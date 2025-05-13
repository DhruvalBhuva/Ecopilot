import torch
from typing import List, Dict
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from src.components.postgreSQLWrapper import PostgreSQLWrapper
from src.components.openAIWrapper import OpenAIWrapper
from src.load_config import LoadConfig
from src.logger import logger


class HybridRetriever:
    def __init__(
        self,
        postgres_wrapper: PostgreSQLWrapper,
        embedder: OpenAIWrapper,
        alpha: float = 0.5,
        enable_reranking: bool = True,
        reranker_model_name: str = "BAAI/bge-reranker-base",
        device: str = "cuda",
    ):
        self.postgres_wrapper = postgres_wrapper
        self.embedder = embedder
        self.alpha = alpha
        self.enable_reranking = enable_reranking
        self.scaler = MinMaxScaler()
        self.device = device

        if self.enable_reranking:
            self.tokenizer = AutoTokenizer.from_pretrained(reranker_model_name)
            self.reranker_model = AutoModelForSequenceClassification.from_pretrained(reranker_model_name).to(self.device)
            self.reranker_model.eval()

    def dense_retrieval(self, query_text, top_k=10):
        try:
            with self.postgres_wrapper.connection.cursor() as cursor:
                query_embedding = self.embedder.embed_query(query_text)
                cursor.execute(
                    f'''
                    SELECT id, source, chunk_num, embedding, text,
                        1 - (embedding <=> %s::vector) AS cosine_similarity
                    FROM "{self.postgres_wrapper.table}"
                    ORDER BY cosine_similarity DESC
                    LIMIT %s;
                    ''',
                    (query_embedding, top_k),
                )
                results = cursor.fetchall()
                return [
                    {
                        "id": row[0],
                        "source": row[1],
                        "chunk_num": row[2],
                        "embedding": row[3],
                        "text": row[4],
                        "dense_score": row[5],
                    }
                    for row in results
                ]
        except Exception as e:
            logger.error(f"Error during dense retrieval: {e}")
            raise

    def sparse_retrieval(self, query_text, top_k=10):
        try:
            with self.postgres_wrapper.connection.cursor() as cursor:
                cursor.execute(
                    f'''
                    SELECT id, source, chunk_num, embedding, text,
                        ts_rank_cd(to_tsvector('simple', text), plainto_tsquery('simple', %s)) AS rank
                    FROM "{self.postgres_wrapper.table}"
                    WHERE to_tsvector('simple', text) @@ plainto_tsquery('simple', %s)
                    ORDER BY rank DESC
                    LIMIT %s;
                    ''',
                    (query_text, query_text, top_k),
                )
                results = cursor.fetchall()
                return [
                    {
                        "id": row[0],
                        "source": row[1],
                        "chunk_num": row[2],
                        "embedding": row[3],
                        "text": row[4],
                        "sparse_score": row[5],
                    }
                    for row in results
                ]
        except Exception as e:
            logger.error(f"Error during sparse retrieval: {e}")
            raise

    def normalize_scores(self, dense_results, sparse_results):
        dense_scores = np.array([doc["dense_score"] for doc in dense_results]).reshape(-1, 1)
        sparse_scores = np.array([doc["sparse_score"] for doc in sparse_results]).reshape(-1, 1)

        all_scores = np.concatenate((dense_scores, sparse_scores), axis=0)
        normalized_scores = self.scaler.fit_transform(all_scores).flatten()

        dense_normalized = normalized_scores[:len(dense_results)]
        sparse_normalized = normalized_scores[len(dense_results):]

        return dense_normalized, sparse_normalized

    def rerank_results(self, query: str, documents: List[Dict], top_k: int = 10):
        try:
            pairs = [[query, doc["text"]] for doc in documents]
            inputs = self.tokenizer(pairs, padding=True, truncation=True, return_tensors="pt").to(self.device)

            with torch.no_grad():
                scores = self.reranker_model(**inputs).logits.squeeze(dim=1).cpu().numpy()

            for doc, score in zip(documents, scores):
                doc["reranker_score"] = float(score)

            return sorted(documents, key=lambda x: x["reranker_score"], reverse=True)[:top_k]

        except Exception as e:
            logger.error(f"Error during reranking: {e}")
            raise

    def retrive(self, query_text, top_k=10, alpha=None):
        alpha = alpha if alpha is not None else self.alpha

        dense_results = self.dense_retrieval(query_text, top_k)
        sparse_results = self.sparse_retrieval(query_text, top_k)

        if not sparse_results:
            logger.warning("No sparse results found. Using dense results only.")
            for doc in dense_results:
                doc["combined_score"] = doc["dense_score"]
            return self.rerank_results(query_text, dense_results[:top_k]) if self.enable_reranking else dense_results[:top_k]

        dense_norm, sparse_norm = self.normalize_scores(dense_results, sparse_results)

        combined_results = []
        for i in range(max(len(dense_results), len(sparse_results))):
            dense_doc = dense_results[i] if i < len(dense_results) else None
            sparse_doc = sparse_results[i] if i < len(sparse_results) else None

            dense_score = dense_norm[i] if dense_doc else 0
            sparse_score = sparse_norm[i] if sparse_doc else 0
            combined_score = alpha * dense_score + (1 - alpha) * sparse_score

            # Prefer dense_doc if both exist and are equal in score
            combined_doc = dense_doc if dense_doc and (not sparse_doc or dense_score >= sparse_score) else sparse_doc

            if combined_doc:
                combined_doc["combined_score"] = combined_score
                combined_results.append(combined_doc)

        combined_results.sort(key=lambda x: x["combined_score"], reverse=True)
        final_results = combined_results[:top_k]

        return self.rerank_results(query_text, final_results) if self.enable_reranking else final_results


# Example usage
if __name__ == "__main__":
    config_loader = LoadConfig()
    postgres_wrapper = PostgreSQLWrapper()
    openai_embedder = OpenAIWrapper(embedding_model_name=config_loader.embedding_model)

    hybrid_retriever = HybridRetriever(
        postgres_wrapper=postgres_wrapper,
        embedder=openai_embedder,
        alpha=0.5,  # Tune as needed
        device=config_loader.device,
    )

    query = "Welche Rolle spielt die OeMAG in der Förderung erneuerbarer Energien in Österreich?"

    results = hybrid_retriever.retrive(query, top_k=10)

    for result in results:
        print(f"Source: {result['source']}, Chunk: {result['chunk_num']}")
        print(f"Text: {result['text']}")
        if "reranker_score" in result:
            print(f"Reranker Score: {result['reranker_score']:.3f}")
        print("-" * 50)
