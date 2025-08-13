import torch
import numpy as np
from src.logger import logger
from typing import List, Dict
from src.load_config import LoadConfig
from sklearn.preprocessing import MinMaxScaler
from src.components.postgreSQLWrapper import PostgreSQLWrapper
from src.components.openAIWrapper import OpenAIWrapper
from src.components.jinaaiWrapper import JinaaiWrapper

from transformers import AutoTokenizer, AutoModelForSequenceClassification


class HybridRetriever:
    def __init__(
        self,
        postgres_wrapper: PostgreSQLWrapper,
        embedder: OpenAIWrapper,
        alpha: float = 0.8,
        enable_reranking: bool = True,
        reranker_model_name: str = "BAAI/bge-reranker-v2-m3",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.postgres_wrapper = postgres_wrapper
        self.embedder = embedder
        self.alpha = alpha
        self.enable_reranking = enable_reranking  
        self.scaler = MinMaxScaler()
        self.device = device

        if self.enable_reranking:
            self.tokenizer = AutoTokenizer.from_pretrained(reranker_model_name)
            self.reranker_model = AutoModelForSequenceClassification.from_pretrained(
                reranker_model_name
            ).to(self.device)
            self.reranker_model.eval()

    def dense_retrieval(self, query_text, top_k=10):
        try:
            with self.postgres_wrapper.connection.cursor() as cursor:
                query_embedding = self.embedder.embed_query(query_text)
                cursor.execute(
                    f"""
                    SELECT id, source, chunk_num, embedding, text,
                        1 - (embedding <=> %s::vector) AS cosine_similarity
                    FROM "{self.postgres_wrapper.table}"
                    ORDER BY cosine_similarity DESC
                    LIMIT %s;
                    """,
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
                    f"""
                    SELECT 
                        id, source, chunk_num, embedding, text,
                        ts_rank_cd(
                            to_tsvector('simple', text), 
                            websearch_to_tsquery('simple', %s)
                        ) AS rank
                    FROM "{self.postgres_wrapper.table}"
                    WHERE 
                        to_tsvector('simple', text) @@ (
                            SELECT to_tsquery('simple', string_agg(word, ' | '))
                            FROM ts_stat($$SELECT to_tsvector('simple', %s)$$)
                            WHERE length(word) > 3
                        )
                    ORDER BY rank DESC
                    LIMIT %s;
                    """,
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
        dense_scores = np.array([doc["dense_score"] for doc in dense_results]).reshape(
            -1, 1
        )
        sparse_scores = np.array(
            [doc["sparse_score"] for doc in sparse_results]
        ).reshape(-1, 1)

        all_scores = np.concatenate((dense_scores, sparse_scores), axis=0)
        normalized_scores = self.scaler.fit_transform(all_scores).flatten()

        dense_normalized = normalized_scores[: len(dense_results)]
        sparse_normalized = normalized_scores[len(dense_results):]

        return dense_normalized, sparse_normalized

    def rerank_results(self, query: str, documents: List[Dict], top_k: int = 10):
        try:
            pairs = [[query, doc["text"]] for doc in documents]
            inputs = self.tokenizer(
                pairs, padding=True, truncation=True, return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                scores = (
                    self.reranker_model(**inputs).logits.squeeze(dim=1).cpu().numpy()
                )

            for doc, score in zip(documents, scores):
                doc["reranker_score"] = float(score)

            return sorted(documents, key=lambda x: x["reranker_score"], reverse=True)[
                :top_k
            ]

        except Exception as e:
            logger.error(f"Error during reranking: {e}")
            raise

    def retrive(self, query_text, top_k=10, alpha=None, use_reranker=None):
        """
        Hybrid retrieval with optional reranking.
        """
        alpha = alpha if alpha is not None else self.alpha
        use_reranker = self.enable_reranking if use_reranker is None else use_reranker

        # Get both dense and sparse results
        dense_results = self.dense_retrieval(query_text, 10)
        sparse_results = self.sparse_retrieval(query_text, 10)

        # If no sparse results, fallback to dense
        if not sparse_results:
            logger.warning("No sparse results found. Using dense results only.")
            for doc in dense_results:
                doc["combined_score"] = doc["dense_score"]
            return (
                self.rerank_results(query_text, dense_results[:top_k])
                if use_reranker
                else dense_results[:top_k]
            )

        # Normalize scores
        dense_norm, sparse_norm = self.normalize_scores(dense_results, sparse_results)

        # Combine results
        all_docs = {}
        for i, doc in enumerate(dense_results):
            doc_id = doc["id"]
            all_docs[doc_id] = doc.copy()
            all_docs[doc_id]["dense_norm"] = dense_norm[i]
            all_docs[doc_id]["sparse_norm"] = 0

        for i, doc in enumerate(sparse_results):
            doc_id = doc["id"]
            if doc_id in all_docs:
                all_docs[doc_id]["sparse_norm"] = sparse_norm[i]
            else:
                all_docs[doc_id] = doc.copy()
                all_docs[doc_id]["dense_norm"] = 0
                all_docs[doc_id]["sparse_norm"] = sparse_norm[i]

        for doc_id in all_docs:
            all_docs[doc_id]["combined_score"] = (
                alpha * all_docs[doc_id]["dense_norm"]
                + (1 - alpha) * all_docs[doc_id]["sparse_norm"]
            )

        combined_results = sorted(
            all_docs.values(), key=lambda x: x["combined_score"], reverse=True
        )
        final_results = combined_results[:top_k]

        return (
            self.rerank_results(query_text, final_results)
            if use_reranker
            else final_results
        )

if __name__ == "__main__":
    config_loader = LoadConfig()
    postgres_wrapper = PostgreSQLWrapper()
    jinaai_embedder = JinaaiWrapper()

    hybrid_retriever = HybridRetriever(
        postgres_wrapper=postgres_wrapper,
        embedder=jinaai_embedder,
        alpha=0.8,
        device=config_loader.device,
    )

    query = "Ab welchem Endenergieverbrauch müssen Unternehmen Umsetzungspläne erstellen?"

    # ➡ With reranker
    print("\nRetrieving with reranker enabled...")
    results_rerank = hybrid_retriever.retrive(query, top_k=5, use_reranker=True)
    for res in results_rerank:
        print(f"[Reranker Score] {res.get('reranker_score', 0):.3f} - {res['text'][:80]}")

    # ➡ Without reranker
    print("\nRetrieving without reranker...")
    results_no_rerank = hybrid_retriever.retrive(query, top_k=5, use_reranker=False)
    for res in results_no_rerank:
        print(f"[Combined Score] {res['combined_score']:.3f} - {res['text'][:80]}")
