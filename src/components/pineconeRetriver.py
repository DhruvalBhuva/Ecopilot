from typing import List, Dict
from src.logger import logger
from rank_bm25 import BM25Okapi
from src.components.pineconeWrapper import PineconeWrapper
from src.components.openAIWrapper import OpenAIWrapper


class HybridRetriever:
    def __init__(
        self,
        pinecone_wrapper: PineconeWrapper,
        embedder: OpenAIWrapper,
    ):
        """
        Initialize the Hybrid Retriever with a single embedder and index.
        """
        self.pinecone_wrapper = pinecone_wrapper
        self.embedder = embedder
        self.index = self.pinecone_wrapper.index

    def dense_retrieval(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Perform dense retrieval using OpenAI embeddings.
        """
        try:
            query_embedding = self.embedder.embed_query(query)
            query_embedding = query_embedding.tolist() if hasattr(query_embedding, "tolist") else query_embedding

            query_response = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
            )

            return [
                {
                    "id": match.id,
                    "score": match.score,
                    "text": match.metadata.get("text", ""),
                    "source": match.metadata.get("source", "unknown"),
                    "chunk_num": match.metadata.get("chunk_num", -1),
                    "bm25_score": match.metadata.get("bm25_score", 0),
                }
                for match in query_response.matches
            ]
        except Exception as e:
            logger.error(f"Error during dense retrieval: {e}")
            return []

    def sparse_retrieval(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Perform sparse retrieval using BM25-style matching and metadata filtering.
        """
        try:
            query_tokens = query.lower().split()

            query_response = self.index.query(
                vector=[0.0] * 1536,  # Use dummy vector
                top_k=500,
                include_metadata=True,
                include_values=False,
                filter={"text": {"$in": query_tokens}},
            )

            documents = [
                {
                    "id": match.id,
                    "text": match.metadata.get("text", ""),
                    "source": match.metadata.get("source", "unknown"),
                    "bm25_score": match.metadata.get("bm25_score", 0),
                }
                for match in query_response.matches
                if query.lower() in match.metadata.get("text", "").lower()
            ]

            if not documents:
                return []

            tokenized_corpus = [doc["text"].lower().split() for doc in documents]
            bm25 = BM25Okapi(tokenized_corpus)
            tokenized_query = query.lower().split()
            doc_scores = bm25.get_scores(tokenized_query)

            sorted_scores = sorted(
                [(i, (doc_scores[i] + documents[i]["bm25_score"]) / 2)
                 for i in range(len(doc_scores))],
                key=lambda x: x[1], reverse=True
            )[:top_k]

            return [
                {
                    "id": documents[i]["id"],
                    "score": score,
                    "text": documents[i]["text"],
                    "source": documents[i]["source"],
                    "chunk_num": i + 1,
                    "bm25_score": documents[i]["bm25_score"],
                }
                for i, score in sorted_scores
            ]
        except Exception as e:
            logger.error(f"Error during sparse retrieval: {e}")
            return []

    def hybrid_retrieval(self, query: str, top_k: int = 10, dense_weight: float = 1.0, sparse_weight: float = 1.0) -> List[Dict]:
        """
        Combine dense and sparse retrieval scores to return top-k results.
        """
        try:
            dense_results = self.dense_retrieval(query, top_k)
            sparse_results = self.sparse_retrieval(query, top_k)

            combined_results = {}
            for result in dense_results + sparse_results:
                doc_id = result["id"]
                if doc_id not in combined_results:
                    combined_results[doc_id] = result
                    combined_results[doc_id]["combined_score"] = 0

                combined_results[doc_id]["combined_score"] += (
                    dense_weight * result.get("score", 0)
                    + sparse_weight * result.get("bm25_score", 0)
                )

            sorted_results = sorted(
                combined_results.values(),
                key=lambda x: x["combined_score"],
                reverse=True
            )[:top_k]

            return sorted_results

        except Exception as e:
            logger.error(f"Error during hybrid retrieval: {e}")
            return []

if __name__ == "__main__":
    from src.load_config import LoadConfig

    config_loader = LoadConfig()

    openai_embedder = OpenAIWrapper(
        embedding_model_name=config_loader.embedding_model,
    )

    # Init pinecone wrapper
    pinecone_wrapper = PineconeWrapper(
        index_name=config_loader.pinecone_index,
    )

    retriever = HybridRetriever(
        pinecone_wrapper=pinecone_wrapper,
        embedder=openai_embedder,
    )

    query = "Welche Rolle spielt die OeMAG in der Förderung erneuerbarer Energien in Österreich?"

    results = retriever.hybrid_retrieval(query, top_k=10)
    for res in results:
        print(f"Score: {res['combined_score']:.3f}")
        print(f"Source: {res['source']}")
        print(f"Text: {res['text']}")
        print("-" * 50)
