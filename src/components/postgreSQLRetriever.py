from src.components.postgreSQLWrapper import PostgreSQLWrapper
from src.components.openAIWrapper import OpenAIWrapper
from src.load_config import LoadConfig
from src.logger import logger

from pgvector.psycopg2 import register_vector
from sklearn.preprocessing import MinMaxScaler
import numpy as np


class HybridRetriever:
    def __init__(self, postgres_wrapper: PostgreSQLWrapper, embedder: OpenAIWrapper, dense_weight: float = 0.5, sparse_weight: float = 0.5):
        """
        Initialize the hybrid retriever.

        :param postgres_wrapper: PostgreSQLWrapper instance for database operations.
        :param dense_weight: Weight for dense retrieval scores.
        :param sparse_weight: Weight for sparse retrieval scores.
        """
        self.postgres_wrapper = postgres_wrapper
        self.embedder = embedder
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.scaler = MinMaxScaler()  # For normalizing scores

    def dense_retrieval(self, query_text, top_k=10):
        """
        Perform dense retrieval using vector search.

        :param query_embedding: Embedding of the query.
        :param top_k: Number of top results to retrieve.
        :return: List of documents with dense retrieval scores.
        """
        try:
            with self.postgres_wrapper.connection.cursor() as cursor:
                # Convert query embedding to a list and cast it to the vector type
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
                        "dense_score": row[5],  # Cosine similarity
                    }
                    for row in results
                ]
        except Exception as e:
            logger.error(f"Error during dense retrieval: {e}")
            raise

    def sparse_retrieval(self, query_text, top_k=10):
        """
        Perform sparse retrieval using full-text search.

        :param query_text: Query text for full-text search.
        :param top_k: Number of top results to retrieve.
        :return: List of documents with sparse retrieval scores.
        """
        try:
            with self.postgres_wrapper.connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    SELECT id, source, chunk_num, embedding, text
                    FROM "{self.postgres_wrapper.table}"
                    WHERE to_tsvector('simple', text) @@ plainto_tsquery('simple', %s)
                    LIMIT %s;
                    """,
                    (query_text, top_k),
                )
                results = cursor.fetchall()
                return [
                    {
                        "id": row[0],
                        "source": row[1],
                        "chunk_num": row[2],
                        "embedding": row[3],
                        "text": row[4],
                        "sparse_score": row[5],  # BM25 score
                    }
                    for row in results
                ]
        except Exception as e:
            logger.error(f"Error during sparse retrieval: {e}")
            raise

    def normalize_scores(self, dense_results, sparse_results):
        """
        Normalize both dense and sparse scores together.

        :param dense_results: List of dense retrieval results.
        :param sparse_results: List of sparse retrieval results.
        :return: Normalized dense and sparse scores.
        """
        dense_scores = np.array([doc["dense_score"] for doc in dense_results]).reshape(
            -1, 1
        )
        sparse_scores = np.array(
            [doc["sparse_score"] for doc in sparse_results]
        ).reshape(-1, 1)

        # Combine both scores for normalization
        all_scores = np.concatenate((dense_scores, sparse_scores), axis=0)
        all_scores_normalized = self.scaler.fit_transform(all_scores).flatten()

        # Split back into dense and sparse normalized scores
        dense_scores_normalized = all_scores_normalized[: len(dense_results)]
        sparse_scores_normalized = all_scores_normalized[len(dense_results) :]

        return dense_scores_normalized, sparse_scores_normalized


    def retrive(self, query_text, top_k=10, dense_weight=None, sparse_weight=None):
        """
        Perform hybrid retrieval by combining dense and sparse retrieval results.

        :param query_text: Query text for sparse and dense retrieval.
        :param top_k: Number of top results to retrieve.
        :param dense_weight: Optional override for dense score weight.
        :param sparse_weight: Optional override for sparse score weight.
        :return: List of documents ranked by hybrid scores.
        """
        # Use instance-level weights if not provided
        dense_w = dense_weight if dense_weight is not None else self.dense_weight
        sparse_w = sparse_weight if sparse_weight is not None else self.sparse_weight

        # Perform dense and sparse retrieval
        dense_results = self.dense_retrieval(query_text, top_k)
        sparse_results = self.sparse_retrieval(query_text, top_k)

        # Handle empty sparse results
        if not sparse_results:
            logger.warning("No sparse results found. Using dense results only.")
            for doc in dense_results:
                doc["combined_score"] = doc["dense_score"]
            return dense_results[:top_k]

        # Normalize both dense and sparse scores together
        dense_scores_normalized, sparse_scores_normalized = self.normalize_scores(
            dense_results, sparse_results
        )

        # Combine results
        combined_results = []
        for i in range(max(len(dense_results), len(sparse_results))):
            dense_doc = dense_results[i] if i < len(dense_results) else None
            sparse_doc = sparse_results[i] if i < len(sparse_results) else None

            # Calculate combined score
            dense_score = dense_scores_normalized[i] if dense_doc else 0
            sparse_score = sparse_scores_normalized[i] if sparse_doc else 0
            combined_score = dense_w * dense_score + sparse_w * sparse_score

            # Use the document with the highest score
            if dense_doc and sparse_doc:
                combined_doc = dense_doc if dense_score > sparse_score else sparse_doc
            elif dense_doc:
                combined_doc = dense_doc
            else:
                combined_doc = sparse_doc

            if combined_doc:
                combined_doc["combined_score"] = combined_score
                combined_results.append(combined_doc)

        # Sort by combined score
        combined_results.sort(key=lambda x: x["combined_score"], reverse=True)

        return combined_results[:top_k]

# Example Usage
if __name__ == "__main__":
    # Load config and initialize PostgreSQLWrapper
    config_loader = LoadConfig()
    
    postgres_wrapper = PostgreSQLWrapper()

    openai_embedder = OpenAIWrapper(
        embedding_model_name=config_loader.embedding_model,
    )
    
    # Initialize hybrid retriever
    hybrid_retriever = HybridRetriever(
        postgres_wrapper, openai_embedder
    )
    
    # Example query
    query_text = "Welche Maßnahmen dürfen bayerische Kommunen im eigenen Wirkungskreis ergreifen, um erneuerbare Energien auszubauen?"
    
    # Perform hybrid retrieval
    results = hybrid_retriever.retrive(query_text, top_k=10)

    # Print results
    for result in results:
        print(f"Source: {result['source']}, Chunk: {result['chunk_num']}")
        print(f"Text: {result['text']}")
        print(f"Combined Score: {result['combined_score']}")
        print("-" * 50)
