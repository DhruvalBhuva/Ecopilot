import torch
from typing import List, Dict
from src.logger import logger
from pinecone import ServerlessSpec
from langchain.schema import Document
from src.load_config import LoadConfig
from pinecone import Pinecone as PineconeClient
from src.components.openAIWrapper import OpenAIWrapper


class PineconeWrapper:
    def __init__(self, api_key: str, environment: str, indexes_config: Dict[str, dict]):
        self.api_key = api_key
        self.environment = environment
        self.indexes_config = indexes_config
        self.pinecone_client = None
        self.indexes = {}  # Store initialized indexes here

        self.initialize_pinecone()

    def initialize_pinecone(self):
        """Initialize Pinecone client and set up multiple indexes."""
        self.pinecone_client = PineconeClient(api_key=self.api_key)
        existing_indexes = [index.name for index in self.pinecone_client.list_indexes()]

        for index_name, config in self.indexes_config.items():

            self.indexes[index_name] = self.pinecone_client.Index(
                index_name
            )  # Use actual index name
            logger.info(f"Connected to Pinecone index: {index_name}")

    def generate_vector_id(self, source: str, doc_id: int):
        """Generate a unique vector ID using a hash of source and doc ID."""
        return f"{hash(source)}-{doc_id}"

    def upsert_documents(
        self, docs: List[Document], index_name: str, batch_size: int = 100
    ):
        """Adds documents with embeddings to the specified Pinecone index in batches."""

        if index_name not in self.indexes:
            logger.error(f"Index '{index_name}' is not initialized.")
            return

        index = self.indexes[index_name]  # Ensure you are using actual index names

        vectors = []
        for i, doc in enumerate(docs):
            vector_id = self.generate_vector_id(
                doc.metadata.get("source", ""), doc.metadata.get("chunk_num", i)
            )

            try:
                embedding = doc.metadata.get("embedding", None)
                if isinstance(embedding, torch.Tensor):
                    embedding = embedding.tolist()

                metadata = {
                    "source": doc.metadata.get("source", None),
                    "text": doc.page_content,
                    "chunk_num": doc.metadata.get("chunk_num", i),
                    "bm25_score": doc.metadata.get("bm25_score", 0),
                    "language": doc.metadata.get("language", "en"),
                }
                vectors.append(
                    {
                        "id": vector_id,
                        "values": embedding,  # Ensure this is a list of floats
                        "metadata": metadata,
                    }
                )
            except Exception as e:
                logger.error(f"Error generating embedding for document {i}: {e}")
                continue

        if vectors:
            try:
                for i in range(0, len(vectors), batch_size):
                    batch = vectors[i : i + batch_size]
                    index.upsert(vectors=batch)  # Use correct index object

                    if (i // batch_size + 1) % 1000 == 0:
                        logger.info(
                            f"Upserted batch {i//batch_size + 1} with {len(batch)} vectors."
                        )

                logger.info(
                    f"Successfully upserted {len(vectors)} documents to {index_name}."
                )
            except Exception as e:
                logger.error(f"Error upserting vectors to {index_name}: {e}")
        else:
            logger.info("No new vectors to upsert.")

    def delete_all_vectors(self, index_name: str):
        """Deletes all vectors from the specified Pinecone index."""
        if index_name not in self.indexes:
            logger.error(f"Index '{index_name}' is not initialized.")
            return

        try:
            index_stats = self.indexes[index_name].describe_index_stats()
            total_vectors = index_stats.get("total_vector_count", 0)

            if total_vectors == 0:
                logger.info(f"No vectors to delete in index '{index_name}'.")
                return

            logger.info(f"Deleting {total_vectors} vectors from '{index_name}'.")

            all_ids = []
            dummy_vector = [0.0] * self.indexes_config[index_name]["dimension"]
            top_k = 1000

            while True:
                query_response = self.indexes[index_name].query(
                    vector=dummy_vector,
                    top_k=top_k,
                    include_metadata=False,
                    include_values=False,
                )
                batch_ids = [match.id for match in query_response.matches]
                all_ids.extend(batch_ids)

                if len(batch_ids) < top_k:
                    break

            batch_size = 1000
            for i in range(0, len(all_ids), batch_size):
                batch_ids = all_ids[i : i + batch_size]
                self.indexes[index_name].delete(ids=batch_ids)
                logger.info(f"Deleted {len(batch_ids)} vectors from '{index_name}'.")

            logger.info(
                f"Successfully deleted all {len(all_ids)} vectors from '{index_name}'."
            )
        except Exception as e:
            logger.error(f"Error deleting vectors from {index_name}: {e}")

    def get_index_statistics(self, index_name: str):
        """Prints index statistics for a given index."""
        if index_name not in self.indexes:
            logger.error(f"Index '{index_name}' is not initialized.")
            return

        index_stats = self.indexes[index_name].describe_index_stats()
        logger.info(
            f"Total vectors in '{index_name}': {index_stats['total_vector_count']}"
        )
        logger.info(f"Index '{index_name}' stats: {index_stats}")


if __name__ == "__main__":
    config_loader = LoadConfig()

    pinecone_wrapper = PineconeWrapper(
        api_key=config_loader.PINECONE_API_KEY,
        environment=config_loader.pinecone_environment,
        indexes_config=config_loader.pinecone_indexes_config,
    )

    # Test document insertion
    test_docs = [
        Document(
            page_content="Deep learning is a subset of machine learning.",
            metadata={
                "source": "source2",
                "chunk_num": 2,
                "bm25_score": 0.8,
                "embedding": [0.2] * 1536,
            },
        ),
    ]

    openai_embedder = OpenAIWrapper(
        api_key=config_loader.OPENAI_API_KEY,
        model_name=config_loader.get_embedding_model_config("openai")["name"],
    )

    for doc in test_docs:
        embedding = openai_embedder.embed_documents(doc.page_content)
        doc.metadata["embedding"] = embedding

    # Insert into OpenAI index
    # pinecone_wrapper.upsert_documents(
    #     test_docs, index_name="ecopilot-openai-embeddings"
    # )

    # Insert into JinaAI index
    pinecone_wrapper.upsert_documents(
        test_docs, index_name="ecopilot-openai-embeddings"
    )

    # delete all vectors
    # pinecone_wrapper.delete_all_vectors(
    #     config_loader.get_pinecone_index_config("ecopilot-openai-embeddings")["index_name"]
    # )
    # pinecone_wrapper.delete_all_vectors(
    #     config_loader.get_pinecone_index_config("ecopilot-jinaai-embeddings")["index_name"]
    # )

    # Get statistics for each index
    # pinecone_wrapper.get_index_statistics(
    #     config_loader.get_pinecone_index_config("ecopilot-openai-embeddings")["index_name"]
    # )
    # pinecone_wrapper.get_index_statistics(
    #     config_loader.get_pinecone_index_config("ecopilot-jinaai-embeddings")["index_name"]
    # )
