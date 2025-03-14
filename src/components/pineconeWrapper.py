import torch
from typing import List, Dict
from src.logger import logger
from pinecone import ServerlessSpec
from langchain.schema import Document
from src.load_config import LoadConfig
from pinecone import Pinecone as PineconeClient
from src.components.openAIWrapper import OpenAIWrapper


class PineconeWrapper:
    def __init__(self, api_key: str = None, index_name: str = "ecopilot-corpus"):
        self.api_key = api_key
        self.pinecone_client = None
        self.index_name = index_name
        self.index = None
        self.initialize_pinecone()

    def initialize_pinecone(self):
        """Initialize Pinecone client and set up multiple indexes."""
        self.pinecone_client = PineconeClient(api_key=self.api_key)
        # self.index = self.pinecone_client.index(self.index_name)

        # Check if index exists, if not create it
        if self.index.exists():
            self.index = self.pinecone_client.index(self.index_name)
            logger.info(f"Index '{self.index_name}' already exists.")
        else:
            logger.info(f"Creating index '{self.index_name}'...")
            self.pinecone_client.create_index(
                name=self.index_name,
                dimension=1536,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1",
                ),
            )

        logger.info(f"Connected to Pinecone index: {self.index_name}")

    def generate_vector_id(self, source: str, doc_id: int):
        """Generate a unique vector ID using a hash of source and doc ID."""
        return f"{hash(source)}-{doc_id}"

    def upsert_documents(
        self, docs: List[Document], index_name: str, batch_size: int = 100
    ):
        """Adds documents with embeddings to the specified Pinecone index in batches."""

        if index_name != self.index_name:
            logger.info(
                f"Provided index name '{index_name}' is different from the initialized index '{self.index_name}'."
            )
            return

        vectors = []
        for i, doc in enumerate(docs):
            vector_id = self.generate_vector_id(
                doc.metadata.get("source", ""), doc.metadata.get(
                    "chunk_num", i)
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
                logger.error(
                    f"Error generating embedding for document {i}: {e}")
                continue

        if vectors:
            try:
                for i in range(0, len(vectors), batch_size):
                    batch = vectors[i: i + batch_size]

                    # Use correct index object
                    self.index.upsert(vectors=batch)

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
        """Deletes all vectors from the specified Pinecone index efficiently."""
        if index_name != self.index_name:
            logger.error(f"Index '{index_name}' is not initialized.")
            return

        try:
            logger.info(
                f"Deleting all vectors from '{index_name}' using delete_all=True.")
            self.indexes[index_name].delete(delete_all=True)
            logger.info(
                f"Successfully deleted all vectors from '{index_name}'.")
        except Exception as e:
            logger.error(f"Error deleting vectors from {index_name}: {e}")

    def get_index_statistics(self, index_name: str):
        """Prints index statistics for a given index."""
        if index_name != self.index:
            logger.error(f"Index '{index_name}' is not initialized.")
            return

        index_stats = self.index.describe_index_stats()
        logger.info(
            f"Total vectors in '{index_name}': {index_stats['total_vector_count']}"
        )
        logger.info(f"Index '{index_name}' stats: {index_stats}")


if __name__ == "__main__":
    config_loader = LoadConfig()

    pinecone_wrapper = PineconeWrapper(config_loader.pinecone_index)

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
        embedding_model_name=config_loader.embedding_model,
    )

    for doc in test_docs:
        embedding = openai_embedder.embed_documents(doc.page_content)
        doc.metadata["embedding"] = embedding

    # Insert into OpenAI index
    pinecone_wrapper.upsert_documents(
        test_docs, index_name=config_loader.pinecone_index
    )

    # delete all vectors
    # pinecone_wrapper.delete_all_vectors(config_loader.pinecone_index)

    # # Get statistics for each index
    # pinecone_wrapper.get_index_statistics(config_loader.pinecone_index)
