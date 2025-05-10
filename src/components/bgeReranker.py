import torch
from src.logger import logger
from typing import List, Dict, Union
from src.load_config import LoadConfig
from src.components.pineconeWrapper import PineconeWrapper
from src.components.postgreSQLWrapper import PostgreSQLWrapper
# from src.components.pineconeRetriver import HybridRetriever
from src.components.postgreSQLRetriever import HybridRetriever
from src.components.openAIWrapper import OpenAIWrapper
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class BGEReranker:
    def __init__(self, model_name="BAAI/bge-reranker-base", device="cuda"):
        """
        Initialize the BGE reranker model.

        :param model_name: Name of the BGE reranker model (default: "BAAI/bge-reranker-base").
        :param device: Device to run the model on (default: "cuda").
        """
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(
            self.device
        )
        self.model.eval()  

    def rerank(self, query, documents, top_k=10):
        """
        Rerank a list of documents based on their relevance to the query using the BGE reranker.

        :param query: The query text.
        :param documents: List of documents (each document is a dictionary with at least a "text" key).
        :param top_k: Number of top reranked documents to return.
        :return: List of reranked documents with their scores.
        """
        try:
            # Prepare input pairs (query, document)
            pairs = [[query, doc["text"]] for doc in documents]

            # Tokenize the pairs
            inputs = self.tokenizer(
                pairs,
                padding=True,
                truncation=True,
                return_tensors="pt",
                # max_length=512,
            ).to(self.device)

            # Perform inference
            with torch.no_grad():
                scores = self.model(**inputs).logits.squeeze(dim=1).cpu().numpy()

            # Add scores to documents
            for doc, score in zip(documents, scores):
                doc["reranker_score"] = float(score)

            # Sort documents by reranker score in descending order
            reranked_documents = sorted(
                documents, key=lambda x: x["reranker_score"], reverse=True
            )

            return reranked_documents[:top_k]

        except Exception as e:
            logger.error(f"Error during reranking: {e}")
            raise

    def retrieve_and_rerank(
        self,
        query: str,
        hybrid_retriever: HybridRetriever,
        top_k_per_embedder: int = 10,
        top_k_final: int = 10,
    ) -> List[Dict]:
        """
        Retrieve documents using each embedder, combine them, and rerank the combined list.

        Args:
            query: The query string.
            hybrid_retriever: Instance of HybridRetriever to retrieve documents.
            embedder: Instance of OpenAIWrapper to embed documents.
            top_k_per_embedder: Number of documents to retrieve per embedder.
            top_k_final: Number of top reranked documents to return.

        Returns:
            List of reranked documents with their scores.
        """
        try:
            # Retrieve documents using each embedder

            retrieved_docs = hybrid_retriever.retrive(
                query, top_k=top_k_per_embedder
            )

            # Rerank the combined documents
            reranked_documents = self.rerank(
                query, retrieved_docs, top_k=top_k_final
            )

            return reranked_documents

        except Exception as e:
            logger.error(f"Error during retrieve_and_rerank: {e}")
            raise


# Example Usage
if __name__ == "__main__":
    # Load config
    config_loader = LoadConfig()

    # Initialize Wrapper
    # pinecone_wrapper = PineconeWrapper(index_name=config_loader.pinecone_index)
    postgres_wrapper = PostgreSQLWrapper()
    
    # Initialize OpenAI Embedder
    openai_embedder = OpenAIWrapper(embedding_model_name=config_loader.embedding_model)
    

    # Initialize HybridRetriever
    hybrid_retriever = HybridRetriever(
        postgres_wrapper, openai_embedder,
    )

    # Initialize BGE Reranker
    bge_reranker = BGEReranker(device=config_loader.device)

    # Example query
    # query = "Welche Rolle spielt die OeMAG in der Förderung erneuerbarer Energien in Österreich?"
    query = "Welche Rolle spielt die OeMAG in der Förderung erneuerbarer Energien in Österreich?"

    reranked_results = bge_reranker.retrieve_and_rerank(
        query,
        hybrid_retriever,
        top_k_per_embedder=10,  # Retrieve 10 docs per embedder
        top_k_final=10,  # Return top 10 reranked docs
    )

    # Print reranked results
    print("Reranked Results:")
    for result in reranked_results:
        print(f"Source: {result['source']}")
        print(f"Text: {result['text']}")
        print(f"Reranker Score: {result['reranker_score']:.3f}")
        print("-" * 50)
