from typing import List, Dict
from src.logger import logger
from src.load_config import LoadConfig
from src.components.openAIWrapper import OpenAIWrapper
from src.components.postgreSQLWrapper import PostgreSQLWrapper
from src.components.postgreSQLRetriever import HybridRetriever
from src.components.llama3GermanWrapper import Llama3GermanWrapper

import torch

class RAGPipeline:
    def __init__(
        self,
        top_k_per_embedder: int = 2,
        top_k_final: int = 1,
    ):
        """
        Initialize the RAG pipeline.

        Args:
            top_k_per_embedder: Number of documents to retrieve per embedder.
            top_k_final: Number of top reranked documents to use for answer generation.
        """
        config_loader = LoadConfig()

        self.top_k_per_embedder = top_k_per_embedder
        self.top_k_final = top_k_final

        # Check for CUDA availability, use CPU if not available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Initialize embedders
        self.openai_embedder = OpenAIWrapper(
            embedding_model_name=config_loader.embedding_model,
        )

        # Initialize PostgreSQLWrapper
        self.postgres_wrapper = PostgreSQLWrapper() 
        
        # Initialize HybridRetriever with correct device
        self.hybrid_retriever = HybridRetriever(
            self.postgres_wrapper, self.openai_embedder, enable_reranking=True, device=self.device
        )

        # Initialize OpenAIWrapper for answer generation
        self.openai_wrapper = OpenAIWrapper(
            llm_model_name=config_loader.get_llm_model_config("GPT")["model_name"],
        )

        # Initialize Llama3GermanWrapper for answer generation (optional)
        # self.llama3_german_wrapper = Llama3GermanWrapper(
        #     model_name=config_loader.get_llm_model_config("llama3-german")["model_name"],
        #     device=self.device,
        # )

    def response(self, query: str, model="GPT") -> Dict:
        """
        Retrieve, rerank documents, and generate an answer in a single function.

        Args:
            query: The query string.
            model: The model to use for answer generation ("GPT" or "Llama3").

        Returns:
            Dictionary containing the generated answer and reranked documents.
        """
        try:
            # Retrieve and rerank documents using HybridRetriever
            reranked_documents = self.hybrid_retriever.retrive(
                query,
                top_k=self.top_k_per_embedder,
                alpha=0.6  # You can set or adjust alpha here
            )

            # Combine the reranked documents into a single context
            context = "\n\n".join([doc["text"] for doc in reranked_documents])

            # Create a prompt for the LLM
            prompt = f"""
            Retrieved context:
            {context}

            User Question:
            {query}
            """

            # Generate the answer using the specified model
            if model == "Llama3":
                answer = self.llama3_german_wrapper.text_generator(prompt)
            else:
                answer = self.openai_wrapper.text_generator(prompt)

            return {
                "answer": answer,
                "reranked_documents": reranked_documents,
            }

        except Exception as e:
            logger.error(f"Error in response: {e}")
            raise


# Example Usage
if __name__ == "__main__":
    # Load config
    config_loader = LoadConfig()

    # Initialize RAG Pipeline
    rag_pipeline = RAGPipeline(
        top_k_per_embedder=10,
        top_k_final=5,
    )

    # Example query
    query = "Welche Rolle spielt die OeMAG in der Förderung erneuerbarer Energien in Österreich?"

    # Run the RAG pipeline
    result = rag_pipeline.response(query, model="GPT")

    # Print the results
    print(f"Query: {query}")
    print("Generated Answer:")
    print(result["answer"])
