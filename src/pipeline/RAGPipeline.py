import torch
from typing import List, Dict
from src.logger import logger
from src.load_config import LoadConfig
from src.components.openAIWrapper import OpenAIWrapper
from src.components.jinaaiWrapper import JinaaiWrapper
from src.components.postgreSQLWrapper import PostgreSQLWrapper
from src.components.postgreSQLRetriever import HybridRetriever
from src.components.llama3GermanWrapper import Llama3GermanWrapper


class RAGPipeline:
    def __init__(
        self,
        top_k: int = 5,
    ):
        """
        Initialize the RAG pipeline.
        """
        config_loader = LoadConfig()
        self.top_k = top_k
        self.device = config_loader.device
        # self.device = "cpu"  # Force CPU for compatibility

        # Initialize components with CPU
        # self.openai_embedder = OpenAIWrapper()
        self.jinaai_embedder = JinaaiWrapper()
        self.postgres_wrapper = PostgreSQLWrapper()
        self.hybrid_retriever = HybridRetriever(
            self.postgres_wrapper,
            self.jinaai_embedder,
            enable_reranking=True,
            device=self.device,
        )
        self.openai_wrapper = OpenAIWrapper()
        self.llama3_german_wrapper = Llama3GermanWrapper(device=self.device)

    def response(self, query: str, model="GPT") -> Dict:
        """
        Retrieve, rerank documents, and generate an answer in a single function.
        """
        try:
            # Retrieve and rerank documents using HybridRetriever
            reranked_documents = self.hybrid_retriever.retrive(
                query, top_k=self.top_k, 
            )

            # Combine the reranked documents into a single context
            context = "\n\n".join([doc["text"] for doc in reranked_documents])

            # Create a prompt for the LLM
            prompt = f"""

            User Question:
            {query}
            
            Retrieved context:
            {context}
            """

            # Generate the answer using the specified model
            if model == "Llama3":
                answer = self.llama3_german_wrapper.rag_text_generator(prompt)
            else:
                answer = self.openai_wrapper.rag_text_generator(prompt)

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
        top_k=5,
    )

    # Example query
    query = "What are the main differences between CSRD and SFDR disclosure obligations? Also write artical number, where I can find complete information about it."

    # Run the RAG pipeline
    result = rag_pipeline.response(query, model="GPT")

    # Print the results
    print(f"Query: {query}")
    print("Generated Answer:")
    print(result["answer"])
