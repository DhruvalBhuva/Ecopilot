from typing import List, Dict, Union
from src.logger import logger
from src.load_config import LoadConfig
from src.components.pineconeWrapper import PineconeWrapper
# from src.components.pineconeRetriver import HybridRetriever
from src.components.openAIWrapper import OpenAIWrapper
from src.components.postgreSQLWrapper import PostgreSQLWrapper
from src.components.postgreSQLRetriever import HybridRetriever
from src.components.llama3GermanWrapper import Llama3GermanWrapper
from src.components.bgeReranker import BGEReranker


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

        # Initialize embedders
        self.openai_embedder = OpenAIWrapper(
            embedding_model_name=config_loader.embedding_model,
        )

        # # Initialize PineconeWrapper
        # self.pinecone_wrapper = PineconeWrapper(
        #     index_name=config_loader.pinecone_index,
        # )
        
        # Initialize PostgreSQLWrapper
        self.postgres_wrapper = PostgreSQLWrapper() 
            
        # # Initialize HybridRetriever
        self.hybrid_retriever = HybridRetriever(
            self.postgres_wrapper, self.openai_embedder
        )

        # Initialize BGE Reranker
        self.bge_reranker = BGEReranker(device=config_loader.device)

        # Initialize OpenAIWrapper for answer generation
        self.openai_wrapper = OpenAIWrapper(
            llm_model_name=config_loader.get_llm_model_config("GPT")["model_name"],
        )
        

        # Initialize Llama3GermanWrapper for answer generation
        self.llama3_german_wrapper = Llama3GermanWrapper(
            model_name=config_loader.get_llm_model_config("llama3-german")[
                "model_name"
            ],
            device=config_loader.device,
        )

    def retrieve_and_rerank(self, query: str) -> List[Dict]:
        """
        Retrieve and rerank documents relevant to the query.

        Args:
            query: The query string.

        Returns:
            List of reranked documents with their scores.
        """
        try:
            # Retrieve and rerank documents using BGE Reranker
            reranked_results = self.bge_reranker.retrieve_and_rerank(
                query,
                self.hybrid_retriever,
                top_k_per_embedder=self.top_k_per_embedder,
                top_k_final=self.top_k_final,
            )
            return reranked_results

        except Exception as e:
            logger.error(f"Error during retrieve_and_rerank: {e}")
            raise

    def generate_answer(
        self, query: str, reranked_documents: List[Dict], model="GPT"
    ) -> str:
        """
        Generate an answer using the reranked documents and the query.

        Args:
            query: The query string.
            reranked_documents: List of reranked documents with their scores.
            model: The model to use for answer generation ("GPT" or "Llama3").

        Returns:
            Generated answer as a string.
        """
        try:
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
            logger.error(f"Error during answer generation: {e}")
            raise

    def response(self, query: str, model="GPT") -> Dict:
        """
        Run the RAG pipeline for a given query.

        Args:
            query: The query string.
            model: The model to use for answer generation ("GPT" or "Llama3").

        Returns:
            Dictionary containing the generated answer and reranked documents.
        """
        try:
            # Step 1: Retrieve and rerank documents
            reranked_documents = self.retrieve_and_rerank(query)

            # Step 2: Generate answer using the reranked documents
            answer = self.generate_answer(query, reranked_documents, model=model)

            return answer

        except Exception as e:
            logger.error(f"Error in RAG pipeline: {e}")
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
