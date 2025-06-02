import json
from typing import List, Dict
from src.load_config import LoadConfig
from src.pipeline.RAGPipeline import RAGPipeline  # Import the RAGPipeline


class TestDatasetGenerator:
    def __init__(self, device="cuda"):
        config_loader = LoadConfig()
        self.rag_pipeline = RAGPipeline(
            top_k_per_embedder=10,
            top_k_final=5,
        )
        self.device = device

    def generate_answers_and_docs(self, question: str) -> Dict:
        """
        Generate answers and retrieve documents for the given question.
        """
        try:
            # Retrieve reranked documents and answers
            gpt_result = self.rag_pipeline.response(query=question, model="GPT")
            llama_result = self.rag_pipeline.response(query=question, model="Llama3")

            retrieved_docs = [
                {"id": doc["id"], "text": doc["text"]}
                for doc in gpt_result.get("reranked_documents", [])
            ]

            return {
                "retrieved_documents": retrieved_docs,
                "gpt_generated_answer": gpt_result.get("answer", ""),
                "llama_generated_answer": llama_result.get("answer", ""),
            }

        except Exception as e:
            print(f"Error generating data for question '{question}': {e}")
            return {
                "retrieved_documents": [],
                "gpt_generated_answer": "",
                "llama_generated_answer": "",
            }

    def process_test_data(self, input_file: str, output_file: str):
        """
        Process the input dataset, generate answers, and save in the desired format.
        """
        try:
            with open(input_file, "r", encoding="utf-8") as f:
                test_data = json.load(f)

            updated_data = []
            for entry in test_data:
                question = entry.get("question", "")
                truth_answer = entry.get("truth", "")  # Rename from 'truth'
                metadata = entry.get("metadata", {})

                if not question.strip():
                    print("Warning: Empty question found, skipping entry.")
                    continue

                generated_data = self.generate_answers_and_docs(question)

                updated_entry = {
                    "query": question,
                    "truth_answer": truth_answer,
                    "retrieved_documents": generated_data["retrieved_documents"],
                    "gpt_generated_answer": generated_data["gpt_generated_answer"],
                    "llama_generated_answer": generated_data["llama_generated_answer"],
                    "metadata": metadata,  # Optional, preserve metadata
                }

                updated_data.append(updated_entry)

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(updated_data, f, ensure_ascii=False, indent=4)

            print(f"Test dataset generated and saved to {output_file}")

        except Exception as e:
            print(f"Error processing test dataset: {e}")


# Example usage
if __name__ == "__main__":
    test_data_generator = TestDatasetGenerator(device="cuda")
    input_file_path = "dataset/test/input_questions.json"  # Your input file with 'question', 'truth', 'metadata'
    output_file_path = "dataset/test/generated_test_data.json"
    test_data_generator.process_test_data(input_file_path, output_file_path)
