import json
from typing import List, Dict
from src.load_config import LoadConfig
from src.pipeline.RAGPipeline import RAGPipeline
from src.components.openAIWrapper import OpenAIWrapper
from src.components.jinaaiWrapper import JinaaiWrapper
from src.components.llama3GermanWrapper import Llama3GermanWrapper
from src.components.postgreSQLWrapper import PostgreSQLWrapper
from src.components.postgreSQLRetriever import HybridRetriever


class TestDatasetAnswerGenerator:
    def __init__(self):
        config_loader = LoadConfig()
        self.rag_pipeline = RAGPipeline(top_k=5)
        self.device = config_loader.device

        self.openai_wrapper = OpenAIWrapper()
        self.jinaai_embedder = JinaaiWrapper()
        self.llama3_german_wrapper = Llama3GermanWrapper(device=self.device)

        self.postgres_wrapper = PostgreSQLWrapper()

        self.hybrid_retriever = HybridRetriever(
            self.postgres_wrapper,
            self.jinaai_embedder,
            enable_reranking=True,
            device=self.device,
        )

    def get_retrieved_docs(self, question: str) -> List[Dict]:
        """
        Retrieve documents for the given question.
        """
        try:
            # Retrieve documents using the hybrid retriever
            retrieved_docs = self.hybrid_retriever.retrive(question, top_k=10, use_reranker=False)
            return [{"id": doc["id"], "text": doc["text"]} for doc in retrieved_docs]
        except Exception as e:
            print(f"Error retrieving documents for question '{question}': {e}")
            return []

    def generate_raw_answers(self, question: str) -> Dict:
        """
        Generate raw answers for the given question.
        """
        try:
            raw_gpt_answers = self.openai_wrapper.raw_text_generator(question)
            raw_llama_answers = self.llama3_german_wrapper.raw_text_generator(question)
            return {
                "raw_gpt_answers": raw_gpt_answers,
                "raw_llama_answers": raw_llama_answers,
            }
        except Exception as e:
            print(f"Error generating raw answers for question '{question}': {e}")
            return {
                "raw_gpt_answers": "",
                "raw_llama_answers": "",
            }

    def generate_rag_answers(self, question: str) -> Dict:
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
            # retrieved_docs = self.get_retrieved_docs(question)

            return {
                "retrieved_documents": retrieved_docs,
                "rag_gpt_answers": gpt_result.get("answer", ""),
                "rag_llama_answers": llama_result.get("answer", ""),
            }

        except Exception as e:
            print(f"Error generating data for question '{question}': {e}")
            return {
                "retrieved_documents": [],
                "rag_gpt_answers": "",
                "rag_llama_answers": "",
            }

    def process_test_data(self, input_file: str, output_file: str):
        """
        Process the input dataset, generate answers, and save in the desired format.
        """
        try:
            with open(input_file, "r", encoding="utf-8") as f:
                test_data = json.load(f)

            updated_data = []
            for idx, entry in enumerate(test_data, start=1):
                print(f"Processing entry #{idx}")

                question = entry.get("question", "")
                truth_answer = entry.get("truth_answer", "")
                # truth_answer_ids = entry.get("truth_answer_ids", [])
                # rag_gpt_answers = entry.get("rag_gpt_answers", "")
                # raw_gpt_answers = entry.get("raw_gpt_answers", "")
                # retrieved_documents = entry.get("retrieved_documents", [])
                meta_data = entry.get("meta_data", {})

                if not question.strip():
                    print(f"Warning: Empty question found at entry #{idx}, skipping.")
                    continue

                rag_generated_data = self.generate_rag_answers(question)
                raw_answers = self.generate_raw_answers(question)

                updated_entry = {
                    "question": question,
                    "truth_answer": truth_answer,
                    # "truth_answer_ids": truth_answer_ids,
                    "retrieved_documents": rag_generated_data.get("retrieved_documents", []),
                    "rag_gpt_answers": rag_generated_data["rag_gpt_answers"],
                    "raw_gpt_answers": raw_answers.get("raw_gpt_answers", ""),
                    "rag_llama_answers": rag_generated_data["rag_llama_answers"],
                    "raw_llama_answers": raw_answers["raw_llama_answers"],
                    "meta_data": meta_data,
                }

                updated_data.append(updated_entry)

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(updated_data, f, ensure_ascii=False, indent=4)

            print(f"Test dataset generated and saved to {output_file}")

        except Exception as e:
            print(f"Error processing test dataset: {e}")


# Example usage
if __name__ == "__main__":
    test_data_generator = TestDatasetAnswerGenerator()
    input_file_path = "dataset/test/questions/question_set_eval.json"
    output_file_path = "dataset/test/generator/answer_set_eval.json"
    test_data_generator.process_test_data(input_file_path, output_file_path)
