import json
import time
from typing import List, Dict
from openai import APIConnectionError, RateLimitError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from ragas import evaluate
from ragas import EvaluationDataset
from ragas.dataset_schema import EvaluationResult
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness
from src.load_config import LoadConfig

# from langchain_openai import ChatOpenAI
from langchain_openai import AzureChatOpenAI


class RAGASEvaluator:
    def __init__(self, llm, batch_size=5, timeout=60):
        self.llm = LangchainLLMWrapper(llm)
        self.metrics = [LLMContextRecall(), Faithfulness(), FactualCorrectness()]
        self.batch_size = batch_size
        self.timeout = timeout

    def _format_to_ragas_dataset(
        self, test_data: List[Dict], answer_key: str
    ) -> List[Dict]:
        formatted = []
        for item in test_data:
            question = item.get("question", "[NO QUESTION]")
            response = item.get(answer_key)
            reference = item.get("truth_answer")
            contexts = item.get("retrieved_documents", [])

            # Validate fields
            if not response:
                print(f"⚠️ Skipping due to missing response for question: {question}")
                continue
            if not reference:
                print(f"⚠️ Skipping due to missing reference for question: {question}")
                continue
            if not contexts or not all(doc.get("text") for doc in contexts):
                print(
                    f"⚠️ Skipping due to missing/invalid retrieved contexts for question: {question}"
                )
                continue

            formatted.append(
                {
                    "user_input": question,
                    "response": response,
                    "reference": reference,
                    "retrieved_contexts": [
                        doc["text"] for doc in contexts if doc.get("text")
                    ],
                }
            )
        return formatted

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=30),
        retry=retry_if_exception_type(
            (TimeoutError, APIConnectionError, RateLimitError)
        ),
    )
    def _evaluate_with_retry(
        self, evaluation_dataset: EvaluationDataset
    ) -> EvaluationResult:
        return evaluate(dataset=evaluation_dataset, metrics=self.metrics, llm=self.llm)

    def evaluate(self, test_data: List[Dict], answer_key: str) -> Dict[str, float]:
        all_results: List[EvaluationResult] = []
        total_batches = (len(test_data) + self.batch_size - 1) // self.batch_size

        for i in range(0, len(test_data), self.batch_size):
            batch = test_data[i : i + self.batch_size]
            batch_num = (i // self.batch_size) + 1

            print(
                f"\n🔄 Starting batch {batch_num}/{total_batches} for {answer_key}..."
            )

            try:
                start_time = time.time()
                formatted_batch = self._format_to_ragas_dataset(batch, answer_key)
                if not formatted_batch:
                    print("⚠️ Skipping empty batch.")
                    continue

                evaluation_dataset = EvaluationDataset.from_list(formatted_batch)

                result = self._evaluate_with_retry(evaluation_dataset)

                print(
                    f"✅ Batch {batch_num} completed in {time.time() - start_time:.2f}s"
                )
                all_results.append(result)

            except Exception as e:
                print(f"❌ Batch {batch_num} failed: {str(e)}")
                continue

        # Aggregate results
        final_scores = {}
        try:
            print("\n📊 Aggregating metrics...")

            all_score_dicts = []
            for res in all_results:
                if isinstance(res.scores, list):
                    all_score_dicts.extend(res.scores)
                elif isinstance(res.scores, dict):
                    all_score_dicts.append(res.scores)

            if not all_score_dicts:
                raise ValueError("No scores to aggregate.")

            metric_names = all_score_dicts[0].keys()
            for metric in metric_names:
                values = [score[metric] for score in all_score_dicts if metric in score]
                final_scores[metric] = sum(values) / len(values)

        except Exception as e:
            print(f"💥 Failed to aggregate results: {e}")
            raise

        return final_scores

    def save_results(self, all_scores: Dict[str, Dict[str, float]], file_path: str):
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(all_scores, f, indent=4)


if __name__ == "__main__":
    try:
        with open("dataset/test/generator/rag_answers.json", encoding="utf-8") as f:
            test_data = json.load(f)

        config_loader = LoadConfig()

        llm = AzureChatOpenAI(
            openai_api_version=config_loader.azure_openai_api_version,
            azure_deployment=config_loader.llm_model_name,
            azure_endpoint=config_loader.azure_openai_api_base,
            api_key=config_loader.azure_openai_api_key,
            temperature=0,
            request_timeout=60,
        )

        evaluator = RAGASEvaluator(llm=llm, batch_size=3)

        all_scores = {}

        # Evaluate RAG LLaMA answers
        rag_llama_scores = evaluator.evaluate(test_data, answer_key="rag_llama_answers")
        print("\n📊 Final Scores for RAG LLaMA Answers:")
        for metric, value in rag_llama_scores.items():
            print(f"{metric}: {value:.4f}")
        all_scores["rag_llama_answers"] = rag_llama_scores

        # Evaluate RAW LLaMA answers
        raw_llama_scores = evaluator.evaluate(test_data, answer_key="raw_llama_answers")
        print("\n📊 Final Scores for RAW LLaMA Answers:")
        for metric, value in raw_llama_scores.items():
            print(f"{metric}: {value:.4f}")
        all_scores["raw_llama_answers"] = raw_llama_scores

        # Evaluate RAG GPT answers
        rag_gpt_scores = evaluator.evaluate(test_data, answer_key="rag_gpt_answers")
        print("\n📊 Final Scores for RAG GPT Answers:")
        for metric, value in rag_gpt_scores.items():
            print(f"{metric}: {value:.4f}")
        all_scores["rag_gpt_answers"] = rag_gpt_scores

        # Evaluate RAW GPT answers
        raw_gpt_scores = evaluator.evaluate(test_data, answer_key="raw_gpt_answers")
        print("\n📊 Final Scores for RAW GPT Answers:")
        for metric, value in raw_gpt_scores.items():
            print(f"{metric}: {value:.4f}")
        all_scores["raw_gpt_answers"] = raw_gpt_scores

        # Save all results to a single file
        evaluator.save_results(all_scores, "dataset/test/matrics/ragas_scores.json")

    except Exception as e:
        print(f"💥 Evaluation failed completely: {e}")
