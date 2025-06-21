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

from langchain_openai import ChatOpenAI


class RAGASEvaluator:
    def __init__(self, llm, batch_size=5, timeout=60):
        self.llm = LangchainLLMWrapper(llm)
        self.metrics = [LLMContextRecall(), Faithfulness(), FactualCorrectness()]
        self.batch_size = batch_size
        self.timeout = timeout

    def _format_to_ragas_dataset(self, test_data: List[Dict]) -> List[Dict]:
        formatted = []
        for item in test_data:
            formatted.append(
                {
                    "user_input": item["question"],
                    "response": item["gpt_generated_answer"],
                    "reference": item["truth_answer"],
                    "retrieved_contexts": [
                        doc["text"] for doc in item["retrieved_documents"][:5]
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
        try:
            return evaluate(
                dataset=evaluation_dataset, metrics=self.metrics, llm=self.llm
            )
        except TimeoutError:
            print("⚠️ TimeoutError during evaluation.")
            raise

    def evaluate(self, test_data: List[Dict]) -> Dict[str, float]:
        all_results: List[EvaluationResult] = []
        total_batches = (len(test_data) + self.batch_size - 1) // self.batch_size

        for i in range(0, len(test_data), self.batch_size):
            batch = test_data[i : i + self.batch_size]
            batch_num = (i // self.batch_size) + 1

            print(f"🔄 Starting batch {batch_num}/{total_batches}...")

            try:
                start_time = time.time()
                formatted_batch = self._format_to_ragas_dataset(batch)
                evaluation_dataset = EvaluationDataset.from_list(formatted_batch)
                print("DEBUG: Evaluation dataset created")

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
            print("\nDEBUG: Aggregating metrics from results...")

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

            print("\n🎉 Evaluation Complete")

        except Exception as e:
            print(f"💥 Failed to aggregate results: {e}")
            raise

        return final_scores

    def save_results(self, scores: Dict[str, float], file_path: str):
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(scores, f, indent=4)


if __name__ == "__main__":
    try:
        with open(
            "dataset/test/generator/generated_gpt_answers_2.json", encoding="utf-8"
        ) as f:
            test_data = json.load(f)

        llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0,
            request_timeout=60,
        )

        evaluator = RAGASEvaluator(llm=llm, batch_size=5)

        scores = evaluator.evaluate(test_data)

        print("\n📊 Final RAGAS Evaluation Scores:")
        for metric, value in scores.items():
            print(f"{metric}: {value:.4f}")

        evaluator.save_results(scores, "dataset/test/generator/ragas_metrics.json")

    except Exception as e:
        print(f"💥 Evaluation failed completely: {e}")
