import os
import json
from typing import List, Dict, Optional
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
import evaluate
import numpy as np
from collections import defaultdict


class RAGEvaluator:
    def __init__(self):
        # Initialize metrics
        self.rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        self.smoothie = SmoothingFunction().method4
        self.bert_score = evaluate.load("bertscore")

    def compute_precision_at_k(
        self, relevant_ids: List[int], retrieved_ids: List[int], k: int
    ) -> float:
        retrieved_k = retrieved_ids[:k]
        return len(set(retrieved_k) & set(relevant_ids)) / k if k > 0 else 0.0

    def compute_recall_at_k(
        self, relevant_ids: List[int], retrieved_ids: List[int], k: int
    ) -> float:
        retrieved_k = retrieved_ids[:k]
        return (
            len(set(retrieved_k) & set(relevant_ids)) / len(relevant_ids)
            if relevant_ids
            else 0.0
        )

    def compute_mrr(self, relevant_ids: List[int], retrieved_ids: List[int]) -> float:
        for rank, doc_id in enumerate(retrieved_ids, start=1):
            if doc_id in relevant_ids:
                return 1.0 / rank
        return 0.0

    def compute_dcg(self, relevances: List[int]) -> float:
        return sum(rel / np.log2(idx + 2) for idx, rel in enumerate(relevances))

    def compute_ndcg_at_k(
        self, relevant_ids: List[int], retrieved_ids: List[int], k: int
    ) -> float:
        retrieved_k = retrieved_ids[:k]
        relevances = [1 if doc_id in relevant_ids else 0 for doc_id in retrieved_k]
        dcg = self.compute_dcg(relevances)
        idcg = self.compute_dcg(sorted(relevances, reverse=True))
        return dcg / idcg if idcg > 0 else 0.0

    def evaluate_retrieval(
        self, test_data: List[Dict], k_values: List[int] = [1, 3, 5, 10]
    ) -> Dict[str, float]:
        results = defaultdict(dict)

        for k in k_values:
            precisions, recalls, ndcgs = [], [], []
            mrrs = []  # MRR doesn't depend on k

            for item in test_data:
                # Skip if no ground truth or retrieved documents
                if not item.get("ground_truth_ids") or not item.get(
                    "retrieved_documents"
                ):
                    continue

                relevant = item["ground_truth_ids"]
                retrieved = [doc["id"] for doc in item["retrieved_documents"]]

                precisions.append(self.compute_precision_at_k(relevant, retrieved, k))
                recalls.append(self.compute_recall_at_k(relevant, retrieved, k))
                ndcgs.append(self.compute_ndcg_at_k(relevant, retrieved, k))

                # Only compute MRR once per item
                if k == k_values[0]:
                    mrrs.append(self.compute_mrr(relevant, retrieved))

            results[f"Precision@{k}"] = np.mean(precisions) if precisions else 0.0
            results[f"Recall@{k}"] = np.mean(recalls) if recalls else 0.0
            results[f"NDCG@{k}"] = np.mean(ndcgs) if ndcgs else 0.0

            if k == k_values[0]:
                results["MRR"] = np.mean(mrrs) if mrrs else 0.0

        return dict(results)

    def evaluate_generation(self, test_data: List[Dict]) -> Dict[str, float]:
        # Prepare data
        references = []
        predictions = []

        for item in test_data:
            if "truth_answer" not in item or "gpt_generated_answer" not in item:
                continue
            references.append(item["truth_answer"])
            predictions.append(item["gpt_generated_answer"])

        if not references or not predictions:
            return {"BLEU": 0.0, "ROUGE-L": 0.0, "METEOR": 0.0, "BERTScore": 0.0}

        # Compute metrics
        bleu_scores, rouge_l_scores, meteor_scores = [], [], []

        for ref, pred in zip(references, predictions):
            try:
                bleu_scores.append(
                    sentence_bleu(
                        [ref.split()], pred.split(), smoothing_function=self.smoothie
                    )
                )
                rouge_score = self.rouge.score(ref, pred)["rougeL"].fmeasure
                rouge_l_scores.append(rouge_score)
                meteor_scores.append(meteor_score([ref], pred))
            except:
                # Skip if there's an error in calculation
                continue

        # Compute BERTScore
        bert = self.bert_score.compute(
            predictions=predictions,
            references=references,
            lang="de",  # Assuming German text
        )

        return {
            "BLEU": np.mean(bleu_scores) if bleu_scores else 0.0,
            "ROUGE-L": np.mean(rouge_l_scores) if rouge_l_scores else 0.0,
            "METEOR": np.mean(meteor_scores) if meteor_scores else 0.0,
            "BERTScore": np.mean(bert["f1"]) if bert["f1"] else 0.0,
        }

    def evaluate_rag(
        self, test_data: List[Dict], k_values: List[int] = [1, 3, 5, 10]
    ) -> Dict[str, Dict[str, float]]:
        return {
            "retrieval_metrics": self.evaluate_retrieval(test_data, k_values),
            "generation_metrics": self.evaluate_generation(test_data),
        }

    def print_metrics(self, metrics: Dict[str, Dict[str, float]]):
        """Pretty print evaluation metrics"""
        print("=== Retrieval Metrics ===")
        for metric, value in metrics["retrieval_metrics"].items():
            print(f"{metric:15}: {value:.4f}")

        print("\n=== Generation Metrics ===")
        for metric, value in metrics["generation_metrics"].items():
            print(f"{metric:20}: {value:.4f}")


if __name__ == "__main__":
    evaluation_data_file_path = "dataset/test/generated_test_data.json"

    try:
        with open(evaluation_data_file_path, "r", encoding="utf-8") as f:
            test_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found at {evaluation_data_file_path}")
        exit(1)
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in file {evaluation_data_file_path}")
        exit(1)

    evaluator = RAGEvaluator()

    print("📊 Evaluating RAG System...")
    metrics = evaluator.evaluate_rag(test_data)
    evaluator.print_metrics(metrics)
