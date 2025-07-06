import os
import json
import evaluate
import numpy as np
from collections import defaultdict
from rouge_score import rouge_scorer
from nltk.tokenize import word_tokenize
from typing import List, Dict, Optional
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score


class RAGEvaluator:
    def __init__(self):
        # Initialize metrics
        self.rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        self.smoothie = SmoothingFunction().method4
        self.bert_score = evaluate.load("bertscore")

    def compute_hit_rate_at_k(
        self, relevant_ids: List[int], retrieved_ids: List[int], k: int
    ) -> float:
        """Compute whether any relevant document is in the top-k retrieved documents"""
        retrieved_k = retrieved_ids[:k]
        return 1.0 if any(doc_id in relevant_ids for doc_id in retrieved_k) else 0.0

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
        self, test_data: List[Dict], k_values: List[int] = [10]
    ) -> Dict[str, float]:
        results = defaultdict(dict)

        for k in k_values:
            precisions, recalls, ndcgs, hit_rates = [], [], [], []
            mrrs = []  # MRR doesn't depend on k

            for item in test_data:
                # Skip if no ground truth or retrieved documents
                if not item.get("truth_answer_ids") or not item.get(
                    "retrieved_documents"
                ):
                    continue

                relevant = item["truth_answer_ids"]
                retrieved = [doc["id"] for doc in item["retrieved_documents"]]

                precisions.append(self.compute_precision_at_k(relevant, retrieved, k))
                recalls.append(self.compute_recall_at_k(relevant, retrieved, k))
                ndcgs.append(self.compute_ndcg_at_k(relevant, retrieved, k))
                hit_rates.append(self.compute_hit_rate_at_k(relevant, retrieved, k))

                # Only compute MRR once per item
                if k == k_values[0]:
                    mrrs.append(self.compute_mrr(relevant, retrieved))

            results[f"Precision@{k}"] = (
                round(np.mean(precisions), 3) if precisions else 0.0
            )
            results[f"Recall@{k}"] = round(np.mean(recalls), 3) if recalls else 0.0
            results[f"NDCG@{k}"] = round(np.mean(ndcgs), 3) if ndcgs else 0.0
            results[f"HitRate@{k}"] = round(np.mean(hit_rates), 3) if hit_rates else 0.0

            if k == k_values[0]:
                results["MRR"] = round(np.mean(mrrs), 3) if mrrs else 0.0

        return dict(results)

    def evaluate_generation(self, test_data: List[Dict]) -> Dict[str, Dict[str, float]]:
        # Store all generations and results by answer type
        answer_types = [
            "rag_llama_answers",
            "raw_llama_answers",
            "rag_gpt_answers",
            "raw_gpt_answers",
        ]

        # Prepare results dict
        results = {}

        for answer_type in answer_types:
            references = []
            predictions = []

            for item in test_data:
                if "truth_answer" not in item or answer_type not in item:
                    continue
                references.append(item["truth_answer"])
                predictions.append(item[answer_type])

            if not references or not predictions:
                results[answer_type] = {
                    "BLEU": 0.0,
                    "ROUGE-L": 0.0,
                    "METEOR": 0.0,
                    "BERTScore": 0.0,
                }
                continue

            # Tokenize for BLEU
            refs_tokenized = [[word_tokenize(ref)] for ref in references]
            preds_tokenized = [word_tokenize(pred) for pred in predictions]

            try:
                bleu = corpus_bleu(
                    refs_tokenized, preds_tokenized, smoothing_function=self.smoothie
                )
            except:
                bleu = 0.0

            rouge_l_scores = []
            meteor_scores = []
            for ref, pred in zip(references, predictions):
                try:
                    rouge_l_scores.append(
                        self.rouge.score(ref, pred)["rougeL"].fmeasure
                    )
                    meteor_scores.append(meteor_score([ref], pred, language="de"))
                except Exception as e:
                    print(f"Error calculating ROUGE/METEOR: {e}")
                    continue

            bert = self.bert_score.compute(
                predictions=predictions,
                references=references,
                lang="de",
                model_type="bert-base-multilingual-cased",
            )

            results[answer_type] = {
                "BLEU": round(bleu, 4),
                "ROUGE-L": round(np.mean(rouge_l_scores), 4) if rouge_l_scores else 0.0,
                "METEOR": round(np.mean(meteor_scores), 4) if meteor_scores else 0.0,
                "BERTScore": round(np.mean(bert["f1"]), 4) if bert["f1"] else 0.0,
            }

        return results

    def evaluate_rag(
        self, test_data: List[Dict], k_values: List[int] = [1, 2, 3, 5, 10]
    ) -> Dict[str, Dict[str, float]]:
        return {
            # "retrieval_metrics": self.evaluate_retrieval(test_data, k_values),
            "generation_metrics": self.evaluate_generation(test_data),
        }

    def print_metrics(self, metrics):
        """Pretty print evaluation metrics"""

        # if "retrieval_metrics" in metrics:
        #     print("=== Retrieval Metrics ===")
        #     for metric, value in metrics["retrieval_metrics"].items():
        #         print(f"{metric:15}: {value:.4f}")

        if "generation_metrics" in metrics:
            print("\n=== Generation Metrics ===")
            for answer_type, metric_dict in metrics["generation_metrics"].items():
                print(f"\n-- {answer_type} --")
                for metric, value in metric_dict.items():
                    print(f"{metric:15}: {value:.4f}")

    def save_metrics_to_file(
        self, metrics: Dict[str, Dict[str, float]], file_path: str
    ):
        """Save evaluation metrics to a JSON file"""
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=4)
            print(f"Metrics saved to {file_path}")
        except IOError as e:
            print(f"Error saving metrics to file: {e}")


if __name__ == "__main__":
    evaluation_data_file_path = "dataset/test/generator/rag_answers.json"

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
    evaluator.save_metrics_to_file(
        metrics, "dataset/test/generator/gen_rag_matrics.json"
    )
