import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from collections import defaultdict
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from typing import List, Dict, Any


class EvaluationModule:
    def __init__(self):
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"], use_stemmer=True
        )

    # --- Retriever Evaluation ---
    def evaluate_retriever(
        self, queries: List[Dict], top_k: int = 10
    ) -> Dict[str, float]:
        """
        queries: list of dicts with keys: 'query', 'ground_truth_ids', 'retrieved_documents'
        Each 'retrieved_documents' is a list of dicts with at least an 'id' field.
        """
        recall_scores = []
        precision_scores = []
        mrr_scores = []

        for q in queries:
            gt_ids = set(q["ground_truth_ids"])
            retrieved_ids = [doc["id"] for doc in q["retrieved_documents"][:top_k]]

            hits = [1 if doc_id in gt_ids else 0 for doc_id in retrieved_ids]
            num_relevant = len(gt_ids)

            recall = sum(hits) / num_relevant if num_relevant > 0 else 0
            precision = sum(hits) / top_k
            recall_scores.append(recall)
            precision_scores.append(precision)

            rr = 0
            for rank, hit in enumerate(hits, start=1):
                if hit:
                    rr = 1 / rank
                    break
            mrr_scores.append(rr)

        return {
            "Recall@{}".format(top_k): np.mean(recall_scores),
            "Precision@{}".format(top_k): np.mean(precision_scores),
            "MRR@{}".format(top_k): np.mean(mrr_scores),
        }

    # --- Generation Evaluation ---
    def evaluate_generation(
        self, predictions: List[str], references: List[str]
    ) -> Dict[str, float]:
        """
        predictions: generated answers
        references: ground-truth answers
        """
        bleu_scores = [
            sentence_bleu([ref.split()], pred.split())
            for pred, ref in zip(predictions, references)
        ]

        rouge1 = []
        rouge2 = []
        rougel = []
        for pred, ref in zip(predictions, references):
            scores = self.rouge_scorer.score(ref, pred)
            rouge1.append(scores["rouge1"].fmeasure)
            rouge2.append(scores["rouge2"].fmeasure)
            rougel.append(scores["rougeL"].fmeasure)

        P, R, F1 = bert_score(
            predictions, references, lang="en", rescale_with_baseline=True
        )
        bertscore_f1 = F1.mean().item()

        return {
            "BLEU": np.mean(bleu_scores),
            "ROUGE-1": np.mean(rouge1),
            "ROUGE-2": np.mean(rouge2),
            "ROUGE-L": np.mean(rougel),
            "BERTScore-F1": bertscore_f1,
        }

    # --- End-to-End Evaluation ---
    def evaluate_end_to_end(
        self, predictions: List[str], references: List[str]
    ) -> Dict[str, float]:
        """
        predictions: generated answers
        references: ground-truth answers
        """
        exact_matches = [
            int(pred.strip().lower() == ref.strip().lower())
            for pred, ref in zip(predictions, references)
        ]
        f1s = [self.compute_f1(pred, ref) for pred, ref in zip(predictions, references)]

        return {"Exact Match": np.mean(exact_matches), "F1": np.mean(f1s)}

    def compute_f1(self, pred: str, ref: str) -> float:
        pred_tokens = pred.lower().split()
        ref_tokens = ref.lower().split()

        common = set(pred_tokens) & set(ref_tokens)
        if len(common) == 0:
            return 0.0

        precision = len(common) / len(pred_tokens)
        recall = len(common) / len(ref_tokens)
        return 2 * precision * recall / (precision + recall)
