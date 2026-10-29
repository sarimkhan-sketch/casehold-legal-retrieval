"""Evaluation with exactly one relevant document per query.

With one relevant document at rank r: AP = 1/r (so MAP = MRR),
Precision@k = 1/k if r <= k, Recall@k = 1 if r <= k, nDCG@k = 1/log2(r + 1) if r <= k.
"""
import numpy as np

NOT_FOUND = 10**9


def ranks_from_scores(scores: np.ndarray, relevant: np.ndarray) -> np.ndarray:
    """1-based rank of the relevant document in the full ranking (ties: lower doc id first)."""
    rel_scores = scores[np.arange(len(scores)), relevant][:, None]
    greater = (scores > rel_scores).sum(axis=1)
    cols = np.arange(scores.shape[1])[None, :]
    tied_before = ((scores == rel_scores) & (cols < relevant[:, None])).sum(axis=1)
    return greater + tied_before + 1


def ranks_from_lists(ranked: np.ndarray, relevant: np.ndarray) -> np.ndarray:
    """1-based rank of the relevant document in a truncated list, NOT_FOUND if absent."""
    hits = ranked == relevant[:, None]
    found = hits.any(axis=1)
    return np.where(found, hits.argmax(axis=1) + 1, NOT_FOUND)


def average_precision(ranks: np.ndarray) -> np.ndarray:
    return 1.0 / ranks


def summary(ranks: np.ndarray) -> dict:
    ranks = np.asarray(ranks)
    ndcg10 = np.where(ranks <= 10, 1.0 / np.log2(ranks + 1), 0.0)
    return {
        "P@10": float(np.mean((ranks <= 10) / 10)),
        "R@1": float(np.mean(ranks <= 1)),
        "R@10": float(np.mean(ranks <= 10)),
        "R@100": float(np.mean(ranks <= 100)),
        "MAP": float(np.mean(average_precision(ranks))),
        "nDCG@10": float(np.mean(ndcg10)),
        "median_rank": float(np.median(ranks)),
    }


def five_choice_summary(ranks: np.ndarray) -> dict:
    ranks = np.asarray(ranks)
    return {
        "Accuracy": float(np.mean(ranks == 1)),
        "MRR": float(np.mean(1.0 / ranks)),
        "R@2": float(np.mean(ranks <= 2)),
        "nDCG@5": float(np.mean(1.0 / np.log2(ranks + 1))),
    }


def pr_curve(ranks: np.ndarray, k_max: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """Mean Precision@k and mean Recall@k for k = 1..k_max."""
    ks = np.arange(1, k_max + 1)
    hit = np.asarray(ranks)[:, None] <= ks[None, :]
    return (hit / ks).mean(axis=0), hit.mean(axis=0)
