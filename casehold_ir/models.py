"""Ranking models: TF-IDF, BM25 (see index.py), dense retrieval, RRF and a cross-encoder."""
from pathlib import Path

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer

from .index import BM25, InvertedIndex
from .preprocess import tokenize

CACHE = Path(__file__).resolve().parent.parent / "cache"


def device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


class TfidfModel:
    """Vector space model: (1 + log tf) * idf weights, unit length, cosine similarity."""

    def __init__(self, remove_stop: bool = True, use_stem: bool = True):
        self.vectorizer = TfidfVectorizer(
            tokenizer=lambda t: tokenize(t, remove_stop, use_stem),
            lowercase=False, token_pattern=None, sublinear_tf=True, norm="l2")

    def fit(self, docs: list[str]) -> "TfidfModel":
        self.doc_matrix = self.vectorizer.fit_transform(docs)
        return self

    def score(self, queries: list[str]) -> np.ndarray:
        return (self.vectorizer.transform(queries) @ self.doc_matrix.T).toarray()


def build_bm25(docs: list[str], k1: float, b: float) -> BM25:
    return BM25(InvertedIndex([tokenize(d) for d in docs]), k1=k1, b=b)


class DenseModel:
    """Sentence Transformer bi-encoder. Embeddings are cached on disk per (model, texts)."""

    def __init__(self, name: str, query_prefix: str = ""):
        from sentence_transformers import SentenceTransformer
        self.name, self.query_prefix = name, query_prefix
        self.model = SentenceTransformer(name, device=device())

    def encode(self, texts: list[str], tag: str, batch_size: int = 128) -> np.ndarray:
        safe = self.name.replace("/", "__").replace(".", "_")
        path = CACHE / f"{safe}__{tag}.npy"
        if path.exists():
            emb = np.load(path)
            if len(emb) == len(texts):
                return emb
        emb = self.model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                                convert_to_numpy=True, show_progress_bar=True).astype(np.float32)
        CACHE.mkdir(exist_ok=True)
        np.save(path, emb)
        return emb

    def score(self, queries: list[str], docs: list[str], tag: str) -> np.ndarray:
        q = self.encode([self.query_prefix + x for x in queries], f"{tag}_queries")
        d = self.encode(docs, f"{tag}_docs")
        return q @ d.T


def top_k(scores: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k best documents per row, best first; ties broken by lower doc id."""
    k = min(k, scores.shape[1])
    part = np.argpartition(-scores, k - 1, axis=1)[:, :k]
    out = np.empty_like(part)
    for i in range(len(scores)):
        cand = part[i]
        order = np.lexsort((cand, -scores[i, cand]))
        out[i] = cand[order]
    return out


def rrf(ranked_lists: list[np.ndarray], k: int = 60, depth: int = 1000) -> np.ndarray:
    """Reciprocal Rank Fusion: score(d) = sum over rankers of 1 / (k + rank of d)."""
    n_queries = ranked_lists[0].shape[0]
    fused = np.empty((n_queries, depth), dtype=np.int64)
    for qi in range(n_queries):
        scores: dict[int, float] = {}
        for lst in ranked_lists:
            for rank, doc in enumerate(lst[qi], start=1):
                scores[int(doc)] = scores.get(int(doc), 0.0) + 1.0 / (k + rank)
        best = sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:depth]
        row = [d for d, _ in best]
        row += [-1] * (depth - len(row))
        fused[qi] = row
    return fused


def cross_encoder_rerank(queries: list[str], docs: list[str], ranked: np.ndarray,
                         depth: int = 20,
                         name: str = "cross-encoder/ms-marco-MiniLM-L6-v2") -> np.ndarray:
    """Re-score the top `depth` documents of each list with a cross-encoder."""
    from sentence_transformers import CrossEncoder
    model = CrossEncoder(name, device=device(), max_length=512)
    pairs = [(queries[qi], docs[d]) for qi in range(len(queries)) for d in ranked[qi, :depth]]
    scores = model.predict(pairs, batch_size=64, show_progress_bar=True).reshape(len(queries), depth)
    out = ranked.copy()
    for qi in range(len(queries)):
        order = np.argsort(-scores[qi], kind="stable")
        out[qi, :depth] = ranked[qi, :depth][order]
    return out
