"""Inverted index with Okapi BM25 scoring."""
import math
from collections import Counter, defaultdict

import numpy as np
from scipy import sparse


class InvertedIndex:
    """term -> postings list of (doc id, term frequency), plus document lengths."""

    def __init__(self, tokenized_docs: list[list[str]]):
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.doc_len = np.array([len(toks) for toks in tokenized_docs], dtype=np.float64)
        for doc_id, toks in enumerate(tokenized_docs):
            for term, tf in Counter(toks).items():
                self.postings[term].append((doc_id, tf))
        self.postings = dict(self.postings)
        self.n_docs = len(tokenized_docs)
        self.avgdl = float(self.doc_len.mean())
        self.vocab = {t: i for i, t in enumerate(sorted(self.postings))}

    def stats(self) -> dict:
        n_postings = sum(len(p) for p in self.postings.values())
        return {"terms": len(self.postings), "postings": n_postings,
                "avg_terms_per_doc": float(self.doc_len.mean()),
                "avg_postings_per_term": n_postings / len(self.postings)}

    def idf(self, term: str) -> float:
        df = len(self.postings.get(term, ()))
        return math.log(1 + (self.n_docs - df + 0.5) / (df + 0.5))


class BM25:
    def __init__(self, index: InvertedIndex, k1: float = 1.2, b: float = 0.75):
        self.index, self.k1, self.b = index, k1, b
        self._matrix = None

    def score_loop(self, query_tokens: list[str]) -> dict[int, float]:
        """Score one query by walking the postings lists (used by the search app)."""
        idx, k1, b = self.index, self.k1, self.b
        scores: dict[int, float] = defaultdict(float)
        # The sum runs over query tokens, so a term repeated in the query counts again.
        for term, qtf in Counter(query_tokens).items():
            postings = idx.postings.get(term)
            if not postings:
                continue
            idf = idx.idf(term)
            for doc_id, tf in postings:
                norm = k1 * (1 - b + b * idx.doc_len[doc_id] / idx.avgdl)
                scores[doc_id] += qtf * idf * tf * (k1 + 1) / (tf + norm)
        return scores

    def doc_term_matrix(self) -> sparse.csr_matrix:
        """BM25 weight of every (doc, term) pair, so a batch of queries is one product."""
        if self._matrix is None:
            idx, k1, b = self.index, self.k1, self.b
            rows, cols, vals = [], [], []
            for term, col in idx.vocab.items():
                idf = idx.idf(term)
                for doc_id, tf in idx.postings[term]:
                    norm = k1 * (1 - b + b * idx.doc_len[doc_id] / idx.avgdl)
                    rows.append(doc_id)
                    cols.append(col)
                    vals.append(idf * tf * (k1 + 1) / (tf + norm))
            self._matrix = sparse.csr_matrix((vals, (rows, cols)),
                                             shape=(idx.n_docs, len(idx.vocab)), dtype=np.float32)
        return self._matrix

    def query_matrix(self, tokenized_queries: list[list[str]]) -> sparse.csr_matrix:
        vocab = self.index.vocab
        rows, cols, vals = [], [], []
        for qi, toks in enumerate(tokenized_queries):
            for term, qtf in Counter(toks).items():
                if term in vocab:
                    rows.append(qi)
                    cols.append(vocab[term])
                    vals.append(qtf)
        return sparse.csr_matrix((np.asarray(vals, dtype=np.float32), (rows, cols)),
                                 shape=(len(tokenized_queries), len(vocab)))

    def score_batch(self, tokenized_queries: list[list[str]]) -> np.ndarray:
        q = self.query_matrix(tokenized_queries)
        return (q @ self.doc_term_matrix().T).toarray()
