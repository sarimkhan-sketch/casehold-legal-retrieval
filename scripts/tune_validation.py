"""Choose every parameter on the validation split (MRR), then freeze them.

Writes results/validation.json, which scripts/evaluate.py reads.
"""
import json
from pathlib import Path

from casehold_ir.data import load_split
from casehold_ir.metrics import ranks_from_lists, ranks_from_scores, summary
from casehold_ir.models import DenseModel, TfidfModel, build_bm25, rrf, top_k
from casehold_ir.preprocess import last_words, tokenize

ROOT = Path(__file__).resolve().parent.parent
BGE_INSTRUCTION = "Represent this sentence for searching relevant passages: "


def mrr(scores, relevant) -> float:
    return round(summary(ranks_from_scores(scores, relevant))["MAP"], 4)


def main():
    val = load_split("validation")
    out: dict = {}

    # 1. How much of the citing context to use as the query (BM25, default k1 and b).
    bm25 = build_bm25(val.docs, k1=1.2, b=0.75)
    out["query_length"] = {}
    for label, n in [("full", None), ("last_64", 64), ("last_32", 32)]:
        toks = [tokenize(last_words(q, n)) for q in val.queries]
        out["query_length"][label] = mrr(bm25.score_batch(toks), val.relevant)
    print("query length:", out["query_length"])
    best_len = max(out["query_length"], key=out["query_length"].get)
    n_words = {"full": None, "last_64": 64, "last_32": 32}[best_len]
    queries = [last_words(q, n_words) for q in val.queries]
    q_tokens = [tokenize(q) for q in queries]

    # 2. BM25 grid over k1 and b.
    grid = {}
    for k1 in [0.2, 0.3, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0]:
        for b in [0.3, 0.5, 0.7, 0.8, 0.9, 1.0]:
            model = build_bm25(val.docs, k1=k1, b=b)
            grid[f"k1={k1},b={b}"] = mrr(model.score_batch(q_tokens), val.relevant)
    best = max(grid, key=grid.get)
    k1, b = (float(x.split("=")[1]) for x in best.split(","))
    out["bm25_grid"] = grid
    out["bm25"] = {"k1": k1, "b": b, "mrr": grid[best]}
    print("bm25 best:", out["bm25"])

    # 3. TF-IDF preprocessing variants.
    out["tfidf_variants"] = {}
    for label, stop, stem in [("tokens_only", False, False), ("stop_words", True, False),
                              ("stop_words+stem", True, True)]:
        model = TfidfModel(remove_stop=stop, use_stem=stem).fit(val.docs)
        out["tfidf_variants"][label] = mrr(model.score(queries), val.relevant)
    print("tfidf:", out["tfidf_variants"])
    best_tfidf = max(out["tfidf_variants"], key=out["tfidf_variants"].get)
    out["tfidf"] = {"remove_stop": best_tfidf != "tokens_only", "use_stem": best_tfidf.endswith("stem")}

    # 4. Which small dense model.
    out["dense_models"] = {}
    for label, name, prefix in [
            ("all-MiniLM-L6-v2", "sentence-transformers/all-MiniLM-L6-v2", ""),
            ("bge-small-en-v1.5", "BAAI/bge-small-en-v1.5", ""),
            ("bge-small-en-v1.5 + instruction", "BAAI/bge-small-en-v1.5", BGE_INSTRUCTION)]:
        model = DenseModel(name, query_prefix=prefix)
        tag = "val" + ("_instr" if prefix else "")
        out["dense_models"][label] = mrr(model.score(queries, val.docs, tag), val.relevant)
    print("dense:", out["dense_models"])

    # 5. RRF constant k for the three-way fusion.
    tfidf = TfidfModel(**out["tfidf"]).fit(val.docs)
    bm25 = build_bm25(val.docs, k1=k1, b=b)
    dense = DenseModel("sentence-transformers/all-MiniLM-L6-v2")
    lists = [top_k(tfidf.score(queries), 1000), top_k(bm25.score_batch(q_tokens), 1000),
             top_k(dense.score(queries, val.docs, "val"), 1000)]
    out["rrf_k"] = {}
    for k in [1, 5, 10, 20, 30, 60, 100]:
        ranks = ranks_from_lists(rrf(lists, k=k), val.relevant)
        out["rrf_k"][str(k)] = round(summary(ranks)["MAP"], 4)
    out["rrf_best_k"] = int(max(out["rrf_k"], key=out["rrf_k"].get))
    print("rrf k:", out["rrf_k"])

    out["query_words"] = n_words
    (ROOT / "results" / "validation.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
