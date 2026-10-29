"""Run every model on the CaseHOLD test split and write tables, figures and error analysis.

Reads the parameters frozen by scripts/tune_validation.py. Outputs go to results/.
"""
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from casehold_ir import plotting
from casehold_ir.data import load_split
from casehold_ir.metrics import (average_precision, five_choice_summary, pr_curve,
                                 ranks_from_lists, ranks_from_scores, summary)
from casehold_ir.models import (DenseModel, TfidfModel, build_bm25, cross_encoder_rerank, rrf,
                                top_k)
from casehold_ir.preprocess import last_words, tokenize

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
FIG = RES / "figures"
BASE_DENSE = "sentence-transformers/all-MiniLM-L6-v2"
FINETUNED = ROOT / "models" / "minilm-casehold"
DEPTH = 1000

REQUIRED = ["TF-IDF", "BM25", "Sentence Transformer", "Hybrid RRF"]
RUN_NAMES = {
    "TF-IDF": "TF-IDF",
    "BM25": "BM25",
    "Sentence Transformer": "Sentence Transformer",
    "Hybrid RRF": "Hybrid RRF (TF-IDF + BM25 + Dense)",
    "Hybrid RRF best k": "Hybrid RRF, k = {k}",
    "Hybrid RRF BM25+Dense": "Hybrid RRF (BM25 + Dense)",
    "Hybrid + cross-encoder": "Hybrid + cross-encoder",
    "Dense, fine-tuned": "Dense, fine-tuned (ext.)",
    "Hybrid RRF (BM25 + fine-tuned)": "Hybrid RRF (BM25 + fine-tuned Dense)",
}


def timed(fn, *args, **kwargs):
    start = time.perf_counter()
    out = fn(*args, **kwargs)
    return out, round(time.perf_counter() - start, 2)


def option_rank(option_scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Rank of the correct option among the five (ties: earlier position first)."""
    correct = option_scores[np.arange(len(labels)), labels][:, None]
    pos = np.arange(5)[None, :]
    return ((option_scores > correct) | ((option_scores == correct) & (pos < labels[:, None]))).sum(1) + 1


def option_rrf(option_score_list: list[np.ndarray], k: int) -> np.ndarray:
    fused = np.zeros_like(option_score_list[0])
    for s in option_score_list:
        order = np.argsort(-s, axis=1, kind="stable")
        ranks = np.empty_like(order)
        np.put_along_axis(ranks, order, np.arange(1, 6)[None, :].repeat(len(s), 0), axis=1)
        fused += 1.0 / (k + ranks)
    return fused


def main():
    plotting.setup()
    FIG.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((RES / "validation.json").read_text())
    test = load_split("test")
    queries = [last_words(q, cfg["query_words"]) for q in test.queries]
    q_tokens = [tokenize(q) for q in queries]
    rel = test.relevant
    timings, scores, lists, ranks = {}, {}, {}, {}

    # Sparse models
    tfidf, timings["tfidf_index_s"] = timed(lambda: TfidfModel(**cfg["tfidf"]).fit(test.docs))
    scores["TF-IDF"], timings["tfidf_score_s"] = timed(tfidf.score, queries)
    bm25, timings["bm25_index_s"] = timed(build_bm25, test.docs, cfg["bm25"]["k1"], cfg["bm25"]["b"])
    scores["BM25"], timings["bm25_score_s"] = timed(bm25.score_batch, q_tokens)
    index_stats = bm25.index.stats()

    # Check: matrix BM25 equals walking the postings lists.
    max_diff = 0.0
    for qi in range(0, len(q_tokens), 360):
        for d, v in bm25.score_loop(q_tokens[qi]).items():
            max_diff = max(max_diff, abs(scores["BM25"][qi, d] - v))

    # Dense models
    dense = DenseModel(BASE_DENSE)
    scores["Sentence Transformer"], timings["dense_encode_s"] = timed(
        dense.score, queries, test.docs, "test")
    if FINETUNED.exists():
        ft = DenseModel(str(FINETUNED))
        scores["Dense, fine-tuned"], timings["finetuned_encode_s"] = timed(
            ft.score, queries, test.docs, "test")

    for name, s in scores.items():
        ranks[name] = ranks_from_scores(s, rel)
        lists[name] = top_k(s, DEPTH)

    # Fusion runs
    k_best = cfg["rrf_best_k"]
    three = [lists["TF-IDF"], lists["BM25"], lists["Sentence Transformer"]]
    lists["Hybrid RRF"] = rrf(three, k=60)
    lists["Hybrid RRF best k"] = rrf(three, k=k_best)
    lists["Hybrid RRF BM25+Dense"] = rrf([lists["BM25"], lists["Sentence Transformer"]], k=60)
    if "Dense, fine-tuned" in lists:
        lists["Hybrid RRF (BM25 + fine-tuned)"] = rrf([lists["BM25"], lists["Dense, fine-tuned"]], k=60)
    ce_cache = ROOT / "cache" / "cross_encoder_test_top20.npy"
    if ce_cache.exists():
        lists["Hybrid + cross-encoder"] = np.load(ce_cache)
        timings["cross_encoder_s"] = json.loads((RES / "timings.json").read_text()).get("cross_encoder_s")
    else:
        lists["Hybrid + cross-encoder"], timings["cross_encoder_s"] = timed(
            cross_encoder_rerank, queries, test.docs, lists["Hybrid RRF"], 20)
        np.save(ce_cache, lists["Hybrid + cross-encoder"])
    for name in ["Hybrid RRF", "Hybrid RRF best k", "Hybrid RRF BM25+Dense",
                 "Hybrid RRF (BM25 + fine-tuned)", "Hybrid + cross-encoder"]:
        if name in lists:
            ranks[name] = ranks_from_lists(lists[name], rel)

    # Full-pool table with Wilcoxon signed-rank test against BM25 on per-query AP
    order = [n for n in RUN_NAMES if n in ranks]
    full = {}
    for name in order:
        row = summary(ranks[name])
        if name != "BM25":
            row["wilcoxon_p_vs_bm25"] = float(wilcoxon(average_precision(ranks[name]),
                                                       average_precision(ranks["BM25"])).pvalue)
        full[RUN_NAMES[name].format(k=k_best)] = row

    # 5-choice setting: score each query only against its own five options
    opt = {n: scores[n][np.arange(len(rel))[:, None], test.options] for n in scores}
    five = {}
    for name in opt:
        five[name] = five_choice_summary(option_rank(opt[name], test.labels))
    five["Hybrid RRF"] = five_choice_summary(option_rank(
        option_rrf([opt["TF-IDF"], opt["BM25"], opt["Sentence Transformer"]], 60), test.labels))
    five["Hybrid RRF BM25+Dense"] = five_choice_summary(option_rank(
        option_rrf([opt["BM25"], opt["Sentence Transformer"]], 60), test.labels))
    if "Dense, fine-tuned" in opt:
        five["Hybrid RRF (BM25 + fine-tuned)"] = five_choice_summary(option_rank(
            option_rrf([opt["BM25"], opt["Dense, fine-tuned"]], 60), test.labels))
    ce5_cache = ROOT / "cache" / "cross_encoder_test_5choice.npy"
    if ce5_cache.exists():
        ce5 = np.load(ce5_cache)
    else:
        from sentence_transformers import CrossEncoder
        from casehold_ir.models import device
        ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2", device=device(), max_length=512)
        pairs = [(queries[i], test.docs[d]) for i in range(len(rel)) for d in test.options[i]]
        ce5 = ce.predict(pairs, batch_size=64, show_progress_bar=True).reshape(-1, 5)
        np.save(ce5_cache, ce5)
    five["Cross-encoder (ms-marco)"] = five_choice_summary(option_rank(ce5, test.labels))
    five_named = {RUN_NAMES.get(k, k).format(k=k_best): v for k, v in five.items()}

    # Figures
    main_runs = [n for n in ["TF-IDF", "BM25", "Sentence Transformer", "Hybrid RRF",
                             "Hybrid + cross-encoder", "Dense, fine-tuned",
                             "Hybrid RRF (BM25 + fine-tuned)"] if n in ranks]
    fig_metrics(main_runs, ranks)
    fig_pr(ranks)
    fig_rank_buckets(main_runs, ranks)

    errors = error_analysis(test, ranks, lists)
    per_query = pd.DataFrame({RUN_NAMES[n].format(k=k_best): ranks[n] for n in order})
    per_query.index.name = "query_id"
    per_query.to_csv(RES / "per_query_ranks.csv")

    timings["n_queries"], timings["n_docs"] = len(rel), len(test.docs)
    (RES / "timings.json").write_text(json.dumps(timings, indent=2))
    out = {"config": cfg | {"rrf_main_k": 60}, "index": index_stats,
           "bm25_matrix_vs_loop_max_abs_diff": max_diff,
           "full_pool": full, "five_choice": five_named, "error_analysis": errors,
           "timings": timings}
    (RES / "test_results.json").write_text(json.dumps(out, indent=2))
    write_markdown(full, five_named)
    print(json.dumps({"full_pool": full, "five_choice": five_named}, indent=1))


def fig_metrics(runs, ranks):
    metrics = [("MAP", "MAP (= MRR)"), ("nDCG@10", "nDCG@10"), ("R@100", "Recall@100")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    y = np.arange(len(runs))[::-1]
    for ax, (key, title) in zip(axes, metrics):
        vals = [summary(ranks[r])[key] for r in runs]
        ax.barh(y, vals, height=0.62, color=[plotting.COLORS[r] for r in runs])
        for yi, v in zip(y, vals):
            ax.text(v + 0.01, yi, f"{v:.3f}", va="center", fontsize=9.5, color=plotting.TEXT)
        ax.set_title(title, loc="left")
        ax.set_xlim(0, max(vals) * 1.22)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y, runs)
    fig.tight_layout()
    fig.savefig(FIG / "fig3_full_pool_results.png")
    plt.close(fig)


def fig_pr(ranks):
    fig, ax = plt.subplots(figsize=(7, 4.8))
    for name in REQUIRED:
        p, r = pr_curve(ranks[name])
        ax.plot(r, p, color=plotting.COLORS[name], label=name)
        ax.plot(r[[0, 9, 99]], p[[0, 9, 99]], "o", color=plotting.COLORS[name], markersize=6)
    ax.set_yscale("log")
    ax.set_xlabel("Mean Recall@k")
    ax.set_ylabel("Mean Precision@k (log scale)")
    ax.set_title("Precision-Recall, k = 1 to 100 (dots at k = 1, 10, 100)", loc="left", fontsize=12)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG / "fig4_precision_recall.png")
    plt.close(fig)


def fig_rank_buckets(runs, ranks):
    bins = [("Rank 1", lambda r: r == 1), ("Rank 2-10", lambda r: (r > 1) & (r <= 10)),
            ("Rank 11-100", lambda r: (r > 10) & (r <= 100)), ("Below 100", lambda r: r > 100)]
    shades = ["#1c5bab", "#4f8fe0", "#9fc2ee", "#dcdcd7"]   # one hue, dark = better
    fig, ax = plt.subplots(figsize=(11, 4.2))
    y = np.arange(len(runs))[::-1]
    left = np.zeros(len(runs))
    for (label, fn), color in zip(bins, shades):
        share = np.array([fn(ranks[r]).mean() for r in runs])
        ax.barh(y, share, left=left, height=0.62, color=color, label=label,
                edgecolor=plotting.SURFACE, linewidth=2)
        for yi, l, s in zip(y, left, share):
            if s > 0.06:
                ax.text(l + s / 2, yi, f"{s:.0%}", ha="center", va="center", fontsize=9,
                        color="white" if color in shades[:2] else plotting.TEXT)
        left += share
    ax.set_yticks(y, runs)
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.grid(axis="y", visible=False)
    ax.set_title("Where the correct holding ends up", loc="left")
    ax.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.4, -0.25))
    fig.tight_layout()
    fig.savefig(FIG / "fig5_rank_buckets.png")
    plt.close(fig)


def error_analysis(test, ranks, lists):
    bm, de, hy = ranks["BM25"], ranks["Sentence Transformer"], ranks["Hybrid RRF"]
    out = {
        "bm25_top1_dense_not": int(((bm == 1) & (de != 1)).sum()),
        "dense_top1_bm25_not": int(((de == 1) & (bm != 1)).sum()),
        "both_top1": int(((bm == 1) & (de == 1)).sum()),
        "neither_top10": int(((bm > 10) & (de > 10)).sum()),
        "median_rank": {n: float(np.median(ranks[n])) for n in ranks},
    }
    miss = np.where(hy != 1)[0]
    top1 = lists["Hybrid RRF"][miss, 0]
    distractor = [top1[i] in set(test.options[q].tolist()) - {test.relevant[q]}
                  for i, q in enumerate(miss)]
    jac = []
    for i, q in enumerate(miss):
        a, b = set(tokenize(test.docs[top1[i]])), set(tokenize(test.docs[test.relevant[q]]))
        jac.append(len(a & b) / max(1, len(a | b)))
    out["hybrid_misses"] = int(len(miss))
    out["hybrid_miss_top1_is_own_distractor"] = float(np.mean(distractor))
    out["hybrid_miss_top1_jaccard_with_correct"] = float(np.mean(jac))
    lengths = np.array([len(test.docs[d].split()) for d in test.relevant])
    buckets = {"<=15": lengths <= 15, "16-25": (lengths > 15) & (lengths <= 25),
               "26-40": (lengths > 25) & (lengths <= 40), ">40": lengths > 40}
    out["bm25_mrr_by_holding_length"] = {k: {"n": int(m.sum()), "mrr": float(np.mean(1 / bm[m]))}
                                         for k, m in buckets.items()}
    # Candidate examples for the report: lexical win, semantic win, failure for every model
    def examples(mask, key):
        idx = np.where(mask)[0]
        idx = idx[np.argsort(key[idx])][:5]
        return [{"query_id": int(q), "bm25_rank": int(bm[q]), "dense_rank": int(de[q]),
                 "hybrid_rank": int(hy[q]), "context_tail": " ".join(test.queries[q].split()[-60:]),
                 "correct": test.docs[test.relevant[q]],
                 "bm25_top1": test.docs[lists["BM25"][q, 0]],
                 "dense_top1": test.docs[lists["Sentence Transformer"][q, 0]]} for q in idx]
    out["examples_bm25_wins"] = examples((bm == 1) & (de > 100), -de)
    out["examples_dense_wins"] = examples((de == 1) & (bm > 1000), -bm)
    all_bad = np.minimum.reduce([ranks[n] for n in ranks])
    out["examples_all_fail"] = examples(all_bad > 500, -all_bad)
    return out


def write_markdown(full, five):
    lines = ["## Full-pool retrieval (test split)", "",
             "| Ranking model | P@10 | Recall@10 | Recall@100 | MAP (= MRR) | nDCG@10 | Wilcoxon p vs BM25 |",
             "|---|---|---|---|---|---|---|"]
    for name, r in full.items():
        p = r.get("wilcoxon_p_vs_bm25")
        lines.append(f"| {name} | {r['P@10']:.3f} | {r['R@10']:.3f} | {r['R@100']:.3f} | "
                     f"{r['MAP']:.3f} | {r['nDCG@10']:.3f} | {'-' if p is None else f'{p:.2g}'} |")
    lines += ["", "## 5-choice setting (original CaseHOLD format)", "",
              "| Ranking model | Accuracy | MRR | Recall@2 | nDCG@5 |", "|---|---|---|---|---|"]
    for name, r in five.items():
        lines.append(f"| {name} | {r['Accuracy']:.3f} | {r['MRR']:.3f} | {r['R@2']:.3f} | {r['nDCG@5']:.3f} |")
    (RES / "tables.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
