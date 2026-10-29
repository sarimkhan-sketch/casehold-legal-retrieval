"""Dataset statistics (Table 1), length plot (Figure 1) and pipeline diagram (Figure 2)."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

from casehold_ir import plotting
from casehold_ir.data import load_split
from casehold_ir.preprocess import tokenize

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "results" / "figures"


def split_stats():
    stats = {}
    for name in ["train", "validation", "test"]:
        s = load_split(name)
        stats[name] = {
            "queries": len(s.queries), "docs": len(s.docs),
            "avg_words_query": round(float(np.mean([len(q.split()) for q in s.queries])), 1),
            "avg_words_doc": round(float(np.mean([len(d.split()) for d in s.docs])), 1),
        }
    test = load_split("test")
    wrong_ids = {int(test.options[i, j]) for i in range(len(test.queries))
                 for j in range(5) if j != test.labels[i]}
    stats["test_extra"] = {
        "answer_options": int(test.options.size),
        "correct_also_wrong_elsewhere_queries": int(sum(int(r) in wrong_ids for r in test.relevant)),
        "correct_also_wrong_elsewhere_unique": len(set(test.relevant.tolist()) & wrong_ids),
        "label_counts": np.bincount(test.labels, minlength=5).tolist(),
        "share_docs_start_holding_that": round(float(np.mean(
            [d.lower().startswith("holding that") for d in test.docs])), 3),
        "example_query_0": {"context": test.queries[0],
                            "options": [test.docs[d] for d in test.options[0]],
                            "label": int(test.labels[0]),
                            "correct_tokens": tokenize(test.docs[test.relevant[0]])},
    }
    return stats, test


def length_figure(test):
    q_len = [len(q.split()) for q in test.queries]
    d_len = [len(d.split()) for d in test.docs]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, data, title, color in [
            (axes[0], q_len, f"Citing contexts (queries, n={len(q_len):,})", plotting.COLORS["TF-IDF"]),
            (axes[1], d_len, f"Unique holdings (documents, n={len(d_len):,})", plotting.COLORS["BM25"])]:
        ax.hist(data, bins=40, color=color, edgecolor=plotting.SURFACE, linewidth=1)
        ax.axvline(np.mean(data), color=plotting.TEXT, linestyle="--", linewidth=1.2)
        ax.text(np.mean(data), ax.get_ylim()[1] * 0.92, f"  mean {np.mean(data):.1f} words",
                color=plotting.TEXT, fontsize=10)
        ax.set_title(title, loc="left", fontsize=12)
        ax.set_xlabel("Length (words)")
        ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(FIG / "fig1_lengths.png")
    plt.close(fig)


def pipeline_figure():
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 4)
    ax.axis("off")

    def box(x, y, w, h, text, color):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                    facecolor=color, edgecolor="none", alpha=0.18))
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                    facecolor="none", edgecolor=color, linewidth=1.6))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10, color=plotting.TEXT)

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=plotting.TEXT_2, lw=1.4))

    gray = plotting.TEXT_2
    box(0.1, 1.5, 1.6, 1.0, "Citing context\n(query) +\n13,297 holdings", gray)
    box(2.3, 2.6, 2.0, 1.0, "Lower-case, remove\ncitations and\nstop words, stem", plotting.COLORS["TF-IDF"])
    box(4.8, 2.6, 1.6, 1.0, "Inverted index\n(postings lists)", plotting.COLORS["TF-IDF"])
    box(6.9, 3.1, 1.4, 0.7, "TF-IDF", plotting.COLORS["TF-IDF"])
    box(6.9, 2.2, 1.4, 0.7, "BM25", plotting.COLORS["BM25"])
    box(2.3, 0.4, 2.0, 1.0, "Raw text", plotting.COLORS["Sentence Transformer"])
    box(4.8, 0.4, 1.6, 1.0, "MiniLM encoder\n(384-d vectors)", plotting.COLORS["Sentence Transformer"])
    box(6.9, 0.55, 1.4, 0.7, "Cosine\nsimilarity", plotting.COLORS["Sentence Transformer"])
    box(8.8, 1.5, 1.0, 1.0, "RRF\nfusion", plotting.COLORS["Hybrid RRF"])
    box(10.0, 1.5, 0.9, 1.0, "Ranked\nholdings", gray)
    arrow(1.7, 2.2, 2.3, 3.0)
    arrow(1.7, 1.8, 2.3, 0.9)
    arrow(4.3, 3.1, 4.8, 3.1)
    arrow(4.3, 0.9, 4.8, 0.9)
    arrow(6.4, 3.2, 6.9, 3.45)
    arrow(6.4, 2.9, 6.9, 2.55)
    arrow(6.4, 0.9, 6.9, 0.9)
    for y in (3.45, 2.55, 0.9):
        arrow(8.3, y, 8.8, 2.0)
    arrow(9.8, 2.0, 10.0, 2.0)
    ax.text(9.9, 1.3, "optional cross-encoder\nre-ranks top 20", ha="center", va="top",
            fontsize=8.5, color=plotting.TEXT_2, style="italic")
    fig.savefig(FIG / "fig2_pipeline.png")
    plt.close(fig)


def main():
    plotting.setup()
    FIG.mkdir(parents=True, exist_ok=True)
    stats, test = split_stats()
    (ROOT / "results" / "dataset_stats.json").write_text(json.dumps(stats, indent=2))
    length_figure(test)
    pipeline_figure()
    for k, v in stats.items():
        if k != "test_extra":
            print(k, v)
    extra = {k: v for k, v in stats["test_extra"].items() if k != "example_query_0"}
    print(extra)


if __name__ == "__main__":
    main()
