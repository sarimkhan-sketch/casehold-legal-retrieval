"""Shared figure style so every chart in the report and slides looks the same."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"

# Fixed colour per model, so a model keeps its colour in every figure.
COLORS = {
    "TF-IDF": "#2a78d6",
    "BM25": "#eb6834",
    "Sentence Transformer": "#1baf7a",
    "Hybrid RRF": "#eda100",
    "Hybrid + cross-encoder": "#e34948",
    "Dense, fine-tuned": "#6250d6",
    "Hybrid RRF (BM25 + fine-tuned)": "#e87ba4",
}


def setup():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": TEXT_2, "text.color": TEXT,
        "xtick.color": TEXT_2, "ytick.color": TEXT_2, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold",
        "legend.frameon": False, "lines.linewidth": 2, "savefig.dpi": 200,
        "savefig.bbox": "tight",
    })
