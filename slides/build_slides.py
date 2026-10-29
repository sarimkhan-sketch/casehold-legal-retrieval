"""Build the presentation from results/*.json and results/figures/*.png.

Every number on the slides is read from the result files, so re-running the
experiments and this script keeps the deck in sync with the code.
"""
import json
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
FIG = RES / "figures"
OUT = ROOT / "slides" / "CaseHOLD_Legal_Holding_Retrieval.pptx"

NAVY = RGBColor(0x14, 0x2B, 0x4C)
BLUE = RGBColor(0x2A, 0x78, 0xD6)
INK = RGBColor(0x0B, 0x0B, 0x0B)
INK_2 = RGBColor(0x52, 0x51, 0x4E)
LIGHT = RGBColor(0xF2, 0xF5, 0xFA)
GREEN = RGBColor(0x0F, 0x7B, 0x4F)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
RULE = RGBColor(0xE4, 0xE3, 0xDF)

R = json.loads((RES / "test_results.json").read_text())
V = json.loads((RES / "validation.json").read_text())
D = json.loads((RES / "dataset_stats.json").read_text())
FULL, FIVE, ERR, T = R["full_pool"], R["five_choice"], R["error_analysis"], R["timings"]

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
BLANK = prs.slide_layouts[6]
W, H = prs.slide_width, prs.slide_height


def f3(x):
    return f"{x:.3f}"


def text(slide, x, y, w, h, content, size=18, color=INK, bold=False, align=PP_ALIGN.LEFT,
         anchor=MSO_ANCHOR.TOP, font="Calibri"):
    """Add a text box. `content` is a string or a list of (text, options) paragraphs."""
    box = slide.shapes.add_textbox(x, y, w, h)
    tf = box.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(0.05)
    paras = content if isinstance(content, list) else [(content, {})]
    for i, (line, opt) in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = opt.get("align", align)
        p.space_after = Pt(opt.get("space", 6))
        run = p.add_run()
        run.text = line
        run.font.name = font
        run.font.size = Pt(opt.get("size", size))
        run.font.bold = opt.get("bold", bold)
        run.font.italic = opt.get("italic", False)
        run.font.color.rgb = opt.get("color", color)
    return box


def bullets(slide, x, y, w, h, items, size=18):
    paras = []
    for item in items:
        if isinstance(item, tuple):          # (heading, detail)
            paras.append(("▸ " + item[0], {"bold": True, "size": size, "space": 2}))
            paras.append(("   " + item[1], {"size": size - 3, "color": INK_2, "space": 10}))
        else:
            paras.append(("▸ " + item, {"size": size, "space": 10}))
    return text(slide, x, y, w, h, paras)


def rect(slide, x, y, w, h, fill, line=None, shape=MSO_SHAPE.RECTANGLE):
    s = slide.shapes.add_shape(shape, x, y, w, h)
    s.fill.solid()
    s.fill.fore_color.rgb = fill
    if line is None:
        s.line.fill.background()
    else:
        s.line.color.rgb = line
    s.shadow.inherit = False
    return s


def new_slide(title, kicker=None):
    s = prs.slides.add_slide(BLANK)
    rect(s, 0, 0, W, Inches(1.05), NAVY)
    text(s, Inches(0.5), Inches(0.12), Inches(11.5), Inches(0.85), title, size=30, color=WHITE,
         bold=True, anchor=MSO_ANCHOR.MIDDLE)
    if kicker:
        text(s, Inches(0.5), Inches(1.15), Inches(12.3), Inches(0.6), kicker, size=17,
             color=BLUE, bold=True)
    n = len(prs.slides)
    text(s, Inches(11.8), Inches(7.0), Inches(1.3), Inches(0.4), str(n), size=11, color=INK_2,
         align=PP_ALIGN.RIGHT)
    text(s, Inches(0.5), Inches(7.0), Inches(8), Inches(0.4),
         "Legal Holding Retrieval on CaseHOLD  ·  Sarim Khan  ·  CSE4144", size=11, color=INK_2)
    return s


def picture(slide, path, x, y, w=None, h=None):
    return slide.shapes.add_picture(str(path), x, y, width=w, height=h)


def table(slide, x, y, w, rows, col_widths, size=13, highlight=(), row_h=0.38):
    shape = slide.shapes.add_table(len(rows), len(rows[0]), x, y, w, Inches(row_h * len(rows)))
    tbl = shape.table
    for j, cw in enumerate(col_widths):
        tbl.columns[j].width = Emu(int(w * cw / sum(col_widths)))
    for i, row in enumerate(rows):
        tbl.rows[i].height = Inches(row_h)
        for j, val in enumerate(row):
            cell = tbl.cell(i, j)
            cell.text = str(val)
            cell.margin_left = cell.margin_right = Inches(0.08)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            p = cell.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT if j == 0 else PP_ALIGN.CENTER
            run = p.runs[0]
            run.font.size = Pt(size)
            run.font.name = "Calibri"
            cell.fill.solid()
            if i == 0:
                cell.fill.fore_color.rgb = NAVY
                run.font.color.rgb = WHITE
                run.font.bold = True
            elif i in highlight:
                cell.fill.fore_color.rgb = RGBColor(0xDD, 0xEA, 0xFA)
                run.font.bold = True
                run.font.color.rgb = INK
            else:
                cell.fill.fore_color.rgb = WHITE if i % 2 else LIGHT
                run.font.color.rgb = INK
    return tbl


def callout(slide, x, y, w, h, big, small, color=BLUE):
    rect(slide, x, y, w, h, LIGHT, shape=MSO_SHAPE.ROUNDED_RECTANGLE).adjustments[0] = 0.08
    rect(slide, x, y, Inches(0.09), h, color)
    text(slide, x + Inches(0.25), y + Inches(0.1), w - Inches(0.35), Inches(0.75), big, size=30,
         color=color, bold=True)
    text(slide, x + Inches(0.25), y + Inches(0.8), w - Inches(0.35), h - Inches(0.85), small,
         size=14, color=INK_2)


def notes(slide, txt):
    slide.notes_slide.notes_text_frame.text = txt


# ---------------------------------------------------------------- 1. Title
s = prs.slides.add_slide(BLANK)
rect(s, 0, 0, W, H, NAVY)
rect(s, Inches(0.8), Inches(2.0), Inches(0.12), Inches(2.3), BLUE)
text(s, Inches(1.15), Inches(1.9), Inches(11), Inches(1.4), "Legal Holding Retrieval on CaseHOLD",
     size=44, color=WHITE, bold=True)
text(s, Inches(1.15), Inches(3.05), Inches(11), Inches(1.3),
     "Comparing TF-IDF, BM25, a Sentence Transformer and Hybrid Reciprocal Rank Fusion",
     size=22, color=RGBColor(0xBF, 0xD3, 0xF0))
text(s, Inches(1.15), Inches(4.8), Inches(11), Inches(1.8), [
    ("Sarim Khan  (23F10CSE00486)", {"size": 20, "bold": True, "color": WHITE}),
    ("CSE4144 Information Retrieval  ·  B.Tech CSE, Semester VII, 2026-27",
     {"size": 16, "color": RGBColor(0xBF, 0xD3, 0xF0)}),
    ("Guide: Ms Shweta Sharma  ·  Manipal University Jaipur",
     {"size": 16, "color": RGBColor(0xBF, 0xD3, 0xF0)}),
])
notes(s, "Introduce the project: retrieving the legal holding a court cites, using classical "
         "and neural IR models on the CaseHOLD dataset.")

# ---------------------------------------------------------------- 2. Problem
s = new_slide("The problem: finding the precedent that supports a point",
              "A lawyer knows the argument. They need the cited case whose holding backs it.")
bullets(s, Inches(0.5), Inches(1.9), Inches(6.6), Inches(4.8), [
    ("Legal arguments rest on precedent",
     "Courts write “other courts have held…” and cite a case, followed by a parenthetical stating its holding."),
    ("Manual search does not scale",
     "Thousands of decisions; a natural information retrieval task."),
    ("Lexical vs semantic matching",
     "TF-IDF/BM25 match exact words. Dense models match meaning but may not know legal language."),
    ("Goal of this project",
     "Recast CaseHOLD as retrieval and compare 4 required rankers + 2 extensions with standard IR metrics."),
], size=19)
rect(s, Inches(7.5), Inches(2.0), Inches(5.3), Inches(4.4), LIGHT, shape=MSO_SHAPE.ROUNDED_RECTANGLE).adjustments[0] = 0.05
text(s, Inches(7.8), Inches(2.15), Inches(4.8), Inches(4.2), [
    ("Vocabulary mismatch", {"bold": True, "size": 18, "color": BLUE}),
    ("Citing passage talks about:", {"size": 15, "color": INK_2}),
    ("“fears of a gas pipeline”", {"size": 20, "italic": True}),
    ("Correct holding talks about:", {"size": 15, "color": INK_2}),
    ("“stigma damages … severance damages”", {"size": 20, "italic": True}),
    ("Few shared words, same legal point. This is why we compare lexical, dense and hybrid rankers.",
     {"size": 15, "color": INK_2}),
])
notes(s, "Explain the motivation and the vocabulary mismatch problem.")

# ---------------------------------------------------------------- 3. Dataset
s = new_slide("Dataset: CaseHOLD (LexGLUE version)",
              "53,000+ citing passages from US court decisions, each with 1 correct and 4 wrong holdings")
rows = [["Split", "Queries", "Unique holdings", "Words / query", "Words / holding"]]
for name in ["train", "validation", "test"]:
    d = D[name]
    rows.append([name.capitalize(), f"{d['queries']:,}", f"{d['docs']:,}", d["avg_words_query"],
                 d["avg_words_doc"]])
table(s, Inches(0.5), Inches(1.95), Inches(6.3), rows, [1.3, 1, 1.4, 1.3, 1.4], size=14, highlight=(3,))
ex = D["test_extra"]
bullets(s, Inches(0.5), Inches(3.75), Inches(6.3), Inches(3.2), [
    "Built from the Harvard Law case law corpus (Zheng et al., ICAIL 2021)",
    "Wrong options = most TF-IDF-similar holdings → deliberately hard",
    f"Correct answer spread evenly over 5 positions ({min(ex['label_counts'])}–{max(ex['label_counts'])} per position)",
    "Long queries (~136 words), short holdings (~25 words)",
], size=16)
picture(s, FIG / "fig1_lengths.png", Inches(7.0), Inches(2.1), w=Inches(6.0))
text(s, Inches(7.0), Inches(4.4), Inches(6), Inches(0.5),
     "Length distributions in the test split", size=13, color=INK_2, align=PP_ALIGN.CENTER)
notes(s, "Hugging Face: coastalcph/lex_glue, config case_hold.")

# ---------------------------------------------------------------- 4. Example
s = new_slide("An example: test query 0", "Choices 1-3 are on the same topic; that is what makes it hard")
q0 = ex["example_query_0"]
ctx = q0["context"]
start = ctx.find("Other courts")
text(s, Inches(0.5), Inches(1.85), Inches(12.3), Inches(1.6), [
    ("Citing context (query)", {"bold": True, "size": 15, "color": BLUE}),
    ("… " + ctx[start:], {"size": 15, "italic": True}),
])
y = Inches(3.75)
for j, opt in enumerate(q0["options"]):
    correct = j == q0["label"]
    rect(s, Inches(0.5), y, Inches(12.3), Inches(0.55), RGBColor(0xDF, 0xF3, 0xE9) if correct else LIGHT)
    text(s, Inches(0.65), y, Inches(12.0), Inches(0.55),
         f"{j}{'  ✓ correct' if correct else ''}:  {opt}", size=15,
         color=GREEN if correct else INK, bold=correct, anchor=MSO_ANCHOR.MIDDLE)
    y += Inches(0.62)
notes(s, "Walk through the example: the passage, the <HOLDING> marker, and the five options.")

# ---------------------------------------------------------------- 5. Recast
s = new_slide("Turning CaseHOLD into a retrieval task")
callout(s, Inches(0.5), Inches(1.5), Inches(6.0), Inches(2.4), "Full-pool setting",
        f"Query = citing passage ({T['n_queries']:,} queries). Documents = every unique holding in "
        f"the test split ({T['n_docs']:,}). Exactly one relevant document per query. "
        "The realistic search setting.")
callout(s, Inches(6.85), Inches(1.5), Inches(6.0), Inches(2.4), "5-choice setting",
        "Each query is ranked only against its own five options, matching the original "
        "multiple-choice CaseHOLD task. Accuracy = correct holding ranked first (random = 0.20).",
        color=RGBColor(0xEB, 0x68, 0x34))
text(s, Inches(0.5), Inches(4.2), Inches(12.3), Inches(0.5), "Evaluation metrics", size=20, bold=True, color=BLUE)
bullets(s, Inches(0.5), Inches(4.75), Inches(12.3), Inches(2.2), [
    "Precision@10, Recall@10/100, MAP and nDCG@10, plus Precision-Recall graphs",
    "With one relevant document at rank r: AP = 1/r, so MAP = MRR, and Precision@10 is capped at 0.1",
    "Wilcoxon signed-rank test on per-query AP to check whether gains over BM25 are significant",
], size=17)

# ---------------------------------------------------------------- 6. Pipeline
s = new_slide("System pipeline", "A sparse path and a dense path, merged by Reciprocal Rank Fusion")
picture(s, FIG / "fig2_pipeline.png", Inches(0.6), Inches(1.9), w=Inches(12.1))
notes(s, "Sparse path: preprocessing, inverted index, TF-IDF and BM25. Dense path: raw text to "
         "MiniLM vectors and cosine similarity. RRF merges the ranked lists.")

# ---------------------------------------------------------------- 7. Preprocessing & index
idx = R["index"]
s = new_slide("Preprocessing and the inverted index")
bullets(s, Inches(0.5), Inches(1.4), Inches(7.0), Inches(4.8), [
    ("Normalise", "Lower-case, remove the <HOLDING> marker, join dotted abbreviations (e.g. → eg)"),
    ("Tokenise", "Split on non-alphanumerics; drop 1-letter tokens and any token with a digit "
                 "(removes reporter citations like 157 F.Supp.2d 949)"),
    ("Stop words", "scikit-learn English list + legal words: holding, held, v, see, eg, id, cf, supra, citing"),
    ("Stem", "Porter stemmer"),
    ("Dense path", "Uses raw text: stemming would destroy the sentences the model was trained on"),
], size=18)
rect(s, Inches(7.9), Inches(1.5), Inches(4.9), Inches(1.9), LIGHT, shape=MSO_SHAPE.ROUNDED_RECTANGLE).adjustments[0] = 0.06
text(s, Inches(8.1), Inches(1.6), Inches(4.6), Inches(1.8), [
    ("Correct holding of query 0 after preprocessing", {"bold": True, "size": 14, "color": BLUE}),
    (", ".join(ex["example_query_0"]["correct_tokens"]), {"size": 15, "italic": True}),
])
rows = [["Inverted index (test pool)", "Value"],
        ["Distinct terms", f"{idx['terms']:,}"],
        ["Postings", f"{idx['postings']:,}"],
        ["Avg. indexed terms / holding", f"{idx['avg_terms_per_doc']:.2f}"],
        ["Avg. postings list length", f"{idx['avg_postings_per_term']:.2f}"],
        ["Index build time (BM25)", f"{T['bm25_index_s']:.2f} s"]]
table(s, Inches(7.9), Inches(3.7), Inches(4.9), rows, [2.6, 1.2], size=14)

# ---------------------------------------------------------------- 8. Models
s = new_slide("Ranking models")
models = [
    ("TF-IDF (vector space)", "Weight (1 + log tf) × idf, unit-length vectors, cosine similarity", BLUE),
    ("Okapi BM25 (own implementation)", "Term-frequency saturation (k1) and length normalisation (b) on our inverted index", RGBColor(0xEB, 0x68, 0x34)),
    ("Sentence Transformer", "all-MiniLM-L6-v2: 6 layers, 384-d vectors, cosine similarity", RGBColor(0x1B, 0xAF, 0x7A)),
    ("Hybrid RRF", "Fuses the top-1000 lists of TF-IDF, BM25 and dense using only rank positions", RGBColor(0xED, 0xA1, 0x00)),
    ("Ext. 1: fine-tuned dense", "MiniLM trained 1 epoch on 45,000 CaseHOLD training pairs (MNR loss, hard negative)", RGBColor(0x62, 0x50, 0xD6)),
    ("Ext. 2: cross-encoder", "ms-marco-MiniLM-L6-v2 re-scores the top 20 of the hybrid list", RGBColor(0xE3, 0x49, 0x48)),
]
y = Inches(1.35)
for name, desc, color in models:
    rect(s, Inches(0.5), y, Inches(0.12), Inches(0.72), color)
    text(s, Inches(0.75), y - Inches(0.03), Inches(6.6), Inches(0.8), [
        (name, {"bold": True, "size": 17, "space": 0}), (desc, {"size": 13, "color": INK_2})])
    y += Inches(0.9)
rect(s, Inches(7.6), Inches(1.4), Inches(5.25), Inches(5.3), LIGHT, shape=MSO_SHAPE.ROUNDED_RECTANGLE).adjustments[0] = 0.04
text(s, Inches(7.85), Inches(1.5), Inches(4.9), Inches(5.2), [
    ("BM25", {"bold": True, "size": 17, "color": BLUE}),
    ("score(q,d) = Σ idf(t) · tf·(k1+1) / (tf + k1·(1 − b + b·|d|/avgdl))", {"size": 14, "font": "Cambria"}),
    ("idf(t) = ln(1 + (N − df + 0.5) / (df + 0.5))", {"size": 14}),
    (f"Chosen on validation: k1 = {V['bm25']['k1']}, b = {V['bm25']['b']}", {"size": 14, "color": INK_2, "space": 18}),
    ("Reciprocal Rank Fusion", {"bold": True, "size": 17, "color": BLUE}),
    ("RRF(d) = Σ over models m of 1 / (k + rank_m(d))", {"size": 14}),
    ("No score normalisation needed: BM25 scores are unbounded, cosine lies in [−1, 1].",
     {"size": 14, "color": INK_2}),
    (f"k = 60 (standard) for main results; k = {V['rrf_best_k']} chosen on validation as an extra run.",
     {"size": 14, "color": INK_2}),
])

# ---------------------------------------------------------------- 9. Validation
s = new_slide("Choosing parameters on the validation split",
              f"All choices made on {D['validation']['queries']:,} validation queries (MRR), then frozen")
ql, tv, dm = V["query_length"], V["tfidf_variants"], V["dense_models"]
rows = [["Decision", "Options tried (validation MRR)", "Chosen"],
        ["Query length (BM25)", f"full {ql['full']:.3f} · last 64 words {ql['last_64']:.3f} · last 32 {ql['last_32']:.3f}", "Full context"],
        ["BM25 k1, b", f"grid of {len(V['bm25_grid'])} settings, best MRR {V['bm25']['mrr']:.3f}", f"k1 = {V['bm25']['k1']}, b = {V['bm25']['b']}"],
        ["TF-IDF preprocessing", f"tokens {tv['tokens_only']:.3f} · + stop words {tv['stop_words']:.3f} · + stemming {tv['stop_words+stem']:.3f}", "Stop words + stem"],
        ["Dense model", f"MiniLM {dm['all-MiniLM-L6-v2']:.3f} · bge-small {dm['bge-small-en-v1.5']:.3f} · bge + instruction {dm['bge-small-en-v1.5 + instruction']:.3f}", "all-MiniLM-L6-v2"],
        ["RRF constant k", " · ".join(f"k={k}: {v:.3f}" for k, v in V["rrf_k"].items() if k in {"1", "5", "10", "60", "100"}), f"k = {V['rrf_best_k']} (extra run)"]]
table(s, Inches(0.5), Inches(1.95), Inches(12.3), rows, [2.2, 6.6, 2.2], size=14, row_h=0.62)
text(s, Inches(0.5), Inches(6.0), Inches(12.3), Inches(0.8),
     "Low k1 and high b suit short documents, where repeating a term adds little meaning. "
     "Stop-word removal alone did not help TF-IDF; adding stemming did.", size=15, color=INK_2)

# ---------------------------------------------------------------- 10. Results table
s = new_slide("Full-pool results on the test split",
              f"{T['n_queries']:,} queries × {T['n_docs']:,} holdings. Max possible Precision@10 is 0.1.")
rows = [["Ranking model", "P@10", "Recall@100", "MAP (= MRR)", "nDCG@10"]]
names = list(FULL)
for n in names:
    r = FULL[n]
    rows.append([n, f3(r["P@10"]), f3(r["R@100"]), f3(r["MAP"]), f3(r["nDCG@10"])])
best_i = 1 + max(range(len(names)), key=lambda i: FULL[names[i]]["MAP"])
table(s, Inches(0.5), Inches(1.85), Inches(12.3), rows, [5, 1.3, 1.5, 1.6, 1.4], size=15,
      highlight=(best_i,), row_h=0.47)
text(s, Inches(0.5), Inches(6.6), Inches(12.3), Inches(0.4),
     "Rows 1-4 are the four required models; “ext.” rows are the extensions. Highlighted = best run.",
     size=13, color=INK_2)

# ---------------------------------------------------------------- 11. Results figure + takeaways
bm, hy, de = FULL["BM25"], FULL["Hybrid RRF (TF-IDF + BM25 + Dense)"], FULL["Sentence Transformer"]
ft = FULL.get("Dense, fine-tuned (ext.)")
fth = FULL.get("Hybrid RRF (BM25 + fine-tuned Dense)")
ce = FULL["Hybrid + cross-encoder"]
s = new_slide("What the results show")
picture(s, FIG / "fig3_full_pool_results.png", Inches(0.4), Inches(1.3), w=Inches(12.5))
y0 = Inches(5.45)
cw = Inches(3.0)
callout(s, Inches(0.5), y0, cw, Inches(1.45), f"{f3(bm['MAP'])}",
        f"BM25 MAP: best required single model (dense {f3(de['MAP'])})", color=RGBColor(0xEB, 0x68, 0x34))
callout(s, Inches(3.65), y0, cw, Inches(1.45), f"{f3(bm['R@100'])} → {f3(hy['R@100'])}",
        "Recall@100, BM25 → three-way RRF", color=RGBColor(0xC9, 0x85, 0x00))
if ft:
    callout(s, Inches(6.8), y0, cw, Inches(1.45), f"{f3(de['MAP'])} → {f3(ft['MAP'])}",
            "Dense MAP after fine-tuning on CaseHOLD", color=RGBColor(0x62, 0x50, 0xD6))
callout(s, Inches(9.95), y0, cw, Inches(1.45), f"{f3(ce['MAP'])}",
        "Cross-encoder MAP: web-trained re-ranker hurts", color=RGBColor(0xE3, 0x49, 0x48))
notes(s, f"Best run overall: BM25 + fine-tuned dense, MAP {f3(fth['MAP']) if fth else 'n/a'}, "
         f"Recall@100 {f3(fth['R@100']) if fth else 'n/a'}. Wilcoxon p for three-way hybrid vs BM25: "
         f"{hy['wilcoxon_p_vs_bm25']:.2g}.")

# ---------------------------------------------------------------- 12. PR graph + buckets
s = new_slide("Precision-Recall and where the correct holding lands")
picture(s, FIG / "fig4_precision_recall.png", Inches(0.3), Inches(1.3), w=Inches(5.6))
picture(s, FIG / "fig5_rank_buckets.png", Inches(6.0), Inches(1.6), w=Inches(7.1))
text(s, Inches(0.5), Inches(5.6), Inches(12.3), Inches(1.3), [
    ("One relevant holding per query, so precision falls roughly as 1/k; plotted on a log scale.", {"size": 15}),
    ("The hybrid curve rises above BM25 as k grows: fusion mainly pulls answers from deep in the list into the top 100.", {"size": 15}),
], color=INK_2)

# ---------------------------------------------------------------- 13. 5-choice
s = new_slide("5-choice setting (original CaseHOLD format)",
              "Every method is far above chance (0.20), but the ordering changes")
rows = [["Ranking model", "Accuracy", "MRR", "Recall@2", "nDCG@5"]]
fnames = list(FIVE)
for n in fnames:
    r = FIVE[n]
    rows.append([n, f3(r["Accuracy"]), f3(r["MRR"]), f3(r["R@2"]), f3(r["nDCG@5"])])
best5 = 1 + max(range(len(fnames)), key=lambda i: FIVE[fnames[i]]["Accuracy"])
table(s, Inches(0.5), Inches(1.85), Inches(7.6), rows, [4.2, 1.2, 1, 1.2, 1.1], size=14,
      highlight=(best5,), row_h=0.46)
bullets(s, Inches(8.4), Inches(1.9), Inches(4.5), Inches(4.8), [
    ("Wrong options were chosen by TF-IDF similarity", "All five candidates share vocabulary, so the lexical signal RRF adds helps less."),
    ("Domain training wins", "The fine-tuned dense model is the most accurate single ranker here."),
    ("Context", "Zheng et al. report ~0.6 macro-F1 for fine-tuned BERT on their split (not directly comparable)."),
], size=16)

# ---------------------------------------------------------------- 14. Error analysis
s = new_slide("Error analysis: the models fail on different queries")
y = Inches(1.4)
for big, small, color in [
        (f"{ERR['bm25_top1_dense_not']:,}", "queries where BM25 ranks the answer first but the dense model does not", RGBColor(0xEB, 0x68, 0x34)),
        (f"{ERR['dense_top1_bm25_not']:,}", "queries where only the dense model ranks it first", RGBColor(0x1B, 0xAF, 0x7A)),
        (f"{ERR['neither_top10']:,}", f"queries ({ERR['neither_top10'] / T['n_queries']:.0%}) where neither puts it in the top 10", INK_2)]:
    callout(s, Inches(0.5), y, Inches(4.6), Inches(1.5), big, small, color=color)
    y += Inches(1.7)
lens = ERR["bm25_mrr_by_holding_length"]
bullets(s, Inches(5.5), Inches(1.4), Inches(7.4), Inches(5.5), [
    ("Complementary errors → fusion helps",
     f"Both rank it first in {ERR['both_top1']:,} queries; the disagreements are what RRF exploits."),
    ("Hybrid's wrong top-1 is rarely a distractor",
     f"Only {ERR['hybrid_miss_top1_is_own_distractor']:.0%} of misses are one of the query's own 4 wrong options; "
     f"word overlap with the correct holding is low (Jaccard {ERR['hybrid_miss_top1_jaccard_with_correct']:.2f})."),
    ("Longer holdings are easier",
     f"BM25 MRR rises from {lens['<=15']['mrr']:.3f} (≤15 words) to {lens['>40']['mrr']:.3f} (>40 words): more terms to match."),
    ("Some citations cannot be recovered",
     "A citation used for a procedural point the surrounding text never mentions defeats every ranker that only reads the context."),
], size=17)
ex_d = next((e for e in ERR["examples_dense_wins"] if e["query_id"] == 296), ERR["examples_dense_wins"][0])
rect(s, Inches(5.6), Inches(4.75), Inches(7.25), Inches(1.95), LIGHT, shape=MSO_SHAPE.ROUNDED_RECTANGLE).adjustments[0] = 0.06
text(s, Inches(5.8), Inches(4.82), Inches(6.9), Inches(1.9), [
    (f"Query {ex_d['query_id']}: dense rank {ex_d['dense_rank']}, BM25 rank {ex_d['bm25_rank']:,}",
     {"bold": True, "size": 14, "color": RGBColor(0x1B, 0xAF, 0x7A)}),
    ("Context: two officers say they were ordered to “cease all contact” with each other …", {"size": 13, "italic": True}),
    ("Correct: " + ex_d["correct"], {"size": 13}),
    ("No shared keywords, so BM25 misses it; the dense model links the meaning.", {"size": 13, "color": INK_2}),
])

# ---------------------------------------------------------------- 15. Demo
s = new_slide("Live demo: Streamlit search interface",
              "Pick a CaseHOLD test query or type any passage; compare all four rankers side by side")
shot = FIG / "fig6_search_app.png"
if shot.exists():
    picture(s, shot, Inches(0.5), Inches(1.8), h=Inches(5.0))
    x_text = Inches(8.2)
else:
    x_text = Inches(0.5)
bullets(s, x_text, Inches(1.9), Inches(4.6), Inches(3.6), [
    "Shows preprocessed query terms",
    "Top 10 from TF-IDF, BM25, dense (base or fine-tuned) and RRF",
    "Marks the correct holding and its rank in each list",
    "BM25 answers by walking the postings lists: well under a second",
], size=15)
rect(s, x_text, Inches(5.6), Inches(4.6), Inches(0.9), NAVY, shape=MSO_SHAPE.ROUNDED_RECTANGLE)
text(s, x_text, Inches(5.6), Inches(4.6), Inches(0.9), "streamlit run app/search_app.py", size=15,
     color=WHITE, font="Menlo", align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
notes(s, "Switch to the browser. Show query 0, then type a custom passage. Toggle the fine-tuned model.")

# ---------------------------------------------------------------- 16. Conclusion
s = new_slide("Conclusion and future work")
text(s, Inches(0.5), Inches(1.35), Inches(6.2), Inches(0.5), "Key lessons", size=22, bold=True, color=BLUE)
bullets(s, Inches(0.5), Inches(1.95), Inches(6.2), Inches(5), [
    ("Lexical models are strong on legal text",
     f"Tuned BM25 (MAP {f3(bm['MAP'])}) beats an off-the-shelf dense model ({f3(de['MAP'])})."),
    ("Fusion helps recall more than the top rank",
     f"Three-way RRF lifts Recall@100 to {f3(hy['R@100'])}; MAP moves to {f3(hy['MAP'])}."),
    ("Domain data matters most",
     (f"Fine-tuning lifts dense MAP to {f3(ft['MAP'])}; BM25 + fine-tuned dense is best ({f3(fth['MAP'])})."
      if ft and fth else "Fine-tuning on CaseHOLD training pairs.")),
], size=17)
text(s, Inches(7.1), Inches(1.35), Inches(5.8), Inches(0.5), "Limits and next steps", size=22, bold=True, color=BLUE)
bullets(s, Inches(7.1), Inches(1.95), Inches(5.8), Inches(5), [
    "One relevant holding per query: legally valid alternatives count as wrong",
    "Pool is the test split, not a full case-law database",
    "Try a legal-domain encoder (Legal-BERT)",
    "Train the cross-encoder on CaseHOLD itself",
    "Weight the rankers inside RRF instead of treating them equally",
], size=17)

# ---------------------------------------------------------------- 17. Thanks
s = prs.slides.add_slide(BLANK)
rect(s, 0, 0, W, H, NAVY)
text(s, Inches(0.8), Inches(2.3), Inches(11.7), Inches(1.2), "Thank you  ·  Questions?", size=44,
     color=WHITE, bold=True, align=PP_ALIGN.CENTER)
text(s, Inches(0.8), Inches(3.7), Inches(11.7), Inches(2.5), [
    ("Code, results and demo: github.com/sarimkhan-sketch/casehold-legal-retrieval",
     {"size": 18, "color": RGBColor(0xBF, 0xD3, 0xF0), "align": PP_ALIGN.CENTER}),
    ("Key references: Robertson & Zaragoza 2009 (BM25) · Reimers & Gurevych 2019 (Sentence-BERT) · "
     "Cormack et al. 2009 (RRF) · Zheng et al. 2021 (CaseHOLD) · Chalkidis et al. 2022 (LexGLUE)",
     {"size": 14, "color": RGBColor(0xBF, 0xD3, 0xF0), "align": PP_ALIGN.CENTER}),
])

prs.save(OUT)
print(f"saved {OUT} ({len(prs.slides)} slides)")
