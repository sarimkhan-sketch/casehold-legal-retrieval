"""Streamlit search interface: compare TF-IDF, BM25, dense and RRF side by side.

Run from the repository root:  streamlit run app/search_app.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
# Make casehold_ir importable without `pip install -e .` (e.g. on Streamlit Cloud).
sys.path.insert(0, str(ROOT))

from casehold_ir.data import load_split  # noqa: E402
from casehold_ir.models import DenseModel, TfidfModel, build_bm25  # noqa: E402
from casehold_ir.preprocess import tokenize  # noqa: E402

FINETUNED = ROOT / "models" / "minilm-casehold"
TOP_N = 10
DEPTH = 1000

st.set_page_config(page_title="CaseHOLD Legal Holding Search", layout="wide")


@st.cache_resource(show_spinner="Loading CaseHOLD test split and building indexes...")
def load_system():
    cfg_path = ROOT / "results" / "validation.json"
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {
        "bm25": {"k1": 0.6, "b": 0.9}, "tfidf": {"remove_stop": True, "use_stem": True}}
    test = load_split("test")
    tfidf = TfidfModel(**cfg["tfidf"]).fit(test.docs)
    bm25 = build_bm25(test.docs, cfg["bm25"]["k1"], cfg["bm25"]["b"])
    dense = {"Base (all-MiniLM-L6-v2)": DenseModel("sentence-transformers/all-MiniLM-L6-v2")}
    if FINETUNED.exists():
        dense["Fine-tuned on CaseHOLD"] = DenseModel(str(FINETUNED))
    doc_emb = {name: m.encode(test.docs, "test_docs") for name, m in dense.items()}
    return test, tfidf, bm25, dense, doc_emb


def ranked(scores: dict[int, float] | np.ndarray, depth: int = DEPTH) -> list[int]:
    if isinstance(scores, dict):
        return [d for d, _ in sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:depth]]
    idx = np.argpartition(-scores, depth)[:depth]
    return idx[np.lexsort((idx, -scores[idx]))].tolist()


def rrf(lists: list[list[int]], k: int) -> list[tuple[int, float]]:
    fused: dict[int, float] = {}
    for lst in lists:
        for rank, d in enumerate(lst, start=1):
            fused[d] = fused.get(d, 0.0) + 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda x: (-x[1], x[0]))


test, tfidf, bm25, dense_models, doc_emb = load_system()

st.title("Legal Holding Retrieval on CaseHOLD")
st.caption(f"Search {len(test.docs):,} holdings from the CaseHOLD test split with four ranking models.")

with st.sidebar:
    st.header("Query")
    mode = st.radio("Source", ["CaseHOLD test query", "Type your own passage"])
    if mode == "CaseHOLD test query":
        qid = st.number_input("Test query id", 0, len(test.queries) - 1, 0)
        query = test.queries[qid]
        correct = int(test.relevant[qid])
    else:
        query = st.text_area("Citing passage", height=200,
                             value="The landowner may recover for the reduced market value "
                                   "caused by buyers' fear of a gas pipeline crossing the land.")
        correct = None
    st.header("Settings")
    dense_name = st.selectbox("Dense model", list(dense_models))
    rrf_k = st.select_slider("RRF constant k", [1, 5, 10, 20, 60, 100], value=60)

st.subheader("Citing context")
st.write(query)
q_tokens = tokenize(query)
with st.expander(f"Preprocessed query terms ({len(q_tokens)})"):
    st.code(" ".join(q_tokens) or "(no indexable terms)")

start = time.perf_counter()
results = {}
results["TF-IDF"] = ranked(tfidf.score([query])[0])
results["BM25"] = ranked(bm25.score_loop(q_tokens))       # walks the postings lists
q_emb = dense_models[dense_name].model.encode([query], normalize_embeddings=True)[0]
results["Sentence Transformer"] = ranked(doc_emb[dense_name] @ q_emb)
fused = rrf([results["TF-IDF"], results["BM25"], results["Sentence Transformer"]], rrf_k)
results["Hybrid RRF"] = [d for d, _ in fused]
elapsed = time.perf_counter() - start
st.caption(f"All four rankings computed in {elapsed * 1000:.0f} ms.")

if correct is not None:
    st.success(f"Correct holding: {test.docs[correct]}")
    cols = st.columns(4)
    for col, (name, lst) in zip(cols, results.items()):
        rank = lst.index(correct) + 1 if correct in lst else None
        col.metric(name, f"rank {rank}" if rank else f"> {DEPTH}")

cols = st.columns(4)
for col, (name, lst) in zip(cols, results.items()):
    with col:
        st.markdown(f"#### {name}")
        for rank, d in enumerate(lst[:TOP_N], start=1):
            text = test.docs[d]
            if d == correct:
                st.markdown(f"**{rank}. :green[✓ {text}]**")
            else:
                st.markdown(f"{rank}. {text}")
