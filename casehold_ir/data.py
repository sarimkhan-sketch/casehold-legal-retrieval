"""Load CaseHOLD (LexGLUE version) and recast it as a retrieval task.

Every citing context is a query. Every unique holding text in the same split
is a document. Each query has exactly one relevant document: its correct
holding. The five answer options are kept for the original 5-choice setting.
"""
from dataclasses import dataclass

import numpy as np
from datasets import load_dataset

HF_NAME = "coastalcph/lex_glue"
HF_CONFIG = "case_hold"


@dataclass
class RetrievalSplit:
    name: str
    queries: list[str]          # citing contexts
    docs: list[str]             # unique holdings (the document pool)
    relevant: np.ndarray        # relevant[i] = doc id of the correct holding of query i
    options: np.ndarray         # options[i, j] = doc id of answer option j of query i
    labels: np.ndarray          # labels[i] = position (0-4) of the correct option


def load_split(name: str) -> RetrievalSplit:
    ds = load_dataset(HF_NAME, HF_CONFIG, split=name)
    doc_id: dict[str, int] = {}
    docs: list[str] = []
    options = np.zeros((len(ds), 5), dtype=np.int64)
    for i, endings in enumerate(ds["endings"]):
        for j, text in enumerate(endings):
            if text not in doc_id:
                doc_id[text] = len(docs)
                docs.append(text)
            options[i, j] = doc_id[text]
    labels = np.asarray(ds["label"], dtype=np.int64)
    relevant = options[np.arange(len(ds)), labels]
    return RetrievalSplit(name, list(ds["context"]), docs, relevant, options, labels)


def load_train_pairs():
    """(context, correct holding, one wrong holding) triples for fine-tuning."""
    ds = load_dataset(HF_NAME, HF_CONFIG, split="train")
    rng = np.random.default_rng(13)
    anchors, positives, negatives = [], [], []
    for ctx, endings, label in zip(ds["context"], ds["endings"], ds["label"]):
        wrong = [j for j in range(5) if j != label]
        anchors.append(ctx)
        positives.append(endings[label])
        negatives.append(endings[int(rng.choice(wrong))])
    return anchors, positives, negatives
