"""Tokenizer used by the sparse models (TF-IDF and BM25)."""
import re
from functools import lru_cache

from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Words that occur in almost every holding or citation and carry no topic.
LEGAL_STOP_WORDS = {"holding", "held", "v", "see", "eg", "id", "cf", "supra", "citing"}
STOP_WORDS = frozenset(ENGLISH_STOP_WORDS) | LEGAL_STOP_WORDS

_MARKER = re.compile(r"<holding>")
_DOTTED = re.compile(r"\b(?:[a-z]\.){2,}")   # e.g. -> eg, i.e. -> ie, u.s. -> us
_SPLIT = re.compile(r"[^a-z0-9]+")
_HAS_DIGIT = re.compile(r"\d")
_stemmer = PorterStemmer()


@lru_cache(maxsize=None)
def stem(token: str) -> str:
    return _stemmer.stem(token)


def tokenize(text: str, remove_stop: bool = True, use_stem: bool = True) -> list[str]:
    text = _MARKER.sub(" ", text.lower())
    text = _DOTTED.sub(lambda m: m.group(0).replace(".", ""), text)
    tokens = [t for t in _SPLIT.split(text) if len(t) > 1 and not _HAS_DIGIT.search(t)]
    if remove_stop:
        tokens = [t for t in tokens if t not in STOP_WORDS]
    if use_stem:
        tokens = [stem(t) for t in tokens]
    return tokens


def last_words(text: str, n: int | None) -> str:
    """Keep only the last n words of a citing context (None keeps everything)."""
    if n is None:
        return text
    return " ".join(text.split()[-n:])
