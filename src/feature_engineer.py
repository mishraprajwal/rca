"""
Feature engineering for IT incident text classification.

Combines three feature representations in a single sparse matrix:
  1. Word n-gram TF-IDF  — captures vocabulary (e.g. "database timeout")
  2. Char n-gram TF-IDF  — captures subword patterns (e.g. "API", "DB", "HTTP")
  3. Meta features       — 14 hand-crafted binary/numeric signals for IT text

Usage
-----
    engineer = FeatureEngineer()
    X_train = engineer.fit_transform(train_texts, y_train)
    X_test  = engineer.transform(test_texts)
    engineer.save("models/features/")
    engineer = FeatureEngineer.load("models/features/")
"""

import logging
import os
import re
from typing import List, Optional, Tuple

import joblib
import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Meta-feature extractor
# ---------------------------------------------------------------------------

class MetaFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Hand-crafted features for IT incident text.

    Numeric:
      text_length, word_count, vocab_richness, uppercase_ratio

    Binary (pattern present / absent):
      has_http_error   — 4xx / 5xx codes
      has_ip_address   — IPv4 pattern
      has_timeout      — timeout / timed out
      has_crash        — crash / crashed
      has_failure      — fail / failure / failed
      has_connection   — connection / connectivity
      has_memory       — memory / RAM / OOM / heap
      has_cpu          — cpu / processor / utilization
      has_disk         — disk / storage / drive / RAID
      has_network      — network / switch / router / firewall
    """

    _PATTERNS: dict = {
        "has_http_error": re.compile(r"\b[45]\d{2}\b"),
        "has_ip_address": re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"),
        "has_timeout": re.compile(r"time[\s\-]?out|timed?\s+out", re.I),
        "has_crash": re.compile(r"crash(?:ed|ing)?", re.I),
        "has_failure": re.compile(r"fail(?:ed|ure|ing)?", re.I),
        "has_connection": re.compile(r"connect(?:ion|ivity|ing|ed)?", re.I),
        "has_memory": re.compile(r"\b(?:memory|ram|oom|heap|cache)\b", re.I),
        "has_cpu": re.compile(r"\b(?:cpu|processor|core|thread|utilization)\b", re.I),
        "has_disk": re.compile(r"\b(?:disk|storage|drive|ssd|raid|volume)\b", re.I),
        "has_network": re.compile(r"\b(?:network|switch|router|firewall|vlan|ethernet)\b", re.I),
    }

    @property
    def feature_names(self) -> List[str]:
        return ["text_length", "word_count", "vocab_richness", "uppercase_ratio"] + list(
            self._PATTERNS.keys()
        )

    def fit(self, X, y=None):
        return self

    def transform(self, X: List[str]) -> np.ndarray:
        rows = []
        for text in X:
            text = text if isinstance(text, str) else ""
            words = text.split()
            row = [
                len(text),
                len(words),
                len(set(words)) / max(len(words), 1),
                sum(c.isupper() for c in text) / max(len(text), 1),
            ] + [float(bool(pat.search(text))) for pat in self._PATTERNS.values()]
            rows.append(row)
        return np.array(rows, dtype=np.float32)


# ---------------------------------------------------------------------------
# Main feature engineer
# ---------------------------------------------------------------------------

class FeatureEngineer:
    """
    Builds a combined sparse feature matrix from incident text.

    Parameters
    ----------
    word_ngram_range : (min_n, max_n) for word TF-IDF  (default (1, 3))
    char_ngram_range : (min_n, max_n) for char TF-IDF  (default (2, 5))
    word_max_features : vocabulary cap for word TF-IDF (default 15 000)
    char_max_features : vocabulary cap for char TF-IDF (default 8 000)
    use_char_features : include char n-gram TF-IDF    (default True)
    use_meta_features : include hand-crafted features (default True)
    feature_selection_k : if set, keep top-k features by chi² (default None)
    """

    def __init__(
        self,
        word_ngram_range: Tuple[int, int] = (1, 3),
        char_ngram_range: Tuple[int, int] = (2, 5),
        word_max_features: int = 15_000,
        char_max_features: int = 8_000,
        use_char_features: bool = True,
        use_meta_features: bool = True,
        feature_selection_k: Optional[int] = None,
    ):
        self.word_ngram_range = word_ngram_range
        self.char_ngram_range = char_ngram_range
        self.word_max_features = word_max_features
        self.char_max_features = char_max_features
        self.use_char_features = use_char_features
        self.use_meta_features = use_meta_features
        self.feature_selection_k = feature_selection_k

        self._word_vec = TfidfVectorizer(
            analyzer="word",
            ngram_range=word_ngram_range,
            max_features=word_max_features,
            sublinear_tf=True,
            min_df=1,
            strip_accents="unicode",
        )
        self._char_vec = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=char_ngram_range,
            max_features=char_max_features,
            sublinear_tf=True,
            min_df=1,
        )
        self._meta = MetaFeatureExtractor()
        self._selector: Optional[SelectKBest] = None
        self._is_fitted = False

    # ------------------------------------------------------------------
    # Fit / transform
    # ------------------------------------------------------------------

    def fit_transform(self, X: List[str], y=None):
        """Fit all extractors on X and return the combined feature matrix."""
        X_word = self._word_vec.fit_transform(X)
        parts = [X_word]

        if self.use_char_features:
            parts.append(self._char_vec.fit_transform(X))

        if self.use_meta_features:
            parts.append(csr_matrix(self._meta.fit_transform(X)))

        X_combined = hstack(parts, format="csr") if len(parts) > 1 else X_word

        if self.feature_selection_k and y is not None:
            k = min(self.feature_selection_k, X_combined.shape[1])
            self._selector = SelectKBest(chi2, k=k)
            X_combined = self._selector.fit_transform(X_combined, y)
            logger.info("Feature selection: kept %d / %d features", k, X_combined.shape[1])

        self._is_fitted = True
        logger.info(
            "Feature matrix: %d samples × %d features "
            "(word=%d, char=%s, meta=%s)",
            X_combined.shape[0],
            X_combined.shape[1],
            X_word.shape[1],
            self._char_vec.transform(X[:1]).shape[1] if self.use_char_features else 0,
            len(self._meta.feature_names) if self.use_meta_features else 0,
        )
        return X_combined

    def transform(self, X: List[str]):
        """Transform new data using the already-fitted extractors."""
        if not self._is_fitted:
            raise RuntimeError("FeatureEngineer must be fitted before transform().")

        parts = [self._word_vec.transform(X)]

        if self.use_char_features:
            parts.append(self._char_vec.transform(X))

        if self.use_meta_features:
            parts.append(csr_matrix(self._meta.transform(X)))

        X_combined = hstack(parts, format="csr") if len(parts) > 1 else parts[0]

        if self._selector is not None:
            X_combined = self._selector.transform(X_combined)

        return X_combined

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

    def top_features_per_class(self, classifier, n: int = 15) -> dict:
        """
        Return top-n discriminative word n-gram features per class.
        Requires classifier with a `coef_` attribute (LR, LinearSVC).
        """
        vocab = self._word_vec.get_feature_names_out()
        result = {}

        if not (hasattr(classifier, "coef_") and hasattr(classifier, "classes_")):
            return result

        coef = classifier.coef_
        for i, cls in enumerate(classifier.classes_):
            row = coef[i] if coef.ndim > 1 else coef
            top_idx = np.argsort(row)[-n:][::-1]
            result[cls] = [vocab[j] for j in top_idx if j < len(vocab)]

        return result

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        joblib.dump(self._word_vec, os.path.join(path, "word_vectorizer.pkl"))
        if self.use_char_features:
            joblib.dump(self._char_vec, os.path.join(path, "char_vectorizer.pkl"))
        if self.use_meta_features:
            joblib.dump(self._meta, os.path.join(path, "meta_extractor.pkl"))
        if self._selector is not None:
            joblib.dump(self._selector, os.path.join(path, "feature_selector.pkl"))
        # Save config flags
        joblib.dump(
            {
                "use_char_features": self.use_char_features,
                "use_meta_features": self.use_meta_features,
                "word_ngram_range": self.word_ngram_range,
                "char_ngram_range": self.char_ngram_range,
            },
            os.path.join(path, "config.pkl"),
        )
        logger.info("Saved FeatureEngineer to %s", path)

    @classmethod
    def load(cls, path: str) -> "FeatureEngineer":
        cfg = joblib.load(os.path.join(path, "config.pkl"))
        eng = cls(
            use_char_features=cfg["use_char_features"],
            use_meta_features=cfg["use_meta_features"],
            word_ngram_range=cfg["word_ngram_range"],
            char_ngram_range=cfg["char_ngram_range"],
        )
        eng._word_vec = joblib.load(os.path.join(path, "word_vectorizer.pkl"))

        char_path = os.path.join(path, "char_vectorizer.pkl")
        if os.path.exists(char_path):
            eng._char_vec = joblib.load(char_path)

        meta_path = os.path.join(path, "meta_extractor.pkl")
        if os.path.exists(meta_path):
            eng._meta = joblib.load(meta_path)

        sel_path = os.path.join(path, "feature_selector.pkl")
        eng._selector = joblib.load(sel_path) if os.path.exists(sel_path) else None

        eng._is_fitted = True
        logger.info("Loaded FeatureEngineer from %s", path)
        return eng
