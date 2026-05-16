"""
Hierarchical incident classifier.

Architecture: Local Classifier Per Parent Node (LCPN) with parent-feature conditioning
---------------------------------------------------------------------------------------
  Level 1: classifier( text_features )
  Level 2: classifier( text_features  ⊕  one_hot( true_level1_label ) )   ← oracle train
  Level 3: classifier( text_features  ⊕  one_hot( true_level1 )
                                      ⊕  one_hot( true_level2 ) )

During **inference**, predicted parent labels replace true ones (greedy top-down).
This is known as "hierarchical top-down classification with parent features".

Strategy flag
-------------
  strategy='hierarchical' (default) — parent-conditioned (LCPN)
  strategy='flat'                   — independent classifiers per level (baseline)

Model backends  (set model_name=...)
--------------------------------------
  'lr'   LogisticRegression        — strong text baseline, calibrated probabilities
  'svm'  LinearSVC                 — fast, excellent for high-dimensional TF-IDF
  'rf'   RandomForestClassifier    — robust, handles feature interactions
  'cnb'  ComplementNB              — especially good for imbalanced text data

Usage
-----
    clf = HierarchicalClassifier(taxonomy_manager, model_name='lr')
    clf.fit(df_train, text_col='processed_text')

    path, confidences = clf.predict_with_confidence("database timeout in production")
    # → (['Software', 'Application', 'Database'], [0.96, 0.89, 0.82])

    clf.save("models/")
    clf = HierarchicalClassifier.load("models/", taxonomy_manager)
"""

import json
import logging
import os
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC

from .feature_engineer import FeatureEngineer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _to_onehot(indices: np.ndarray, n_classes: int) -> np.ndarray:
    """Convert integer class indices to a (n_samples, n_classes) one-hot matrix."""
    out = np.zeros((len(indices), n_classes), dtype=np.float32)
    out[np.arange(len(indices)), indices] = 1.0
    return out


# ---------------------------------------------------------------------------
# Classifier factory
# ---------------------------------------------------------------------------

class ClassifierFactory:
    """
    Build sklearn-compatible classifiers by name.

    All classifiers use `class_weight='balanced'` (where supported) to
    handle class-imbalanced incident data.
    """

    @staticmethod
    def build(name: str, n_samples: int = 100):
        """
        Parameters
        ----------
        name      : one of 'lr', 'svm', 'rf', 'cnb'
        n_samples : used to set safe CV splits for SVM calibration
        """
        if name == "lr":
            return LogisticRegression(
                C=1.0,
                max_iter=2000,
                random_state=42,
                class_weight="balanced",
                solver="lbfgs",
                multi_class="auto",
            )
        if name == "svm":
            # LinearSVC has no predict_proba — wrap with Platt scaling.
            # Use cv=min(3, floor(n_samples / n_classes)) but at least 2.
            cv_splits = max(2, min(3, n_samples // 10))
            return CalibratedClassifierCV(
                LinearSVC(
                    C=1.0,
                    max_iter=3000,
                    random_state=42,
                    class_weight="balanced",
                ),
                cv=cv_splits,
            )
        if name == "rf":
            return RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                class_weight="balanced",
                n_jobs=-1,
                min_samples_leaf=1,
            )
        if name == "cnb":
            # ComplementNB — great for imbalanced multi-class text.
            # Input must be non-negative (TF-IDF already is).
            return ComplementNB(alpha=0.1)
        raise ValueError(
            f"Unknown model '{name}'. Choose from: 'lr', 'svm', 'rf', 'cnb'."
        )


# ---------------------------------------------------------------------------
# Hierarchical classifier
# ---------------------------------------------------------------------------

class HierarchicalClassifier:
    """
    Three-level hierarchical incident classifier.

    See module docstring for architecture details.
    """

    METADATA_FILE = "classifier_meta.json"

    def __init__(
        self,
        taxonomy_manager,
        model_name: str = "lr",
        strategy: str = "hierarchical",
    ):
        self.taxonomy_manager = taxonomy_manager
        self.model_name = model_name
        self.strategy = strategy  # 'hierarchical' | 'flat'

        # Shared text feature extractor (fitted once)
        self.feature_engineer = FeatureEngineer()

        # Per-level artifacts
        self._classifiers: Dict[int, object] = {}
        self._label_encoders: Dict[int, LabelEncoder] = {}
        self._is_fitted = False

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame, text_col: str = "processed_text") -> None:
        """
        Fit classifiers for all three taxonomy levels.

        Parameters
        ----------
        df       : DataFrame with columns `text_col`, `level_1`, `level_2`, `level_3`
        text_col : column containing preprocessed incident text
        """
        levels = [c for c in ["level_1", "level_2", "level_3"] if c in df.columns]
        if not levels:
            raise ValueError("DataFrame must have at least one level_N column.")

        X_text = df[text_col].tolist()

        # Fit the shared text feature extractor once on level-1 labels
        y1 = df["level_1"].tolist()
        X_base = self.feature_engineer.fit_transform(X_text, y1)

        # Level 1 — text features only
        self._fit_level(X_base, y1, level=1)

        if "level_2" in df.columns:
            y2 = df["level_2"].tolist()
            X_level2 = self._augment_with_parent(X_base, y1, level=1)
            self._fit_level(X_level2, y2, level=2)

            if "level_3" in df.columns:
                y3 = df["level_3"].tolist()
                # Augment with both level-1 and level-2 true labels
                X_level3 = self._augment_with_parent(X_level2, y2, level=2)
                self._fit_level(X_level3, y3, level=3)

        self._is_fitted = True
        logger.info(
            "HierarchicalClassifier fitted (strategy=%s, model=%s, levels=%s)",
            self.strategy,
            self.model_name,
            list(self._classifiers.keys()),
        )

    def _fit_level(self, X, y: List[str], level: int) -> None:
        """Encode labels, build classifier, fit, store."""
        le = LabelEncoder()
        y_enc = le.fit_transform(y)
        self._label_encoders[level] = le

        clf = ClassifierFactory.build(self.model_name, n_samples=len(y))
        try:
            clf.fit(X, y_enc)
        except ValueError:
            # Fallback for edge cases (e.g. CalibratedClassifierCV on tiny data)
            logger.warning(
                "  Level %d: primary classifier failed, falling back to LR.", level
            )
            from sklearn.linear_model import LogisticRegression
            clf = LogisticRegression(
                class_weight="balanced", max_iter=500, solver="lbfgs"
            )
            clf.fit(X, y_enc)
        self._classifiers[level] = clf

        logger.info(
            "  Level %d: %d samples, %d classes (%s)",
            level,
            len(y),
            len(le.classes_),
            list(le.classes_),
        )

    def _augment_with_parent(self, X_base, y_parent: List[str], level: int):
        """
        If strategy='hierarchical', append one-hot parent-label features.
        If strategy='flat', return X_base unchanged.
        """
        if self.strategy == "flat":
            return X_base

        le = self._label_encoders[level]
        try:
            enc = le.transform(y_parent)
        except Exception:
            # Fallback for unseen labels (shouldn't happen at train time)
            enc = np.zeros(len(y_parent), dtype=int)

        onehot = csr_matrix(_to_onehot(enc, len(le.classes_)))
        return hstack([X_base, onehot], format="csr")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_with_confidence(
        self, text: str
    ) -> Tuple[List[str], List[float]]:
        """
        Predict the full taxonomy path and per-level confidence for one text.

        Returns
        -------
        path        : ['Level1', 'Level2', 'Level3']
        confidences : [0.96, 0.89, 0.82]
        """
        if not self._is_fitted:
            raise RuntimeError("Classifier is not fitted. Call fit() or load() first.")

        X_base = self.feature_engineer.transform([text])
        path: List[str] = []
        confidences: List[float] = []

        X_current = X_base

        for level in sorted(self._classifiers.keys()):
            pred_label, conf = self._predict_single_level(X_current, level)
            if pred_label is None:
                break
            path.append(pred_label)
            confidences.append(conf)

            # Prepare augmented features for next level
            if level + 1 in self._classifiers and self.strategy == "hierarchical":
                le = self._label_encoders[level]
                try:
                    enc = le.transform([pred_label])
                except Exception:
                    enc = np.array([0])
                onehot = csr_matrix(_to_onehot(enc, len(le.classes_)))
                X_current = hstack([X_current, onehot], format="csr")

        return path, confidences

    def predict_hierarchy(self, text: str) -> List[str]:
        """Return just the predicted path (backward-compatible API)."""
        path, _ = self.predict_with_confidence(text)
        return path

    def predict_batch(
        self, texts: List[str]
    ) -> List[Tuple[List[str], List[float]]]:
        """Predict paths and confidence scores for a list of texts."""
        return [self.predict_with_confidence(t) for t in texts]

    def _predict_single_level(
        self, X, level: int
    ) -> Tuple[Optional[str], float]:
        """
        Run one level's classifier and return (label_string, confidence).
        """
        clf = self._classifiers.get(level)
        le = self._label_encoders.get(level)
        if clf is None or le is None:
            return None, 0.0

        idx = int(clf.predict(X)[0])

        if hasattr(clf, "predict_proba"):
            proba = clf.predict_proba(X)[0]
            confidence = float(np.max(proba))
        else:
            confidence = 1.0  # fallback for non-probabilistic models

        label = le.inverse_transform([idx])[0]
        return label, confidence

    # ------------------------------------------------------------------
    # Model introspection
    # ------------------------------------------------------------------

    def top_features(self, level: int = 1, n: int = 15) -> Dict[str, List[str]]:
        """
        Return top-n discriminative word features per class at `level`.
        Only works for LR and SVM (have coef_).
        """
        clf = self._classifiers.get(level)
        le = self._label_encoders.get(level)

        # Unwrap CalibratedClassifierCV
        base_clf = getattr(clf, "estimator", clf)
        if not (hasattr(base_clf, "coef_") and le is not None):
            return {}

        vocab = self.feature_engineer._word_vec.get_feature_names_out()
        coef = base_clf.coef_
        result = {}

        for i, cls in enumerate(le.classes_):
            # Binary LR stores only one row (positive class coef)
            if coef.ndim > 1 and i < coef.shape[0]:
                row = coef[i]
            elif coef.ndim == 1:
                row = coef
            else:
                continue
            top_idx = np.argsort(row)[-n:][::-1]
            result[cls] = [vocab[j] for j in top_idx if j < len(vocab)]

        return result

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save the full classifier (feature engineer + per-level models)."""
        os.makedirs(path, exist_ok=True)

        # Feature engineer
        feat_path = os.path.join(path, "features")
        self.feature_engineer.save(feat_path)

        # Per-level classifiers + label encoders
        for level, clf in self._classifiers.items():
            joblib.dump(clf, os.path.join(path, f"clf_level_{level}.pkl"))
            joblib.dump(
                self._label_encoders[level],
                os.path.join(path, f"le_level_{level}.pkl"),
            )

        # Metadata
        meta = {
            "model_name": self.model_name,
            "strategy": self.strategy,
            "levels": sorted(self._classifiers.keys()),
        }
        with open(os.path.join(path, self.METADATA_FILE), "w") as f:
            json.dump(meta, f, indent=2)

        logger.info("Saved HierarchicalClassifier to %s", path)

    @classmethod
    def load(cls, path: str, taxonomy_manager) -> "HierarchicalClassifier":
        """Load a previously saved classifier."""
        meta_path = os.path.join(path, cls.METADATA_FILE)
        with open(meta_path) as f:
            meta = json.load(f)

        obj = cls(
            taxonomy_manager,
            model_name=meta["model_name"],
            strategy=meta["strategy"],
        )
        obj.feature_engineer = FeatureEngineer.load(os.path.join(path, "features"))

        for level in meta["levels"]:
            obj._classifiers[level] = joblib.load(
                os.path.join(path, f"clf_level_{level}.pkl")
            )
            obj._label_encoders[level] = joblib.load(
                os.path.join(path, f"le_level_{level}.pkl")
            )

        obj._is_fitted = True
        logger.info(
            "Loaded HierarchicalClassifier from %s (strategy=%s, model=%s)",
            path,
            meta["strategy"],
            meta["model_name"],
        )
        return obj

    # ------------------------------------------------------------------
    # Legacy compatibility shims
    # ------------------------------------------------------------------

    def load_models(self, model_path: str = "models") -> None:
        """Legacy: load from a path (use HierarchicalClassifier.load() instead)."""
        loaded = HierarchicalClassifier.load(model_path, self.taxonomy_manager)
        self._classifiers = loaded._classifiers
        self._label_encoders = loaded._label_encoders
        self.feature_engineer = loaded.feature_engineer
        self.model_name = loaded.model_name
        self.strategy = loaded.strategy
        self._is_fitted = True

    def save_models(self, model_path: str = "models") -> None:
        """Legacy: save to a path (use save() instead)."""
        self.save(model_path)

