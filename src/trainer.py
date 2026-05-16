"""
Training pipeline for the Hierarchical RCA system.

Steps
-----
1. Load raw CSV data
2. Preprocess text
3. (Optional) Run EDA  → eda_results/
4. Model selection via stratified k-fold cross-validation
   Compares: Logistic Regression, LinearSVC, Random Forest, ComplementNB
5. Train the winning model on the full training set
6. Save model + experiment log

Usage
-----
    trainer = ModelTrainer()
    trainer.run_training_pipeline()
"""

import json
import logging
import os
import time
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder

from .data_loader import DataLoader
from .eda import EDAAnalyzer
from .model import ClassifierFactory, HierarchicalClassifier
from .preprocessor import TextPreprocessor
from .taxonomy import TaxonomyManager

logger = logging.getLogger(__name__)

_MODEL_NAMES = ["lr", "svm", "rf", "cnb"]


class ModelTrainer:
    """
    End-to-end training orchestrator.

    Config keys (all optional)
    --------------------------
    model_name       : force a backend ('lr'|'svm'|'rf'|'cnb'), or 'auto' to run CV
    strategy         : 'hierarchical' (default) | 'flat'
    data_file        : CSV filename in data/raw/
    taxonomy_file    : JSON filename in data/
    model_path       : where to save the trained model
    cv_splits        : StratifiedKFold splits for model comparison
    run_eda          : run EDA before training
    eda_output_dir   : where to save EDA plots/report
    experiment_log   : path to append experiment JSON lines
    """

    _DEFAULTS: Dict[str, Any] = {
        "model_name": "auto",
        "strategy": "hierarchical",
        "data_file": "incidents.csv",
        "taxonomy_file": "taxonomy.json",
        "model_path": "models/",
        "cv_splits": 3,
        "run_eda": True,
        "eda_output_dir": "eda_results/",
        "experiment_log": "experiments.jsonl",
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = {**self._DEFAULTS, **(config or {})}
        self._data_loader = DataLoader()
        self._preprocessor = TextPreprocessor()
        self._taxonomy_manager = TaxonomyManager()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_training_pipeline(self) -> None:
        t0 = time.time()
        logger.info("━" * 60)
        logger.info("HIERARCHICAL RCA — TRAINING PIPELINE")
        logger.info("━" * 60)

        logger.info("Step 1/4  Loading and preprocessing data …")
        df = self._load_and_prepare()

        if self.config["run_eda"]:
            logger.info("Step 2/4  Exploratory Data Analysis …")
            EDAAnalyzer(df, output_dir=self.config["eda_output_dir"]).run_full_eda()
        else:
            logger.info("Step 2/4  EDA skipped (run_eda=False)")

        logger.info("Step 3/4  Model selection via cross-validation …")
        best_model_name, cv_results = self._select_best_model(df)

        logger.info("Step 4/4  Training best model (%s) on full data …", best_model_name)
        self._train_full(df, best_model_name)

        elapsed = time.time() - t0
        self._log_experiment(cv_results, best_model_name, df, elapsed)

        logger.info("━" * 60)
        logger.info("Done in %.1fs.  Model → %s", elapsed, self.config["model_path"])
        logger.info("━" * 60)

    # ------------------------------------------------------------------
    # Steps
    # ------------------------------------------------------------------

    def _load_and_prepare(self) -> pd.DataFrame:
        df = self._data_loader.load_incident_data(self.config["data_file"])
        taxonomy_path = os.path.join("data", self.config["taxonomy_file"])
        self._taxonomy_manager.load_taxonomy(taxonomy_path)
        df = self._preprocessor.preprocess_incident_data(df)
        df = df.dropna(subset=["level_1"])
        return df

    def _select_best_model(self, df: pd.DataFrame) -> Tuple[str, Dict]:
        """
        Cross-validate all model backends on level-1 labels using a fast
        word TF-IDF pipeline.  Returns (best_name, results_dict).
        """
        if self.config["model_name"] != "auto":
            name = self.config["model_name"]
            logger.info("  model_name='%s' fixed — skipping CV.", name)
            return name, {}

        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.pipeline import Pipeline

        X_text = df["processed_text"].tolist()
        y_enc = LabelEncoder().fit_transform(df["level_1"].tolist())

        min_class = min(Counter(y_enc).values())
        n_splits = min(self.config["cv_splits"], min_class)
        if n_splits < 2:
            logger.warning("Too few samples for CV — defaulting to 'lr'.")
            return "lr", {}

        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        results: Dict[str, Dict] = {}

        logger.info("  CV splits=%d  scoring=f1_weighted", n_splits)
        logger.info("  %-8s  %-10s  %s", "Model", "mean F1", "± std")
        logger.info("  " + "-" * 34)

        for name in _MODEL_NAMES:
            pipe = Pipeline([
                ("tfidf", TfidfVectorizer(
                    analyzer="word", ngram_range=(1, 2),
                    max_features=10_000, sublinear_tf=True, min_df=1,
                )),
                ("clf", ClassifierFactory.build(name, n_samples=len(X_text))),
            ])
            try:
                scores = cross_val_score(pipe, X_text, y_enc,
                                         cv=cv, scoring="f1_weighted", n_jobs=1)
                results[name] = {"mean_f1": float(scores.mean()),
                                 "std_f1": float(scores.std())}
                logger.info("  %-8s  %.4f      ±%.4f", name, scores.mean(), scores.std())
            except Exception as exc:
                logger.warning("  %s failed: %s", name, exc)
                results[name] = {"mean_f1": 0.0, "std_f1": 0.0}

        best = max(results, key=lambda k: results[k]["mean_f1"])
        logger.info("  → Best: %s  (F1=%.4f)", best, results[best]["mean_f1"])
        return best, results

    def _train_full(self, df: pd.DataFrame, model_name: str) -> HierarchicalClassifier:
        clf = HierarchicalClassifier(
            self._taxonomy_manager,
            model_name=model_name,
            strategy=self.config["strategy"],
        )
        clf.fit(df, text_col="processed_text")
        self._log_top_features(clf)
        clf.save(self.config["model_path"])
        return clf

    def _log_top_features(self, clf: HierarchicalClassifier, n: int = 8) -> None:
        top = clf.top_features(level=1, n=n)
        if not top:
            return
        logger.info("  Top discriminative features (Level 1):")
        for cls, terms in top.items():
            logger.info("    %-20s → %s", cls, ", ".join(terms[:n]))

    def _log_experiment(self, cv_results: Dict, best_model: str,
                        df: pd.DataFrame, elapsed: float) -> None:
        record = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "model_name": best_model,
            "strategy": self.config["strategy"],
            "n_train_samples": len(df),
            "cv_results": cv_results,
            "elapsed_seconds": round(elapsed, 2),
        }
        log_path = self.config["experiment_log"]
        with open(log_path, "a") as f:
            f.write(json.dumps(record) + "\n")
        logger.info("Experiment logged → %s", log_path)
