"""
Exploratory Data Analysis (EDA) for the incident ticket classification dataset.

Analyses produced
-----------------
  - Class distribution at each taxonomy level (counts, imbalance ratio, entropy)
  - Text length statistics (chars and words)
  - Top discriminative terms per class (TF-IDF proxy)
  - Label co-occurrence (Level 1 → Level 2 heatmap)

Plots saved to `output_dir/`
  - class_distributions.png
  - text_length_distribution.png
  - label_hierarchy_heatmap.png
  - top_terms_per_class.png

Usage
-----
    from src.eda import EDAAnalyzer
    analyzer = EDAAnalyzer(df, output_dir="eda_results")
    report = analyzer.run_full_eda()
"""

import json
import logging
import os

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe for scripts
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_PALETTE = "viridis"


class EDAAnalyzer:
    """Run a full EDA pass on the incident DataFrame and persist results."""

    def __init__(self, df: pd.DataFrame, output_dir: str = "eda_results"):
        self.df = df.copy()
        self.output_dir = output_dir
        self._report: Dict = {}

    # ------------------------------------------------------------------
    # Analysis methods
    # ------------------------------------------------------------------

    def analyze_class_distribution(self) -> Dict:
        """
        For each taxonomy level column (level_1/2/3) compute:
          - per-class counts
          - number of distinct classes
          - imbalance ratio  (max count / min count)
          - Shannon entropy of the label distribution
        """
        dist: Dict = {}
        for lvl in range(1, 4):
            col = f"level_{lvl}"
            if col not in self.df.columns:
                continue
            counts = self.df[col].dropna().value_counts()
            probs = counts / counts.sum()
            dist[col] = {
                "counts": counts.to_dict(),
                "num_classes": int(len(counts)),
                "imbalance_ratio": float(counts.max() / counts.min()),
                "entropy_bits": float(-(probs * np.log2(probs + 1e-12)).sum()),
            }
        self._report["class_distribution"] = dist
        return dist

    def analyze_text_statistics(self, text_col: str = "description") -> Dict:
        """Compute character-length and word-count statistics."""
        col = text_col if text_col in self.df.columns else self.df.columns[0]
        series = self.df[col].dropna()

        char_len = series.str.len()
        word_cnt = series.str.split().str.len()

        def _stats(s: pd.Series) -> Dict:
            return {
                "mean": float(s.mean()),
                "median": float(s.median()),
                "std": float(s.std()),
                "min": int(s.min()),
                "max": int(s.max()),
                "p25": float(s.quantile(0.25)),
                "p75": float(s.quantile(0.75)),
            }

        stats = {
            "num_samples": len(self.df),
            "char_length": _stats(char_len),
            "word_count": _stats(word_cnt),
        }
        self._report["text_statistics"] = stats
        return stats

    def analyze_top_terms(
        self,
        text_col: str = "description",
        n_terms: int = 10,
    ) -> Dict:
        """
        For each class at every level, fit a TF-IDF vectorizer on the class
        subset and return the top-n terms.  Useful as a feature-importance
        proxy before model training.
        """
        col = text_col if text_col in self.df.columns else self.df.columns[0]
        top_terms: Dict = {}

        for lvl in range(1, 4):
            level_col = f"level_{lvl}"
            if level_col not in self.df.columns:
                continue
            top_terms[level_col] = {}
            df_lvl = self.df[[col, level_col]].dropna()

            for label, group in df_lvl.groupby(level_col):
                texts = group[col].tolist()
                vec = TfidfVectorizer(
                    max_features=n_terms,
                    stop_words="english",
                    ngram_range=(1, 2),
                    min_df=1,
                )
                try:
                    vec.fit(texts)
                    top_terms[level_col][label] = list(vec.get_feature_names_out())
                except ValueError:
                    top_terms[level_col][label] = []

        self._report["top_terms_per_class"] = top_terms
        return top_terms

    def analyze_label_transition(self) -> Dict:
        """
        For each (Level-1, Level-2) pair show how often they co-occur.
        Useful to verify the taxonomy is consistent and to spot anomalies.
        """
        if "level_1" not in self.df.columns or "level_2" not in self.df.columns:
            return {}
        transitions: Dict = {}
        for l1, grp in self.df.groupby("level_1"):
            transitions[l1] = grp["level_2"].value_counts().to_dict() if "level_2" in grp else {}
        self._report["label_transitions"] = transitions
        return transitions

    # ------------------------------------------------------------------
    # Plot methods
    # ------------------------------------------------------------------

    def plot_class_distributions(self) -> None:
        """Horizontal bar charts — one per taxonomy level."""
        os.makedirs(self.output_dir, exist_ok=True)
        n_levels = sum(f"level_{i}" in self.df.columns for i in range(1, 4))
        if n_levels == 0:
            return

        fig, axes = plt.subplots(1, n_levels, figsize=(7 * n_levels, 5))
        if n_levels == 1:
            axes = [axes]

        ax_idx = 0
        for lvl in range(1, 4):
            col = f"level_{lvl}"
            if col not in self.df.columns:
                continue
            ax = axes[ax_idx]
            counts = self.df[col].dropna().value_counts()
            colors = sns.color_palette(_PALETTE, len(counts))
            ax.barh(counts.index, counts.values, color=colors)
            ax.invert_yaxis()
            ax.set_title(f"Level {lvl} Distribution", fontsize=13, fontweight="bold")
            ax.set_xlabel("Sample Count")
            for i, v in enumerate(counts.values):
                ax.text(v + 0.1, i, str(v), va="center", fontsize=9)
            ax_idx += 1

        plt.tight_layout()
        out = os.path.join(self.output_dir, "class_distributions.png")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Saved → %s", out)

    def plot_text_length_distribution(self, text_col: str = "description") -> None:
        """Histograms of character length and word count."""
        os.makedirs(self.output_dir, exist_ok=True)
        col = text_col if text_col in self.df.columns else self.df.columns[0]

        char_len = self.df[col].dropna().str.len()
        word_cnt = self.df[col].dropna().str.split().str.len()

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        for ax, data, label, color in [
            (axes[0], char_len, "Characters", "steelblue"),
            (axes[1], word_cnt, "Words",      "darkorange"),
        ]:
            ax.hist(data, bins=20, color=color, edgecolor="white", alpha=0.85)
            ax.axvline(data.mean(), color="red", linestyle="--",
                       label=f"Mean: {data.mean():.1f}")
            ax.axvline(data.median(), color="green", linestyle=":",
                       label=f"Median: {data.median():.1f}")
            ax.set_title(f"{label} Distribution", fontweight="bold")
            ax.set_xlabel(label)
            ax.legend(fontsize=9)

        plt.tight_layout()
        out = os.path.join(self.output_dir, "text_length_distribution.png")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Saved → %s", out)

    def plot_label_hierarchy_heatmap(self) -> None:
        """Level 1 → Level 2 co-occurrence heatmap."""
        if "level_1" not in self.df.columns or "level_2" not in self.df.columns:
            return

        os.makedirs(self.output_dir, exist_ok=True)
        pivot = self.df.groupby(["level_1", "level_2"]).size().unstack(fill_value=0)

        fig, ax = plt.subplots(figsize=(max(8, len(pivot.columns) * 1.2), max(4, len(pivot) * 0.8)))
        sns.heatmap(
            pivot,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=ax,
            linewidths=0.5,
            cbar_kws={"label": "Count"},
        )
        ax.set_title("Label Co-occurrence: Level 1 → Level 2", fontsize=13, fontweight="bold")
        ax.set_ylabel("Level 1")
        ax.set_xlabel("Level 2")
        plt.tight_layout()

        out = os.path.join(self.output_dir, "label_hierarchy_heatmap.png")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Saved → %s", out)

    def plot_top_terms(self, n_terms: int = 8) -> None:
        """
        Bar chart of top TF-IDF terms per Level-1 class.
        Shows the most discriminative vocabulary for each domain.
        """
        top_terms = self._report.get("top_terms_per_class", {})
        if "level_1" not in top_terms:
            top_terms = self.analyze_top_terms(n_terms=n_terms)

        classes = list(top_terms.get("level_1", {}).keys())
        if not classes:
            return

        os.makedirs(self.output_dir, exist_ok=True)
        n_cols = min(len(classes), 3)
        n_rows = (len(classes) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(6 * n_cols, 3.5 * n_rows))
        axes = np.array(axes).flatten()

        for i, cls in enumerate(classes):
            terms = top_terms["level_1"].get(cls, [])[:n_terms]
            ax = axes[i]
            ax.barh(range(len(terms)), range(len(terms), 0, -1),
                    color=sns.color_palette(_PALETTE, len(terms)))
            ax.set_yticks(range(len(terms)))
            ax.set_yticklabels(terms, fontsize=9)
            ax.invert_yaxis()
            ax.set_title(f'Top terms: "{cls}"', fontsize=11, fontweight="bold")
            ax.set_xlabel("TF-IDF rank")

        for j in range(len(classes), len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout()
        out = os.path.join(self.output_dir, "top_terms_per_class.png")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Saved → %s", out)

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def run_full_eda(self) -> Dict:
        """
        Run all analyses and plots; save a JSON summary report.

        Returns the full report dictionary.
        """
        logger.info("=" * 50)
        logger.info("EXPLORATORY DATA ANALYSIS  (%d samples)", len(self.df))
        logger.info("=" * 50)

        self.analyze_class_distribution()
        self.analyze_text_statistics()
        self.analyze_top_terms()
        self.analyze_label_transition()

        self.plot_class_distributions()
        self.plot_text_length_distribution()
        self.plot_label_hierarchy_heatmap()
        self.plot_top_terms()

        os.makedirs(self.output_dir, exist_ok=True)
        report_path = os.path.join(self.output_dir, "eda_report.json")
        with open(report_path, "w") as f:
            json.dump(self._report, f, indent=2)

        self._log_summary()
        logger.info("EDA complete. All outputs in %s/", self.output_dir)
        return self._report

    def _log_summary(self) -> None:
        stats = self._report.get("text_statistics", {})
        dist = self._report.get("class_distribution", {})

        logger.info("── Text statistics ──────────────────────────────")
        logger.info(
            "  Samples: %d  |  Avg words: %.1f  |  Avg chars: %.0f",
            stats.get("num_samples", 0),
            stats.get("word_count", {}).get("mean", 0),
            stats.get("char_length", {}).get("mean", 0),
        )
        logger.info("── Class distribution ───────────────────────────")
        for col, info in dist.items():
            logger.info(
                "  %-10s  classes: %2d  |  imbalance: %.1fx  |  entropy: %.2f bits",
                col,
                info["num_classes"],
                info["imbalance_ratio"],
                info["entropy_bits"],
            )
        logger.info("─────────────────────────────────────────────────")
