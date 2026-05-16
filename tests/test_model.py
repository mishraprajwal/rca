"""Tests for HierarchicalClassifier (new LCPN API)."""

import pytest
import pandas as pd
from src.taxonomy import TaxonomyManager
from src.model import HierarchicalClassifier
from src.preprocessor import TextPreprocessor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tm(taxonomy_file: str) -> TaxonomyManager:
    tm = TaxonomyManager()
    tm.load_taxonomy(taxonomy_file)
    return tm


def _preprocess(df: pd.DataFrame) -> pd.DataFrame:
    return TextPreprocessor().preprocess_incident_data(df)


@pytest.fixture
def trained_classifier(taxonomy_file, sample_incidents):
    """Return a classifier trained on the sample fixtures using new API."""
    df = _preprocess(sample_incidents)
    tm = _make_tm(taxonomy_file)
    clf = HierarchicalClassifier(tm, model_name="lr", strategy="hierarchical")
    clf.fit(df, text_col="processed_text")
    return clf, tm, taxonomy_file


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

class TestInit:
    def test_default_model_name(self, taxonomy_file):
        tm = _make_tm(taxonomy_file)
        clf = HierarchicalClassifier(tm)
        assert clf.model_name == "lr"

    def test_strategy_stored(self, taxonomy_file):
        tm = _make_tm(taxonomy_file)
        clf = HierarchicalClassifier(tm, strategy="hierarchical")
        assert clf.strategy == "hierarchical"

    def test_classifiers_empty_before_fit(self, taxonomy_file):
        tm = _make_tm(taxonomy_file)
        clf = HierarchicalClassifier(tm)
        assert clf._classifiers == {}


# ---------------------------------------------------------------------------
# Training (fit)
# ---------------------------------------------------------------------------

class TestFit:
    def test_fit_populates_classifiers(self, taxonomy_file, sample_incidents):
        df = _preprocess(sample_incidents)
        clf = HierarchicalClassifier(_make_tm(taxonomy_file))
        clf.fit(df, text_col="processed_text")
        assert len(clf._classifiers) > 0

    def test_fit_all_models(self, taxonomy_file, sample_incidents):
        df = _preprocess(sample_incidents)
        for name in ["lr", "svm", "rf", "cnb"]:
            clf = HierarchicalClassifier(_make_tm(taxonomy_file), model_name=name)
            clf.fit(df, text_col="processed_text")
            assert clf._classifiers, f"{name} produced no classifiers"

    def test_fit_hierarchical_vs_flat(self, taxonomy_file, sample_incidents):
        df = _preprocess(sample_incidents)
        for strategy in ["hierarchical", "flat"]:
            clf = HierarchicalClassifier(
                _make_tm(taxonomy_file), strategy=strategy
            )
            clf.fit(df, text_col="processed_text")
            assert clf._classifiers


# ---------------------------------------------------------------------------

class TestPrediction:
    def test_predict_hierarchy_returns_list(self, trained_classifier):
        clf, *_ = trained_classifier
        result = clf.predict_hierarchy("cpu server failure")
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_predict_with_confidence_returns_tuple(self, trained_classifier):
        clf, *_ = trained_classifier
        path, confidences = clf.predict_with_confidence("cpu server failure")
        assert isinstance(path, list)
        assert isinstance(confidences, list)
        assert len(path) == len(confidences)

    def test_confidence_values_in_range(self, trained_classifier):
        clf, *_ = trained_classifier
        _, confidences = clf.predict_with_confidence("database timeout production")
        for c in confidences:
            assert 0.0 <= c <= 1.0

    def test_predict_unknown_text_returns_path(self, trained_classifier):
        clf, *_ = trained_classifier
        path, _ = clf.predict_with_confidence("something completely random xyz")
        assert isinstance(path, list)

    def test_predict_batch(self, trained_classifier):
        clf, *_ = trained_classifier
        texts = ["cpu failure", "network down", "user login error"]
        results = clf.predict_batch(texts)
        assert len(results) == len(texts)
        # predict_batch returns List[Tuple[List[str], List[float]]]
        for path, confs in results:
            assert isinstance(path, list)
            assert isinstance(confs, list)

    def test_predict_empty_string(self, trained_classifier):
        clf, *_ = trained_classifier
        # Should not raise; may return any path
        path, _ = clf.predict_with_confidence("")
        assert isinstance(path, list)


# ---------------------------------------------------------------------------
# Save / Load round-trip
# ---------------------------------------------------------------------------

class TestSaveLoad:
    def test_save_creates_files(self, trained_classifier, tmp_path):
        clf, *_ = trained_classifier
        clf.save(str(tmp_path))
        pkl_files = list(tmp_path.glob("**/*.pkl"))
        assert pkl_files, "Expected at least one .pkl file"

    def test_load_predicts_correctly(self, trained_classifier, tmp_path):
        clf, tm, taxonomy_file = trained_classifier
        clf.save(str(tmp_path))

        clf2 = HierarchicalClassifier.load(str(tmp_path), _make_tm(taxonomy_file))
        path, conf = clf2.predict_with_confidence("server cpu overload")
        assert isinstance(path, list)
        assert all(0.0 <= c <= 1.0 for c in conf)

    def test_predictions_consistent_after_reload(self, trained_classifier, tmp_path):
        clf, tm, taxonomy_file = trained_classifier
        text = "database connection timeout"
        orig_path, _ = clf.predict_with_confidence(text)

        clf.save(str(tmp_path))
        clf2 = HierarchicalClassifier.load(str(tmp_path), _make_tm(taxonomy_file))
        reloaded_path, _ = clf2.predict_with_confidence(text)

        assert orig_path == reloaded_path


# ---------------------------------------------------------------------------
# Feature inspection
# ---------------------------------------------------------------------------

class TestTopFeatures:
    def test_top_features_lr(self, taxonomy_file, sample_incidents):
        df = _preprocess(sample_incidents)
        clf = HierarchicalClassifier(_make_tm(taxonomy_file), model_name="lr")
        clf.fit(df)
        top = clf.top_features(level=1, n=3)
        # May be empty dict for very small data (LR may not expose coef_ on 1 class)
        # but should never raise and must be a dict
        assert isinstance(top, dict)

    def test_top_features_rf_returns_empty(self, taxonomy_file, sample_incidents):
        """RF has no coef_ so top_features should return {} gracefully."""
        df = _preprocess(sample_incidents)
        clf = HierarchicalClassifier(_make_tm(taxonomy_file), model_name="rf")
        clf.fit(df)
        top = clf.top_features(level=1, n=3)
        assert top == {}
