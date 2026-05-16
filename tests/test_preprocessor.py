"""Tests for TextPreprocessor."""

import pytest
from src.preprocessor import TextPreprocessor


@pytest.fixture(scope="module")
def preprocessor():
    return TextPreprocessor()


class TestCleanText:
    def test_lowercase(self, preprocessor):
        # CPU expands to 'cpu processor' — output must still be lowercase
        result = preprocessor.clean_text("SERVER CPU FAILURE")
        assert result == result.lower()

    def test_preserves_numbers(self, preprocessor):
        # Error codes like 500 are preserved as signal for classification
        result = preprocessor.clean_text("error code 500 occurred")
        assert "500" in result

    def test_removes_special_chars(self, preprocessor):
        result = preprocessor.clean_text("DB@connection#timeout!")
        assert "@" not in result
        assert "#" not in result
        assert "!" not in result

    def test_handles_empty_string(self, preprocessor):
        assert preprocessor.clean_text("") == ""

    def test_handles_non_string(self, preprocessor):
        assert preprocessor.clean_text(None) == ""
        assert preprocessor.clean_text(123) == ""

    def test_strips_extra_whitespace(self, preprocessor):
        result = preprocessor.clean_text("  too   many   spaces  ")
        assert "  " not in result
        assert result == result.strip()


class TestTokenizeAndLemmatize:
    def test_returns_list(self, preprocessor):
        result = preprocessor.tokenize_and_lemmatize("server cpu failure")
        assert isinstance(result, list)

    def test_removes_stopwords(self, preprocessor):
        result = preprocessor.tokenize_and_lemmatize("the server is down")
        assert "the" not in result
        assert "is" not in result

    def test_filters_short_tokens(self, preprocessor):
        result = preprocessor.tokenize_and_lemmatize("a b cpu")
        # tokens of length <= 2 should be excluded
        assert "a" not in result
        assert "b" not in result

    def test_lemmatization(self, preprocessor):
        result = preprocessor.tokenize_and_lemmatize("servers crashing")
        # 'servers' should be lemmatized to 'server'
        assert "server" in result


class TestPreprocessIncidentData:
    def test_adds_clean_text_column(self, preprocessor, sample_incidents):
        df = preprocessor.preprocess_incident_data(sample_incidents)
        assert "clean_text" in df.columns

    def test_adds_tokens_column(self, preprocessor, sample_incidents):
        df = preprocessor.preprocess_incident_data(sample_incidents)
        assert "tokens" in df.columns

    def test_adds_processed_text_column(self, preprocessor, sample_incidents):
        df = preprocessor.preprocess_incident_data(sample_incidents)
        assert "processed_text" in df.columns

    def test_row_count_preserved(self, preprocessor, sample_incidents):
        df = preprocessor.preprocess_incident_data(sample_incidents)
        assert len(df) == len(sample_incidents)

    def test_processed_text_is_string(self, preprocessor, sample_incidents):
        df = preprocessor.preprocess_incident_data(sample_incidents)
        assert all(isinstance(t, str) for t in df["processed_text"])
