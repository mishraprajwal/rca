"""Tests for DataLoader."""

import pytest
import pandas as pd
from src.data_loader import DataLoader


class TestDataLoader:
    def test_load_missing_file_raises(self, tmp_path):
        loader = DataLoader(str(tmp_path))
        with pytest.raises(FileNotFoundError):
            loader.load_incident_data("nonexistent.csv")

    def test_save_and_reload(self, tmp_path, sample_incidents):
        loader = DataLoader(str(tmp_path))
        loader.save_processed_data(sample_incidents, "test_output.csv")

        reloaded = pd.read_csv(tmp_path / "processed" / "test_output.csv")
        assert len(reloaded) == len(sample_incidents)
        assert list(reloaded.columns) == list(sample_incidents.columns)

    def test_get_data_info_keys(self, sample_incidents):
        loader = DataLoader()
        info = loader.get_data_info(sample_incidents)
        assert "num_samples" in info
        assert "columns" in info
        assert "missing_values" in info

    def test_get_data_info_counts(self, sample_incidents):
        loader = DataLoader()
        info = loader.get_data_info(sample_incidents)
        assert info["num_samples"] == len(sample_incidents)
