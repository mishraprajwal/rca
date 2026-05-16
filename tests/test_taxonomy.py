"""Tests for TaxonomyManager."""

import pytest
from src.taxonomy import TaxonomyManager


class TestLoadTaxonomy:
    def test_loads_levels(self, taxonomy_file):
        tm = TaxonomyManager(taxonomy_file)
        assert len(tm.levels) == 3

    def test_loads_taxonomy_keys(self, taxonomy_file):
        tm = TaxonomyManager(taxonomy_file)
        assert "Hardware" in tm.taxonomy
        assert "Software" in tm.taxonomy

    def test_missing_file_does_not_raise(self, tmp_path):
        tm = TaxonomyManager()
        # Should not raise; prints a warning
        tm.load_taxonomy(str(tmp_path / "nonexistent.json"))
        assert tm.taxonomy == {}


class TestGetCategoriesAtLevel:
    def test_level_1_categories(self, taxonomy_file):
        tm = TaxonomyManager(taxonomy_file)
        cats = tm.get_categories_at_level(1)
        assert "Hardware" in cats
        assert "Software" in cats

    def test_level_2_contains_server(self, taxonomy_file):
        tm = TaxonomyManager(taxonomy_file)
        cats = tm.get_categories_at_level(2)
        assert "Server" in cats
        assert "Application" in cats

    def test_level_3_contains_cpu(self, taxonomy_file):
        tm = TaxonomyManager(taxonomy_file)
        cats = tm.get_categories_at_level(3)
        assert "CPU" in cats
        assert "Database" in cats


class TestGetAllPaths:
    def test_returns_list(self, taxonomy_file):
        tm = TaxonomyManager(taxonomy_file)
        paths = tm.get_all_paths()
        assert isinstance(paths, list)
        assert len(paths) > 0

    def test_all_paths_are_lists(self, taxonomy_file):
        tm = TaxonomyManager(taxonomy_file)
        for path in tm.get_all_paths():
            assert isinstance(path, list)


class TestSaveTaxonomy:
    def test_round_trip(self, taxonomy_file, tmp_path):
        tm = TaxonomyManager(taxonomy_file)
        out = str(tmp_path / "saved.json")
        tm.save_taxonomy(out)

        tm2 = TaxonomyManager(out)
        assert tm2.levels == tm.levels
        assert set(tm2.taxonomy.keys()) == set(tm.taxonomy.keys())
