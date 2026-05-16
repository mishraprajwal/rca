"""Shared pytest fixtures for the RCA test suite."""

import pytest
import pandas as pd
import os
import json
import tempfile


@pytest.fixture
def sample_incidents():
    """A small DataFrame of labelled incident tickets."""
    return pd.DataFrame(
        {
            "description": [
                "Server CPU usage is at 100% causing application slowdown",
                "Database connection timeout error in production",
                "Network switch failure in data center",
                "Web server returning 500 internal server error",
                "User unable to login due to password issues",
            ],
            "level_1": ["Hardware", "Software", "Hardware", "Software", "Software"],
            "level_2": ["Server", "Application", "Network", "Application", "Application"],
            "level_3": ["CPU", "Database", "Switch", "Web Server", "Authentication"],
        }
    )


@pytest.fixture
def taxonomy_file(tmp_path):
    """Write a minimal taxonomy JSON to a temp file and return its path."""
    taxonomy = {
        "levels": ["Level_1", "Level_2", "Level_3"],
        "taxonomy": {
            "Hardware": {
                "description": "Hardware issues",
                "subcategories": {
                    "Server": {
                        "description": "Server hardware",
                        "subcategories": {
                            "CPU": {"description": "CPU failures"},
                            "Memory": {"description": "RAM problems"},
                        },
                    },
                    "Network": {
                        "description": "Network infrastructure",
                        "subcategories": {
                            "Switch": {"description": "Switch problems"},
                            "Router": {"description": "Router failures"},
                        },
                    },
                },
            },
            "Software": {
                "description": "Software issues",
                "subcategories": {
                    "Application": {
                        "description": "Application problems",
                        "subcategories": {
                            "Database": {"description": "DB issues"},
                            "Web Server": {"description": "Web server errors"},
                            "Authentication": {"description": "Auth problems"},
                        },
                    },
                },
            },
        },
    }
    path = tmp_path / "taxonomy.json"
    path.write_text(json.dumps(taxonomy))
    return str(path)
