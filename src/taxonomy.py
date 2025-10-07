from typing import Dict, List, Set, Tuple, Any
import json
import os
from collections import defaultdict

class TaxonomyManager:
    """Class for managing hierarchical taxonomy structure."""

    def __init__(self, taxonomy_file: str = None):
        self.taxonomy = {}
        self.levels = []
        self.reverse_taxonomy = defaultdict(set)
        self.level_hierarchy = {}

        if taxonomy_file:
            self.load_taxonomy(taxonomy_file)

    def load_taxonomy(self, filepath: str) -> None:
        """Load taxonomy from a JSON file.

        Args:
            filepath: Path to the taxonomy JSON file
        """
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                data = json.load(f)

            self.taxonomy = data.get('taxonomy', {})
            self.levels = data.get('levels', [])
            self._build_reverse_taxonomy()
            print(f"Loaded taxonomy with {len(self.levels)} levels")
        else:
            print(f"Taxonomy file {filepath} not found. Using empty taxonomy.")

    def save_taxonomy(self, filepath: str) -> None:
        """Save taxonomy to a JSON file.

        Args:
            filepath: Path to save the taxonomy
        """
        data = {
            'taxonomy': self.taxonomy,
            'levels': self.levels
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"Saved taxonomy to {filepath}")

    def add_category(self, path: List[str], description: str = "") -> None:
        """Add a category to the taxonomy.

        Args:
            path: Hierarchical path to the category (e.g., ['Hardware', 'Network', 'Connectivity'])
            description: Optional description of the category
        """
        if len(path) == 0:
            return

        current_level = self.taxonomy
        for i, category in enumerate(path):
            if category not in current_level:
                current_level[category] = {'description': description if i == len(path) - 1 else '',
                                         'subcategories': {}}
            current_level = current_level[category]['subcategories']

        # Update levels
        if len(path) > len(self.levels):
            self.levels = [f"Level_{i+1}" for i in range(len(path))]

        self._build_reverse_taxonomy()

    def get_all_paths(self) -> List[List[str]]:
        """Get all hierarchical paths in the taxonomy.

        Returns:
            List of all category paths
        """
        paths = []

        def traverse(current_path, node):
            if not node.get('subcategories'):
                paths.append(current_path)
                return

            for category, data in node['subcategories'].items():
                traverse(current_path + [category], data)

        traverse([], {'subcategories': self.taxonomy})
        return paths

    def get_categories_at_level(self, level: int) -> List[str]:
        """Get all categories at a specific level.

        Args:
            level: The level to get categories for (1-indexed)

        Returns:
            List of category names at the specified level
        """
        categories = []

        def traverse(current_level, node):
            if current_level == level:
                categories.extend(node.get('subcategories', {}).keys())
                return

            for data in node.get('subcategories', {}).values():
                traverse(current_level + 1, data)

        traverse(1, {'subcategories': self.taxonomy})
        return categories

    def get_parent_categories(self, category_path: List[str]) -> List[List[str]]:
        """Get all parent category paths for a given category.

        Args:
            category_path: Path to the category

        Returns:
            List of parent paths
        """
        parents = []
        for i in range(1, len(category_path)):
            parents.append(category_path[:i])
        return parents

    def _build_reverse_taxonomy(self) -> None:
        """Build reverse taxonomy mapping from category to its path."""
        self.reverse_taxonomy = defaultdict(set)

        def traverse(path, node):
            for category, data in node.get('subcategories', {}).items():
                full_path = path + [category]
                self.reverse_taxonomy[category].add(tuple(full_path))
                traverse(full_path, data)

        traverse([], {'subcategories': self.taxonomy})

    def validate_category_path(self, path: List[str]) -> bool:
        """Validate if a category path exists in the taxonomy.

        Args:
            path: Category path to validate

        Returns:
            True if path exists, False otherwise
        """
        current = self.taxonomy
        for category in path:
            if category not in current:
                return False
            current = current[category].get('subcategories', {})
        return True

    def get_taxonomy_stats(self) -> Dict[str, Any]:
        """Get statistics about the taxonomy.

        Returns:
            Dictionary with taxonomy statistics
        """
        all_paths = self.get_all_paths()
        max_depth = max(len(path) for path in all_paths) if all_paths else 0

        stats = {
            "num_levels": len(self.levels),
            "max_depth": max_depth,
            "total_categories": len(all_paths),
            "categories_per_level": [len(self.get_categories_at_level(i+1)) for i in range(max_depth)]
        }

        return stats