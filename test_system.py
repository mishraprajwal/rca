#!/usr/bin/env python3

from src.data_loader import DataLoader
from src.preprocessor import TextPreprocessor
from src.taxonomy import TaxonomyManager

# Test data loading
loader = DataLoader()
df = loader.load_incident_data('incidents.csv')
print(f"Loaded {len(df)} incidents")

# Test preprocessing
preprocessor = TextPreprocessor()
df_processed = preprocessor.preprocess_incident_data(df)
print("Preprocessing completed")

# Test taxonomy
taxonomy = TaxonomyManager()
taxonomy.load_taxonomy('data/taxonomy.json')
print(f"Taxonomy has {len(taxonomy.levels)} levels")

print("All tests passed!")