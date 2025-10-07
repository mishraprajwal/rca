#!/usr/bin/env python3

import sys
from src.evaluator import Evaluator
from src.preprocessor import TextPreprocessor

def main():
    if len(sys.argv) < 2:
        print("Usage: python predict.py 'incident description'")
        sys.exit(1)

    description = sys.argv[1]

    # Load evaluator (which loads models)
    evaluator = Evaluator()
    evaluator.taxonomy_manager.load_taxonomy('data/taxonomy.json')
    evaluator.load_models()

    # Preprocess text
    preprocessor = TextPreprocessor()
    clean_text = preprocessor.clean_text(description)
    processed_text = ' '.join(preprocessor.tokenize_and_lemmatize(clean_text))

    # Predict
    prediction = evaluator.classifier.predict_hierarchy(processed_text)

    print(f"Incident: {description}")
    print(f"Processed: {processed_text}")
    print(f"Predicted categories: {prediction}")

if __name__ == '__main__':
    main()