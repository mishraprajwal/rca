#!/usr/bin/env python3
"""Main entry point for the Hierarchical RCA System."""

import argparse
import sys
from .trainer import Trainer
from .evaluator import Evaluator

def main():
    parser = argparse.ArgumentParser(description='Hierarchical Root-Cause Analysis System')
    parser.add_argument('action', choices=['train', 'evaluate'],
                       help='Action to perform: train or evaluate')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to configuration file')
    parser.add_argument('--data-file', type=str, default=None,
                       help='Data file for training/evaluation')
    parser.add_argument('--model-type', choices=['tfidf_lr', 'transformer'],
                       default='tfidf_lr', help='Model type to use')

    args = parser.parse_args()

    if args.action == 'train':
        config = {}
        if args.config:
            # Load config from file if provided
            pass
        if args.data_file:
            config['data_file'] = args.data_file
        if args.model_type:
            config['model_type'] = args.model_type

        trainer = Trainer(config)
        trainer.run_training_pipeline()

    elif args.action == 'evaluate':
        config = {}
        if args.config:
            # Load config from file if provided
            pass
        if args.data_file:
            config['test_data_file'] = args.data_file
        if args.model_type:
            config['model_type'] = args.model_type

        evaluator = Evaluator(config)
        results = evaluator.run_evaluation_pipeline()

        print("\nEvaluation Results:")
        print(f"Hierarchical Accuracy: {results['accuracy']:.4f}")
        print(f"Precision: {results['precision']:.4f}")
        print(f"Recall: {results['recall']:.4f}")
        print(f"F1-Score: {results['f1']:.4f}")

if __name__ == '__main__':
    main()