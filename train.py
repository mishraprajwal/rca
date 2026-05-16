#!/usr/bin/env python3
"""Convenience training entry point."""

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)

from src.trainer import ModelTrainer

ModelTrainer().run_training_pipeline()