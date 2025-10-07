from typing import List, Dict, Any, Tuple, Optional
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
import joblib
import os
import numpy as np

class HierarchicalClassifier:
    """Hierarchical classifier for multi-level taxonomy classification."""

    def __init__(self, taxonomy_manager, model_type: str = "tfidf_lr"):
        """
        Args:
            taxonomy_manager: TaxonomyManager instance
            model_type: Type of model ('tfidf_lr' or 'transformer')
        """
        self.taxonomy_manager = taxonomy_manager
        self.model_type = model_type
        self.models = {}  # level -> model
        self.vectorizers = {}  # level -> vectorizer (for tfidf)
        self.tokenizers = {}  # level -> tokenizer (for transformer)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def train_level(self, X: List[str], y: List[str], level: int,
                   model_path: str = "models") -> None:
        """Train model for a specific level.

        Args:
            X: List of text samples
            y: List of labels for this level
            level: The taxonomy level to train
            model_path: Directory to save models
        """
        os.makedirs(model_path, exist_ok=True)

        if self.model_type == "tfidf_lr":
            self._train_tfidf_lr(X, y, level, model_path)
        elif self.model_type == "transformer":
            self._train_transformer(X, y, level, model_path)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def _train_tfidf_lr(self, X: List[str], y: List[str], level: int, model_path: str) -> None:
        """Train TF-IDF + Logistic Regression model."""
        # Vectorize text
        vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        X_vec = vectorizer.fit_transform(X)

        # Train classifier
        classifier = LogisticRegression(random_state=42, max_iter=1000)
        classifier.fit(X_vec, y)

        # Save model and vectorizer
        model_file = os.path.join(model_path, f"level_{level}_model.pkl")
        vectorizer_file = os.path.join(model_path, f"level_{level}_vectorizer.pkl")

        joblib.dump(classifier, model_file)
        joblib.dump(vectorizer, vectorizer_file)

        self.models[level] = classifier
        self.vectorizers[level] = vectorizer

        print(f"Trained TF-IDF+LR model for level {level}")

    def _train_transformer(self, X: List[str], y: List[str], level: int, model_path: str) -> None:
        """Train transformer-based model."""
        # Load pre-trained model
        model_name = "bert-base-uncased"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=len(set(y)),
            problem_type="multi_label_classification"
        )

        # Prepare data (simplified - in practice, use proper data loading)
        # This is a placeholder for actual transformer training
        model.to(self.device)

        # Save model and tokenizer
        model_dir = os.path.join(model_path, f"level_{level}")
        model.save_pretrained(model_dir)
        tokenizer.save_pretrained(model_dir)

        self.models[level] = model
        self.tokenizers[level] = tokenizer

        print(f"Trained transformer model for level {level}")

    def predict_hierarchy(self, text: str) -> List[str]:
        """Predict the full hierarchical path for a given text.

        Args:
            text: Input text to classify

        Returns:
            Predicted hierarchical path
        """
        path = []

        for level in range(1, len(self.taxonomy_manager.levels) + 1):
            if level not in self.models:
                break

            prediction = self._predict_level(text, level)
            if prediction:
                path.append(prediction)
            else:
                break

        return path

    def _predict_level(self, text: str, level: int) -> Optional[str]:
        """Predict category at a specific level.

        Args:
            text: Input text
            level: Taxonomy level

        Returns:
            Predicted category or None
        """
        if self.model_type == "tfidf_lr":
            return self._predict_tfidf_lr(text, level)
        elif self.model_type == "transformer":
            return self._predict_transformer(text, level)
        else:
            return None

    def _predict_tfidf_lr(self, text: str, level: int) -> Optional[str]:
        """Predict using TF-IDF + LR model."""
        if level not in self.vectorizers or level not in self.models:
            return None

        vectorizer = self.vectorizers[level]
        model = self.models[level]

        X_vec = vectorizer.transform([text])
        prediction = model.predict(X_vec)

        # Return the predicted category
        if len(prediction) > 0:
            return prediction[0]
        return None

    def _predict_transformer(self, text: str, level: int) -> Optional[str]:
        """Predict using transformer model."""
        # Placeholder for transformer prediction
        return None

    def save_models(self, model_path: str = "models") -> None:
        """Save all trained models."""
        for level, model in self.models.items():
            if self.model_type == "tfidf_lr":
                model_file = os.path.join(model_path, f"level_{level}_model.pkl")
                joblib.dump(model, model_file)
            elif self.model_type == "transformer":
                model_dir = os.path.join(model_path, f"level_{level}")
                model.save_pretrained(model_dir)

        print(f"Saved models to {model_path}")

    def load_models(self, model_path: str = "models") -> None:
        """Load trained models."""
        for level in range(1, len(self.taxonomy_manager.levels) + 1):
            model_file = os.path.join(model_path, f"level_{level}_model.pkl")
            vectorizer_file = os.path.join(model_path, f"level_{level}_vectorizer.pkl")

            if os.path.exists(model_file) and os.path.exists(vectorizer_file):
                self.models[level] = joblib.load(model_file)
                self.vectorizers[level] = joblib.load(vectorizer_file)
                print(f"Loaded model for level {level}")
            else:
                print(f"Model files not found for level {level}")