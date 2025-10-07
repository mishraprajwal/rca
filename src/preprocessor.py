import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from typing import List, Dict, Any
import pandas as pd

class TextPreprocessor:
    """Class for preprocessing incident ticket text data."""

    def __init__(self):
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')

        try:
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            nltk.download('punkt_tab')

        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')

        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet')

        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

        # Add domain-specific stop words
        self.stop_words.update(['incident', 'issue', 'problem', 'error', 'failed', 'unable'])

    def clean_text(self, text: str) -> str:
        """Clean and normalize text data.

        Args:
            text: Raw text to clean

        Returns:
            Cleaned text
        """
        if not isinstance(text, str):
            return ""

        # Convert to lowercase
        text = text.lower()

        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def tokenize_and_lemmatize(self, text: str) -> List[str]:
        """Tokenize and lemmatize text.

        Args:
            text: Cleaned text

        Returns:
            List of lemmatized tokens
        """
        tokens = nltk.word_tokenize(text)

        # Remove stop words and lemmatize
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens
                 if token not in self.stop_words and len(token) > 2]

        return tokens

    def preprocess_incident_data(self, df: pd.DataFrame,
                               text_column: str = 'description') -> pd.DataFrame:
        """Preprocess the entire incident dataset.

        Args:
            df: DataFrame with incident data
            text_column: Name of the column containing text

        Returns:
            DataFrame with additional processed columns
        """
        df_processed = df.copy()

        # Clean text
        df_processed['clean_text'] = df_processed[text_column].apply(self.clean_text)

        # Tokenize and lemmatize
        df_processed['tokens'] = df_processed['clean_text'].apply(self.tokenize_and_lemmatize)

        # Create text for modeling (joined tokens)
        df_processed['processed_text'] = df_processed['tokens'].apply(lambda x: ' '.join(x))

        print(f"Preprocessed {len(df_processed)} incident tickets")
        return df_processed

    def get_vocabulary_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get vocabulary statistics from processed data.

        Args:
            df: DataFrame with processed data

        Returns:
            Dictionary with vocabulary statistics
        """
        all_tokens = [token for tokens in df['tokens'] for token in tokens]

        vocab = set(all_tokens)
        word_freq = pd.Series(all_tokens).value_counts()

        stats = {
            "vocab_size": len(vocab),
            "total_tokens": len(all_tokens),
            "avg_tokens_per_doc": len(all_tokens) / len(df),
            "most_common_words": word_freq.head(20).to_dict()
        }

        return stats