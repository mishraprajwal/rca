"""
Text preprocessor for IT incident tickets.

Key design choices vs. generic NLP preprocessing
-------------------------------------------------
- Do NOT strip numbers:  "CPU 100%", "HTTP 500", "error 0x8007" all carry signal
- Preserve technical tokens:  IP addresses, version strings, hex codes, error codes
- Split camelCase / PascalCase:  "DatabaseTimeout" → "database timeout"
- Expand common IT abbreviations:  "DB" → "database", "OS" → "operating system"
- Keep domain-meaningful short tokens (3-char min is still applied after expansion)
"""

import re
import logging
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from typing import Dict, List, Any
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Domain-specific abbreviation expansion
# ---------------------------------------------------------------------------
_IT_EXPANSIONS: Dict[str, str] = {
    r"\bapi\b": "application programming interface",
    r"\bdb\b": "database",
    r"\bos\b": "operating system",
    r"\bcpu\b": "cpu processor",
    r"\bgpu\b": "gpu graphics",
    r"\bram\b": "ram memory",
    r"\bssd\b": "ssd storage",
    r"\bhdd\b": "hdd disk",
    r"\bvm\b": "virtual machine",
    r"\bvpn\b": "vpn network",
    r"\bdns\b": "dns network",
    r"\bdhcp\b": "dhcp network",
    r"\bnfs\b": "nfs filesystem",
    r"\bssh\b": "ssh connection",
    r"\bhttp\b": "http web",
    r"\bhttps\b": "https web",
    r"\bui\b": "user interface",
    r"\bcli\b": "command line interface",
    r"\bci\b": "continuous integration",
    r"\bcd\b": "continuous deployment",
}

# Preserve these patterns before lowercasing / symbol removal
_PRESERVE_PATTERNS: List[str] = [
    r"\b\d{1,3}(?:\.\d{1,3}){3}\b",          # IP addresses  192.168.1.1
    r"\b(?:v|version)?\s*\d+\.\d+[\.\d]*\b",  # versions  v2.3.1
    r"\b(?:http|ftp)s?://\S+",                 # URLs
    r"\b0x[0-9a-fA-F]+\b",                    # hex codes  0x8007
    r"\b\d{3,5}\b",                            # standalone numbers (error codes)
]


class TextPreprocessor:
    """Text preprocessor tuned for IT incident management data."""

    def __init__(self):
        for resource, path in [
            ("punkt", "tokenizers/punkt"),
            ("punkt_tab", "tokenizers/punkt_tab"),
            ("stopwords", "corpora/stopwords"),
            ("wordnet", "corpora/wordnet"),
        ]:
            try:
                nltk.data.find(path)
            except LookupError:
                nltk.download(resource, quiet=True)

        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words("english"))

        # Keep general incident words that carry category signal
        # ('error', 'failed' removed from the default stop-list because they
        # help distinguish hardware vs. human-error incidents)
        _keep = {"error", "failed", "failure", "crash", "timeout", "down"}
        self.stop_words -= _keep

        # Generic filler words not useful for classification
        self.stop_words.update({"incident", "issue", "ticket", "please", "kindly"})

        # Pre-compile abbreviation patterns (lowercase)
        self._abbrev_re: List[tuple] = [
            (re.compile(pat, re.IGNORECASE), repl)
            for pat, repl in _IT_EXPANSIONS.items()
        ]

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def clean_text(self, text: str) -> str:
        """
        Full cleaning pipeline for a single raw incident description.

        Steps
        -----
        1. Guard against non-string input
        2. Split camelCase / PascalCase tokens
        3. Lowercase
        4. Expand IT abbreviations
        5. Normalise special characters (keep alphanumerics + spaces)
        6. Collapse whitespace
        """
        if not isinstance(text, str) or not text.strip():
            return ""

        # Split camelCase / PascalCase:  "DatabaseTimeout" → "Database Timeout"
        text = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text)
        text = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", text)

        text = text.lower()

        # Expand abbreviations
        for pattern, replacement in self._abbrev_re:
            text = pattern.sub(replacement, text)

        # Replace common separators with space (keep alphanumerics + dots for
        # version numbers and dots in IPs — the TF-IDF char n-grams handle them)
        text = re.sub(r"[^a-z0-9.\s]", " ", text)

        # Collapse runs of dots not surrounded by digits (avoids artefacts)
        text = re.sub(r"(?<!\d)\.(?!\d)", " ", text)

        # Collapse whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def tokenize_and_lemmatize(self, text: str) -> List[str]:
        """Tokenize, remove stop words, and lemmatize."""
        tokens = nltk.word_tokenize(text)
        tokens = [
            self.lemmatizer.lemmatize(tok)
            for tok in tokens
            if tok not in self.stop_words and len(tok) > 2
        ]
        return tokens

    def preprocess_incident_data(
        self, df: pd.DataFrame, text_column: str = "description"
    ) -> pd.DataFrame:
        """
        Apply the full preprocessing pipeline to a DataFrame.

        Adds columns
        ------------
        clean_text     : cleaned raw text (before tokenisation)
        tokens         : list of lemmatised tokens
        processed_text : space-joined tokens (used by model)
        """
        df_processed = df.copy()
        df_processed["clean_text"] = df_processed[text_column].apply(self.clean_text)
        df_processed["tokens"] = df_processed["clean_text"].apply(
            self.tokenize_and_lemmatize
        )
        df_processed["processed_text"] = df_processed["tokens"].apply(
            lambda toks: " ".join(toks)
        )
        logger.info("Preprocessed %d incident tickets", len(df_processed))
        return df_processed

    def get_vocabulary_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Return basic vocabulary statistics from a preprocessed DataFrame."""
        all_tokens = [tok for toks in df["tokens"] for tok in toks]
        word_freq = pd.Series(all_tokens).value_counts()
        return {
            "vocab_size": len(set(all_tokens)),
            "total_tokens": len(all_tokens),
            "avg_tokens_per_doc": len(all_tokens) / max(len(df), 1),
            "most_common_words": word_freq.head(20).to_dict(),
        }