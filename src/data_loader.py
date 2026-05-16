import logging
import pandas as pd
from typing import List, Dict, Any
import os

logger = logging.getLogger(__name__)

class DataLoader:
    """Class for loading and managing incident ticket data."""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.raw_dir = os.path.join(data_dir, "raw")
        self.processed_dir = os.path.join(data_dir, "processed")

    def load_incident_data(self, filename: str) -> pd.DataFrame:
        """Load incident ticket data from a CSV file.

        Args:
            filename: Name of the CSV file in the raw data directory

        Returns:
            DataFrame containing the incident data
        """
        filepath = os.path.join(self.raw_dir, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Data file {filepath} not found")

        df = pd.read_csv(filepath)
        logger.info("Loaded %d incident tickets from %s", len(df), filename)
        return df

    def save_processed_data(self, df: pd.DataFrame, filename: str) -> None:
        """Save processed data to the processed directory.

        Args:
            df: DataFrame to save
            filename: Name for the output file
        """
        os.makedirs(self.processed_dir, exist_ok=True)
        filepath = os.path.join(self.processed_dir, filename)
        df.to_csv(filepath, index=False)
        logger.info("Saved processed data to %s", filepath)

    def get_data_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get basic information about the dataset.

        Args:
            df: DataFrame to analyze

        Returns:
            Dictionary with dataset statistics
        """
        info = {
            "num_samples": len(df),
            "columns": list(df.columns),
            "missing_values": df.isnull().sum().to_dict(),
            "data_types": df.dtypes.to_dict()
        }
        return info