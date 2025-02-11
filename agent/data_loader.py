import os
import logging
import pandas as pd
from typing import List

def load_json_file(file_path: str) -> pd.DataFrame:
    """
    Load a JSON file into a pandas DataFrame.
    """
    if not os.path.exists(file_path):
        logging.error(f"File not found: {file_path}")
        return pd.DataFrame()
    
    try:
        df = pd.read_json(file_path)
        logging.info(f"Loaded {len(df)} records from {file_path}")
        return df
    except Exception as e:
        logging.exception(f"Error loading file {file_path}: {e}")
        return pd.DataFrame()

def combine_text_columns(df: pd.DataFrame, expected_columns: List[str], new_col_name: str) -> pd.DataFrame:
    """
    Combine specified text columns into a single column.
    """
    present_columns = [col for col in expected_columns if col in df.columns]
    if not present_columns:
        present_columns = df.select_dtypes(include="object").columns.tolist()
        logging.warning(f"No expected columns found for '{new_col_name}'; using all object columns: {present_columns}")
    
    df[new_col_name] = df[present_columns].fillna("").agg(" ".join, axis=1)
    return df
