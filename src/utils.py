"""
utils.py
Shared utilities used across the project.
"""
import os
import pandas as pd
from typing import List, Dict
import re
import string

def normalize_text(text: str) -> str:
    """
    Normalize text for LLM prompts:
    - Lowercase
    - Remove extra spaces
    - Remove punctuation
    - Strip leading/trailing spaces
    """
    if not isinstance(text, str):
        return ""
    
    # Lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    
    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text)
    
    # Strip leading/trailing spaces
    text = text.strip()
    
    return text

def load_csv_data(path: str) -> pd.DataFrame:
    """Load CSV with safe parsing."""
    df = pd.read_csv(path)
    if 'FL_DATE' in df.columns:
        df['FL_DATE'] = pd.to_datetime(df['FL_DATE'], errors='coerce')
    return df

def validate_columns(df: pd.DataFrame, required_cols: List[str]) -> None:
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

def df_to_dict_list(df: pd.DataFrame, cols: List[str] = None) -> List[Dict]:
    if cols:
        df = df[cols]
    return df.to_dict(orient="records")

def score_flight(row, preference="balanced"):
    """
    Lightweight scoring function:
    - preference 'balanced' considers elapsed time, arr delay, and estimated CO2
    - returns higher-is-better score
    """
    # safe getters
    elapsed = float(row.get('ELAPSED_TIME') or 0)
    arr_delay = float(row.get('ARR_DELAY') or 0)
    co2 = float(row.get('ESTIMATED_CO2_KG') or 0)
    # small epsilon to avoid division by zero
    eps = 1e-6

    if preference == "balanced":
        score = (1.0/(1+elapsed)) + (1.0/(1+max(0,arr_delay))) + (1.0/(1+co2))
    elif preference == "fastest":
        score = 1.0/(1+elapsed)
    elif preference == "reliable":
        score = 1.0/(1+max(0,arr_delay))
    elif preference == "eco":
        score = 1.0/(1+co2)
    else:
        score = (1.0/(1+elapsed)) + (1.0/(1+max(0,arr_delay))) + (1.0/(1+co2))
    return score
