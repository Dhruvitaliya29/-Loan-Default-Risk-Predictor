"""
data_loader.py
Loads the Home Credit application_train.csv file and performs initial checks.
"""

import pandas as pd
import numpy as np
import os

def load_data(filepath: str) -> pd.DataFrame:
    """Load CSV file into a pandas DataFrame."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    df = pd.read_csv(filepath)
    print(f"✅ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

def initial_inspection(df: pd.DataFrame) -> None:
    """Print basic info, missing values summary, and class balance."""
    print("\n--- Data Types ---")
    print(df.dtypes.value_counts())
    
    print("\n--- Missing Values (top 15) ---")
    missing = df.isnull().sum().sort_values(ascending=False)
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame({
        'Count': missing,
        'Percent': missing_pct
    })
    print(missing_df.head(15))
    
    print("\n--- Target Distribution ---")
    target = 'TARGET'
    if target in df.columns:
        counts = df[target].value_counts()
        print(counts)
        print(f"Imbalance ratio: {counts[0]/counts[1]:.2f}:1 (non-default : default)")
    else:
        print("TARGET column not found!")

if __name__ == "__main__":
    # Example usage
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(BASE_DIR, "data", "raw", "application_train.csv")
    
    df = load_data(data_path)
    initial_inspection(df)