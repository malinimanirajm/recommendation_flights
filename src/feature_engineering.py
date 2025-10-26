"""
feature_engineering.py
Memory-efficient advanced feature engineering that:
- streams the original CSV and writes a 25% sampled CSV
- creates basic + advanced features
- uses 'category' dtype for high-cardinality columns to avoid MemoryError
Output: data/flights_feature_engineered_25.csv
"""

import pandas as pd
import numpy as np
import holidays
from pathlib import Path

INPUT_CSV = Path("data/flights_sample_3m.csv")                          # large raw dataset (place in data/)
SAMPLED_CSV = Path("data/flights_sample.csv")              # intermediate 25% sample
OUTPUT_CSV = Path("data/flights_feature_engineered_25.csv")   # final engineered dataset
CHUNKSIZE = 100_000
SAMPLE_FRAC = 0.25
RANDOM_STATE = 42

def stream_sample_csv(input_path: Path, output_sample_path: Path, frac=SAMPLE_FRAC,
                      chunksize=CHUNKSIZE, random_state=RANDOM_STATE):
    """Stream the big CSV in chunks and write sampled rows directly to disk."""
    print(f"Streaming sample from {input_path} -> writing to {output_sample_path}")
    first = True
    for chunk in pd.read_csv(input_path, chunksize=chunksize):
        sampled = chunk.sample(frac=frac, random_state=random_state)
        sampled.to_csv(output_sample_path, mode="a", index=False, header=first)
        first = False
    print("Sampling complete.")

def bucketize_time(hour):
    try:
        hour = int(hour)
    except Exception:
        return "Unknown"
    if 5 <= hour < 12: return "Morning"
    if 12 <= hour < 17: return "Afternoon"
    if 17 <= hour < 21: return "Evening"
    return "Night"

def delay_category(delay):
    try:
        d = float(delay)
    except Exception:
        return "Unknown"
    if d <= 0: return "Early/On-time"
    if d <= 15: return "Slight Delay"
    if d <= 60: return "Moderate Delay"
    return "Severe Delay"

def reduce_cardinality(df, col, min_freq=0.001):
    """
    Group very rare categories into 'OTHER' using a frequency threshold.
    min_freq is fraction (e.g. 0.001 = 0.1%).
    """
    freqs = df[col].value_counts(normalize=True)
    small = freqs[freqs < min_freq].index
    if len(small) > 0:
        df[col] = df[col].replace(small, "OTHER")
    return df

def run_feature_engineering():
    # Step A: ensure data folder and input exists
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV}")

    # Step 1: stream-sample 25%
    if not SAMPLED_CSV.exists():
        stream_sample_csv(INPUT_CSV, SAMPLED_CSV)
    else:
        print(f"Found existing sample: {SAMPLED_CSV}")

    # Step 2: load sampled csv (this should fit in memory)
    df = pd.read_csv(SAMPLED_CSV)
    print(f"Loaded sample shape: {df.shape}")

    # Parse FL_DATE
    df['FL_DATE'] = pd.to_datetime(df['FL_DATE'], errors='coerce')

    # Basic date/time features
    df['DAY_OF_WEEK'] = df['FL_DATE'].dt.dayofweek
    df['WEEK_OF_YEAR'] = df['FL_DATE'].dt.isocalendar().week.astype('Int64')
    df['MONTH'] = df['FL_DATE'].dt.month
    df['IS_WEEKEND'] = df['DAY_OF_WEEK'].isin([5,6]).astype(int)

    # Holidays (USA by default; adjust if your data is different)
    years = sorted(df['FL_DATE'].dt.year.dropna().unique().astype(int).tolist())
    us_holidays = holidays.US(years=years)
    df['IS_HOLIDAY'] = df['FL_DATE'].isin(us_holidays).astype(int)

    # Time features
    df['DEP_HOUR'] = (df['CRS_DEP_TIME'].fillna(0)//100).astype(int)
    df['ARR_HOUR'] = (df['CRS_ARR_TIME'].fillna(0)//100).astype(int)
    df['DEP_TIME_BUCKET'] = df['DEP_HOUR'].apply(bucketize_time)
    df['ARR_TIME_BUCKET'] = df['ARR_HOUR'].apply(bucketize_time)

    # Delay categories
    df['DEP_DELAY_CATEGORY'] = df['DEP_DELAY'].fillna(0).apply(delay_category)
    df['ARR_DELAY_CATEGORY'] = df['ARR_DELAY'].fillna(0).apply(delay_category)

    # Flight speed & estimated CO2
    df['FLIGHT_SPEED_MPH'] = (df['DISTANCE'] / (df['AIR_TIME']/60)).replace([np.inf, -np.inf], np.nan)
    # ESTIMATED_CO2_KG: approximate 115 g CO2 per passenger-km -> convert miles->km (1 mile = 1.60934 km) -> grams
    df['ESTIMATED_CO2_KG'] = df['DISTANCE'].fillna(0) * 1.60934 * 0.115

    # Airline performance
    df['AIRLINE_AVG_ARR_DELAY'] = df.groupby('AIRLINE_CODE')['ARR_DELAY'].transform('mean')

    # Distance bucket
    df['DISTANCE_BUCKET'] = pd.cut(df['DISTANCE'].fillna(0),
                                   bins=[-1, 500, 1500, 3000, np.inf],
                                   labels=['Short','Medium','Long','Ultra-Long'])

    # Advanced features
    df['AIRLINE_7DAY_AVG_DELAY'] = df.groupby('AIRLINE_CODE')['ARR_DELAY'].transform(lambda x: x.rolling(7, min_periods=1).mean())
    df['DAILY_ORIGIN_FLIGHTS'] = df.groupby(['ORIGIN','FL_DATE'])['FL_DATE'].transform('count')
    df['DAILY_DEST_FLIGHTS'] = df.groupby(['DEST','FL_DATE'])['FL_DATE'].transform('count')
    df['IS_SUMMER'] = df['MONTH'].isin([6,7,8]).astype(int)
    df['IS_WINTER'] = df['MONTH'].isin([12,1,2]).astype(int)
    df['IS_PEAK_SEASON'] = df['MONTH'].isin([6,7,12]).astype(int)
    df['DISTANCE_X_DELAY'] = df['DISTANCE'].fillna(0) * df['ARR_DELAY'].fillna(0)
    df['SPEED_X_CONGESTION'] = df['FLIGHT_SPEED_MPH'] / (1 + df['DAILY_ORIGIN_FLIGHTS'].fillna(0))
    df['HIGH_DELAY_RISK'] = (((df['AIRLINE_AVG_ARR_DELAY'] > 30).fillna(False)) |
                            (df['DAILY_ORIGIN_FLIGHTS'] > df['DAILY_ORIGIN_FLIGHTS'].quantile(0.9))).astype(int)

    # Efficient categorical encoding: convert to 'category' dtype
    categorical_cols = ['AIRLINE_CODE','ORIGIN','DEST','CANCELLATION_CODE',
                        'DEP_TIME_BUCKET','ARR_TIME_BUCKET','DEP_DELAY_CATEGORY','ARR_DELAY_CATEGORY','DISTANCE_BUCKET']
    # Reduce cardinality for ORIGIN/DEST if extremely high
    for col in ['ORIGIN','DEST']:
        if col in df.columns:
            df = reduce_cardinality(df, col, min_freq=0.001)  # e.g. 0.1% threshold

    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype('category')

    # Save engineered dataset
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved engineered dataset to {OUTPUT_CSV} with shape {df.shape}")

if __name__ == "__main__":
    run_feature_engineering()
