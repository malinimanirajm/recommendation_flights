import pandas as pd
from src.feature_engineering import run_feature_engineering

def test_feature_engineering_output_shape():
    df = pd.DataFrame({
        "FL_DATE": ["2023-01-01"],
        "CRS_DEP_TIME": [1230],
        "CRS_ARR_TIME": [1400],
        "DISTANCE": [500]
    })
    processed_df = run_feature_engineering(df)
    
    # Check if important engineered columns exist
    assert "DEP_HOUR" in processed_df.columns
    assert "ARR_HOUR" in processed_df.columns
    assert "DISTANCE_BUCKET" in processed_df.columns
    assert not processed_df.empty
