# tests/test_flights_recommender.py
import pandas as pd
from flights_recommender import recommend_flights

def test_recommend_flights_shortest():
    flights = pd.DataFrame({
        "FL_NUMBER": [1, 2],
        "DISTANCE": [300, 1000],
        "ARR_DELAY": [10, 20],
        "ESTIMATED_CO2_KG": [50, 150]
    })
    result = recommend_flights(flights, top_n=1)
    assert len(result) == 1
    assert result.iloc[0]["FL_NUMBER"] == 1  # shortest + least delayed
