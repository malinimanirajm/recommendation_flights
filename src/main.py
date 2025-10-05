"""
main.py
End-to-end runner:
- ensures feature engineering has been run (if not, instructs)
- runs recommender
- calls Anthropic explainer if configured
"""
import os
from pathlib import Path
from logger_config import logger
from flights_recommender import FlightRecommender
from llm_explainer import LLMExplainer, _has_anthropic
from dotenv import load_dotenv

load_dotenv()

DATA_PATH = os.getenv("DATA_PATH", "data/flights_feature_engineered_25.csv")

def run_example(origin="JFK", dest="LAX", preference="balanced", top_k=3):
    if not Path(DATA_PATH).exists():
        logger.error(f"{DATA_PATH} not found. Run feature_engineering.py first.")
        return
    rec = FlightRecommender(data_path=DATA_PATH)
    top = rec.recommend(origin, dest, preference=preference, top_k=top_k)
    if top.empty:
        logger.info("No recommendations returned.")
        return
    print("Recommendations:\n", top.to_string(index=False))

    # Prepare explainer
    explainer = None
    if _has_anthropic and os.getenv("ANTHROPIC_API_KEY"):
        explainer = LLMExplainer(provider="anthropic")
        explanation = explainer.explain_flights(top.to_dict(orient='records'), user_preference=preference)
        print("\nLLM Explanation:\n", explanation)
    else:
        # fallback
        explanation = "\n".join([f"Flight {r['AIRLINE_CODE']}{r['FL_NUMBER']} score={r['SCORE']:.3f}" for _,r in top.iterrows()])
        print("\nExplanation (fallback):\n", explanation)

if __name__ == "__main__":
    run_example(origin="JFK", dest="LAX", preference="eco", top_k=3)
