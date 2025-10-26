# flights_recommender.py
import pandas as pd
import logging
from llm_explainer import LLMExplainer
from utils import normalize_text, compute_shortest_path, eco_score

# -----------------------------
# Logging configuration
# -----------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

# -----------------------------
# Flight Recommender
# -----------------------------
class FlightRecommender:
    def __init__(self, flight_data: pd.DataFrame):
        self.df = flight_data.copy()
        self.llm = LLMExplainer()  # ✅ Initialize optimized LLMExplainer
        logger.info("FlightRecommender initialized.")

    def recommend(self, origin: str, dest: str, preference: str = "eco", top_k: int = 5):
        """
        Recommend top_k flights between origin and dest
        preference: "eco", "fastest", or "cheapest"
        """
        logger.info(f"Generating recommendations from {origin} to {dest} based on preference: {preference}")

        # Filter flights
        flights = self.df[(self.df["ORIGIN"] == origin) & (self.df["DEST"] == dest)]
        if flights.empty:
            logger.warning("No flights found for the given route.")
            return []

        # Rank flights based on preference
        if preference == "fastest":
            flights = flights.nsmallest(top_k, "ELAPSED_TIME")
        elif preference == "cheapest":
            flights = flights.nsmallest(top_k, "PRICE") if "PRICE" in flights.columns else flights
        elif preference == "eco":
            flights["eco_score"] = flights.apply(lambda x: eco_score(x["DISTANCE"], x["ESTIMATED_CO2_KG"]), axis=1)
            flights = flights.nlargest(top_k, "eco_score")
        else:
            logger.info("Unknown preference, defaulting to balanced ranking.")
            flights = flights.head(top_k)

        recommendations = flights.to_dict(orient="records")

        # Generate explanation using LLM
        explanation = self.llm.explain_flights(recommendations, user_preference=preference)

        logger.info("Flight recommendations and explanation generated.")
        return {
            "flights": recommendations,
            "explanation": explanation
        }

# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    # Load preprocessed feature-engineered dataset (25% sampled)
    flight_data = pd.read_csv("data/flights_feature_engineered_25.csv")

    recommender = FlightRecommender(flight_data)
    result = recommender.recommend(origin="JFK", dest="LAX", preference="eco", top_k=3)

    print("\nTop Flights:")
    for f in result["flights"]:
        print(f"Flight {f['FL_NUMBER']} | {f['ORIGIN_CITY']} -> {f['DEST_CITY']} | CO2: {f['ESTIMATED_CO2_KG']} kg | Delay: {f['ARR_DELAY']} min")

    print("\nLLM Explanation:\n")
    print(result["explanation"])
