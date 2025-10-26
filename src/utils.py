# utils.py
import re
import math

def normalize_text(text: str) -> str:
    """
    Simple text normalization:
    - lowercase
    - remove extra whitespace
    - remove special characters except punctuation
    """
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)  # collapse multiple spaces
    text = re.sub(r'[^a-z0-9,.!? ]', '', text)  # remove special chars
    return text.strip()

def eco_score(distance_miles: float, co2_kg: float) -> float:
    """
    Compute a simple eco-friendliness score for a flight.
    Higher score = more eco-friendly
    """
    if co2_kg == 0:
        return 0
    return max(0, distance_miles / co2_kg)  # simple efficiency metric

def compute_shortest_path(flights: list, origin: str, dest: str) -> list:
    """
    Placeholder for shortest-path computation.
    flights: list of flight dicts with ORIGIN, DEST, ELAPSED_TIME
    Returns: list of flight dicts representing shortest path
    """
    # Simple greedy approach (for demo purposes)
    path = []
    current = origin
    remaining = flights.copy()
    while current != dest and remaining:
        # choose next flight from current with smallest ELAPSED_TIME
        candidates = [f for f in remaining if f["ORIGIN"] == current]
        if not candidates:
            break
        next_flight = min(candidates, key=lambda x: x.get("ELAPSED_TIME", float("inf")))
        path.append(next_flight)
        current = next_flight["DEST"]
        remaining.remove(next_flight)
    return path
