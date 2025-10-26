# llm_explainer.py
import os
import logging
import time
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from utils import normalize_text  # your utility functions
from dotenv import load_dotenv
import torch

# -----------------------------
# Load environment variables
# -----------------------------
load_dotenv()  # loads variables from .env in project root
HF_API_TOKEN = os.getenv("HF_API_TOKEN")
if not HF_API_TOKEN:
    raise ValueError("HF_API_TOKEN not found in environment variables. Please add it to your .env file.")

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
# Hugging Face LLM Explainer
# -----------------------------
class LLMExplainer:
    def __init__(self, model_name: str = "distilgpt2"):
        """
        Initialize Hugging Face LLM Explainer
        :param model_name: HF model name (smaller model for faster load)
        """
        # Detect device
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            logger.info("Using Apple MPS (Metal GPU)")
        else:
            self.device = torch.device("cpu")
            logger.info("MPS not available, using CPU")

        # Load model
        start_time = time.time()
        logger.info(f"Loading Hugging Face model {model_name} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=HF_API_TOKEN)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,   # required for MPS
            low_cpu_mem_usage=True,
            use_auth_token=HF_API_TOKEN
        ).to(self.device)

        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if self.device.type != "cpu" else -1
        )
        end_time = time.time()
        logger.info(f"Model loaded in {end_time - start_time:.2f} seconds.")
        logger.info("Model loaded successfully.")

    def explain_flights(self, flights: list, user_preference: str = "eco") -> str:
        """
        Generate explanation for recommended flights.
        """
        if not flights:
            return "No flights available to explain."

        prompt = self._build_prompt(flights, user_preference)

        try:
            gen_start_time = time.time()
            response = self.generator(prompt, max_new_tokens=50)  # increased for better explanation
            gen_end_time = time.time()
            logger.info(f"Explanation generated in {gen_end_time - gen_start_time:.2f} seconds.")
            explanation = response[0]["generated_text"]
            return explanation.strip()
        except Exception as e:
            logger.error(f"Error generating explanation: {e}")
            return "Failed to generate explanation."

    def _build_prompt(self, flights: list, preference: str) -> str:
        """
        Build prompt for the LLM
        """
        flight_summary = ""
        for f in flights:
            flight_summary += (
                f"Flight {f.get('FL_NUMBER', 'N/A')} from {f.get('ORIGIN', '')} to "
                f"{f.get('DEST', '')}, departs at {f.get('CRS_DEP_TIME', 'N/A')}, "
                f"arrives at {f.get('CRS_ARR_TIME', 'N/A')}, "
                f"ETA CO2 {f.get('ESTIMATED_CO2_KG', 'N/A')} kg, "
                f"arrival delay {f.get('ARR_DELAY', 'N/A')} minutes.\n"
            )
        prompt = (
            f"As a travel advisor AI, explain which flights are best for a user "
            f"preferring '{preference}' considering cost, eco-friendliness, and delays.\n"
            f"Flights:\n{flight_summary}\nProvide a clear and concise explanation."
        )
        return normalize_text(prompt)
