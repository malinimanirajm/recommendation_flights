

# ✈️ Flight Recommendation System with LLM Explanations

A **smart flight recommendation system** that suggests the best flights based on:

* ✅ Shortest travel time
* ✅ Least delays
* ✅ Eco-friendly options (low CO₂ emissions)
* ✅ Dynamic natural language explanations using LLMs

This project integrates **traditional data science pipelines** with **Large Language Models (LLMs)** to make flight recommendations **explainable, user-friendly, and intelligent**.

---

## 📂 Project Structure

```
.
├── feature_engineering.py   # Preprocess raw flight dataset (25% sampled)
├── flights_recommender.py   # Flight ranking logic (time, delays, CO2)
├── llm_explainer.py         # Hugging Face LLM explanations
├── utils.py                 # Shared utility functions
├── logging_config.py        # Central logging setup
├── requirements.txt         # Dependencies
├── .env                     # Store your API tokens here (not committed)
└── README.md                # This file
```

---

## ⚙️ Features

* 🛠 **Feature Engineering**:

  * Delay categorization
  * Time bucketization (hour of day, weekday/weekend)
  * Flight speed & CO₂ estimation
  * Distance bucketing

* 🔍 **Recommendation Engine**:

  * Rank flights based on weighted criteria (time, delay, CO₂)
  * Flexible scoring logic

* 🧠 **LLM Explanations**:

  * Generates natural-language justifications for chosen flights
  * Runs with Hugging Face models (e.g., `distilgpt2`)
  * Compatible with **MPS acceleration** on Mac (Apple Silicon)

* 📊 **Logging & Debugging**:

  * Tracks model loading time, explanation generation, and pipeline steps

---

## 🚀 Quick Start

### 1️⃣ Clone the repo

```bash
git clone <your-repo-url>
cd flight-recommendation
```

### 2️⃣ Create and activate virtual environment

```bash
python3 -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Set environment variables

Create a `.env` file in the root:

```
HF_API_TOKEN=your_huggingface_token
```

Get a free Hugging Face token here 👉 [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

### 5️⃣ Run the pipeline

**Feature engineering**

```bash
python feature_engineering.py
```

**Get flight recommendations**

```bash
python flights_recommender.py
```

**Generate LLM explanations**

```bash
python llm_explainer.py
```

---

## 🖥 Device Acceleration

* ✅ **Mac (Apple Silicon)** → uses **MPS (Metal Performance Shaders)** automatically
* ✅ **CPU fallback** if MPS is unavailable
* 🚀 Future-ready for **CUDA (NVIDIA GPUs)** if running on Linux/Windows

---

## 📚 Tech Stack

* **Python** 🐍
* **Pandas, NumPy** → Feature engineering
* **Scikit-learn** → Preprocessing utilities
* **Hugging Face Transformers** → LLM-based explanations
* **Dotenv** → Secure API key management
* **Logging** → Monitoring and debugging

---

## ✅ Example Output

**Input:** Flight dataset (25% sample)
**Output (ranked flights):**

```
Top recommended flights:
- Flight 123 from JFK to LAX, delay 5 mins, CO₂ 120kg
- Flight 456 from SFO to ORD, delay 2 mins, CO₂ 90kg
```

**LLM Explanation:**

```
The best choice is Flight 456 because it balances eco-friendliness with a very low delay risk, 
while Flight 123 is also good but has slightly higher emissions.
```

---

## 🔮 Future Enhancements

* 🌍 Integration with live flight APIs
* 📈 Full dataset training & scalability with Spark/Dask
* 🧩 Multi-modal AI (combine text + structured data)
* 🤝 LLMOps integration for monitoring and feedback loops




[1] DEFINE ROLE & CONTEXT
   -> Specify expertise and purpose
   -> Include environment, libraries, constraints
   -> Example: “Act as senior ML engineer, AWS/Databricks environment”

[2] PROVIDE DATA & GOALS
   -> Sample datasets, schemas, metrics
   -> Clear objective for output
   -> Example: “Feature engineering for flight delay prediction, optimize F1”

[3] STRUCTURED PROMPT
   -> Specify output format & constraints
   -> Few-shot examples if needed
   -> Example: “Return only Python code, single block, with comments”

[4] MULTI-STEP CHAINING
   -> Break tasks into logical steps
   -> Step outputs feed next prompts
   -> Example: Features -> Code -> Refactor -> Explain -> Summarize

[5] ITERATIVE REFINEMENT
   -> Review output, request optimizations
   -> Handle edge cases, errors, performance
   -> Example: “Refactor code to handle missing data efficiently”

[6] AUTOMATION & INTEGRATION
   -> Connect AI output to pipelines, dashboards, CI/CD
   -> Example: Generate ETL scripts or reports automatically

[7] VALIDATION & AUDIT
   -> Check AI outputs against reality/data
   -> Ensure correctness, safety, regulatory compliance

[8] SAVE & REUSE
   -> Store useful prompts, code snippets, summaries
   -> Build personal AI knowledge base

[9] CONTINUOUS IMPROVEMENT
   -> Adjust prompts for efficiency and clarity
   -> Learn from past interactions for faster, better results

