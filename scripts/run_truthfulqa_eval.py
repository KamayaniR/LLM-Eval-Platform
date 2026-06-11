import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

import requests
from data.datasets.truthfulqa import load_truthfulqa

API_URL = "http://127.0.0.1:8080"

def run_eval(max_samples=50, model="gemini-2.5-flash"):
    prompts = load_truthfulqa(max_samples=max_samples)
    questions = [p["question"] for p in prompts]
    
    print(f"Submitting {len(questions)} TruthfulQA prompts to {model}...")
    
    response = requests.post(f"{API_URL}/jobs", json={
        "prompts": questions,
        "model": model
    })
    
    job_id = response.json()["job_id"]
    print(f"Job ID: {job_id}")
    print("Worker is processing... check results in a few minutes:")
    print(f"curl {API_URL}/results/{job_id}")
    return job_id

if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else "gemini-2.5-flash"
    run_eval(max_samples=50, model=model)
