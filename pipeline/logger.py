import os
from google.cloud import bigquery
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

PROJECT_ID = os.getenv("PROJECT_ID")

class Logger:
    def __init__(self):
        self.client = bigquery.Client(project=PROJECT_ID)

    def log_result(self, job_id, prompt, response, scores):
        rows = [{
            "job_id": job_id,
            "prompt": prompt,
            "response": response,
            "reward_score": scores.get("reward"),
            "toxicity_score": scores.get("toxicity"),
            "instruction_score": scores.get("instruction"),
            "factuality_score": scores.get("factuality"),
            "composite_score": scores.get("composite"),
            "created_date": datetime.now(timezone.utc).date().isoformat(),
        }]
        self.client.insert_rows_json(
            f"{PROJECT_ID}.llm_eval.eval_results", rows
        )

    def update_job_status(self, job_id, status):
        query = f"""
            UPDATE `{PROJECT_ID}.llm_eval.eval_jobs`
            SET status = '{status}'
            WHERE job_id = '{job_id}'
        """
        self.client.query(query).result()
