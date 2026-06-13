import os
import json
import uuid
from fastapi import FastAPI
from dotenv import load_dotenv
from google.cloud import pubsub_v1, bigquery
from api.schemas import EvalJobRequest, EvalJobResponse, JobStatusResponse
from datetime import datetime, timezone

load_dotenv()

app = FastAPI(title="LLM Eval Platform")
PROJECT_ID = os.getenv("PROJECT_ID")
TOPIC = f"projects/{PROJECT_ID}/topics/eval-jobs"

bq_client = bigquery.Client(project=PROJECT_ID)
publisher = pubsub_v1.PublisherClient()

@app.on_event("startup")
def startup_event():
    from pipeline.worker import start_worker_thread
    start_worker_thread()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/sample-prompts")
def sample_prompts():
    from data.datasets.truthfulqa import load_truthfulqa
    prompts = load_truthfulqa(max_samples=10)
    return {
        "prompts": [p["question"] for p in prompts]
    }

@app.post("/jobs", response_model=EvalJobResponse)
def submit_job(request: EvalJobRequest):
    job_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()

    rows = [{
        "job_id": job_id,
        "status": "pending",
        "model": request.model,
        "prompt_count": len(request.prompts),
        "created_at": now,
    }]
    bq_client.insert_rows_json(
        f"{PROJECT_ID}.llm_eval.eval_jobs", rows
    )

    message = json.dumps({
        "job_id": job_id,
        "prompts": request.prompts,
        "model": request.model,
        "max_tokens": request.max_tokens,
    }).encode()
    publisher.publish(TOPIC, message)

    return EvalJobResponse(
        job_id=job_id,
        status="pending",
        prompt_count=len(request.prompts)
    )

@app.get("/jobs/{job_id}", response_model=JobStatusResponse)
def get_job_status(job_id: str):
    query = f"""
        SELECT job_id, status, prompt_count, created_at
        FROM `{PROJECT_ID}.llm_eval.eval_jobs`
        WHERE job_id = '{job_id}'
        ORDER BY created_at DESC
        LIMIT 1
    """
    results = list(bq_client.query(query).result())
    if not results:
        return JobStatusResponse(job_id=job_id, status="not_found")
    row = results[0]
    return JobStatusResponse(
        job_id=row.job_id,
        status=row.status,
        prompt_count=row.prompt_count,
        created_at=str(row.created_at)
    )

@app.get("/results/{job_id}")
def get_results(job_id: str):
    query = f"""
        SELECT prompt, response, reward_score, toxicity_score,
               instruction_score, factuality_score, composite_score
        FROM `{PROJECT_ID}.llm_eval.eval_results`
        WHERE job_id = '{job_id}'
    """
    results = list(bq_client.query(query).result())
    return {"job_id": job_id, "results": [dict(row) for row in results]}
