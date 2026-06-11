import os
import json
import threading
from google.cloud import pubsub_v1
from dotenv import load_dotenv

load_dotenv()

PROJECT_ID = os.getenv("PROJECT_ID")
SUBSCRIPTION = f"projects/{PROJECT_ID}/subscriptions/eval-jobs-sub"

processed_jobs = set()
scorer = None
runner = None
logger = None

def init_components():
    global scorer, runner, logger
    from runners.gemini import GeminiRunner
    from pipeline.scorer import Scorer
    from pipeline.logger import Logger
    print("Initializing components once...")
    runner = GeminiRunner()
    scorer = Scorer()
    logger = Logger()
    print("Components ready.")

def process_message(message):
    try:
        data = json.loads(message.data.decode())
        job_id = data["job_id"]

        if job_id in processed_jobs:
            message.ack()
            return

        prompts = data["prompts"]
        model = data["model"]
        print(f"Processing job {job_id} with {len(prompts)} prompts")

        for prompt in prompts:
            response = runner.run(prompt, model=model)
            scores = scorer.score(prompt, response)
            logger.log_result(job_id, prompt, response, scores)
            print(f"Scored: {scores}")

        processed_jobs.add(job_id)
        message.ack()
        print(f"Job {job_id} completed and acked")

    except Exception as e:
        print(f"Worker error: {e}")
        message.ack()

def start_worker():
    init_components()
    subscriber = pubsub_v1.SubscriberClient()
    print(f"Starting worker, listening on {SUBSCRIPTION}")
    future = subscriber.subscribe(SUBSCRIPTION, callback=process_message)
    try:
        future.result()
    except Exception as e:
        print(f"Worker stopped: {e}")
        future.cancel()

def start_worker_thread():
    t = threading.Thread(target=start_worker, daemon=True)
    t.start()
    print("Worker thread started.")
