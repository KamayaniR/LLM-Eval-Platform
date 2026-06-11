import os
from google.cloud import aiplatform
from dotenv import load_dotenv

load_dotenv()

PROJECT_ID = os.getenv("PROJECT_ID")
REGION = os.getenv("REGION", "us-central1")
GCS_BUCKET = os.getenv("GCS_BUCKET")
WANDB_API_KEY = os.getenv("WANDB_API_KEY")
IMAGE_URI = f"{REGION}-docker.pkg.dev/{PROJECT_ID}/llm-eval/trainer:v1"

def launch_training_job(
    epochs=3,
    batch_size=8,
    max_samples=None,
    job_name="reward-model-v1"
):
    aiplatform.init(
        project=PROJECT_ID,
        location=REGION,
        staging_bucket=f"gs://{GCS_BUCKET}"
    )

    job = aiplatform.CustomContainerTrainingJob(
        display_name=job_name,
        container_uri=IMAGE_URI,
    )

    args = [
        f"--epochs={epochs}",
        f"--batch-size={batch_size}",
        f"--save-dir=gs://{GCS_BUCKET}/checkpoints",
        "--use-wandb",
    ]
    if max_samples:
        args.append(f"--max-samples={max_samples}")

    job.run(
        args=args,
        replica_count=1,
        machine_type="n1-standard-8",
        accelerator_type="NVIDIA_TESLA_T4",
        accelerator_count=1,
        environment_variables={
            "GCS_BUCKET": GCS_BUCKET,
            "PROJECT_ID": PROJECT_ID,
            "WANDB_API_KEY": WANDB_API_KEY,
        },
        base_output_dir=f"gs://{GCS_BUCKET}/checkpoints",
        sync=False,
    )
    print(f"Training job launched: {job_name}")
    print(f"Monitor at: https://console.cloud.google.com/vertex-ai/training/custom-jobs?project={PROJECT_ID}")

if __name__ == "__main__":
    launch_training_job(epochs=3, batch_size=8)
