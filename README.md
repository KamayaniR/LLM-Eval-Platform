# LLM Eval Platform 🚧 Work in Progress

A mini RLHF evaluation platform built on GCP. The idea: instead of paying a commercial model to judge every LLM response, train your own reward model on human preference data and use it as the judge. Cheaper, faster, and you own it.

> **Status:** Core pipeline is working end-to-end. Reward model trained (71.8% train accuracy, 61.3% eval accuracy on held-out set). Active development ongoing — see What's next for what's being worked on.

---

## What it does

You submit a list of prompts and a model name. The platform:

1. Sends each prompt to the LLM (Gemini Flash or Pro)
2. Scores each response across four dimensions: toxicity, instruction following, factuality, and overall quality (reward model)
3. Stores results in BigQuery
4. Shows everything in a Looker Studio dashboard

The reward model is a DeBERTa-v3-base model trained from scratch on 160k human preference pairs from Anthropic's HH-RLHF dataset using Bradley-Terry loss. Training ran on Vertex AI on NVIDIA T4 GPUs using PyTorch FSDP.

---

## Architecture

```
User
  │
  ▼
Cloud Run API (FastAPI)
  │
  ├── POST /jobs      → publishes to Pub/Sub, returns job_id
  ├── GET /jobs/{id}  → job status from BigQuery
  └── GET /results/{id} → scores from BigQuery

Pub/Sub (eval-jobs)
  │
  ▼
Worker (background thread in Cloud Run)
  │
  ├── Gemini runner → gets LLM response
  └── Scorer (parallel)
        ├── Reward model  → DeBERTa trained on HH-RLHF
        ├── Toxicity      → Detoxify
        ├── Instruction   → rule-based
        └── Factuality    → Gemini-as-judge

BigQuery (llm_eval)
  ├── eval_jobs
  ├── eval_results
  └── preference_labels

Looker Studio → dashboard
```

---

## Stack

| Component | Technology |
|---|---|
| Reward model | DeBERTa-v3-base, PyTorch FSDP |
| Distributed training | Vertex AI Custom Training, NVIDIA T4 |
| Training data | Anthropic HH-RLHF (160k preference pairs) |
| API | FastAPI, Cloud Run |
| Job queue | Pub/Sub |
| Storage | BigQuery, Google Cloud Storage |
| LLM inference | Vertex AI (Gemini Flash, Gemini Pro) |
| Dashboard | Looker Studio |
| Container registry | Artifact Registry |

---

## Training results

The reward model was trained for 3 epochs on 160k preference pairs using Bradley-Terry loss:

| Epoch | Train loss | Train accuracy | Eval accuracy |
|---|---|---|---|
| 1 | 0.6889 | 54.96% | — |
| 2 | 0.5920 | 66.40% | 59.92% |
| 3 | 0.5254 | 71.78% | 61.31% |

Trained on a single NVIDIA T4 GPU on Vertex AI. Loss curves and GPU utilization tracked in W&B:
`kamayanirai771-arizona-state-university/llm-eval-platform`

---

## Eval results (TruthfulQA)

Ran 50 questions from TruthfulQA — a benchmark designed to expose model hallucinations — through Gemini 2.5 Flash:

- Factuality scores ranged from 0.25 to 1.0 (not all 1.0 like simple factual questions)
- Composite scores ranged from 0.69 to 0.9999
- Questions on contested or ambiguous topics scored below 0.75

*Gemini Pro comparison in progress.*

---

## Project structure

```
llm-eval-platform/
├── api/
│   ├── main.py              # FastAPI app
│   └── schemas.py           # Pydantic models
├── evaluators/
│   ├── base.py
│   ├── toxicity.py          # Detoxify
│   ├── instruction.py       # rule-based
│   ├── factuality.py        # Gemini-as-judge
│   └── reward_model.py      # DeBERTa checkpoint (WIP)
├── reward_model/
│   ├── model.py             # DeBERTa + scalar head
│   ├── dataset.py           # HH-RLHF loader
│   └── trainer.py           # Bradley-Terry training loop
├── runners/
│   └── gemini.py            # Vertex AI Gemini runner
├── pipeline/
│   ├── worker.py            # Pub/Sub subscriber
│   ├── scorer.py            # parallel evaluator orchestration
│   └── logger.py            # BigQuery writer
├── data/
│   └── datasets/
│       └── truthfulqa.py    # TruthfulQA loader
├── scripts/
│   └── run_truthfulqa_eval.py
├── infra/
│   ├── vertex_training_job.py
│   └── bigquery_schema.json
├── Dockerfile.training      # training container (CUDA + FSDP)
├── cloudbuild.yaml          # Cloud Build config
├── requirements.txt
└── requirements-train.txt
```

---

## Setup

### Prerequisites
- GCP project with billing enabled
- gcloud CLI installed and authenticated
- Python 3.11+

### 1. Clone and configure

```bash
git clone https://github.com/KamayaniR/llm-eval-platform.git
cd llm-eval-platform
cp .env.example .env
# fill in your values
```

### 2. Enable GCP APIs

```bash
gcloud services enable \
  aiplatform.googleapis.com \
  run.googleapis.com \
  pubsub.googleapis.com \
  bigquery.googleapis.com \
  artifactregistry.googleapis.com \
  storage.googleapis.com
```

### 3. Install dependencies

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 4. Run the API locally

```bash
python -m uvicorn api.main:app --port 8080
```

### 5. Submit an eval job

```bash
curl -X POST http://localhost:8080/jobs \
  -H "Content-Type: application/json" \
  -d '{"prompts": ["What is the capital of France?"], "model": "gemini-2.5-flash"}'
```

### 6. Run TruthfulQA eval

```bash
python scripts/run_truthfulqa_eval.py gemini-2.5-flash
```

---

## What's next

**In progress**
- Fix GCS checkpoint upload so reward scores are populated end-to-end (currently null)
- Run Gemini Flash vs Pro comparison on the full TruthfulQA dataset
- Deploy API to Cloud Run (currently runs locally only)

**Planned improvements**
- Faster training: increase to batch_size=32, use 2 GPU replicas, reduce sequence length from 256 to 128 tokens — combined this should cut training time by ~4x
- React frontend for job submission, result visualization, and model comparison
- Use a separate model family for factuality judging (currently Gemini judging Gemini is circular)
- Add MMLU benchmark support
- Cloud Scheduler for automatic retraining when new preference labels accumulate
- Vertex AI Vector Search for failure clustering — group low-scoring responses by semantic similarity to identify failure patterns

**Known issues**
- GCS checkpoint upload fails silently in Vertex AI training container — explicit ADC setup needed
- Factuality evaluator occasionally hits Vertex AI rate limits on large eval batches
- Worker reloads Detoxify model on every restart — needs caching improvement

---

## Author

Kamayani Rai · [GitHub](https://github.com/KamayaniR) · [LinkedIn](https://linkedin.com/in/kamayanirai) · [Medium](https://medium.com/@kamayanirai771)
