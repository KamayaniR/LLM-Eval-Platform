# LLM Eval Platform - Work in Progress

A mini RLHF evaluation platform built on GCP. The idea: instead of paying a commercial model to judge every LLM response, train your own reward model on human preference data and use it as the judge. Cheaper, faster, and you own it.

> **Status:** Core pipeline is working end-to-end. Reward model trained (71.8% train accuracy, 61.3% eval accuracy). React frontend built with model comparison and verdict summary. GCS checkpoint upload fix merged — verification pending on next training run.

---

## Demo

Not deployed publicly to avoid uncontrolled API credit consumption. Run locally with the setup below.

**What it looks like:**

The platform has two views:

**Submit job** — enter prompts, pick a model (Flash or Pro), submit. Results auto-poll and appear in ~30 seconds. Each prompt shows the response with four score dimensions.

**Compare models** — pick a set of TruthfulQA questions (designed to expose hallucinations on common misconceptions) or enter your own custom prompts. Runs both models in parallel and shows a side-by-side metric breakdown with diffs and an automatic verdict on which model performed better.

---

## What it does

You submit a list of prompts and a model name. The platform:

1. Sends each prompt to the LLM (Gemini Flash or Pro) via Vertex AI
2. Scores each response across four dimensions in parallel
3. Stores results in BigQuery
4. Surfaces everything in the React frontend (job view + model comparison)

The reward model is a DeBERTa-v3-base model trained from scratch on 160k human preference pairs from Anthropic's HH-RLHF dataset using Bradley-Terry loss. Training ran on Vertex AI on NVIDIA T4 GPUs.

---

## Architecture

```
React frontend (Vite + Tailwind)
  │
  ├── Submit job → POST /jobs
  ├── View results → GET /results/{id}
  └── Compare models → parallel jobs + verdict

FastAPI (Cloud Run — local for now)
  │
  ├── Inserts job to BigQuery
  └── Publishes to Pub/Sub

Pub/Sub (eval-jobs topic)
  │
  └── Worker thread (background)
        │
        ├── Gemini runner → response
        └── Scorer (parallel, ThreadPoolExecutor)
              ├── Reward model   → DeBERTa trained on HH-RLHF (0.4 weight)
              ├── Factuality     → Gemini-as-judge, retry + cache (0.25 weight)
              ├── Instruction    → rule-based (0.2 weight)
              └── Toxicity       → Detoxify (0.15 weight)

BigQuery (llm_eval dataset)
  ├── eval_jobs
  ├── eval_results
  └── preference_labels

Offline training pipeline (Vertex AI)
  └── DeBERTa + Bradley-Terry → GCS checkpoint → reward model evaluator
```

---

## Evaluation dimensions

| Dimension | Method | Weight | What it measures |
|---|---|---|---|
| Reward | DeBERTa-v3-base, HH-RLHF | 40% | Overall response quality as humans would judge it |
| Factuality | Gemini-2.5-flash as judge | 25% | Accuracy of factual claims |
| Instruction | Rule-based | 20% | Format compliance, length, relevance |
| Toxicity | Detoxify | 15% | Safety of language |

---

## Stack

| Component | Technology |
|---|---|
| Frontend | React, Vite, Tailwind v4 |
| Reward model | DeBERTa-v3-base, PyTorch, Bradley-Terry loss |
| Distributed training | Vertex AI Custom Training, NVIDIA T4 |
| Training data | Anthropic HH-RLHF (160k preference pairs) |
| API | FastAPI |
| Job queue | Pub/Sub |
| Storage | BigQuery, Google Cloud Storage |
| LLM inference | Vertex AI (Gemini 2.5 Flash, Gemini 2.5 Pro) |
| Container registry | Artifact Registry |
| Experiment tracking | Weights & Biases |

---

## Training results

| Epoch | Train loss | Train accuracy | Eval accuracy |
|---|---|---|---|
| 1 | 0.6889 | 54.96% | — |
| 2 | 0.5920 | 66.40% | 59.92% |
| 3 | 0.5254 | 71.78% | 61.31% |

Trained on a single NVIDIA T4 GPU on Vertex AI.

---

## Eval results (TruthfulQA — Flash vs Pro)

Ran 100 questions from TruthfulQA on both models:

| Metric | Gemini 2.5 Flash | Gemini 2.5 Pro |
|---|---|---|
| Composite | 0.9818 | 0.9805 |
| Factuality | 0.9745 | 0.9821 |
| Toxicity | 0.9886 | 0.9875 |

Pro scores slightly higher on factuality (more capable model). Flash scores marginally higher on overall composite. Differences are small — both models handle common misconceptions reasonably well on this benchmark.

---

## Project structure

```
llm-eval-platform/
├── frontend/                    # React + Vite + Tailwind frontend
│   └── src/
│       ├── App.jsx
│       └── components/
│           ├── JobSubmit.jsx
│           ├── JobList.jsx
│           ├── ResultsView.jsx
│           └── CompareView.jsx  # model comparison + verdict
├── api/
│   ├── main.py                  # FastAPI app + /sample-prompts endpoint
│   └── schemas.py
├── evaluators/
│   ├── base.py
│   ├── toxicity.py              # Detoxify
│   ├── instruction.py           # rule-based
│   ├── factuality.py            # Gemini-as-judge + retry/cache
│   └── reward_model.py          # DeBERTa checkpoint (WIP — checkpoint upload fix merged)
├── reward_model/
│   ├── model.py                 # DeBERTa-v3-base + scalar head
│   ├── dataset.py               # HH-RLHF loader
│   └── trainer.py               # Bradley-Terry training loop + GCS upload
├── runners/
│   └── gemini.py                # Vertex AI Gemini runner
├── pipeline/
│   ├── worker.py                # Pub/Sub subscriber, loads evaluators once
│   ├── scorer.py                # parallel evaluator orchestration
│   └── logger.py                # BigQuery writer
├── data/
│   └── datasets/
│       └── truthfulqa.py        # TruthfulQA loader (817 questions)
├── scripts/
│   └── run_truthfulqa_eval.py   # submit TruthfulQA batch (configurable size)
├── infra/
│   ├── vertex_training_job.py
│   └── bigquery_schema.json
├── Dockerfile.training
├── cloudbuild.yaml
├── requirements.txt
└── requirements-train.txt
```

---

## Setup

### Prerequisites
- GCP project with billing enabled
- gcloud CLI installed and authenticated
- Python 3.11+
- Node 18+

### 1. Clone and configure

```bash
git clone https://github.com/KamayaniR/llm-eval-platform.git
cd llm-eval-platform
cp .env.example .env
# fill in: PROJECT_ID, GCS_BUCKET, WANDB_API_KEY, GEMINI_API_KEY
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

### 4. Run the API

```bash
python -m uvicorn api.main:app --port 8080
```

### 5. Run the frontend

```bash
cd frontend
npm install
npm run dev
# opens at http://localhost:5173
```

### 6. Run TruthfulQA eval

```bash
python scripts/run_truthfulqa_eval.py gemini-2.5-flash 50
python scripts/run_truthfulqa_eval.py gemini-2.5-pro 50
```

---

## What's next

**In progress**
- Verify GCS checkpoint upload fix on a new Vertex AI training run — once confirmed, reward scores will populate in eval results
- Cloud Run deployment with bring-your-own-API-key auth gate (to allow public usage without exposing GCP credits)

**Planned**
- Faster training: batch_size=32, 2 GPU replicas, seq_len=128 → ~4x speedup
- Multi-model support: add Claude and GPT-4 runners alongside Gemini
- MMLU benchmark support
- Cloud Scheduler for automatic retraining when preference labels accumulate

**Known issues**
- GCS checkpoint upload fails silently in Vertex AI container — fix merged in PR #1, needs verification
- Factuality evaluator occasionally hits rate limits on very large batches (>100 prompts) despite retry logic
- Reward score shows null until checkpoint is verified and wired in


