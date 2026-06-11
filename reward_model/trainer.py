import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
from reward_model.model import RewardModel
from reward_model.dataset import PreferenceDataset
import wandb
import os

def upload_to_gcs(local_path, gcs_path):
    try:
        from google.cloud import storage
        from google.oauth2 import service_account
        import google.auth

        try:
            credentials, project = google.auth.default()
        except Exception:
            credentials = None

        client = storage.Client(credentials=credentials)
        bucket_name = gcs_path.split("/")[2]
        blob_path = "/".join(gcs_path.split("/")[3:])
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_path)
        blob.upload_from_filename(local_path)
        print(f"Uploaded {local_path} to {gcs_path}")
        return True
    except Exception as e:
        print(f"GCS upload failed: {e}")
        return False

def compute_loss(chosen_rewards, rejected_rewards):
    return -F.logsigmoid(chosen_rewards - rejected_rewards).mean()

def compute_accuracy(chosen_rewards, rejected_rewards):
    return (chosen_rewards > rejected_rewards).float().mean().item()

def train(
    epochs=3,
    batch_size=8,
    lr=1e-5,
    max_samples=None,
    save_dir="checkpoints",
    use_wandb=True
):
    os.makedirs("checkpoints", exist_ok=True)

    if use_wandb:
        wandb.init(project="llm-eval-platform", name="reward-model-training")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading datasets...")
    train_ds = PreferenceDataset(split="train", max_samples=max_samples)
    test_ds = PreferenceDataset(split="test", max_samples=500)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    model = RewardModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    total_steps = len(train_loader) * epochs
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    print(f"Training for {epochs} epochs, {len(train_loader)} steps each...")

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_acc = 0

        for step, batch in enumerate(train_loader):
            chosen_ids = batch["chosen_input_ids"].to(device)
            chosen_mask = batch["chosen_attention_mask"].to(device)
            rejected_ids = batch["rejected_input_ids"].to(device)
            rejected_mask = batch["rejected_attention_mask"].to(device)

            chosen_rewards = model(chosen_ids, chosen_mask)
            rejected_rewards = model(rejected_ids, rejected_mask)

            loss = compute_loss(chosen_rewards, rejected_rewards)
            acc = compute_accuracy(chosen_rewards, rejected_rewards)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            total_acc += acc

            if step % 100 == 0:
                avg_loss = total_loss / (step + 1)
                avg_acc = total_acc / (step + 1)
                print(f"Epoch {epoch+1} | Step {step} | Loss: {avg_loss:.4f} | Acc: {avg_acc:.4f}")
                if use_wandb:
                    wandb.log({"train/loss": avg_loss, "train/accuracy": avg_acc, "step": step})

        local_ckpt = f"checkpoints/epoch_{epoch+1}.pt"
        torch.save(model.state_dict(), local_ckpt)
        print(f"Saved checkpoint locally: {local_ckpt}")

        if save_dir.startswith("gs://"):
            gcs_ckpt = f"{save_dir}/epoch_{epoch+1}.pt"
            success = upload_to_gcs(local_ckpt, gcs_ckpt)
            if success:
                print(f"Checkpoint saved to GCS: {gcs_ckpt}")
            else:
                print(f"WARNING: GCS upload failed for epoch {epoch+1}. Checkpoint only saved locally.")

        model.eval()
        eval_loss, eval_acc = 0, 0
        with torch.no_grad():
            for batch in test_loader:
                chosen_ids = batch["chosen_input_ids"].to(device)
                chosen_mask = batch["chosen_attention_mask"].to(device)
                rejected_ids = batch["rejected_input_ids"].to(device)
                rejected_mask = batch["rejected_attention_mask"].to(device)
                chosen_rewards = model(chosen_ids, chosen_mask)
                rejected_rewards = model(rejected_ids, rejected_mask)
                eval_loss += compute_loss(chosen_rewards, rejected_rewards).item()
                eval_acc += compute_accuracy(chosen_rewards, rejected_rewards)

        eval_loss /= len(test_loader)
        eval_acc /= len(test_loader)
        print(f"Epoch {epoch+1} EVAL | Loss: {eval_loss:.4f} | Acc: {eval_acc:.4f}")
        if use_wandb:
            wandb.log({"eval/loss": eval_loss, "eval/accuracy": eval_acc, "epoch": epoch+1})

    if use_wandb:
        wandb.finish()
    print("Training complete.")

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument("--use-wandb", action="store_true")
    args = parser.parse_args()

    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        save_dir=args.save_dir,
        use_wandb=args.use_wandb
    )
