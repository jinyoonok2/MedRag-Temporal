import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import pandas as pd
import matplotlib.pyplot as plt
import random
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR


def set_seed(seed):
    print(f"🌱 Setting random seed to {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class CLIPTrainer:
    def __init__(self, model_wrapper, train_loader, val_loader, seed, best_metric_key="recall@5"):
        set_seed(seed)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🚀 Using device: {self.device}")

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.best_metric_key = best_metric_key

        self.model = model_wrapper.model.to(self.device)
        self.tokenizer = model_wrapper.tokenizer
        self.train_cfg = model_wrapper.train_cfg

        self.optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=float(self.train_cfg["learning_rate"]),
            weight_decay=float(self.train_cfg["weight_decay"])
        )

        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.train_cfg["epochs"])
        self.criterion = nn.CrossEntropyLoss()

        self.evaluator = CLIPEvaluator(model_wrapper, val_loader)  # Pass text_key here
        self.best_metric_value = -float("inf") if self._metric_higher_is_better(best_metric_key) else float("inf")
        self.metrics_log = []


    def _metric_higher_is_better(self, key):
        return key in {"recall@1", "recall@5"}

    def process_batch(self, images, tokens):
        images = images.to(self.device)
        tokens = tokens.to(self.device)

        image_features, text_features, logit_scale = self.model(images, tokens)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return image_features, text_features, logit_scale

    def train(self):
        print("🚀 Starting Training...")
        print(f"🏅 Using '{self.best_metric_key}' to determine best.pt")
        self.model.train()

        save_dir = self.train_cfg["weights_dir"]
        os.makedirs(save_dir, exist_ok=True)

        for epoch in range(self.train_cfg["epochs"]):
            print(f"\n📘 Epoch {epoch + 1}/{self.train_cfg['epochs']} — [TRAINING]")
            total_loss = 0.0
            pbar = tqdm(self.train_loader, desc="Training", leave=False)

            for batch in pbar:
                images = batch["image_pt"]
                tokens = batch["refined_report_token"]

                self.optimizer.zero_grad()
                image_features, text_features, logit_scale = self.process_batch(images, tokens)
                logits_per_image = logit_scale * image_features @ text_features.t()
                logits_per_text = logits_per_image.t()
                targets = torch.arange(image_features.size(0), device=self.device)

                loss = (self.criterion(logits_per_image, targets) +
                        self.criterion(logits_per_text, targets)) / 2

                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                pbar.set_postfix({"batch_loss": loss.item()})

            avg_train_loss = total_loss / len(self.train_loader)
            print(f"✅ Epoch {epoch + 1} | Avg Train Loss: {avg_train_loss:.4f}")

            val_loss, recall1, recall5, recall10 = self.evaluator.evaluate(self.model)

            metrics = {
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "val_loss": val_loss,
                "recall@1": recall1,
                "recall@5": recall5,
                "recall@10": recall10,
            }

            current_metric = metrics[self.best_metric_key]
            print(f"🔎 Using '{self.best_metric_key}' = {current_metric:.4f} for best model selection")

            is_new_best = (
                current_metric > self.best_metric_value if self._metric_higher_is_better(self.best_metric_key)
                else current_metric < self.best_metric_value
            )

            if is_new_best:
                self.best_metric_value = current_metric
                for entry in self.metrics_log:
                    entry["is_best"] = False

            metrics["is_best"] = is_new_best
            self.metrics_log.append(metrics)

            if is_new_best:
                best_path = self.train_cfg["best_pt"]
                torch.save(self.model.state_dict(), best_path)
                print(f"💾 Saved new best model to: {best_path} (Epoch {metrics['epoch']})")

            self.scheduler.step()
            print("────────────────────────────────────────────")

        last_path = self.train_cfg["last_pt"]
        torch.save(self.model.state_dict(), last_path)
        print(f"💾 Saved final model to: {last_path}")

        csv_path = os.path.join(save_dir, "metrics_log.csv")
        pd.DataFrame(self.metrics_log).to_csv(csv_path, index=False)
        print(f"📈 Saved metrics to: {csv_path}")
        plot_metrics(csv_path, save_dir, best_metric_key=self.best_metric_key)

        best_epoch = next((entry["epoch"] for entry in self.metrics_log if entry["is_best"] == True), None)
        if best_epoch:
            print(f"🏁 Best model was found at epoch {best_epoch} based on '{self.best_metric_key}'")


class CLIPEvaluator:
    def __init__(self, model_wrapper, val_loader):
        self.model = model_wrapper.model
        self.tokenizer = model_wrapper.tokenizer
        self.context_length = model_wrapper.train_cfg["context_length"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.val_loader = val_loader


    def evaluate(self, model=None, weights_path=None):
        if model is None and weights_path is None:
            raise ValueError("❌ Either `model` or `weights_path` must be provided.")

        model = model or self.model

        if weights_path:
            model.load_state_dict(torch.load(weights_path, map_location=self.device))
            print(f"📦 Loaded weights from {weights_path}")

        model.to(self.device)
        model.eval()

        criterion = nn.CrossEntropyLoss()
        total_loss = 0.0
        all_image_features = []
        all_text_features = []

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Evaluating", leave=False)
            for batch in pbar:
                images = batch["image_pt"].to(self.device)
                tokens = batch["refined_report_token"].to(self.device)

                image_features, text_features, logit_scale = model(images, tokens)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

                logits_per_image = logit_scale * image_features @ text_features.t()
                logits_per_text = logits_per_image.t()
                targets = torch.arange(image_features.size(0), device=self.device)

                loss = (criterion(logits_per_image, targets) + criterion(logits_per_text, targets)) / 2
                total_loss += loss.item()

                all_image_features.append(image_features)
                all_text_features.append(text_features)
                pbar.set_postfix({"loss": loss.item()})

        avg_loss = total_loss / len(self.val_loader)
        sim_matrix = torch.cat(all_image_features) @ torch.cat(all_text_features).t()
        targets = torch.arange(sim_matrix.size(0), device=self.device)
        recall1 = self.recall_at_k(sim_matrix, targets, k=1)
        recall5 = self.recall_at_k(sim_matrix, targets, k=5)
        recall10 = self.recall_at_k(sim_matrix, targets, k=10)

        print(f"📉 Validation Loss: {avg_loss:.4f}")
        print(f"🎯 Recall@1: {recall1:.4f} | Recall@5: {recall5:.4f} | Recall@10: {recall10:.4f}")
        return avg_loss, recall1, recall5, recall10

    def recall_at_k(self, sims, targets, k):
        topk = sims.topk(k, dim=1).indices
        correct = (topk == targets.unsqueeze(1)).any(dim=1).float()
        return correct.mean().item()


def plot_metrics(csv_path, save_dir=None, best_metric_key="recall@5"):
    df = pd.read_csv(csv_path)
    if save_dir is None:
        save_dir = os.path.dirname(csv_path)

    metrics_to_plot = [col for col in df.columns if col not in ["epoch", "is_best"]]
    best_row = df[df["is_best"] == True]
    best_epoch = int(best_row["epoch"].values[0]) if not best_row.empty else None

    for metric in metrics_to_plot:
        plt.figure()
        plt.plot(df["epoch"], df[metric], marker='o', label=metric)

        if best_epoch is not None and metric == best_metric_key:
            best_value = float(best_row[metric].values[0])
            plt.scatter(best_epoch, best_value, color='red', label='Best', zorder=5)
            plt.annotate(f'Best @ {best_epoch}', (best_epoch, best_value),
                         textcoords="offset points", xytext=(0, 10), ha='center', color='red')

        plt.title(f"{metric} over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel(metric)
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, f"{metric}.png"))
        plt.close()

    print(f"📊 Saved metric plots with best epoch highlighted to: {save_dir}")
