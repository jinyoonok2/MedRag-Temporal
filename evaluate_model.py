import os
import argparse
import pandas as pd
import torch
from config import load_config
from data import split_merged_df_subjectwise, split_train_val_studywise, get_val_loader
from models.biomedclip import BioMedCLIP
from trainer import CLIPEvaluator


def evaluate_model(dataset_name, model_name, use_pretrained=False):
    print(f"📌 Evaluating CLIP model '{model_name}' on '{dataset_name}'"
          f" using {'pretrained' if use_pretrained else 'trained'} weights")

    dataset_cfg_map = {
        "mimic": "mimic_cxr.yaml"
    }

    model_cfg_map = {
        "biomedclip": "biomedclip.yaml"
    }

    if dataset_name not in dataset_cfg_map:
        raise ValueError(f"❌ Unknown dataset: {dataset_name}")
    if model_name not in model_cfg_map:
        raise ValueError(f"❌ Unknown model: {model_name}")

    # Load configs
    data_cfg = load_config(dataset_cfg_map[dataset_name])
    model_cfg = load_config(model_cfg_map[model_name])

    # Load merged dataset
    merged_path = os.path.join(data_cfg.dataset.download_path, "dataset_processed_merged.csv")
    if not os.path.exists(merged_path):
        raise FileNotFoundError(f"❌ Merged dataset not found at: {merged_path}")
    merged_df = pd.read_csv(merged_path)

    # ✅ Create loaders
    model_wrapper = BioMedCLIP(model_cfg)

    # ✅ Split merged_df into train_full/graphrag/test
    split_dfs = split_merged_df_subjectwise(
        merged_df,
        split_ratios=data_cfg.dataset.dataset_split,
        sample_frac=args.sample_frac
    )

    # ✅ Further split train_full into train/val by study_id
    train_df, val_df = split_train_val_studywise(
        split_dfs["train_full"],
        train_split=data_cfg.dataset.train_split
    )

    val_loader = get_val_loader(
        val_df,
        data_cfg.dataset.download_path,
        model_wrapper.train_cfg["batch_size"],
        model_wrapper.train_cfg["num_workers"]
    )
    # Load weights
    model = model_wrapper.model
    if use_pretrained:
        print("📦 Using pretrained weights")
    else:
        weights_path = model_cfg.clip.best_pt
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"❌ Trained weights not found at: {weights_path}")
        model.load_state_dict(torch.load(weights_path, map_location="cpu"))
        print(f"📦 Loaded trained weights from: {weights_path}")

    # Evaluate
    evaluator = CLIPEvaluator(model_wrapper, val_loader)
    val_loss, recall1, recall5, recall10 = evaluator.evaluate(model=model)

    print("\n📊 Final Evaluation Results:")
    print(f"{'val_loss':20}: {val_loss:.4f}")
    print(f"{'recall@1':20}: {recall1:.4f}")
    print(f"{'recall@5':20}: {recall5:.4f}")
    print(f"{'recall@10':20}: {recall10:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate CLIP model on validation set")
    parser.add_argument("dataset", help="Dataset name (e.g., 'mimic')")
    parser.add_argument("model", help="Model name (e.g., 'biomedclip')")
    parser.add_argument("--pretrained", action="store_true", help="Use pretrained weights instead of trained")
    args = parser.parse_args()

    evaluate_model(
        dataset_name=args.dataset,
        model_name=args.model,
        use_pretrained=args.pretrained
    )
