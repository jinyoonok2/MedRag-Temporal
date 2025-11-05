import os
import argparse
import pandas as pd
from config import load_config, random_seed
from trainer import CLIPTrainer
from models.biomedclip import BioMedCLIP
from data import (
    split_merged_df_subjectwise,
    get_train_loader,
    get_rag_loader,
)


def main():
    parser = argparse.ArgumentParser(description="Train CLIP model")
    parser.add_argument("dataset", help="Dataset name (e.g., 'mimic')")
    parser.add_argument("model", help="Model name (e.g., 'biomedclip')")
    # Optional: how much of the RAG split to keep for validation (by study_id)
    parser.add_argument(
        "rag_keep_frac",
        nargs="?",
        type=float,
        default=0.25,
        help="Fraction of the RAG split to keep by study_id for validation (default 0.25). "
             "Use 1.0 to keep the full RAG split."
    )
    args = parser.parse_args()

    dataset_cfg_map = {"mimic": "mimic_cxr.yaml"}
    model_cfg_map = {"biomedclip": "biomedclip.yaml"}

    if args.dataset not in dataset_cfg_map:
        raise ValueError(f"❌ Unsupported dataset: {args.dataset}")
    if args.model not in model_cfg_map:
        raise ValueError(f"❌ Unsupported model: {args.model}")

    data_cfg = load_config(dataset_cfg_map[args.dataset])
    model_cfg = load_config(model_cfg_map[args.model])

    # Load merged dataset
    merged_path = data_cfg.dataset.merged_csv_path
    if not os.path.exists(merged_path):
        raise FileNotFoundError(f"❌ Merged dataset not found at: {merged_path}")
    merged_df = pd.read_csv(merged_path)
    print(f"\nLoaded full merged dataset: {len(merged_df)} rows")

    # Subject-wise split (reproducible shuffle via seed), ratios by study count
    train_df, rag_df, _ = split_merged_df_subjectwise(
        merged_df,
        split_ratios=data_cfg.dataset.dataset_split,  # e.g., [0.6, 0.3, 0.1]
        seed=int(random_seed),
    )

    # Initialize model wrapper
    model_wrapper = BioMedCLIP(model_cfg, use_best=False)
    tokenizer = model_wrapper.tokenizer
    context_length = model_wrapper.train_cfg["context_length"]
    transforms = model_wrapper.transforms
    batch_size = model_wrapper.train_cfg["batch_size"]
    num_workers = model_wrapper.train_cfg["num_workers"]

    # Build loaders
    train_loader = get_train_loader(
        train_df=train_df,
        download_path=data_cfg.dataset.download_path,
        batch_size=batch_size,
        num_workers=num_workers,
        tokenizer=tokenizer,
        context_length=context_length,
        transforms=transforms,
    )

    # Validation uses RAG split; optionally downsample by study_id via rag_keep_frac
    rag_loader = get_rag_loader(
        rag_df=rag_df,
        download_path=data_cfg.dataset.download_path,
        batch_size=batch_size,
        num_workers=num_workers,
        tokenizer=tokenizer,
        context_length=context_length,
        transforms=transforms,
        rag_keep_frac=args.rag_keep_frac,
        seed=int(random_seed),
    )

    # Train
    trainer = CLIPTrainer(
        model_wrapper=model_wrapper,
        train_loader=train_loader,
        val_loader=rag_loader,
        seed=random_seed,
    )
    trainer.train()


if __name__ == "__main__":
    main()

# usage:
#   python train.py mimic biomedclip           # uses default 0.25 of RAG for validation
#   python train.py mimic biomedclip 1.0       # keep full RAG split for validation
#   python train.py mimic biomedclip 0.33      # keep ~33% of RAG
