import os
import argparse
import pandas as pd
from config import load_config, random_seed
from models import BioMedCLIP
from data import (
    get_train_loader, get_rag_loader, get_test_loader,
    split_merged_df_subjectwise,
)
# ⬇️ refiner: should remove ONLY underscores and keep everything else as-is
from utils import generate_refined_reports


def main():
    parser = argparse.ArgumentParser(description="Generate cleaned report text files")
    parser.add_argument("dataset", help="Dataset name (e.g., 'mimic')")
    parser.add_argument("model", help="Model name (e.g., 'biomedclip')")
    args = parser.parse_args()

    dataset_cfg_map = {"mimic": "mimic_cxr.yaml"}
    model_cfg_map = {"biomedclip": "biomedclip.yaml"}

    if args.dataset not in dataset_cfg_map:
        raise ValueError(f"❌ Unsupported dataset: {args.dataset}")
    if args.model not in model_cfg_map:
        raise ValueError(f"❌ Unsupported model: {args.model}")

    data_cfg = load_config(dataset_cfg_map[args.dataset])
    model_cfg = load_config(model_cfg_map[args.model])

    # ✅ Load merged metadata
    merged_path = data_cfg.dataset.merged_csv_path
    if not os.path.exists(merged_path):
        raise FileNotFoundError(f"❌ Merged dataset not found at: {merged_path}")
    merged_df = pd.read_csv(merged_path)
    print(f"\n📊 Loaded full merged dataset: {len(merged_df)} rows")

    # ✅ 3-way subject-wise split → train_df, rag_df, test_df
    train_df, rag_df, test_df = split_merged_df_subjectwise(
        merged_df,
        split_ratios=data_cfg.dataset.dataset_split,  # e.g., [0.6, 0.3, 0.1]
    )

    download_path = data_cfg.dataset.download_path

    # ✅ Model wrapper (for tokenizer/context length/transforms)
    model_wrapper = BioMedCLIP(model_cfg, use_best=False)
    tokenizer = model_wrapper.tokenizer
    context_length = model_wrapper.train_cfg["context_length"]
    transforms = model_wrapper.transforms

    # Dataloader params
    bs = 1
    nw = 0

    # ✅ Build loaders directly from each DF
    train_loader = get_train_loader(
        train_df, download_path, batch_size=bs, num_workers=nw,
        tokenizer=tokenizer, context_length=context_length, transforms=transforms
    )
    rag_loader = get_rag_loader(
        rag_df, download_path, batch_size=bs, num_workers=nw,
        tokenizer=tokenizer, context_length=context_length, transforms=transforms,
        rag_keep_frac=1.0, seed=int(random_seed)
    )
    test_loader = get_test_loader(
        test_df, download_path, batch_size=bs, num_workers=nw,
        tokenizer=tokenizer, context_length=context_length, transforms=transforms
    )

    # ✅ Run refinement on all
    generate_refined_reports(train_loader, mode="refine")
    generate_refined_reports(rag_loader, mode="refine")
    generate_refined_reports(test_loader, mode="refine")
    print("\n✅ Finished generating refined reports.")


if __name__ == "__main__":
    main()

# usage:
#   python refine_data.py mimic biomedclip      # keep full RAG split
