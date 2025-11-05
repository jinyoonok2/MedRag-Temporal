# graph_annotate.py
import os
import argparse
import pandas as pd

from config import load_config, random_seed
from models import BioMedCLIP
from data import (
    get_rag_loader,
    split_merged_df_subjectwise,
)
from graphrag_interface import generate_radgraph_annotations

# === RadGraph annotator ===
import json
from tqdm import tqdm
from radgraph import RadGraph

def main():
    parser = argparse.ArgumentParser(description="Generate RadGraph JSON annotations for RAG split only")
    parser.add_argument("dataset", help="Dataset name (e.g., 'mimic')")
    parser.add_argument("model", help="Model name (e.g., 'biomedclip')")
    args = parser.parse_args()

    dataset_cfg_map = {"mimic": "mimic_cxr.yaml"}
    model_cfg_map   = {"biomedclip": "biomedclip.yaml"}

    if args.dataset not in dataset_cfg_map:
        raise ValueError(f"❌ Unsupported dataset: {args.dataset}")
    if args.model not in model_cfg_map:
        raise ValueError(f"❌ Unsupported model: {args.model}")

    data_cfg  = load_config(dataset_cfg_map[args.dataset])
    model_cfg = load_config(model_cfg_map[args.model])

    # Load merged metadata
    merged_path = data_cfg.dataset.merged_csv_path
    if not os.path.exists(merged_path):
        raise FileNotFoundError(f"❌ Merged dataset not found at: {merged_path}")
    merged_df = pd.read_csv(merged_path)
    print(f"\n📊 Loaded full merged dataset: {len(merged_df)} rows")

    # Subject-wise split → we only need the RAG split
    train_df, rag_df, test_df = split_merged_df_subjectwise(
        merged_df,
        split_ratios=data_cfg.dataset.dataset_split,  # e.g., [0.6, 0.3, 0.1]
    )

    download_path = data_cfg.dataset.download_path

    # Model wrapper gives tokenizer/context_length/transforms required by your Dataset
    model_wrapper  = BioMedCLIP(model_cfg, use_best=False)
    tokenizer      = model_wrapper.tokenizer
    context_length = model_wrapper.train_cfg["context_length"]
    transforms     = model_wrapper.transforms

    # Build only the RAG loader
    bs = 1
    nw = 0
    rag_loader = get_rag_loader(
        rag_df,
        download_path,
        batch_size=bs,
        num_workers=nw,
        tokenizer=tokenizer,
        context_length=context_length,
        transforms=transforms,
        rag_keep_frac=1.0,
        seed=random_seed,
    )

    # Run annotations on RAG split
    generate_radgraph_annotations(rag_loader)
    print("\n✅ Finished generating RadGraph annotations for RAG split.")


if __name__ == "__main__":
    main()

# python generate_radgraph_annotations.py mimic biomedclip
