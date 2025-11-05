import os
import argparse
import pandas as pd
from config import load_config, random_seed
from models.biomedclip import BioMedCLIP
from data import split_merged_df_subjectwise, get_rag_loader
from graphrag_interface import GraphRAGBuilder


def main():
    parser = argparse.ArgumentParser(description="Build RAG and graph from rag split")
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

    # ✅ Load merged dataset
    merged_path = data_cfg.dataset.merged_csv_path
    if not os.path.exists(merged_path):
        raise FileNotFoundError(f"❌ Merged dataset not found at: {merged_path}")
    merged_df = pd.read_csv(merged_path)
    print(f"\n📊 Loaded full merged dataset: {len(merged_df)} rows")

    # ✅ Subject-wise split → (train_df, rag_df, test_df)
    train_df, rag_df, test_df = split_merged_df_subjectwise(
        merged_df,
        split_ratios=data_cfg.dataset.dataset_split,  # e.g., [0.6, 0.3, 0.1]
    )

    # ✅ Create model wrapper (tokenizer / context_length / transforms)
    model_wrapper = BioMedCLIP(model_cfg, use_best=False)
    tokenizer = model_wrapper.tokenizer
    context_length = model_wrapper.train_cfg["context_length"]
    transforms = model_wrapper.transforms
    download_path = data_cfg.dataset.download_path

    rag_loader = get_rag_loader(
        rag_df=rag_df,
        download_path=download_path,
        batch_size=1,
        num_workers=0,
        tokenizer=tokenizer,
        context_length=context_length,
        transforms=transforms,
        rag_keep_frac=1.0,
        seed=random_seed,
    )

    # === GraphRAG config ===
    image_collection_name = data_cfg.graphrag.image_collection_name
    text_collection_name = data_cfg.graphrag.text_collection_name
    chroma_dir = data_cfg.graphrag.chroma_dir
    local_graph_dir = data_cfg.graphrag.local_graph_dir
    global_graph_dir = data_cfg.graphrag.global_graph_dir
    postings_dir = data_cfg.graphrag.postings_dir

    builder = GraphRAGBuilder(
        model_wrapper=model_wrapper,
        dataloader=rag_loader,
        image_collection_name=image_collection_name,
        text_collection_name=text_collection_name,
        chroma_dir=chroma_dir,
        local_graph_dir=local_graph_dir,
        global_graph_dir=global_graph_dir,
        postings_dir=postings_dir,
    )

    # 1) Save embeddings
    builder.save_embeddings()

    # 2) Build local graphs
    builder.build_and_save_local_subgraphs()

    # 2.5) Build postings (connect global tokens/edges back to studies)
    builder.build_postings_from_subgraphs()

    # 3) Build global graphs (relations + co-occurrence)
    builder.build_global_graph_from_subgraphs()


if __name__ == "__main__":
    main()

# usage:
# python generate_graphrag_index.py mimic biomedclip
