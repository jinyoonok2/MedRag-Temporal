# test_graphrag_retriever.py
import os
import argparse
import time
import inspect
import pandas as pd
import torch

from graphrag_interface import GraphAugmentedRetriever
from models.biomedclip import BioMedCLIP
from data import split_merged_df_subjectwise, get_test_loader
from config import load_config
from typing import Tuple, cast


def _require_refined_query_from_batch(batch, expect_tokens: bool = True):
    """
    Build the query from *refined* report only. No fallbacks.
    Returns:
      - torch.LongTensor shaped [1, L] (preferred) if 'refined_report_token' present
      - raw string if only 'refined_report_txt' present
    Otherwise raises a RuntimeError with a clear message.
    """
    if "refined_report_token" in batch:
        tok = batch["refined_report_token"]
        if not isinstance(tok, torch.Tensor):
            raise RuntimeError("refined_report_token is not a Tensor.")
        if tok.dim() == 2 and tok.size(0) >= 1:
            return tok[0:1]  # [1, L]
        if tok.dim() == 1:
            return tok.unsqueeze(0)  # [1, L]
        raise RuntimeError(f"Unexpected refined_report_token shape: {tuple(tok.shape)}")

    if "refined_report_txt" in batch:
        txt = batch["refined_report_txt"]
        if isinstance(txt, (list, tuple)):
            if len(txt) == 0:
                raise RuntimeError("refined_report_txt is an empty list.")
            txt = txt[0]
        if not isinstance(txt, str) or not txt.strip():
            raise RuntimeError("refined_report_txt is missing or empty.")
        return txt

    pkey = "refined_report_txt_path"
    if pkey in batch:
        path = batch[pkey]
        if isinstance(path, (list, tuple)):
            if len(path) == 0:
                raise RuntimeError("refined_report_txt_path is an empty list.")
            path = path[0]
        if not isinstance(path, str) or not os.path.exists(path):
            raise RuntimeError("refined_report_txt_path is missing or does not exist.")
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            txt = f.read().strip()
        if not txt:
            raise RuntimeError(f"Refined report file exists but is empty: {path}")
        return txt

    raise RuntimeError(
        "Refined report is required but not found. "
        "Expected one of: 'refined_report_token', 'refined_report_txt', or 'refined_report_txt_path'."
    )


def _print_results(label, results, batch, retriever, K, M, limits, topN, elapsed, snippet_len: int = 30):
    def _ok(p):
        return "✅" if (isinstance(p, str) and len(p) > 0 and os.path.exists(p)) else "❌"

    def _snip(s, n=30):
        if not isinstance(s, str) or not s:
            return "∅"
        s = s.replace("\n", " ").replace("\r", " ")
        return (s[:n] + "…") if len(s) > n else s

    print(f"\n=== {label} ===")
    dbg = results.get("debug", {})
    anchors = dbg.get("anchors", {})
    expanded_ids = dbg.get("expanded_ids", [])
    print(f"anchors.tokens: {len(anchors.get('tokens', []))}")
    print(f"anchors.relations: {len(anchors.get('relations', []))}")
    print(f"expanded_ids: {len(expanded_ids)}")

    sid_batch = batch.get("study_id")
    if isinstance(sid_batch, (list, tuple)):
        sid_batch = sid_batch[0] if sid_batch else None
    print(f"Query study_id: {sid_batch}")

    # Baseline
    print("\n--- Baseline (CLIP-only) top results ---")
    baseline = results.get("baseline", [])
    if not baseline:
        print("  (no baseline results)")
    else:
        for r in baseline:
            sid = r.get("study_id", "")
            img_p = r.get("image_path", "")
            ref_p = r.get("refined_report_txt_path", "")
            ref_txt = r.get("refined_report_txt", "")
            print(f"  {str(sid):>8}  score_emb={r.get('score_emb', float('nan')):.4f}")
            print(f"      image_path:               {_ok(img_p)}  {img_p}")
            print(f"      refined_report_txt_path:  {_ok(ref_p)}  {ref_p}")
            print(f"      refined_report_txt[0:{snippet_len}]: \"{_snip(ref_txt, snippet_len)}\"")

    # Graph-augmented
    print("\n--- Graph-augmented top results ---")
    graph = results.get("graph", [])
    if not graph:
        print("  (no graph results)")
    else:
        for r in graph:
            sid = r.get("study_id", "")
            img_p = r.get("image_path", "")
            ref_p = r.get("refined_report_txt_path", "")
            ref_txt = r.get("refined_report_txt", "")
            final = r.get("final_score", float('nan'))
            emb = r.get("score_emb_norm", float('nan'))
            rel = r.get("score_rel_norm", float('nan'))
            cooc = r.get("score_cooc_norm", float('nan'))
            print(
                "  {sid:>8}  final={final:.4f}  emb={emb:.3f}  rel={rel:.3f}  cooc={cooc:.3f}".format(
                    sid=str(sid), final=final, emb=emb, rel=rel, cooc=cooc
                )
            )
            print(f"      image_path:               {_ok(img_p)}  {img_p}")
            print(f"      refined_report_txt_path:  {_ok(ref_p)}  {ref_p}")
            print(f"      refined_report_txt[0:{snippet_len}]: \"{_snip(ref_txt, snippet_len)}\"")

    print(
        f"\n⏱️  Retrieval time: {elapsed:.2f}s "
        f"(K={K}, M={M}, limits={limits}, topN={topN}, FAST_POSTINGS_ONLY={retriever.FAST_POSTINGS_ONLY})"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Test GraphRAG retriever with a single sample (image-priority and refined-text-only)"
    )
    parser.add_argument("dataset", help="Dataset name (e.g., 'mimic')")
    parser.add_argument("model", help="Model name (e.g., 'biomedclip')")
    parser.add_argument("--idx", type=int, default=0, help="Index of test sample to retrieve for (default: 0)")
    parser.add_argument("--snip", type=int, default=30, help="Chars to show from refined text (default: 30)")
    args = parser.parse_args()

    dataset_cfg_map = {"mimic": "mimic_cxr.yaml"}
    model_cfg_map = {"biomedclip": "biomedclip.yaml"}

    if args.dataset not in dataset_cfg_map:
        raise ValueError(f"❌ Unsupported dataset: {args.dataset}")
    if args.model not in model_cfg_map:
        raise ValueError(f"❌ Unsupported model: {args.model}")

    # Load configs
    data_cfg = load_config(dataset_cfg_map[args.dataset])
    model_cfg = load_config(model_cfg_map[args.model])
    retr_cfg = load_config("retriever.yaml")

    # Load merged CSV
    merged_path = data_cfg.dataset.merged_csv_path
    if not os.path.exists(merged_path):
        raise FileNotFoundError(f"❌ Merged dataset not found at: {merged_path}")
    merged_df = pd.read_csv(merged_path)

    # Split into train/rag/test
    train_df, rag_df, test_df = split_merged_df_subjectwise(
        merged_df,
        split_ratios=data_cfg.dataset.dataset_split,
    )

    # Model wrapper
    model_wrapper = BioMedCLIP(model_cfg, use_best=True)
    download_path = data_cfg.dataset.download_path
    context_length = getattr(model_wrapper, "train_cfg", {}).get("context_length", model_cfg.context_length)
    tokenizer = model_wrapper.tokenizer
    transforms = model_wrapper.transforms

    # Create test loader
    test_loader = get_test_loader(
        test_df=test_df,
        download_path=download_path,
        batch_size=1,
        num_workers=0,
        tokenizer=tokenizer,
        context_length=context_length,
        transforms=transforms,
    )

    # Prepare retriever
    retriever = GraphAugmentedRetriever(
        model_wrapper=model_wrapper,
        chroma_dir=data_cfg.graphrag.chroma_dir,
        image_collection_name=data_cfg.graphrag.image_collection_name,
        text_collection_name=data_cfg.graphrag.text_collection_name,
        local_graph_dir=data_cfg.graphrag.local_graph_dir,
        global_graph_dir=data_cfg.graphrag.global_graph_dir,
        postings_dir=data_cfg.graphrag.postings_dir,
    )

    # --- Verify which class path is being used (helps avoid stale imports)
    import_path = inspect.getsourcefile(GraphAugmentedRetriever)
    print(f"\nℹ️ GraphAugmentedRetriever loaded from: {import_path}\n")

    retriever.FAST_POSTINGS_ONLY = bool(getattr(retr_cfg, "FAST_POSTINGS_ONLY", True))
    retriever.FILTER_ANCHOR_BY_DF = bool(getattr(retr_cfg, "FILTER_ANCHOR_BY_DF", True))
    retriever.DF_MIN = int(getattr(retr_cfg, "DF_MIN", 5))
    retriever.DF_MAX = int(getattr(retr_cfg, "DF_MAX", 3000))
    retriever.DEG_MAX_REL = int(getattr(retr_cfg, "DEG_MAX_REL", 2000))
    retriever.DEG_MAX_COOC = int(getattr(retr_cfg, "DEG_MAX_COOC", 2000))

    # --- Retrieval settings from retr_cfg
    K = int(getattr(retr_cfg, "K", 100))
    M = int(getattr(retr_cfg, "M", 15))
    topN = int(getattr(retr_cfg, "topN", 20))
    weights = cast(Tuple[float, float, float],
                   tuple(float(x) for x in getattr(retr_cfg, "weights", [0.7, 0.2, 0.1])))
    limits = {
        "k_rel_per_anchor_dir": int(getattr(retr_cfg, "k_rel_per_anchor_dir", 50)),
        "k_cooc_per_anchor": int(getattr(retr_cfg, "k_cooc_per_anchor", 50)),
        "pool_cap": int(getattr(retr_cfg, "pool_cap", 1200)),
    }

    # Get one batch
    batch = None
    for i, b in enumerate(test_loader):
        if i == args.idx:
            batch = b
            break
    if batch is None:
        raise ValueError(f"❌ No batch found at index {args.idx}")

    # Pull the image tensor (kept as-is)
    query_image_tensor = batch.get("image_pt")

    # Build a refined-only query (token tensor or raw string). Error if absent.
    query_text = _require_refined_query_from_batch(batch)

    # 1) Image-priority run (image + refined text)
    print("🔍 Running GraphRAG retriever (image-priority: image + refined text)...\n")
    t0 = time.perf_counter()
    results_img_priority = retriever.retrieve_pairs(
        query_image_tensor=query_image_tensor,
        query_text=query_text,
        K=K,
        M=M,
        limits=limits,
        weights=weights,
        topN=topN,
        normalize_scores=True,
        include_text_docs=False,   # retriever hydrates refined text for returned Top-N
        return_debug=True,
    )
    t1 = time.perf_counter()
    _print_results("Image-priority (image + refined text)", results_img_priority, batch, retriever, K, M, limits, topN, t1 - t0, snippet_len=args.snip)

    # 2) Refined-text-only run
    print("\n🔍 Running GraphRAG retriever (refined-text-only)...\n")
    t2 = time.perf_counter()
    results_text_only = retriever.retrieve_pairs(
        query_text=query_text,
        K=K,
        M=M,
        limits=limits,
        weights=weights,
        topN=topN,
        normalize_scores=True,
        include_text_docs=False,   # retriever hydrates refined text for returned Top-N
        return_debug=True,
    )
    t3 = time.perf_counter()
    _print_results("Refined-text-only", results_text_only, batch, retriever, K, M, limits, topN, t3 - t2, snippet_len=args.snip)

    print("\nDone.")


if __name__ == "__main__":
    main()

# Example:
# python test_graphrag_retriever.py mimic biomedclip --snip 30
