# test_graphrag_eval.py
import os
import argparse
import time
import json
import pandas as pd
from tqdm import tqdm
from typing import Tuple, Union

from graphrag_interface import GraphAugmentedRetriever
from models.biomedclip import BioMedCLIP
from data import split_merged_df_subjectwise, get_test_loader
from config import load_config
from graphrag_interface import (
    BERTScoreEvaluator,
    PerQueryBERTScoreRunner,
    CorpusBERTScoreAggregator,
)
import torch

def _require_refined_query_from_batch(batch) -> Union[str, "torch.Tensor"]:
    """
    Build the query from *refined* report only. No fallbacks.
    Returns:
      - torch.LongTensor shaped [1, L] if 'refined_report_token' present
      - raw string if only 'refined_report_txt' present
    Raises if neither is available.
    """
    if "refined_report_token" in batch:
        tok = batch["refined_report_token"]
        if not isinstance(tok, torch.Tensor):
            raise RuntimeError("refined_report_token is not a Tensor.")
        if tok.dim() == 2 and tok.size(0) >= 1:
            return tok[0:1]
        if tok.dim() == 1:
            return tok.unsqueeze(0)
        raise RuntimeError(f"Unexpected refined_report_token shape: {tuple(tok.shape)}")

    if "refined_report_txt" in batch:
        txt = batch["refined_report_txt"]
        if isinstance(txt, (list, tuple)):
            if not txt:
                raise RuntimeError("refined_report_txt is an empty list.")
            txt = txt[0]
        if not isinstance(txt, str) or not txt.strip():
            raise RuntimeError("refined_report_txt is missing or empty.")
        return txt

    raise RuntimeError(
        "Refined report is required but not found. Expected one of: "
        "'refined_report_token' or 'refined_report_txt'."
    )


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate GraphRAG retrieval over the entire test set with BERTScore"
    )
    parser.add_argument("dataset", help="Dataset name (e.g., 'mimic')")
    parser.add_argument("model", help="Model name (e.g., 'biomedclip')")
    parser.add_argument(
        "--mode",
        choices=["image", "text", "auto"],
        default="image",
        help="Query mode: image | text | auto (default: image)",
    )
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
    qwen_cfg = load_config("qwen_eval.yaml")

    # Evaluation knobs (support both naming styles)
    TOPN_EVAL = getattr(qwen_cfg, "top_eval")
    MODEL_NAME = getattr(qwen_cfg, "model_name", "bert-base-uncased")
    DEVICE = getattr(qwen_cfg, "device", "cuda:0")
    EVAL_OUT_DIR = getattr(qwen_cfg, "eval_output_dir")

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

    # Create test loader (batch_size=1)
    test_loader = get_test_loader(
        test_df=test_df,
        download_path=download_path,
        batch_size=1,
        num_workers=0,
        tokenizer=tokenizer,
        context_length=context_length,
        transforms=transforms,
    )
    print(f"✅ Test dataset size: {len(test_loader)}")

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

    # Speed knobs from retr_cfg
    retriever.FAST_POSTINGS_ONLY = bool(getattr(retr_cfg, "FAST_POSTINGS_ONLY", True))
    retriever.FILTER_ANCHOR_BY_DF = bool(getattr(retr_cfg, "FILTER_ANCHOR_BY_DF", True))
    retriever.DF_MIN = int(getattr(retr_cfg, "DF_MIN", 5))
    retriever.DF_MAX = int(getattr(retr_cfg, "DF_MAX", 3000))
    retriever.DEG_MAX_REL = int(getattr(retr_cfg, "DEG_MAX_REL", 2000))
    retriever.DEG_MAX_COOC = int(getattr(retr_cfg, "DEG_MAX_COOC", 2000))

    # --- Retrieval settings from retr_cfg (validate weights shape) ---
    K = int(getattr(retr_cfg, "K", 100))
    M = int(getattr(retr_cfg, "M", 15))
    topN = int(getattr(retr_cfg, "topN", 20))
    raw_w = getattr(retr_cfg, "weights", [0.7, 0.2, 0.1])
    if not (isinstance(raw_w, (list, tuple)) and len(raw_w) == 3):
        raise ValueError("retriever.yaml: 'weights' must be exactly three numbers [alpha, beta, gamma].")
    weights: Tuple[float, float, float] = (float(raw_w[0]), float(raw_w[1]), float(raw_w[2]))
    limits = {
        "k_rel_per_anchor_dir": int(getattr(retr_cfg, "k_rel_per_anchor_dir", 50)),
        "k_cooc_per_anchor": int(getattr(retr_cfg, "k_cooc_per_anchor", 50)),
        "pool_cap": int(getattr(retr_cfg, "pool_cap", 1200)),
    }

    # --- Build evaluator + runner (STRICT) ---
    TOPN_EVAL = getattr(qwen_cfg, "top_eval")
    MODEL_NAME = getattr(qwen_cfg, "model_name", "emilyalsentzer/Bio_ClinicalBERT")
    DEVICE = getattr(qwen_cfg, "device", "cuda:0")
    EVAL_OUT_DIR = getattr(qwen_cfg, "eval_output_dir")

    # Strict config sanity: make sure retrieval returns at least TOPN_EVAL
    if topN < TOPN_EVAL:
        raise ValueError(f"Config mismatch: retriever topN={topN} < TOPN_EVAL={TOPN_EVAL}. "
                         f"Increase 'topN' in retriever.yaml or lower 'top_eval'.")

    evaluator = BERTScoreEvaluator(model_name=MODEL_NAME, device=DEVICE)
    runner = PerQueryBERTScoreRunner(evaluator=evaluator, topN_eval=TOPN_EVAL, out_dir=EVAL_OUT_DIR)

    # ------------- Stage 1: per-query JSONs -------------
    t0 = time.perf_counter()
    prog = tqdm(test_loader, total=len(test_loader), desc=f"Evaluating queries [{args.mode}]", ncols=100)
    num_ok, num_err = 0, 0

    for batch in prog:
        sid = runner._get_query_sid(batch)
        try:
            if args.mode == "image":
                results = retriever.retrieve_pairs(
                    query_image_tensor=batch["image_pt"],
                    K=K, M=M, limits=limits, weights=weights, topN=topN,
                    normalize_scores=True, include_text_docs=False, return_debug=False,
                )
            elif args.mode == "text":
                qtext = _require_refined_query_from_batch(batch)  # <-- refined-only
                results = retriever.retrieve_pairs(
                    query_text=qtext,
                    K=K, M=M, limits=limits, weights=weights, topN=topN,
                    normalize_scores=True, include_text_docs=False, return_debug=False,
                )
            else:  # auto
                if "image_pt" in batch and batch["image_pt"] is not None:
                    results = retriever.retrieve_pairs(
                        query_image_tensor=batch["image_pt"],
                        K=K, M=M, limits=limits, weights=weights, topN=topN,
                        normalize_scores=True, include_text_docs=False, return_debug=False,
                    )
                else:
                    qtext = _require_refined_query_from_batch(batch)  # <-- refined-only fallback
                    results = retriever.retrieve_pairs(
                        query_text=qtext,
                        K=K, M=M, limits=limits, weights=weights, topN=topN,
                        normalize_scores=True, include_text_docs=False, return_debug=False,
                    )

            if not results.get("baseline") and not results.get("graph"):
                tqdm.write(f"⚠️  Empty retrieval for study_id={sid} (mode={args.mode})")

            out_path = runner.run_and_save(batch=batch, retrieve_results=results)
            num_ok += 1
            prog.set_postfix_str(f"ok={num_ok}, err={num_err}")

        except Exception as e:
            num_err += 1
            prog.set_postfix_str(f"ok={num_ok}, err={num_err}")
            tqdm.write(f"\n❌ Error on study_id={sid} [{args.mode}]: {type(e).__name__}: {e}")

            # Keep corpus aggregation structure; still fail-fast per-query.
            stub = {
                "query_study_id": str(sid),
                "status": "error",
                "error": f"{type(e).__name__}: {e}",
                "topN_eval": TOPN_EVAL,
                "baseline": {"ids": [], "precision": [], "recall": [], "f1": [],
                             "summary": {"best_f1": 0.0, "mean_f1": 0.0}},
                "graph": {"ids": [], "precision": [], "recall": [], "f1": [],
                          "summary": {"best_f1": 0.0, "mean_f1": 0.0}},
            }
            os.makedirs(EVAL_OUT_DIR, exist_ok=True)
            with open(os.path.join(EVAL_OUT_DIR, f"{sid}.json"), "w", encoding="utf-8") as f:
                json.dump(stub, f, ensure_ascii=False, indent=2)

    t1 = time.perf_counter()
    print(f"\n⏱️  Stage 1 completed in {(t1 - t0):.2f}s  |  ok={num_ok}, err={num_err}")
    print(f"📁 Per-query JSONs written to: {os.path.abspath(EVAL_OUT_DIR)}")

    # ------------- Stage 2: aggregate -------------
    aggregator = CorpusBERTScoreAggregator(in_dir=EVAL_OUT_DIR)
    summary = aggregator.aggregate()
    aggregator.print_summary(summary)
    aggregator.save_summary(os.path.join(EVAL_OUT_DIR, "aggregate_summary.json"), summary)
    print(f"\n📝 Saved corpus summary to: {os.path.join(EVAL_OUT_DIR, 'aggregate_summary.json')}")


if __name__ == "__main__":
    main()

# Usage:
#   python evaluate_graphrag.py mimic biomedclip --mode image   # default, matches your working path
#   python evaluate_graphrag.py mimic biomedclip --mode text
#   python evaluate_graphrag.py mimic biomedclip --mode auto
