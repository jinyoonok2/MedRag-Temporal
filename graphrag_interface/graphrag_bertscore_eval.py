# graphrag_bertscore_eval.py (STRICT, refined-only, no fallbacks)
import os
import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

@dataclass
class EvalSummary:
    best_f1: float
    mean_f1: float

class BERTScoreEvaluator:
    """Strict wrapper around BERTScore (refined-only; no fallbacks)."""
    def __init__(self, model_name: str = "emilyalsentzer/Bio_ClinicalBERT", device: Optional[str] = None):
        try:
            from bert_score import BERTScorer  # noqa: F401
        except Exception as e:
            raise RuntimeError(
                "bert-score is not installed. Run:\n  pip install bert-score transformers torch --upgrade"
            ) from e

        self.model_name = model_name
        self.device = device

        from transformers import AutoTokenizer
        self._tok = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        self._max_tokens = 512

        from bert_score import BERTScorer
        self.scorer = BERTScorer(
            model_type=self.model_name,
            num_layers=12,
            device=self.device if self.device is not None else None,
            lang="en",
            rescale_with_baseline=False,
        )

    def _truncate_text(self, text: str) -> str:
        ids = self._tok.encode(text, add_special_tokens=True, max_length=self._max_tokens, truncation=True)
        return self._tok.decode(ids, skip_special_tokens=True)

    def resolve_gt_text(self, batch: Dict[str, Any]) -> str:
        """Must come from refined_report_txt in batch; error if missing/empty."""
        if "refined_report_txt" not in batch:
            raise KeyError("Batch missing 'refined_report_txt'.")
        txt = batch["refined_report_txt"]
        if isinstance(txt, list):
            if not txt:
                raise ValueError("'refined_report_txt' list is empty.")
            txt = txt[0]
        if not isinstance(txt, str) or not txt.strip():
            raise ValueError("'refined_report_txt' is not a non-empty string.")
        return self._truncate_text(txt)

    def _collect_ids_with_paths(self, items: List[Dict[str, Any]], topN: int) -> List[str]:
        """Require refined_report_txt_path to exist; raise if any missing."""
        out_ids: List[str] = []
        for i, r in enumerate(items[:topN]):
            sid = str(r.get("study_id", "<?>"))
            if "refined_report_txt_path" not in r:
                raise KeyError(f"[candidate #{i} sid={sid}] missing 'refined_report_txt_path' in retrieval results.")
            p = r["refined_report_txt_path"]
            if not isinstance(p, str) or not p:
                raise ValueError(f"[candidate #{i} sid={sid}] 'refined_report_txt_path' is empty or not a string.")
            if not os.path.exists(p):
                raise FileNotFoundError(f"[candidate #{i} sid={sid}] refined path does not exist: {p}")
            out_ids.append(sid)
        return out_ids

    def _collect_candidate_texts(self, items: List[Dict[str, Any]], topN: int) -> List[str]:
        """Require refined_report_txt as text; raise if any missing/empty."""
        texts: List[str] = []
        for i, r in enumerate(items[:topN]):
            sid = str(r.get("study_id", "<?>"))
            if "refined_report_txt" not in r:
                raise KeyError(f"[candidate #{i} sid={sid}] missing 'refined_report_txt' in retrieval results.")
            txt = r["refined_report_txt"]
            if isinstance(txt, list):
                if not txt:
                    raise ValueError(f"[candidate #{i} sid={sid}] 'refined_report_txt' list is empty.")
                txt = txt[0]
            if not isinstance(txt, str) or not txt.strip():
                raise ValueError(f"[candidate #{i} sid={sid}] 'refined_report_txt' is empty or not a string.")
            texts.append(self._truncate_text(txt))
        if not texts:
            raise RuntimeError("No candidate texts collected; retrieval likely returned missing fields.")
        return texts

    def score(self, ref: str, cands: List[str]) -> Dict[str, Any]:
        if not cands:
            raise RuntimeError("BERTScore called with empty candidate list.")
        refs = [ref] * len(cands)
        P, R, F1 = self.scorer.score(cands, refs)
        P = [float(p) for p in P]
        R = [float(r) for r in R]
        F1 = [float(f) for f in F1]
        return {
            "precision": P,
            "recall": R,
            "f1": F1,
            "summary": {"best_f1": max(F1), "mean_f1": sum(F1)/len(F1)},
        }


class PerQueryBERTScoreRunner:
    """Stage 1: run retrieval + BERTScore for each query and save JSON per query (strict; propagate errors)."""
    def __init__(self, evaluator: BERTScoreEvaluator, topN_eval: int, out_dir: str):
        self.evaluator = evaluator
        self.topN_eval = int(topN_eval)
        if self.topN_eval <= 0:
            raise ValueError("topN_eval must be a positive integer.")
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)

    @staticmethod
    def _get_query_sid(batch: Dict[str, Any]) -> str:
        sid = batch.get("study_id")
        try:
            if isinstance(sid, list) and sid:
                sid = sid[0]
            if hasattr(sid, "item"):
                sid = sid.item()
        except Exception:
            pass
        return str(sid)

    def run_and_save(self, batch: Dict[str, Any], retrieve_results: Dict[str, Any]) -> str:
        """Strict mode: raises on any missing fields so the caller can log the exact reason."""
        gt_text = self.evaluator.resolve_gt_text(batch)

        # Require both groups present
        if "baseline" not in retrieve_results:
            raise KeyError("Retriever results missing 'baseline'.")
        if "graph" not in retrieve_results:
            raise KeyError("Retriever results missing 'graph'.")

        baseline_items = retrieve_results["baseline"]
        graph_items    = retrieve_results["graph"]

        # Require enough candidates to score (surface short/empty lists explicitly)
        if not isinstance(baseline_items, list) or len(baseline_items) < self.topN_eval:
            raise RuntimeError(f"Baseline returned {len(baseline_items) if isinstance(baseline_items, list) else 'non-list'} "
                               f"items; need at least topN_eval={self.topN_eval}.")
        if not isinstance(graph_items, list) or len(graph_items) < self.topN_eval:
            raise RuntimeError(f"Graph returned {len(graph_items) if isinstance(graph_items, list) else 'non-list'} "
                               f"items; need at least topN_eval={self.topN_eval}.")

        # Validate fields and collect data (any missing key raises with rank + sid)
        baseline_ids   = self.evaluator._collect_ids_with_paths(baseline_items, self.topN_eval)
        graph_ids      = self.evaluator._collect_ids_with_paths(graph_items,    self.topN_eval)
        baseline_texts = self.evaluator._collect_candidate_texts(baseline_items, self.topN_eval)
        graph_texts    = self.evaluator._collect_candidate_texts(graph_items,    self.topN_eval)

        # Score
        baseline_scores = self.evaluator.score(gt_text, baseline_texts)
        graph_scores    = self.evaluator.score(gt_text, graph_texts)

        # Persist per-query JSON
        sid = self._get_query_sid(batch)
        out = {
            "query_study_id": sid,
            "status": "ok",
            "topN_eval": self.topN_eval,
            "baseline": {**baseline_scores, "ids": baseline_ids},
            "graph":    {**graph_scores,    "ids": graph_ids},
        }
        out_path = os.path.join(self.out_dir, f"{sid}.json")
        # debugging process during evaluation
        # print("\n[DEBUG] Per-query JSON (OK case):")
        # print(json.dumps(out, ensure_ascii=False, indent=2))
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        return out_path


class CorpusBERTScoreAggregator:
    """Stage 2: reads per-query JSONs and computes overall metrics."""
    def __init__(self, in_dir: str):
        self.in_dir = in_dir

    def _iter_jsons(self):
        for name in os.listdir(self.in_dir):
            if name.lower().endswith(".json"):
                path = os.path.join(self.in_dir, name)
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        yield json.load(f)
                except Exception:
                    continue

    def aggregate(self) -> Dict[str, Any]:
        import numpy as np
        n_ok = 0
        base_best, base_mean, graph_best, graph_mean = [], [], [], []
        for rec in self._iter_jsons():
            if rec.get("status") != "ok":
                continue
            n_ok += 1
            base_best.append(rec["baseline"]["summary"]["best_f1"] if rec["baseline"]["f1"] else 0.0)
            base_mean.append(rec["baseline"]["summary"]["mean_f1"] if rec["baseline"]["f1"] else 0.0)
            graph_best.append(rec["graph"]["summary"]["best_f1"] if rec["graph"]["f1"] else 0.0)
            graph_mean.append(rec["graph"]["summary"]["mean_f1"] if rec["graph"]["f1"] else 0.0)

        def _avg(x): return float(np.mean(x)) if x else 0.0
        best_wins = [g > b for g, b in zip(graph_best, base_best)]
        mean_wins = [g > b for g, b in zip(graph_mean, base_mean)]

        return {
            "num_queries_ok": n_ok,
            "avg_best_baseline": _avg(base_best),
            "avg_best_graph": _avg(graph_best),
            "avg_mean_baseline": _avg(base_mean),
            "avg_mean_graph": _avg(graph_mean),
            "win_rate_best_graph_over_baseline": (sum(best_wins) / len(best_wins)) if best_wins else 0.0,
            "win_rate_mean_graph_over_baseline": (sum(mean_wins) / len(mean_wins)) if mean_wins else 0.0,
        }

    @staticmethod
    def print_summary(summary: Dict[str, Any]) -> None:
        print("\n=== Corpus BERTScore Summary ===")
        print(f"Queries evaluated (status=ok): {summary['num_queries_ok']}")
        print(f"Avg Best-of-5  — Baseline: {summary['avg_best_baseline']:.4f}  |  Graph: {summary['avg_best_graph']:.4f}")
        print(f"Avg Mean-of-5  — Baseline: {summary['avg_mean_baseline']:.4f}  |  Graph: {summary['avg_mean_graph']:.4f}")

    def save_summary(self, path: str, summary: Dict[str, Any]) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
