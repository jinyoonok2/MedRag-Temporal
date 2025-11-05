#!/usr/bin/env python3
import os
import argparse
from typing import Iterable, Tuple
from tqdm import tqdm

# Adjust if your module paths differ
from config import load_config  # type: ignore

# ---------- core ----------

SAFE_FILE_EXTS = {".jpg", ".jpeg", ".png", ".txt", ".pt", ".json"}

def _is_study_dir(name: str) -> bool:
    """Heuristic: study folders are numeric (e.g., '51062323')."""
    return name.isdigit()

def _iter_study_dirs(root: str) -> Iterable[str]:
    for name in os.listdir(root):
        p = os.path.join(root, name)
        if os.path.isdir(p) and _is_study_dir(name):
            yield p

def _safety_guard(path: str) -> None:
    ap = os.path.abspath(path)
    # very basic guardrails to avoid nuking "/" or "~"
    if ap in ("/", os.path.expanduser("~")):
        raise RuntimeError(f"Refusing to clean suspicious path: {ap}")
    if not os.path.isdir(ap):
        raise FileNotFoundError(f"Download path not found: {ap}")
    if len(ap) < 5:
        raise RuntimeError(f"Path too short to be safe: {ap}")

def _delete_files_in_dir(dirpath: str, remove_exts=SAFE_FILE_EXTS) -> Tuple[int, int]:
    files_removed = 0
    dirs_removed = 0
    # delete only files with known, safe extensions
    for fname in os.listdir(dirpath):
        fpath = os.path.join(dirpath, fname)
        if os.path.isfile(fpath):
            ext = os.path.splitext(fname)[1].lower()
            if ext in remove_exts:
                try:
                    os.remove(fpath)
                    files_removed += 1
                except Exception as e:
                    print(f"⚠️  Could not remove {fpath}: {e}")
    # try to remove folder if now empty
    try:
        if not os.listdir(dirpath):
            os.rmdir(dirpath)
            dirs_removed += 1
    except Exception:
        pass
    return files_removed, dirs_removed

def purge_downloads(download_path: str, remove_csv: bool = False) -> None:
    """
    Removes all per-study files and folders under `download_path`.
    By default, keeps top-level CSVs (metadata.csv, chexpert_labels.csv, dataset_processed_merged.csv).
    Set `remove_csv=True` to delete those as well.
    """
    _safety_guard(download_path)

    total_files = 0
    total_dirs = 0

    study_dirs = list(_iter_study_dirs(download_path))
    for sdir in tqdm(study_dirs, desc="Cleaning studies"):
        f, d = _delete_files_in_dir(sdir)
        total_files += f
        total_dirs  += d

    # optionally remove top-level CSVs
    if remove_csv:
        for fname in ("metadata.csv", "chexpert_labels.csv", "dataset_processed_merged.csv"):
            fpath = os.path.join(download_path, fname)
            if os.path.exists(fpath) and os.path.isfile(fpath):
                try:
                    os.remove(fpath)
                    total_files += 1
                except Exception as e:
                    print(f"⚠️  Could not remove {fpath}: {e}")

    # also remove stray .part files in the top-level (if any)
    for fname in os.listdir(download_path):
        fpath = os.path.join(download_path, fname)
        if os.path.isfile(fpath) and fpath.endswith(".part"):
            try:
                os.remove(fpath)
                total_files += 1
            except Exception as e:
                print(f"⚠️  Could not remove {fpath}: {e}")

    print(f"✅ Removed files: {total_files} | Removed empty study dirs: {total_dirs}")
    if not remove_csv:
        print("ℹ️ Top-level CSVs kept. Set remove_csv=True to delete them as well.")

# ---------- CLI ----------

def main():
    parser = argparse.ArgumentParser(description="Clean downloaded dataset files safely.")
    parser.add_argument("dataset", help="Dataset name (e.g., mimic)")
    parser.add_argument("--hard", action="store_true",
                        help="Also delete top-level CSVs (metadata/labels/merged).")
    args = parser.parse_args()

    dataset_cfg_map = {"mimic": "mimic_cxr.yaml"}
    if args.dataset not in dataset_cfg_map:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    data_cfg = load_config(dataset_cfg_map[args.dataset])
    download_path = data_cfg.dataset.download_path
    print(f"🧹 Cleaning under: {download_path}")
    purge_downloads(download_path, remove_csv=args.hard)

if __name__ == "__main__":
    main()

# python clean_datasets.py mimic