# print_subject_reports.py
import os
import argparse
import random
import textwrap
import pandas as pd

def load_merged_csv(download_path: str) -> pd.DataFrame:
    merged_csv = os.path.join(download_path, "dataset_processed_merged.csv")
    if not os.path.exists(merged_csv):
        raise FileNotFoundError(
            f"Missing merged CSV at: {merged_csv}\n"
            "Run your MIMIC-CXR downloader first so this file is created."
        )
    df = pd.read_csv(merged_csv)
    # Basic sanity columns
    required_cols = {"subject_id", "study_id"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Merged CSV missing columns: {missing}")
    return df

def report_path_for_study(download_path: str, study_id: int | str) -> str:
    study_str = str(study_id)
    return os.path.join(download_path, study_str, f"s{study_str}.txt")

def read_report(path: str) -> str:
    if not os.path.exists(path):
        return "[Report file not found]"
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception as e:
        return f"[Error reading report: {e}]"

def print_subject_reports(download_path: str, n: int = 5, seed: int = 42) -> None:
    df = load_merged_csv(download_path)

    # Ensure deterministic sampling if desired
    rng = random.Random(seed)

    # Unique subject_ids that actually have at least one study
    subjects = sorted(df["subject_id"].unique())
    if len(subjects) == 0:
        print("No subjects found in merged CSV. Nothing to print.")
        return

    if n > len(subjects):
        n = len(subjects)

    sampled_subjects = rng.sample(subjects, n)

    print("=" * 80)
    print(f"Showing reports for {n} randomly sampled subjects "
          f"(seed={seed}) from {len(subjects)} total subjects.")
    print("=" * 80)

    for sidx, subj in enumerate(sampled_subjects, start=1):
        sub_df = df[df["subject_id"] == subj].copy()

        # Sort by study_id as a workable proxy for temporal order.
        # (If you add a proper timestamp column later, sort by that.)
        sub_df = sub_df.sort_values(["study_id"], ascending=True)

        studies = sub_df["study_id"].unique().tolist()

        print("\n" + "#" * 80)
        print(f"[{sidx}/{len(sampled_subjects)}] SUBJECT {subj} — {len(studies)} study(ies)")
        print("#" * 80)

        for i, study_id in enumerate(studies, start=1):
            path = report_path_for_study(download_path, study_id)
            text = read_report(path)

            header = f"Subject {subj} | Study {study_id} ({i}/{len(studies)})"
            print("\n" + "-" * len(header))
            print(header)
            print("-" * len(header))

            # Wrap for more readable terminal output (optional)
            wrapped = textwrap.fill(text, width=100, replace_whitespace=False,
                                    drop_whitespace=False)
            print(wrapped)

def main():
    p = argparse.ArgumentParser(
        description="Print all text reports for N random subject_ids to inspect temporal changes."
    )
    p.add_argument(
        "--download-path",
        required=True,
        help="Root download directory used by your downloader "
             "(contains dataset_processed_merged.csv and per-study folders).",
    )
    p.add_argument("--n", type=int, default=5, help="Number of subjects to sample (default: 5)")
    p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = p.parse_args()

    print_subject_reports(args.download_path, n=args.n, seed=args.seed)

if __name__ == "__main__":
    main()
