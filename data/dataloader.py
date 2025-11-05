from collections import defaultdict
import random
from typing import Tuple, Optional

from torch.utils.data import DataLoader
from .mimic_dataset import MimicUnifiedDataset
import pandas as pd


def split_merged_df_subjectwise(
    merged_df: "pd.DataFrame",
    split_ratios,
    seed: int = 42,
) -> Tuple["pd.DataFrame", "pd.DataFrame", "pd.DataFrame"]:
    """
    Subject-wise partition into three splits: train_df, rag_df, test_df.
    - Keeps subjects intact across splits.
    - split_ratios must sum to 1.0 (e.g., [0.6, 0.3, 0.1]).
    - Uses a subject-level random shuffle (seeded) for reproducible random sampling.
    - Greedy assignment tries to meet ratios by total *study* counts.

    Returns:
        (train_df, rag_df, test_df)
    """
    assert abs(sum(split_ratios) - 1.0) < 1e-6, "Split ratios must sum to 1.0"

    split_names = ["train", "graphrag", "test"]
    split_study_counts = defaultdict(int)
    split_subjects = defaultdict(list)

    # Precompute study counts per subject
    subject_groups = merged_df.groupby("subject_id")
    subject_study_counts = {subject: len(group) for subject, group in subject_groups}

    # Randomized subject order (reproducible)
    subjects = list(subject_study_counts.keys())
    rng = random.Random(seed)
    rng.shuffle(subjects)

    # Targets in terms of total *study* counts
    target_total = len(merged_df)
    target_counts = [r * target_total for r in split_ratios]

    # Greedy assignment in randomized subject order
    for subject_id in subjects:
        count = subject_study_counts[subject_id]
        ratios = [
            (split_study_counts[name] / target if target > 0 else float("inf"))
            for name, target in zip(split_names, target_counts)
        ]
        split_name = split_names[ratios.index(min(ratios))]
        split_subjects[split_name].append(subject_id)
        split_study_counts[split_name] += count

    train_df = merged_df[merged_df["subject_id"].isin(split_subjects["train"])].reset_index(drop=True)
    rag_df   = merged_df[merged_df["subject_id"].isin(split_subjects["graphrag"])].reset_index(drop=True)
    test_df  = merged_df[merged_df["subject_id"].isin(split_subjects["test"])].reset_index(drop=True)
    return train_df, rag_df, test_df


def get_train_loader(train_df, download_path, batch_size, num_workers, tokenizer, context_length, transforms):
    dataset = MimicUnifiedDataset(
        merged_df=train_df,
        download_path=download_path,
        tokenizer=tokenizer,
        context_length=context_length,
        transforms=transforms,
    )
    if len(dataset) == 0:
        raise ValueError("❌ Train dataset is empty after loading! Check dataframe and file paths.")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    print(f"✅ Train dataset size: {len(dataset)}")
    return loader


def get_rag_loader(
    rag_df,
    download_path,
    batch_size,
    num_workers,
    tokenizer,
    context_length,
    transforms,
    rag_keep_frac: float = 0.25,
    seed: int = 42,
):
    """
    Build a RAG dataloader with optional downsampling by unique study_id.

    Behavior:
      - rag_keep_frac >= 1.0  → keep FULL rag_df (no downsampling)
      - 0 < rag_keep_frac < 1 → keep that fraction of unique study_id (seeded shuffle)
      - otherwise             → error

    Note: Downsampling is by study_id (subjects may be partially retained inside RAG).
    """
    if rag_keep_frac <= 0:
        raise ValueError("rag_keep_frac must be > 0. Use >= 1.0 to keep the full RAG split.")

    df = rag_df
    if rag_keep_frac < 1.0:
        uniq = list(df["study_id"].unique())
        n = len(uniq)
        if n == 0:
            print("⚠️ graphrag split is empty; cannot downsample.")
        else:
            rng = random.Random(seed)
            uniq_sorted = sorted(uniq)  # stable base order for reproducibility
            rng.shuffle(uniq_sorted)
            n_keep = max(1, int(round(n * rag_keep_frac)))
            keep_ids = set(uniq_sorted[:n_keep])
            df = df[df["study_id"].isin(keep_ids)].reset_index(drop=True)
            print(f"🟡 Using {rag_keep_frac:.0%} of graphrag by study_id: {len(df)} rows (of original {len(rag_df)}).")
    else:
        print(f"🟢 Using FULL graphrag split (rag_keep_frac={rag_keep_frac}).")

    dataset = MimicUnifiedDataset(
        merged_df=df,
        download_path=download_path,
        tokenizer=tokenizer,
        context_length=context_length,
        transforms=transforms,
    )
    if len(dataset) == 0:
        raise ValueError("❌ RAG dataset is empty after loading! Check dataframe and file paths.")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    print(f"✅ RAG dataset size: {len(dataset)} (rag_keep_frac={rag_keep_frac})")
    return loader


def get_test_loader(test_df, download_path, batch_size, num_workers, tokenizer, context_length, transforms):
    dataset = MimicUnifiedDataset(
        merged_df=test_df,
        download_path=download_path,
        tokenizer=tokenizer,
        context_length=context_length,
        transforms=transforms,
    )
    if len(dataset) == 0:
        raise ValueError("❌ Test dataset is empty after loading! Check dataframe and file paths.")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    print(f"✅ Test dataset size: {len(dataset)}")
    return loader
