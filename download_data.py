# download_data.py
import argparse
import os
import gcsfs
import pandas as pd
from config import load_config, random_seed

from utils import (
    MIMICCXRDownloader, MIMICCXRConfig,
    MIMICIVDownloader, MIMICIVConfig,
)

def run_mimic_cxr():
    data_cfg = load_config("mimic_cxr.yaml")
    fs = gcsfs.GCSFileSystem()
    os.makedirs(data_cfg.dataset.download_path, exist_ok=True)

    # Local caches
    metadata_save_path = os.path.join(data_cfg.dataset.download_path, "metadata.csv")
    chexpert_save_path = os.path.join(data_cfg.dataset.download_path, "chexpert_labels.csv")

    # 1) Metadata (gzip on GCS → CSV local)
    if os.path.exists(metadata_save_path):
        print(f"Metadata file already exists: {metadata_save_path}")
        metadata_df = pd.read_csv(metadata_save_path)
    else:
        with fs.open(data_cfg.gcs_path.gcs_jpg_metadata, "rb") as f:
            metadata_df = pd.read_csv(f, compression="gzip")
        metadata_df.to_csv(metadata_save_path, index=False)
        print(f"Saved metadata CSV to: {metadata_save_path}")

    # 2) CheXpert labels (clean -1 → "")
    if os.path.exists(chexpert_save_path):
        print(f"CheXpert label file already exists: {chexpert_save_path}")
        chexpert_df = pd.read_csv(chexpert_save_path)
    else:
        with fs.open(data_cfg.gcs_path.gcs_jpg_label, "rb") as f:
            chexpert_df = pd.read_csv(f, compression="gzip")
        chexpert_df = chexpert_df.replace(-1, "")
        chexpert_df.to_csv(chexpert_save_path, index=False)
        print(f"Saved cleaned CheXpert labels to: {chexpert_save_path}")

    # 3) Run class-based downloader (full sample)
    print("▶ Downloading with full sample (sample_split=1.0)")
    cxr_downloader = MIMICCXRDownloader(
        MIMICCXRConfig(dataset=data_cfg.dataset, gcs_path=data_cfg.gcs_path,
                       target_size=384, jpg_quality=90),
        random_seed=random_seed,
    )
    cxr_downloader.run(metadata_df=metadata_df, chexpert_df=chexpert_df)

def run_mimic_iv():
    iv_cfg = load_config("mimic_iv.yaml")
    paths = dict(
        base_dir=iv_cfg.paths.base_dir,                    # e.g., ./datasets/mimic_iv
        filtered_csv_dir=iv_cfg.paths.filtered_csv_dir,    # e.g., ./datasets/mimic_iv/csv_filtered
    )
    gcp = dict(project=iv_cfg.gcp.project, location=iv_cfg.gcp.location)
    tables = dict(hosp=list(iv_cfg.tables.hosp), icu=list(iv_cfg.tables.icu))

    dl = MIMICIVDownloader(MIMICIVConfig(
        paths=paths,
        gcp=gcp,
        tables=tables,
        filter_subjects_csv=iv_cfg.filter.subjects_csv,
        chunk_size=getattr(iv_cfg, "chunk_size", 250_000),
        gzip_output=True,
    ))
    dl.run()



def main():
    p = argparse.ArgumentParser(description="Dataset downloader")
    p.add_argument("dataset", choices=["mimic_cxr", "mimic_iv"], help="Which dataset to download")
    args = p.parse_args()

    if args.dataset == "mimic_cxr":
        run_mimic_cxr()
    elif args.dataset == "mimic_iv":
        run_mimic_iv()

if __name__ == "__main__":
    main()

# # MIMIC-CXR (unchanged behavior)
# python download_data.py mimic_cxr
#
# # MIMIC-IV (new)
# python download_data.py mimic_iv

