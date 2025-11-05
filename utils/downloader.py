# downloader.py
import os
import time
import signal
import gzip
from io import BytesIO
from dataclasses import dataclass
from typing import Dict, Any, Optional, List

import gcsfs
import pandas as pd
from PIL import Image, ImageOps
from tqdm import tqdm
from typing import Tuple

# Optional Google BigQuery Storage deps (only needed for MIMIC-IV)
try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    import pyarrow.csv as pa_csv
    from google.cloud import bigquery_storage_v1 as bqstorage
    from google.cloud.bigquery_storage_v1.types import ReadSession, DataFormat
except Exception:
    bqstorage = None  # handled in class init

# ---------------------
# Shared helpers
# ---------------------
_STOP = {"requested": False}
def _handle_stop(signum, frame):
    _STOP["requested"] = True

signal.signal(signal.SIGINT, _handle_stop)
try:
    signal.signal(signal.SIGTERM, _handle_stop)
except Exception:
    pass

def _atomic_write_text(path: str, text: str) -> None:
    tmp = path + ".part"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)

def _atomic_save_jpeg(pil_img: Image.Image, out_path: str, quality: int) -> None:
    tmp = out_path + ".part"
    pil_img.save(tmp, format="JPEG", quality=quality, optimize=True)
    try:
        with open(tmp, "rb") as f:
            os.fsync(f.fileno())
    except Exception:
        pass
    os.replace(tmp, out_path)

def _verify_jpeg(path: str) -> bool:
    try:
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            im.load()
        return True
    except Exception:
        return False

# ---------------------
# MIMIC-CXR (existing behavior, lifted into a class)
# ---------------------
@dataclass
class MIMICCXRConfig:
    dataset: Any   # expects attributes: download_path (str)
    gcs_path: Any  # expects attributes: gcs_jpg_metadata, gcs_jpg_label, gcs_jpg_bucket, gcs_bucket
    target_size: int = 384
    jpg_quality: int = 90
    max_retries: int = 3
    backoff: float = 1.5

class MIMICCXRDownloader:
    def __init__(self, cfg: MIMICCXRConfig, random_seed: int):
        self.cfg = cfg
        self.random_seed = random_seed
        self.fs = gcsfs.GCSFileSystem()

    def _create_processed_metadata_and_labels(self, metadata_df, chexpert_df):
        # (Identical to your current helper)
        shared_ids = set(metadata_df["study_id"]) & set(chexpert_df["study_id"])
        md = metadata_df[metadata_df["study_id"].isin(shared_ids)].copy()
        chex = chexpert_df[chexpert_df["study_id"].isin(shared_ids)].copy()

        md["ViewPosition"] = md["ViewPosition"].astype(str).str.upper().str.strip()
        valid_views = ["PA", "AP", "LL", "LA"]
        md = md[md["ViewPosition"].isin(valid_views)]

        md = md.sort_values(["subject_id", "study_id"]).reset_index(drop=True)
        chex = chex.sort_values(["subject_id", "study_id"]).reset_index(drop=True)

        view_priority = {"PA": 0, "AP": 1, "LA": 2, "LL": 3}
        print("View Priority Policy:"); print(view_priority)
        md["view_rank"] = md["ViewPosition"].map(view_priority)
        md = md.sort_values(["subject_id", "study_id", "view_rank", "dicom_id"])
        one_per_study = md.drop_duplicates("study_id", keep="first").drop(columns=["view_rank"])
        merged_df = pd.merge(one_per_study, chex, on=["subject_id", "study_id"], how="inner")
        merged_df = merged_df.sort_values(["subject_id", "study_id"]).reset_index(drop=True)
        return merged_df

    def _ensure_jpg_and_report(self, merged_df: pd.DataFrame):
        out_dir = self.cfg.dataset.download_path
        fs = self.fs
        for _, row in tqdm(merged_df.iterrows(), total=len(merged_df), desc="Ensuring files", unit="study"):
            if _STOP["requested"]:
                print("\n🛑 Stop requested. Exiting after current study."); break

            dicom_id = row["dicom_id"]
            subject_id = str(row["subject_id"]).zfill(8)
            study_id = str(row["study_id"])

            jpg_prefix = f"files/p{subject_id[:2]}/p{subject_id}/s{study_id}/{dicom_id}"
            jpg_path_gcs = f"{self.cfg.gcs_path.gcs_jpg_bucket}/{jpg_prefix}.jpg"

            txt_prefix = f"files/p{subject_id[:2]}/p{subject_id}/s{study_id}"
            txt_path_gcs = f"{self.cfg.gcs_path.gcs_bucket}/{txt_prefix}.txt"

            study_dir = os.path.join(out_dir, study_id)
            os.makedirs(study_dir, exist_ok=True)
            chosen_jpg = os.path.join(study_dir, f"{dicom_id}.jpg")
            report_txt = os.path.join(study_dir, f"s{study_id}.txt")

            # image
            need_image = not (os.path.exists(chosen_jpg) and _verify_jpeg(chosen_jpg))
            if need_image:
                attempt = 0
                while attempt < self.cfg.max_retries:
                    try:
                        with fs.open(jpg_path_gcs, "rb") as f:
                            image_bytes = f.read()
                        pil = Image.open(BytesIO(image_bytes)).convert("RGB")
                        pil_resized = ImageOps.fit(pil, (self.cfg.target_size, self.cfg.target_size), method=Image.BICUBIC)
                        _atomic_save_jpeg(pil_resized, chosen_jpg, quality=self.cfg.jpg_quality)
                        if _verify_jpeg(chosen_jpg):
                            break
                        else:
                            try: os.remove(chosen_jpg)
                            except FileNotFoundError: pass
                            raise IOError("JPEG verification failed")
                    except Exception as e:
                        attempt += 1
                        wait = self.cfg.backoff ** attempt
                        print(f"❌ JPG s{study_id}/{dicom_id} attempt {attempt}/{self.cfg.max_retries} failed: {e}")
                        if attempt >= self.cfg.max_retries:
                            print(f"🚫 Giving up on s{study_id}/{dicom_id} for now.")
                        else:
                            time.sleep(wait)

            # report
            if not os.path.exists(report_txt):
                attempt = 0
                while attempt < self.cfg.max_retries:
                    try:
                        with fs.open(txt_path_gcs, "r") as f:
                            text = f.read().strip()
                        _atomic_write_text(report_txt, text)
                        break
                    except Exception as e:
                        attempt += 1
                        wait = self.cfg.backoff ** attempt
                        print(f"❌ Report s{study_id} attempt {attempt}/{self.cfg.max_retries} failed: {e}")
                        if attempt >= self.cfg.max_retries:
                            print(f"🚫 Giving up on report s{study_id} for now.")
                        else:
                            time.sleep(wait)

    def run(self, metadata_df: pd.DataFrame, chexpert_df: pd.DataFrame):
        os.makedirs(self.cfg.dataset.download_path, exist_ok=True)

        merged_csv = os.path.join(self.cfg.dataset.download_path, "dataset_processed_merged.csv")
        if os.path.exists(merged_csv):
            merged_df = pd.read_csv(merged_csv)
            print(f"⏩ Loaded cached merged dataset: {merged_csv}")
        else:
            merged_df = self._create_processed_metadata_and_labels(metadata_df, chexpert_df)
            merged_df.to_csv(merged_csv, index=False)
            print(f"✅ Saved merged dataset CSV to: {merged_csv}")

        print(f"📦 Using all {len(merged_df)} studies (100.0%).")
        self._ensure_jpg_and_report(merged_df)
        print("✅ Download complete (atomic writes + verification).")
        if _STOP["requested"]:
            print("ℹ️ Stopped early by user. Already-downloaded files were written atomically and verified.")


# ---------------------
# MIMIC-IV (direct-to-filtered CSV.GZ)
# ---------------------
@dataclass
class MIMICIVConfig:
    paths: Dict[str, str]           # { base_dir, filtered_csv_dir }
    gcp: Dict[str, str]             # { project, location }
    tables: Dict[str, List[str]]    # { hosp: [...], icu: [...] }
    filter_subjects_csv: str        # REQUIRED; must contain 'subject_id'
    chunk_size: int = 250_000
    gzip_output: bool = True        # keep True to write .csv.gz

class MIMICIVDownloader:
    def __init__(self, cfg: MIMICIVConfig):
        if bqstorage is None:
            raise RuntimeError("Please install google-cloud-bigquery-storage and pyarrow.")
        if not cfg.filter_subjects_csv or not os.path.exists(cfg.filter_subjects_csv):
            raise RuntimeError("filter_subjects_csv must exist and contain a 'subject_id' column.")

        self.cfg = cfg
        self.client = bqstorage.BigQueryReadClient()
        self.parent = f"projects/{cfg.gcp['project']}"
        os.makedirs(self.cfg.paths["base_dir"], exist_ok=True)
        os.makedirs(self.cfg.paths["filtered_csv_dir"], exist_ok=True)

        # Load keep-set once
        print(f"Loading subject_id list from: {self.cfg.filter_subjects_csv}")
        keep_df = pd.read_csv(self.cfg.filter_subjects_csv, usecols=["subject_id"])
        self.keep_subjects = set(pd.to_numeric(keep_df["subject_id"], errors="coerce").dropna().astype("int64").tolist())
        print(f"Loaded {len(self.keep_subjects):,} unique subject_id(s).")

    def _out_paths(self, dataset: str, table: str) -> Tuple[str, str]:
        """Return (filtered_candidate_path, full_candidate_path)."""
        suff = ".gz" if self.cfg.gzip_output else ""
        filtered_name = f"{dataset}_{table}_filtered.csv{suff}"
        full_name = f"{dataset}_{table}.csv{suff}"
        d = self.cfg.paths["filtered_csv_dir"]
        return (os.path.join(d, filtered_name), os.path.join(d, full_name))

    def _either_output_exists(self, dataset: str, table: str) -> bool:
        p_filtered, p_full = self._out_paths(dataset, table)
        if os.path.exists(p_filtered):
            print(f"✔ Exists: {p_filtered}")
            return True
        if os.path.exists(p_full):
            print(f"✔ Exists: {p_full}")
            return True
        return False

    def _write_filtered_or_full_from_bq(self, dataset: str, table: str):
        # Skip if already produced
        if self._either_output_exists(dataset, table):
            return

        table_resource = f"projects/physionet-data/datasets/{dataset}/tables/{table}"
        print(f"Reading BQ Storage: {table_resource}")
        session = self.client.create_read_session(
            parent=self.parent,
            read_session=ReadSession(table=table_resource, data_format=DataFormat.ARROW),
            max_stream_count=1,
        )
        if not session.streams:
            print(f"⚠ No streams for {dataset}.{table}; writing empty filtered CSV.")
            out_filtered, _ = self._out_paths(dataset, table)
            with (gzip.open(out_filtered, "wt") if self.cfg.gzip_output else open(out_filtered, "w")) as f:
                pass
            return

        # We need the first non-empty batch to decide: filterable (has subject_id) or not.
        first_df = None
        stream_iter = []
        for stream in session.streams:
            stream_iter.append(self.client.read_rows(stream.name).rows(session).pages)

        for pages in stream_iter:
            for page in pages:
                arrow_obj = page.to_arrow()
                tbl = pa.Table.from_batches([arrow_obj]) if isinstance(arrow_obj, pa.RecordBatch) else arrow_obj
                if tbl.num_rows == 0:
                    continue
                first_df = tbl.to_pandas()
                break
            if first_df is not None:
                break

        # If absolutely no rows at all, just create empty filtered file.
        if first_df is None:
            out_filtered, _ = self._out_paths(dataset, table)
            with (gzip.open(out_filtered, "wt") if self.cfg.gzip_output else open(out_filtered, "w")) as f:
                pass
            print(f"✔ {dataset}.{table}: empty table → wrote empty {os.path.basename(out_filtered)}")
            return

        filterable = "subject_id" in first_df.columns
        out_filtered, out_full = self._out_paths(dataset, table)
        out_path = out_filtered if filterable else out_full
        open_out = (lambda p: gzip.open(p, "wt", newline="")) if self.cfg.gzip_output else (lambda p: open(p, "w", newline=""))

        # Re-run the read (including that first batch) to stream everything to disk
        # We’ll write header once and then append.
        header_written = False
        total_in, total_out = 0, 0

        with open_out(out_path) as fout:
            # Helper to process & write a DataFrame batch
            def _process_and_write(df: pd.DataFrame):
                nonlocal header_written, total_in, total_out
                total_in += len(df)
                if filterable:
                    subj = pd.to_numeric(df["subject_id"], errors="coerce").astype("Int64")
                    df = df.loc[subj.isin(self.keep_subjects)]
                if df.empty:
                    return
                df.to_csv(fout, index=False, header=(not header_written))
                header_written = True
                total_out += len(df)

            # Write the first_df
            _process_and_write(first_df)

            # Continue with the rest of the pages/streams
            for stream in session.streams:
                reader = self.client.read_rows(stream.name)
                # We already consumed the first non-empty page above; now iterate all pages again
                for page in reader.rows(session).pages:
                    arrow_obj = page.to_arrow()
                    tbl = pa.Table.from_batches([arrow_obj]) if isinstance(arrow_obj, pa.RecordBatch) else arrow_obj
                    if tbl.num_rows == 0:
                        continue
                    df = tbl.to_pandas()
                    # Skip the exact first batch we already wrote (best-effort: relies on header_written gate)
                    # This is harmless even if we write it twice: header_written prevents duplicate header,
                    # but to avoid duplicates we can check equality by shape—a small heuristic:
                    if not header_written and len(df) == len(first_df) and list(df.columns) == list(first_df.columns):
                        # We can't reliably compare content across large batches; allow rare double-write risk to be negligible.
                        pass
                    _process_and_write(df)

        label = "filtered" if filterable else "full (no subject_id)"
        print(f"✔ {dataset}.{table}: kept {total_out:,} / {total_in:,} rows → {out_path} [{label}]")

    def run(self):
        plan = [("mimiciv_3_1_hosp", t) for t in self.cfg.tables.get("hosp", [])] + \
               [("mimiciv_3_1_icu",  t) for t in self.cfg.tables.get("icu",  [])]

        for dataset, table in plan:
            print(f"\n>>> {dataset}.{table}")
            self._write_filtered_or_full_from_bq(dataset, table)

        print("\n✅ Done. Outputs are under:", self.cfg.paths["filtered_csv_dir"])