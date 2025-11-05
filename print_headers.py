# print_csv_heads.py
import os
import glob
import pandas as pd


def print_csv_heads(base_dir="./datasets/mimic_iv/csv_filtered", nrows=5):
    """
    Print the head of every CSV/CSV.GZ file under base_dir.

    Args:
        base_dir (str): Directory containing CSV/CSV.GZ files.
        nrows (int): Number of rows from head() to print.
    """
    if not os.path.exists(base_dir):
        print(f"❌ Base directory not found: {base_dir}")
        return

    files = sorted(glob.glob(os.path.join(base_dir, "*.csv")) +
                   glob.glob(os.path.join(base_dir, "*.csv.gz")))
    if not files:
        print(f"⚠️ No CSV files found in {base_dir}")
        return

    for f in files:
        try:
            print(f"\n=== {os.path.basename(f)} ===")
            df = pd.read_csv(f, nrows=nrows)
            print(df.head(nrows))
        except Exception as e:
            print(f"❌ Could not read {f}: {e}")


if __name__ == "__main__":
    # Default: show first 5 rows of each CSV
    print_csv_heads(nrows=5)
