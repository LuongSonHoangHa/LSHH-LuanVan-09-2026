"""
group_numbering.py — Re-number posting_id and label_group sequentially.

Usage:
    # For Shopee-only experiment:
    python src/data/group_numbering.py --input data/raw/Shopee_group_sort.csv \
                                      --output data/processed/shopee_cleaned.csv

    # For Merged experiment (run merge_data.py first):
    python src/data/group_numbering.py --input data/processed/merged_raw.csv \
                                      --output data/processed/merged_cleaned.csv
"""

import os
import sys
import argparse
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))               # src/data/
ROOT  = os.path.dirname(os.path.dirname(_HERE))                 # project root

DEFAULT_IN  = os.path.join(ROOT, "data", "raw", "Shopee_group_sort.csv")
DEFAULT_OUT = os.path.join(ROOT, "data", "processed", "shopee_cleaned.csv")


def renumber_dataset(input_path: str, output_path: str, preview: int = 5):
    print("=" * 60)
    print("  RE-NUMBERING POSTING_ID AND LABEL_GROUP")
    print("=" * 60)
    print(f"Input : {input_path}")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    df = pd.read_csv(input_path)
    print(f"Shape : {df.shape}")
    print("\nSample (original):")
    print(df[["posting_id", "title", "label_group"]].head(preview).to_string(index=False))

    # 1. Sequential posting_id
    df["posting_id"] = range(1, len(df) + 1)

    # 2. Sequential label_group (preserves encounter order)
    unique_groups = df["label_group"].unique()
    group_map     = {orig: new for new, orig in enumerate(unique_groups, start=1)}
    df["label_group"] = df["label_group"].map(group_map)

    print("\nSample (after re-numbering):")
    print(df[["posting_id", "title", "label_group"]].head(preview).to_string(index=False))
    print(f"\nTotal rows   : {len(df):,}")
    print(f"Total groups : {df['label_group'].nunique():,}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nSaved -> {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Re-number posting_id and label_group.")
    parser.add_argument("--input",   default=DEFAULT_IN,  help="Input CSV path")
    parser.add_argument("--output",  default=DEFAULT_OUT, help="Output CSV path")
    parser.add_argument("--preview", type=int, default=5, help="Rows to preview")
    args = parser.parse_args()

    # Resolve relative paths against ROOT if not absolute
    in_path  = args.input if os.path.isabs(args.input) else os.path.join(ROOT, args.input)
    out_path = args.output if os.path.isabs(args.output) else os.path.join(ROOT, args.output)

    renumber_dataset(in_path, out_path, preview=args.preview)


if __name__ == "__main__":
    main()
