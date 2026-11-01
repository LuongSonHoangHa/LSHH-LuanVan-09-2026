"""
merge_data.py — Merge separate training splits into a single unified train set.

Takes:
  1. data/processed/shopee_train.csv (80% Shopee train split from data_split.py)
  2. data/processed/vietnamese_train.csv (80% Vietnamese train split from data_split.py)

Outputs:
  - data/processed/merge_train.csv (Combined 80% Shopee + 80% Vietnamese train set)

Also cleans up intermediate train split files so only 3 files remain in data/processed:
  1. merge_train.csv
  2. shopee_test.csv
  3. vietnamese_test.csv

Usage:
    python src/data/merge_data.py
    python src/data/merge_data.py --keep-intermediates
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

DEFAULT_PROCESSED     = os.path.join(ROOT, "data", "processed")
DEFAULT_SHOPEE_TRAIN  = os.path.join(DEFAULT_PROCESSED, "shopee_train.csv")
DEFAULT_VN_TRAIN      = os.path.join(DEFAULT_PROCESSED, "vietnamese_train.csv")
DEFAULT_OUT           = os.path.join(DEFAULT_PROCESSED, "merge_train.csv")
DEFAULT_SHOPEE_TEST   = os.path.join(DEFAULT_PROCESSED, "shopee_test.csv")
DEFAULT_VN_TEST       = os.path.join(DEFAULT_PROCESSED, "vietnamese_test.csv")


def merge_train_sets(shopee_train_path: str = DEFAULT_SHOPEE_TRAIN,
                     vn_train_path: str = DEFAULT_VN_TRAIN,
                     out_path: str = DEFAULT_OUT,
                     shopee_test_path: str = DEFAULT_SHOPEE_TEST,
                     vn_test_path: str = DEFAULT_VN_TEST,
                     cleanup_intermediates: bool = True) -> pd.DataFrame:
    """
    Merges Shopee and Vietnamese train splits into a single training CSV.
    Validates column alignment and verifies zero group leakage against test sets.
    """
    print("=" * 65)
    print("  MERGING TRAINING SETS (Shopee Train + Vietnamese Train)")
    print("=" * 65)

    if not os.path.exists(shopee_train_path):
        raise FileNotFoundError(
            f"Shopee train split not found: {shopee_train_path}\n"
            "Please run `python src/data/data_split.py` first."
        )
    if not os.path.exists(vn_train_path):
        raise FileNotFoundError(
            f"Vietnamese train split not found: {vn_train_path}\n"
            "Please run `python src/data/data_split.py` first."
        )

    print(f"Reading Shopee Train     : {shopee_train_path}")
    df_s_train = pd.read_csv(shopee_train_path)
    print(f"  Rows: {len(df_s_train):,} | Groups: {df_s_train['label_group'].nunique():,}")

    print(f"\nReading Vietnamese Train : {vn_train_path}")
    df_v_train = pd.read_csv(vn_train_path)
    print(f"  Rows: {len(df_v_train):,} | Groups: {df_v_train['label_group'].nunique():,}")

    # Validate columns
    if list(df_s_train.columns) != list(df_v_train.columns):
        raise ValueError(
            f"Column mismatch!\n  Shopee : {list(df_s_train.columns)}\n  VN     : {list(df_v_train.columns)}"
        )

    # Check for duplicate group IDs between datasets
    train_overlap = set(df_s_train["label_group"]) & set(df_v_train["label_group"])
    if len(train_overlap) > 0:
        print(f"  Warning: {len(train_overlap)} overlapping label_groups between Shopee and VN")
    else:
        print("  Verified: Disjoint label groups between Shopee and VN train sets")

    # Merge
    df_merged = pd.concat([df_s_train, df_v_train], ignore_index=True)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df_merged.to_csv(out_path, index=False)

    print(f"\nSaved Merged Train CSV   -> {out_path}")
    print(f"  Total rows             : {len(df_merged):,}")
    print(f"  Total unique groups    : {df_merged['label_group'].nunique():,}")

    # Verification against test sets if they exist
    merged_groups = set(df_merged["label_group"])
    if os.path.exists(shopee_test_path):
        df_s_test = pd.read_csv(shopee_test_path)
        shopee_test_groups = set(df_s_test["label_group"])
        leakage_shopee = merged_groups & shopee_test_groups
        assert len(leakage_shopee) == 0, f"DATA LEAKAGE: {len(leakage_shopee)} Shopee test groups in Merged Train!"
        print(f"  Verified zero leakage with {os.path.basename(shopee_test_path)}")

    if os.path.exists(vn_test_path):
        df_v_test = pd.read_csv(vn_test_path)
        vn_test_groups = set(df_v_test["label_group"])
        leakage_vn = merged_groups & vn_test_groups
        assert len(leakage_vn) == 0, f"DATA LEAKAGE: {len(leakage_vn)} VN test groups in Merged Train!"
        print(f"  Verified zero leakage with {os.path.basename(vn_test_path)}")

    # Clean up intermediate files if requested
    if cleanup_intermediates:
        cleaned = []
        for path in [shopee_train_path, vn_train_path]:
            if os.path.exists(path) and os.path.abspath(path) != os.path.abspath(out_path):
                try:
                    os.remove(path)
                    cleaned.append(os.path.basename(path))
                except OSError as e:
                    print(f"  Notice: Could not delete intermediate {path}: {e}")
        if cleaned:
            print(f"\nCleaned up intermediate files: {', '.join(cleaned)}")

    # Summary of final files in destination folder
    dest_dir = os.path.dirname(out_path)
    print("\n" + "=" * 65)
    print(f"  PROCESSED DIRECTORY STATUS ({dest_dir}):")
    print("=" * 65)
    if os.path.exists(dest_dir):
        for f in sorted(os.listdir(dest_dir)):
            fp = os.path.join(dest_dir, f)
            if os.path.isfile(fp):
                size_kb = os.path.getsize(fp) / 1024
                print(f"  - {f:<25} ({size_kb:>8.1f} KB)")
    print("=" * 65 + "\n")

    return df_merged


# Backward-compatibility alias
merge_csv = merge_train_sets


def main():
    parser = argparse.ArgumentParser(
        description="Merge Shopee and Vietnamese train splits into a unified train set."
    )
    parser.add_argument(
        "--shopee-train", "--shopee",
        dest="shopee_train",
        default=DEFAULT_SHOPEE_TRAIN,
        help="Path to Shopee train CSV (default: data/processed/shopee_train.csv)"
    )
    parser.add_argument(
        "--vn-train", "--vietnamese",
        dest="vn_train",
        default=DEFAULT_VN_TRAIN,
        help="Path to Vietnamese train CSV (default: data/processed/vietnamese_train.csv)"
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUT,
        help="Path to merged train output CSV (default: data/processed/merge_train.csv)"
    )
    parser.add_argument(
        "--shopee-test",
        default=DEFAULT_SHOPEE_TEST,
        help="Path to Shopee test CSV for leakage verification"
    )
    parser.add_argument(
        "--vn-test",
        default=DEFAULT_VN_TEST,
        help="Path to Vietnamese test CSV for leakage verification"
    )
    parser.add_argument(
        "--keep-intermediates",
        action="store_true",
        help="Do not delete shopee_train.csv and vietnamese_train.csv after merging"
    )
    args = parser.parse_args()

    # Resolve relative paths against ROOT
    s_train = args.shopee_train if os.path.isabs(args.shopee_train) else os.path.join(ROOT, args.shopee_train)
    v_train = args.vn_train if os.path.isabs(args.vn_train) else os.path.join(ROOT, args.vn_train)
    out_csv = args.output if os.path.isabs(args.output) else os.path.join(ROOT, args.output)
    s_test  = args.shopee_test if os.path.isabs(args.shopee_test) else os.path.join(ROOT, args.shopee_test)
    v_test  = args.vn_test if os.path.isabs(args.vn_test) else os.path.join(ROOT, args.vn_test)

    merge_train_sets(
        shopee_train_path=s_train,
        vn_train_path=v_train,
        out_path=out_csv,
        shopee_test_path=s_test,
        vn_test_path=v_test,
        cleanup_intermediates=not args.keep_intermediates,
    )


if __name__ == "__main__":
    main()
