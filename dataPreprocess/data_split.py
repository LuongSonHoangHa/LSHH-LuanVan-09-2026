"""
data_split.py — Group-aware 80/20 train/test split and merge pipeline.

Splits datasets on `label_group` (not individual rows), ensuring no group leakage
between train and test splits, stratified by group size bucket.

Workflow:
    1. Input raw files:
       - data/raw/Shopee.csv
       - data/raw/vietnamese.csv
       - data/raw/shopee_no_indo.csv
    2. Split each into 80% train and 20% test (6 splits):
       - Shopee:          shopee_train.csv          & shopee_test.csv
       - Vietnamese:      vietnamese_train.csv      & vietnamese_test.csv
       - Shopee No-Indo:  shopee_no_indo_train.csv  & shopee_no_indo_test.csv
    3. Merge train sets in data/processed/:
       - Shopee Train + Vietnamese Train         -> merge_train.csv
       - Shopee No-Indo Train + Vietnamese Train -> merge_train_no_indo.csv
    4. Clean up individual train files so 5 final files remain:
       1. vietnamese_test.csv
       2. shopee_test.csv
       3. shopee_no_indo_test.csv
       4. merge_train.csv
       5. merge_train_no_indo.csv
    5. Comprehensive zero-leakage verification across all sets.

Usage:
    python src/dataPreprocess/data_split.py
"""

import os
import sys
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))               # src/dataPreprocess/
ROOT  = os.path.dirname(os.path.dirname(_HERE))                 # project root

DEFAULT_SHOPEE_RAW         = os.path.join(ROOT, "data", "raw", "Shopee.csv")
DEFAULT_VN_RAW             = os.path.join(ROOT, "data", "raw", "vietnamese.csv")
DEFAULT_SHOPEE_NO_INDO_RAW = os.path.join(ROOT, "data", "raw", "shopee_no_indo.csv")
DEFAULT_PROCESSED          = os.path.join(ROOT, "data", "processed")

# Output files in processed/
DEFAULT_SHOPEE_TEST         = os.path.join(DEFAULT_PROCESSED, "shopee_test.csv")
DEFAULT_VN_TEST             = os.path.join(DEFAULT_PROCESSED, "vietnamese_test.csv")
DEFAULT_SHOPEE_NO_INDO_TEST = os.path.join(DEFAULT_PROCESSED, "shopee_no_indo_test.csv")
DEFAULT_MERGE_TRAIN         = os.path.join(DEFAULT_PROCESSED, "merge_train.csv")
DEFAULT_MERGE_TRAIN_NO_INDO = os.path.join(DEFAULT_PROCESSED, "merge_train_no_indo.csv")


def split_data(csv_or_df, train_out: str = None, test_out: str = None,
               test_size: float = 0.20, seed: int = 42, dataset_name: str = "Dataset") -> tuple:
    """Group-aware 80/20 split on label_group with size bucket stratification."""
    print("=" * 65)
    print(f"  GROUP-AWARE TRAIN-TEST SPLIT: {dataset_name}")
    print("=" * 65)

    if isinstance(csv_or_df, str):
        print(f"Input : {csv_or_df}")
        if not os.path.exists(csv_or_df):
            raise FileNotFoundError(f"Input file not found: {csv_or_df}")
        df = pd.read_csv(csv_or_df)
    else:
        df = csv_or_df.copy()

    # Drop rows that are missing critical fields if any
    initial_len = len(df)
    df = df.dropna(subset=["image", "title", "label_group"]).reset_index(drop=True)
    if "posting_id" in df.columns:
        df["posting_id"] = df["posting_id"].astype(int)
    if "label_group" in df.columns:
        df["label_group"] = df["label_group"].astype(int)

    if len(df) < initial_len:
        print(f"  Note: Dropped {initial_len - len(df)} rows with missing fields")

    print(f"Loaded {len(df):,} rows | {df['label_group'].nunique():,} unique groups")

    # Group size summary
    group_info = (
        df.groupby("label_group")
        .size()
        .reset_index(name="group_size")
    )

    # Stratify by group-size bucket so both splits get representative distribution
    group_info["size_bucket"] = pd.cut(
        group_info["group_size"],
        bins=[0, 1, 2, 3, 5, 10, 9999],
        labels=["1", "2", "3", "4-5", "6-10", "11+"]
    )
    group_info["size_bucket"] = group_info["size_bucket"].cat.remove_unused_categories()

    # Fallback stratification if any bucket has fewer than 2 elements
    bucket_counts = group_info["size_bucket"].value_counts()
    rare_buckets = bucket_counts[bucket_counts < 2].index
    if len(rare_buckets) > 0:
        group_info["stratify_col"] = group_info["size_bucket"].astype(str)
        group_info.loc[group_info["size_bucket"].isin(rare_buckets), "stratify_col"] = "rare"
        strat_col = group_info["stratify_col"]
    else:
        strat_col = group_info["size_bucket"]

    print("\nGroup-size bucket distribution:")
    print(group_info["size_bucket"].value_counts().sort_index())

    # Split groups (preventing leakage)
    train_groups, test_groups = train_test_split(
        group_info["label_group"],
        test_size=test_size,
        random_state=seed,
        stratify=strat_col,
    )

    train_groups = set(train_groups)
    test_groups  = set(test_groups)

    # Map back to listing rows
    df_train = df[df["label_group"].isin(train_groups)].reset_index(drop=True)
    df_test  = df[df["label_group"].isin(test_groups)].reset_index(drop=True)

    print(f"\n{'':=<55}")
    print(f"  Train : {len(df_train):>6,} rows | {len(train_groups):>5,} groups "
          f"({len(train_groups)/len(group_info)*100:.1f}%)")
    print(f"  Test  : {len(df_test):>6,} rows | {len(test_groups):>5,} groups "
          f"({len(test_groups)/len(group_info)*100:.1f}%)")
    print(f"{'':=<55}")

    # Integrity assertions
    overlap = train_groups & test_groups
    assert len(overlap) == 0, f"DATA LEAKAGE: {len(overlap)} groups appear in both sets!"
    print("  Verified: No group leakage between train and test")

    if train_out:
        os.makedirs(os.path.dirname(train_out), exist_ok=True)
        df_train.to_csv(train_out, index=False)
        print(f"Saved Train CSV -> {train_out}")

    if test_out:
        os.makedirs(os.path.dirname(test_out), exist_ok=True)
        df_test.to_csv(test_out, index=False)
        print(f"Saved Test CSV  -> {test_out}")

    return df_train, df_test


def run_split_pipeline(shopee_raw: str = DEFAULT_SHOPEE_RAW,
                       vn_raw: str = DEFAULT_VN_RAW,
                       shopee_no_indo_raw: str = DEFAULT_SHOPEE_NO_INDO_RAW,
                       out_dir: str = DEFAULT_PROCESSED,
                       test_size: float = 0.20,
                       seed: int = 42,
                       keep_intermediates: bool = False):
    """
    Executes full data split and merge pipeline:
      1. Splits Shopee.csv (80/20) -> shopee_train.csv & shopee_test.csv.
      2. Splits vietnamese.csv (80/20) -> vietnamese_train.csv & vietnamese_test.csv.
      3. Splits shopee_no_indo.csv (80/20) -> shopee_no_indo_train.csv & shopee_no_indo_test.csv.
      4. Merges Shopee Train + Vietnamese Train -> merge_train.csv.
      5. Merges Shopee No-Indo Train + Vietnamese Train -> merge_train_no_indo.csv.
      6. Deletes standalone train sets (shopee, vietnamese, shopee_no_indo) unless keep_intermediates is True.
      7. Asserts zero group leakage across all train and test sets.
    """
    os.makedirs(out_dir, exist_ok=True)

    # Intermediate train split paths
    shopee_train_out = os.path.join(out_dir, "shopee_train.csv")
    vn_train_out     = os.path.join(out_dir, "vietnamese_train.csv")
    ni_train_out     = os.path.join(out_dir, "shopee_no_indo_train.csv")

    # Final 5 outputs
    shopee_test_out         = os.path.join(out_dir, "shopee_test.csv")
    vn_test_out             = os.path.join(out_dir, "vietnamese_test.csv")
    ni_test_out             = os.path.join(out_dir, "shopee_no_indo_test.csv")
    merge_train_out         = os.path.join(out_dir, "merge_train.csv")
    merge_train_no_indo_out = os.path.join(out_dir, "merge_train_no_indo.csv")

    print("\n" + "=" * 65)
    print("  STEP 1: 80/20 SPLIT FOR SHOPEE, VIETNAMESE & SHOPEE NO-INDO")
    print(f"  Shopee raw         : {shopee_raw}")
    print(f"  VN raw             : {vn_raw}")
    print(f"  Shopee No-Indo raw : {shopee_no_indo_raw}")
    print(f"  Output directory   : {out_dir}")
    print("=" * 65 + "\n")

    # 1. Split Shopee
    df_s_train, df_s_test = split_data(
        csv_or_df=shopee_raw,
        train_out=shopee_train_out,
        test_out=shopee_test_out,
        test_size=test_size,
        seed=seed,
        dataset_name="Shopee"
    )

    # 2. Split Vietnamese
    print()
    df_v_train, df_v_test = split_data(
        csv_or_df=vn_raw,
        train_out=vn_train_out,
        test_out=vn_test_out,
        test_size=test_size,
        seed=seed,
        dataset_name="Vietnamese"
    )

    # 3. Split Shopee No-Indo
    print()
    df_ni_train, df_ni_test = split_data(
        csv_or_df=shopee_no_indo_raw,
        train_out=ni_train_out,
        test_out=ni_test_out,
        test_size=test_size,
        seed=seed,
        dataset_name="Shopee No-Indo"
    )

    print("\n" + "=" * 65)
    print("  STEP 2: MERGING TRAINING SETS")
    print("=" * 65)

    # Validate column schemas match
    expected_cols = list(df_s_train.columns)
    assert list(df_v_train.columns) == expected_cols, "Column mismatch: Shopee vs Vietnamese!"
    assert list(df_ni_train.columns) == expected_cols, "Column mismatch: Shopee vs Shopee No-Indo!"

    # 4. Merge Shopee Train + Vietnamese Train
    print("\nMerging Shopee Train + Vietnamese Train...")
    df_merge_train = pd.concat([df_s_train, df_v_train], ignore_index=True)
    df_merge_train.to_csv(merge_train_out, index=False)
    print(f"  -> Saved: {merge_train_out}")
    print(f"     Rows: {len(df_merge_train):,} | Unique groups: {df_merge_train['label_group'].nunique():,}")

    # 5. Merge Shopee No-Indo Train + Vietnamese Train
    print("\nMerging Shopee No-Indo Train + Vietnamese Train...")
    df_merge_ni_train = pd.concat([df_ni_train, df_v_train], ignore_index=True)
    df_merge_ni_train.to_csv(merge_train_no_indo_out, index=False)
    print(f"  -> Saved: {merge_train_no_indo_out}")
    print(f"     Rows: {len(df_merge_ni_train):,} | Unique groups: {df_merge_ni_train['label_group'].nunique():,}")

    # 6. Delete intermediate train sets
    if not keep_intermediates:
        print("\n" + "=" * 65)
        print("  STEP 3: CLEANING UP INTERMEDIATE TRAIN SPLIT FILES")
        print("=" * 65)
        for path in [shopee_train_out, vn_train_out, ni_train_out]:
            if os.path.exists(path):
                os.remove(path)
                print(f"  Deleted intermediate file: {os.path.basename(path)}")
    else:
        print("\nIntermediate train split files retained (--keep-intermediates was passed).")

    # 7. Verification of zero leakage across all combinations
    print("\n" + "=" * 65)
    print("  STEP 4: INTEGRITY & ZERO-LEAKAGE VERIFICATION")
    print("=" * 65)

    s_test_groups       = set(df_s_test["label_group"])
    v_test_groups       = set(df_v_test["label_group"])
    ni_test_groups      = set(df_ni_test["label_group"])
    merge_train_groups  = set(df_merge_train["label_group"])
    merge_ni_train_groups = set(df_merge_ni_train["label_group"])

    # Within-dataset leak checks
    assert len(set(df_s_train["label_group"]) & s_test_groups) == 0, "Leakage in Shopee!"
    assert len(set(df_v_train["label_group"]) & v_test_groups) == 0, "Leakage in Vietnamese!"
    assert len(set(df_ni_train["label_group"]) & ni_test_groups) == 0, "Leakage in Shopee No-Indo!"

    # Merged train vs test checks
    assert len(merge_train_groups & s_test_groups) == 0, "Leakage: Shopee test groups in merge_train!"
    assert len(merge_train_groups & v_test_groups) == 0, "Leakage: Vietnamese test groups in merge_train!"
    assert len(merge_ni_train_groups & ni_test_groups) == 0, "Leakage: Shopee No-Indo test groups in merge_train_no_indo!"
    assert len(merge_ni_train_groups & v_test_groups) == 0, "Leakage: Vietnamese test groups in merge_train_no_indo!"

    # Cross dataset leak checks
    assert len(set(df_s_train["label_group"]) & v_test_groups) == 0, "Cross-leakage: Shopee train & VN test!"
    assert len(set(df_v_train["label_group"]) & s_test_groups) == 0, "Cross-leakage: VN train & Shopee test!"
    assert len(set(df_ni_train["label_group"]) & v_test_groups) == 0, "Cross-leakage: Shopee No-Indo train & VN test!"

    print("  [PASS] Zero group leakage verified across all splits and merged training sets!")

    # 8. Summary of final files in data/processed/
    print("\n" + "=" * 65)
    print(f"  FINAL PROCESSED DIRECTORY STATUS ({out_dir}):")
    print("=" * 65)
    final_files = [
        ("vietnamese_test.csv", vn_test_out),
        ("shopee_test.csv", shopee_test_out),
        ("shopee_no_indo_test.csv", ni_test_out),
        ("merge_train.csv", merge_train_out),
        ("merge_train_no_indo.csv", merge_train_no_indo_out),
    ]

    for fname, fpath in final_files:
        if os.path.exists(fpath):
            df_curr = pd.read_csv(fpath)
            size_kb = os.path.getsize(fpath) / 1024
            print(f"  - {fname:<26} : {len(df_curr):>6,} rows | {df_curr['label_group'].nunique():>5,} groups | {size_kb:>8.1f} KB")
        else:
            print(f"  - {fname:<26} : MISSING")

    # List all files currently in output directory to confirm count
    actual_files = [f for f in sorted(os.listdir(out_dir)) if os.path.isfile(os.path.join(out_dir, f))]
    print(f"\n  Total files in {out_dir}: {len(actual_files)}")
    print("=" * 65 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Group-aware 80/20 train/test split and merge pipeline for Shopee, Vietnamese, and Shopee No-Indo."
    )
    parser.add_argument("--shopee-raw",         default=DEFAULT_SHOPEE_RAW,         help="Raw Shopee CSV path")
    parser.add_argument("--vn-raw",             default=DEFAULT_VN_RAW,             help="Raw Vietnamese CSV path")
    parser.add_argument("--shopee-no-indo-raw", default=DEFAULT_SHOPEE_NO_INDO_RAW, help="Raw Shopee No-Indo CSV path")
    parser.add_argument("--out-dir",            default=DEFAULT_PROCESSED,          help="Processed output directory")
    parser.add_argument("--test-size",          type=float, default=0.20,           help="Test fraction (default: 0.20)")
    parser.add_argument("--seed",               type=int,   default=42,             help="Random seed (default: 42)")
    parser.add_argument("--keep-intermediates", action="store_true",                help="Retain intermediate single-dataset train CSVs")

    # Custom single CSV options
    parser.add_argument("--input",     default=None, help="Custom single input CSV")
    parser.add_argument("--train-out", default=None, help="Train output for custom input")
    parser.add_argument("--test-out",  default=None, help="Test output for custom input")

    args = parser.parse_args()

    if args.input is not None:
        in_path    = args.input if os.path.isabs(args.input) else os.path.join(ROOT, args.input)
        train_path = (args.train_out if os.path.isabs(args.train_out) else os.path.join(ROOT, args.train_out)) if args.train_out else None
        test_path  = (args.test_out if os.path.isabs(args.test_out) else os.path.join(ROOT, args.test_out)) if args.test_out else None
        split_data(in_path, train_out=train_path, test_out=test_path, test_size=args.test_size, seed=args.seed)
    else:
        s_raw   = args.shopee_raw if os.path.isabs(args.shopee_raw) else os.path.join(ROOT, args.shopee_raw)
        v_raw   = args.vn_raw if os.path.isabs(args.vn_raw) else os.path.join(ROOT, args.vn_raw)
        ni_raw  = args.shopee_no_indo_raw if os.path.isabs(args.shopee_no_indo_raw) else os.path.join(ROOT, args.shopee_no_indo_raw)
        out_dir = args.out_dir if os.path.isabs(args.out_dir) else os.path.join(ROOT, args.out_dir)

        run_split_pipeline(
            shopee_raw=s_raw,
            vn_raw=v_raw,
            shopee_no_indo_raw=ni_raw,
            out_dir=out_dir,
            test_size=args.test_size,
            seed=args.seed,
            keep_intermediates=args.keep_intermediates,
        )


if __name__ == "__main__":
    main()
