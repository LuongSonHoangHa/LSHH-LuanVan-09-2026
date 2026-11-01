"""
filter_indo.py — Filter out Indonesian language from Shopee dataset and generate no-indo splits.

Steps:
  1. Detect language of product titles in data/raw/Shopee_group_sort.csv using `langdetect`.
  2. Drop groups where >50% of the items are Indonesian (`lang == 'id'`).
  3. Save filtered dataset -> data/raw/shopee_no_indo.csv.
  4. Perform 80/20 group-aware train/test split on shopee_no_indo.csv:
     - 20% test -> data/processed/shopee_no_indo_test.csv
  5. Merge 80% Shopee (no-indo) train + 80% Vietnamese train:
     - Merged train -> data/processed/merge_train_no_indo.csv
  6. Assert zero group leakage across train and test sets.

Usage:
    python src/data/filter_indo.py
"""

import os
import sys
import time
import argparse
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
from langdetect import detect, DetectorFactory

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))               # src/data/
ROOT  = os.path.dirname(os.path.dirname(_HERE))                 # project root

DEFAULT_SHOPEE_RAW    = os.path.join(ROOT, "data", "raw", "Shopee_group_sort.csv")
DEFAULT_VN_RAW        = os.path.join(ROOT, "data", "raw", "vietnamese1200.csv")
DEFAULT_SHOPEE_NO_INDO = os.path.join(ROOT, "data", "raw", "shopee_no_indo.csv")
DEFAULT_PROCESSED     = os.path.join(ROOT, "data", "processed")

OUT_TEST_SHOPEE_NO_INDO = os.path.join(DEFAULT_PROCESSED, "shopee_no_indo_test.csv")
OUT_MERGE_TRAIN_NO_INDO = os.path.join(DEFAULT_PROCESSED, "merge_train_no_indo.csv")
EXISTING_VN_TEST        = os.path.join(DEFAULT_PROCESSED, "vietnamese_test.csv")


def _detect_chunk(titles):
    DetectorFactory.seed = 42
    results = []
    for t in titles:
        try:
            results.append(detect(t))
        except Exception:
            results.append("unknown")
    return results


def filter_shopee_no_indo(input_csv: str = DEFAULT_SHOPEE_RAW,
                          output_csv: str = DEFAULT_SHOPEE_NO_INDO,
                          threshold: float = 0.50,
                          chunk_size: int = 500,
                          force: bool = False) -> pd.DataFrame:
    """Detects languages and removes groups where >50% of titles are Indonesian."""
    print("=" * 65)
    print("  STEP 1: FILTERING INDONESIAN GROUPS FROM SHOPEE DATASET")
    print("=" * 65)

    if os.path.exists(output_csv) and not force:
        print(f"Filtered file already exists at: {output_csv}")
        print("Loading directly (pass --force to re-run language detection)...")
        df_filtered = pd.read_csv(output_csv)
        print(f"Loaded {len(df_filtered):,} rows | {df_filtered['label_group'].nunique():,} unique groups")
        print("=" * 65 + "\n")
        return df_filtered

    print(f"Reading input: {input_csv}")
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input file not found: {input_csv}")

    df = pd.read_csv(input_csv)
    initial_rows = len(df)
    initial_groups = df["label_group"].nunique()
    print(f"Loaded {initial_rows:,} rows across {initial_groups:,} unique groups")

    # Detect language using multiprocessing for speed
    titles = df["title"].tolist()
    chunks = [titles[i:i + chunk_size] for i in range(0, len(titles), chunk_size)]

    print(f"\nDetecting languages across {len(titles):,} titles using ProcessPoolExecutor...")
    t0 = time.time()
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(_detect_chunk, chunks))
    dt = time.time() - t0

    df["detected_lang"] = [item for sub in results for item in sub]
    df["is_indo"] = (df["detected_lang"] == "id")

    indo_count = df["is_indo"].sum()
    print(f"Language detection completed in {dt:.1f}s ({len(titles)/dt:.1f} titles/sec)")
    print(f"Total Indonesian titles detected: {indo_count:,} ({indo_count/len(df)*100:.1f}%)")

    print("\nTop 5 detected languages:")
    print(df["detected_lang"].value_counts().head(5).to_string())

    # Calculate per-group Indonesian ratio
    group_stats = df.groupby("label_group").agg(
        group_size=("is_indo", "count"),
        indo_count=("is_indo", "sum"),
    )
    group_stats["indo_ratio"] = group_stats["indo_count"] / group_stats["group_size"]

    # Filter out groups where > threshold are Indonesian
    groups_to_drop = group_stats[group_stats["indo_ratio"] > threshold].index
    groups_to_keep = group_stats[group_stats["indo_ratio"] <= threshold].index

    print("\n" + "-" * 55)
    print(f"Groups with >{int(threshold*100)}% Indonesian (DROPPED) : {len(groups_to_drop):,}")
    print(f"Groups with <={int(threshold*100)}% Indonesian (KEPT)   : {len(groups_to_keep):,}")
    print("-" * 55)

    df_filtered = df[df["label_group"].isin(groups_to_keep)].copy()
    # Retain the exact original raw columns
    original_cols = ["posting_id", "image", "image_phash", "title", "label_group"]
    df_filtered = df_filtered[original_cols].reset_index(drop=True)

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_filtered.to_csv(output_csv, index=False)

    print(f"\nSaved filtered dataset -> {output_csv}")
    print(f"  Rows remaining   : {len(df_filtered):,} (dropped {initial_rows - len(df_filtered):,})")
    print(f"  Groups remaining : {df_filtered['label_group'].nunique():,} (dropped {initial_groups - df_filtered['label_group'].nunique():,})")
    print("=" * 65 + "\n")

    return df_filtered


def create_no_indo_splits(shopee_no_indo_csv: str = DEFAULT_SHOPEE_NO_INDO,
                          vn_raw_csv: str = DEFAULT_VN_RAW,
                          test_out: str = OUT_TEST_SHOPEE_NO_INDO,
                          merge_train_out: str = OUT_MERGE_TRAIN_NO_INDO,
                          test_size: float = 0.20,
                          seed: int = 42):
    """
    Applies the same 80/20 split and merge rule to produce:
      - shopee_no_indo_test.csv
      - merge_train_no_indo.csv
    """
    # Import split_data
    if _HERE not in sys.path:
        sys.path.insert(0, _HERE)
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    try:
        from data_split import split_data
    except ImportError:
        from src.dataPreprocess.data_split import split_data

    print("=" * 65)
    print("  STEP 2: SPLITTING & MERGING NO-INDO DATASET")
    print("=" * 65)

    # 1. Split shopee_no_indo (80% train / 20% test)
    df_s_train, df_s_test = split_data(
        csv_or_df=shopee_no_indo_csv,
        train_out=None,
        test_out=test_out,
        test_size=test_size,
        seed=seed,
        dataset_name="Shopee No-Indo"
    )

    # 2. Split Vietnamese (80% train / 20% test) with same seed
    df_v_train, df_v_test = split_data(
        csv_or_df=vn_raw_csv,
        train_out=None,
        test_out=None,
        test_size=test_size,
        seed=seed,
        dataset_name="Vietnamese"
    )

    # 3. Merge train splits
    print("\n" + "-" * 55)
    print("  MERGING NO-INDO SHOPEE TRAIN + VIETNAMESE TRAIN")
    print("-" * 55)
    df_merge_train = pd.concat([df_s_train, df_v_train], ignore_index=True)

    os.makedirs(os.path.dirname(merge_train_out), exist_ok=True)
    df_merge_train.to_csv(merge_train_out, index=False)
    print(f"Saved Merged Train (No-Indo) -> {merge_train_out}")
    print(f"  Total rows   : {len(df_merge_train):,}")
    print(f"  Total groups : {df_merge_train['label_group'].nunique():,}")

    # 4. Zero-leakage assertions
    train_groups = set(df_merge_train["label_group"])
    shopee_test_groups = set(df_s_test["label_group"])
    vn_test_groups = set(df_v_test["label_group"])

    leakage_s = train_groups & shopee_test_groups
    leakage_v = train_groups & vn_test_groups
    assert len(leakage_s) == 0, f"DATA LEAKAGE: {len(leakage_s)} Shopee test groups in Merged Train!"
    assert len(leakage_v) == 0, f"DATA LEAKAGE: {len(leakage_v)} VN test groups in Merged Train!"

    print("\n" + "=" * 65)
    print("  NO-INDO PIPELINE COMPLETE — ZERO LEAKAGE VERIFIED")
    print("=" * 65)
    print(f"  1. Raw No-Indo Shopee : {shopee_no_indo_csv}")
    print(f"  2. Merged Train       : {len(df_merge_train):>6,} rows | {len(train_groups):>5,} groups -> {merge_train_out}")
    print(f"  3. Shopee Test (20%)  : {len(df_s_test):>6,} rows | {len(shopee_test_groups):>5,} groups -> {test_out}")
    print("=" * 65 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Filter out Indonesian groups and generate no-indo splits."
    )
    parser.add_argument("--shopee-raw", default=DEFAULT_SHOPEE_RAW, help="Path to raw Shopee CSV")
    parser.add_argument("--vn-raw",     default=DEFAULT_VN_RAW,     help="Path to raw Vietnamese CSV")
    parser.add_argument("--shopee-no-indo", default=DEFAULT_SHOPEE_NO_INDO, help="Path to output shopee_no_indo.csv")
    parser.add_argument("--test-out",   default=OUT_TEST_SHOPEE_NO_INDO, help="Path to output shopee_no_indo_test.csv")
    parser.add_argument("--merge-train-out", default=OUT_MERGE_TRAIN_NO_INDO, help="Path to output merge_train_no_indo.csv")
    parser.add_argument("--threshold",  type=float, default=0.50, help="Indonesian ratio threshold (default: 0.50)")
    parser.add_argument("--test-size",  type=float, default=0.20, help="Test split fraction (default: 0.20)")
    parser.add_argument("--seed",       type=int,   default=42,   help="Random seed (default: 42)")
    parser.add_argument("--force",      action="store_true",      help="Force re-run language detection even if output exists")
    args = parser.parse_args()

    # Step 1: Filter
    filter_shopee_no_indo(
        input_csv=args.shopee_raw,
        output_csv=args.shopee_no_indo,
        threshold=args.threshold,
        force=args.force,
    )

    # Step 2: Split & Merge
    create_no_indo_splits(
        shopee_no_indo_csv=args.shopee_no_indo,
        vn_raw_csv=args.vn_raw,
        test_out=args.test_out,
        merge_train_out=args.merge_train_out,
        test_size=args.test_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
