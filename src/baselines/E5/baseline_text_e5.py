"""
baseline_text_e5.py — Phase 1B: Text-only Baseline (multilingual-E5-large)
============================================================================
Designed to run on Kaggle (GPU T4 / P100).

Strategy:
  - Training set : 80% Shopee + 80% Vietnamese (merged split) — NOT used here
                   (E5 is frozen/zero-shot; no training needed for Phase 1B)
  - Test set A   : 20% Shopee      → evaluates EN/ID in-domain retrieval
  - Test set B   : 20% Vietnamese  → evaluates VI cross-market retrieval
  - NO fine-tuning: E5 backbone is frozen (zero-shot evaluation)

Pipeline:
  1. Mount datasets / set paths for Kaggle input directories
  2. Load split CSVs (split_shopee_test, split_vn_test)
  3. Build E5Encoder (frozen multilingual-e5-large, mean-pool + L2-norm)
  4. Encode titles for BOTH test sets
  5. Threshold sweep (cosine similarity) -> best Mean F1 on each test set
  6. Save embeddings (.npy) and results (.csv) to /kaggle/working/outputs/

Expected runtime on Kaggle T4:
  - Encoding ~37K titles  -> ~4 min
  - Threshold sweep       -> ~2 min

How to run on Kaggle:
  1. Create a new Kaggle notebook and upload this file (or paste code into cells).
  2. Add your dataset via "Add Data":
       Dataset must contain: split_shopee_test.csv, split_vn_test.csv
  3. Set GPU accelerator: Settings -> Accelerator -> GPU T4 x2 (or P100)
  4. Run All.

Outputs saved to /kaggle/working/outputs/:
  txt_shopee_test_emb.npy       -- Shopee test embeddings [N_s, 1024]
  txt_vn_test_emb.npy           -- Vietnamese test embeddings [N_v, 1024]
  txt_baseline_shopee_sweep.csv -- Shopee full threshold sweep
  txt_baseline_vn_sweep.csv     -- Vietnamese full threshold sweep
  txt_baseline_summary.csv      -- Final best-F1 summary (both test sets)
"""

# ===============================================================================
# 0. IMPORTS
# ===============================================================================

import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from typing import List, Tuple

print(f"PyTorch  : {torch.__version__}")
print(f"CUDA     : {torch.cuda.is_available()}  |  GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")

# ===============================================================================
# 1. CONFIGURATION
# ===============================================================================

# Kaggle dataset paths:
# Kaggle creates unique slug names under /kaggle/input/ that often differ from the UI title.
# find_csv_dir() searches /kaggle/input dynamically so you never have to guess the exact folder name.
def find_csv_dir() -> str:
    if os.path.exists("/kaggle/input"):
        for root, _, files in os.walk("/kaggle/input"):
            if "shopee_test.csv" in files or "split_shopee_test.csv" in files:
                return root
    local_path = os.path.abspath("data/processed")
    if (os.path.exists(os.path.join(local_path, "shopee_test.csv")) or
            os.path.exists(os.path.join(local_path, "split_shopee_test.csv"))):
        return local_path
    return "/kaggle/input/Split_csv"

CSV_DIR = find_csv_dir()

# Note: Images in 'Merged_Image_set/shopee' are not used in this Phase 1B text baseline.
# They will be used in Phase 1A (Vision) and Phase 2 (ArcFace fine-tuning).
IMAGE_DIR = "/kaggle/input/Merged_Image_set/shopee"

OUTPUT_DIR = "/kaggle/working/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Test CSV paths (supports both shopee_test.csv and legacy split_shopee_test.csv)
SHOPEE_TEST_CSV = (os.path.join(CSV_DIR, "shopee_test.csv")
                  if os.path.exists(os.path.join(CSV_DIR, "shopee_test.csv"))
                  else os.path.join(CSV_DIR, "split_shopee_test.csv"))
VN_TEST_CSV     = (os.path.join(CSV_DIR, "vietnamese_test.csv")
                  if os.path.exists(os.path.join(CSV_DIR, "vietnamese_test.csv"))
                  else os.path.join(CSV_DIR, "split_vn_test.csv"))

# Model
E5_MODEL_NAME = "intfloat/multilingual-e5-large"
# Swap to base if VRAM is tight (< 8 GB):
# E5_MODEL_NAME = "intfloat/multilingual-e5-base"

# Encoding
TXT_BATCH   = 64     # lower to 32 if GPU OOM
TXT_MAX_LEN = 128    # max token length for product titles

# Evaluation
THRESHOLD_RANGE = [round(t, 2) for t in np.arange(0.30, 0.85, 0.05).tolist()]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"\n  Device         : {DEVICE.upper()}")
print(f"  E5 Model       : {E5_MODEL_NAME}")
print(f"  Batch size     : {TXT_BATCH}")
print(f"  Threshold range: {THRESHOLD_RANGE}")

# ===============================================================================
# 2. EVALUATION UTILITIES
# ===============================================================================

def build_ground_truth(df: pd.DataFrame) -> List[List[int]]:
    """
    For each row, return the row-indices of ALL items in the same label_group
    (including itself — a product is always its own match).
    """
    group_to_idx = df.groupby("label_group").apply(lambda g: g.index.tolist()).to_dict()
    return [group_to_idx[row["label_group"]] for _, row in df.iterrows()]


def compute_mean_f1(pred_matches: List[List[int]],
                    true_matches: List[List[int]]) -> float:
    """Mean F1 score across all queries."""
    scores = []
    for pred, true in zip(pred_matches, true_matches):
        pred_set, true_set = set(pred), set(true)
        tp = len(pred_set & true_set)
        if tp == 0:
            scores.append(0.0)
            continue
        p = tp / len(pred_set)
        r = tp / len(true_set)
        scores.append(2 * p * r / (p + r))
    return float(np.mean(scores))


def threshold_sweep_to_df(embeddings: np.ndarray,
                           true_matches: List[List[int]],
                           thresholds: List[float],
                           batch_size: int = 512) -> pd.DataFrame:
    """
    Sweep cosine-similarity thresholds. Returns a DataFrame with columns
    [threshold, mean_f1] for all thresholds tested.

    Uses batched matmul to avoid building the full N*N matrix in one shot.
    """
    n = len(embeddings)
    rows = []
    print(f"\n  Sweeping {len(thresholds)} thresholds over {n} embeddings ...")

    for thresh in thresholds:
        pred_matches = []
        for start in range(0, n, batch_size):
            end   = min(start + batch_size, n)
            batch = embeddings[start:end]           # [B, D]
            sim   = batch @ embeddings.T            # [B, N] (cosine since L2-normed)
            for i, row_sim in enumerate(sim):
                matched = np.where(row_sim >= thresh)[0].tolist()
                pred_matches.append(matched if matched else [start + i])

        f1   = compute_mean_f1(pred_matches, true_matches)
        rows.append({"threshold": thresh, "mean_f1": round(f1, 6)})
        print(f"    threshold={thresh:.2f}  Mean F1={f1:.4f}")

    return pd.DataFrame(rows)


# ===============================================================================
# 3. E5 ENCODER
# ===============================================================================

class E5Encoder:
    """
    Wraps multilingual-E5-large for batch title encoding.
    Uses mean pooling over token outputs + L2 normalisation.
    All inputs are prefixed with "query: " (required for symmetric retrieval).
    """

    def __init__(self, model_name: str = E5_MODEL_NAME, device: str = DEVICE):
        print(f"\n  Loading E5 tokenizer and model: {model_name} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model     = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()
        self.device    = device
        n_params = sum(p.numel() for p in self.model.parameters()) / 1e6
        print(f"  Model loaded on {device.upper()}. Parameters: {n_params:.0f}M")

    @torch.no_grad()
    def encode(self,
               texts: List[str],
               batch_size: int = TXT_BATCH,
               prefix: str = "query: ") -> np.ndarray:
        """
        Encode a list of raw title strings into L2-normalised embeddings.

        Args:
            texts      : list of product title strings (raw, no prefix needed)
            batch_size : forward-pass batch size (reduce to 32 if OOM)
            prefix     : E5 instruction prefix — always "query: " for
                         symmetric retrieval (both query and gallery use same prefix)

        Returns:
            np.ndarray of shape [N, 1024]  (float32, L2-normalised)
        """
        prefixed = [prefix + t for t in texts]
        all_emb  = []
        total    = len(prefixed)
        t0       = time.time()

        for start in range(0, total, batch_size):
            end   = min(start + batch_size, total)
            batch = prefixed[start:end]

            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=TXT_MAX_LEN,
                return_tensors="pt",
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            outputs  = self.model(**encoded)

            # Mean pool (masked) over token dimension
            token_emb = outputs.last_hidden_state          # [B, T, D]
            mask      = encoded["attention_mask"]          # [B, T]
            mask_exp  = mask.unsqueeze(-1).expand(token_emb.size()).float()
            pooled    = (torch.sum(token_emb * mask_exp, dim=1)
                         / torch.clamp(mask_exp.sum(dim=1), min=1e-9))
            normalised = F.normalize(pooled, p=2, dim=1)

            all_emb.append(normalised.cpu().numpy())

            elapsed = time.time() - t0
            print(f"\r  Encoded {end:>6}/{total}  ({end/total*100:.1f}%)  "
                  f"[{elapsed:.0f}s]", end="", flush=True)

        print()
        return np.vstack(all_emb).astype(np.float32)


# ===============================================================================
# 4. HELPERS
# ===============================================================================

def load_csv_safe(path: str, label: str) -> pd.DataFrame:
    """Load CSV with informative error if not found."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"[ERROR] {label} CSV not found:\n  {path}\n"
            f"  Make sure you added the Kaggle dataset that contains this file."
        )
    df = pd.read_csv(path).reset_index(drop=True)
    print(f"  {label}: {len(df):,} rows | {df['label_group'].nunique():,} groups")
    return df


def encode_or_load(encoder: E5Encoder,
                   df: pd.DataFrame,
                   cache_path: str,
                   label: str) -> np.ndarray:
    """Encode titles, or load cached embeddings if already computed."""
    if os.path.exists(cache_path):
        print(f"  [CACHE HIT] Loading {label} embeddings from {cache_path} ...")
        emb = np.load(cache_path)
    else:
        titles = df["title"].fillna("").tolist()
        print(f"\n  Encoding {len(titles):,} {label} titles ...")
        emb = encoder.encode(titles, batch_size=TXT_BATCH)
        np.save(cache_path, emb)
        print(f"  Saved: {cache_path}")
    print(f"  {label} embedding shape: {emb.shape}")
    return emb


def print_result_box(label: str, best_thresh: float, best_f1: float, n: int):
    print("\n" + "=" * 55)
    print(f"  RESULT -- {label}")
    print(f"  Test rows  : {n:,}")
    print(f"  Best thresh: {best_thresh:.2f}")
    print(f"  Mean F1    : {best_f1:.4f}")
    print("=" * 55)


# ===============================================================================
# 5. MAIN
# ===============================================================================

def main():
    print("\n" + "=" * 60)
    print("  PHASE 1B -- TEXT BASELINE  (multilingual-E5-large, zero-shot)")
    print("=" * 60)
    print(f"  Model    : {E5_MODEL_NAME}")
    print(f"  Device   : {DEVICE.upper()}")
    print(f"  CSV Path : {CSV_DIR}")
    print(f"  Output   : {OUTPUT_DIR}")

    # ── 5.1 Load test CSVs ────────────────────────────────────────────────────
    print("\n-- Loading test sets -------------------------------------------")
    df_shopee_test = load_csv_safe(SHOPEE_TEST_CSV, "Shopee Test (20%)")
    df_vn_test     = load_csv_safe(VN_TEST_CSV,     "Vietnamese Test (20%)")

    # ── 5.2 Build ground truth ────────────────────────────────────────────────
    print("\n-- Building ground truth ---------------------------------------")
    shopee_gt = build_ground_truth(df_shopee_test)
    vn_gt     = build_ground_truth(df_vn_test)
    print(f"  Shopee    : {len(shopee_gt):,} queries")
    print(f"  Vietnamese: {len(vn_gt):,} queries")

    # ── 5.3 Load E5 model ─────────────────────────────────────────────────────
    print("\n-- Loading E5 model --------------------------------------------")
    encoder = E5Encoder(E5_MODEL_NAME, DEVICE)

    # ── 5.4 Encode test sets ──────────────────────────────────────────────────
    print("\n-- Encoding test sets ------------------------------------------")
    shopee_emb = encode_or_load(
        encoder, df_shopee_test,
        os.path.join(OUTPUT_DIR, "txt_shopee_test_emb.npy"),
        "Shopee Test"
    )
    vn_emb = encode_or_load(
        encoder, df_vn_test,
        os.path.join(OUTPUT_DIR, "txt_vn_test_emb.npy"),
        "Vietnamese Test"
    )

    # ── 5.5 Threshold sweep -- Shopee ─────────────────────────────────────────
    print("\n-- Threshold sweep: Shopee Test --------------------------------")
    df_shopee_sweep = threshold_sweep_to_df(shopee_emb, shopee_gt, THRESHOLD_RANGE)
    best_shopee     = df_shopee_sweep.loc[df_shopee_sweep["mean_f1"].idxmax()]
    df_shopee_sweep.to_csv(
        os.path.join(OUTPUT_DIR, "txt_baseline_shopee_sweep.csv"), index=False
    )
    print_result_box("Shopee Test (20%)",
                     best_shopee["threshold"],
                     best_shopee["mean_f1"],
                     len(df_shopee_test))

    # ── 5.6 Threshold sweep -- Vietnamese ─────────────────────────────────────
    print("\n-- Threshold sweep: Vietnamese Test ----------------------------")
    df_vn_sweep = threshold_sweep_to_df(vn_emb, vn_gt, THRESHOLD_RANGE)
    best_vn     = df_vn_sweep.loc[df_vn_sweep["mean_f1"].idxmax()]
    df_vn_sweep.to_csv(
        os.path.join(OUTPUT_DIR, "txt_baseline_vn_sweep.csv"), index=False
    )
    print_result_box("Vietnamese Test (20%)",
                     best_vn["threshold"],
                     best_vn["mean_f1"],
                     len(df_vn_test))

    # ── 5.7 Summary CSV ───────────────────────────────────────────────────────
    summary = pd.DataFrame([
        {
            "phase"         : "1B",
            "model"         : "multilingual-E5-large (frozen, zero-shot)",
            "aQE"           : False,
            "DBA"           : False,
            "test_set"      : "Shopee Test (20%)",
            "n_rows"        : len(df_shopee_test),
            "n_groups"      : df_shopee_test["label_group"].nunique(),
            "best_threshold": best_shopee["threshold"],
            "mean_f1"       : best_shopee["mean_f1"],
        },
        {
            "phase"         : "1B",
            "model"         : "multilingual-E5-large (frozen, zero-shot)",
            "aQE"           : False,
            "DBA"           : False,
            "test_set"      : "Vietnamese Test (20%)",
            "n_rows"        : len(df_vn_test),
            "n_groups"      : df_vn_test["label_group"].nunique(),
            "best_threshold": best_vn["threshold"],
            "mean_f1"       : best_vn["mean_f1"],
        },
    ])
    summary_path = os.path.join(OUTPUT_DIR, "txt_baseline_summary.csv")
    summary.to_csv(summary_path, index=False)

    # ── 5.8 Final print ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  PHASE 1B COMPLETE")
    print(f"  Shopee Test    Mean F1 : {best_shopee['mean_f1']:.4f}"
          f"  (thresh={best_shopee['threshold']:.2f})")
    print(f"  Vietnamese Test Mean F1: {best_vn['mean_f1']:.4f}"
          f"  (thresh={best_vn['threshold']:.2f})")
    print("=" * 60)
    print(f"\n  All outputs saved to: {OUTPUT_DIR}/")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        fpath = os.path.join(OUTPUT_DIR, f)
        size  = os.path.getsize(fpath)
        print(f"    {f}  ({size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
