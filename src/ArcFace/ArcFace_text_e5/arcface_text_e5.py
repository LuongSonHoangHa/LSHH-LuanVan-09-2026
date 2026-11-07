"""
arcface_text_e5.py — Phase 2 (Text): ArcFace Fine-tuning on multilingual-E5-large
====================================================================================
Designed to run on Kaggle (GPU T4 x2 / P100).

Architecture:
  - Backbone  : intfloat/multilingual-e5-large (24 encoder layers)
                Frozen  : embeddings + encoder.layer[0..FREEZE_LAYERS-1]
                Trainable: encoder.layer[FREEZE_LAYERS..23] + projection + arc_head
  - Projection: Linear(1024 -> EMBED_DIM, bias=False) + BatchNorm1d
  - Loss head : ArcMarginProduct (s=30, m=0.50)
  - Inference : mean-pool + L2-norm -> cosine threshold sweep -> Mean F1

Training data : TRAIN_CSV_NAME  (default: merge_train.csv)
Test set A    : shopee_test.csv   (Shopee 20%)
Test set B    : vietnamese_test.csv (Vietnamese 20%)

Ablation: set TRAIN_CSV_NAME = 'merge_train_no_indo.csv' to measure the effect
          of Indonesian data on Vietnamese test F1.

Expected runtime on Kaggle T4 x2:
  - 10 epochs, ~28 K titles  -> ~15-20 min
  - Threshold sweep           -> ~2 min

Outputs -> /kaggle/working/outputs/:
  arc_txt_train_emb.npy           [N_train, EMBED_DIM]
  arc_txt_shopee_test_emb.npy     [N_s, EMBED_DIM]
  arc_txt_vn_test_emb.npy         [N_v, EMBED_DIM]
  arc_txt_shopee_sweep.csv        threshold sweep (Shopee)
  arc_txt_vn_sweep.csv            threshold sweep (Vietnamese)
  arc_txt_summary.csv             best F1 summary (both test sets)
  arc_txt_best_model.pth          best checkpoint (by train loss)
  arc_txt_training_log.csv        epoch-level loss / acc history
"""

# ==============================================================================
# 0. IMPORTS
# ==============================================================================
import os, gc, time, math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from typing import List, Tuple

print(f"PyTorch : {torch.__version__}")
print(f"CUDA    : {torch.cuda.is_available()}  |  "
      f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")

# ==============================================================================
# 1. CONFIGURATION  <- change variables here to switch experiment
# ==============================================================================

# -- CHANGE THESE TO SWITCH EXPERIMENT -----------------------------------------
TRAIN_CSV_NAME   = "merge_train.csv"           # e.g. "merge_train.csv" or "merge_train_no_indo.csv"
SHOPEE_TEST_NAME = "shopee_test.csv"           # e.g. "shopee_test.csv" or "shopee_no_indo_test.csv"
VN_TEST_NAME     = "vietnamese_test.csv"
# ------------------------------------------------------------------------------

def find_file(filename: str) -> str:
    """Search for a specific file by name across /kaggle/input or local directories."""
    if os.path.isabs(filename) and os.path.exists(filename):
        return filename

    local_candidates = [
        os.path.join("data/processed", filename),
        os.path.join("/kaggle/working", filename),
        filename
    ]
    for p in local_candidates:
        if os.path.isfile(p):
            return os.path.abspath(p)

    if os.path.exists("/kaggle/input"):
        for root, _, files in os.walk("/kaggle/input"):
            if filename in files:
                return os.path.join(root, filename)

    return ""

OUTPUT_DIR = "/kaggle/working/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Model
E5_MODEL_NAME = "intfloat/multilingual-e5-large"
EMBED_DIM     = 512          # projection output dimension
FREEZE_LAYERS = 18           # freeze encoder.layer[0..17], fine-tune [18..23]

# ArcFace
ARC_S = 30.0
ARC_M = 0.50

# Training
EPOCHS        = 10
BATCH_SIZE    = 32            # reduce to 16 if OOM
LR_BACKBONE   = 1e-5          # unfrozen transformer layers
LR_HEAD       = 1e-4          # projection + ArcFace head
WEIGHT_DECAY  = 1e-4
TXT_MAX_LEN   = 128
WARMUP_EPOCHS = 1
SEED          = 42

# Encoding / Evaluation
ENC_BATCH       = 64
THRESHOLD_RANGE = [round(t, 2) for t in np.arange(0.30, 0.85, 0.05).tolist()]
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"

print(f"\n  Device        : {DEVICE.upper()}")
print(f"  Train CSV     : {TRAIN_CSV_NAME}")
print(f"  E5 Model      : {E5_MODEL_NAME}")
print(f"  Embed dim     : {EMBED_DIM}  |  Freeze: {FREEZE_LAYERS}/24 layers")
print(f"  ArcFace (s,m) : ({ARC_S}, {ARC_M})")
print(f"  Epochs        : {EPOCHS}  |  Batch: {BATCH_SIZE}")


# ==============================================================================
# 2. SEED
# ==============================================================================
def set_seed(seed: int = SEED):
    import random; random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed()


# ==============================================================================
# 3. EVALUATION UTILITIES
# ==============================================================================

def build_ground_truth(df: pd.DataFrame) -> List[List[int]]:
    """Row-indices of all items in the same label_group (including self)."""
    g2i = df.groupby("label_group").apply(lambda g: g.index.tolist()).to_dict()
    return [g2i[row["label_group"]] for _, row in df.iterrows()]


def compute_mean_f1(pred_matches: List[List[int]],
                    true_matches: List[List[int]]) -> float:
    scores = []
    for p, t in zip(pred_matches, true_matches):
        ps, ts = set(p), set(t)
        tp = len(ps & ts)
        if tp == 0:
            scores.append(0.0); continue
        scores.append(2 * tp / (len(ps) + len(ts)))
    return float(np.mean(scores))


def threshold_sweep(emb: np.ndarray,
                    gt: List[List[int]],
                    thresholds: List[float],
                    bs: int = 512) -> pd.DataFrame:
    """Batched cosine-similarity threshold sweep (emb must be L2-normalised)."""
    n = len(emb)
    rows = []
    print(f"\n  Sweeping {len(thresholds)} thresholds over {n} embeddings ...")
    for thresh in thresholds:
        preds = []
        for s in range(0, n, bs):
            e   = min(s + bs, n)
            sim = emb[s:e] @ emb.T          # [B, N] cosine (L2-normed)
            for i, row_sim in enumerate(sim):
                matched = np.where(row_sim >= thresh)[0].tolist()
                preds.append(matched if matched else [s + i])
        f1 = compute_mean_f1(preds, gt)
        rows.append({"threshold": thresh, "mean_f1": round(f1, 6)})
        print(f"    threshold={thresh:.2f}  Mean F1={f1:.4f}")
    return pd.DataFrame(rows)


def print_result_box(label: str, thresh: float, f1: float, n: int):
    print("\n" + "=" * 55)
    print(f"  RESULT -- {label}")
    print(f"  Rows       : {n:,}")
    print(f"  Best thresh: {thresh:.2f}")
    print(f"  Mean F1    : {f1:.4f}")
    print("=" * 55)


# ==============================================================================
# 4. DATASET
# ==============================================================================

class TitleDataset(Dataset):
    """
    Tokenises product titles on-the-fly.
    In train mode, encodes label_group as a dense integer for ArcFace.
    """
    def __init__(self, df: pd.DataFrame, tokenizer,
                 max_len: int = TXT_MAX_LEN,
                 prefix: str = "query: ",
                 is_train: bool = True):
        self.df        = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len   = max_len
        self.prefix    = prefix
        self.is_train  = is_train
        if is_train:
            groups = sorted(self.df["label_group"].unique())
            self.label_map = {g: i for i, g in enumerate(groups)}
            self.df["label_idx"] = self.df["label_group"].map(self.label_map)

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        title = self.prefix + str(row["title"])
        enc   = self.tokenizer(
            title, max_length=self.max_len, padding="max_length",
            truncation=True, return_tensors="pt"
        )
        item = {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }
        if self.is_train:
            item["label"] = torch.tensor(row["label_idx"], dtype=torch.long)
        return item


# ==============================================================================
# 5. ARCFACE HEAD
# ==============================================================================

class ArcMarginProduct(nn.Module):
    """
    ArcFace additive angular margin loss head.
    Deng et al. (2019) — s=30, m=0.50 are standard defaults for product matching.
    """
    def __init__(self, in_features: int, num_classes: int,
                 s: float = ARC_S, m: float = ARC_M):
        super().__init__()
        self.s = s; self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.cos_m = math.cos(m);  self.sin_m = math.sin(m)
        self.th    = math.cos(math.pi - m)
        self.mm    = math.sin(math.pi - m) * m

    def forward(self, feat: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        cosine  = F.linear(F.normalize(feat), F.normalize(self.weight))  # [B, C]
        sine    = torch.sqrt((1.0 - cosine.pow(2)).clamp(0, 1))
        phi     = cosine * self.cos_m - sine * self.sin_m               # cos(theta+m)
        phi     = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        return (one_hot * phi + (1.0 - one_hot) * cosine) * self.s


# ==============================================================================
# 6. MODEL
# ==============================================================================

class ArcFaceTextModel(nn.Module):
    """
    multilingual-E5-large + projection head + ArcMarginProduct.

    Layer-freezing strategy (E5-large has 24 encoder layers):
      Frozen  : backbone.embeddings  +  encoder.layer[0 .. FREEZE_LAYERS-1]
      Trainable: encoder.layer[FREEZE_LAYERS .. 23]  (low LR = LR_BACKBONE)
                 projection + bn + arc_head            (high LR = LR_HEAD)
    """
    def __init__(self, model_name: str, num_classes: int,
                 embed_dim: int = EMBED_DIM,
                 freeze_layers: int = FREEZE_LAYERS):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)

        # Freeze embedding layer
        for p in self.backbone.embeddings.parameters():
            p.requires_grad = False
        # Freeze first N transformer blocks
        for i, layer in enumerate(self.backbone.encoder.layer):
            if i < freeze_layers:
                for p in layer.parameters():
                    p.requires_grad = False

        hidden = self.backbone.config.hidden_size   # 1024 for E5-large
        self.projection = nn.Linear(hidden, embed_dim, bias=False)
        self.bn         = nn.BatchNorm1d(embed_dim)
        nn.init.xavier_normal_(self.projection.weight)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

        self.arc_head = ArcMarginProduct(embed_dim, num_classes)

    def _mean_pool(self, hidden_state: torch.Tensor,
                   attn_mask: torch.Tensor) -> torch.Tensor:
        mask   = attn_mask.unsqueeze(-1).expand(hidden_state.size()).float()
        pooled = torch.sum(hidden_state * mask, dim=1)
        return pooled / torch.clamp(mask.sum(dim=1), min=1e-9)

    def extract_features(self, input_ids: torch.Tensor,
                         attn_mask: torch.Tensor) -> torch.Tensor:
        """Returns L2-normalised embeddings [B, embed_dim]."""
        out    = self.backbone(input_ids=input_ids, attention_mask=attn_mask)
        pooled = self._mean_pool(out.last_hidden_state, attn_mask)
        feat   = self.bn(self.projection(pooled))
        return F.normalize(feat, p=2, dim=1)

    def forward(self, input_ids, attn_mask, labels=None):
        feat = self.extract_features(input_ids, attn_mask)
        if labels is not None:
            return self.arc_head(feat, labels), feat
        return feat


# ==============================================================================
# 7. OPTIMIZER & SCHEDULER
# ==============================================================================

def build_optimizer(model: ArcFaceTextModel):
    """Two LR groups: low for unfrozen backbone, high for projection + head."""
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    head_params     = (list(model.projection.parameters())
                       + list(model.bn.parameters())
                       + list(model.arc_head.parameters()))
    return torch.optim.AdamW(
        [{"params": backbone_params, "lr": LR_BACKBONE},
         {"params": head_params,     "lr": LR_HEAD}],
        weight_decay=WEIGHT_DECAY
    )


def cosine_warmup_scheduler(optimizer, total_steps: int, warmup_steps: int):
    def lr_fn(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        p = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * p))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_fn)


# ==============================================================================
# 8. TRAINING LOOP
# ==============================================================================

def train_one_epoch(model, loader, optimizer, scheduler, criterion,
                    device, epoch, total_epochs) -> Tuple[float, float]:
    model.train()
    total_loss = n_correct = n_total = 0
    t0 = time.time()
    for step, batch in enumerate(loader, 1):
        ids   = batch["input_ids"].to(device)
        mask  = batch["attention_mask"].to(device)
        label = batch["label"].to(device)
        optimizer.zero_grad()
        logits, _ = model(ids, mask, label)
        loss = criterion(logits, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
        n_correct  += (logits.argmax(dim=1) == label).sum().item()
        n_total    += label.size(0)
        if step % 50 == 0 or step == len(loader):
            print(f"\r  Epoch [{epoch}/{total_epochs}] "
                  f"step {step}/{len(loader)} "
                  f"loss={total_loss/step:.4f} "
                  f"acc={n_correct/n_total*100:.1f}% "
                  f"[{time.time()-t0:.0f}s]", end="", flush=True)
    print()
    return total_loss / len(loader), n_correct / n_total


# ==============================================================================
# 9. ENCODE (inference)
# ==============================================================================

@torch.no_grad()
def encode_df(model: ArcFaceTextModel, df: pd.DataFrame,
              tokenizer, device: str,
              batch_size: int = ENC_BATCH) -> np.ndarray:
    """Encode all titles in df -> L2-normalised float32 array [N, embed_dim]."""
    model.eval()
    ds     = TitleDataset(df, tokenizer, is_train=False)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=2, pin_memory=True)
    embs = []; total = len(df); t0 = time.time()
    for step, batch in enumerate(loader, 1):
        ids  = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        embs.append(model.extract_features(ids, mask).cpu().numpy())
        done = min(step * batch_size, total)
        print(f"\r  Encoded {done:>6}/{total} ({done/total*100:.1f}%)"
              f"  [{time.time()-t0:.0f}s]", end="", flush=True)
    print()
    return np.vstack(embs).astype(np.float32)


def load_csv(filename: str, label: str) -> pd.DataFrame:
    path = find_file(filename)
    if not path or not os.path.exists(path):
        attached = list(os.listdir('/kaggle/input')) if os.path.exists('/kaggle/input') else []
        raise FileNotFoundError(
            f"[ERROR] {label} ('{filename}') not found in /kaggle/input or local data/processed.\n"
            f"  Attached datasets in /kaggle/input: {attached}\n"
            f"  Please ensure the dataset containing '{filename}' is attached."
        )
    df = pd.read_csv(path).reset_index(drop=True)
    print(f"  {label:<24s}: {len(df):>6,} rows | {df['label_group'].nunique():>5,} groups | Path: {path}")
    return df


def param_counts(model) -> Tuple[int, int]:
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total


# ==============================================================================
# 10. MAIN
# ==============================================================================

def main():
    print("\n" + "=" * 65)
    print("  PHASE 2 (TEXT) -- ArcFace Fine-tuning: multilingual-E5-large")
    print("=" * 65)

    # -- Load CSVs ------------------------------------------------------------
    print("\n-- Loading datasets --------------------------------------------")
    df_train = load_csv(TRAIN_CSV_NAME,   f"Train ({TRAIN_CSV_NAME})")
    df_sh    = load_csv(SHOPEE_TEST_NAME, "Shopee Test")
    df_vn    = load_csv(VN_TEST_NAME,     "Vietnamese Test")
    num_classes = df_train["label_group"].nunique()
    print(f"\n  num_classes: {num_classes:,}")

    # -- Tokenizer & DataLoader -----------------------------------------------
    print("\n-- Building tokenizer & dataloader ----------------------------")
    tokenizer = AutoTokenizer.from_pretrained(E5_MODEL_NAME)
    ds_train  = TitleDataset(df_train, tokenizer, is_train=True)
    dl_train  = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True,
                           num_workers=2, pin_memory=True, drop_last=True)
    print(f"  {len(dl_train)} steps/epoch")

    # -- Build model ----------------------------------------------------------
    print("\n-- Building model ----------------------------------------------")
    model = ArcFaceTextModel(E5_MODEL_NAME, num_classes).to(DEVICE)
    t, total = param_counts(model)
    print(f"  Total: {total/1e6:.1f}M  |  Trainable: {t/1e6:.1f}M ({t/total*100:.1f}%)")
    print(f"  Frozen layers  : 0..{FREEZE_LAYERS-1}")
    print(f"  Trainable layers: {FREEZE_LAYERS}..23")

    # -- Optimizer & Scheduler ------------------------------------------------
    optimizer    = build_optimizer(model)
    total_steps  = EPOCHS * len(dl_train)
    warmup_steps = WARMUP_EPOCHS * len(dl_train)
    scheduler    = cosine_warmup_scheduler(optimizer, total_steps, warmup_steps)
    criterion    = nn.CrossEntropyLoss()

    # -- Training loop --------------------------------------------------------
    print("\n-- Training ----------------------------------------------------")
    best_loss = float("inf")
    ckpt_path = os.path.join(OUTPUT_DIR, "arc_txt_best_model.pth")
    history   = []

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        loss, acc = train_one_epoch(model, dl_train, optimizer, scheduler,
                                    criterion, DEVICE, epoch, EPOCHS)
        history.append({"epoch": epoch, "loss": loss, "train_acc": acc})
        print(f"  -> Epoch {epoch:02d}  loss={loss:.4f}  acc={acc*100:.1f}%"
              f"  [{time.time()-t0:.0f}s]")
        if loss < best_loss:
            best_loss = loss
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "loss":        loss,
                "config": {
                    "embed_dim":     EMBED_DIM,
                    "freeze_layers": FREEZE_LAYERS,
                    "num_classes":   num_classes,
                    "arc_s":         ARC_S,
                    "arc_m":         ARC_M,
                    "train_csv":     TRAIN_CSV_NAME,
                },
            }, ckpt_path)
            print(f"  + Checkpoint saved (loss={loss:.4f})")
        gc.collect(); torch.cuda.empty_cache()

    pd.DataFrame(history).to_csv(
        os.path.join(OUTPUT_DIR, "arc_txt_training_log.csv"), index=False
    )

    # -- Load best checkpoint -------------------------------------------------
    print("\n-- Loading best checkpoint for evaluation ----------------------")
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    print(f"  Loaded epoch {ckpt['epoch']} (loss={ckpt['loss']:.4f})")

    # -- Encode ---------------------------------------------------------------
    print("\n-- Encoding all splits ----------------------------------------")
    train_emb = encode_df(model, df_train, tokenizer, DEVICE)
    sh_emb    = encode_df(model, df_sh,    tokenizer, DEVICE)
    vn_emb    = encode_df(model, df_vn,    tokenizer, DEVICE)
    np.save(os.path.join(OUTPUT_DIR, "arc_txt_train_emb.npy"),       train_emb)
    np.save(os.path.join(OUTPUT_DIR, "arc_txt_shopee_test_emb.npy"), sh_emb)
    np.save(os.path.join(OUTPUT_DIR, "arc_txt_vn_test_emb.npy"),     vn_emb)
    print(f"  train emb : {train_emb.shape}")
    print(f"  shopee emb: {sh_emb.shape}")
    print(f"  vn emb    : {vn_emb.shape}")

    # -- Threshold sweep ------------------------------------------------------
    sh_gt = build_ground_truth(df_sh)
    vn_gt = build_ground_truth(df_vn)

    print("\n-- Threshold sweep: Shopee Test --------------------------------")
    df_sh_sweep = threshold_sweep(sh_emb, sh_gt, THRESHOLD_RANGE)
    best_sh     = df_sh_sweep.loc[df_sh_sweep["mean_f1"].idxmax()]
    df_sh_sweep.to_csv(os.path.join(OUTPUT_DIR, "arc_txt_shopee_sweep.csv"), index=False)
    print_result_box("Shopee Test (20%)", best_sh["threshold"],
                     best_sh["mean_f1"], len(df_sh))

    print("\n-- Threshold sweep: Vietnamese Test ----------------------------")
    df_vn_sweep = threshold_sweep(vn_emb, vn_gt, THRESHOLD_RANGE)
    best_vn     = df_vn_sweep.loc[df_vn_sweep["mean_f1"].idxmax()]
    df_vn_sweep.to_csv(os.path.join(OUTPUT_DIR, "arc_txt_vn_sweep.csv"), index=False)
    print_result_box("Vietnamese Test (20%)", best_vn["threshold"],
                     best_vn["mean_f1"], len(df_vn))

    # -- Summary CSV ----------------------------------------------------------
    summary = pd.DataFrame([
        {"phase": "2-Text",
         "model": f"E5-large+ArcFace(freeze={FREEZE_LAYERS},s={ARC_S},m={ARC_M})",
         "train_csv":       TRAIN_CSV_NAME,
         "embed_dim":       EMBED_DIM,
         "epochs":          EPOCHS,
         "best_train_loss": best_loss,
         "test_set":        "Shopee Test (20%)",
         "n_rows":          len(df_sh),
         "n_groups":        df_sh["label_group"].nunique(),
         "best_threshold":  best_sh["threshold"],
         "mean_f1":         best_sh["mean_f1"]},
        {"phase": "2-Text",
         "model": f"E5-large+ArcFace(freeze={FREEZE_LAYERS},s={ARC_S},m={ARC_M})",
         "train_csv":       TRAIN_CSV_NAME,
         "embed_dim":       EMBED_DIM,
         "epochs":          EPOCHS,
         "best_train_loss": best_loss,
         "test_set":        "Vietnamese Test (20%)",
         "n_rows":          len(df_vn),
         "n_groups":        df_vn["label_group"].nunique(),
         "best_threshold":  best_vn["threshold"],
         "mean_f1":         best_vn["mean_f1"]},
    ])
    summary.to_csv(os.path.join(OUTPUT_DIR, "arc_txt_summary.csv"), index=False)

    # -- Final summary --------------------------------------------------------
    print("\n" + "=" * 65)
    print("  PHASE 2 (TEXT) COMPLETE")
    print(f"  Train CSV         : {TRAIN_CSV_NAME}")
    print(f"  Shopee F1         : {best_sh['mean_f1']:.4f}"
          f"  (thresh={best_sh['threshold']:.2f})")
    print(f"  Vietnamese F1     : {best_vn['mean_f1']:.4f}"
          f"  (thresh={best_vn['threshold']:.2f})")
    print("=" * 65)
    print(f"\n  All outputs -> {OUTPUT_DIR}/")
    for fname in sorted(os.listdir(OUTPUT_DIR)):
        sz = os.path.getsize(os.path.join(OUTPUT_DIR, fname))
        print(f"    {fname}  ({sz/1024:.1f} KB)")


if __name__ == "__main__":
    main()
