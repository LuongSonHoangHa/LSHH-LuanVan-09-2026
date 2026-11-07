"""
arcface_image_nfnet.py — Phase 2 (Image): ArcFace Fine-tuning on NFNet-F0
===========================================================================
Designed to run on Kaggle (GPU T4 x2 / P100).

Architecture:
  - Backbone  : timm NFNet-F0 (nfnet_f0) — pretrained on ImageNet-21k
                Frozen  : stem  +  stages[0]  +  stages[1]
                Trainable: stages[2]  +  stages[3]  +  head (lower LR)
  - Pooling   : GeM (Generalised Mean Pooling, p=3)
  - Projection: Linear(backbone.num_features -> EMBED_DIM, bias=False) + BatchNorm1d
  - Loss head : ArcMarginProduct (s=30, m=0.50)
  - Inference : GeM pool + L2-norm -> cosine threshold sweep -> Mean F1

Training data : TRAIN_CSV_NAME (default: merge_train.csv)
Image dirs    : IMG_DIRS  (shopee/ and vietnamese/ — looked up dynamically)
Test set A    : shopee_test.csv   (Shopee 20%)
Test set B    : vietnamese_test.csv (Vietnamese 20%)

Ablation: set TRAIN_CSV_NAME = "merge_train_no_indo.csv" to measure the effect
          of Indonesian data on Vietnamese test F1.

Inspired by the Shopee 2021 2nd-place solution (lyakaap):
  - ShopeeNet: NFNet backbone + reset_classifier + GeM pool + FC + BN
  - ArcMarginProduct loss head
  - Two LR groups: backbone vs head

Expected runtime on Kaggle T4 x2:
  - 10 epochs, ~28 K images  -> ~30-40 min
  - Threshold sweep           -> ~2 min

Outputs -> /kaggle/working/outputs/:
  arc_img_train_emb.npy           [N_train, EMBED_DIM]
  arc_img_shopee_test_emb.npy     [N_s, EMBED_DIM]
  arc_img_vn_test_emb.npy         [N_v, EMBED_DIM]
  arc_img_shopee_sweep.csv        threshold sweep (Shopee)
  arc_img_vn_sweep.csv            threshold sweep (Vietnamese)
  arc_img_summary.csv             best F1 summary (both test sets)
  arc_img_best_model.pth          best checkpoint (by train loss)
  arc_img_training_log.csv        epoch-level loss / acc history
"""

# ==============================================================================
# 0. IMPORTS
# ==============================================================================
import os, gc, time, math
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from typing import List, Tuple

try:
    import timm
    print(f"timm    : {timm.__version__}")
except ImportError:
    raise ImportError("timm not found. Run: pip install timm")

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
    """Search for a file in /kaggle/working or across attached /kaggle/input datasets (case-insensitive)."""
    if os.path.isabs(filename) and os.path.exists(filename):
        return filename

    # 1. Check current working directory (/kaggle/working)
    working_path = os.path.join("/kaggle/working", filename)
    if os.path.isfile(working_path):
        return os.path.abspath(working_path)

    # 2. Case-insensitive search across attached datasets in /kaggle/input
    target = filename.lower()
    if os.path.exists("/kaggle/input"):
        for root, _, files in os.walk("/kaggle/input"):
            for f in files:
                if f.lower() == target:
                    return os.path.abspath(os.path.join(root, f))

    return ""


def find_img_dirs() -> Tuple[str, str]:
    """
    Dynamically locate 'shopee' and 'vietnamese' image directories
    inside /kaggle/input (case-insensitive name match).
    """
    shopee_dir = ""
    vn_dir = ""
    search_root = "/kaggle/input"
    if os.path.exists(search_root):
        for root, dirs, _ in os.walk(search_root):
            for d in dirs:
                dl = d.lower()
                if dl == "shopee" and not shopee_dir:
                    shopee_dir = os.path.abspath(os.path.join(root, d))
                elif dl == "vietnamese" and not vn_dir:
                    vn_dir = os.path.abspath(os.path.join(root, d))
            if shopee_dir and vn_dir:
                break

    if not shopee_dir:
        print(f"[WARNING] find_img_dirs: could not locate a 'shopee' directory under {search_root}")
    if not vn_dir:
        print(f"[WARNING] find_img_dirs: could not locate a 'vietnamese' directory under {search_root}")
    return shopee_dir, vn_dir

SHOPEE_IMG_DIR, VN_IMG_DIR = find_img_dirs()


def _build_img_index(img_dir: str) -> dict:
    """
    Recursively scan img_dir and build {filename_lower: full_path} dict.
    This handles:
      - Images stored in subdirectories (e.g. shopee/train_images/*.jpg)
      - Case-mismatch between CSV filenames and on-disk filenames (Kaggle lowercases)
    """
    index = {}
    if not img_dir or not os.path.isdir(img_dir):
        return index
    for root, _, files in os.walk(img_dir):
        for fname in files:
            index[fname.lower()] = os.path.join(root, fname)
    return index


print("\n-- Indexing image directories (recursive scan) -----------------")
_SHOPEE_IDX: dict = _build_img_index(SHOPEE_IMG_DIR)
_VN_IDX:     dict = _build_img_index(VN_IMG_DIR)

# Fallback global index in case images were unzipped directly under dataset root
_GLOBAL_IMG_IDX: dict = {}
if (len(_SHOPEE_IDX) == 0 or len(_VN_IDX) == 0) and os.path.exists("/kaggle/input"):
    for root, _, files in os.walk("/kaggle/input"):
        for fname in files:
            fl = fname.lower()
            if fl.endswith((".jpg", ".jpeg", ".png", ".webp")):
                if fl not in _GLOBAL_IMG_IDX:
                    _GLOBAL_IMG_IDX[fl] = os.path.join(root, fname)
    if _GLOBAL_IMG_IDX:
        print(f"  Fallback global image index: {len(_GLOBAL_IMG_IDX):,} files discovered")

print(f"  Shopee index    : {len(_SHOPEE_IDX):,} files  ({SHOPEE_IMG_DIR})")
print(f"  Vietnamese index: {len(_VN_IDX):,} files  ({VN_IMG_DIR})")

OUTPUT_DIR = "/kaggle/working/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Model
BACKBONE   = "dm_nfnet_f0"   # timm model name with verified pretrained weights
IMG_SIZE   = 224             # training image size
IMG_SIZE_V = 256             # val/test image size (slightly larger for CenterCrop)
EMBED_DIM  = 512             # projection output dimension
GEM_P      = 3               # GeM pooling exponent

# Freezing strategy for NFNet-F0:
#   NFNet-F0 structure: stem -> stages[0..3] -> head
#   Frozen  : stem + stages[0] + stages[1]
#   Trainable: stages[2] + stages[3] + head
FREEZE_STAGES = 2            # freeze first N stages (0-indexed)

# ArcFace
ARC_S = 30.0
ARC_M = 0.50

# Training
EPOCHS        = 10
BATCH_SIZE    = 32            # reduce to 16 if OOM
LR_BACKBONE   = 1e-5          # trainable backbone stages
LR_HEAD       = 1e-4          # projection + ArcFace head
WEIGHT_DECAY  = 1e-4
WARMUP_EPOCHS = 1
SEED          = 42

# Encoding / Evaluation
ENC_BATCH       = 64
THRESHOLD_RANGE = [round(t, 2) for t in np.arange(0.30, 0.85, 0.05).tolist()]
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS     = 2

print(f"\n  Device        : {DEVICE.upper()}")
print(f"  Train CSV     : {TRAIN_CSV_NAME}")
print(f"  Backbone      : {BACKBONE}")
print(f"  Embed dim     : {EMBED_DIM}  |  GeM p={GEM_P}")
print(f"  Freeze stages : 0..{FREEZE_STAGES-1} (fine-tune {FREEZE_STAGES}..3 + head)")
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
    n = len(emb)
    rows = []
    print(f"\n  Sweeping {len(thresholds)} thresholds over {n} embeddings ...")
    for thresh in thresholds:
        preds = []
        for s in range(0, n, bs):
            e   = min(s + bs, n)
            sim = emb[s:e] @ emb.T
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

def get_img_path(row: pd.Series) -> str:
    """
    Resolve absolute image path from a CSV row using pre-built lookup indices.

    Lookup order (case-insensitive, handles nested subdirectories):
      1. Explicit 'source' column  -> check primary index directly.
      2. Smart routing by filename  -> .webp / vn-/sg-/cn- prefix => VN index,
                                       MD5 hex .jpg                => Shopee index.
      3. Cross-index fallback       -> try the other index.
      4. Global index fallback      -> check search roots if index was incomplete.
      5. Last resort: raw os.path.join (will return missing path for caller to handle).
    """
    img_name  = os.path.basename(str(row["image"]).strip())
    key       = img_name.lower()          # case-insensitive key

    # 1. Honour explicit 'source' column when present
    src = str(row.get("source", "")).lower()
    if src == "vietnamese":
        if key in _VN_IDX:
            return _VN_IDX[key]
    elif src == "shopee":
        if key in _SHOPEE_IDX:
            return _SHOPEE_IDX[key]

    # 2. Smart routing by filename pattern
    is_vn = key.endswith(".webp") or key.startswith(("vn-", "sg-", "cn-"))
    primary_idx   = _VN_IDX     if is_vn else _SHOPEE_IDX
    secondary_idx = _SHOPEE_IDX if is_vn else _VN_IDX
    primary_dir   = VN_IMG_DIR  if is_vn else SHOPEE_IMG_DIR

    if key in primary_idx:
        return primary_idx[key]

    # 3. Cross-index fallback (images that landed in the wrong folder)
    if key in secondary_idx:
        return secondary_idx[key]

    # 4. Global index fallback
    if key in _GLOBAL_IMG_IDX:
        return _GLOBAL_IMG_IDX[key]

    # 5. Last resort — path will be missing; ImageDataset catches the exception
    return os.path.join(primary_dir if primary_dir else (SHOPEE_IMG_DIR or "."), img_name)


TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize(IMG_SIZE + 32),
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

VAL_TRANSFORM = transforms.Compose([
    transforms.Resize(IMG_SIZE_V),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


class ImageDataset(Dataset):
    """
    Loads product images from shopee/ or vietnamese/ folders.
    In train mode, encodes label_group as a dense integer for ArcFace.
    Missing images are replaced with a black tensor.
    """
    def __init__(self, df: pd.DataFrame, transform,
                 is_train: bool = True):
        self.df        = df.reset_index(drop=True)
        self.transform = transform
        self.is_train  = is_train
        if is_train:
            groups = sorted(self.df["label_group"].unique())
            self.label_map = {g: i for i, g in enumerate(groups)}
            self.df["label_idx"] = self.df["label_group"].map(self.label_map)

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row  = self.df.iloc[idx]
        path = get_img_path(row)
        try:
            img = Image.open(path).convert("RGB")
            img = self.transform(img)
        except Exception:
            img = torch.zeros(3, IMG_SIZE, IMG_SIZE)

        item = {"image": img}
        if self.is_train:
            item["label"] = torch.tensor(row["label_idx"], dtype=torch.long)
        return item


# ==============================================================================
# 5. GEM POOLING
# ==============================================================================

def gem_pool(x: torch.Tensor, p: float = GEM_P, eps: float = 1e-6) -> torch.Tensor:
    """
    Generalised Mean Pooling.
    x: [B, C, H, W]  ->  [B, C]
    """
    return F.avg_pool2d(
        x.clamp(min=eps).pow(p),
        (x.size(-2), x.size(-1))
    ).pow(1.0 / p)


# ==============================================================================
# 6. ARCFACE HEAD
# ==============================================================================

class ArcMarginProduct(nn.Module):
    """
    ArcFace additive angular margin loss head.
    Deng et al. (2019) — s=30, m=0.50 are standard defaults.
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
        cosine  = F.linear(F.normalize(feat), F.normalize(self.weight))
        sine    = torch.sqrt((1.0 - cosine.pow(2)).clamp(0, 1))
        phi     = cosine * self.cos_m - sine * self.sin_m
        phi     = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        return (one_hot * phi + (1.0 - one_hot) * cosine) * self.s


# ==============================================================================
# 7. MODEL
# ==============================================================================

class ArcFaceImageModel(nn.Module):
    """
    NFNet-F0 backbone + GeM pooling + projection + ArcMarginProduct.

    Layer-freezing strategy (NFNet-F0):
      Frozen   : stem  +  stages[0]  +  stages[1]
      Trainable: stages[2]  +  stages[3]  (LR_BACKBONE)
                 projection  +  bn  +  arc_head  (LR_HEAD)

    This mirrors the 2nd-place Shopee solution: backbone.reset_classifier(0)
    removes the original head, then GeM + FC + BN replace it.
    """
    def __init__(self, backbone_name: str, num_classes: int,
                 embed_dim: int = EMBED_DIM,
                 gem_p: float = GEM_P,
                 freeze_stages: int = FREEZE_STAGES):
        super().__init__()
        self.gem_p     = gem_p
        self.embed_dim = embed_dim

        # Load pretrained backbone, strip classifier
        backbone_candidates = [backbone_name]
        if "dm_" not in backbone_name:
            backbone_candidates.append(f"dm_{backbone_name}")
        if backbone_name.startswith("dm_"):
            backbone_candidates.append(backbone_name.replace("dm_", "", 1))
        for fallback in ["dm_nfnet_f0", "eca_nfnet_l0", "nfnet_l0"]:
            if fallback not in backbone_candidates:
                backbone_candidates.append(fallback)

        model_loaded = None
        last_err = None
        for cand in backbone_candidates:
            try:
                model_loaded = timm.create_model(cand, pretrained=True)
                if cand != backbone_name:
                    print(f"  [INFO] Backbone '{backbone_name}' replaced with '{cand}' (pretrained=True)")
                break
            except Exception as e:
                last_err = e

        if model_loaded is None:
            raise RuntimeError(
                f"Could not load any pretrained backbone from {backbone_candidates}. Last error: {last_err}"
            )

        self.backbone = model_loaded
        self.backbone.reset_classifier(num_classes=0)  # remove softmax head
        backbone_dim = self.backbone.num_features

        # Freeze: stem
        for p in self.backbone.stem.parameters():
            p.requires_grad = False

        # Freeze: first `freeze_stages` stages
        for i, stage in enumerate(self.backbone.stages):
            if i < freeze_stages:
                for p in stage.parameters():
                    p.requires_grad = False
            # else: remains trainable

        # Projection head
        self.projection = nn.Linear(backbone_dim, embed_dim, bias=False)
        self.bn         = nn.BatchNorm1d(embed_dim)
        nn.init.xavier_normal_(self.projection.weight)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

        self.arc_head = ArcMarginProduct(embed_dim, num_classes)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Returns L2-normalised embeddings [B, embed_dim]."""
        feat = self.backbone.forward_features(x)   # [B, C, H, W]
        feat = gem_pool(feat, p=self.gem_p)        # [B, C, 1, 1]
        feat = feat.view(feat.size(0), -1)         # [B, C]
        feat = self.bn(self.projection(feat))      # [B, embed_dim]
        return F.normalize(feat, p=2, dim=1)

    def forward(self, x: torch.Tensor,
                labels: torch.Tensor = None):
        feat = self.extract_features(x)
        if labels is not None:
            return self.arc_head(feat, labels), feat
        return feat


# ==============================================================================
# 8. OPTIMIZER & SCHEDULER
# ==============================================================================

def build_optimizer(model: ArcFaceImageModel):
    """Two LR groups: low for trainable backbone stages, high for head."""
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
# 9. TRAINING LOOP
# ==============================================================================

def train_one_epoch(model, loader, optimizer, scheduler, criterion,
                    device, epoch, total_epochs) -> Tuple[float, float]:
    model.train()
    total_loss = n_correct = n_total = 0
    t0 = time.time()
    for step, batch in enumerate(loader, 1):
        imgs  = batch["image"].to(device)
        label = batch["label"].to(device)
        optimizer.zero_grad()
        logits, _ = model(imgs, label)
        loss = criterion(logits, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
        n_correct  += (logits.argmax(dim=1) == label).sum().item()
        n_total    += label.size(0)
        if step % 20 == 0 or step == len(loader):
            print(f"\r  Epoch [{epoch}/{total_epochs}] "
                  f"step {step}/{len(loader)} "
                  f"loss={total_loss/step:.4f} "
                  f"acc={n_correct/n_total*100:.1f}% "
                  f"[{time.time()-t0:.0f}s]", end="", flush=True)
    print()
    return total_loss / len(loader), n_correct / n_total


# ==============================================================================
# 10. ENCODE (inference)
# ==============================================================================

@torch.no_grad()
def encode_df(model: ArcFaceImageModel, df: pd.DataFrame,
              device: str, batch_size: int = ENC_BATCH) -> np.ndarray:
    """Extract L2-normalised image embeddings [N, embed_dim]."""
    model.eval()
    ds     = ImageDataset(df, VAL_TRANSFORM, is_train=False)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=True)
    embs = []; total = len(df); t0 = time.time()
    for step, batch in enumerate(loader, 1):
        imgs = batch["image"].to(device)
        embs.append(model.extract_features(imgs).cpu().numpy())
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
            f"[ERROR] {label} ('{filename}') not found in /kaggle/input or /kaggle/working.\n"
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


def verify_image_loading(df_sh: pd.DataFrame, df_vn: pd.DataFrame):
    """Sanity-check image resolution for Shopee (.jpg) and Vietnamese (.webp)."""
    print("\n-- Image directory verification --------------------------------")
    print(f"  Shopee img folder     : {SHOPEE_IMG_DIR} (exists: {os.path.isdir(SHOPEE_IMG_DIR) if SHOPEE_IMG_DIR else False})")
    print(f"  Vietnamese img folder : {VN_IMG_DIR} (exists: {os.path.isdir(VN_IMG_DIR) if VN_IMG_DIR else False})")
    print(f"  Shopee index size     : {len(_SHOPEE_IDX):,} files")
    print(f"  Vietnamese index size : {len(_VN_IDX):,} files")

    sh_ok = 0
    for idx in range(min(5, len(df_sh))):
        row = df_sh.iloc[idx]
        p   = get_img_path(row)
        found = os.path.exists(p)
        if found:
            sh_ok += 1
        else:
            key = str(row["image"]).strip().lower()
            print(f"    [MISS] Shopee row {idx}: '{row['image']}' -> '{p}'")
            print(f"           key='{key}' in shopee_idx={key in _SHOPEE_IDX}  vn_idx={key in _VN_IDX}")
    print(f"  Sample Shopee check   : {sh_ok}/5 images found on disk")

    vn_ok = 0
    for idx in range(min(5, len(df_vn))):
        row = df_vn.iloc[idx]
        p   = get_img_path(row)
        found = os.path.exists(p)
        if found:
            vn_ok += 1
        else:
            key = str(row["image"]).strip().lower()
            print(f"    [MISS] VN row {idx}: '{row['image']}' -> '{p}'")
            print(f"           key='{key}' in shopee_idx={key in _SHOPEE_IDX}  vn_idx={key in _VN_IDX}")
    print(f"  Sample Vietnamese check: {vn_ok}/5 images found on disk")

    if sh_ok == 0 and len(df_sh) > 0:
        # Print a few keys from the index to help diagnose mismatches
        sample_keys = list(_SHOPEE_IDX.keys())[:3]
        print(f"  [WARNING] Could not find Shopee images! Sample index keys: {sample_keys}")
    if vn_ok == 0 and len(df_vn) > 0:
        sample_keys = list(_VN_IDX.keys())[:3]
        print(f"  [WARNING] Could not find Vietnamese images! Sample index keys: {sample_keys}")


# ==============================================================================
# 11. MAIN
# ==============================================================================

def main():
    print("\n" + "=" * 65)
    print("  PHASE 2 (IMAGE) -- ArcFace Fine-tuning: NFNet-F0")
    print("=" * 65)

    # -- Load CSVs ------------------------------------------------------------
    print("\n-- Loading datasets --------------------------------------------")
    df_train = load_csv(TRAIN_CSV_NAME,   f"Train ({TRAIN_CSV_NAME})")
    df_sh    = load_csv(SHOPEE_TEST_NAME, "Shopee Test")
    df_vn    = load_csv(VN_TEST_NAME,     "Vietnamese Test")
    num_classes = df_train["label_group"].nunique()
    print(f"\n  num_classes  : {num_classes:,}")

    # -- Verify Image Setup ---------------------------------------------------
    verify_image_loading(df_sh, df_vn)

    # -- DataLoader -----------------------------------------------------------
    print("\n-- Building dataloader -----------------------------------------")
    ds_train = ImageDataset(df_train, TRAIN_TRANSFORM, is_train=True)
    dl_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
    print(f"  {len(dl_train)} steps/epoch")

    # -- Build model ----------------------------------------------------------
    print("\n-- Building model ----------------------------------------------")
    model = ArcFaceImageModel(BACKBONE, num_classes).to(DEVICE)
    t, total = param_counts(model)
    print(f"  Total: {total/1e6:.1f}M  |  Trainable: {t/1e6:.1f}M ({t/total*100:.1f}%)")
    print(f"  Frozen  : stem + stages[0..{FREEZE_STAGES-1}]")
    print(f"  Trainable: stages[{FREEZE_STAGES}..3] + projection + arc_head")

    # -- Optimizer & Scheduler ------------------------------------------------
    optimizer    = build_optimizer(model)
    total_steps  = EPOCHS * len(dl_train)
    warmup_steps = WARMUP_EPOCHS * len(dl_train)
    scheduler    = cosine_warmup_scheduler(optimizer, total_steps, warmup_steps)
    criterion    = nn.CrossEntropyLoss()

    # -- Training loop --------------------------------------------------------
    print("\n-- Training ----------------------------------------------------")
    best_loss = float("inf")
    ckpt_path = os.path.join(OUTPUT_DIR, "arc_img_best_model.pth")
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
                    "backbone":      BACKBONE,
                    "embed_dim":     EMBED_DIM,
                    "gem_p":         GEM_P,
                    "freeze_stages": FREEZE_STAGES,
                    "num_classes":   num_classes,
                    "arc_s":         ARC_S,
                    "arc_m":         ARC_M,
                    "train_csv":     TRAIN_CSV_NAME,
                },
            }, ckpt_path)
            print(f"  + Checkpoint saved (loss={loss:.4f})")
        gc.collect(); torch.cuda.empty_cache()

    pd.DataFrame(history).to_csv(
        os.path.join(OUTPUT_DIR, "arc_img_training_log.csv"), index=False
    )

    # -- Load best checkpoint -------------------------------------------------
    print("\n-- Loading best checkpoint for evaluation ----------------------")
    try:
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    print(f"  Loaded epoch {ckpt['epoch']} (loss={ckpt['loss']:.4f})")

    # -- Encode ---------------------------------------------------------------
    print("\n-- Encoding all splits ----------------------------------------")
    train_emb = encode_df(model, df_train, DEVICE)
    sh_emb    = encode_df(model, df_sh,    DEVICE)
    vn_emb    = encode_df(model, df_vn,    DEVICE)
    np.save(os.path.join(OUTPUT_DIR, "arc_img_train_emb.npy"),       train_emb)
    np.save(os.path.join(OUTPUT_DIR, "arc_img_shopee_test_emb.npy"), sh_emb)
    np.save(os.path.join(OUTPUT_DIR, "arc_img_vn_test_emb.npy"),     vn_emb)
    print(f"  train emb : {train_emb.shape}")
    print(f"  shopee emb: {sh_emb.shape}")
    print(f"  vn emb    : {vn_emb.shape}")

    # -- Threshold sweep ------------------------------------------------------
    sh_gt = build_ground_truth(df_sh)
    vn_gt = build_ground_truth(df_vn)

    print("\n-- Threshold sweep: Shopee Test --------------------------------")
    df_sh_sweep = threshold_sweep(sh_emb, sh_gt, THRESHOLD_RANGE)
    best_sh     = df_sh_sweep.loc[df_sh_sweep["mean_f1"].idxmax()]
    df_sh_sweep.to_csv(os.path.join(OUTPUT_DIR, "arc_img_shopee_sweep.csv"), index=False)
    print_result_box("Shopee Test (20%)", best_sh["threshold"],
                     best_sh["mean_f1"], len(df_sh))

    print("\n-- Threshold sweep: Vietnamese Test ----------------------------")
    df_vn_sweep = threshold_sweep(vn_emb, vn_gt, THRESHOLD_RANGE)
    best_vn     = df_vn_sweep.loc[df_vn_sweep["mean_f1"].idxmax()]
    df_vn_sweep.to_csv(os.path.join(OUTPUT_DIR, "arc_img_vn_sweep.csv"), index=False)
    print_result_box("Vietnamese Test (20%)", best_vn["threshold"],
                     best_vn["mean_f1"], len(df_vn))

    # -- Summary CSV ----------------------------------------------------------
    summary = pd.DataFrame([
        {"phase": "2-Image",
         "model": f"NFNet-F0+ArcFace(freeze_stages={FREEZE_STAGES},s={ARC_S},m={ARC_M})",
         "train_csv":        TRAIN_CSV_NAME,
         "embed_dim":        EMBED_DIM,
         "gem_p":            GEM_P,
         "epochs":           EPOCHS,
         "best_train_loss":  best_loss,
         "test_set":         "Shopee Test (20%)",
         "n_rows":           len(df_sh),
         "n_groups":         df_sh["label_group"].nunique(),
         "best_threshold":   best_sh["threshold"],
         "mean_f1":          best_sh["mean_f1"]},
        {"phase": "2-Image",
         "model": f"NFNet-F0+ArcFace(freeze_stages={FREEZE_STAGES},s={ARC_S},m={ARC_M})",
         "train_csv":        TRAIN_CSV_NAME,
         "embed_dim":        EMBED_DIM,
         "gem_p":            GEM_P,
         "epochs":           EPOCHS,
         "best_train_loss":  best_loss,
         "test_set":         "Vietnamese Test (20%)",
         "n_rows":           len(df_vn),
         "n_groups":         df_vn["label_group"].nunique(),
         "best_threshold":   best_vn["threshold"],
         "mean_f1":          best_vn["mean_f1"]},
    ])
    summary.to_csv(os.path.join(OUTPUT_DIR, "arc_img_summary.csv"), index=False)

    # -- Final summary --------------------------------------------------------
    print("\n" + "=" * 65)
    print("  PHASE 2 (IMAGE) COMPLETE")
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
