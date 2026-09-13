"""
ResNet-UNet v5 — TNBC / MoNuSeg — Exact SBA-Attention + Gated Skips
====================================================================
This is v4 upgraded with exact Softmax SBA-Attention and enhanced 
decoder skip connections (Phase 5).

  PHASE 5 — Attention Upgrades (SBA + Decoder Skips)
    (a) Exact Softmax SBA-Attention: Replaces the linear attention 
        approximation from v4. Since the ASPP bottleneck operates at a 
        low resolution (e.g., 16x16, N=256), exact O(N^2) Softmax 
        attention is computationally trivial while offering a perfectly 
        sharp attention distribution. Replaces explicit F.unfold window 
        attention with a highly optimized depthwise convolution on V.
    (b) Attention Gates (Semantic Filtering): Standard decoder skip 
        connections pull in high-resolution background noise. Attention 
        Gates use the semantically rich decoder features to mute 
        background noise in the skip connection before merging.
    (c) Boundary-Gated Skips (Structural Enhancement): Optional. Reuses 
        the Phase 4a pre-bottleneck Scharr gate on the skip connections 
        themselves to guarantee razor-sharp boundary features are boosted 
        before being handed to the decoder.
    Toggles: USE_ATTENTION_GATES, USE_BOUNDARY_SKIPS.

  PHASE 1 — Instance separation (the actual bottleneck for dense nuclei).
    Adds a HoVer-Net-style auxiliary head that regresses, per pixel, the
    (horizontal, vertical) offset to its instance's centroid.
    Toggle: USE_HV_HEAD.

  PHASE 2 — Fixed + strengthened boundary descriptor (feeds SBA-Attention).
    Orientation bug fix + optional Laplacian-of-Gaussian channel.
    Toggles: USE_LOG_CHANNEL, BOUNDARY_DESC_MODE.

  PHASE 3 — Direct boundary supervision.
    Added `BoundaryConsistencyLoss`: a small, fixed-kernel Scharr edge
    map penalized with L1.
    Toggle: USE_BOUNDARY_LOSS.

  PHASE 4 — Pre-bottleneck Scharr refinement + post-bottleneck multi-
  scale Hessian, wired through the existing kernelized-attention.
    Toggles: USE_PRE_BOTTLENECK_REFINE, USE_HESSIAN_CHANNELS.
----------------------------------------------------------------------------
"""

# ─────────────────────────────────────────────────────────────────────────────
# 0. Imports
# ─────────────────────────────────────────────────────────────────────────────
import os, glob, random, warnings, copy
import numpy as np
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms.functional as TF
import torchvision.transforms as T

from sklearn.model_selection import train_test_split, KFold

from skimage.measure import label as cc_label
from skimage.segmentation import watershed
from skimage.morphology import erosion, disk, remove_small_objects
from scipy import ndimage as ndi

# ─────────────────────────────────────────────────────────────────────────────
# 1. Configuration
# ─────────────────────────────────────────────────────────────────────────────

# ---- MoNuSeg (Train/Test only) ------------------------------------------------
MONUSEG_BASE_DIR = Path(
    "/kaggle/input/datasets/kartikmaity/tnbc-monuseg-segmentation/"
    "MoNuSeg_official_split/MoNuSeg"
)
TRAIN_IMG_DIR  = MONUSEG_BASE_DIR / "Train" / "Images"
TRAIN_MASK_DIR = MONUSEG_BASE_DIR / "Train" / "Masks"
TEST_IMG_DIR   = MONUSEG_BASE_DIR / "Test"  / "Images"
TEST_MASK_DIR  = MONUSEG_BASE_DIR / "Test"  / "Masks"

# ---- TNBC (Train/Validation/Test, all pre-made) -------------------------------
TNBC_BASE_DIR = Path(
    "/kaggle/input/datasets/kartikmaity/tnbc-monuseg-segmentation/"
    "TNBC/TNBC/TNBC_with_split/TNBC_split"
)
TNBC_TRAIN_IMG_DIR  = TNBC_BASE_DIR / "Train"      / "Images"
TNBC_TRAIN_MASK_DIR = TNBC_BASE_DIR / "Train"      / "Masks"
TNBC_VAL_IMG_DIR    = TNBC_BASE_DIR / "Validation" / "Images"
TNBC_VAL_MASK_DIR   = TNBC_BASE_DIR / "Validation" / "Masks"
TNBC_TEST_IMG_DIR   = TNBC_BASE_DIR / "Test"       / "Images"
TNBC_TEST_MASK_DIR  = TNBC_BASE_DIR / "Test"       / "Masks"

IMG_SIZE     = 512
BATCH_SIZE   = 4
NUM_EPOCHS   = 500         
LR           = 3e-4
WEIGHT_DECAY = 1e-4
POS_WEIGHT_CAP = 15.0      
PRINT_EVERY  = 20          
VAL_FRAC     = 0.15        
EMA_DECAY    = 0.999
USE_AMP      = True
N_FOLDS      = 5           
SEED         = 42

# --- SBA-Attention hyperparameters -----------------------------------------
SBA_HEADS   = 4

# --- Phase 5: Skip Connection and Exact Attention enhancements -------------
USE_ATTENTION_GATES   = True   # Semantic filtering on skip connections
USE_BOUNDARY_SKIPS    = False  # Apply Scharr gating to skip connections

# --- Phase 2: boundary-descriptor toggles -----------------------------------
USE_LOG_CHANNEL      = False
USE_HESSIAN_CHANNELS = True
HESSIAN_SCALES       = (1, 2)      
BOUNDARY_DESC_MODE   = "dilation"  
SBA_K_DIM = 4 + (1 if USE_LOG_CHANNEL else 0) + (2 if USE_HESSIAN_CHANNELS else 0)

# --- Phase 4: pre-bottleneck Scharr-gated refinement toggle -----------------
USE_PRE_BOTTLENECK_REFINE = True

# --- Phase 1: instance-aware HV head toggles --------------------------------
USE_HV_HEAD            = True
USE_CC_FALLBACK        = True   
HV_MSE_WEIGHT           = 1.0
HV_MSGE_WEIGHT          = 1.0
HV_TOTAL_WEIGHT         = 1.0   
WATERSHED_MIN_SIZE      = 10
WATERSHED_SEED_THRESH   = 0.4   
WATERSHED_SEED_EROSION  = 2     

# --- Phase 3: boundary-consistency loss toggles -----------------------------
USE_BOUNDARY_LOSS    = True
BOUNDARY_LOSS_WEIGHT = 0.15   
BOUNDARY_LOSS_BETA   = 3.0    

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Mask loading
# ─────────────────────────────────────────────────────────────────────────────
def load_mask_as_binary(path: Path) -> np.ndarray:
    arr = np.array(Image.open(path).convert("I"), dtype=np.int32)
    if arr.max() > 1:
        binary = (arr > 0).astype(np.float32)
    else:
        binary = arr.astype(np.float32)
    return binary


_cc_fallback_warned = set()


def load_mask_instances(path: Path, use_cc_fallback=USE_CC_FALLBACK):
    arr = np.array(Image.open(path).convert("I"), dtype=np.int32)
    if arr.max() > 1:
        return arr, True

    binary = (arr > 0).astype(np.uint8)
    if use_cc_fallback:
        inst = cc_label(binary, connectivity=2).astype(np.int32)
    else:
        inst = binary.astype(np.int32)

    key = str(Path(path).parent)
    if key not in _cc_fallback_warned:
        _cc_fallback_warned.add(key)
        print(f"  [NOTE] Masks in {key} carry no real instance IDs — using "
              f"connected-component fallback for HV supervision.")
    return inst, False


def compute_hv_maps(inst_map: np.ndarray) -> np.ndarray:
    h_map = np.zeros(inst_map.shape, dtype=np.float32)
    v_map = np.zeros(inst_map.shape, dtype=np.float32)
    inst_ids = np.unique(inst_map)
    inst_ids = inst_ids[inst_ids != 0]
    for iid in inst_ids:
        ys, xs = np.nonzero(inst_map == iid)
        if len(xs) == 0:
            continue
        cx, cy = xs.mean(), ys.mean()
        hx = xs.astype(np.float32) - cx
        vy = ys.astype(np.float32) - cy
        hx_max = np.abs(hx).max()
        vy_max = np.abs(vy).max()
        h_map[ys, xs] = hx / (hx_max + 1e-8) if hx_max > 0 else hx
        v_map[ys, xs] = vy / (vy_max + 1e-8) if vy_max > 0 else vy
    return np.stack([h_map, v_map], axis=0)


def diagnose_masks(mask_dir: Path, n_samples: int = 5):
    mask_dir = Path(mask_dir)
    paths = sorted(mask_dir.glob("*.png"))[:n_samples]
    if not paths:
        print(f"  [WARN] No masks found in {mask_dir}")
        return
    print(f"\n  Mask diagnostics ({mask_dir}):")
    for p in paths:
        arr = np.array(Image.open(p).convert("I"), dtype=np.int32)
        binary = (arr > 0).astype(np.float32)
        fg_pct = binary.mean() * 100
        n_inst = len(np.unique(arr)) - (1 if 0 in arr else 0)
        print(f"    {p.name:40s}  dtype={arr.dtype}  "
              f"min={arr.min()}  max={arr.max()}  fg%={fg_pct:.1f}%  "
              f"unique_labels={n_inst}")


def compute_pos_weight(mask_paths, cap=POS_WEIGHT_CAP):
    fg_fracs = [load_mask_as_binary(mp).mean() for mp in mask_paths]
    fg_mean = float(np.mean(fg_fracs))
    fg_mean = max(fg_mean, 1e-4)
    pw = min((1 - fg_mean) / fg_mean, cap)
    print(f"  Auto pos_weight: mean fg={fg_mean*100:.1f}%  ->  pos_weight={pw:.2f}")
    return pw


@torch.no_grad()
def find_best_threshold(model, loader, device=None, thresholds=None, use_hv=USE_HV_HEAD):
    device = device or next(model.parameters()).device
    thresholds = thresholds if thresholds is not None else np.linspace(0.05, 0.95, 19)
    model.eval()
    all_probs, all_targets = [], []
    for batch in loader:
        imgs, masks = batch["image"], batch["mask"]
        imgs = imgs.to(device)
        out = model(imgs)
        probs = torch.sigmoid(out["seg"]).cpu()
        all_probs.append(probs)
        all_targets.append(masks)
    probs   = torch.cat(all_probs).view(-1)
    targets = torch.cat(all_targets).view(-1)

    best_thr, best_dice = 0.5, -1.0
    for thr in thresholds:
        pred = (probs > thr).float()
        tp = (pred * targets).sum()
        fp = (pred * (1 - targets)).sum()
        fn = ((1 - pred) * targets).sum()
        dice = ((2 * tp) / (2 * tp + fp + fn + 1e-7)).item()
        if dice > best_dice:
            best_dice, best_thr = dice, float(thr)

    print(f"  Threshold tuning: best_thr={best_thr:.2f}  (val Dice={best_dice:.4f})")
    return best_thr, best_dice


# ─────────────────────────────────────────────────────────────────────────────
# 3. Dataset
# ─────────────────────────────────────────────────────────────────────────────
class SegDataset(Dataset):
    MEAN = [0.485, 0.456, 0.406]
    STD  = [0.229, 0.224, 0.225]

    def __init__(self, image_paths, mask_paths, img_size=256, augment=False,
                 use_hv=USE_HV_HEAD, use_cc_fallback=USE_CC_FALLBACK):
        self.imgs    = image_paths
        self.masks   = mask_paths
        self.sz      = img_size
        self.augment = augment
        self.use_hv  = use_hv
        self.use_cc_fallback = use_cc_fallback
        self.norm    = T.Normalize(self.MEAN, self.STD)
        self.eraser  = T.RandomErasing(p=0.3, scale=(0.01, 0.05))

    def __len__(self): return len(self.imgs)

    def _geom_multi(self, img, nearest_tensors):
        flip_h = random.random() > 0.5
        flip_v = random.random() > 0.5
        k = random.randint(0, 3)
        do_affine = random.random() > 0.5
        angle = random.uniform(-15, 15) if do_affine else 0.0
        shear = random.uniform(-5, 5) if do_affine else 0.0

        def apply(t, interp):
            if flip_h: t = TF.hflip(t)
            if flip_v: t = TF.vflip(t)
            if k: t = torch.rot90(t, k, [1, 2])
            if do_affine:
                t = TF.affine(t, angle=angle, translate=[0, 0], scale=1.0,
                               shear=shear, interpolation=interp)
            return t

        img = apply(img, TF.InterpolationMode.BILINEAR)
        nearest_out = [apply(t, TF.InterpolationMode.NEAREST) for t in nearest_tensors]
        return img, nearest_out

    def _photo(self, img):
        if random.random() > 0.4:
            img = TF.adjust_hue(img,        random.uniform(-0.05, 0.05))
            img = TF.adjust_saturation(img, random.uniform(0.8,   1.2))
        if random.random() > 0.4:
            img = TF.adjust_brightness(img, random.uniform(0.75, 1.25))
            img = TF.adjust_contrast(img,   random.uniform(0.75, 1.25))
        if random.random() > 0.5:
            ks = random.choice([3, 5])
            img = TF.gaussian_blur(img, kernel_size=ks, sigma=random.uniform(0.1, 1.5))
        if random.random() > 0.5:
            noise = torch.randn_like(img) * random.uniform(0.01, 0.04)
            img   = torch.clamp(img + noise, 0.0, 1.0)
        return img

    def __getitem__(self, idx):
        img = Image.open(self.imgs[idx]).convert("RGB")
        img = img.resize((self.sz, self.sz), Image.BILINEAR)
        img = T.ToTensor()(img)

        inst_arr, has_real_inst = load_mask_instances(self.masks[idx], self.use_cc_fallback)
        inst_arr = np.ascontiguousarray(inst_arr, dtype=np.int32)
        inst_resized = np.array(
            Image.fromarray(inst_arr).resize((self.sz, self.sz), Image.NEAREST),
            dtype=np.int32,
        )
        inst_t = torch.from_numpy(inst_resized).unsqueeze(0).float()

        if self.augment:
            img, (inst_t,) = self._geom_multi(img, [inst_t])
            img = self._photo(img)
            img = self.eraser(img)

        inst_np = inst_t.squeeze(0).round().long().numpy().astype(np.int32)
        binary = torch.from_numpy((inst_np > 0).astype(np.float32)).unsqueeze(0)

        if self.use_hv:
            hv = torch.from_numpy(compute_hv_maps(inst_np))
        else:
            hv = torch.zeros(2, self.sz, self.sz)

        img = self.norm(img)
        return {
            "image": img,
            "mask": binary,
            "hv": hv,
            "has_real_inst": torch.tensor(has_real_inst),
        }


def _worker_init_fn(worker_id):
    seed = (torch.initial_seed() + worker_id) % 2**32
    random.seed(seed)
    np.random.seed(seed)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Building datasets
# ─────────────────────────────────────────────────────────────────────────────
def _strip_known_suffixes(stem: str) -> str:
    suffixes = ["_bin_mask", "_binary_mask", "_binary", "_bin",
                "_mask", "_masks", "_label", "_labels", "_gt",
                "_instance", "_instances", "_ann", "_seg"]
    s = stem
    changed = True
    while changed:
        changed = False
        for suf in suffixes:
            if s.lower().endswith(suf):
                s = s[: -len(suf)]
                changed = True
    return s


def gather_pairs(image_dir: Path, mask_dir: Path):
    image_dir, mask_dir = Path(image_dir), Path(mask_dir)
    ext = ["*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff"]
    img_paths = sorted(p for e in ext for p in image_dir.glob(e))
    if not img_paths:
        raise FileNotFoundError(f"No images found in {image_dir}")

    mask_paths = sorted(p for e in ext for p in mask_dir.glob(e))
    if not mask_paths:
        raise FileNotFoundError(f"No masks found in {mask_dir}")

    by_exact_stem = {}
    by_norm_stem  = {}
    for mp in mask_paths:
        by_exact_stem.setdefault(mp.stem, []).append(mp)
        by_norm_stem.setdefault(_strip_known_suffixes(mp.stem), []).append(mp)

    pairs, unmatched = [], []
    for ip in img_paths:
        cands = by_exact_stem.get(ip.stem)
        if not cands:
            cands = by_norm_stem.get(_strip_known_suffixes(ip.stem))
        if not cands:
            cands = [mp for mp in mask_paths
                     if ip.stem in mp.stem or mp.stem in ip.stem]
        if not cands:
            unmatched.append(ip.name)
            continue
        pairs.append((ip, sorted(cands)[0]))

    if unmatched:
        sample_imgs  = [p.name for p in img_paths[:5]]
        sample_masks = [p.name for p in mask_paths[:5]]
        raise FileNotFoundError(
            f"Could not match {len(unmatched)}/{len(img_paths)} images to masks.\n"
        )
    return pairs


def fg_density_buckets(pairs, n_buckets=4):
    fg_pcts = []
    for _, mp in pairs:
        arr = load_mask_as_binary(mp)
        fg_pcts.append(arr.mean())
    fg_pcts = np.array(fg_pcts)
    quantiles = np.quantile(fg_pcts, np.linspace(0, 1, n_buckets + 1))
    quantiles[-1] += 1e-6
    buckets = np.digitize(fg_pcts, quantiles[1:-1], right=True)
    return buckets


def build_train_val(train_img_dir, train_mask_dir, img_size,
                     val_frac=VAL_FRAC, seed=SEED):
    pairs = gather_pairs(train_img_dir, train_mask_dir)
    buckets = fg_density_buckets(pairs)

    idx = np.arange(len(pairs))
    tr_idx, vl_idx = train_test_split(
        idx, test_size=val_frac, random_state=seed, stratify=buckets
    )

    def sub(indices, aug):
        imgs  = [pairs[i][0] for i in indices]
        masks = [pairs[i][1] for i in indices]
        return SegDataset(imgs, masks, img_size=img_size, augment=aug)

    train_ds = sub(tr_idx, aug=True)
    val_ds   = sub(vl_idx, aug=False)
    print(f"  Train pool split  →  train: {len(train_ds)} | val: {len(val_ds)}")
    return train_ds, val_ds


def build_test(test_img_dir, test_mask_dir, img_size):
    pairs = gather_pairs(test_img_dir, test_mask_dir)
    imgs  = [p[0] for p in pairs]
    masks = [p[1] for p in pairs]
    ds = SegDataset(imgs, masks, img_size=img_size, augment=False)
    print(f"  Official test set  →  {len(ds)} images")
    return ds


def build_official_split(train_img_dir, train_mask_dir,
                          val_img_dir, val_mask_dir,
                          test_img_dir, test_mask_dir,
                          img_size):
    train_pairs = gather_pairs(train_img_dir, train_mask_dir)
    val_pairs   = gather_pairs(val_img_dir,   val_mask_dir)
    test_pairs  = gather_pairs(test_img_dir,  test_mask_dir)

    def to_dataset(pairs, augment):
        imgs  = [p[0] for p in pairs]
        masks = [p[1] for p in pairs]
        return SegDataset(imgs, masks, img_size=img_size, augment=augment)

    train_ds = to_dataset(train_pairs, augment=True)
    val_ds   = to_dataset(val_pairs,   augment=False)
    test_ds  = to_dataset(test_pairs,  augment=False)

    print(f"  Official TNBC split  →  train: {len(train_ds)} | "
          f"val: {len(val_ds)} | test: {len(test_ds)}")

    return train_ds, val_ds, test_ds


def fg_weighted_sampler(dataset: SegDataset):
    weights = []
    for mp in dataset.masks:
        arr = load_mask_as_binary(mp)
        weights.append(max(arr.mean(), 1e-3))
    weights = torch.as_tensor(weights, dtype=torch.double)
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Model building blocks
# ─────────────────────────────────────────────────────────────────────────────
class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.bn1   = nn.BatchNorm2d(in_ch)
        self.relu  = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.skip  = nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False) \
                     if (stride != 1 or in_ch != out_ch) else nn.Identity()

    def forward(self, x):
        return self.conv2(self.relu(self.bn2(
               self.conv1(self.relu(self.bn1(x)))))) + self.skip(x)


class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, n=2):
        super().__init__()
        self.blocks = nn.Sequential(
            ResBlock(in_ch, out_ch),
            *[ResBlock(out_ch, out_ch) for _ in range(n - 1)]
        )
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        s = self.blocks(x)
        return s, self.pool(s)


class ASPPBottleneck(nn.Module):
    def __init__(self, ch, rates=(3, 6, 12)):
        super().__init__()
        def _branch(k, d=1):
            return nn.Sequential(
                nn.Conv2d(ch, ch, k, padding=d * (k // 2), dilation=d, bias=False),
                nn.BatchNorm2d(ch), nn.ReLU(inplace=True)
            )
        self.b0   = _branch(1)
        self.brs  = nn.ModuleList([_branch(3, r) for r in rates])
        self.gap  = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch, ch, 1, bias=False),
            nn.BatchNorm2d(ch), nn.ReLU(inplace=True)
        )
        self.proj = nn.Sequential(
            nn.Conv2d((2 + len(rates)) * ch, ch, 1, bias=False),
            nn.BatchNorm2d(ch), nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )

    def forward(self, x):
        h, w = x.shape[2:]
        feats = [self.b0(x)] + [b(x) for b in self.brs]
        gap = F.interpolate(self.gap(x), (h, w), mode="bilinear", align_corners=False)
        feats.append(gap)
        return self.proj(torch.cat(feats, 1))

# ─────────────────────────────────────────────────────────────────────────────
# 5b. Boundary Descriptors and Exact Attention (Phase 5)
# ─────────────────────────────────────────────────────────────────────────────
class ScharrBoundaryDescriptor(nn.Module):
    def __init__(self, in_ch, beta_init=1.0, k_dim=SBA_K_DIM,
                 use_log_channel=USE_LOG_CHANNEL,
                 use_hessian_channels=USE_HESSIAN_CHANNELS,
                 multiscale_mode=BOUNDARY_DESC_MODE,
                 hessian_scales=HESSIAN_SCALES):
        super().__init__()
        expected_k_dim = 4 + (1 if use_log_channel else 0) + (2 if use_hessian_channels else 0)
        assert k_dim == expected_k_dim
        self.use_log_channel = use_log_channel
        self.use_hessian_channels = use_hessian_channels
        self.multiscale_mode = multiscale_mode
        self.hessian_scales = hessian_scales

        gx = torch.tensor([[-3., 0., 3.],
                            [-10., 0., 10.],
                            [-3., 0., 3.]]) / 16.0
        gy = gx.t().contiguous()
        self.register_buffer("gx", gx.view(1, 1, 3, 3))
        self.register_buffer("gy", gy.view(1, 1, 3, 3))

        if self.use_log_channel:
            log_kernel = torch.tensor([
                [0.,  0., -1.,  0.,  0.],
                [0., -1., -2., -1.,  0.],
                [-1., -2., 16., -2., -1.],
                [0., -1., -2., -1.,  0.],
                [0.,  0., -1.,  0.,  0.],
            ])
            self.register_buffer("log_kernel", log_kernel.view(1, 1, 5, 5))

        if self.multiscale_mode == "blur":
            g1d = torch.tensor([1., 4., 6., 4., 1.])
            g2d = torch.outer(g1d, g1d)
            g2d = g2d / g2d.sum()
            self.register_buffer("blur_kernel", g2d.view(1, 1, 5, 5))

        if self.use_hessian_channels:
            hxx = torch.tensor([[0., 0., 0.], [1., -2., 1.], [0., 0., 0.]])
            hyy = hxx.t().contiguous()
            hxy = torch.tensor([[1., 0., -1.], [0., 0., 0.], [-1., 0., 1.]]) / 4.0
            self.register_buffer("hxx", hxx.view(1, 1, 3, 3))
            self.register_buffer("hyy", hyy.view(1, 1, 3, 3))
            self.register_buffer("hxy", hxy.view(1, 1, 3, 3))

        self.to_gray = nn.Conv2d(in_ch, 1, 1, bias=False)
        nn.init.constant_(self.to_gray.weight, 1.0 / in_ch)
        self.raw_beta = nn.Parameter(torch.tensor(float(beta_init)))

    def _scharr(self, gray, dilation):
        pad = dilation
        ex = F.conv2d(gray, self.gx.float(), padding=pad, dilation=dilation)
        ey = F.conv2d(gray, self.gy.float(), padding=pad, dilation=dilation)
        return ex, ey

    def _hessian_channels(self, gray, beta):
        eps = 1e-4
        norm_traces, raw_traces, raw_dets = [], [], []
        for s in self.hessian_scales:
            pad = s
            gray_p = F.pad(gray, (pad, pad, pad, pad), mode="reflect")
            ixx = F.conv2d(gray_p, self.hxx.float(), dilation=s)
            iyy = F.conv2d(gray_p, self.hyy.float(), dilation=s)
            ixy = F.conv2d(gray_p, self.hxy.float(), dilation=s)
            raw_trace = ixx + iyy
            raw_det = ixx * iyy - ixy ** 2
            norm_traces.append((s ** 2) * raw_trace)
            raw_traces.append(raw_trace)
            raw_dets.append(raw_det)

        norm_stack = torch.stack(norm_traces, dim=0)
        raw_trace_stack = torch.stack(raw_traces, dim=0)
        raw_det_stack = torch.stack(raw_dets, dim=0)

        best_idx = norm_stack.abs().argmax(dim=0, keepdim=True)
        T_norm_sel = torch.gather(norm_stack, 0, best_idx).squeeze(0)
        trace_sel = torch.gather(raw_trace_stack, 0, best_idx).squeeze(0)
        det_sel = torch.gather(raw_det_stack, 0, best_idx).squeeze(0)

        T_relnorm = T_norm_sel / (T_norm_sel.abs().amax(dim=(2, 3), keepdim=True) + eps)
        polarity_strength = torch.tanh(beta * T_relnorm)
        blobness = (4.0 * det_sel / (trace_sel ** 2 + eps)).clamp(0.0, 1.0)
        return polarity_strength, blobness

    def forward(self, x):
        with torch.autocast(device_type=x.device.type, enabled=False):
            x32 = x.float()
            gray = F.conv2d(x32, self.to_gray.weight.float())

            if self.multiscale_mode == "blur":
                gray_scale2 = F.conv2d(gray, self.blur_kernel.float(), padding=2)
                ex1, ey1 = self._scharr(gray, dilation=1)
                ex2, ey2 = self._scharr(gray_scale2, dilation=1)
            else:  
                ex1, ey1 = self._scharr(gray, dilation=1)
                ex2, ey2 = self._scharr(gray, dilation=2)

            eps = 1e-4
            n1 = torch.sqrt(ex1 ** 2 + ey1 ** 2 + eps)
            n2 = torch.sqrt(ex2 ** 2 + ey2 ** 2 + eps)

            mag_norm = n1 / (n1.amax(dim=(2, 3), keepdim=True) + eps)
            beta = F.softplus(self.raw_beta.float()) + 1e-3
            boundary_likeness = torch.tanh(beta * mag_norm)

            u1x, u1y = ex1 / n1, ey1 / n1
            u2x, u2y = ex2 / n2, ey2 / n2
            cos_agree = (u1x * u2x + u1y * u2y).clamp(-1.0, 1.0)
            r = 0.5 * (cos_agree + 1.0)

            cos2t = u1x ** 2 - u1y ** 2
            sin2t = 2.0 * u1x * u1y

            channels = [boundary_likeness, r, r * cos2t, r * sin2t]

            if self.use_log_channel:
                log_resp = F.conv2d(gray, self.log_kernel.float(), padding=2)
                log_norm = log_resp / (log_resp.abs().amax(dim=(2, 3), keepdim=True) + eps)
                blob_likeness = torch.tanh(beta * log_norm)
                channels.append(blob_likeness)

            if self.use_hessian_channels:
                polarity, blobness = self._hessian_channels(gray, beta)
                channels.append(polarity)
                channels.append(blobness)

            psi = torch.cat(channels, dim=1)

            if not torch.isfinite(psi).all():
                psi = torch.nan_to_num(psi, nan=0.0, posinf=0.0, neginf=0.0)

        return psi.to(x.dtype)


class ScharrPreBottleneckRefine(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.descriptor = ScharrBoundaryDescriptor(
            channels, k_dim=4, use_log_channel=False, use_hessian_channels=False
        )
        self.sqrt_gamma = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        psi = self.descriptor(x)
        boundary_likeness = psi[:, 0:1]
        gamma = self.sqrt_gamma ** 2
        gate = 1.0 + gamma * boundary_likeness
        out = x * gate
        return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


class SBAAttention(nn.Module):
    """
    Phase 5: Exact Softmax SBA-Attention.
    Calculates standard visual similarity (Q * K^T) and adds explicit 
    boundary similarity (Psi * Psi^T) prior to the exact Softmax. 
    Replaces F.unfold window attention with a depthwise conv local bias.
    """
    def __init__(self, channels, heads=SBA_HEADS, k_dim=SBA_K_DIM,
                 use_log_channel=USE_LOG_CHANNEL, use_hessian_channels=USE_HESSIAN_CHANNELS,
                 multiscale_mode=BOUNDARY_DESC_MODE, hessian_scales=HESSIAN_SCALES):
        super().__init__()
        assert channels % heads == 0, "channels must be divisible by heads"
        self.heads  = heads
        self.dh     = channels // heads
        self.k_dim  = k_dim

        self.q_proj   = nn.Conv2d(channels, channels, 1, bias=False)
        self.k_proj   = nn.Conv2d(channels, channels, 1, bias=False)
        self.v_proj   = nn.Conv2d(channels, channels, 1, bias=False)
        
        # Replaces explicit F.unfold local window attention 
        self.v_local  = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        
        self.out_proj = nn.Conv2d(channels, channels, 1, bias=False)

        self.boundary = ScharrBoundaryDescriptor(
            channels, k_dim=k_dim, use_log_channel=use_log_channel,
            use_hessian_channels=use_hessian_channels,
            multiscale_mode=multiscale_mode, hessian_scales=hessian_scales,
        )

        self.sqrt_lambda = nn.Parameter(torch.full((heads,), 0.1))
        self.norm = nn.GroupNorm(min(8, channels), channels)

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        residual = x
        xn = self.norm(x)

        q = self.q_proj(xn).view(B, self.heads, self.dh, N)
        k = self.k_proj(xn).view(B, self.heads, self.dh, N)
        v_map = self.v_proj(xn)
        v = v_map.view(B, self.heads, self.dh, N)
        
        v_loc = self.v_local(v_map)

        psi = self.boundary(xn) 
        psi_flat = psi.view(B, self.k_dim, N)

        # --- FLOAT32 GUARD ---
        with torch.autocast(device_type=x.device.type, enabled=False):
            q32, k32, psi32 = q.float(), k.float(), psi_flat.float()
            
            sim_visual = torch.einsum('bhdn,bhdm->bhnm', q32, k32) * (self.dh ** -0.5)
            sim_boundary = torch.einsum('bdn,bdm->bnm', psi32, psi32)

            lam = (self.sqrt_lambda.float() ** 2).view(1, self.heads, 1, 1)
            sim_total = sim_visual + lam * sim_boundary.unsqueeze(1)

            attn = torch.softmax(sim_total, dim=-1)

        attn = attn.to(v.dtype)
        out_attn = torch.einsum('bhnm,bhdm->bhdn', attn, v)
        out_attn = out_attn.reshape(B, C, H, W)

        out = out_attn + v_loc
        out = self.out_proj(out)
        out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        
        return residual + out


# ─────────────────────────────────────────────────────────────────────────────
# 5c. Enhanced Decoder with Attention Gates
# ─────────────────────────────────────────────────────────────────────────────
class AttentionGate(nn.Module):
    def __init__(self, skip_ch, dec_ch, inter_ch=None):
        super().__init__()
        inter_ch = inter_ch or skip_ch // 2
        self.W_g = nn.Conv2d(dec_ch, inter_ch, 1, bias=False)
        self.W_x = nn.Conv2d(skip_ch, inter_ch, 1, bias=False)
        self.psi = nn.Sequential(
            nn.BatchNorm2d(inter_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_ch, 1, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

    def forward(self, x_skip, x_dec):
        g = self.W_g(x_dec)
        g = F.interpolate(g, size=x_skip.shape[2:], mode="bilinear", align_corners=False)
        x = self.W_x(x_skip)
        alpha = self.psi(g + x)
        return x_skip * alpha


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, n=2, use_ag=True, use_boundary_gate=False):
        super().__init__()
        self.use_ag = use_ag
        self.use_boundary_gate = use_boundary_gate
        
        if self.use_ag:
            self.ag = AttentionGate(skip_ch=skip_ch, dec_ch=in_ch)
        if self.use_boundary_gate:
            self.bg = ScharrPreBottleneckRefine(skip_ch)

        self.merge = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)
        )
        self.blocks = nn.Sequential(*[ResBlock(out_ch, out_ch) for _ in range(n)])

    def forward(self, x, skip):
        if self.use_boundary_gate:
            skip = self.bg(skip)
        if self.use_ag:
            skip = self.ag(skip, x)
            
        x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        return self.blocks(self.merge(torch.cat([x, skip], 1)))


def hv_watershed_postprocess(binary_mask: np.ndarray, hv_map, min_size=WATERSHED_MIN_SIZE,
                              seed_thresh=WATERSHED_SEED_THRESH,
                              seed_erosion=WATERSHED_SEED_EROSION):
    binary_mask = binary_mask.astype(bool)
    if hv_map is None:
        inst = cc_label(binary_mask, connectivity=2)
        return remove_small_objects(inst, min_size=min_size)

    h, v = hv_map[0], hv_map[1]
    sob_hx = ndi.sobel(h, axis=1)
    sob_hy = ndi.sobel(h, axis=0)
    sob_vx = ndi.sobel(v, axis=1)
    sob_vy = ndi.sobel(v, axis=0)
    grad_mag = np.sqrt(sob_hx ** 2 + sob_hy ** 2 + sob_vx ** 2 + sob_vy ** 2)
    grad_norm = grad_mag / (grad_mag.max() + 1e-8)

    seed_map = (grad_norm < seed_thresh) & binary_mask
    seed_map = erosion(seed_map, disk(seed_erosion))
    markers = cc_label(seed_map, connectivity=2)

    if markers.max() == 0:
        inst = cc_label(binary_mask, connectivity=2)
        return remove_small_objects(inst, min_size=min_size)

    inst = watershed(grad_norm, markers=markers, mask=binary_mask)
    return remove_small_objects(inst, min_size=min_size)


def aggregated_jaccard_index(gt_inst: np.ndarray, pred_inst: np.ndarray) -> float:
    gt_ids = np.unique(gt_inst); gt_ids = gt_ids[gt_ids != 0]
    pred_ids = np.unique(pred_inst); pred_ids = pred_ids[pred_ids != 0]
    if len(gt_ids) == 0:
        return 1.0 if len(pred_ids) == 0 else 0.0

    pred_areas = {pid: int((pred_inst == pid).sum()) for pid in pred_ids}
    used_pred = set()
    inter_sum, union_sum = 0, 0

    for gid in gt_ids:
        gmask = gt_inst == gid
        g_area = int(gmask.sum())
        overlap_ids, counts = np.unique(pred_inst[gmask], return_counts=True)
        best_iou, best_pid, best_inter = 0.0, None, 0
        for pid, cnt in zip(overlap_ids, counts):
            if pid == 0:
                continue
            union = g_area + pred_areas[pid] - cnt
            iou = cnt / union if union > 0 else 0.0
            if iou > best_iou:
                best_iou, best_pid, best_inter = iou, pid, int(cnt)
        if best_pid is not None:
            union = g_area + pred_areas[best_pid] - best_inter
            inter_sum += best_inter
            union_sum += union
            used_pred.add(best_pid)
        else:
            union_sum += g_area

    for pid in pred_ids:
        if pid not in used_pred:
            union_sum += pred_areas[pid]

    return inter_sum / union_sum if union_sum > 0 else 1.0


class ResNetUNetV5(nn.Module):
    def __init__(self, in_ch=3, use_hv_head=USE_HV_HEAD):
        super().__init__()
        self.use_hv_head = use_hv_head

        self.enc1 = EncoderBlock(in_ch, 64,  n=2)
        self.enc2 = EncoderBlock(64,    128, n=2)
        self.enc3 = EncoderBlock(128,   256, n=3)
        self.enc4 = EncoderBlock(256,   512, n=2)

        self.use_pre_bottleneck_refine = USE_PRE_BOTTLENECK_REFINE
        if self.use_pre_bottleneck_refine:
            self.pre_bottleneck_refine = ScharrPreBottleneckRefine(512)

        self.bottleneck = ASPPBottleneck(512, rates=(3, 6, 12))
        
        # Phase 5: Replaced SBALinearAttention with exact Softmax SBAAttention
        self.sba = SBAAttention(512, heads=SBA_HEADS, k_dim=SBA_K_DIM)

        # Phase 5: Enhanced Decoders
        self.dec4 = DecoderBlock(512, 512, 256, n=2, use_ag=USE_ATTENTION_GATES, use_boundary_gate=USE_BOUNDARY_SKIPS)
        self.dec3 = DecoderBlock(256, 256, 128, n=2, use_ag=USE_ATTENTION_GATES, use_boundary_gate=USE_BOUNDARY_SKIPS)
        self.dec2 = DecoderBlock(128, 128, 64,  n=2, use_ag=USE_ATTENTION_GATES, use_boundary_gate=USE_BOUNDARY_SKIPS)
        self.dec1 = DecoderBlock(64,   64, 32,  n=2, use_ag=USE_ATTENTION_GATES, use_boundary_gate=USE_BOUNDARY_SKIPS)

        self.head       = nn.Conv2d(32,  1, 1)
        self.aux_head3  = nn.Conv2d(128, 1, 1)
        self.aux_head2  = nn.Conv2d(64,  1, 1)

        if self.use_hv_head:
            self.hv_head = nn.Conv2d(32, 2, 1)

        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x):
        s1, x = self.enc1(x)
        s2, x = self.enc2(x)
        s3, x = self.enc3(x)
        s4, x = self.enc4(x)

        if self.use_pre_bottleneck_refine:
            x = self.pre_bottleneck_refine(x)

        x = self.bottleneck(x)
        x = self.sba(x)

        d4 = self.dec4(x,  s4)
        d3 = self.dec3(d4, s3)
        d2 = self.dec2(d3, s2)
        d1 = self.dec1(d2, s1)

        out = self.head(d1)
        hv_out = torch.tanh(self.hv_head(d1)) if self.use_hv_head else None

        if self.training:
            aux3 = F.interpolate(self.aux_head3(d3), size=d1.shape[2:],
                                  mode="bilinear", align_corners=False)
            aux2 = F.interpolate(self.aux_head2(d2), size=d1.shape[2:],
                                  mode="bilinear", align_corners=False)
            return {"seg": out, "aux3": aux3, "aux2": aux2, "hv": hv_out}

        return {"seg": out, "hv": hv_out}


# ─────────────────────────────────────────────────────────────────────────────
# 6. Losses
# ─────────────────────────────────────────────────────────────────────────────
class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, gamma=0.75,
                 bce_weight=0.4, pos_weight=5.0, smooth=1.0):
        super().__init__()
        self.alpha, self.beta, self.gamma = alpha, beta, gamma
        self.bce_w  = bce_weight
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))

    def tversky(self, logits, targets):
        p = torch.sigmoid(logits).view(logits.size(0), -1)
        t = targets.view(targets.size(0), -1)
        tp = (p * t).sum(1)
        fn = ((1 - p) * t).sum(1)
        fp = (p * (1 - t)).sum(1)
        ti = (tp + self.smooth) / (tp + self.alpha * fn + self.beta * fp + self.smooth)
        return (1 - ti).mean()

    def forward(self, logits, targets):
        targets = targets.to(logits.device)
        if self.bce.pos_weight.device != logits.device:
            self.bce.pos_weight = self.bce.pos_weight.to(logits.device)
        ft = self.tversky(logits, targets) ** self.gamma
        return (1 - self.bce_w) * ft + self.bce_w * self.bce(logits, targets)


_SOBEL_X = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]])
_SOBEL_Y = _SOBEL_X.t().contiguous()

class HVLoss(nn.Module):
    def __init__(self, mse_weight=HV_MSE_WEIGHT, msge_weight=HV_MSGE_WEIGHT):
        super().__init__()
        self.mse_w, self.msge_w = mse_weight, msge_weight
        self.register_buffer("sx", _SOBEL_X.view(1, 1, 3, 3))
        self.register_buffer("sy", _SOBEL_Y.view(1, 1, 3, 3))

    def _grad(self, hv):
        B, C, H, W = hv.shape
        hv_flat = hv.reshape(B * C, 1, H, W)
        with torch.autocast(device_type=hv.device.type, enabled=False):
            hv_flat32 = hv_flat.float()
            gx = F.conv2d(hv_flat32, self.sx.float(), padding=1)
            gy = F.conv2d(hv_flat32, self.sy.float(), padding=1)
        gx = gx.reshape(B, C, H, W).to(hv.dtype)
        gy = gy.reshape(B, C, H, W).to(hv.dtype)
        return gx, gy

    def forward(self, pred_hv, gt_hv, fg_mask):
        gt_hv = gt_hv.to(pred_hv.device)
        fg_mask = fg_mask.to(pred_hv.device)

        mse = F.mse_loss(pred_hv, gt_hv)

        pgx, pgy = self._grad(pred_hv)
        ggx, ggy = self._grad(gt_hv)
        fg = fg_mask.expand_as(pgx)
        denom = fg.sum().clamp_min(1.0)
        msge = (((pgx - ggx) ** 2 + (pgy - ggy) ** 2) * fg).sum() / denom

        return self.mse_w * mse + self.msge_w * msge


def _fixed_edge_map(x_1ch, beta=BOUNDARY_LOSS_BETA):
    with torch.autocast(device_type=x_1ch.device.type, enabled=False):
        x32 = x_1ch.float()
        gx = torch.tensor([[-3., 0., 3.], [-10., 0., 10.], [-3., 0., 3.]],
                           device=x32.device) / 16.0
        gy = gx.t()
        ex = F.conv2d(x32, gx.view(1, 1, 3, 3), padding=1)
        ey = F.conv2d(x32, gy.view(1, 1, 3, 3), padding=1)
        mag = torch.sqrt(ex ** 2 + ey ** 2 + 1e-8)
        mag_norm = mag / (mag.amax(dim=(2, 3), keepdim=True) + 1e-8)
        edge = torch.tanh(beta * mag_norm)
    return edge.to(x_1ch.dtype)


class BoundaryConsistencyLoss(nn.Module):
    def __init__(self, weight=BOUNDARY_LOSS_WEIGHT, beta=BOUNDARY_LOSS_BETA):
        super().__init__()
        self.weight, self.beta = weight, beta

    def forward(self, logits, targets):
        targets = targets.to(logits.device)
        prob = torch.sigmoid(logits)
        pred_edge = _fixed_edge_map(prob, self.beta)
        with torch.no_grad():
            gt_edge = _fixed_edge_map(targets, self.beta)
        return self.weight * F.l1_loss(pred_edge, gt_edge)


class CombinedLoss(nn.Module):
    def __init__(self, pos_weight, use_hv=USE_HV_HEAD, use_boundary=USE_BOUNDARY_LOSS):
        super().__init__()
        self.seg_loss = FocalTverskyLoss(alpha=0.7, beta=0.3, gamma=0.75,
                                          bce_weight=0.4, pos_weight=pos_weight)
        self.use_hv = use_hv
        self.use_boundary = use_boundary
        if self.use_hv:
            self.hv_loss = HVLoss(mse_weight=HV_MSE_WEIGHT, msge_weight=HV_MSGE_WEIGHT)
        if self.use_boundary:
            self.boundary_loss = BoundaryConsistencyLoss(weight=BOUNDARY_LOSS_WEIGHT,
                                                           beta=BOUNDARY_LOSS_BETA)

    def forward(self, model_out, masks, hv_targets=None):
        loss = self.seg_loss(model_out["seg"], masks)
        if "aux3" in model_out and model_out["aux3"] is not None:
            loss = loss + 0.4 * self.seg_loss(model_out["aux3"], masks)
        if "aux2" in model_out and model_out["aux2"] is not None:
            loss = loss + 0.2 * self.seg_loss(model_out["aux2"], masks)

        components = {"seg": loss.detach().clone()}

        if self.use_hv and model_out.get("hv") is not None and hv_targets is not None:
            hv_l = self.hv_loss(model_out["hv"], hv_targets, masks)
            loss = loss + HV_TOTAL_WEIGHT * hv_l
            components["hv"] = hv_l.detach()

        if self.use_boundary:
            b_l = self.boundary_loss(model_out["seg"], masks)
            loss = loss + b_l
            components["boundary"] = b_l.detach()

        return loss, components


# ─────────────────────────────────────────────────────────────────────────────
# 7. Metrics
# ─────────────────────────────────────────────────────────────────────────────
class SegMetrics:
    def __init__(self, thr=0.5): self.thr = thr; self.reset()
    def reset(self): self.tp = self.fp = self.fn = self.tn = 0.0

    @torch.no_grad()
    def update(self, logits, targets):
        p = (torch.sigmoid(logits) > self.thr).float()
        t = targets.to(p.device)
        self.tp += (p * t).sum().item()
        self.fp += (p * (1 - t)).sum().item()
        self.fn += ((1 - p) * t).sum().item()
        self.tn += ((1 - p) * (1 - t)).sum().item()

    def compute(self):
        e = 1e-7
        tp, fp, fn, tn = self.tp, self.fp, self.fn, self.tn
        return dict(
            dice=(2 * tp) / (2 * tp + fp + fn + e),
            iou=tp / (tp + fp + fn + e),
            precision=tp / (tp + fp + e),
            recall=tp / (tp + fn + e),
            f1=(2 * tp) / (2 * tp + fp + fn + e),
            specificity=tn / (tn + fp + e),
        )


class InstanceMetrics:
    def __init__(self): self.reset()
    def reset(self):
        self.ajis = []
        self.merge_events = 0
        self.n_patches = 0

    def update(self, gt_inst: np.ndarray, pred_inst: np.ndarray):
        self.ajis.append(aggregated_jaccard_index(gt_inst, pred_inst))
        n_gt = len(np.unique(gt_inst)) - (1 if 0 in gt_inst else 0)
        n_pred = len(np.unique(pred_inst)) - (1 if 0 in pred_inst else 0)
        if n_pred < n_gt:
            self.merge_events += 1
        self.n_patches += 1

    def compute(self):
        if self.n_patches == 0:
            return {"aji": float("nan"), "merge_rate": float("nan")}
        return {
            "aji": float(np.mean(self.ajis)),
            "merge_rate": self.merge_events / self.n_patches,
        }


# ─────────────────────────────────────────────────────────────────────────────
# 8. EMA (warmup decay)
# ─────────────────────────────────────────────────────────────────────────────
class ModelEMA:
    def __init__(self, model, decay=EMA_DECAY, warmup=True):
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.decay = decay
        self.warmup = warmup
        self.updates = 0

    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        d = self.decay
        if self.warmup:
            d = min(self.decay, (self.updates) / (self.updates + 10))
        msd = model.state_dict()
        for k, v in self.module.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(d).add_(msd[k].detach(), alpha=1 - d)
            else:
                v.copy_(msd[k])


# ─────────────────────────────────────────────────────────────────────────────
# 9. Training loop
# ─────────────────────────────────────────────────────────────────────────────
def run_epoch(model, loader, criterion, metrics, optimizer=None, scheduler=None,
              scaler=None, ema=None, device=DEVICE, phase="train", verbose=True):
    is_train = (phase == "train")
    model.train(is_train)
    metrics.reset()
    total_loss = 0.0
    n_batches = len(loader)
    last_components = {}

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for i, batch in enumerate(loader, start=1):
            imgs  = batch["image"].to(device)
            masks = batch["mask"].to(device)
            hv_t  = batch["hv"].to(device)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=USE_AMP and device.type == "cuda"):
                    out = model(imgs)
                    loss, components = criterion(out, masks, hv_t)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                if scheduler is not None: scheduler.step()
                if ema is not None: ema.update(model)
            else:
                with torch.autocast(device_type=device.type, enabled=USE_AMP and device.type == "cuda"):
                    out = model(imgs)
                    loss, components = criterion(out, masks, hv_t)

            total_loss += loss.item() * imgs.size(0)
            metrics.update(out["seg"], masks)
            last_components = components

            if verbose and (i % PRINT_EVERY == 0 or i == n_batches):
                m = metrics.compute()
                comp_str = "  ".join(f"{k}={v.item():.4f}" for k, v in last_components.items())
                print(f"    [{phase:>5}] batch {i:4d}/{n_batches}  "
                      f"loss={loss.item():.4f}  ({comp_str})  "
                      f"dice={m['dice']:.4f}  iou={m['iou']:.4f}")

    return total_loss / len(loader.dataset), metrics.compute()


@torch.no_grad()
def evaluate(model, loader, metrics, device=DEVICE, verbose=True,
             instance_metrics=None, watershed_thr=0.5):
    model.eval()
    metrics.reset()
    n_batches = len(loader)
    for i, batch in enumerate(loader, start=1):
        imgs  = batch["image"].to(device)
        masks = batch["mask"].to(device)

        out = model(imgs)
        probs = torch.sigmoid(out["seg"])
        metrics.update(out["seg"], masks)

        if instance_metrics is not None and out.get("hv") is not None:
            hv_np = out["hv"].cpu().numpy()
            binary_np = (probs.cpu().numpy() > watershed_thr)
            gt_np = masks.cpu().numpy() > 0.5
            for b in range(imgs.size(0)):
                pred_inst = hv_watershed_postprocess(binary_np[b, 0], hv_np[b])
                gt_inst = cc_label(gt_np[b, 0], connectivity=2)  
                instance_metrics.update(gt_inst, pred_inst)

        if verbose and (i % PRINT_EVERY == 0 or i == n_batches):
            print(f"    [  test] batch {i:4d}/{n_batches}")

    return metrics.compute()


# ─────────────────────────────────────────────────────────────────────────────
# 10a. MoNuSeg entry point 
# ─────────────────────────────────────────────────────────────────────────────
def resolve_dir(d: Path, label: str):
    d = Path(d)
    if not d.exists():
        raise FileNotFoundError(f"{label} directory not found: {d}")
    return d


def train(train_img_dir=TRAIN_IMG_DIR, train_mask_dir=TRAIN_MASK_DIR,
          test_img_dir=TEST_IMG_DIR, test_mask_dir=TEST_MASK_DIR,
          img_size=IMG_SIZE, batch_size=BATCH_SIZE, num_epochs=NUM_EPOCHS,
          lr=LR, weight_decay=WEIGHT_DECAY,
          use_weighted_sampler=False, checkpoint="best_model_v5.pth", seed=SEED):

    for d, label in [(train_img_dir, "train images"), (train_mask_dir, "train masks"),
                      (test_img_dir, "test images"), (test_mask_dir, "test masks")]:
        resolve_dir(d, label)

    print(f"\n{'='*62}\n  ResNet-UNet v5  |  MoNuSeg official split (Train/Test)")
    print(f"  Device : {DEVICE}  |  img_size={img_size}  batch={batch_size}")
    print(f"  Epochs : {num_epochs}  (no early stopping — full run every time)")
    print(f"  Phase5 : attention_gates={USE_ATTENTION_GATES}  boundary_skips={USE_BOUNDARY_SKIPS}")
    print(f"  SBA    : heads={SBA_HEADS}  k_dim={SBA_K_DIM}  (Exact Softmax, No Linear Approx)  "
          f"desc_mode={BOUNDARY_DESC_MODE}  log_channel={USE_LOG_CHANNEL}  "
          f"hessian_channels={USE_HESSIAN_CHANNELS}")
    print(f"  Phase1 : HV head={USE_HV_HEAD}  cc_fallback={USE_CC_FALLBACK}")
    print(f"  Phase3 : boundary_loss={USE_BOUNDARY_LOSS}  weight={BOUNDARY_LOSS_WEIGHT}")
    print(f"  Phase4 : pre_bottleneck_refine={USE_PRE_BOTTLENECK_REFINE}  "
          f"hessian_scales={HESSIAN_SCALES}")
    print(f"{'='*62}")

    diagnose_masks(Path(train_mask_dir))

    train_ds, val_ds = build_train_val(train_img_dir, train_mask_dir, img_size,
                                        val_frac=VAL_FRAC, seed=seed)
    test_ds = build_test(test_img_dir, test_mask_dir, img_size)

    sampler = fg_weighted_sampler(train_ds) if use_weighted_sampler else None
    kw = dict(num_workers=4, pin_memory=True, worker_init_fn=_worker_init_fn)
    train_loader = DataLoader(train_ds, batch_size, shuffle=(sampler is None),
                               sampler=sampler, drop_last=True, **kw)
    val_loader   = DataLoader(val_ds,   batch_size, shuffle=False, **kw)
    test_loader  = DataLoader(test_ds,  batch_size, shuffle=False, **kw)

    model, history, te = _run_training_loop(
        train_loader, val_loader, test_loader, train_ds,
        num_epochs, lr, weight_decay, checkpoint,
        header="MoNuSeg Test set"
    )
    return model, history, te


# ─────────────────────────────────────────────────────────────────────────────
# 10b. TNBC entry point
# ─────────────────────────────────────────────────────────────────────────────
def train_official_split(train_img_dir=TNBC_TRAIN_IMG_DIR, train_mask_dir=TNBC_TRAIN_MASK_DIR,
                          val_img_dir=TNBC_VAL_IMG_DIR,     val_mask_dir=TNBC_VAL_MASK_DIR,
                          test_img_dir=TNBC_TEST_IMG_DIR,   test_mask_dir=TNBC_TEST_MASK_DIR,
                          img_size=IMG_SIZE, batch_size=BATCH_SIZE, num_epochs=NUM_EPOCHS,
                          lr=LR, weight_decay=WEIGHT_DECAY,
                          use_weighted_sampler=False, checkpoint="best_model_v5_tnbc.pth",
                          seed=SEED):
    print(f"\n{'='*62}\n  ResNet-UNet v5  |  TNBC official Train/Validation/Test split")
    print(f"  Device : {DEVICE}  |  img_size={img_size}  batch={batch_size}")
    print(f"  Phase5 : attention_gates={USE_ATTENTION_GATES}  boundary_skips={USE_BOUNDARY_SKIPS}")
    print(f"  SBA    : heads={SBA_HEADS}  k_dim={SBA_K_DIM}  (Exact Softmax, No Linear Approx)  "
          f"desc_mode={BOUNDARY_DESC_MODE}  log_channel={USE_LOG_CHANNEL}  "
          f"hessian_channels={USE_HESSIAN_CHANNELS}")
    print(f"{'='*62}")

    diagnose_masks(Path(train_mask_dir))

    train_ds, val_ds, test_ds = build_official_split(
        train_img_dir, train_mask_dir,
        val_img_dir, val_mask_dir,
        test_img_dir, test_mask_dir,
        img_size=img_size,
    )

    sampler = fg_weighted_sampler(train_ds) if use_weighted_sampler else None
    kw = dict(num_workers=4, pin_memory=True, worker_init_fn=_worker_init_fn)
    train_loader = DataLoader(train_ds, batch_size, shuffle=(sampler is None),
                               sampler=sampler, drop_last=True, **kw)
    val_loader   = DataLoader(val_ds,   batch_size, shuffle=False, **kw)
    test_loader  = DataLoader(test_ds,  batch_size, shuffle=False, **kw)

    model, history, te = _run_training_loop(
        train_loader, val_loader, test_loader, train_ds,
        num_epochs, lr, weight_decay, checkpoint,
        header="TNBC Test set"
    )
    return model, history, te


# ─────────────────────────────────────────────────────────────────────────────
# 10c. Shared training loop body
# ─────────────────────────────────────────────────────────────────────────────
def _run_training_loop(train_loader, val_loader, test_loader, train_ds,
                        num_epochs, lr, weight_decay, checkpoint, header):
    pos_weight = compute_pos_weight(train_ds.masks)

    model     = ResNetUNetV5(use_hv_head=USE_HV_HEAD).to(DEVICE)
    ema       = ModelEMA(model, decay=EMA_DECAY, warmup=True)
    criterion = CombinedLoss(pos_weight, use_hv=USE_HV_HEAD, use_boundary=USE_BOUNDARY_LOSS).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler    = torch.cuda.amp.GradScaler(enabled=USE_AMP and DEVICE.type == "cuda")

    total_steps = num_epochs * len(train_loader)
    scheduler   = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, total_steps=total_steps,
        pct_start=0.3, anneal_strategy="cos", div_factor=10, final_div_factor=100
    )

    train_m, val_m = SegMetrics(), SegMetrics()
    best_dice = -1.0
    history = {k: [] for k in ["tr_loss", "vl_loss", "tr_dice", "vl_dice", "tr_iou", "vl_iou"]}

    for ep in range(1, num_epochs + 1):
        print(f"\nEpoch [{ep:03d}/{num_epochs}]")
        tr_loss, tr = run_epoch(model, train_loader, criterion, train_m,
                                 optimizer, scheduler, scaler, ema, DEVICE, "train")
        vl_loss, vl = run_epoch(ema.module, val_loader, criterion, val_m,
                                 scaler=scaler, device=DEVICE, phase="val")

        for k, v in zip(["tr_loss", "vl_loss", "tr_dice", "vl_dice", "tr_iou", "vl_iou"],
                         [tr_loss, vl_loss, tr["dice"], vl["dice"], tr["iou"], vl["iou"]]):
            history[k].append(v)

        print(f"  Train        loss={tr_loss:.4f}  dice={tr['dice']:.4f}  iou={tr['iou']:.4f}")
        print(f"  Val (EMA)    loss={vl_loss:.4f}  dice={vl['dice']:.4f}  iou={vl['iou']:.4f}")

        if vl["dice"] > best_dice:
            best_dice = vl["dice"]
            torch.save({"epoch": ep, "state": ema.module.state_dict(),
                        "val_dice": best_dice, "use_hv_head": USE_HV_HEAD},
                       checkpoint)
            print(f"  ✓ Saved EMA checkpoint  (val Dice = {best_dice:.4f})")

    print(f"\nLoading best EMA checkpoint ({checkpoint}) for test …")
    ckpt = torch.load(checkpoint, map_location=DEVICE)
    model.load_state_dict(ckpt["state"])

    best_thr, _ = find_best_threshold(model, val_loader, device=DEVICE)

    test_m = SegMetrics(thr=best_thr)
    inst_m = InstanceMetrics() if USE_HV_HEAD else None
    te = evaluate(model, test_loader, test_m, device=DEVICE,
                  instance_metrics=inst_m, watershed_thr=best_thr)

    print(f"\n{'='*62}\n  TEST RESULTS ({header}, "
          f"thr={best_thr:.2f})\n{'='*62}")
    for name, val in [("Dice (F1)", te["dice"]), ("IoU", te["iou"]),
                       ("Precision", te["precision"]), ("Recall", te["recall"]),
                       ("Specificity", te["specificity"])]:
        print(f"  {name:<14}: {val:.4f}")
    if inst_m is not None:
        inst_scores = inst_m.compute()
        print(f"  {'AJI (approx GT)':<14}: {inst_scores['aji']:.4f}")
        print(f"  {'Merge rate':<14}: {inst_scores['merge_rate']:.4f}")
    print(f"{'='*62}\n")

    return model, history, te

if __name__ == "__main__":
    train_official_split()
