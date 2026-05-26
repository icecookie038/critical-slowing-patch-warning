# train_patch_model.py
# -*- coding: utf-8 -*-

import os
import sys
import csv
import argparse
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from torch.cuda.amp import autocast, GradScaler

from sklearn.metrics import (
    r2_score,
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    confusion_matrix,
)

from tqdm import tqdm

from seir_model import generate_patch_dataset
from features.dynamic_patch_indicators import extract_dynamic_feature_matrix
from features.feature_groups import get_dynamic_columns


# =========================
# 默认参数
# =========================
SEED = 42
TRAIN_RATIO = 0.8

DEFAULT_INPUT_SEQ_LEN = 10
DEFAULT_HORIZON = 30

EARLY_STOP_PATIENCE = 12
MIN_EPOCHS = 5
FEATURE_GROUPS = [
    "legacy",
    "image_only",
    "classic_patch_only",
    "dynamic_patch_only",
    "classic_dynamic_patch",
    "full",
]

# =========================
# 随机种子
# =========================
def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =========================
# Dataset
# =========================
class PatchSequenceDataset(Dataset):
    def __init__(
        self,
        X_img,
        X_patch,
        y_remaining,
        y_risk,
        train=True,
    ):
        self.X_img = torch.tensor(X_img, dtype=torch.float32)
        self.X_patch = torch.tensor(X_patch, dtype=torch.float32)
        self.y_remaining = torch.tensor(y_remaining, dtype=torch.float32)
        self.y_risk = torch.tensor(y_risk, dtype=torch.float32)
        self.train = train

    def __len__(self):
        return len(self.y_remaining)

    def __getitem__(self, idx):
        img = self.X_img[idx]          # (T, C, L, L)
        patch = self.X_patch[idx]      # (T, F)
        y_remain = self.y_remaining[idx]
        y_risk = self.y_risk[idx]

        if self.train:
            # 空间翻转增强：不会改变临界慢化性质
            if torch.rand(1).item() > 0.5:
                img = torch.flip(img, dims=[-1])

            if torch.rand(1).item() > 0.5:
                img = torch.flip(img, dims=[-2])

        return img, patch, y_remain, y_risk


# =========================
# 模型
# =========================
class SpatialCNN(nn.Module):
    def __init__(self, in_channels=1, out_dim=64):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.fc = nn.Sequential(
            nn.Linear(64, out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
        )

    def forward(self, x):
        feat = self.conv(x)
        feat = feat.view(feat.size(0), -1)
        return self.fc(feat)


class PatchCriticalPredictor(nn.Module):
    """
    输入：
        x_img:   (B, T, C, L, L)
        x_patch: (B, T, F)

    输出：
        remaining_pred: 归一化 remaining，范围 0~1
        risk_logit:     未来 horizon 内进入临界慢化状态的 logit

    input_mode:
        full       使用图像 + 斑块指标
        img_only   只使用图像，斑块指标置零
        patch_only 只使用斑块指标，图像特征置零
    """

    def __init__(
        self,
        in_channels=1,
        patch_feature_dim=20,
        cnn_out_dim=64,
        rnn_hidden=64,
        dropout=0.50,
        input_mode="full",
    ):
        super().__init__()

        if input_mode not in ["full", "img_only", "patch_only"]:
            raise ValueError("input_mode must be one of: full, img_only, patch_only")

        self.input_mode = input_mode

        self.cnn = SpatialCNN(
            in_channels=in_channels,
            out_dim=cnn_out_dim,
        )

        self.rnn = nn.GRU(
            input_size=cnn_out_dim + patch_feature_dim,
            hidden_size=rnn_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.attention = nn.Linear(rnn_hidden * 2, 1)

        self.head = nn.Sequential(
            nn.Linear(rnn_hidden * 2, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        self.remaining_head = nn.Linear(64, 1)
        self.risk_head = nn.Linear(64, 1)

    def forward(self, x_img, x_patch):
        B, T, C, H, W = x_img.shape

        img_flat = x_img.view(B * T, C, H, W)
        img_feat = self.cnn(img_flat)
        img_feat = img_feat.view(B, T, -1)

        if self.input_mode == "img_only":
            x_patch = torch.zeros_like(x_patch)

        elif self.input_mode == "patch_only":
            img_feat = torch.zeros_like(img_feat)

        seq = torch.cat([img_feat, x_patch], dim=-1)

        rnn_out, _ = self.rnn(seq)

        attn = torch.softmax(self.attention(rnn_out), dim=1)
        context = torch.sum(attn * rnn_out, dim=1)

        h = self.head(context)

        # remaining 是 0~1 归一化值，所以使用 sigmoid 限制范围
        remaining_pred = torch.sigmoid(self.remaining_head(h))

        risk_logit = self.risk_head(h)

        return remaining_pred, risk_logit


# =========================
# Trainer
# =========================
class Trainer:
    def __init__(
        self,
        model,
        device,
        sim_steps,
        lr=1e-4,
        weight_decay=1e-3,
        use_amp=True,
        pos_weight=None,
        reg_weight=0.0,
        cls_weight=1.0,
    ):
        self.model = model.to(device)
        self.device = device
        self.sim_steps = sim_steps

        self.reg_weight = float(reg_weight)
        self.cls_weight = float(cls_weight)

        self.reg_loss = nn.SmoothL1Loss()

        if pos_weight is not None:
            pos_weight_tensor = torch.tensor(
                [pos_weight],
                dtype=torch.float32,
                device=device,
            )
            self.cls_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        else:
            self.cls_loss = nn.BCEWithLogitsLoss()

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            patience=4,
            factor=0.5,
        )

        self.use_amp = bool(use_amp and device.type == "cuda")
        self.scaler = GradScaler(enabled=self.use_amp)

        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_reg": [],
            "train_cls": [],
            "val_reg": [],
            "val_cls": [],
        }

    def _remaining_target_transform(self, y_remaining):
        """
        log 归一化：
            y = log(1 + remaining) / log(1 + sim_steps)
        """
        return torch.log1p(y_remaining) / np.log1p(float(self.sim_steps))

    def _compute_loss(self, pred_remaining_norm, risk_logit, y_remaining, y_risk):
        y_remaining_norm = self._remaining_target_transform(y_remaining)

        loss_reg = self.reg_loss(pred_remaining_norm, y_remaining_norm)
        loss_cls = self.cls_loss(risk_logit, y_risk)

        loss = self.reg_weight * loss_reg + self.cls_weight * loss_cls

        return loss, loss_reg, loss_cls

    def train_epoch(self, loader, epoch):
        self.model.train()

        total_loss = 0.0
        total_reg = 0.0
        total_cls = 0.0

        pbar = tqdm(loader, desc=f"Epoch {epoch + 1} train", leave=False)

        for x_img, x_patch, y_remaining, y_risk in pbar:
            x_img = x_img.to(self.device, non_blocking=True)
            x_patch = x_patch.to(self.device, non_blocking=True)

            y_remaining = y_remaining.to(self.device, non_blocking=True).view(-1, 1)
            y_risk = y_risk.to(self.device, non_blocking=True).view(-1, 1)

            self.optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=self.use_amp):
                pred_remaining_norm, risk_logit = self.model(x_img, x_patch)

                loss, loss_reg, loss_cls = self._compute_loss(
                    pred_remaining_norm,
                    risk_logit,
                    y_remaining,
                    y_risk,
                )

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += float(loss.item())
            total_reg += float(loss_reg.item())
            total_cls += float(loss_cls.item())

            pbar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "reg": f"{loss_reg.item():.4f}",
                    "cls": f"{loss_cls.item():.4f}",
                }
            )

        n = max(1, len(loader))
        return total_loss / n, total_reg / n, total_cls / n

    @torch.no_grad()
    def validate(self, loader):
        self.model.eval()

        total_loss = 0.0
        total_reg = 0.0
        total_cls = 0.0

        for x_img, x_patch, y_remaining, y_risk in loader:
            x_img = x_img.to(self.device, non_blocking=True)
            x_patch = x_patch.to(self.device, non_blocking=True)

            y_remaining = y_remaining.to(self.device, non_blocking=True).view(-1, 1)
            y_risk = y_risk.to(self.device, non_blocking=True).view(-1, 1)

            with autocast(enabled=self.use_amp):
                pred_remaining_norm, risk_logit = self.model(x_img, x_patch)

                loss, loss_reg, loss_cls = self._compute_loss(
                    pred_remaining_norm,
                    risk_logit,
                    y_remaining,
                    y_risk,
                )

            total_loss += float(loss.item())
            total_reg += float(loss_reg.item())
            total_cls += float(loss_cls.item())

        n = max(1, len(loader))
        return total_loss / n, total_reg / n, total_cls / n

    def fit(self, train_loader, val_loader, epochs, save_path):
        best_val = float("inf")
        best_epoch = -1
        patience = 0

        for epoch in range(epochs):
            train_loss, train_reg, train_cls = self.train_epoch(train_loader, epoch)
            val_loss, val_reg, val_cls = self.validate(val_loader)

            self.scheduler.step(val_loss)

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_reg"].append(train_reg)
            self.history["train_cls"].append(train_cls)
            self.history["val_reg"].append(val_reg)
            self.history["val_cls"].append(val_cls)

            print(
                f"Epoch {epoch + 1:03d} | "
                f"Train {train_loss:.5f} "
                f"(Reg {train_reg:.5f}, Cls {train_cls:.5f}) | "
                f"Val {val_loss:.5f} "
                f"(Reg {val_reg:.5f}, Cls {val_cls:.5f})"
            )

            if val_loss < best_val - 1e-5:
                best_val = val_loss
                best_epoch = epoch + 1
                patience = 0

                torch.save(
                    {
                        "model_state_dict": self.model.state_dict(),
                        "best_val": best_val,
                        "best_epoch": best_epoch,
                    },
                    save_path,
                )
            else:
                patience += 1

            if epoch + 1 >= MIN_EPOCHS and patience >= EARLY_STOP_PATIENCE:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        print(f"Best epoch: {best_epoch}, best val loss: {best_val:.6f}")

        checkpoint = torch.load(save_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])

        return self.model


# =========================
# 数据处理
# =========================
def clean_dataset(X_img, X_patch, y_remaining, y_risk, sim_id):
    mask = (
        np.isfinite(X_img).all(axis=tuple(range(1, X_img.ndim)))
        & np.isfinite(X_patch).all(axis=(1, 2))
        & np.isfinite(y_remaining)
        & np.isfinite(y_risk)
        & np.isfinite(sim_id)
    )

    X_img = X_img[mask]
    X_patch = X_patch[mask]
    y_remaining = y_remaining[mask]
    y_risk = y_risk[mask]
    sim_id = sim_id[mask]

    X_img = np.clip(X_img, 0.0, 1.0)
    y_remaining = np.maximum(y_remaining, 1.0)
    y_risk = np.clip(y_risk, 0.0, 1.0)

    return X_img, X_patch, y_remaining, y_risk, sim_id


def split_by_sim_id(sim_id, train_ratio=0.8, seed=42):
    rng = np.random.default_rng(seed)

    unique_sims = np.unique(sim_id)
    rng.shuffle(unique_sims)

    n_train = int(len(unique_sims) * train_ratio)

    train_sims = set(unique_sims[:n_train].tolist())
    val_sims = set(unique_sims[n_train:].tolist())

    tr_idx = np.array([i for i, sid in enumerate(sim_id) if int(sid) in train_sims])
    val_idx = np.array([i for i, sid in enumerate(sim_id) if int(sid) in val_sims])

    return tr_idx, val_idx


def standardize_patch_features(X_patch, tr_idx):
    mean = X_patch[tr_idx].mean(axis=(0, 1), keepdims=True)
    std = X_patch[tr_idx].std(axis=(0, 1), keepdims=True) + 1e-8

    X_patch_norm = (X_patch - mean) / std

    X_patch_norm = np.nan_to_num(
        X_patch_norm,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(np.float32)

    return X_patch_norm, mean.astype(np.float32), std.astype(np.float32)
def build_dynamic_patch_sequence_features(
    X_img,
    threshold=0.5,
    baseline_end=3,
    mode="expansion",
):
    """
    Build dynamic patch feature sequence from image sequence.

    Input:
        X_img shape: (N, T, C, H, W)

    Output:
        X_dynamic shape: (N, T, 4)

    For SEIR expansion, the dynamic features are:
        PDSI + DPCI + FAI + PSII

    DPCR is computed inside dynamic_patch_indicators.py,
    but it is not used as the main SEIR expansion feature.
    """
    if X_img.ndim != 5:
        raise ValueError(f"X_img must have shape (N,T,C,H,W), got {X_img.shape}")

    n_samples, seq_len, channels, height, width = X_img.shape

    dynamic_columns = get_dynamic_columns(mode=mode)

    dynamic_features = []

    for i in range(n_samples):
        # Use the first channel as the infection field / binary mask.
        field_series = X_img[i, :, 0, :, :]

        _, df = extract_dynamic_feature_matrix(
            field_series=field_series,
            threshold=threshold,
            positive=True,
            connectivity=8,
            lag=1,
            mode=mode,
            baseline_end=baseline_end,
            feature_columns=dynamic_columns,
        )

        feature_matrix = df[dynamic_columns].to_numpy(dtype=np.float32)

        if feature_matrix.shape[0] != seq_len:
            raise RuntimeError(
                f"Dynamic feature length mismatch: expected {seq_len}, "
                f"got {feature_matrix.shape[0]}"
            )

        dynamic_features.append(feature_matrix)

    X_dynamic = np.stack(dynamic_features, axis=0).astype(np.float32)

    X_dynamic = np.nan_to_num(
        X_dynamic,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    return X_dynamic
def apply_feature_group_to_dataset(
    X_img,
    X_patch,
    feature_group="legacy",
    input_mode="full",
):
    """
    Convert original v0.7 dataset into v0.9 feature-group dataset.

    Original dataset:
        X_img   = image sequence
        X_patch = classic patch features

    v0.9 groups:
        image_only:
            image + zero classic patch

        classic_patch_only:
            zero image + classic patch

        dynamic_patch_only:
            zero image + dynamic patch

        classic_dynamic_patch:
            zero image + classic + dynamic patch

        full:
            image + classic + dynamic patch

    legacy:
        keep original v0.7 behavior.
    """
    if feature_group == "legacy":
        return X_img, X_patch, input_mode

    if feature_group not in [
        "image_only",
        "classic_patch_only",
        "dynamic_patch_only",
        "classic_dynamic_patch",
        "full",
    ]:
        raise ValueError(f"Unknown feature_group: {feature_group}")

    print()
    print(f"Building v0.9 feature group: {feature_group}")

    X_dynamic = build_dynamic_patch_sequence_features(
        X_img=X_img,
        threshold=0.5,
        baseline_end=max(3, min(5, X_img.shape[1] // 2)),
        mode="expansion",
    )

    if feature_group == "image_only":
        new_X_img = X_img
        new_X_patch = np.zeros_like(X_patch)
        new_input_mode = "img_only"

    elif feature_group == "classic_patch_only":
        new_X_img = np.zeros_like(X_img)
        new_X_patch = X_patch
        new_input_mode = "patch_only"

    elif feature_group == "dynamic_patch_only":
        new_X_img = np.zeros_like(X_img)
        new_X_patch = X_dynamic
        new_input_mode = "patch_only"

    elif feature_group == "classic_dynamic_patch":
        new_X_img = np.zeros_like(X_img)
        new_X_patch = np.concatenate([X_patch, X_dynamic], axis=-1)
        new_input_mode = "patch_only"

    elif feature_group == "full":
        new_X_img = X_img
        new_X_patch = np.concatenate([X_patch, X_dynamic], axis=-1)
        new_input_mode = "full"

    else:
        raise RuntimeError(f"Unhandled feature_group: {feature_group}")

    print(f"Original X_img:   {X_img.shape}")
    print(f"Original X_patch: {X_patch.shape}")
    print(f"Dynamic X_patch:  {X_dynamic.shape}")
    print(f"New X_img:        {new_X_img.shape}")
    print(f"New X_patch:      {new_X_patch.shape}")
    print(f"Model input_mode: {new_input_mode}")

    return new_X_img, new_X_patch, new_input_mode

# =========================
# 评估
# =========================
@torch.no_grad()
def predict(model, loader, device, sim_steps):
    model.eval()

    remaining_preds = []
    risk_probs = []
    remaining_true = []
    risk_true = []

    for x_img, x_patch, y_remaining, y_risk in loader:
        x_img = x_img.to(device)
        x_patch = x_patch.to(device)

        pred_remaining_norm, risk_logit = model(x_img, x_patch)

        pred_norm = pred_remaining_norm.detach().cpu().numpy().reshape(-1)
        pred_norm = np.clip(pred_norm, 0.0, 1.0)

        pred_remaining = np.expm1(
            pred_norm * np.log1p(float(sim_steps))
        )

        pred_remaining = np.clip(pred_remaining, 0.0, float(sim_steps))

        prob = torch.sigmoid(risk_logit).detach().cpu().numpy().reshape(-1)

        remaining_preds.append(pred_remaining)
        risk_probs.append(prob)

        remaining_true.append(y_remaining.numpy().reshape(-1))
        risk_true.append(y_risk.numpy().reshape(-1))

    remaining_preds = np.concatenate(remaining_preds)
    risk_probs = np.concatenate(risk_probs)
    remaining_true = np.concatenate(remaining_true)
    risk_true = np.concatenate(risk_true)

    return remaining_true, remaining_preds, risk_true, risk_probs


def compute_metrics(y_remain, pred_remain, y_risk, risk_prob):
    y_remain = np.asarray(y_remain, dtype=np.float64)
    pred_remain = np.asarray(pred_remain, dtype=np.float64)
    y_risk = np.asarray(y_risk, dtype=np.float32)
    risk_prob = np.asarray(risk_prob, dtype=np.float32)

    errors = pred_remain - y_remain

    rmse = float(np.sqrt(np.mean(errors ** 2)))
    mae = float(np.mean(np.abs(errors)))

    try:
        r2 = float(r2_score(y_remain, pred_remain))
    except Exception:
        r2 = float("nan")

    if len(np.unique(y_risk)) >= 2:
        auc = float(roc_auc_score(y_risk, risk_prob))
        auprc = float(average_precision_score(y_risk, risk_prob))
    else:
        auc = float("nan")
        auprc = float("nan")

    best_f1 = -1.0
    best_threshold = 0.5
    best_acc = 0.0
    best_precision = 0.0
    best_recall = 0.0
    best_tn, best_fp, best_fn, best_tp = 0, 0, 0, 0

    for th in np.linspace(0.05, 0.95, 91):
        pred_label = (risk_prob >= th).astype(np.float32)

        f1_tmp = f1_score(y_risk, pred_label, zero_division=0)

        if f1_tmp > best_f1:
            best_f1 = float(f1_tmp)
            best_threshold = float(th)
            best_acc = float(accuracy_score(y_risk, pred_label))
            best_precision = float(
                precision_score(y_risk, pred_label, zero_division=0)
            )
            best_recall = float(
                recall_score(y_risk, pred_label, zero_division=0)
            )

            cm = confusion_matrix(y_risk, pred_label, labels=[0, 1])
            best_tn, best_fp, best_fn, best_tp = cm.ravel()

    return {
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "AUC": auc,
        "AUPRC": auprc,
        "ACC": best_acc,
        "F1": best_f1,
        "Precision": best_precision,
        "Recall": best_recall,
        "Best_Threshold": best_threshold,
        "Positive_Rate": float(np.mean(y_risk)),
        "TN": int(best_tn),
        "FP": int(best_fp),
        "FN": int(best_fn),
        "TP": int(best_tp),
    }


# =========================
# 可视化
# =========================
def plot_training_history(history, save_path):
    plt.figure(figsize=(10, 4))

    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training History")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_predictions(y_true, y_pred, save_path):
    plt.figure(figsize=(5, 5))

    plt.scatter(y_true, y_pred, s=8, alpha=0.5)

    max_v = max(float(np.max(y_true)), float(np.max(y_pred)), 1.0)
    plt.plot([0, max_v], [0, max_v], linestyle="--")

    plt.xlabel("True remaining steps")
    plt.ylabel("Predicted remaining steps")
    plt.title("Remaining Time Prediction")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_risk_scores(y_risk, risk_prob, save_path):
    plt.figure(figsize=(8, 4))

    idx = np.argsort(risk_prob)

    plt.plot(
        risk_prob[idx],
        label="Predicted risk probability",
        linewidth=2,
    )

    plt.plot(
        y_risk[idx],
        label="True risk label",
        alpha=0.55,
    )

    plt.xlabel("Validation samples sorted by predicted risk")
    plt.ylabel("Risk")
    plt.title("Risk Prediction")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_label_diagnostics(num_cases=5, L=64, sim_steps=200, seed=42, save_dir="results"):
    """
    用于检查 detect_critical_point 标注是否合理。
    图像保存到 save_dir / diagnostics。
    """
    from scipy.ndimage import gaussian_filter1d
    from seir_model import PatchSEIR, detect_critical_point

    def safe_corrcoef_local(x, y):
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        if len(x) < 2 or len(y) < 2:
            return 0.0
        if np.std(x) < 1e-12 or np.std(y) < 1e-12:
            return 0.0

        c = np.corrcoef(x, y)[0, 1]
        return 0.0 if not np.isfinite(c) else float(c)

    def compute_csd_series_local(density_history, window=20):
        density_history = np.asarray(density_history, dtype=np.float64)
        n = len(density_history)

        ac1_vals = np.zeros(n)
        var_vals = np.zeros(n)
        trend_vals = np.zeros(n)

        for t in range(window, n):
            w = density_history[t - window:t]

            ac1_vals[t] = safe_corrcoef_local(w[:-1], w[1:])
            var_vals[t] = np.var(w)

            x = np.arange(len(w))
            if np.std(w) > 1e-12:
                trend_vals[t] = np.polyfit(x, w, 1)[0]

        ac1_smooth = gaussian_filter1d(ac1_vals, sigma=2.0)
        var_smooth = gaussian_filter1d(var_vals, sigma=2.0)
        trend_smooth = gaussian_filter1d(trend_vals, sigma=2.0)

        return ac1_smooth, var_smooth, trend_smooth

    save_dir = Path(save_dir) / "diagnostics"
    save_dir.mkdir(parents=True, exist_ok=True)

    for k in range(num_cases):
        print(f"Generating label diagnostic case {k + 1}/{num_cases}...")

        model = PatchSEIR(
            L=L,
            sim_steps=sim_steps,
            seed=seed + 1000 + k,
        )

        density_hist = []

        for _ in range(sim_steps):
            model.step()
            density_hist.append(model.history["density"][-1])

        density_hist = np.asarray(density_hist)
        critical_point = detect_critical_point(density_hist)

        ac1, var, trend = compute_csd_series_local(density_hist)

        plt.figure(figsize=(10, 8))

        plt.subplot(4, 1, 1)
        plt.plot(density_hist)
        if critical_point is not None:
            plt.axvline(critical_point, color="r", linestyle="--")
        plt.ylabel("Density")
        plt.title(f"Diagnostic case {k}, critical point = {critical_point}")

        plt.subplot(4, 1, 2)
        plt.plot(ac1)
        if critical_point is not None:
            plt.axvline(critical_point, color="r", linestyle="--")
        plt.ylabel("AC1")

        plt.subplot(4, 1, 3)
        plt.plot(var)
        if critical_point is not None:
            plt.axvline(critical_point, color="r", linestyle="--")
        plt.ylabel("Variance")

        plt.subplot(4, 1, 4)
        plt.plot(trend)
        if critical_point is not None:
            plt.axvline(critical_point, color="r", linestyle="--")
        plt.ylabel("Trend")
        plt.xlabel("Time")

        plt.tight_layout()

        save_path = save_dir / f"label_diagnostic_{k}.png"
        plt.savefig(save_path, dpi=150)
        plt.close()

        print(f"Saved diagnostic figure: {save_path}")

    return save_dir


def append_experiment_summary(summary_path, args, metrics, result_dir):
    summary_path = Path(summary_path)
    file_exists = summary_path.exists()

    fieldnames = [
        "num_sims",
        "grid",
        "sim_steps",
        "seq_len",
        "horizon",
        "input_mode",
        "feature_group",
        "effective_input_mode",
        "reg_weight",
        "cls_weight",
        "dropout",
        "weight_decay",
        "AUC",
        "AUPRC",
        "F1",
        "Precision",
        "Recall",
        "ACC",
        "Best_Threshold",
        "Positive_Rate",
        "TP",
        "FP",
        "TN",
        "FN",
        "result_dir",
    ]

    row = {
        "num_sims": args.num_sims,
        "grid": args.grid,
        "sim_steps": args.sim_steps,
        "seq_len": args.seq_len,
        "horizon": args.horizon,
        "input_mode": args.input_mode,
        "feature_group": getattr(args, "feature_group", "legacy"),
        "effective_input_mode": getattr(args, "effective_input_mode", args.input_mode),
        "reg_weight": args.reg_weight,
        "cls_weight": args.cls_weight,
        "dropout": args.dropout,
        "weight_decay": args.weight_decay,
        "AUC": metrics.get("AUC", np.nan),
        "AUPRC": metrics.get("AUPRC", np.nan),
        "F1": metrics.get("F1", np.nan),
        "Precision": metrics.get("Precision", np.nan),
        "Recall": metrics.get("Recall", np.nan),
        "ACC": metrics.get("ACC", np.nan),
        "Best_Threshold": metrics.get("Best_Threshold", np.nan),
        "Positive_Rate": metrics.get("Positive_Rate", np.nan),
        "TP": metrics.get("TP", 0),
        "FP": metrics.get("FP", 0),
        "TN": metrics.get("TN", 0),
        "FN": metrics.get("FN", 0),
        "result_dir": str(result_dir),
    }

    with open(summary_path, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


# =========================
# 主程序
# =========================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--num-sims", type=int, default=100)
    parser.add_argument("--grid", type=int, default=64)
    parser.add_argument("--sim-steps", type=int, default=200)
    parser.add_argument("--seq-len", type=int, default=DEFAULT_INPUT_SEQ_LEN)
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)

    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)

    parser.add_argument("--dropout", type=float, default=0.50)
    parser.add_argument("--reg-weight", type=float, default=0.0)
    parser.add_argument("--cls-weight", type=float, default=1.0)

    parser.add_argument("--input-mode", type=str, default="full",
                        choices=["full", "img_only", "patch_only"])
    parser.add_argument(
        "--feature-group",
        type=str,
        default="legacy",
        choices=FEATURE_GROUPS,
        help=(
            "v0.9 feature group. Use legacy to keep v0.7 behavior. "
            "Options: image_only, classic_patch_only, dynamic_patch_only, "
            "classic_dynamic_patch, full."
        ),
    )

    parser.add_argument("--use-pos-weight", action="store_true")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=SEED)

    parser.add_argument("--force-regenerate", action="store_true")
    parser.add_argument("--run-diagnostics", action="store_true")
    parser.add_argument("--diagnostics-only", action="store_true")
    parser.add_argument("--diagnostic-cases", type=int, default=8)

    args = parser.parse_args()

    set_seed(args.seed)

    root_dir = Path(__file__).parent
    data_dir = root_dir / "data"
    model_dir = root_dir / "models"

    if args.feature_group == "legacy":
        run_name = (
            f"seed{args.seed}_"
            f"sims{args.num_sims}_"
            f"L{args.grid}_"
            f"steps{args.sim_steps}_"
            f"h{args.horizon}_"
            f"mode{args.input_mode}_"
            f"rw{args.reg_weight}_"
            f"wd{args.weight_decay}"
        )
    else:
        run_name = (
            f"seed{args.seed}_"
            f"sims{args.num_sims}_"
            f"L{args.grid}_"
            f"steps{args.sim_steps}_"
            f"h{args.horizon}_"
            f"group{args.feature_group}_"
            f"rw{args.reg_weight}_"
            f"wd{args.weight_decay}"
        )

    base_result_dir = root_dir / "results"
    result_dir = base_result_dir / run_name

    data_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    # 保存最近一次运行路径，方便找文件
    latest_path_file = base_result_dir / "latest_run_path.txt"
    latest_path_file.parent.mkdir(parents=True, exist_ok=True)

    with open(latest_path_file, "w", encoding="utf-8") as f:
        f.write(str(result_dir))

    # 线程设置
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["MKL_NUM_THREADS"] = "4"
    os.environ["OPENBLAS_NUM_THREADS"] = "4"
    os.environ["NUMEXPR_NUM_THREADS"] = "4"

    torch.set_num_threads(4)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available.")
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    use_amp = device.type == "cuda"

    if device.type == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GB")
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        print("Using CPU only.")

    print("\n===== Run information =====")
    print(f"run_name: {run_name}")
    print(f"result_dir: {result_dir}")
    print(f"latest_run_path saved to: {latest_path_file}")

    # 只生成诊断图，不训练
    if args.diagnostics_only:
        print("\nDiagnostics-only mode.")
        diag_dir = plot_label_diagnostics(
            num_cases=args.diagnostic_cases,
            L=args.grid,
            sim_steps=args.sim_steps,
            seed=args.seed,
            save_dir=result_dir,
        )
        print("\nDiagnostics finished.")
        print(f"Open this folder to view label diagnostics:")
        print(diag_dir)
        return

    # 训练前生成诊断图
    if args.run_diagnostics:
        print("\nRunning label diagnostics...")
        diag_dir = plot_label_diagnostics(
            num_cases=args.diagnostic_cases,
            L=args.grid,
            sim_steps=args.sim_steps,
            seed=args.seed,
            save_dir=result_dir,
        )
        print(f"Label diagnostics saved to: {diag_dir}")

    cache_name = (
        f"patch_dataset_"
        f"seed{args.seed}_"
        f"sims{args.num_sims}_"
        f"L{args.grid}_"
        f"steps{args.sim_steps}_"
        f"seq{args.seq_len}_"
        f"h{args.horizon}.npz"
    )

    dataset_path = data_dir / cache_name

    if dataset_path.exists() and not args.force_regenerate:
        print(f"\nLoading cached dataset: {dataset_path}")
        data = np.load(dataset_path)

        X_img = data["X_img"]
        X_patch = data["X_patch"]
        y_remaining = data["y_remaining"]
        y_risk = data["y_risk"]
        sim_id = data["sim_id"]

    else:
        X_img, X_patch, y_remaining, y_risk, sim_id = generate_patch_dataset(
            num_sims=args.num_sims,
            L=args.grid,
            sim_steps=args.sim_steps,
            input_seq_len=args.seq_len,
            horizon=args.horizon,
            seed=args.seed,
            num_workers=args.workers,
        )

        np.savez_compressed(
            dataset_path,
            X_img=X_img,
            X_patch=X_patch,
            y_remaining=y_remaining,
            y_risk=y_risk,
            sim_id=sim_id,
        )

        print(f"Dataset saved to: {dataset_path}")

    X_img, X_patch, y_remaining, y_risk, sim_id = clean_dataset(
        X_img,
        X_patch,
        y_remaining,
        y_risk,
        sim_id,
    )

    print("\nAfter cleaning:")
    X_img, X_patch, effective_input_mode = apply_feature_group_to_dataset(
        X_img=X_img,
        X_patch=X_patch,
        feature_group=args.feature_group,
        input_mode=args.input_mode,
    )

    print()
    print("After feature group construction:")
    print(f"Feature group: {args.feature_group}")
    print(f"Effective input_mode: {effective_input_mode}")
    print(f"X_img:   {X_img.shape}")
    print(f"X_patch: {X_patch.shape}")
    args.effective_input_mode = effective_input_mode
    print(f"X_img:       {X_img.shape}")
    print(f"X_patch:     {X_patch.shape}")
    print(f"y_remaining: {y_remaining.shape}")
    print(f"Positive rate: {float(np.mean(y_risk)):.4f}")

    tr_idx, val_idx = split_by_sim_id(
        sim_id,
        train_ratio=TRAIN_RATIO,
        seed=args.seed,
    )

    print("\nSplit by simulation ID:")
    print(f"Train samples: {len(tr_idx)}")
    print(f"Val samples:   {len(val_idx)}")
    print(f"Train positive rate: {float(np.mean(y_risk[tr_idx])):.4f}")
    print(f"Val positive rate:   {float(np.mean(y_risk[val_idx])):.4f}")

    X_patch, patch_mean, patch_std = standardize_patch_features(X_patch, tr_idx)

    scaler_path = model_dir / f"patch_feature_scaler_{run_name}.npz"
    np.savez_compressed(
        scaler_path,
        mean=patch_mean,
        std=patch_std,
    )

    print(f"Patch feature scaler saved to: {scaler_path}")

    train_ds = PatchSequenceDataset(
        X_img[tr_idx],
        X_patch[tr_idx],
        y_remaining[tr_idx],
        y_risk[tr_idx],
        train=True,
    )

    val_ds = PatchSequenceDataset(
        X_img[val_idx],
        X_patch[val_idx],
        y_remaining[val_idx],
        y_risk[val_idx],
        train=False,
    )

    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
    }

    if args.workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(
        train_ds,
        shuffle=True,
        **loader_kwargs,
    )

    val_loader = DataLoader(
        val_ds,
        shuffle=False,
        **loader_kwargs,
    )

    patch_feature_dim = X_patch.shape[-1]

    model = PatchCriticalPredictor(
        in_channels=X_img.shape[2],
        patch_feature_dim=patch_feature_dim,
        cnn_out_dim=64,
        rnn_hidden=64,
        dropout=args.dropout,
        input_mode=effective_input_mode,
    )

    if args.use_pos_weight:
        pos = float(np.sum(y_risk[tr_idx] == 1))
        neg = float(np.sum(y_risk[tr_idx] == 0))

        if pos > 0:
            pos_weight = neg / max(pos, 1.0)
            pos_weight = float(np.clip(pos_weight, 0.5, 2.0))
        else:
            pos_weight = None
    else:
        pos_weight = None

    print(f"\nUse pos_weight: {args.use_pos_weight}")
    print(f"Risk pos_weight: {pos_weight}")
    print(f"reg_weight: {args.reg_weight}")
    print(f"cls_weight: {args.cls_weight}")
    print(f"input_mode: {args.input_mode}")
    print(f"feature_group: {args.feature_group}")
    print(f"effective_input_mode: {effective_input_mode}")

    trainer = Trainer(
        model=model,
        device=device,
        sim_steps=args.sim_steps,
        lr=args.lr,
        weight_decay=args.weight_decay,
        use_amp=use_amp,
        pos_weight=pos_weight,
        reg_weight=args.reg_weight,
        cls_weight=args.cls_weight,
    )

    best_model_path = model_dir / f"best_patch_csd_model_{run_name}.pth"

    print("\nStarting training...")
    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        save_path=best_model_path,
    )

    print("\nEvaluating...")
    y_remain_true, y_remain_pred, y_risk_true, risk_prob = predict(
        trainer.model,
        val_loader,
        device,
        sim_steps=args.sim_steps,
    )

    metrics = compute_metrics(
        y_remain_true,
        y_remain_pred,
        y_risk_true,
        risk_prob,
    )

    print("\n===== Validation Metrics =====")

    for k, v in metrics.items():
        if args.reg_weight == 0 and k in ["RMSE", "MAE", "R2"]:
            continue
        print(f"{k}: {v:.4f}")

    if args.reg_weight == 0:
        print("\nNote: RMSE/MAE/R2 are skipped because reg_weight=0 and remaining_head is not trained.")

    metric_path = result_dir / "validation_metrics.txt"

    with open(metric_path, "w", encoding="utf-8") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v:.6f}\n")

        if args.reg_weight == 0:
            f.write("\nNote: RMSE/MAE/R2 are not interpreted because reg_weight=0.\n")

    print(f"Metrics saved to: {metric_path}")

    np.savez_compressed(
        result_dir / "validation_predictions.npz",
        y_remaining_true=y_remain_true,
        y_remaining_pred=y_remain_pred,
        y_risk_true=y_risk_true,
        risk_prob=risk_prob,
    )

    plot_training_history(
        trainer.history,
        result_dir / "training_history.png",
    )

    plot_risk_scores(
        y_risk_true,
        risk_prob,
        result_dir / "risk_prediction.png",
    )

    # 如果 reg_weight=0，remaining_head 没有训练，不保存 remaining 图，避免误读
    if args.reg_weight > 0:
        plot_predictions(
            y_remain_true,
            y_remain_pred,
            result_dir / "remaining_prediction.png",
        )
    else:
        print("Skipping remaining_prediction.png because reg_weight=0.")

    summary_path = base_result_dir / "experiment_summary.csv"
    append_experiment_summary(
        summary_path=summary_path,
        args=args,
        metrics=metrics,
        result_dir=result_dir,
    )

    print("\nSaved figures:")
    print(result_dir / "training_history.png")
    print(result_dir / "risk_prediction.png")

    if args.reg_weight > 0:
        print(result_dir / "remaining_prediction.png")

    print(f"\nExperiment summary updated:")
    print(summary_path)

    print("\nOpen this folder to view this run:")
    print(result_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()