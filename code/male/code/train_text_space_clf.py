#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train a lightweight TextSpaceClassifier on flan-t5-small sentence embeddings.

- 用 T5EncoderModel(mean-pool) 得到 512 维句向量
- 默认模型路径：/root/public/lcx/GenderCARE-ccs24-main/Code/flan-t5-small
- ✅ 支持 3-class 标签 (0/1/2)
- 可选：--num_labels 3 强制标签空间为 {0,1,2}
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import csv
import argparse
import random
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.utils.class_weight import compute_class_weight

from transformers import AutoTokenizer, T5EncoderModel, AutoModel


# ----------------- Utils -----------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def detect_columns(fieldnames: List[str]) -> Tuple[str, str]:
    text_keys = ["text", "sentence", "content", "texts", "input", "prompt"]
    label_keys = ["label", "y", "target", "class", "labels"]
    lower2orig = {c.lower(): c for c in fieldnames}

    def first(keys):
        for k in keys:
            if k in lower2orig:
                return lower2orig[k]
        return None

    tcol = first(text_keys)
    ycol = first(label_keys)
    if tcol is None or ycol is None:
        raise ValueError(
            f"CSV 未找到文本/标签列。字段有：{fieldnames}\n"
            f"文本列候选：{text_keys}\n标签列候选：{label_keys}"
        )
    return tcol, ycol


def load_texts_and_labels(csv_path: str) -> Tuple[List[str], List[str]]:
    texts, labels = [], []
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        r = csv.DictReader(f)
        if not r.fieldnames:
            raise ValueError(f"{csv_path} 无表头或为空")
        tcol, ycol = detect_columns(r.fieldnames)
        for row in r:
            t = (row.get(tcol, "") or "").strip()
            y = (row.get(ycol, "") or "").strip()
            if t == "":
                continue
            texts.append(t)
            labels.append(y)
    if len(texts) == 0:
        raise ValueError(f"{csv_path} 没有有效的数据行")
    return texts, labels


def build_label_mapping(
    raw_labels: List[str],
    num_labels: Optional[int] = None,
    enforce_numeric: bool = True,
) -> Tuple[Dict[str, int], List[int]]:
    """
    - 如果 num_labels=3：强制标签空间为 {'0','1','2'}，并检查非法值
    - 否则：沿用你原来的逻辑（若全数字则按 uniq 排序映射到 0..K-1；否则按字符串 uniq）
    """
    raw_labels = [(y or "").strip() for y in raw_labels]

    # ✅ 强制 3 类 (0/1/2)
    if num_labels is not None:
        if enforce_numeric:
            allowed = {str(i) for i in range(num_labels)}
            bad = sorted(list(set([y for y in raw_labels if y not in allowed])))
            if bad:
                raise ValueError(
                    f"发现不在 0..{num_labels-1} 范围内的标签：{bad[:20]} "
                    f"(共 {len(bad)} 个不同非法标签). 请检查 CSV 的 label 列。"
                )
            mapping = {str(i): i for i in range(num_labels)}
            mapped = [mapping[y] for y in raw_labels]
            return mapping, mapped

        # 不强制 numeric 的情况（一般你用不到）
        uniq = sorted(list(set(raw_labels)))
        if len(uniq) != num_labels:
            raise ValueError(f"期望 {num_labels} 类，但数据中发现 {len(uniq)} 类: {uniq}")
        mapping = {u: i for i, u in enumerate(uniq)}
        mapped = [mapping[y] for y in raw_labels]
        return mapping, mapped

    # ===== 否则：保持你原来的“自动映射” =====
    all_numeric = True
    for y in raw_labels:
        try:
            _ = int(y)
        except Exception:
            all_numeric = False
            break

    if all_numeric:
        uniq = sorted(list(set(int(y) for y in raw_labels)))
        mapping = {str(v): i for i, v in enumerate(uniq)}
        mapped = [mapping[str(int(y))] for y in raw_labels]
        return mapping, mapped

    uniq = sorted(list(set(raw_labels)))
    mapping = {u: i for i, u in enumerate(uniq)}
    mapped = [mapping[y] for y in raw_labels]
    return mapping, mapped


@torch.no_grad()
def t5_embed_texts(
    texts: List[str],
    t5_path: str,
    device: torch.device,
    batch_size: int = 128,
    max_length: int = 128
) -> torch.Tensor:
    """
    用 flan-t5-small 做 encoder-only embedding，输出 [B, 512]
    """
    tok = AutoTokenizer.from_pretrained(t5_path, local_files_only=True)

    try:
        enc = T5EncoderModel.from_pretrained(t5_path, local_files_only=True).to(device).eval()

        def encode(batch_tok):
            out = enc(input_ids=batch_tok["input_ids"], attention_mask=batch_tok["attention_mask"])
            return out.last_hidden_state  # [B, L, 512]

    except Exception:
        base = AutoModel.from_pretrained(t5_path, local_files_only=True).to(device).eval()
        encoder_only = base.get_encoder() if hasattr(base, "get_encoder") else base

        def encode(batch_tok):
            out = encoder_only(input_ids=batch_tok["input_ids"], attention_mask=batch_tok["attention_mask"])
            if hasattr(out, "last_hidden_state"):
                return out.last_hidden_state
            elif isinstance(out, (list, tuple)) and len(out) > 0:
                return out[0]
            else:
                raise RuntimeError("Encoder output missing last_hidden_state")

    outs = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i + batch_size]
        toks = tok(chunk, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(device)
        last = encode(toks)  # [B, L, 512]
        mask = toks["attention_mask"].unsqueeze(-1)  # [B, L, 1]
        mean = (last * mask).sum(1) / mask.sum(1).clamp(min=1)  # [B, 512]
        outs.append(torch.nan_to_num(mean))
    return torch.cat(outs, dim=0) if outs else torch.empty(0, 512, device=device)


class TextSpaceClassifier(nn.Module):
    def __init__(self, in_dim=512, hidden=256, num_labels=3, p_drop=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(p_drop),
            nn.Linear(hidden, num_labels),
        )

    def forward(self, x):
        return self.net(torch.nan_to_num(x))


def make_loader(X: torch.Tensor, y: torch.Tensor, batch_size: int, shuffle: bool):
    ds = torch.utils.data.TensorDataset(X, y)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def train_one_epoch(model, loader, opt, device, class_weights=None):
    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = F.cross_entropy(logits, yb, weight=class_weights) if class_weights is not None else F.cross_entropy(logits, yb)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        loss_sum += float(loss.detach().cpu())
        pred = logits.argmax(-1)
        total += yb.numel()
        correct += (pred == yb).sum().item()

    acc = correct / max(1, total)
    return loss_sum / max(1, len(loader)), acc


@torch.no_grad()
def eval_model(model, loader, device):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = F.cross_entropy(logits, yb)
        loss_sum += float(loss.detach().cpu())
        pred = logits.argmax(-1)
        total += yb.numel()
        correct += (pred == yb).sum().item()
    acc = correct / max(1, total)
    return loss_sum / max(1, len(loader)), acc


def build_argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", type=str, default="train_dataset_sampled.csv")
    ap.add_argument("--save_path", type=str, default="checkpoints/text_space_clf.pt")

    ap.add_argument("--t5_path", type=str, default="/root/public/lcx/GenderCARE-ccs24-main/Code/flan-t5-small")

    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--use_class_weights", action="store_true")
    ap.add_argument("--seed", type=int, default=42)

    # ✅ 新增：强制 3 类标签(0/1/2)。建议你训练三类时总是显式传这个。
    ap.add_argument("--num_labels", type=int, default=3)

    return ap


def main():
    args = build_argparser().parse_args()
    set_seed(args.seed)

    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device={device}, t5_path={args.t5_path}")

    # 1) load data
    texts, raw_labels = load_texts_and_labels(args.train_csv)
    print(f"Loaded {len(texts)} rows from {args.train_csv}")

    # 2) label mapping (✅ enforce 0/1/2 by default)
    label_map, y_list = build_label_mapping(raw_labels, num_labels=args.num_labels, enforce_numeric=True)
    num_labels = args.num_labels
    y = torch.tensor(y_list, dtype=torch.long)

    # save label map
    base, _ = os.path.splitext(args.save_path)
    with open(base + "_label_map.json", "w", encoding="utf-8") as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)

    # quick stats
    binc = np.bincount(np.asarray(y_list, dtype=np.int64), minlength=num_labels)
    print("[i] label counts:", {i: int(binc[i]) for i in range(num_labels)})

    # 3) embeddings
    X = t5_embed_texts(texts, t5_path=args.t5_path, device=device, batch_size=128, max_length=128)
    assert X.shape[0] == y.shape[0] and X.shape[1] == 512, f"Emb shape mismatch: {X.shape}"

    # 4) split train/val
    N = X.shape[0]
    idx = np.arange(N)
    np.random.shuffle(idx)
    val_n = int(N * args.val_ratio)
    val_idx = idx[:val_n]
    tr_idx = idx[val_n:]

    X_tr, y_tr = X[tr_idx], y[tr_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    # 5) class weights (optional, ✅ aligned to 0..num_labels-1, and safe for missing class)
    class_weights = None
    if args.use_class_weights:
        y_np = y_tr.detach().cpu().numpy()
        present = np.unique(y_np)

        weights = np.ones((num_labels,), dtype=np.float32)
        # only compute for present classes (avoid sklearn error if a class is missing)
        w_present = compute_class_weight(class_weight="balanced", classes=present, y=y_np)
        for c, w in zip(present, w_present):
            weights[int(c)] = float(w)

        class_weights = torch.tensor(weights, dtype=torch.float, device=device)
        print("[i] class weights:", class_weights.detach().cpu().numpy().tolist())

    # 6) model/opt
    model = TextSpaceClassifier(in_dim=512, hidden=256, num_labels=num_labels, p_drop=0.1).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # 7) loaders
    train_loader = make_loader(X_tr, y_tr, batch_size=args.batch_size, shuffle=True)
    val_loader = make_loader(X_val, y_val, batch_size=args.batch_size, shuffle=False)

    # 8) train
    best_val_acc = 0.0
    best_state = None
    for ep in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, opt, device, class_weights=class_weights)
        val_loss, val_acc = eval_model(model, val_loader, device)
        print(f"[ep{ep:02d}] train loss={tr_loss:.4f} acc={tr_acc:.4f} | val loss={val_loss:.4f} acc={val_acc:.4f}")
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    # 9) save
    payload = {
        "state_dict": best_state if best_state is not None else {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "num_labels": num_labels,
        "label_map": label_map,
        "t5_path": args.t5_path,
        "config": {
            "in_dim": 512,
            "hidden": 256,
            "p_drop": 0.1,
        },
    }
    torch.save(payload, args.save_path)
    print(f"Saved classifier to {args.save_path} (best val acc={best_val_acc:.4f})")


if __name__ == "__main__":
    os.environ["HF_HUB_OFFLINE"] = os.environ.get("HF_HUB_OFFLINE", "1")
    os.environ["TRANSFORMERS_OFFLINE"] = os.environ.get("TRANSFORMERS_OFFLINE", "1")
    main()
