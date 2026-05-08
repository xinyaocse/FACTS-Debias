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
            f"Cannot find text or label columns in the CSV file. "
            f"Available columns: {fieldnames}\n"
            f"Candidate text columns: {text_keys}\n"
            f"Candidate label columns: {label_keys}"
        )

    return tcol, ycol


def load_texts_and_labels(csv_path: str) -> Tuple[List[str], List[str]]:
    texts, labels = [], []

    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)

        if not reader.fieldnames:
            raise ValueError(f"{csv_path} has no header or is empty.")

        tcol, ycol = detect_columns(reader.fieldnames)

        for row in reader:
            text = (row.get(tcol, "") or "").strip()
            label = (row.get(ycol, "") or "").strip()

            if text == "":
                continue

            texts.append(text)
            labels.append(label)

    if len(texts) == 0:
        raise ValueError(f"{csv_path} contains no valid data rows.")

    return texts, labels


def build_label_mapping(
    raw_labels: List[str],
    num_labels: Optional[int] = None,
    enforce_numeric: bool = True,
) -> Tuple[Dict[str, int], List[int]]:

    raw_labels = [(y or "").strip() for y in raw_labels]

    if num_labels is not None:
        if enforce_numeric:
            allowed = {str(i) for i in range(num_labels)}
            bad = sorted(list(set([y for y in raw_labels if y not in allowed])))

            if bad:
                raise ValueError(
                    f"Found labels outside the range 0..{num_labels - 1}: {bad[:20]} "
                    f"({len(bad)} unique invalid labels in total). "
                    f"Please check the label column in the CSV file."
                )

            mapping = {str(i): i for i in range(num_labels)}
            mapped = [mapping[y] for y in raw_labels]

            return mapping, mapped

        unique_labels = sorted(list(set(raw_labels)))

        if len(unique_labels) != num_labels:
            raise ValueError(
                f"Expected {num_labels} classes, but found {len(unique_labels)}: {unique_labels}"
            )

        mapping = {label: i for i, label in enumerate(unique_labels)}
        mapped = [mapping[y] for y in raw_labels]

        return mapping, mapped

    all_numeric = True

    for y in raw_labels:
        try:
            _ = int(y)
        except Exception:
            all_numeric = False
            break

    if all_numeric:
        unique_labels = sorted(list(set(int(y) for y in raw_labels)))
        mapping = {str(v): i for i, v in enumerate(unique_labels)}
        mapped = [mapping[str(int(y))] for y in raw_labels]

        return mapping, mapped

    unique_labels = sorted(list(set(raw_labels)))
    mapping = {label: i for i, label in enumerate(unique_labels)}
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

    tokenizer = AutoTokenizer.from_pretrained(
        t5_path,
        local_files_only=True
    )

    try:
        encoder = T5EncoderModel.from_pretrained(
            t5_path,
            local_files_only=True
        ).to(device).eval()

        def encode(batch_tok):
            output = encoder(
                input_ids=batch_tok["input_ids"],
                attention_mask=batch_tok["attention_mask"]
            )
            return output.last_hidden_state

    except Exception:
        base_model = AutoModel.from_pretrained(
            t5_path,
            local_files_only=True
        ).to(device).eval()

        encoder_only = base_model.get_encoder() if hasattr(base_model, "get_encoder") else base_model

        def encode(batch_tok):
            output = encoder_only(
                input_ids=batch_tok["input_ids"],
                attention_mask=batch_tok["attention_mask"]
            )

            if hasattr(output, "last_hidden_state"):
                return output.last_hidden_state

            if isinstance(output, (list, tuple)) and len(output) > 0:
                return output[0]

            raise RuntimeError("Encoder output does not contain last_hidden_state.")

    outputs = []

    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i + batch_size]

        tokens = tokenizer(
            chunk,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        ).to(device)

        last_hidden_state = encode(tokens)
        attention_mask = tokens["attention_mask"].unsqueeze(-1)

        mean_pooled = (
            (last_hidden_state * attention_mask).sum(dim=1)
            / attention_mask.sum(dim=1).clamp(min=1)
        )

        outputs.append(torch.nan_to_num(mean_pooled))

    if outputs:
        return torch.cat(outputs, dim=0)

    return torch.empty(0, 512, device=device)


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


def make_loader(
    X: torch.Tensor,
    y: torch.Tensor,
    batch_size: int,
    shuffle: bool
):
    dataset = torch.utils.data.TensorDataset(X, y)

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False
    )


def train_one_epoch(model, loader, optimizer, device, class_weights=None):
    model.train()

    total = 0
    correct = 0
    loss_sum = 0.0

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        logits = model(xb)

        if class_weights is not None:
            loss = F.cross_entropy(logits, yb, weight=class_weights)
        else:
            loss = F.cross_entropy(logits, yb)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        loss_sum += float(loss.detach().cpu())

        pred = logits.argmax(dim=-1)
        total += yb.numel()
        correct += (pred == yb).sum().item()

    acc = correct / max(1, total)
    avg_loss = loss_sum / max(1, len(loader))

    return avg_loss, acc


@torch.no_grad()
def eval_model(model, loader, device):
    model.eval()

    total = 0
    correct = 0
    loss_sum = 0.0

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        logits = model(xb)
        loss = F.cross_entropy(logits, yb)

        loss_sum += float(loss.detach().cpu())

        pred = logits.argmax(dim=-1)
        total += yb.numel()
        correct += (pred == yb).sum().item()

    acc = correct / max(1, total)
    avg_loss = loss_sum / max(1, len(loader))

    return avg_loss, acc


def build_argparser():
    parser = argparse.ArgumentParser(
        description="Train a lightweight classifier on T5 encoder embeddings."
    )

    parser.add_argument(
        "--train_csv",
        type=str,
        default="train_dataset_sampled.csv"
    )

    parser.add_argument(
        "--save_path",
        type=str,
        default="checkpoints/text_space_clf.pt"
    )

    parser.add_argument(
        "--t5_path",
        type=str,
        default=os.environ.get("T5_MODEL_PATH", "models/flan-t5-small"),
        help="Local path to the T5 model."
    )

    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--use_class_weights", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--num_labels",
        type=int,
        default=3,
        help="Number of labels. By default, labels are expected to be 0, 1, and 2."
    )

    return parser


def main():
    args = build_argparser().parse_args()

    set_seed(args.seed)

    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device={device}, t5_path={args.t5_path}")


    texts, raw_labels = load_texts_and_labels(args.train_csv)
    print(f"Loaded {len(texts)} rows from {args.train_csv}")


    label_map, y_list = build_label_mapping(
        raw_labels,
        num_labels=args.num_labels,
        enforce_numeric=True
    )

    num_labels = args.num_labels
    y = torch.tensor(y_list, dtype=torch.long)


    base, _ = os.path.splitext(args.save_path)

    with open(base + "_label_map.json", "w", encoding="utf-8") as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)

    label_counts = np.bincount(
        np.asarray(y_list, dtype=np.int64),
        minlength=num_labels
    )

    print(
        "Label counts:",
        {i: int(label_counts[i]) for i in range(num_labels)}
    )


    X = t5_embed_texts(
        texts,
        t5_path=args.t5_path,
        device=device,
        batch_size=128,
        max_length=128
    )

    assert X.shape[0] == y.shape[0] and X.shape[1] == 512, (
        f"Embedding shape mismatch: {X.shape}"
    )


    N = X.shape[0]
    indices = np.arange(N)
    np.random.shuffle(indices)

    val_n = int(N * args.val_ratio)
    val_idx = indices[:val_n]
    train_idx = indices[val_n:]

    X_train = X[train_idx]
    y_train = y[train_idx]

    X_val = X[val_idx]
    y_val = y[val_idx]


    class_weights = None

    if args.use_class_weights:
        y_np = y_train.detach().cpu().numpy()
        present = np.unique(y_np)

        weights = np.ones((num_labels,), dtype=np.float32)

        w_present = compute_class_weight(
            class_weight="balanced",
            classes=present,
            y=y_np
        )

        for c, w in zip(present, w_present):
            weights[int(c)] = float(w)

        class_weights = torch.tensor(
            weights,
            dtype=torch.float,
            device=device
        )

        print("Class weights:", class_weights.detach().cpu().numpy().tolist())


    model = TextSpaceClassifier(
        in_dim=512,
        hidden=256,
        num_labels=num_labels,
        p_drop=0.1
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr
    )


    train_loader = make_loader(
        X_train,
        y_train,
        batch_size=args.batch_size,
        shuffle=True
    )

    val_loader = make_loader(
        X_val,
        y_val,
        batch_size=args.batch_size,
        shuffle=False
    )


    best_val_acc = 0.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            class_weights=class_weights
        )

        val_loss, val_acc = eval_model(
            model,
            val_loader,
            device
        )

        print(
            f"[Epoch {epoch:02d}] "
            f"train loss={train_loss:.4f}, acc={train_acc:.4f} | "
            f"val loss={val_loss:.4f}, acc={val_acc:.4f}"
        )

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_state = {
                k: v.detach().cpu()
                for k, v in model.state_dict().items()
            }


    payload = {
        "state_dict": (
            best_state
            if best_state is not None
            else {k: v.detach().cpu() for k, v in model.state_dict().items()}
        ),
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

    print(
        f"Saved classifier to {args.save_path} "
        f"(best val acc={best_val_acc:.4f})"
    )


if __name__ == "__main__":
    os.environ["HF_HUB_OFFLINE"] = os.environ.get("HF_HUB_OFFLINE", "1")
    os.environ["TRANSFORMERS_OFFLINE"] = os.environ.get("TRANSFORMERS_OFFLINE", "1")

    main()
