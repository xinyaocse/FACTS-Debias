#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
- Train representation decomposition model
- Extract z_specific and z_shared
- Save model checkpoints and vector outputs
- Evaluate:
  (1) z_specific -> label accuracy
  (2) z_shared -> label accuracy using adversarial head
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
sys.path.append("/root/bias")

import torch
import numpy as np
import random
from torch.utils.data import DataLoader

from model import Encoder, Classifier, AdversarialClassifier, Decoder, train_model
from dataset import BiasFactDataset
from utils import load_data, get_gtr_token_embeddings

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score


# ==============================
# Helper functions
# ==============================

@torch.no_grad()
def extract_z_specific(encoder, dataloader, device):
    encoder.eval()
    all_z_specific = []

    for x, _ in dataloader:
        x = x.to(device)
        _, z_spec = encoder(x)
        all_z_specific.append(z_spec.cpu())

    return torch.cat(all_z_specific, dim=0)


@torch.no_grad()
def extract_z_shared(encoder, dataloader, device):
    encoder.eval()
    all_z_shared = []

    for x, _ in dataloader:
        x = x.to(device)
        z_shared, _ = encoder(x)
        all_z_shared.append(z_shared.cpu())

    return torch.cat(all_z_shared, dim=0)


@torch.no_grad()
def evaluate_model(encoder, classifier, dataloader, device, name=""):
    encoder.eval()
    classifier.eval()

    all_preds = []
    all_labels = []

    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)

        _, z_specific = encoder(x)
        logits = classifier(z_specific)
        preds = torch.argmax(logits, dim=1)

        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(y.cpu().tolist())

    acc = accuracy_score(all_labels, all_preds)
    print(f"z_specific 分类准确率 ({name}): {acc * 100:.2f}%")

    return acc


@torch.no_grad()
def evaluate_z_shared_adversarial(encoder, adv_classifier, dataloader, device, name=""):
    encoder.eval()
    adv_classifier.eval()

    all_preds = []
    all_labels = []

    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)

        z_shared, _ = encoder(x)
        logits = adv_classifier(z_shared)
        preds = torch.argmax(logits, dim=1)

        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(y.cpu().tolist())

    acc = accuracy_score(all_labels, all_preds)
    print(f"z_shared 分类准确率 (adv head, {name}): {acc * 100:.2f}%")

    return acc


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_aligned_class_weights(train_labels, num_classes, device):
    """
    Return length=num_classes weights aligned to class index 0..num_classes-1.
    If some class is missing, use weight=1.0 for that class.
    """
    labels = np.asarray(train_labels, dtype=np.int64)
    present = np.unique(labels)

    weights = np.ones((num_classes,), dtype=np.float32)

    present_sorted = np.array(sorted(present.tolist()), dtype=np.int64)
    w_present = compute_class_weight(
        class_weight="balanced",
        classes=present_sorted,
        y=labels
    )

    for c, w in zip(present_sorted, w_present):
        if 0 <= int(c) < num_classes:
            weights[int(c)] = float(w)

    return torch.tensor(weights, dtype=torch.float, device=device)


def save_per_class_z_and_dirs(z_specific, labels, out_dir, num_classes=3, eps=1e-8):
    """
    Save:
      - z_specific_train_class{c}.pt
      - dir_specific_class{c}.pt
    """
    os.makedirs(out_dir, exist_ok=True)

    labels = np.asarray(labels, dtype=np.int64)

    for c in range(num_classes):
        mask = labels == c
        z_c = z_specific[mask]

        z_path = os.path.join(out_dir, f"z_specific_train_class{c}.pt")
        torch.save(z_c, z_path)
        print(f"Saved {z_path}, shape={tuple(z_c.shape)}")

        if z_c.numel() == 0:
            print(f"class {c} missing in train set, skip dir_specific_class{c}.pt")
            continue

        mu = z_c.mean(dim=0)
        mu = mu / (mu.norm(p=2) + eps)

        dir_path = os.path.join(out_dir, f"dir_specific_class{c}.pt")
        torch.save(mu, dir_path)
        print(f"Saved {dir_path}, shape={tuple(mu.shape)}")


# ==============================
# Main
# ==============================

def main():
    set_seed()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t5_model_path = "/root/public/lcx/GenderCARE-ccs24-main/Code/flan-t5-small"

    num_classes = 3

    # ==============================
    # Train data
    # ==============================

    train_texts, train_labels = load_data("train_dataset_sampled.csv")
    train_embeddings = get_gtr_token_embeddings(
        train_texts,
        t5_model_path,
        device=device
    )

    train_dataset = BiasFactDataset(train_embeddings, train_labels)

    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        drop_last=True
    )

    train_loader_for_extract = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=False,
        drop_last=False
    )

    class_weights = compute_aligned_class_weights(
        train_labels,
        num_classes=num_classes,
        device=device
    )

    print("class_weights =", class_weights.detach().cpu().numpy().tolist())

    # ==============================
    # Model
    # ==============================

    encoder = Encoder(token_dim=512, latent_dim=512).to(device)
    classifier = Classifier(latent_dim=512, num_classes=num_classes).to(device)
    adv_classifier = AdversarialClassifier(latent_dim=512, num_classes=num_classes).to(device)
    decoder = Decoder(latent_dim=512, token_dim=512).to(device)

    opt_main = torch.optim.Adam(
        list(encoder.parameters()) +
        list(classifier.parameters()) +
        list(decoder.parameters()),
        lr=1e-4
    )

    opt_adv = torch.optim.Adam(
        list(adv_classifier.parameters()),
        lr=3e-4
    )

    # ==============================
    # Train
    # ==============================

    train_model(
        encoder=encoder,
        classifier=classifier,
        adv_classifier=adv_classifier,
        decoder=decoder,
        dataloader=train_loader,
        opt_main=opt_main,
        opt_adv=opt_adv,
        device=device,
        class_weights=class_weights,
        num_classes=num_classes,
        adv_alpha=1.0,
        recon_alpha=0.2,
        proto_intra_alpha=0.05,
        proto_inter_alpha=0.10,
        proto_warmup_epochs=3,
        epochs=12
    )

    evaluate_model(
        encoder,
        classifier,
        train_loader_for_extract,
        device,
        name="Train Set"
    )

    # ==============================
    # Save checkpoints
    # ==============================

    os.makedirs("checkpoints", exist_ok=True)

    torch.save(encoder.state_dict(), "checkpoints/encoder.pt")
    torch.save(classifier.state_dict(), "checkpoints/classifier.pt")
    torch.save(adv_classifier.state_dict(), "checkpoints/adv_classifier.pt")

    print("Model weights saved to checkpoints/")

    # ==============================
    # Test data
    # ==============================

    test_texts, test_labels = load_data("test_dataset_sampled.csv")
    test_embeddings = get_gtr_token_embeddings(
        test_texts,
        t5_model_path,
        device=device
    )

    test_dataset = BiasFactDataset(test_embeddings, test_labels)

    test_loader = DataLoader(
        test_dataset,
        batch_size=4,
        shuffle=False,
        drop_last=False
    )

    evaluate_model(
        encoder,
        classifier,
        test_loader,
        device,
        name="Test Set"
    )

    # ==============================
    # Extract and save test vectors
    # ==============================

    os.makedirs("outputs/vectors", exist_ok=True)

    z_specific_test = extract_z_specific(
        encoder,
        test_loader,
        device
    )

    z_shared_test = extract_z_shared(
        encoder,
        test_loader,
        device
    )

    torch.save(z_specific_test, "outputs/vectors/z_specific_test.pt")
    torch.save(z_shared_test, "outputs/vectors/z_shared_test.pt")

    print("Saved z_specific_test.pt and z_shared_test.pt")

    # ==============================
    # Extract and save train vectors
    # ==============================

    z_specific_train = extract_z_specific(
        encoder,
        train_loader_for_extract,
        device
    )

    z_shared_train = extract_z_shared(
        encoder,
        train_loader_for_extract,
        device
    )

    torch.save(z_specific_train, "outputs/vectors/z_specific_train.pt")
    torch.save(z_shared_train, "outputs/vectors/z_shared_train.pt")

    print("Saved z_specific_train.pt and z_shared_train.pt")

    # ==============================
    # Save per-class z_specific and directions
    # ==============================

    save_per_class_z_and_dirs(
        z_specific=z_specific_train,
        labels=train_labels,
        out_dir="outputs/vectors",
        num_classes=num_classes
    )

    # ==============================
    # Adversarial head evaluation
    # ==============================

    print("=" * 60)
    print("adv head evaluation: z_shared to label, lower is better")

    evaluate_z_shared_adversarial(
        encoder,
        adv_classifier,
        train_loader_for_extract,
        device,
        name="Train Set"
    )

    evaluate_z_shared_adversarial(
        encoder,
        adv_classifier,
        test_loader,
        device,
        name="Test Set"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()