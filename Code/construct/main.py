
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import argparse
import random

import torch
import numpy as np
from torch.utils.data import DataLoader

from model import Encoder, Classifier, AdversarialClassifier, Decoder, train_model
from dataset import BiasFactDataset
from utils import load_data, get_gtr_token_embeddings

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score


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
    print(f"z_specific classification accuracy ({name}): {acc * 100:.2f}%")

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
    print(f"z_shared classification accuracy (adv head, {name}): {acc * 100:.2f}%")

    return acc


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_aligned_class_weights(train_labels, num_classes, device):
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
    os.makedirs(out_dir, exist_ok=True)

    labels = np.asarray(labels, dtype=np.int64)

    for c in range(num_classes):
        mask = labels == c
        z_c = z_specific[mask]

        z_path = os.path.join(out_dir, f"z_specific_train_class{c}.pt")
        torch.save(z_c, z_path)
        print(f"Saved {z_path}, shape={tuple(z_c.shape)}")

        if z_c.numel() == 0:
            print(f"Class {c} is missing in the training set. Skip dir_specific_class{c}.pt")
            continue

        mu = z_c.mean(dim=0)
        mu = mu / (mu.norm(p=2) + eps)

        dir_path = os.path.join(out_dir, f"dir_specific_class{c}.pt")
        torch.save(mu, dir_path)
        print(f"Saved {dir_path}, shape={tuple(mu.shape)}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a representation decomposition model."
    )

    parser.add_argument(
        "--project_root",
        type=str,
        default=os.environ.get("PROJECT_ROOT", ""),
        help="Optional project root added to sys.path. Leave empty if modules are in the current directory."
    )

    parser.add_argument(
        "--embedding_model_path",
        type=str,
        default=os.environ.get("EMBEDDING_MODEL_PATH", "models/flan-t5-small"),
        help="Path to the local embedding model."
    )

    parser.add_argument(
        "--train_file",
        type=str,
        default="train_dataset_sampled.csv",
        help="Path to the training CSV file."
    )

    parser.add_argument(
        "--test_file",
        type=str,
        default="test_dataset_sampled.csv",
        help="Path to the test CSV file."
    )

    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints",
        help="Directory for saving model checkpoints."
    )

    parser.add_argument(
        "--vector_dir",
        type=str,
        default=os.path.join("outputs", "vectors"),
        help="Directory for saving extracted vectors."
    )

    parser.add_argument("--num_classes", type=int, default=3)
    parser.add_argument("--token_dim", type=int, default=512)
    parser.add_argument("--latent_dim", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--lr_main", type=float, default=1e-4)
    parser.add_argument("--lr_adv", type=float, default=3e-4)

    parser.add_argument("--adv_alpha", type=float, default=1.0)
    parser.add_argument("--recon_alpha", type=float, default=0.2)
    parser.add_argument("--proto_intra_alpha", type=float, default=0.05)
    parser.add_argument("--proto_inter_alpha", type=float, default=0.10)
    parser.add_argument("--proto_warmup_epochs", type=int, default=3)

    return parser.parse_args()



def main():
    args = parse_args()

    if args.project_root:
        sys.path.append(args.project_root)

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")


    train_texts, train_labels = load_data(args.train_file)

    train_embeddings = get_gtr_token_embeddings(
        train_texts,
        args.embedding_model_path,
        device=device
    )

    train_dataset = BiasFactDataset(train_embeddings, train_labels)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True
    )

    train_loader_for_extract = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False
    )

    class_weights = compute_aligned_class_weights(
        train_labels,
        num_classes=args.num_classes,
        device=device
    )

    print("class_weights =", class_weights.detach().cpu().numpy().tolist())


    encoder = Encoder(
        token_dim=args.token_dim,
        latent_dim=args.latent_dim
    ).to(device)

    classifier = Classifier(
        latent_dim=args.latent_dim,
        num_classes=args.num_classes
    ).to(device)

    adv_classifier = AdversarialClassifier(
        latent_dim=args.latent_dim,
        num_classes=args.num_classes
    ).to(device)

    decoder = Decoder(
        latent_dim=args.latent_dim,
        token_dim=args.token_dim
    ).to(device)

    opt_main = torch.optim.Adam(
        list(encoder.parameters()) +
        list(classifier.parameters()) +
        list(decoder.parameters()),
        lr=args.lr_main
    )

    opt_adv = torch.optim.Adam(
        list(adv_classifier.parameters()),
        lr=args.lr_adv
    )


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
        num_classes=args.num_classes,
        adv_alpha=args.adv_alpha,
        recon_alpha=args.recon_alpha,
        proto_intra_alpha=args.proto_intra_alpha,
        proto_inter_alpha=args.proto_inter_alpha,
        proto_warmup_epochs=args.proto_warmup_epochs,
        epochs=args.epochs
    )

    evaluate_model(
        encoder,
        classifier,
        train_loader_for_extract,
        device,
        name="Train Set"
    )


    os.makedirs(args.checkpoint_dir, exist_ok=True)

    torch.save(encoder.state_dict(), os.path.join(args.checkpoint_dir, "encoder.pt"))
    torch.save(classifier.state_dict(), os.path.join(args.checkpoint_dir, "classifier.pt"))
    torch.save(adv_classifier.state_dict(), os.path.join(args.checkpoint_dir, "adv_classifier.pt"))

    print(f"Model weights saved to {args.checkpoint_dir}/")


    test_texts, test_labels = load_data(args.test_file)

    test_embeddings = get_gtr_token_embeddings(
        test_texts,
        args.embedding_model_path,
        device=device
    )

    test_dataset = BiasFactDataset(test_embeddings, test_labels)

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
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


    os.makedirs(args.vector_dir, exist_ok=True)

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

    torch.save(z_specific_test, os.path.join(args.vector_dir, "z_specific_test.pt"))
    torch.save(z_shared_test, os.path.join(args.vector_dir, "z_shared_test.pt"))

    print("Saved z_specific_test.pt and z_shared_test.pt")


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

    torch.save(z_specific_train, os.path.join(args.vector_dir, "z_specific_train.pt"))
    torch.save(z_shared_train, os.path.join(args.vector_dir, "z_shared_train.pt"))

    print("Saved z_specific_train.pt and z_shared_train.pt")


    save_per_class_z_and_dirs(
        z_specific=z_specific_train,
        labels=train_labels,
        out_dir=args.vector_dir,
        num_classes=args.num_classes
    )


    print("=" * 60)
    print("Adversarial head evaluation: z_shared -> label. Lower is better.")

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