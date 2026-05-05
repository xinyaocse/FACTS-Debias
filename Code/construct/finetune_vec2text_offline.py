# -*- coding: utf-8 -*-

import os
os.environ["HF_HUB_OFFLINE"] = os.environ.get("HF_HUB_OFFLINE", "1")
os.environ["TRANSFORMERS_OFFLINE"] = os.environ.get("TRANSFORMERS_OFFLINE", "1")

import json
import argparse
import random
import numpy as np
import pandas as pd
from collections import deque
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from transformers import (
    AutoTokenizer,
    AutoModel,
    T5EncoderModel,
    T5ForConditionalGeneration,
)
from transformers.modeling_outputs import BaseModelOutput


# ==============================
# Local-only loading patch
# ==============================

try:
    from transformers.modeling_utils import PreTrainedModel

    _orig_from_pretrained = PreTrainedModel.from_pretrained

    def _patched(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        kwargs.setdefault("low_cpu_mem_usage", False)
        kwargs.setdefault("local_files_only", True)
        kwargs.setdefault("device_map", None)

        return _orig_from_pretrained.__func__(
            cls,
            pretrained_model_name_or_path,
            *model_args,
            **kwargs
        )

    PreTrainedModel.from_pretrained = classmethod(_patched)
    print("[patch] Local-only model loading is enabled.")

except Exception as e:
    print("[warning] Failed to apply local-only loading patch:", e)


DEFAULT_MODEL_PATH = os.environ.get("T5_MODEL_PATH", "models/flan-t5-small")
DEFAULT_GTR_PATH = os.environ.get("GTR_MODEL_PATH", DEFAULT_MODEL_PATH)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path: str):
    if path:
        os.makedirs(path, exist_ok=True)


# ==============================
# Text-space classifier
# ==============================

class TextSpaceClassifier(nn.Module):
    def __init__(self, in_dim=512, hidden=256, num_labels=2, p_drop=0.1):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(p_drop),
            nn.Linear(hidden, num_labels),
        )

    def forward(self, x):
        return self.net(torch.nan_to_num(x))


def load_tsc(ckpt_path, device):
    blob = torch.load(ckpt_path, map_location="cpu")

    num_labels = blob.get("num_labels")
    label_map = blob.get("label_map", {})

    # Backward-compatible with checkpoints that store either "gtr_path" or "t5_path".
    gtr_path = blob.get("gtr_path", None) or blob.get("t5_path", None)

    cfg = blob.get(
        "config",
        {
            "in_dim": 512,
            "hidden": 256,
            "p_drop": 0.1,
        }
    )

    model = TextSpaceClassifier(
        in_dim=cfg.get("in_dim", 512),
        hidden=cfg.get("hidden", 256),
        num_labels=num_labels,
        p_drop=cfg.get("p_drop", 0.1),
    )

    model.load_state_dict(blob["state_dict"])
    model.to(device).eval()

    for param in model.parameters():
        param.requires_grad = False

    return model, label_map, gtr_path, cfg


# ==============================
# Encoder wrapper for reward
# ==============================

class GTRWrapper:
    def __init__(self, gtr_path, device, max_length=128):
        self.device = device
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(
            gtr_path,
            local_files_only=True
        )

        try:
            self.encoder = T5EncoderModel.from_pretrained(
                gtr_path,
                local_files_only=True
            ).to(device).eval()

        except Exception:
            base_model = AutoModel.from_pretrained(
                gtr_path,
                local_files_only=True
            ).to(device).eval()

            self.encoder = (
                base_model.get_encoder()
                if hasattr(base_model, "get_encoder")
                else base_model
            )

    @torch.no_grad()
    def encode(self, texts: List[str], batch_size=128):
        outputs = []

        for i in range(0, len(texts), batch_size):
            chunk = texts[i:i + batch_size]

            tokens = self.tokenizer(
                chunk,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            ).to(self.device)

            output = self.encoder(
                input_ids=tokens["input_ids"],
                attention_mask=tokens["attention_mask"]
            )

            last_hidden_state = (
                output.last_hidden_state
                if hasattr(output, "last_hidden_state")
                else output[0]
            )

            attention_mask = tokens["attention_mask"].unsqueeze(-1)

            mean_pooled = (
                (last_hidden_state * attention_mask).sum(dim=1)
                / attention_mask.sum(dim=1).clamp(min=1)
            )

            outputs.append(torch.nan_to_num(mean_pooled))

        if outputs:
            return torch.cat(outputs, dim=0)

        return torch.empty(0, 512, device=self.device)


# ==============================
# Helper functions
# ==============================

def is_cjk_ratio(text: str) -> float:
    total = len(text)
    cjk = 0

    for ch in text:
        if (
            "\u4e00" <= ch <= "\u9fff"
            or "\u3040" <= ch <= "\u30ff"
            or "\uac00" <= ch <= "\ud7af"
        ):
            cjk += 1

    return cjk / max(1, total)


# ==============================
# Projection: z -> encoder space
# ==============================

class ZtoGTR(nn.Module):
    def __init__(self, in_dim, out_dim=512):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(),
            nn.Linear(512, out_dim, bias=False),
        )

    def forward(self, z):
        return self.net(z)


# ==============================
# Inversion model
# ==============================

class SimpleInversionModel(nn.Module):
    def __init__(
        self,
        t5_path: str,
        z_dim: int,
        num_repeat_tokens: int,
        pad_id: int,
        eos_id: int,
        device: torch.device,
        train_t5: bool = False,
        decode_prefix: str = "A sentence is: ",
        t5_tokenizer_path: Optional[str] = None,
    ):
        super().__init__()

        self.tok_pad = pad_id
        self.tok_eos = eos_id
        self.num_repeat = num_repeat_tokens
        self.decode_prefix = decode_prefix

        tokenizer_path = t5_tokenizer_path if t5_tokenizer_path is not None else t5_path

        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            local_files_only=True
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = (
                self.tokenizer.eos_token
                or self.tokenizer.unk_token
            )

        self.t5 = T5ForConditionalGeneration.from_pretrained(
            t5_path,
            local_files_only=True
        )

        self.t5.to(device)

        if not train_t5:
            for param in self.t5.parameters():
                param.requires_grad = False

        d_model = self.t5.config.d_model

        self.proj = nn.Linear(
            z_dim,
            num_repeat_tokens * d_model,
            bias=False
        ).to(device)

        self.d_model = d_model

    def _enc_out_from_z(self, z: torch.Tensor) -> BaseModelOutput:
        batch_size = z.size(0)

        hidden = self.proj(z)
        hidden = hidden.view(
            batch_size,
            self.num_repeat,
            self.d_model
        )

        return BaseModelOutput(last_hidden_state=hidden)

    @torch.no_grad()
    def generate(
        self,
        inputs: dict,
        generation_kwargs: dict,
        decoder_input_ids: Optional[torch.Tensor] = None
    ):
        z = inputs["frozen_embeddings"]
        encoder_outputs = self._enc_out_from_z(z)

        if decoder_input_ids is None:
            prefix_ids = self.tokenizer(
                self.decode_prefix,
                return_tensors="pt"
            ).input_ids.to(z.device)

            prefix_ids = prefix_ids.repeat(z.size(0), 1)
        else:
            prefix_ids = decoder_input_ids

        generated = self.t5.generate(
            encoder_outputs=encoder_outputs,
            decoder_input_ids=prefix_ids,
            max_new_tokens=generation_kwargs.get("max_new_tokens", 48),
            do_sample=generation_kwargs.get("do_sample", True),
            top_p=generation_kwargs.get("top_p", 0.9),
            top_k=generation_kwargs.get("top_k", 40),
            temperature=generation_kwargs.get("temperature", 0.7),
            repetition_penalty=generation_kwargs.get("repetition_penalty", 1.15),
            no_repeat_ngram_size=3,
            pad_token_id=self.tok_pad,
            eos_token_id=self.tok_eos,
        )

        return generated

    def forward(
        self,
        frozen_embeddings: torch.Tensor,
        labels=None,
        decoder_input_ids=None
    ):
        encoder_outputs = self._enc_out_from_z(frozen_embeddings)

        output = self.t5(
            encoder_outputs=encoder_outputs,
            labels=labels,
            decoder_input_ids=decoder_input_ids,
            use_cache=False
        )

        return output


# ==============================
# Stage 1: z -> T5 encoder embedding
# ==============================

def pretrain_z2enc(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_seed(args.seed)

    Z = torch.load(args.z_train)

    if not torch.is_tensor(Z):
        Z = torch.tensor(Z)

    Z = Z.float().to(device)

    df = pd.read_csv(args.labels_csv)

    text_col = None
    for candidate in ["text", "sentence", "content", "prompt", "input"]:
        if candidate in df.columns:
            text_col = candidate
            break

    if text_col is None:
        raise ValueError(
            "Stage-1 projection pretraining requires a text column "
            "such as text, sentence, content, prompt, or input."
        )

    texts = df[text_col].astype(str).tolist()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        local_files_only=True
    )

    encoder = T5EncoderModel.from_pretrained(
        args.model_name,
        local_files_only=True
    ).to(device).eval()

    with torch.no_grad():
        all_targets = []
        batch_size = 64

        for i in range(0, len(texts), batch_size):
            chunk = texts[i:i + batch_size]

            tokens = tokenizer(
                chunk,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt"
            ).to(device)

            output = encoder(
                input_ids=tokens["input_ids"],
                attention_mask=tokens["attention_mask"]
            )

            last_hidden_state = output.last_hidden_state
            attention_mask = tokens["attention_mask"].unsqueeze(-1)

            mean_pooled = (
                (last_hidden_state * attention_mask).sum(dim=1)
                / attention_mask.sum(dim=1).clamp(min=1)
            )

            all_targets.append(mean_pooled)

        target_embeddings = torch.cat(all_targets, dim=0)

    proj = ZtoGTR(
        in_dim=Z.shape[1],
        out_dim=512
    ).to(device)

    optimizer = torch.optim.AdamW(
        proj.parameters(),
        lr=args.proj_lr
    )

    dataset = torch.utils.data.TensorDataset(Z, target_embeddings)

    loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=True
    )

    for epoch in range(1, args.pretrain_epochs + 1):
        for z_batch, target_batch in loader:
            z_batch = z_batch.to(device)
            target_batch = target_batch.to(device)

            pred_batch = proj(z_batch)

            cos_loss = 1 - F.cosine_similarity(
                pred_batch,
                target_batch,
                dim=-1
            ).mean()

            mse_loss = F.mse_loss(pred_batch, target_batch)

            loss = cos_loss + mse_loss * 0.3

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                proj.parameters(),
                max_norm=1.0
            )

            optimizer.step()

        print(f"[pretrain] epoch {epoch} loss={loss.item():.4f}")

    ensure_dir(args.out_dir)

    save_path = os.path.join(args.out_dir, "z2enc.pt")
    torch.save(proj.state_dict(), save_path)

    print("Saved pre-trained projection to", save_path)


# ==============================
# Stage 1b: MLE warm-start
# ==============================

def pretrain_mle(args):
    """
    Stage 1b: MLE warm-start with paired (z, text) samples.

    By default, only the projection layer is trained. With --train_t5_mle,
    the T5 decoder can also be lightly fine-tuned.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_seed(args.seed)
    ensure_dir(args.out_dir)

    Z = torch.load(args.z_train)

    if not torch.is_tensor(Z):
        Z = torch.tensor(Z)

    Z = Z.float()

    df = pd.read_csv(args.labels_csv)

    assert len(df) == Z.shape[0], (
        f"labels_csv has {len(df)} rows, but z has {Z.shape[0]} rows."
    )

    text_col = None
    for candidate in ["text", "sentence", "content", "prompt", "input"]:
        if candidate in df.columns:
            text_col = candidate
            break

    if text_col is None:
        raise ValueError(
            "MLE warm-start requires a text column such as text, sentence, "
            "content, prompt, or input."
        )

    texts = df[text_col].astype(str).tolist()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        local_files_only=True
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id or pad_id

    model = SimpleInversionModel(
        t5_path=args.model_name,
        z_dim=Z.shape[1],
        num_repeat_tokens=args.num_repeat_tokens,
        pad_id=pad_id,
        eos_id=eos_id,
        device=device,
        train_t5=args.train_t5_mle,
        decode_prefix=args.decode_prefix,
        t5_tokenizer_path=args.model_name,
    )

    model.to(device).train()

    z2enc_path = os.path.join(args.out_dir, "z2enc.pt")

    if os.path.exists(z2enc_path):
        print("[info] Loading z2enc.pt before MLE warm-start.")
        state_dict = torch.load(z2enc_path, map_location="cpu")
        model.proj.load_state_dict(state_dict, strict=False)

    if args.train_t5_mle:
        params = [p for p in model.parameters() if p.requires_grad]
    else:
        params = list(model.proj.parameters())

    optimizer = torch.optim.AdamW(
        params,
        lr=args.mle_lr
    )

    class TextDataset(torch.utils.data.Dataset):
        def __init__(self, Z, texts):
            self.Z = Z
            self.texts = texts

        def __len__(self):
            return self.Z.size(0)

        def __getitem__(self, index):
            return self.Z[index], self.texts[index]

    dataset = TextDataset(Z, texts)

    loader = DataLoader(
        dataset,
        batch_size=args.mle_batch_size,
        shuffle=True
    )

    for epoch in range(1, args.mle_epochs + 1):
        for z_batch, text_batch in loader:
            z_batch = z_batch.to(device).float()

            prefixed_texts = [
                args.decode_prefix + text
                for text in text_batch
            ]

            label_tokens = tokenizer(
                prefixed_texts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt"
            ).to(device)

            output = model(
                frozen_embeddings=z_batch,
                labels=label_tokens["input_ids"]
            )

            loss = output.loss

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            optimizer.step()

        print(f"[MLE] epoch {epoch}/{args.mle_epochs} loss={loss.item():.4f}")

    save_path = os.path.join(args.out_dir, "z2enc_mle.pt")
    torch.save(model.proj.state_dict(), save_path)

    print("Saved MLE-tuned projection to", save_path)


# ==============================
# Stage 2: RL training
# ==============================

def train_label_only(args):
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ensure_dir(args.out_dir)

    Z = torch.load(args.z_train)

    if not torch.is_tensor(Z):
        Z = torch.tensor(Z)

    Z = Z.float()

    df = pd.read_csv(args.labels_csv)

    assert len(df) == Z.shape[0], (
        f"labels_csv has {len(df)} rows, but z_specific has {Z.shape[0]} rows."
    )

    classifier, label_map, gtr_path_from_ckpt, _ = load_tsc(
        args.clf_ckpt,
        device
    )

    gtr_path = (
        args.gtr_path
        if args.gtr_path
        else (gtr_path_from_ckpt or DEFAULT_GTR_PATH)
    )

    gtr = GTRWrapper(
        gtr_path=gtr_path,
        device=device
    )

    try:
        classifier_in_dim = next(classifier.parameters()).shape[1]
        probe = gtr.encode(["hello"])

        print(
            f"[debug] classifier input dim={classifier_in_dim}, "
            f"encoder output dim={probe.shape[-1]}"
        )

        assert classifier_in_dim == probe.shape[-1], (
            "The classifier input dimension does not match the encoder output dimension. "
            "Please check the classifier checkpoint and gtr_path."
        )

    except Exception as e:
        print("[warning] Dimension check failed:", e)

    if args.label_col not in df.columns:
        raise ValueError(
            f"{args.labels_csv} must contain the column {args.label_col}."
        )

    raw_labels = df[args.label_col].astype(str).tolist()

    if not all(label in label_map for label in raw_labels):
        missing = sorted(set(raw_labels) - set(label_map.keys()))

        raise ValueError(
            f"Some labels are not included in label_map: {missing[:10]}"
        )

    y = torch.tensor(
        [label_map[label] for label in raw_labels],
        dtype=torch.long
    )

    tokenizer_train = AutoTokenizer.from_pretrained(
        args.model_name,
        local_files_only=True
    )

    if tokenizer_train.pad_token is None:
        tokenizer_train.pad_token = (
            tokenizer_train.eos_token
            or tokenizer_train.unk_token
        )

    pad_id = tokenizer_train.pad_token_id
    eos_id = tokenizer_train.eos_token_id or pad_id

    model = SimpleInversionModel(
        t5_path=args.model_name,
        z_dim=Z.shape[1],
        num_repeat_tokens=args.num_repeat_tokens,
        pad_id=pad_id,
        eos_id=eos_id,
        device=device,
        train_t5=args.train_t5,
        decode_prefix=args.decode_prefix,
        t5_tokenizer_path=args.model_name,
    )

    mle_path = os.path.join(args.out_dir, "z2enc_mle.pt")
    pretrain_path = os.path.join(args.out_dir, "z2enc.pt")

    if os.path.exists(mle_path):
        print("[info] Loading z2enc_mle.pt.")
        state_dict = torch.load(mle_path, map_location="cpu")
        model.proj.load_state_dict(state_dict, strict=False)

    elif os.path.exists(pretrain_path):
        print("[info] Loading pre-trained z2enc.pt.")
        state_dict = torch.load(pretrain_path, map_location="cpu")
        model.proj.load_state_dict(state_dict, strict=False)

    else:
        print(
            "[warning] No z2enc*.pt found. "
            "RL training will start from a randomly initialized projection."
        )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.rl_lr
    )

    text_col = None
    for candidate in ["text", "sentence", "content", "prompt", "input"]:
        if candidate in df.columns:
            text_col = candidate
            break

    if text_col is not None:
        print("[info] Encoding instance-level target texts for semantic reward.")

        instance_texts = df[text_col].astype(str).tolist()

        with torch.no_grad():
            instance_targets = gtr.encode(
                instance_texts,
                batch_size=128
            )

    else:
        instance_targets = None
        print("[info] No text column found. Instance-level semantic reward is disabled.")

    indices = torch.arange(
        Z.shape[0],
        dtype=torch.long
    )

    dataset = torch.utils.data.TensorDataset(
        Z,
        y,
        indices
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True
    )

    reward_ma = None
    beta = args.reward_ma_beta

    recent_texts = deque(maxlen=args.recent_capacity)
    recent_heads = deque(maxlen=args.recent_capacity)

    for epoch in range(args.epochs):
        step = 0

        for embeddings, y_batch, index_batch in loader:
            step += 1

            embeddings = embeddings.to(device).float()
            y_batch = y_batch.to(device)
            index_batch = index_batch.to(device)

            with torch.no_grad():
                generated_ids = model.generate(
                    inputs={"frozen_embeddings": embeddings},
                    generation_kwargs={
                        "max_new_tokens": args.gen_max_new_tokens,
                        "do_sample": True,
                        "top_p": args.top_p,
                        "top_k": args.top_k,
                        "temperature": args.temperature,
                        "repetition_penalty": args.train_repetition_penalty,
                    }
                )

                texts = model.tokenizer.batch_decode(
                    generated_ids,
                    skip_special_tokens=True
                )

                generated_ids = generated_ids.to(device)

            batch_size = len(texts)

            with torch.no_grad():
                gtr_embeddings = gtr.encode(texts)

                logits = classifier(gtr_embeddings)
                log_probs = F.log_softmax(logits, dim=-1)

                r_cls = log_probs[
                    torch.arange(y_batch.size(0), device=device),
                    y_batch
                ]

                other = log_probs.clone()
                other[
                    torch.arange(other.size(0), device=device),
                    y_batch
                ] = -1e9

                other_max = other.max(dim=-1).values
                r_margin = r_cls - other_max

                probs = F.softmax(logits, dim=-1)
                entropy = -(
                    probs * torch.log(probs + 1e-8)
                ).sum(dim=-1)

                r_conf = -entropy

                lengths = (
                    generated_ids != tokenizer_train.pad_token_id
                ).sum(dim=-1).float()

                r_len = -(
                    (lengths - args.len_target) ** 2
                ) / (2 * (args.len_sigma ** 2))

                reward = (
                    args.w_cls * r_cls
                    + args.w_cls_margin * r_margin
                    + args.w_len * r_len
                    + 0.05 * r_conf
                )

                if args.w_lang > 0:
                    lang_bonus = []

                    for text in texts:
                        cjk_ratio = is_cjk_ratio(text)

                        if cjk_ratio > 0.3:
                            lang_bonus.append(0.0)
                        else:
                            ascii_count = sum(
                                1
                                for ch in text
                                if 32 <= ord(ch) < 127
                            )
                            ratio = ascii_count / max(1, len(text))
                            lang_bonus.append(ratio - 0.9)

                    lang_bonus = torch.tensor(
                        lang_bonus,
                        device=device,
                        dtype=reward.dtype
                    )

                    reward = reward + args.w_lang * lang_bonus

                if instance_targets is not None and args.w_inst > 0:
                    instance_embeddings = instance_targets[index_batch].to(device)

                    r_inst = F.cosine_similarity(
                        gtr_embeddings,
                        instance_embeddings,
                        dim=-1
                    )

                    reward = reward + args.w_inst * r_inst

                lower_texts = [
                    text.strip().lower()
                    for text in texts
                ]

                for i, text in enumerate(lower_texts):
                    if lower_texts.count(text) > batch_size * args.collapse_ratio:
                        reward[i] = reward[i] - args.collapse_penalty

                if args.w_ngram > 0:
                    heads = [
                        " ".join(text.split()[:4]).lower()
                        for text in texts
                    ]

                    for i, head in enumerate(heads):
                        if heads.count(head) > 1:
                            reward[i] = reward[i] - args.w_ngram

                for i, text in enumerate(lower_texts):
                    if text in recent_texts:
                        reward[i] = reward[i] - args.collapse_penalty * 0.5

                heads = [
                    " ".join(text.split()[:4]).lower()
                    for text in texts
                ]

                for i, head in enumerate(heads):
                    if head in recent_heads:
                        reward[i] = reward[i] - args.collapse_penalty * 0.3

                recent_texts.extend(lower_texts)
                recent_heads.extend(heads)

                if args.w_div > 0:
                    sim_mat = F.cosine_similarity(
                        gtr_embeddings.unsqueeze(1),
                        gtr_embeddings.unsqueeze(0),
                        dim=-1
                    )

                    sim_mat = sim_mat - torch.eye(
                        batch_size,
                        device=device
                    ) * 10

                    max_sim, _ = sim_mat.max(dim=-1)

                    r_div = -max_sim
                    reward = reward + args.w_div * r_div

                if reward_ma is None:
                    reward_ma = reward.mean().item()

                advantage = reward - reward_ma
                reward_ma = (
                    beta * reward_ma
                    + (1 - beta) * reward.mean().item()
                )

            output = model(
                frozen_embeddings=embeddings,
                labels=None,
                decoder_input_ids=generated_ids[:, :-1]
            )

            logits2 = output.logits
            target = generated_ids[:, 1:].contiguous()

            log_probs2 = F.log_softmax(logits2, dim=-1)

            target_log_probs = log_probs2.gather(
                dim=-1,
                index=target.unsqueeze(-1)
            ).squeeze(-1)

            mask = (
                target != tokenizer_train.pad_token_id
            ).float()

            target_log_prob_mean = (
                (target_log_probs * mask).sum(dim=-1)
                / mask.sum(dim=-1).clamp(min=1)
            )

            normalized_advantage = advantage.detach()

            if args.adv_clip:
                normalized_advantage = normalized_advantage.clamp(
                    -args.adv_clip,
                    args.adv_clip
                )

            normalized_advantage = (
                (normalized_advantage - normalized_advantage.mean())
                / (normalized_advantage.std() + 1e-6)
            )

            rl_loss = -(
                normalized_advantage * target_log_prob_mean
            ).mean()

            optimizer.zero_grad()
            rl_loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            optimizer.step()

            if step % 20 == 0:
                print(
                    f"[RL] epoch {epoch + 1}/{args.epochs} step {step} | "
                    f"rl={rl_loss.item():.4f} | "
                    f"reward={reward.mean().item():.3f} | "
                    f"cls={r_cls.mean().item():.3f}"
                )

        save_dir = os.path.join(args.out_dir, f"ep{epoch + 1}")

        ensure_dir(save_dir)

        model.tokenizer.save_pretrained(save_dir)
        model.t5.save_pretrained(os.path.join(save_dir, "t5"))

        torch.save(
            model.proj.state_dict(),
            os.path.join(save_dir, "proj.pt")
        )

        meta = {
            "num_repeat_tokens": args.num_repeat_tokens,
            "pad_token_id": int(model.tokenizer.pad_token_id),
            "eos_token_id": int(
                model.tokenizer.eos_token_id
                or model.tokenizer.pad_token_id
            ),
            "decode_prefix": args.decode_prefix,
        }

        with open(os.path.join(save_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(
                meta,
                f,
                ensure_ascii=False,
                indent=2
            )

        print(f"Saved checkpoint to {save_dir}")


# ==============================
# Inference
# ==============================

@torch.no_grad()
def inference(args):
    """
    Generate text with K candidates per z and rerank by classifier score
    and semantic consistency.

    The output CSV preserves row alignment:
        output row i corresponds to Z[i] and labels_csv row i.

    Optional prompt-constrained inference:
        If --prompt_json and --prompt_n_supervised are provided, the first N
        samples use identity and attribute parsed from prompt_json as decoder
        prefixes.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(
        args.load_from,
        local_files_only=True
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    with open(os.path.join(args.load_from, "meta.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)

    num_repeat = int(meta["num_repeat_tokens"])
    pad_id = int(meta["pad_token_id"])
    eos_id = int(meta["eos_token_id"])
    decode_prefix = meta.get("decode_prefix", "A sentence is: ")

    proj_state_dict = torch.load(
        os.path.join(args.load_from, "proj.pt"),
        map_location="cpu"
    )

    z_dim = proj_state_dict["weight"].shape[1]

    t5_dir = os.path.join(args.load_from, "t5")

    model = SimpleInversionModel(
        t5_path=t5_dir,
        z_dim=z_dim,
        num_repeat_tokens=num_repeat,
        pad_id=pad_id,
        eos_id=eos_id,
        device=device,
        train_t5=False,
        decode_prefix=decode_prefix,
        t5_tokenizer_path=args.load_from,
    )

    model.eval()
    model.proj.load_state_dict(proj_state_dict)
    model.proj.to(device)

    classifier, label_map, gtr_path_from_ckpt, _ = load_tsc(
        args.clf_ckpt,
        device
    )

    gtr_path = (
        args.gtr_path
        if args.gtr_path
        else (gtr_path_from_ckpt or DEFAULT_GTR_PATH)
    )

    gtr = GTRWrapper(
        gtr_path=gtr_path,
        device=device
    )

    try:
        classifier_in_dim = next(classifier.parameters()).shape[1]
        probe = gtr.encode(["hello"])

        print(
            f"[debug] classifier input dim={classifier_in_dim}, "
            f"encoder output dim={probe.shape[-1]}"
        )

        assert classifier_in_dim == probe.shape[-1], (
            "The classifier input dimension does not match the encoder output dimension. "
            "Please check the classifier checkpoint and gtr_path."
        )

    except Exception as e:
        print("[warning] Dimension check failed:", e)

    df_labels = pd.read_csv(args.labels_csv)

    Z = torch.load(args.z_test)

    if not torch.is_tensor(Z):
        Z = torch.tensor(Z)

    Z = Z.float()

    assert len(df_labels) == Z.shape[0], (
        f"labels_csv has {len(df_labels)} rows, but z_test has {Z.shape[0]} rows."
    )

    y_raw = df_labels[args.label_col].astype(str).tolist()
    y_true = [label_map[label] for label in y_raw]

    prompt_meta = None

    if args.prompt_json and args.prompt_n_supervised > 0:
        with open(args.prompt_json, "r", encoding="utf-8") as f:
            raw_prompts = json.load(f)

        if not isinstance(raw_prompts, list):
            raise ValueError("prompt_json must be a list.")

        def extract_identity_attr(prompt_text: str):
            """
            Extract the first two {...} spans from a prompt.

            The first span is treated as identity and the second span as
            attribute. Later empty braces are ignored.
            """
            spans = []
            current = []
            inside = False

            for ch in prompt_text:
                if ch == "{":
                    inside = True
                    current = []
                elif ch == "}":
                    if inside:
                        spans.append("".join(current).strip())
                        inside = False
                else:
                    if inside:
                        current.append(ch)

            identity = spans[0] if len(spans) >= 1 else None
            attribute = spans[1] if len(spans) >= 2 else None

            return identity, attribute

        prompt_meta = []

        for item in raw_prompts:
            if isinstance(item, dict) and "prompt" in item:
                prompt_text = item["prompt"]
            else:
                prompt_text = str(item)

            identity, attribute = extract_identity_attr(prompt_text)

            prompt_meta.append(
                {
                    "prompt": prompt_text,
                    "identity": identity,
                    "attr": attribute,
                }
            )

        if len(prompt_meta) < args.prompt_n_supervised:
            print(
                f"[warning] prompt_json contains only {len(prompt_meta)} items. "
                f"Prompt constraints will be used only for these samples."
            )
        else:
            print(
                f"[info] Prompt constraints will be used for the first "
                f"{args.prompt_n_supervised} samples."
            )

    batch_size = args.batch_size
    generations = []
    num_candidates = args.infer_n_candidates

    for i in range(0, Z.shape[0], batch_size):
        embeddings = Z[i:i + batch_size].to(device).float()

        current_batch_size = embeddings.size(0)

        y_batch = torch.tensor(
            y_true[i:i + current_batch_size],
            device=device,
            dtype=torch.long
        )

        semantic_target = model._enc_out_from_z(
            embeddings
        ).last_hidden_state.mean(dim=1)

        if prompt_meta is not None and args.prompt_n_supervised > 0:
            prefix_texts = []

            for j in range(current_batch_size):
                global_index = i + j

                if (
                    global_index < args.prompt_n_supervised
                    and global_index < len(prompt_meta)
                ):
                    meta_j = prompt_meta[global_index]

                    identity = meta_j.get("identity")
                    attribute = meta_j.get("attr")

                    if identity and attribute:
                        prefix = f"{identity} who is {{{attribute}}}"
                    elif attribute:
                        prefix = f"{{{attribute}}}"
                    else:
                        prefix = decode_prefix
                else:
                    prefix = decode_prefix

                prefix_texts.append(prefix)

            decoder_input_ids = tokenizer(
                prefix_texts,
                padding=True,
                truncation=True,
                max_length=64,
                return_tensors="pt"
            ).input_ids.to(device)

        else:
            decoder_input_ids = None

        best_texts = [""] * current_batch_size

        best_scores = torch.full(
            (current_batch_size,),
            -1e9,
            device=device,
            dtype=torch.float32
        )

        for _ in range(num_candidates):
            generated_ids = model.generate(
                inputs={"frozen_embeddings": embeddings},
                generation_kwargs={
                    "max_new_tokens": args.gen_max_new_tokens,
                    "do_sample": True,
                    "top_p": args.infer_top_p,
                    "top_k": args.infer_top_k,
                    "temperature": args.infer_temperature,
                    "repetition_penalty": 1.15,
                },
                decoder_input_ids=decoder_input_ids,
            )

            texts = tokenizer.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )

            gtr_embeddings = gtr.encode(texts)

            logits = classifier(gtr_embeddings)
            log_probs = F.log_softmax(logits, dim=-1)

            cls_score = log_probs[
                torch.arange(current_batch_size, device=device),
                y_batch
            ]

            sem_score = F.cosine_similarity(
                gtr_embeddings,
                semantic_target,
                dim=-1
            )

            score = (
                args.infer_rerank_alpha * cls_score
                + args.infer_rerank_beta * sem_score
            )

            for j in range(current_batch_size):
                if score[j] > best_scores[j]:
                    best_scores[j] = score[j]
                    best_texts[j] = texts[j]

        generations.extend(best_texts)

    output_dir = (
        os.path.dirname(args.infer_out)
        if os.path.dirname(args.infer_out)
        else "."
    )

    ensure_dir(output_dir)

    output_df = pd.DataFrame(
        {
            "gen_text": generations,
            "true_label": y_raw[:len(generations)],
            "row_index": np.arange(len(generations)),
        }
    )

    for col in ["id", "text", "sentence", "content", "prompt", "input"]:
        if col in df_labels.columns:
            output_df[col] = (
                df_labels[col]
                .astype(str)
                .iloc[:len(generations)]
                .values
            )

    output_df.to_csv(
        args.infer_out,
        index=False
    )

    print("Wrote", len(generations), "rows to", args.infer_out)


# ==============================
# Evaluation
# ==============================

@torch.no_grad()
def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    classifier, label_map, gtr_path_from_ckpt, _ = load_tsc(
        args.clf_ckpt,
        device
    )

    gtr_path = (
        args.gtr_path
        if args.gtr_path
        else (gtr_path_from_ckpt or DEFAULT_GTR_PATH)
    )

    gtr = GTRWrapper(
        gtr_path=gtr_path,
        device=device
    )

    df_gen = pd.read_csv(args.gen_csv)

    if "gen_text" in df_gen.columns:
        texts = df_gen["gen_text"].astype(str).tolist()
    else:
        texts = df_gen.iloc[:, 0].astype(str).tolist()

    generated_embeddings = gtr.encode(texts)

    logits = classifier(generated_embeddings)
    pred = logits.argmax(dim=-1).cpu().numpy()

    df_labels = pd.read_csv(args.labels_csv)

    y_raw = (
        df_labels[args.label_col]
        .astype(str)
        .tolist()[:len(pred)]
    )

    y_true = np.array(
        [label_map[label] for label in y_raw]
    )

    acc = (pred == y_true).mean()

    print(f"[Eval] Accuracy on generated texts: {acc:.4f}")


# ==============================
# Arguments
# ==============================

def parse_args():
    parser = argparse.ArgumentParser()

    # Data.
    parser.add_argument("--z_train", type=str)
    parser.add_argument("--labels_csv", type=str)
    parser.add_argument("--label_col", type=str, default="label")

    # Classifier.
    parser.add_argument("--clf_ckpt", type=str)

    # Base model.
    parser.add_argument(
        "--model_name",
        type=str,
        default=DEFAULT_MODEL_PATH
    )

    parser.add_argument(
        "--decode_prefix",
        type=str,
        default="A sentence is: "
    )

    parser.add_argument(
        "--num_repeat_tokens",
        type=int,
        default=24
    )

    # RL.
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--rl_lr", type=float, default=1e-5)
    parser.add_argument("--gen_max_new_tokens", type=int, default=48)
    parser.add_argument("--top_p", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=40)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--w_cls", type=float, default=1.6)
    parser.add_argument("--w_cls_margin", type=float, default=0.6)
    parser.add_argument("--w_len", type=float, default=0.02)
    parser.add_argument("--len_target", type=float, default=18.0)
    parser.add_argument("--len_sigma", type=float, default=8.0)
    parser.add_argument("--w_lang", type=float, default=0.0)
    parser.add_argument("--w_inst", type=float, default=0.0)
    parser.add_argument("--adv_clip", type=float, default=1.0)
    parser.add_argument("--reward_ma_beta", type=float, default=0.9)
    parser.add_argument("--train_repetition_penalty", type=float, default=1.15)
    parser.add_argument("--collapse_ratio", type=float, default=0.05)
    parser.add_argument("--collapse_penalty", type=float, default=1.0)
    parser.add_argument("--w_div", type=float, default=0.0)
    parser.add_argument("--w_ngram", type=float, default=0.0)
    parser.add_argument("--recent_capacity", type=int, default=128)

    # Projection.
    parser.add_argument("--use_proj_semantic", action="store_true")
    parser.add_argument("--proj_lr", type=float, default=5e-4)

    # Encoder model for reward.
    parser.add_argument("--gtr_path", type=str, default=None)

    # Directories.
    parser.add_argument(
        "--out_dir",
        type=str,
        default="saves/zspec_label_only"
    )

    # Inference.
    parser.add_argument("--load_from", type=str)
    parser.add_argument("--z_test", type=str)

    parser.add_argument(
        "--infer_out",
        type=str,
        default="saves/test_generations.csv"
    )

    parser.add_argument("--infer_do_sample", action="store_true")
    parser.add_argument("--infer_top_p", type=float, default=0.7)
    parser.add_argument("--infer_top_k", type=int, default=50)
    parser.add_argument("--infer_temperature", type=float, default=0.6)
    parser.add_argument("--infer_n_candidates", type=int, default=10)
    parser.add_argument("--infer_rerank_alpha", type=float, default=1.0)
    parser.add_argument("--infer_rerank_beta", type=float, default=0.4)

    # Prompt-constrained inference.
    parser.add_argument(
        "--prompt_json",
        type=str,
        default=None,
        help=(
            "A JSON list file. Each item may contain a 'prompt' field. "
            "Used to constrain the first N generated samples."
        )
    )

    parser.add_argument(
        "--prompt_n_supervised",
        type=int,
        default=0,
        help=(
            "Number of initial samples that should also serve as answers "
            "to the prompts in prompt_json."
        )
    )

    # Evaluation.
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--gen_csv", type=str)

    # Stage 1.
    parser.add_argument("--pretrain_z2enc", action="store_true")
    parser.add_argument("--pretrain_epochs", type=int, default=10)

    # Stage 1b.
    parser.add_argument("--pretrain_mle", action="store_true")
    parser.add_argument("--mle_epochs", type=int, default=2)
    parser.add_argument("--mle_lr", type=float, default=5e-5)
    parser.add_argument("--mle_batch_size", type=int, default=16)
    parser.add_argument("--train_t5_mle", action="store_true")

    # Misc.
    parser.add_argument(
        "--train_t5",
        action="store_true",
        help="Fine-tune T5 during RL training."
    )

    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


# ==============================
# Entry
# ==============================

def main():
    args = parse_args()

    if args.pretrain_z2enc:
        pretrain_z2enc(args)
        return

    if args.pretrain_mle:
        pretrain_mle(args)
        return

    if (
        args.z_train
        and args.labels_csv
        and args.clf_ckpt
        and not args.load_from
        and not args.evaluate
    ):
        train_label_only(args)
        return

    if args.load_from and args.z_test and not args.evaluate:
        inference(args)
        return

    if args.evaluate and args.gen_csv and args.labels_csv and args.clf_ckpt:
        evaluate(args)
        return

    print(
        "Run one of the following modes: "
        "--pretrain_z2enc | --pretrain_mle | RL training | inference | --evaluate"
    )


if __name__ == "__main__":
    main()