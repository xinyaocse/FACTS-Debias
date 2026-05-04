# -*- coding: utf-8 -*-
"""
Label-consistent inversion (2-stage, MLE warmup, anti-collapse, instance-semantic, rerank-inference)

Stage 1: z -> T5-encoder space (MSE + cosine), freeze T5
Stage 1b: MLE warm-start on paired (z, text), teacher forcing (optional decoder finetuning)
Stage 2: RL on (optionally) frozen T5 decoder
Reward = cls + margin + len + (optional) lang + diversity + instance_semantic - collapse
Inference = K candidates per z, classifier + instance-sem rerank
"""

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

# ---- safe patch ----
try:
    from transformers.modeling_utils import PreTrainedModel
    _orig_from_pretrained = PreTrainedModel.from_pretrained

    def _patched(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        kwargs.setdefault("low_cpu_mem_usage", False)
        kwargs.setdefault("local_files_only", True)
        kwargs.setdefault("device_map", None)
        return _orig_from_pretrained.__func__(cls, pretrained_model_name_or_path, *model_args, **kwargs)

    PreTrainedModel.from_pretrained = classmethod(_patched)
    print("[patch] local-only T5 enabled.")
except Exception as e:
    print("[warn] patch failed:", e)

DEFAULT_MODEL_PATH = "/root/public/lcx/GenderCARE-ccs24-main/Code/flan-t5-small"
DEFAULT_GTR_PATH   = "/root/public/lcx/GenderCARE-ccs24-main/Code/flan-t5-small"


def set_seed(s=42):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(p: str):
    if p:
        os.makedirs(p, exist_ok=True)


# ============ classifier ============

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
    gtr_path = blob.get("gtr_path", None)
    cfg = blob.get("config", {"in_dim": 512, "hidden": 256, "p_drop": 0.1})
    model = TextSpaceClassifier(
        in_dim=cfg.get("in_dim", 512),
        hidden=cfg.get("hidden", 256),
        num_labels=num_labels,
        p_drop=cfg.get("p_drop", 0.1),
    )
    model.load_state_dict(blob["state_dict"])
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, label_map, gtr_path, cfg


# ============ encoder for reward ============

class GTRWrapper:
    def __init__(self, gtr_path, device, max_length=128):
        self.device = device
        self.max_length = max_length
        self.tok = AutoTokenizer.from_pretrained(gtr_path, local_files_only=True)
        try:
            self.enc = T5EncoderModel.from_pretrained(gtr_path, local_files_only=True).to(device).eval()
        except Exception:
            base = AutoModel.from_pretrained(gtr_path, local_files_only=True).to(device).eval()
            self.enc = base.get_encoder() if hasattr(base, "get_encoder") else base

    @torch.no_grad()
    def encode(self, texts: List[str], batch_size=128):
        outs = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i:i + batch_size]
            toks = self.tok(
                chunk,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            ).to(self.device)
            out = self.enc(input_ids=toks["input_ids"], attention_mask=toks["attention_mask"])
            last = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
            mask = toks["attention_mask"].unsqueeze(-1)
            mean = (last * mask).sum(1) / mask.sum(1).clamp(min=1)
            outs.append(torch.nan_to_num(mean))
        return torch.cat(outs, dim=0) if outs else torch.empty(0, 512, device=self.device)


# ============ helpers ============

def is_cjk_ratio(s: str) -> float:
    total = len(s)
    cjk = 0
    for ch in s:
        if '\u4e00' <= ch <= '\u9fff' or '\u3040' <= ch <= '\u30ff' or '\uac00' <= ch <= '\ud7af':
            cjk += 1
    return cjk / max(1, total)


# ============ proj z -> 512 ============

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


# ============ main inversion model ============

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

        tok_path = t5_tokenizer_path if t5_tokenizer_path is not None else t5_path
        self.tok = AutoTokenizer.from_pretrained(tok_path, local_files_only=True)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token or self.tok.unk_token

        self.t5 = T5ForConditionalGeneration.from_pretrained(t5_path, local_files_only=True)
        self.t5.to(device)
        if not train_t5:
            for p in self.t5.parameters():
                p.requires_grad = False

        d_model = self.t5.config.d_model  # 512
        self.proj = nn.Linear(z_dim, num_repeat_tokens * d_model, bias=False).to(device)
        self.d_model = d_model

    def _enc_out_from_z(self, z: torch.Tensor) -> BaseModelOutput:
        B = z.size(0)
        h = self.proj(z)
        h = h.view(B, self.num_repeat, self.d_model)
        return BaseModelOutput(last_hidden_state=h)

    @torch.no_grad()
    def generate(self, inputs: dict, generation_kwargs: dict, decoder_input_ids: Optional[torch.Tensor] = None):
        z = inputs["frozen_embeddings"]
        enc = self._enc_out_from_z(z)

        if decoder_input_ids is None:
            prefix_ids = self.tok(self.decode_prefix, return_tensors="pt").input_ids.to(z.device)
            prefix_ids = prefix_ids.repeat(z.size(0), 1)
        else:
            prefix_ids = decoder_input_ids

        gen = self.t5.generate(
            encoder_outputs=enc,
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
        return gen

    def forward(self, frozen_embeddings: torch.Tensor, labels=None, decoder_input_ids=None):
        enc = self._enc_out_from_z(frozen_embeddings)
        out = self.t5(
            encoder_outputs=enc,
            labels=labels,
            decoder_input_ids=decoder_input_ids,
            use_cache=False
        )
        return out


# ============ Stage 1: z -> T5 encoder embedding ============

def pretrain_z2enc(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    Z = torch.load(args.z_train)
    if not torch.is_tensor(Z):
        Z = torch.tensor(Z)
    Z = Z.float().to(device)

    df = pd.read_csv(args.labels_csv)
    text_col = None
    for cand in ["text", "sentence", "content", "prompt", "input"]:
        if cand in df.columns:
            text_col = cand
            break
    if text_col is None:
        raise ValueError("预训练投影需要 CSV 里有 text / sentence 这一类列。")

    texts = df[text_col].astype(str).tolist()
    tok = AutoTokenizer.from_pretrained(args.model_name, local_files_only=True)
    enc = T5EncoderModel.from_pretrained(args.model_name, local_files_only=True).to(device).eval()

    with torch.no_grad():
        all_t = []
        bs = 64
        for i in range(0, len(texts), bs):
            chunk = texts[i:i + bs]
            toks = tok(chunk, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
            out = enc(input_ids=toks["input_ids"], attention_mask=toks["attention_mask"])
            last = out.last_hidden_state
            mask = toks["attention_mask"].unsqueeze(-1)
            mean = (last * mask).sum(1) / mask.sum(1).clamp(min=1)
            all_t.append(mean)
        tgt_emb = torch.cat(all_t, dim=0)

    proj = ZtoGTR(in_dim=Z.shape[1], out_dim=512).to(device)
    opt = torch.optim.AdamW(proj.parameters(), lr=args.proj_lr)

    ds = torch.utils.data.TensorDataset(Z, tgt_emb)
    dl = DataLoader(ds, batch_size=64, shuffle=True)

    for ep in range(1, args.pretrain_epochs + 1):
        for z_b, t_b in dl:
            z_b = z_b.to(device)
            t_b = t_b.to(device)
            out_b = proj(z_b)
            cos_loss = 1 - F.cosine_similarity(out_b, t_b, dim=-1).mean()
            mse_loss = F.mse_loss(out_b, t_b)
            loss = cos_loss + mse_loss * 0.3
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(proj.parameters(), 1.0)
            opt.step()
        print(f"[pretrain] ep {ep} loss={loss.item():.4f}")

    ensure_dir(args.out_dir)
    torch.save(proj.state_dict(), os.path.join(args.out_dir, "z2enc.pt"))
    print("[✓] saved pre-trained projection to", os.path.join(args.out_dir, "z2enc.pt"))


# ============ Stage 1b: MLE warm-start ============

def pretrain_mle(args):
    """
    Stage-1b: 用 paired (z, text) 做 MLE 暖启动
    仅训练 model.proj（默认），可选 --train_t5_mle 轻微微调解码器
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    ensure_dir(args.out_dir)

    Z = torch.load(args.z_train)
    if not torch.is_tensor(Z):
        Z = torch.tensor(Z)
    Z = Z.float()

    df = pd.read_csv(args.labels_csv)
    assert len(df) == Z.shape[0], f"labels_csv 行数({len(df)})需与 z 行数({Z.shape[0]})一致"

    text_col = None
    for cand in ["text", "sentence", "content", "prompt", "input"]:
        if cand in df.columns:
            text_col = cand
            break
    if text_col is None:
        raise ValueError("MLE 需要 CSV 里有文本列（text/sentence/content/prompt/input）。")
    texts = df[text_col].astype(str).tolist()

    tok = AutoTokenizer.from_pretrained(args.model_name, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token or tok.unk_token
    pad_id = tok.pad_token_id
    eos_id = tok.eos_token_id or pad_id

    model = SimpleInversionModel(
        t5_path=args.model_name,
        z_dim=Z.shape[1],
        num_repeat_tokens=args.num_repeat_tokens,
        pad_id=pad_id,
        eos_id=eos_id,
        device=device,
        train_t5=args.train_t5_mle,           # 默认 False
        decode_prefix=args.decode_prefix,
        t5_tokenizer_path=args.model_name,
    )
    model.to(device).train()

    # 如果有 z2enc.pt，则先载入
    z2enc_path = os.path.join(args.out_dir, "z2enc.pt")
    if os.path.exists(z2enc_path):
        print("[info] loading z2enc.pt before MLE")
        sd = torch.load(z2enc_path, map_location="cpu")
        model.proj.load_state_dict(sd, strict=False)

    params = list(model.proj.parameters()) if not args.train_t5_mle else [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(params, lr=args.mle_lr)

    class _TDS(torch.utils.data.Dataset):
        def __init__(self, Z, T):
            self.Z = Z
            self.T = T
        def __len__(self): return self.Z.size(0)
        def __getitem__(self, i): return self.Z[i], self.T[i]
    ds = _TDS(Z, texts)
    dl = DataLoader(ds, batch_size=args.mle_batch_size, shuffle=True)

    for ep in range(1, args.mle_epochs + 1):
        for z_b, t_b in dl:
            z_b = z_b.to(device).float()
            t_b_prefixed = [args.decode_prefix + s for s in t_b]
            toks_lab = tok(t_b_prefixed, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
            out = model(frozen_embeddings=z_b, labels=toks_lab["input_ids"])  # 不要传 decoder_input_ids
            loss = out.loss
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
        print(f"[MLE] ep {ep}/{args.mle_epochs} loss={loss.item():.4f}")

    torch.save(model.proj.state_dict(), os.path.join(args.out_dir, "z2enc_mle.pt"))
    print("[✓] saved MLE-tuned projection to", os.path.join(args.out_dir, "z2enc_mle.pt"))


# ============ Stage 2: RL ============

def train_label_only(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ensure_dir(args.out_dir)

    # z
    Z = torch.load(args.z_train)
    if not torch.is_tensor(Z):
        Z = torch.tensor(Z)
    Z = Z.float()

    # csv
    df = pd.read_csv(args.labels_csv)
    assert len(df) == Z.shape[0], f"labels_csv 行数({len(df)})需与 z_specific 行数({Z.shape[0]})一致"

    # classifier
    clf, label_map, gtr_path_from_ckpt, _ = load_tsc(args.clf_ckpt, device)
    gtr_path = args.gtr_path if args.gtr_path else (gtr_path_from_ckpt or DEFAULT_GTR_PATH)
    gtr = GTRWrapper(gtr_path=gtr_path, device=device)

    # debug shape check
    try:
        clf_in_dim = next(clf.parameters()).shape[1]
        probe = gtr.encode(["hello"])
        print(f"[dbg] clf in_dim={clf_in_dim}, gtr output dim={probe.shape[-1]}")
        assert clf_in_dim == probe.shape[-1], "分类器输入维度与嵌入器输出维度不一致，请检查 ckpt 与 gtr_path。"
    except Exception as e:
        print("[warn] dim check failed:", e)

    # label
    if args.label_col not in df.columns:
        raise ValueError(f"{args.labels_csv} 必须含列 {args.label_col}")
    raw_y = df[args.label_col].astype(str).tolist()
    if not all(y in label_map for y in raw_y):
        miss = sorted(set(raw_y) - set(label_map.keys()))
        raise ValueError(f"label 不在 label_map: {miss[:10]}")
    y = torch.tensor([label_map[y] for y in raw_y], dtype=torch.long)

    # tokenizer for training
    tok_train = AutoTokenizer.from_pretrained(args.model_name, local_files_only=True)
    if tok_train.pad_token is None:
        tok_train.pad_token = tok_train.eos_token or tok_train.unk_token
    pad_id = tok_train.pad_token_id
    eos_id = tok_train.eos_token_id or pad_id

    # model
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

    # load stage1 proj (prefer MLE)
    mle_path = os.path.join(args.out_dir, "z2enc_mle.pt")
    pt_path  = os.path.join(args.out_dir, "z2enc.pt")
    if os.path.exists(mle_path):
        print("[info] loading z2enc_mle.pt")
        sd = torch.load(mle_path, map_location="cpu")
        model.proj.load_state_dict(sd, strict=False)
    elif os.path.exists(pt_path):
        print("[info] loading pre-trained z2enc.pt")
        sd = torch.load(pt_path, map_location="cpu")
        model.proj.load_state_dict(sd, strict=False)
    else:
        print("[warn] no z2enc*.pt found, RL will start from scratch projection.")

    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.rl_lr)

    # ========= instance-level target embedding =========
    text_col = None
    for cand in ["text", "sentence", "content", "prompt", "input"]:
        if cand in df.columns:
            text_col = cand
            break
    if text_col is not None:
        print("[info] encoding instance-level target texts for semantic reward ...")
        inst_texts = df[text_col].astype(str).tolist()
        with torch.no_grad():
            inst_targets = gtr.encode(inst_texts, batch_size=128)
    else:
        inst_targets = None
        print("[info] no text column found, instance-semantic reward disabled.")

    # dataloader
    idxs = torch.arange(Z.shape[0], dtype=torch.long)
    ds = torch.utils.data.TensorDataset(Z, y, idxs)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True)

    reward_ma = None
    beta = args.reward_ma_beta
    recent_texts = deque(maxlen=args.recent_capacity)
    recent_heads = deque(maxlen=args.recent_capacity)

    for ep in range(args.epochs):
        step = 0
        for emb, yb, idxb in dl:
            step += 1
            emb = emb.to(device).float()
            yb = yb.to(device)
            idxb = idxb.to(device)

            # sample
            with torch.no_grad():
                out = model.generate(
                    inputs={"frozen_embeddings": emb},
                    generation_kwargs={
                        "max_new_tokens": args.gen_max_new_tokens,
                        "do_sample": True,
                        "top_p": args.top_p,
                        "top_k": args.top_k,
                        "temperature": args.temperature,
                        "repetition_penalty": args.train_repetition_penalty,
                    }
                )
                texts = model.tok.batch_decode(out, skip_special_tokens=True)
                gen_ids = out.to(device)

            B = len(texts)
            with torch.no_grad():
                gtr_emb = gtr.encode(texts)
                logits = clf(gtr_emb)
                logp = F.log_softmax(logits, dim=-1)

                # cls score（目标类对数概率）
                r_cls = logp[torch.arange(yb.size(0), device=device), yb]

                # margin: 目标类 vs 最强负类
                other = logp.clone()
                other[torch.arange(other.size(0), device=device), yb] = -1e9
                other_max = other.max(dim=-1).values
                r_margin = r_cls - other_max

                # 置信度（负熵）
                probs = F.softmax(logits, dim=-1)
                ent = -(probs * torch.log(probs + 1e-8)).sum(-1)
                r_conf = -ent

                # 长度
                lens = (gen_ids != tok_train.pad_token_id).sum(dim=-1).float()
                r_len = -((lens - args.len_target) ** 2) / (2 * (args.len_sigma ** 2))

                # base reward
                reward = (
                    args.w_cls * r_cls
                    + args.w_cls_margin * r_margin
                    + args.w_len * r_len
                    + 0.05 * r_conf
                )

                # 语言奖励（默认关闭；若开启，对中文不惩罚）
                if args.w_lang > 0:
                    lang_bonus = []
                    for t in texts:
                        cjk_r = is_cjk_ratio(t)
                        if cjk_r > 0.3:
                            lang_bonus.append(0.0)
                        else:
                            ascii_cnt = sum(1 for ch in t if 32 <= ord(ch) < 127)
                            ratio = ascii_cnt / max(1, len(t))
                            lang_bonus.append(ratio - 0.9)
                    lang_bonus = torch.tensor(lang_bonus, device=device, dtype=reward.dtype)
                    reward = reward + args.w_lang * lang_bonus

                # 样本级语义奖励
                if inst_targets is not None and args.w_inst > 0:
                    inst_emb = inst_targets[idxb].to(device)
                    r_inst = F.cosine_similarity(gtr_emb, inst_emb, dim=-1)
                    reward = reward + args.w_inst * r_inst

                # 批内整句塌缩
                lower = [t.strip().lower() for t in texts]
                for i_, t_ in enumerate(lower):
                    if lower.count(t_) > B * args.collapse_ratio:
                        reward[i_] = reward[i_] - args.collapse_penalty

                # 批内 n-gram 塌缩（前4词一样扣）
                if args.w_ngram > 0:
                    heads = [" ".join(t.split()[:4]).lower() for t in texts]
                    for i_, h in enumerate(heads):
                        if heads.count(h) > 1:
                            reward[i_] = reward[i_] - args.w_ngram

                # 跨 batch 塌缩（全文/前4词）
                for i_, t_ in enumerate(lower):
                    if t_ in recent_texts:
                        reward[i_] = reward[i_] - args.collapse_penalty * 0.5
                heads = [" ".join(t.split()[:4]).lower() for t in texts]
                for i_, h in enumerate(heads):
                    if h in recent_heads:
                        reward[i_] = reward[i_] - args.collapse_penalty * 0.3
                recent_texts.extend(lower)
                recent_heads.extend(heads)

                # 多样性（embedding 角度）
                if args.w_div > 0:
                    sim_mat = F.cosine_similarity(
                        gtr_emb.unsqueeze(1),
                        gtr_emb.unsqueeze(0),
                        dim=-1
                    )
                    sim_mat = sim_mat - torch.eye(B, device=device) * 10
                    max_sim, _ = sim_mat.max(dim=-1)
                    r_div = -max_sim
                    reward = reward + args.w_div * r_div

                if reward_ma is None:
                    reward_ma = reward.mean().item()
                adv = reward - reward_ma
                reward_ma = beta * reward_ma + (1 - beta) * reward.mean().item()

            # === REINFORCE（token 平均 + 归一化 advantage）===
            out2 = model(frozen_embeddings=emb, labels=None, decoder_input_ids=gen_ids[:, :-1])
            logits2 = out2.logits
            tgt = gen_ids[:, 1:].contiguous()
            log_probs2 = F.log_softmax(logits2, dim=-1)
            tgt_logp = log_probs2.gather(dim=-1, index=tgt.unsqueeze(-1)).squeeze(-1)
            mask = (tgt != tok_train.pad_token_id).float()
            tgt_logp_mean = (tgt_logp * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1)

            adv_n = adv.detach()
            if args.adv_clip:
                adv_n = adv_n.clamp(-args.adv_clip, args.adv_clip)
            adv_n = (adv_n - adv_n.mean()) / (adv_n.std() + 1e-6)

            rl_loss = -(adv_n * tgt_logp_mean).mean()

            optim.zero_grad()
            rl_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

            if step % 20 == 0:
                print(
                    f"[RL] ep {ep + 1}/{args.epochs} step {step} | rl={rl_loss.item():.4f} | "
                    f"R={reward.mean().item():.3f} | cls={r_cls.mean().item():.3f}"
                )

        # save epoch
        save_dir = os.path.join(args.out_dir, f"ep{ep + 1}")
        ensure_dir(save_dir)
        model.tok.save_pretrained(save_dir)
        model.t5.save_pretrained(os.path.join(save_dir, "t5"))
        torch.save(model.proj.state_dict(), os.path.join(save_dir, "proj.pt"))
        meta = {
            "num_repeat_tokens": args.num_repeat_tokens,
            "pad_token_id": int(model.tok.pad_token_id),
            "eos_token_id": int(model.tok.eos_token_id or model.tok.pad_token_id),
            "decode_prefix": args.decode_prefix,
        }
        with open(os.path.join(save_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        print(f"[✓] saved {save_dir}")


@torch.no_grad()
def inference(args):
    """
    inference = K candidates per z + classifier rerank + instance-semantic rerank
    保证输出 CSV 第 i 行与 Z[i] / labels_csv 第 i 行一致

    新增：支持 --prompt_json 和 --prompt_n_supervised
    对前 N 条样本，用 prompt.json 中解析出的 identity + {attribute} 作为 decoder 前缀，
    使生成文本可以作为对应 prompt 的回答。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tok = AutoTokenizer.from_pretrained(args.load_from, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token or tok.unk_token
    with open(os.path.join(args.load_from, "meta.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)
    num_repeat = int(meta["num_repeat_tokens"])
    pad_id = int(meta["pad_token_id"])
    eos_id = int(meta["eos_token_id"])
    decode_prefix = meta.get("decode_prefix", "A sentence is: ")

    proj_sd = torch.load(os.path.join(args.load_from, "proj.pt"), map_location="cpu")
    z_dim = proj_sd["weight"].shape[1]

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
    model.proj.load_state_dict(proj_sd)
    model.proj.to(device)

    # ---- 分类器用于重排 ----
    clf, label_map, gtr_path_from_ckpt, _ = load_tsc(args.clf_ckpt, device)
    gtr_path = args.gtr_path if args.gtr_path else (gtr_path_from_ckpt or DEFAULT_GTR_PATH)
    gtr = GTRWrapper(gtr_path=gtr_path, device=device)

    # debug shape check
    try:
        clf_in_dim = next(clf.parameters()).shape[1]
        probe = gtr.encode(["hello"])
        print(f"[dbg] clf in_dim={clf_in_dim}, gtr output dim={probe.shape[-1]}")
        assert clf_in_dim == probe.shape[-1], "分类器输入维度与嵌入器输出维度不一致，请检查 ckpt 与 gtr_path。"
    except Exception as e:
        print("[warn] dim check failed:", e)

    # ---- labels & Z ----
    df_lab = pd.read_csv(args.labels_csv)
    Z = torch.load(args.z_test)
    if not torch.is_tensor(Z):
        Z = torch.tensor(Z)
    Z = Z.float()
    assert len(df_lab) == Z.shape[0], f"labels_csv 行数({len(df_lab)})需与 z_test 行数({Z.shape[0]})一致"

    y_raw = df_lab[args.label_col].astype(str).tolist()
    y_true = [label_map[y] for y in y_raw]

    # ---- NOTE (modified): 不再使用 CSV 的实例文本做 sem_score（避免把 z_shared 拉回来）
    # sem_score 将改为：cos( gtr.encode(candidate_text), mean_pool( model._enc_out_from_z(z).last_hidden_state ) )
    # 因此此处不再构造 inst_targets。

    # ---- 可选：前 N 条使用 prompt.json 约束 ----
    prompt_meta = None
    if args.prompt_json and args.prompt_n_supervised > 0:
        with open(args.prompt_json, "r", encoding="utf-8") as f:
            raw_prompts = json.load(f)

        if not isinstance(raw_prompts, list):
            raise ValueError("prompt_json 必须是 list。")

        def _extract_identity_attr(ptext: str):
            """
            从 prompt 里抽出前两个 {...} 内容：
            第一个当作 identity，第二个当作 attribute。
            后面那个空的 '{}'（提示标记用）会被忽略。
            """
            spans = []
            cur = []
            inside = False
            for ch in ptext:
                if ch == "{":
                    inside = True
                    cur = []
                elif ch == "}":
                    if inside:
                        spans.append("".join(cur).strip())
                        inside = False
                else:
                    if inside:
                        cur.append(ch)
            identity = spans[0] if len(spans) >= 1 else None
            attr = spans[1] if len(spans) >= 2 else None
            return identity, attr

        prompt_meta = []
        for item in raw_prompts:
            if isinstance(item, dict) and "prompt" in item:
                ptext = item["prompt"]
            else:
                ptext = str(item)
            identity, attr = _extract_identity_attr(ptext)
            prompt_meta.append({
                "prompt": ptext,
                "identity": identity,
                "attr": attr,
            })

        if len(prompt_meta) < args.prompt_n_supervised:
            print(f"[warn] prompt_json 只有 {len(prompt_meta)} 条，"
                  f"将只对前 {len(prompt_meta)} 条样本使用 prompt 约束。")
        else:
            print(f"[info] 将对前 {args.prompt_n_supervised} 条样本使用 prompt 约束。")

    bs = args.batch_size
    gens = []
    K = args.infer_n_candidates

    for i in range(0, Z.shape[0], bs):
        emb = Z[i:i + bs].to(device).float()
        this_bs = emb.size(0)
        yb = torch.tensor(y_true[i:i+this_bs], device=device, dtype=torch.long)
        # new: semantic target from z itself (encoder-space, 512-d)
        inst_b = model._enc_out_from_z(emb).last_hidden_state.mean(dim=1)  # [B, 512]

        # ---- 为本 batch 构造 decoder 前缀 ----
        # 前 prompt_n_supervised 条且有 prompt_meta 时，用 identity + {attr} 作为开头；
        # 其它样本则用默认 decode_prefix。
        if prompt_meta is not None and args.prompt_n_supervised > 0:
            prefix_texts = []
            for j in range(this_bs):
                gidx = i + j  # 全局行号
                if gidx < args.prompt_n_supervised and gidx < len(prompt_meta):
                    meta_j = prompt_meta[gidx]
                    ident = meta_j.get("identity")
                    attr = meta_j.get("attr")
                    if ident and attr:
                        # 满足：包含 identity，且只对 attribute 用一次 {}
                        # 例如："Willow who is {busy}"
                        prefix = f"{ident} who is {{{attr}}}"
                    elif attr:
                        prefix = f"{{{attr}}}"
                    else:
                        prefix = decode_prefix
                else:
                    prefix = decode_prefix
                prefix_texts.append(prefix)

            dec_inp = tok(
                prefix_texts,
                padding=True,
                truncation=True,
                max_length=64,
                return_tensors="pt"
            ).input_ids.to(device)
        else:
            dec_inp = None

        best_texts = [""] * this_bs
        best_scores = torch.full((this_bs,), -1e9, device=device, dtype=torch.float32)

        for _ in range(K):
            out = model.generate(
                inputs={"frozen_embeddings": emb},
                generation_kwargs={
                    "max_new_tokens": args.gen_max_new_tokens,
                    "do_sample": True,
                    "top_p": args.infer_top_p,
                    "top_k": args.infer_top_k,
                    "temperature": args.infer_temperature,
                    "repetition_penalty": 1.15,
                },
                decoder_input_ids=dec_inp,
            )
            texts = tok.batch_decode(out, skip_special_tokens=True)
            g_emb = gtr.encode(texts)
            logits = clf(g_emb)
            logp = F.log_softmax(logits, dim=-1)
            cls_score = logp[torch.arange(this_bs, device=device), yb]

            sem_score = F.cosine_similarity(g_emb, inst_b, dim=-1)

            score = (
                args.infer_rerank_alpha * cls_score
                + args.infer_rerank_beta * sem_score
            )

            for j in range(this_bs):
                if score[j] > best_scores[j]:
                    best_scores[j] = score[j]
                    best_texts[j] = texts[j]

        gens.extend(best_texts)

    ensure_dir(os.path.dirname(args.infer_out) if os.path.dirname(args.infer_out) else ".")
    out_df = pd.DataFrame({
        "gen_text": gens,
        "true_label": y_raw[:len(gens)],
        "row_index": np.arange(len(gens)),
    })
    for col in ["id", "text", "sentence", "content", "prompt", "input"]:
        if col in df_lab.columns:
            out_df[col] = df_lab[col].astype(str).iloc[:len(gens)].values
    out_df.to_csv(args.infer_out, index=False)
    print("[✓] wrote", len(gens), "rows to", args.infer_out)


@torch.no_grad()
def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clf, label_map, gtr_path_from_ckpt, _ = load_tsc(args.clf_ckpt, device)
    gtr_path = args.gtr_path if args.gtr_path else (gtr_path_from_ckpt or DEFAULT_GTR_PATH)
    gtr = GTRWrapper(gtr_path=gtr_path, device=device)

    df_gen = pd.read_csv(args.gen_csv)
    texts = (df_gen["gen_text"] if "gen_text" in df_gen.columns else df_gen.iloc[:, 0]).astype(str).tolist()
    gemb = gtr.encode(texts)
    logits = clf(gemb)
    pred = logits.argmax(dim=-1).cpu().numpy()

    df_lab = pd.read_csv(args.labels_csv)
    y_raw = df_lab[args.label_col].astype(str).tolist()[:len(pred)]
    y_true = np.array([label_map[y] for y in y_raw])
    acc = (pred == y_true).mean()
    print(f"[Eval] Accuracy on generated texts: {acc:.4f}")


def parse_args():
    p = argparse.ArgumentParser()
    # data
    p.add_argument("--z_train", type=str)
    p.add_argument("--labels_csv", type=str)
    p.add_argument("--label_col", type=str, default="label")
    # classifier
    p.add_argument("--clf_ckpt", type=str)
    # model
    p.add_argument("--model_name", type=str, default=DEFAULT_MODEL_PATH)
    p.add_argument("--decode_prefix", type=str, default="A sentence is: ")
    p.add_argument("--num_repeat_tokens", type=int, default=24)
    # RL
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--rl_lr", type=float, default=1e-5)
    p.add_argument("--gen_max_new_tokens", type=int, default=48)
    p.add_argument("--top_p", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=40)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--w_cls", type=float, default=1.6)
    p.add_argument("--w_cls_margin", type=float, default=0.6)
    p.add_argument("--w_len", type=float, default=0.02)
    p.add_argument("--len_target", type=float, default=18.0)
    p.add_argument("--len_sigma", type=float, default=8.0)
    p.add_argument("--w_lang", type=float, default=0.0)
    p.add_argument("--w_inst", type=float, default=0.0)
    p.add_argument("--adv_clip", type=float, default=1.0)
    p.add_argument("--reward_ma_beta", type=float, default=0.9)
    p.add_argument("--train_repetition_penalty", type=float, default=1.15)
    p.add_argument("--collapse_ratio", type=float, default=0.05)
    p.add_argument("--collapse_penalty", type=float, default=1.0)
    p.add_argument("--w_div", type=float, default=0.0)
    p.add_argument("--w_ngram", type=float, default=0.0)
    p.add_argument("--recent_capacity", type=int, default=128)
    # proj
    p.add_argument("--use_proj_semantic", action="store_true")
    p.add_argument("--proj_lr", type=float, default=5e-4)
    # gtr
    p.add_argument("--gtr_path", type=str, default=None)
    # dirs
    p.add_argument("--out_dir", type=str, default="saves/zspec_label_only")
    # inference
    p.add_argument("--load_from", type=str)
    p.add_argument("--z_test", type=str)
    p.add_argument("--infer_out", type=str, default="saves/test_generations.csv")
    p.add_argument("--infer_do_sample", action="store_true")
    p.add_argument("--infer_top_p", type=float, default=0.7)
    p.add_argument("--infer_top_k", type=int, default=50)
    p.add_argument("--infer_temperature", type=float, default=0.6)
    p.add_argument("--infer_n_candidates", type=int, default=10)
    p.add_argument("--infer_rerank_alpha", type=float, default=1.0)
    p.add_argument("--infer_rerank_beta", type=float, default=0.4)
    # prompt-constrained inference
    p.add_argument(
        "--prompt_json",
        type=str,
        default=None,
        help="JSON 列表文件，元素带 'prompt' 字段；用于约束前 N 条生成文本"
    )
    p.add_argument(
        "--prompt_n_supervised",
        type=int,
        default=0,
        help="前多少条样本要同时作为 prompt.json 的回答（行对齐）"
    )
    # evaluate
    p.add_argument("--evaluate", action="store_true")
    p.add_argument("--gen_csv", type=str)
    # stage 1
    p.add_argument("--pretrain_z2enc", action="store_true")
    p.add_argument("--pretrain_epochs", type=int, default=10)
    # stage 1b
    p.add_argument("--pretrain_mle", action="store_true")
    p.add_argument("--mle_epochs", type=int, default=2)
    p.add_argument("--mle_lr", type=float, default=5e-5)
    p.add_argument("--mle_batch_size", type=int, default=16)
    p.add_argument("--train_t5_mle", action="store_true")
    # misc
    p.add_argument("--train_t5", action="store_true", help="really finetune T5 (not recommended at first)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    if args.pretrain_z2enc:
        pretrain_z2enc(args)
        return
    if args.pretrain_mle:
        pretrain_mle(args)
        return
    if args.z_train and args.labels_csv and args.clf_ckpt and not args.load_from and not args.evaluate:
        train_label_only(args)
        return
    if args.load_from and args.z_test and not args.evaluate:
        inference(args)
        return
    if args.evaluate and args.gen_csv and args.labels_csv and args.clf_ckpt:
        evaluate(args)
        return
    print("Run one of: --pretrain_z2enc | --pretrain_mle | RL train | inference | --evaluate")


if __name__ == "__main__":
    main()
