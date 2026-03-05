"""
Unified training script for PolyGEC.

Runs experiments across:
  - 2 architectures : RNN-Attention, LSTM-Attention
  - 3 tokenization  : chopped→common, common→common, chopped→chopped

Total = 6 experiments (can be run selectively via --model and --tok_config).

Usage examples:
  # Train all 6 experiments sequentially
  python train.py --run_all

  # Train one specific experiment
  python train.py --model lstm --tok_config chopped_common

  # Quick smoke test (small data, few epochs)
  python train.py --model rnn --tok_config common_common --max_samples 1000 --epochs 2
"""

import os
import sys
import time
import argparse
import json
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

# ── project imports ────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from tok.bpe_tokenizer import ChoppedTokenizer, CommonTokenizer, build_tokenizers
from data.dataset import build_dataloader, build_train_val_dataloaders
from models.rnn_attention import RNNSeq2Seq
from models.lstm_attention import LSTMSeq2Seq
from config import (
    TRAIN_CSV, TEST_CSV, TOK_DIR, CKPT_DIR,
    TOKENIZATION_CONFIGS, HYPERPARAMS as DEFAULT_HYPERPARAMS, PAD_IDX
)


# ══════════════════════════════════════════════════════════════════════════════
#  Tokenizer loading helpers
# ══════════════════════════════════════════════════════════════════════════════
def load_or_build_tokenizers(bpe_vocab: int, word_vocab: int,
                             train_csv: str = None, lang: str = "en"):
    """
    Return (chopped_tok, common_tok).
    Tokenizers are saved/loaded from tok/saved/<lang>/ so each language
    gets its own vocabulary and never overwrites another language's files.
    """
    tok_dir      = os.path.join(TOK_DIR, lang)
    chopped_path = os.path.join(tok_dir, "chopped_bpe.json")
    common_path  = os.path.join(tok_dir, "common_word.json")

    chopped = ChoppedTokenizer(vocab_size=bpe_vocab)
    common  = CommonTokenizer(vocab_size=word_vocab)

    if os.path.exists(chopped_path) and os.path.exists(common_path):
        print(f"Loading tokenizers from {tok_dir} …")
        chopped.load(chopped_path)
        common.load(common_path)
    else:
        csv_to_use = train_csv or TRAIN_CSV
        print(f"Tokenizers not found — training from {csv_to_use} …")
        chopped, common = build_tokenizers(
            csv_to_use, tok_dir, bpe_vocab, word_vocab
        )
    return chopped, common, tok_dir


def get_tokenizers_for_config(tok_config: str, chopped, common):
    """Return (src_tok, tgt_tok) for the given config name."""
    src_type, tgt_type = TOKENIZATION_CONFIGS[tok_config]
    src_tok = chopped if src_type == "chopped" else common
    tgt_tok = chopped if tgt_type == "chopped" else common
    return src_tok, tgt_tok


# ══════════════════════════════════════════════════════════════════════════════
#  Model builder
# ══════════════════════════════════════════════════════════════════════════════
def build_model(model_name: str, src_vocab: int, tgt_vocab: int, hp: dict, device):
    common_kwargs = dict(
        src_vocab_size=src_vocab,
        tgt_vocab_size=tgt_vocab,
        embed_dim=hp["embed_dim"],
        hidden_dim=hp["hidden_dim"],
        enc_layers=hp["enc_layers"],
        dropout=hp["dropout"],
    )
    if model_name == "rnn":
        model = RNNSeq2Seq(**common_kwargs)
    elif model_name == "lstm":
        model = LSTMSeq2Seq(**common_kwargs)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[{model_name.upper()}] Parameters: {total_params:,}")
    return model


# ══════════════════════════════════════════════════════════════════════════════
#  Train one epoch
# ══════════════════════════════════════════════════════════════════════════════
def train_epoch(model, loader, optimizer, criterion, device, clip, tf_ratio):
    model.train()
    total_loss = 0.0

    for src, tgt in loader:
        src, tgt = src.to(device), tgt.to(device)
        optimizer.zero_grad()

        output = model(src, tgt, teacher_forcing_ratio=tf_ratio)
        # output: (batch, tgt_len-1, vocab)
        # tgt   : (batch, tgt_len)  → target is tgt[:, 1:]
        output_flat = output.reshape(-1, output.size(-1))
        target_flat = tgt[:, 1:].reshape(-1)

        loss = criterion(output_flat, target_flat)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


# ══════════════════════════════════════════════════════════════════════════════
#  Evaluate one epoch
# ══════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    for src, tgt in loader:
        src, tgt = src.to(device), tgt.to(device)
        output = model(src, tgt, teacher_forcing_ratio=0.0)
        output_flat = output.reshape(-1, output.size(-1))
        target_flat = tgt[:, 1:].reshape(-1)
        loss = criterion(output_flat, target_flat)
        total_loss += loss.item()
    return total_loss / len(loader)


# ══════════════════════════════════════════════════════════════════════════════
#  Main training routine for one experiment
# ══════════════════════════════════════════════════════════════════════════════
def run_experiment(
    model_name: str,
    tok_config: str,
    hp: dict,
    device,
    max_samples: int = None,
    train_csv: str = None,
    test_csv: str = None,
    lang: str = "en",
):
    train_csv = train_csv or TRAIN_CSV
    exp_name  = f"{lang}_{model_name}_{tok_config}"

    print(f"\n{'='*60}")
    print(f"  Experiment : {exp_name}")
    print(f"  Language   : {lang}")
    print(f"  Train CSV  : {train_csv}")
    print(f"  Val source : {'10% split from train' if test_csv is None else test_csv}")
    print(f"  Device     : {device}")
    print(f"{'='*60}")

    # ── Tokenizers ────────────────────────────────────────────────────────────
    chopped, common, tok_dir = load_or_build_tokenizers(
        hp["bpe_vocab"], hp["word_vocab"], train_csv=train_csv, lang=lang
    )
    src_tok, tgt_tok = get_tokenizers_for_config(tok_config, chopped, common)

    # ── Data loaders ──────────────────────────────────────────────────────────
    # If an explicit test_csv is given, use it as the validation set.
    # Otherwise split 10% off train_csv so the test CSV stays unseen.
    if test_csv is not None:
        train_loader = build_dataloader(
            train_csv, src_tok, tgt_tok,
            batch_size=hp["batch_size"], shuffle=True,
            max_len=hp["max_len"], max_samples=max_samples,
        )
        val_loader = build_dataloader(
            test_csv, src_tok, tgt_tok,
            batch_size=hp["batch_size"], shuffle=False,
            max_len=hp["max_len"], max_samples=max_samples,
        )
    else:
        train_loader, val_loader = build_train_val_dataloaders(
            train_csv, src_tok, tgt_tok,
            val_split=0.1,
            batch_size=hp["batch_size"],
            max_len=hp["max_len"], max_samples=max_samples,
        )

    # ── Model ─────────────────────────────────────────────────────────────────
    src_vocab = src_tok.vocab_size_actual()
    tgt_vocab = tgt_tok.vocab_size_actual()
    model = build_model(model_name, src_vocab, tgt_vocab, hp, device)

    # ── Optimizer / Loss ──────────────────────────────────────────────────────
    optimizer = Adam(model.parameters(), lr=hp["lr"])
    scheduler = ReduceLROnPlateau(optimizer, patience=2, factor=0.5)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": []}

    os.makedirs(CKPT_DIR, exist_ok=True)
    ckpt_path = os.path.join(CKPT_DIR, f"{exp_name}_best.pt")

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(1, hp["epochs"] + 1):
        t0 = time.time()
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device,
            hp["clip"], hp["tf_ratio"]
        )
        val_loss = eval_epoch(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        elapsed = time.time() - t0

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        print(
            f"  Epoch {epoch:>3}/{hp['epochs']} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Time: {elapsed:.1f}s"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "hp": hp,
                    "model_name": model_name,
                    "tok_config": tok_config,
                    "lang": lang,
                    "tok_dir": tok_dir,
                    "train_csv": train_csv,
                    "val_source": test_csv if test_csv else "10% split from train",
                },
                ckpt_path,
            )
            print(f"  ✓ Best model saved → {ckpt_path}")

    # ── Save history ──────────────────────────────────────────────────────────
    hist_path = os.path.join(CKPT_DIR, f"{exp_name}_history.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n  Best Val Loss : {best_val_loss:.4f}")
    print(f"  History saved : {hist_path}")
    return best_val_loss


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(description="PolyGEC Training Script")
    p.add_argument("--model",       type=str, default="lstm",
                   choices=["rnn", "lstm"],
                   help="Model architecture")
    p.add_argument("--tok_config",  type=str, default="chopped_common",
                   choices=list(TOKENIZATION_CONFIGS.keys()),
                   help="Tokenization configuration")
    p.add_argument("--run_all",     action="store_true",
                   help="Run all 6 experiments sequentially")
    p.add_argument("--epochs",      type=int,   default=DEFAULT_HYPERPARAMS["epochs"])
    p.add_argument("--batch_size",  type=int,   default=DEFAULT_HYPERPARAMS["batch_size"])
    p.add_argument("--hidden_dim",  type=int,   default=DEFAULT_HYPERPARAMS["hidden_dim"])
    p.add_argument("--embed_dim",   type=int,   default=DEFAULT_HYPERPARAMS["embed_dim"])
    p.add_argument("--lr",          type=float, default=DEFAULT_HYPERPARAMS["lr"])
    p.add_argument("--dropout",     type=float, default=DEFAULT_HYPERPARAMS["dropout"])
    p.add_argument("--max_samples", type=int,   default=None,
                   help="Limit dataset size (for debugging)")
    p.add_argument("--device",      type=str,   default="auto",
                   help="cuda / cpu / auto")
    p.add_argument("--train_csv",   type=str,   default=None,
                   help="Path to training CSV (overrides config TRAIN_CSV)")
    p.add_argument("--test_csv",    type=str,   default=None,
                   help="Path to test/val CSV (overrides config TEST_CSV)")
    p.add_argument("--lang",        type=str,   default="en",
                   help="Language tag used in checkpoint naming (e.g. en, ta, hi)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # Hyperparams (override defaults with CLI args)
    hp = dict(DEFAULT_HYPERPARAMS)
    hp.update({
        "epochs":     args.epochs,
        "batch_size": args.batch_size,
        "hidden_dim": args.hidden_dim,
        "embed_dim":  args.embed_dim,
        "lr":         args.lr,
        "dropout":    args.dropout,
    })

    if args.run_all:
        results = {}
        for model_name in ["rnn", "lstm"]:
            for tok_config in TOKENIZATION_CONFIGS:
                best = run_experiment(
                    model_name, tok_config, hp, device, args.max_samples,
                    train_csv=args.train_csv,
                    test_csv=args.test_csv,
                    lang=args.lang,
                )
                results[f"{args.lang}_{model_name}_{tok_config}"] = best

        print("\n" + "="*60)
        print("  SUMMARY OF ALL EXPERIMENTS")
        print("="*60)
        for name, loss in sorted(results.items(), key=lambda x: x[1]):
            print(f"  {name:<35} Val Loss: {loss:.4f}")
    else:
        run_experiment(
            args.model, args.tok_config, hp, device, args.max_samples,
            train_csv=args.train_csv,
            test_csv=args.test_csv,
            lang=args.lang,
        )
