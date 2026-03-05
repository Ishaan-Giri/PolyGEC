"""
Evaluation script for PolyGEC.

Metrics implemented:
  1. GLEU  — sentence-level BLEU variant optimised for GEC (via sacrebleu)
  2. Corpus BLEU — overall translation quality
  3. Token Accuracy — exact token match rate

Usage:
    # Evaluate a single checkpoint
    python evaluation/evaluate.py \
        --checkpoint checkpoints/ta_lstm_common_common_best.pt \
        --test_csv   data/tamil_test.csv

    # Evaluate ALL checkpoints for a language at once
    python evaluation/evaluate.py \
        --run_all \
        --lang       ta \
        --test_csv   data/tamil_test.csv \
        --ckpt_dir   checkpoints \
        --save_json  results_ta.json

    # Evaluate all, specific model only
    python evaluation/evaluate.py \
        --run_all --lang ta --model lstm --test_csv data/tamil_test.csv

    # Evaluate all, specific tok_config only
    python evaluation/evaluate.py \
        --run_all --lang ta --tok_config common_common --test_csv data/tamil_test.csv
"""

import os
import sys
import csv
import json
import argparse
import torch
import sacrebleu

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tok.bpe_tokenizer import ChoppedTokenizer, CommonTokenizer
from models.rnn_attention  import RNNSeq2Seq
from models.lstm_attention import LSTMSeq2Seq

PAD_IDX = 0
EOS_IDX = 2


# ══════════════════════════════════════════════════════════════════════════════
#  Load checkpoint
# ══════════════════════════════════════════════════════════════════════════════
def load_checkpoint(ckpt_path: str, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    return ckpt


def rebuild_tokenizers(tok_config: str, tok_dir: str):
    """Re-load tokenizers from the language-specific saved directory."""

    # Safe split: configs are  chopped_common | common_common | chopped_chopped
    parts    = tok_config.split("_")
    src_type = parts[0]
    tgt_type = parts[-1]   # last part is always the tgt type

    def _load(tok_type):
        if tok_type == "chopped":
            t = ChoppedTokenizer()
            t.load(os.path.join(tok_dir, "chopped_bpe.json"))
        else:
            t = CommonTokenizer()
            t.load(os.path.join(tok_dir, "common_word.json"))
        return t

    return _load(src_type), _load(tgt_type)


def rebuild_model(model_name: str, src_vocab: int, tgt_vocab: int, hp: dict, device):
    common_kwargs = dict(
        src_vocab_size=src_vocab,
        tgt_vocab_size=tgt_vocab,
        embed_dim=hp["embed_dim"],
        hidden_dim=hp["hidden_dim"],
        enc_layers=hp["enc_layers"],
        dropout=0.0,  # no dropout at inference
    )
    if model_name == "rnn":
        return RNNSeq2Seq(**common_kwargs).to(device)
    else:
        return LSTMSeq2Seq(**common_kwargs).to(device)


# ══════════════════════════════════════════════════════════════════════════════
#  Decode helpers
# ══════════════════════════════════════════════════════════════════════════════
def ids_to_sentence(ids, tokenizer, skip_special=True) -> str:
    # Strip padding and EOS
    clean = []
    for i in ids:
        if i == EOS_IDX:
            break
        clean.append(int(i))
    return tokenizer.decode(clean, skip_special=skip_special).strip()


# ══════════════════════════════════════════════════════════════════════════════
#  GLEU computation
# ══════════════════════════════════════════════════════════════════════════════
def compute_gleu(hypotheses, references):
    """
    Sentence-level GLEU averaged over all sentences.
    Uses sacrebleu BLEU at sentence level as a proxy.
    """
    scores = []
    for hyp, ref in zip(hypotheses, references):
        score = sacrebleu.sentence_bleu(hyp, [ref]).score
        scores.append(score)
    return sum(scores) / len(scores) if scores else 0.0


def compute_corpus_bleu(hypotheses, references):
    result = sacrebleu.corpus_bleu(hypotheses, [references])
    return result.score


def compute_token_accuracy(hypotheses, references):
    total, correct = 0, 0
    for hyp, ref in zip(hypotheses, references):
        h_tok = hyp.split()
        r_tok = ref.split()
        length = min(len(h_tok), len(r_tok))
        correct += sum(h == r for h, r in zip(h_tok[:length], r_tok[:length]))
        total   += max(len(h_tok), len(r_tok))
    return correct / total * 100 if total else 0.0


# ══════════════════════════════════════════════════════════════════════════════
#  Main evaluation
# ══════════════════════════════════════════════════════════════════════════════
def evaluate(ckpt_path: str, test_csv: str, batch_size: int = 64, max_samples: int = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    print(f"Loading checkpoint: {ckpt_path}")

    ckpt       = load_checkpoint(ckpt_path, device)
    model_name = ckpt["model_name"]
    tok_config = ckpt["tok_config"]
    hp         = ckpt["hp"]
    lang       = ckpt.get("lang", "en")

    # tok_dir is stored in the checkpoint; fall back to legacy global path
    tok_dir = ckpt.get(
        "tok_dir",
        os.path.join(os.path.dirname(__file__), "..", "tok", "saved")
    )

    print(f"Model      : {model_name}")
    print(f"Tok Config : {tok_config}")
    print(f"Language   : {lang}")
    print(f"Tok Dir    : {tok_dir}")

    src_tok, tgt_tok = rebuild_tokenizers(tok_config, tok_dir)
    src_vocab = src_tok.vocab_size_actual()
    tgt_vocab = tgt_tok.vocab_size_actual()

    model = rebuild_model(model_name, src_vocab, tgt_vocab, hp, device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # ── Read test data ─────────────────────────────────────────────────────────
    sources, targets = [], []
    with open(test_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        src_col = "source" if "source" in fieldnames else "src"
        tgt_col = "target" if "target" in fieldnames else "trg"
        for i, row in enumerate(reader):
            if max_samples and i >= max_samples:
                break
            src = row.get(src_col, "").strip()
            tgt = row.get(tgt_col, "").strip()
            if src and tgt:
                sources.append(src)
                targets.append(tgt)

    print(f"Test samples: {len(sources):,}")

    # ── Generate predictions in batches ────────────────────────────────────────
    hypotheses = []
    for i in range(0, len(sources), batch_size):
        batch_src = sources[i: i + batch_size]
        src_ids   = [
            torch.tensor(src_tok.encode(s)[:hp["max_len"]], dtype=torch.long)
            for s in batch_src
        ]
        # Pad
        max_l  = max(t.size(0) for t in src_ids)
        padded = torch.zeros(len(src_ids), max_l, dtype=torch.long)
        for j, t in enumerate(src_ids):
            padded[j, :t.size(0)] = t
        padded = padded.to(device)

        with torch.no_grad():
            gen = model.generate(padded, max_len=hp["max_len"])

        for ids in gen:
            hyp = ids_to_sentence(ids.cpu().tolist(), tgt_tok)
            hypotheses.append(hyp)

        if (i // batch_size) % 20 == 0:
            print(f"  Generated {min(i+batch_size, len(sources)):,}/{len(sources):,} …")

    # ── Metrics ────────────────────────────────────────────────────────────────
    gleu       = compute_gleu(hypotheses, targets)
    bleu       = compute_corpus_bleu(hypotheses, targets)
    tok_acc    = compute_token_accuracy(hypotheses, targets)

    print(f"\n{'='*50}")
    print(f"  RESULTS  [{model_name.upper()} | {tok_config}]")
    print(f"{'='*50}")
    print(f"  GLEU (avg sentence-level) : {gleu:.2f}")
    print(f"  Corpus BLEU               : {bleu:.2f}")
    print(f"  Token Accuracy            : {tok_acc:.2f}%")
    print(f"{'='*50}")

    # ── Show a few examples ────────────────────────────────────────────────────
    print("\n  Sample Predictions:")
    print(f"  {'─'*70}")
    for src, hyp, ref in zip(sources[:5], hypotheses[:5], targets[:5]):
        print(f"  SRC : {src}")
        print(f"  HYP : {hyp}")
        print(f"  REF : {ref}")
        print(f"  {'─'*70}")

    return {"gleu": gleu, "bleu": bleu, "token_acc": tok_acc}


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="PolyGEC Evaluation Script")
    p.add_argument("--checkpoint",  default=None,
                   help="Path to a single .pt checkpoint (single-run mode)")
    p.add_argument("--test_csv",    required=True,
                   help="Path to test CSV file")
    p.add_argument("--run_all",     action="store_true",
                   help="Evaluate all matching checkpoints in --ckpt_dir")
    p.add_argument("--lang",        type=str, default=None,
                   help="Language tag to filter checkpoints (e.g. ta, en, hi)")
    p.add_argument("--model",       type=str, default=None, choices=["rnn", "lstm"],
                   help="Filter --run_all to one model architecture")
    p.add_argument("--tok_config",  type=str, default=None,
                   choices=["chopped_common", "common_common", "chopped_chopped"],
                   help="Filter --run_all to one tokenization config")
    p.add_argument("--ckpt_dir",    type=str, default="checkpoints",
                   help="Directory containing checkpoints (used with --run_all)")
    p.add_argument("--batch_size",  type=int, default=64)
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--save_json",   type=str, default=None,
                   help="Save all results to this JSON file (e.g. results_ta.json)")
    args = p.parse_args()

    all_results = {}

    if args.run_all:
        # ── Collect all matching checkpoint files ──────────────────────────────
        ckpt_files = sorted([
            f for f in os.listdir(args.ckpt_dir)
            if f.endswith("_best.pt")
        ])

        # Apply filters
        if args.lang:
            ckpt_files = [f for f in ckpt_files if f.startswith(f"{args.lang}_")]
        if args.model:
            ckpt_files = [f for f in ckpt_files if f"_{args.model}_" in f]
        if args.tok_config:
            ckpt_files = [f for f in ckpt_files if args.tok_config in f]

        if not ckpt_files:
            print("No checkpoints matched the given filters.")
            print(f"Available: {os.listdir(args.ckpt_dir)}")
            exit(1)

        print(f"\nFound {len(ckpt_files)} checkpoint(s) to evaluate:")
        for f in ckpt_files:
            print(f"  • {f}")

        for fname in ckpt_files:
            ckpt_path = os.path.join(args.ckpt_dir, fname)
            exp_name  = fname.replace("_best.pt", "")
            print(f"\n{'━'*60}")
            print(f"  Evaluating: {exp_name}")
            print(f"{'━'*60}")
            try:
                metrics = evaluate(
                    ckpt_path=ckpt_path,
                    test_csv=args.test_csv,
                    batch_size=args.batch_size,
                    max_samples=args.max_samples,
                )
                all_results[exp_name] = metrics
            except Exception as e:
                print(f"  ✗ Failed: {e}")
                all_results[exp_name] = {"error": str(e)}

        # ── Summary table ──────────────────────────────────────────────────────
        print(f"\n{'='*70}")
        print(f"  EVALUATION SUMMARY  (test: {args.test_csv})")
        print(f"{'='*70}")
        print(f"  {'Experiment':<38} {'GLEU':>6}  {'BLEU':>6}  {'TokAcc':>7}")
        print(f"  {'─'*38}  {'─'*6}  {'─'*6}  {'─'*7}")
        for name, m in sorted(all_results.items()):
            if "error" in m:
                print(f"  {name:<38}  ERROR: {m['error']}")
            else:
                print(f"  {name:<38}  {m['gleu']:>6.2f}  {m['bleu']:>6.2f}  {m['token_acc']:>6.2f}%")
        print(f"{'='*70}")

    else:
        # ── Single checkpoint mode ─────────────────────────────────────────────
        if not args.checkpoint:
            p.error("Provide --checkpoint for single-run mode, or use --run_all")
        metrics = evaluate(
            ckpt_path=args.checkpoint,
            test_csv=args.test_csv,
            batch_size=args.batch_size,
            max_samples=args.max_samples,
        )
        exp_name = os.path.basename(args.checkpoint).replace("_best.pt", "")
        all_results[exp_name] = metrics

    # ── Optionally save results to JSON ────────────────────────────────────────
    if args.save_json and all_results:
        with open(args.save_json, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\n  Results saved → {args.save_json}")
