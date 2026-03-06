"""
Evaluation script for PolyGEC.

Metrics (matching project outline):
  1. GLEU            — sentence-level BLEU variant optimised for GEC
  2. Corpus BLEU     — overall translation quality (sacrebleu)
  3. Token Accuracy  — exact token match rate
  4. P / R / F0.5    — ERRANT-style span-level edit precision, recall, F0.5
  5. Efficiency      — inference latency (ms/sentence) + model size (MB)

Usage:
    # Evaluate a single checkpoint
    python evaluation/evaluate.py \
        --checkpoint checkpoints/ta_rnn_common_common_best.pt \
        --test_csv   data/tamil_test.csv

    # Evaluate ALL checkpoints for a language → summary table + JSON
    python evaluation/evaluate.py \
        --run_all --lang ta \
        --test_csv  data/tamil_test.csv \
        --save_json results_ta.json

    # Filter by model or config
    python evaluation/evaluate.py --run_all --lang ta --model rnn \
        --test_csv data/tamil_test.csv
    python evaluation/evaluate.py --run_all --lang ta --tok_config common_common \
        --test_csv data/tamil_test.csv
"""

import os
import sys
import csv
import json
import time
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
#  Metric 1 — GLEU
# ══════════════════════════════════════════════════════════════════════════════
def compute_gleu(hypotheses, references):
    """Sentence-level GLEU averaged over corpus (sacrebleu proxy)."""
    scores = [sacrebleu.sentence_bleu(h, [r]).score
              for h, r in zip(hypotheses, references)]
    return sum(scores) / len(scores) if scores else 0.0


# ══════════════════════════════════════════════════════════════════════════════
#  Metric 2 — Corpus BLEU
# ══════════════════════════════════════════════════════════════════════════════
def compute_corpus_bleu(hypotheses, references):
    return sacrebleu.corpus_bleu(hypotheses, [references]).score


# ══════════════════════════════════════════════════════════════════════════════
#  Metric 3 — Token Accuracy
# ══════════════════════════════════════════════════════════════════════════════
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
#  Metric 4 — ERRANT-style Precision / Recall / F0.5
#  Span-level: counts token edits (insertions, deletions, substitutions)
#  using a greedy token-diff approach.  No external ERRANT install needed.
# ══════════════════════════════════════════════════════════════════════════════
def _token_edits(hyp_tokens, ref_tokens):
    """
    Returns (tp, fp, fn) counts by comparing token sequences.
      tp = tokens in hyp that match ref (correct edits / unchanged correct tokens)
      fp = tokens in hyp not in ref  (over-corrections)
      fn = tokens in ref not in hyp  (missed corrections)
    Uses longest-common-subsequence matching.
    """
    from difflib import SequenceMatcher
    matcher = SequenceMatcher(None, hyp_tokens, ref_tokens, autojunk=False)
    tp = sum(block.size for block in matcher.get_matching_blocks())
    fp = len(hyp_tokens) - tp
    fn = len(ref_tokens)  - tp
    return tp, fp, fn


def compute_prf05(hypotheses, references):
    """
    Corpus-level Precision, Recall, F0.5 (ERRANT-style).
    F0.5 weights precision twice as much as recall (standard for GEC).
    """
    total_tp = total_fp = total_fn = 0
    for hyp, ref in zip(hypotheses, references):
        tp, fp, fn = _token_edits(hyp.split(), ref.split())
        total_tp += tp
        total_fp += fp
        total_fn += fn

    precision = total_tp / (total_tp + total_fp) * 100 if (total_tp + total_fp) > 0 else 0.0
    recall    = total_tp / (total_tp + total_fn) * 100 if (total_tp + total_fn) > 0 else 0.0
    beta = 0.5
    f05 = ((1 + beta**2) * precision * recall /
           (beta**2 * precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f05


# ══════════════════════════════════════════════════════════════════════════════
#  Metric 5 — Computational Efficiency
# ══════════════════════════════════════════════════════════════════════════════
def compute_efficiency(model, src_tok, hp, device, n_sentences=200):
    """
    Measures:
      - Inference latency  (ms per sentence)
      - Model size         (MB, parameter count)
    Runs n_sentences dummy forward passes with avg-length inputs.
    """
    model.eval()
    dummy = ["dummy sentence for timing"] * n_sentences
    src_ids = [
        torch.tensor(src_tok.encode(s)[:hp["max_len"]], dtype=torch.long)
        for s in dummy
    ]
    max_l  = max(t.size(0) for t in src_ids)
    padded = torch.zeros(len(src_ids), max_l, dtype=torch.long).to(device)
    for j, t in enumerate(src_ids):
        padded[j, :t.size(0)] = t

    # Warm-up
    with torch.no_grad():
        model.generate(padded[:4], max_len=hp["max_len"])

    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        model.generate(padded, max_len=hp["max_len"])
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - t0) * 1000

    latency_ms  = elapsed_ms / n_sentences
    param_count = sum(p.numel() for p in model.parameters())
    size_mb     = param_count * 4 / (1024 ** 2)   # float32 = 4 bytes
    return latency_ms, size_mb, param_count


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
    gleu              = compute_gleu(hypotheses, targets)
    bleu              = compute_corpus_bleu(hypotheses, targets)
    tok_acc           = compute_token_accuracy(hypotheses, targets)
    precision, recall, f05 = compute_prf05(hypotheses, targets)
    latency_ms, size_mb, param_count = compute_efficiency(model, src_tok, hp, device)

    print(f"\n{'='*55}")
    print(f"  RESULTS  [{model_name.upper()} | {tok_config} | {lang}]")
    print(f"{'='*55}")
    print(f"  GLEU (sentence-level avg) : {gleu:.2f}")
    print(f"  Corpus BLEU               : {bleu:.2f}")
    print(f"  Token Accuracy            : {tok_acc:.2f}%")
    print(f"  ─────────────────────────────────────────────")
    print(f"  Precision  (ERRANT-style) : {precision:.2f}%")
    print(f"  Recall     (ERRANT-style) : {recall:.2f}%")
    print(f"  F0.5       (ERRANT-style) : {f05:.2f}%")
    print(f"  ─────────────────────────────────────────────")
    print(f"  Latency                   : {latency_ms:.2f} ms/sentence")
    print(f"  Model Size                : {size_mb:.1f} MB  ({param_count:,} params)")
    print(f"{'='*55}")

    # ── Show a few examples ────────────────────────────────────────────────────
    print("\n  Sample Predictions:")
    print(f"  {'─'*70}")
    for src, hyp, ref in zip(sources[:5], hypotheses[:5], targets[:5]):
        print(f"  SRC : {src}")
        print(f"  HYP : {hyp}")
        print(f"  REF : {ref}")
        print(f"  {'─'*70}")

    return {
        "gleu":       gleu,
        "bleu":       bleu,
        "token_acc":  tok_acc,
        "precision":  precision,
        "recall":     recall,
        "f0.5":       f05,
        "latency_ms": latency_ms,
        "size_mb":    size_mb,
        "params":     param_count,
    }


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
        W = 108
        print(f"\n{'='*W}")
        print(f"  EVALUATION SUMMARY  (test: {args.test_csv})")
        print(f"{'='*W}")
        hdr = (f"  {'Experiment':<34} {'GLEU':>6}  {'BLEU':>6}  {'TokAcc':>7}"
               f"  {'Prec':>7}  {'Rec':>7}  {'F0.5':>7}"
               f"  {'Lat(ms)':>8}  {'Size(MB)':>8}  {'Params':>12}")
        print(hdr)
        print(f"  {'─'*34}  {'─'*6}  {'─'*6}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*8}  {'─'*8}  {'─'*12}")
        for name, m in sorted(all_results.items()):
            if "error" in m:
                print(f"  {name:<34}  ERROR: {m['error']}")
            else:
                print(
                    f"  {name:<34}"
                    f"  {m['gleu']:>6.2f}"
                    f"  {m['bleu']:>6.2f}"
                    f"  {m['token_acc']:>6.2f}%"
                    f"  {m['precision']:>6.2f}%"
                    f"  {m['recall']:>6.2f}%"
                    f"  {m['f0.5']:>6.2f}%"
                    f"  {m['latency_ms']:>8.2f}"
                    f"  {m['size_mb']:>8.1f}"
                    f"  {m['params']:>12,}"
                )
        print(f"{'='*W}")

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

    # ── Auto-save results to results/ folder ──────────────────────────────────
    if all_results:
        import datetime
        results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
        os.makedirs(results_dir, exist_ok=True)

        # Build a filename stem from lang + timestamp
        lang_tag  = args.lang or "all"
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        stem      = f"{lang_tag}_{timestamp}"

        # 1. JSON — full numeric results
        json_path = os.path.join(results_dir, f"{stem}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)

        # 2. Plain-text report — human-readable
        txt_path = os.path.join(results_dir, f"{stem}.txt")
        W = 108
        lines = []
        lines.append("=" * W)
        lines.append(f"  PolyGEC Evaluation Report — {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"  Test CSV : {args.test_csv}")
        lines.append("=" * W)
        hdr = (f"  {'Experiment':<34} {'GLEU':>6}  {'BLEU':>6}  {'TokAcc':>7}"
               f"  {'Prec':>7}  {'Rec':>7}  {'F0.5':>7}"
               f"  {'Lat(ms)':>8}  {'Size(MB)':>8}  {'Params':>12}")
        lines.append(hdr)
        lines.append(f"  {'─'*34}  {'─'*6}  {'─'*6}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*8}  {'─'*8}  {'─'*12}")
        for name, m in sorted(all_results.items()):
            if "error" in m:
                lines.append(f"  {name:<34}  ERROR: {m['error']}")
            else:
                lines.append(
                    f"  {name:<34}"
                    f"  {m['gleu']:>6.2f}"
                    f"  {m['bleu']:>6.2f}"
                    f"  {m['token_acc']:>6.2f}%"
                    f"  {m['precision']:>6.2f}%"
                    f"  {m['recall']:>6.2f}%"
                    f"  {m['f0.5']:>6.2f}%"
                    f"  {m['latency_ms']:>8.2f}"
                    f"  {m['size_mb']:>8.1f}"
                    f"  {m['params']:>12,}"
                )
        lines.append("=" * W)
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

        print(f"\n  ✓ Results auto-saved:")
        print(f"      JSON : {json_path}")
        print(f"      TXT  : {txt_path}")

        # 3. Honour explicit --save_json if provided
        if args.save_json:
            with open(args.save_json, "w", encoding="utf-8") as f:
                json.dump(all_results, f, indent=2)
            print(f"      JSON : {args.save_json}  (--save_json override)")
