"""
Inference script for PolyGEC.
Loads a trained checkpoint and corrects input sentences.

Usage:
    # Correct a single sentence
    python predict.py --checkpoint checkpoints/lstm_chopped_common_best.pt \
                      --sentence "I goes to school yesterday."

    # Correct sentences from a text file (one per line)
    python predict.py --checkpoint checkpoints/lstm_chopped_common_best.pt \
                      --input_file my_sentences.txt \
                      --output_file corrected.txt
"""

import os
import sys
import argparse
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tok.bpe_tokenizer  import ChoppedTokenizer, CommonTokenizer
from models.rnn_attention  import RNNSeq2Seq
from models.lstm_attention import LSTMSeq2Seq
from config import TOK_DIR, EOS_IDX, HYPERPARAMS

# ══════════════════════════════════════════════════════════════════════════════
#  Load helpers
# ══════════════════════════════════════════════════════════════════════════════
def load_tokenizer(tok_type: str):
    if tok_type == "chopped":
        t = ChoppedTokenizer()
        t.load(os.path.join(TOK_DIR, "chopped_bpe.json"))
    else:
        t = CommonTokenizer()
        t.load(os.path.join(TOK_DIR, "common_word.json"))
    return t


def load_model_from_checkpoint(ckpt_path: str, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    model_name = ckpt["model_name"]
    tok_config = ckpt["tok_config"]
    hp         = ckpt["hp"]

    src_type, tgt_type = tok_config.split("_", 1)
    # Handle 3-part configs like "chopped_chopped"
    parts = tok_config.split("_")
    src_type = parts[0]
    tgt_type = "_".join(parts[1:]) if len(parts) > 2 else parts[1]

    src_tok = load_tokenizer(src_type)
    tgt_tok = load_tokenizer(tgt_type)

    src_vocab = src_tok.vocab_size_actual()
    tgt_vocab = tgt_tok.vocab_size_actual()

    kwargs = dict(
        src_vocab_size=src_vocab,
        tgt_vocab_size=tgt_vocab,
        embed_dim=hp["embed_dim"],
        hidden_dim=hp["hidden_dim"],
        enc_layers=hp["enc_layers"],
        dropout=0.0,
    )
    if model_name == "rnn":
        model = RNNSeq2Seq(**kwargs)
    else:
        model = LSTMSeq2Seq(**kwargs)

    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()

    print(f"Loaded [{model_name.upper()}] | tok={tok_config} | epoch={ckpt['epoch']} | val_loss={ckpt['val_loss']:.4f}")
    return model, src_tok, tgt_tok, hp


# ══════════════════════════════════════════════════════════════════════════════
#  Correct a single sentence
# ══════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def correct_sentence(sentence: str, model, src_tok, tgt_tok, max_len: int, device) -> str:
    ids = src_tok.encode(sentence)[:max_len]
    src = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)  # (1, src_len)

    gen = model.generate(src, max_len=max_len)  # (1, gen_len)
    out_ids = gen[0].cpu().tolist()

    # Strip up to first EOS
    clean = []
    for i in out_ids:
        if i == EOS_IDX:
            break
        clean.append(i)

    return tgt_tok.decode(clean, skip_special=True).strip()


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════
def main():
    p = argparse.ArgumentParser(description="PolyGEC Inference")
    p.add_argument("--checkpoint",   required=True, help="Path to .pt checkpoint")
    p.add_argument("--sentence",     type=str, default=None,
                   help="Single sentence to correct")
    p.add_argument("--input_file",   type=str, default=None,
                   help="Text file with one sentence per line")
    p.add_argument("--output_file",  type=str, default=None,
                   help="Where to write corrected sentences")
    p.add_argument("--device",       type=str, default="auto")
    args = p.parse_args()

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    model, src_tok, tgt_tok, hp = load_model_from_checkpoint(args.checkpoint, device)
    max_len = hp.get("max_len", HYPERPARAMS["max_len"])

    # ── Single sentence mode ───────────────────────────────────────────────────
    if args.sentence:
        result = correct_sentence(args.sentence, model, src_tok, tgt_tok, max_len, device)
        print(f"\nInput    : {args.sentence}")
        print(f"Corrected: {result}")

    # ── File mode ──────────────────────────────────────────────────────────────
    elif args.input_file:
        with open(args.input_file, "r", encoding="utf-8") as fin:
            sentences = [line.strip() for line in fin if line.strip()]

        print(f"Correcting {len(sentences):,} sentences …")
        results = []
        for i, sent in enumerate(sentences):
            corrected = correct_sentence(sent, model, src_tok, tgt_tok, max_len, device)
            results.append(corrected)
            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{len(sentences)}")

        if args.output_file:
            with open(args.output_file, "w", encoding="utf-8") as fout:
                fout.write("\n".join(results) + "\n")
            print(f"Saved to {args.output_file}")
        else:
            for src, out in zip(sentences[:10], results[:10]):
                print(f"  IN : {src}")
                print(f"  OUT: {out}")
                print()
    else:
        p.print_help()


if __name__ == "__main__":
    main()
