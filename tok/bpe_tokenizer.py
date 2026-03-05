"""
BPE Tokenizer module for PolyGEC project.

Two tokenization strategies as described in the project outline:
  1. "Chopped"  (Aggressive sub-word): sentences → sub-word tokens via BPE
  2. "Common"   (Word-level baseline):  sentences → word tokens (space-split)

Uses the `tokenizers` library (HuggingFace) for BPE training.
Special tokens: <pad>=0  <sos>=1  <eos>=2  <unk>=3
"""

import os
import csv
import json
from pathlib import Path
from typing import List, Tuple

# ── HuggingFace fast tokenizers ────────────────────────────────────────────
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing

# ── Constants ──────────────────────────────────────────────────────────────
PAD_TOKEN = "<pad>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"
SPECIAL_TOKENS = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN]

PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3


# ══════════════════════════════════════════════════════════════════════════════
#  ChoppedTokenizer  — aggressive BPE sub-word tokenizer
# ══════════════════════════════════════════════════════════════════════════════
class ChoppedTokenizer:
    """
    Aggressive BPE tokenizer.
    Breaks words into sub-word units, e.g. 'learning' → 'learn' + '##ing'.
    Trained jointly on source + target sentences.
    """

    def __init__(self, vocab_size: int = 8000):
        self.vocab_size = vocab_size
        self.tokenizer: Tokenizer = None

    # ── Training ─────────────────────────────────────────────────────────────
    def train(self, sentences: List[str], save_path: str = None):
        """Train BPE on a list of sentences."""
        tokenizer = Tokenizer(BPE(unk_token=UNK_TOKEN))
        tokenizer.pre_tokenizer = Whitespace()

        trainer = BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=SPECIAL_TOKENS,
            min_frequency=2,
            show_progress=True,
        )
        tokenizer.train_from_iterator(sentences, trainer=trainer)

        # Wrap each sequence with <sos> … <eos>
        tokenizer.post_processor = TemplateProcessing(
            single=f"{SOS_TOKEN} $A {EOS_TOKEN}",
            special_tokens=[
                (SOS_TOKEN, tokenizer.token_to_id(SOS_TOKEN)),
                (EOS_TOKEN, tokenizer.token_to_id(EOS_TOKEN)),
            ],
        )

        self.tokenizer = tokenizer
        if save_path:
            self.save(save_path)
        print(f"[ChoppedTokenizer] Trained. Vocab size = {self.tokenizer.get_vocab_size()}")

    # ── Encode / Decode ───────────────────────────────────────────────────────
    def encode(self, sentence: str) -> List[int]:
        return self.tokenizer.encode(sentence).ids

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=skip_special)

    def vocab_size_actual(self) -> int:
        return self.tokenizer.get_vocab_size()

    # ── Save / Load ───────────────────────────────────────────────────────────
    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.tokenizer.save(path)
        print(f"[ChoppedTokenizer] Saved to {path}")

    def load(self, path: str):
        self.tokenizer = Tokenizer.from_file(path)
        print(f"[ChoppedTokenizer] Loaded from {path}")


# ══════════════════════════════════════════════════════════════════════════════
#  CommonTokenizer  — word-level tokenizer (baseline)
# ══════════════════════════════════════════════════════════════════════════════
class CommonTokenizer:
    """
    Word-level tokenizer.
    Splits on whitespace and punctuation; builds a fixed vocabulary.
    Keeps the top `vocab_size` most-frequent tokens.
    """

    def __init__(self, vocab_size: int = 20000):
        self.vocab_size = vocab_size
        self.word2idx = {}
        self.idx2word = {}

    # ── Training ─────────────────────────────────────────────────────────────
    def train(self, sentences: List[str], save_path: str = None):
        from collections import Counter
        counter = Counter()
        for sent in sentences:
            for tok in sent.lower().split():
                counter[tok] += 1

        # Reserve slots for special tokens
        vocab = SPECIAL_TOKENS + [
            w for w, _ in counter.most_common(self.vocab_size - len(SPECIAL_TOKENS))
        ]

        self.word2idx = {w: i for i, w in enumerate(vocab)}
        self.idx2word = {i: w for w, i in self.word2idx.items()}

        if save_path:
            self.save(save_path)
        print(f"[CommonTokenizer] Trained. Vocab size = {len(self.word2idx)}")

    # ── Encode / Decode ───────────────────────────────────────────────────────
    def encode(self, sentence: str) -> List[int]:
        tokens = [SOS_TOKEN] + sentence.lower().split() + [EOS_TOKEN]
        return [self.word2idx.get(t, UNK_IDX) for t in tokens]

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        special = set(SPECIAL_TOKENS) if skip_special else set()
        words = [
            self.idx2word.get(i, UNK_TOKEN)
            for i in ids
            if self.idx2word.get(i, UNK_TOKEN) not in special
        ]
        return " ".join(words)

    def vocab_size_actual(self) -> int:
        return len(self.word2idx)

    # ── Save / Load ───────────────────────────────────────────────────────────
    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.word2idx, f, ensure_ascii=False, indent=2)
        print(f"[CommonTokenizer] Saved to {path}")

    def load(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            self.word2idx = json.load(f)
        self.idx2word = {int(i): w for w, i in self.word2idx.items()}
        print(f"[CommonTokenizer] Loaded from {path}. Vocab = {len(self.word2idx)}")


# ══════════════════════════════════════════════════════════════════════════════
#  Helper: build & save both tokenizers from the training CSV
# ══════════════════════════════════════════════════════════════════════════════
def build_tokenizers(
    train_csv: str,
    save_dir: str = "tokenizers/saved",
    bpe_vocab_size: int = 8000,
    word_vocab_size: int = 20000,
) -> Tuple[ChoppedTokenizer, CommonTokenizer]:
    """
    Read the training CSV, collect all sentences (src + tgt),
    train both tokenizers, and save them.

    Returns (chopped_tok, common_tok).
    """
    print(f"Reading sentences from {train_csv} ...")
    sentences = []
    with open(train_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        src_col = "source" if "source" in fieldnames else "src"
        tgt_col = "target" if "target" in fieldnames else "trg"
        for row in reader:
            if row.get(src_col):
                sentences.append(row[src_col].strip())
            if row.get(tgt_col):
                sentences.append(row[tgt_col].strip())
    print(f"  Total sentences: {len(sentences):,}")

    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # Chopped (BPE)
    chopped = ChoppedTokenizer(vocab_size=bpe_vocab_size)
    chopped.train(sentences, save_path=os.path.join(save_dir, "chopped_bpe.json"))

    # Common (word-level)
    common = CommonTokenizer(vocab_size=word_vocab_size)
    common.train(sentences, save_path=os.path.join(save_dir, "common_word.json"))

    return chopped, common


# ══════════════════════════════════════════════════════════════════════════════
#  Quick test
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    TRAIN_CSV = "lang8_train.csv"
    chopped, common = build_tokenizers(
        train_csv=TRAIN_CSV,
        save_dir="tok/saved",
        bpe_vocab_size=8000,
        word_vocab_size=20000,
    )

    sample = "I goes to the school yesterday."
    print("\nSample:", sample)
    print("Chopped IDs :", chopped.encode(sample))
    print("Chopped back:", chopped.decode(chopped.encode(sample)))
    print("Common  IDs :", common.encode(sample))
    print("Common  back:", common.decode(common.encode(sample)))
