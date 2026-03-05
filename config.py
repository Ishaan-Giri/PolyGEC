"""
Central configuration for PolyGEC.
All hyperparameters and paths are defined here.
"""

import os

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
TRAIN_CSV  = os.path.join(BASE_DIR, "lang8_train.csv")
TEST_CSV   = os.path.join(BASE_DIR, "lang8_test.csv")
TOK_DIR    = os.path.join(BASE_DIR, "tok", "saved")
CKPT_DIR   = os.path.join(BASE_DIR, "checkpoints")

# ── Special token indices ──────────────────────────────────────────────────────
PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3

# ── Tokenization configs ───────────────────────────────────────────────────────
# name  →  (src_tokenizer_type, tgt_tokenizer_type)
TOKENIZATION_CONFIGS = {
    "chopped_common"  : ("chopped", "common"),
    "common_common"   : ("common",  "common"),
    "chopped_chopped" : ("chopped", "chopped"),
}

# ── Vocabulary sizes ───────────────────────────────────────────────────────────
BPE_VOCAB_SIZE  = 8_000
WORD_VOCAB_SIZE = 20_000

# ── Model hyperparameters ──────────────────────────────────────────────────────
HYPERPARAMS = {
    "embed_dim"  : 256,
    "hidden_dim" : 512,
    "enc_layers" : 2,
    "dropout"    : 0.3,
    "batch_size" : 64,
    "epochs"     : 15,
    "lr"         : 1e-3,
    "clip"       : 1.0,    # gradient clipping norm
    "max_len"    : 100,    # max sequence length
    "tf_ratio"   : 0.5,    # teacher-forcing ratio
    "bpe_vocab"  : BPE_VOCAB_SIZE,
    "word_vocab" : WORD_VOCAB_SIZE,
}
