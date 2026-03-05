# PolyGEC — Multilingual Grammatical Error Correction

## Project Overview
PolyGEC compares two seq2seq architectures — **RNN with Attention** and **LSTM with Attention** — under three BPE tokenization configurations for the GEC task.

---

## Project Structure
```
project/
├── tok/
│   ├── bpe_tokenizer.py     # ChoppedTokenizer (BPE) + CommonTokenizer (word-level)
│   └── saved/               # Saved tokenizer files (generated at runtime)
│       ├── chopped_bpe.json
│       └── common_word.json
├── data/
│   └── dataset.py           # GECDataset + DataLoader builder
├── models/
│   ├── rnn_attention.py     # Encoder-Decoder GRU + Bahdanau Attention
│   └── lstm_attention.py    # Encoder-Decoder LSTM + Bahdanau Attention
├── evaluation/
│   └── evaluate.py          # GLEU, Corpus BLEU, Token Accuracy
├── checkpoints/             # Saved model checkpoints (.pt files)
├── train.py                 # Unified training script
├── requirements.txt
└── README.md
```

---

## Dataset
- **Lang-8 Corpus** (`lang8_train.csv` / `lang8_test.csv`)
- 180,000 training pairs / 20,000 test pairs
- Format: `source` (incorrect sentence) → `target` (corrected sentence)

Split: 90% train / 10% test (random shuffle, seed=42)

---

## Tokenization Strategies

| Config Name       | Source Tokenizer | Target Tokenizer |
|-------------------|-----------------|-----------------|
| `chopped_common`  | BPE (sub-word)  | Word-level      |
| `common_common`   | Word-level      | Word-level      |
| `chopped_chopped` | BPE (sub-word)  | BPE (sub-word)  |

- **Chopped (BPE)**: vocab size = 8,000 sub-word units
- **Common (Word)**: vocab size = 20,000 most-frequent words

---

## Models

### 1. RNN with Bahdanau Attention (`models/rnn_attention.py`)
- Bidirectional GRU encoder
- GRU decoder with additive (Bahdanau) attention
- Reference: Bahdanau et al. (2015), ICLR

### 2. LSTM with Bahdanau Attention (`models/lstm_attention.py`)
- Multi-layer bidirectional LSTM encoder
- LSTM decoder with additive (Bahdanau) attention
- Captures long-term dependencies via cell state gating
- Reference: Cherian & Balakrishnan (2022)

---

## Setup

```bash
# Install dependencies
pip install torch tokenizers sacrebleu numpy

# Split dataset (if not already done)
python split_lang8.py
```

---

## Training

```bash
# Train a single experiment
python train.py --model lstm --tok_config chopped_common --epochs 15

# Train all 6 experiments (2 models × 3 tok configs)
python train.py --run_all

# Quick smoke test (small subset)
python train.py --model rnn --tok_config common_common --max_samples 1000 --epochs 2
```

### Hyperparameters (defaults)

| Parameter     | Value  |
|---------------|--------|
| embed_dim     | 256    |
| hidden_dim    | 512    |
| enc_layers    | 2      |
| dropout       | 0.3    |
| batch_size    | 64     |
| epochs        | 15     |
| learning_rate | 1e-3   |
| grad_clip     | 1.0    |
| max_seq_len   | 100    |
| teacher_force | 0.5    |

---

## Evaluation

```bash
python evaluation/evaluate.py \
    --checkpoint checkpoints/lstm_chopped_common_best.pt \
    --test_csv   lang8_test.csv
```

### Metrics
- **GLEU** — Sentence-level BLEU variant optimised for GEC
- **Corpus BLEU** — Overall translation quality (sacrebleu)
- **Token Accuracy** — Exact token match rate

---

## Experimental Design (3 × 2 Factorial)

| Experiment            | Model | Tok Config      |
|-----------------------|-------|-----------------|
| rnn_chopped_common    | RNN   | chopped→common  |
| rnn_common_common     | RNN   | common→common   |
| rnn_chopped_chopped   | RNN   | chopped→chopped |
| lstm_chopped_common   | LSTM  | chopped→common  |
| lstm_common_common    | LSTM  | common→common   |
| lstm_chopped_chopped  | LSTM  | chopped→chopped |

---

## References
1. Bahdanau et al. (2015). *Neural Machine Translation by Jointly Learning to Align and Translate*. ICLR.
2. Cherian & Balakrishnan (2022). *Evaluating Grammatical Correctness of Malayalam Text using improved Text GCN and LSTM*. IJETT.
3. Sharma & Bhattacharyya (2025). *IndiGEC: Multilingual Grammar Error Correction*. EMNLP.
