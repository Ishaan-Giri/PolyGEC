# PolyGEC — Multilingual Grammatical Error Correction

Compares **RNN+Attention** and **LSTM+Attention** seq2seq models for GEC across three tokenization configurations and multiple languages.

---

## Project Structure

```
PolyGEC/
├── config.py            # All paths and hyperparameters
├── train.py             # Training script
├── predict.py           # Inference script
├── requirements.txt
├── models/
│   ├── rnn_attention.py   # BiGRU encoder + Bahdanau attention decoder
│   └── lstm_attention.py  # BiLSTM encoder + Bahdanau attention decoder
├── tok/
│   └── bpe_tokenizer.py   # ChoppedTokenizer (BPE) + CommonTokenizer (word-level)
├── data/
│   └── dataset.py         # GECDataset + DataLoader
├── evaluation/
│   └── evaluate.py        # GLEU, Corpus BLEU, Token Accuracy
└── checkpoints/           # Saved .pt model files (git-ignored)
```

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Tokenization Configs

| Config            | Source        | Target        |
|-------------------|---------------|---------------|
| `chopped_common`  | BPE (8 000)   | Word (20 000) |
| `common_common`   | Word (20 000) | Word (20 000) |
| `chopped_chopped` | BPE (8 000)   | BPE (8 000)   |

---

## Data Format

Training and test CSVs must have two columns. Either naming works:

```
src,trg               ← or →    source,target
incorrect sentence              incorrect sentence
...                             ...
```

---

## Training

```bash
# Single model + single config  (val split auto-carved from train_csv)
python train.py --model lstm --tok_config common_common \
    --train_csv data/tamil_train.csv --lang ta

# All 6 experiments at once (2 models × 3 configs)
python train.py --run_all \
    --train_csv data/tamil_train.csv --lang ta

# All 3 configs for one model
for cfg in chopped_common common_common chopped_chopped; do
  python train.py --model lstm --tok_config $cfg \
      --train_csv data/tamil_train.csv --lang ta
done
```

> **Note:** `--test_csv` is intentionally omitted during training.  
> A 10% validation split is carved automatically from `--train_csv` to monitor val loss.  
> The test CSV is used only for final evaluation (`evaluation/evaluate.py`).

### Key flags

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `lstm` | `rnn` or `lstm` |
| `--tok_config` | `chopped_common` | tokenization config (see table above) |
| `--run_all` | off | run all 6 experiments sequentially |
| `--train_csv` | config default | path to training CSV |
| `--lang` | `en` | language tag used in checkpoint naming |
| `--epochs` | 15 | training epochs |
| `--batch_size` | 64 | batch size |
| `--max_samples` | None | limit data size (debugging) |

Checkpoints → `checkpoints/<lang>_<model>_<tok_config>_best.pt`  
Tokenizers  → `tok/saved/<lang>/` (rebuilt automatically if missing)

---

## Evaluation

```bash
# Single checkpoint
python evaluation/evaluate.py \
    --checkpoint checkpoints/ta_lstm_common_common_best.pt \
    --test_csv   data/tamil_test.csv

# All checkpoints for a language → prints summary table, saves JSON
python evaluation/evaluate.py \
    --run_all --lang ta \
    --test_csv   data/tamil_test.csv \
    --save_json  results_ta.json

# Filter by model or config
python evaluation/evaluate.py --run_all --lang ta --model lstm \
    --test_csv data/tamil_test.csv

python evaluation/evaluate.py --run_all --lang ta --tok_config common_common \
    --test_csv data/tamil_test.csv
```

**Metrics:** GLEU · Corpus BLEU · Token Accuracy · Precision / Recall / F0.5 (ERRANT-style) · Inference Latency · Model Size

---

## Inference

```bash
# Single sentence
python predict.py \
    --checkpoint checkpoints/ta_lstm_common_common_best.pt \
    --sentence "இந்த வாக்கியம் தவறானது"

# File (one sentence per line)
python predict.py \
    --checkpoint checkpoints/ta_lstm_common_common_best.pt \
    --input_file  sentences.txt \
    --output_file corrected.txt
```

---

## Models

| Model | Encoder | Decoder |
|-------|---------|---------|
| RNN   | 2-layer Bidirectional GRU  | GRU + Bahdanau attention  |
| LSTM  | 2-layer Bidirectional LSTM | LSTM + Bahdanau attention |

Both: embed_dim=256 · hidden_dim=512 · dropout=0.3

---

## References

1. Bahdanau et al. (2015). *Neural Machine Translation by Jointly Learning to Align and Translate.* ICLR.
2. Sennrich et al. (2016). *Neural Machine Translation of Rare Words with Subword Units.* ACL.
3. Sharma & Bhattacharyya (2025). *IndiGEC: Multilingual Grammar Error Correction.* EMNLP.
