"""
PyTorch Dataset and DataLoader utilities for PolyGEC.

Supports three input→output tokenization configurations:
  1. chopped → common   (BPE source, word-level target)
  2. common  → common   (word-level both)
  3. chopped → chopped  (BPE both)

Usage:
    from data.dataset import GECDataset, build_dataloader
"""

import csv
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
from typing import List, Tuple, Optional

# Token indices (must match tokenizer definitions)
PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3


# ══════════════════════════════════════════════════════════════════════════════
#  GECDataset
# ══════════════════════════════════════════════════════════════════════════════
class GECDataset(Dataset):
    """
    Grammatical Error Correction dataset.
    Reads a CSV with columns: source (incorrect), target (correct).
    Applies the given src_tokenizer and tgt_tokenizer to each row.

    Args:
        csv_path      : path to lang8_train.csv or lang8_test.csv
        src_tokenizer : tokenizer for source (incorrect) sentences
        tgt_tokenizer : tokenizer for target (correct) sentences
        max_len       : maximum token length (longer sequences are truncated)
        max_samples   : if set, only load first N samples (useful for debugging)
    """

    def __init__(
        self,
        csv_path: str,
        src_tokenizer,
        tgt_tokenizer,
        max_len: int = 128,
        max_samples: Optional[int] = None,
    ):
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.max_len = max_len

        self.src_data: List[List[int]] = []
        self.tgt_data: List[List[int]] = []

        self._load(csv_path, max_samples)

    def _load(self, csv_path: str, max_samples: Optional[int]):
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            # Auto-detect column names: support source/target AND src/trg
            fieldnames = reader.fieldnames or []
            src_col = "source" if "source" in fieldnames else "src"
            tgt_col = "target" if "target" in fieldnames else "trg"
            for i, row in enumerate(reader):
                if max_samples is not None and i >= max_samples:
                    break

                src = row.get(src_col, "").strip()
                tgt = row.get(tgt_col, "").strip()
                if not src or not tgt:
                    continue

                src_ids = self.src_tokenizer.encode(src)[: self.max_len]
                tgt_ids = self.tgt_tokenizer.encode(tgt)[: self.max_len]

                self.src_data.append(src_ids)
                self.tgt_data.append(tgt_ids)

        print(f"[GECDataset] Loaded {len(self.src_data):,} samples from {csv_path}")

    def __len__(self) -> int:
        return len(self.src_data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        src = torch.tensor(self.src_data[idx], dtype=torch.long)
        tgt = torch.tensor(self.tgt_data[idx], dtype=torch.long)
        return src, tgt


# ══════════════════════════════════════════════════════════════════════════════
#  Collate function — pads variable-length sequences in a batch
# ══════════════════════════════════════════════════════════════════════════════
def collate_fn(
    batch: List[Tuple[torch.Tensor, torch.Tensor]]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pads src and tgt sequences to the same length within a batch.
    Returns:
        src_padded : (batch_size, src_max_len)
        tgt_padded : (batch_size, tgt_max_len)
    """
    src_batch, tgt_batch = zip(*batch)
    src_padded = pad_sequence(src_batch, batch_first=True, padding_value=PAD_IDX)
    tgt_padded = pad_sequence(tgt_batch, batch_first=True, padding_value=PAD_IDX)
    return src_padded, tgt_padded


# ══════════════════════════════════════════════════════════════════════════════
#  Convenience builder
# ══════════════════════════════════════════════════════════════════════════════
def build_dataloader(
    csv_path: str,
    src_tokenizer,
    tgt_tokenizer,
    batch_size: int = 64,
    shuffle: bool = True,
    max_len: int = 128,
    max_samples: Optional[int] = None,
    num_workers: int = 2,
) -> DataLoader:
    """Build a DataLoader from a CSV file."""
    dataset = GECDataset(
        csv_path=csv_path,
        src_tokenizer=src_tokenizer,
        tgt_tokenizer=tgt_tokenizer,
        max_len=max_len,
        max_samples=max_samples,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )
    return loader


def build_train_val_dataloaders(
    csv_path: str,
    src_tokenizer,
    tgt_tokenizer,
    val_split: float = 0.1,
    batch_size: int = 64,
    max_len: int = 128,
    max_samples: Optional[int] = None,
    num_workers: int = 2,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """
    Load one CSV, split it into train/val (no test CSV needed).
    val_split=0.1 reserves 10% of the training data for validation.
    The test CSV is kept completely unseen until final evaluation.
    """
    dataset = GECDataset(
        csv_path=csv_path,
        src_tokenizer=src_tokenizer,
        tgt_tokenizer=tgt_tokenizer,
        max_len=max_len,
        max_samples=max_samples,
    )
    n_val   = max(1, int(len(dataset) * val_split))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(seed)
    )
    print(f"[Split] train={n_train:,}  val={n_val:,}  (val_split={val_split})")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader
