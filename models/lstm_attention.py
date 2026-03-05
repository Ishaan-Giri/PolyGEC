"""
LSTM Encoder-Decoder with Bahdanau (Additive) Attention for GEC.

Architecture:
  Encoder : multi-layer bidirectional LSTM
  Attention: Bahdanau (additive) — aligns decoder hidden state to encoder outputs
  Decoder : multi-layer LSTM with attention context injected at each step

LSTM captures long-term dependencies better than vanilla RNN thanks to the
cell state gating mechanism (input, forget, output gates).

Reference:
  Bahdanau et al. (2015) ICLR — attention mechanism
  Cherian & Balakrishnan (2022) — LSTM for GEC
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

PAD_IDX = 0


# ══════════════════════════════════════════════════════════════════════════════
#  Bahdanau Attention  (shared with RNN model, duplicated for independence)
# ══════════════════════════════════════════════════════════════════════════════
class BahdanauAttention(nn.Module):
    """
    Additive attention.
    score(s_t, h_i) = v^T · tanh(W_dec·s_t + W_enc·h_i)
    """

    def __init__(self, dec_hidden_dim: int, enc_hidden_dim: int):
        super().__init__()
        self.W_dec = nn.Linear(dec_hidden_dim, dec_hidden_dim, bias=False)
        self.W_enc = nn.Linear(enc_hidden_dim, dec_hidden_dim, bias=False)
        self.v     = nn.Linear(dec_hidden_dim, 1, bias=False)

    def forward(
        self,
        decoder_hidden: torch.Tensor,   # (batch, dec_hidden)
        encoder_outputs: torch.Tensor,  # (batch, src_len, enc_hidden)
        src_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        src_len = encoder_outputs.size(1)
        dec_h   = self.W_dec(decoder_hidden).unsqueeze(1).repeat(1, src_len, 1)
        enc_h   = self.W_enc(encoder_outputs)
        energy  = self.v(torch.tanh(dec_h + enc_h)).squeeze(2)

        if src_mask is not None:
            energy = energy.masked_fill(src_mask, float("-inf"))

        weights = F.softmax(energy, dim=1)
        context = torch.bmm(weights.unsqueeze(1), encoder_outputs).squeeze(1)
        return context, weights


# ══════════════════════════════════════════════════════════════════════════════
#  LSTM Encoder
# ══════════════════════════════════════════════════════════════════════════════
class LSTMEncoder(nn.Module):
    """
    Bidirectional LSTM encoder.

    Args:
        vocab_size : source vocabulary size
        embed_dim  : embedding dimension
        hidden_dim : LSTM hidden units per direction
        num_layers : stacked LSTM layers
        dropout    : dropout (between layers, ignored when num_layers=1)
        pad_idx    : padding index
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        dropout: float = 0.3,
        pad_idx: int = PAD_IDX,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        # Project bidirectional hidden/cell → single direction for decoder
        self.fc_hidden = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc_cell   = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(
        self, src: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            src : (batch, src_len)
        Returns:
            outputs : (batch, src_len, hidden_dim*2)
            (h, c)  : each (batch, hidden_dim)  — last layer combined
        """
        embedded = self.dropout(self.embedding(src))
        outputs, (hidden, cell) = self.lstm(embedded)

        # Take the last layer's forward + backward states
        h_fwd, h_bwd = hidden[-2], hidden[-1]   # each (batch, hidden_dim)
        c_fwd, c_bwd = cell[-2],   cell[-1]

        h = torch.tanh(self.fc_hidden(torch.cat([h_fwd, h_bwd], dim=1)))
        c = torch.tanh(self.fc_cell(torch.cat([c_fwd, c_bwd], dim=1)))
        return outputs, (h, c)


# ══════════════════════════════════════════════════════════════════════════════
#  LSTM Decoder
# ══════════════════════════════════════════════════════════════════════════════
class LSTMDecoder(nn.Module):
    """
    Single-layer LSTM decoder with Bahdanau attention.

    At each step the decoder receives:
        [embedding ; attention_context]  as LSTM input
    and predicts the next token from:
        [LSTM_output ; context ; embedding]
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        enc_hidden_dim: int,
        dropout: float = 0.3,
        pad_idx: int = PAD_IDX,
    ):
        super().__init__()
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.attention = BahdanauAttention(hidden_dim, enc_hidden_dim)
        self.lstm = nn.LSTM(
            embed_dim + enc_hidden_dim,
            hidden_dim,
            batch_first=True,
        )
        self.fc_out  = nn.Linear(hidden_dim + enc_hidden_dim + embed_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        tgt_token: torch.Tensor,                      # (batch,)
        hidden: torch.Tensor,                          # (batch, hidden_dim)
        cell: torch.Tensor,                            # (batch, hidden_dim)
        encoder_outputs: torch.Tensor,                 # (batch, src_len, enc_hidden)
        src_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        One decoder step.
        Returns:
            prediction  : (batch, vocab_size)
            hidden      : (batch, hidden_dim)
            cell        : (batch, hidden_dim)
            attn_weights: (batch, src_len)
        """
        embedded = self.dropout(self.embedding(tgt_token.unsqueeze(1)))  # (B,1,E)
        context, attn_weights = self.attention(hidden, encoder_outputs, src_mask)

        rnn_input         = torch.cat([embedded, context.unsqueeze(1)], dim=2)
        output, (h, c)    = self.lstm(rnn_input, (hidden.unsqueeze(0), cell.unsqueeze(0)))
        hidden, cell      = h.squeeze(0), c.squeeze(0)

        prediction = self.fc_out(
            torch.cat([output.squeeze(1), context, embedded.squeeze(1)], dim=1)
        )
        return prediction, hidden, cell, attn_weights


# ══════════════════════════════════════════════════════════════════════════════
#  Seq2Seq Model
# ══════════════════════════════════════════════════════════════════════════════
class LSTMSeq2Seq(nn.Module):
    """
    Full LSTM encoder-decoder with attention for GEC.

    Args:
        src_vocab_size : source vocabulary size
        tgt_vocab_size : target vocabulary size
        embed_dim      : embedding dimension
        hidden_dim     : LSTM hidden units
        enc_layers     : encoder LSTM layers
        dropout        : dropout rate
        sos_idx        : <sos> token index
        eos_idx        : <eos> token index
        pad_idx        : <pad> token index
    """

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        embed_dim: int = 256,
        hidden_dim: int = 512,
        enc_layers: int = 2,
        dropout: float = 0.3,
        sos_idx: int = 1,
        eos_idx: int = 2,
        pad_idx: int = PAD_IDX,
    ):
        super().__init__()
        self.sos_idx       = sos_idx
        self.eos_idx       = eos_idx
        self.pad_idx       = pad_idx
        self.tgt_vocab_size = tgt_vocab_size

        self.encoder = LSTMEncoder(
            src_vocab_size, embed_dim, hidden_dim, enc_layers, dropout, pad_idx
        )
        enc_hidden_dim = hidden_dim * 2  # bidirectional
        self.decoder = LSTMDecoder(
            tgt_vocab_size, embed_dim, hidden_dim, enc_hidden_dim, dropout, pad_idx
        )

    def make_src_mask(self, src: torch.Tensor) -> torch.Tensor:
        return src == self.pad_idx  # (batch, src_len)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        teacher_forcing_ratio: float = 0.5,
    ) -> torch.Tensor:
        """
        Training forward pass.
        Returns:
            outputs : (batch, tgt_len-1, tgt_vocab_size)
        """
        batch_size  = src.size(0)
        tgt_len     = tgt.size(1)
        device      = src.device

        src_mask = self.make_src_mask(src)
        enc_outputs, (hidden, cell) = self.encoder(src)

        dec_input = tgt[:, 0]
        outputs   = torch.zeros(batch_size, tgt_len - 1, self.tgt_vocab_size, device=device)

        for t in range(1, tgt_len):
            pred, hidden, cell, _ = self.decoder(
                dec_input, hidden, cell, enc_outputs, src_mask
            )
            outputs[:, t - 1, :] = pred
            use_teacher = torch.rand(1).item() < teacher_forcing_ratio
            dec_input   = tgt[:, t] if use_teacher else pred.argmax(dim=1)

        return outputs

    @torch.no_grad()
    def generate(
        self,
        src: torch.Tensor,
        max_len: int = 128,
    ) -> torch.Tensor:
        """
        Greedy decoding for inference.
        Returns:
            generated : (batch, max_len)
        """
        batch_size = src.size(0)
        device     = src.device

        src_mask = self.make_src_mask(src)
        enc_outputs, (hidden, cell) = self.encoder(src)

        dec_input = torch.full((batch_size,), self.sos_idx, dtype=torch.long, device=device)
        generated = []

        for _ in range(max_len):
            pred, hidden, cell, _ = self.decoder(
                dec_input, hidden, cell, enc_outputs, src_mask
            )
            dec_input = pred.argmax(dim=1)
            generated.append(dec_input.unsqueeze(1))

        return torch.cat(generated, dim=1)
