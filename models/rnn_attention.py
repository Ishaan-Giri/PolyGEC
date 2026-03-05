"""
RNN Encoder-Decoder with Bahdanau (Additive) Attention for GEC.

Architecture:
  Encoder : single-layer GRU (bidirectional)
  Attention: Bahdanau (additive) — aligns each decoder step to encoder outputs
  Decoder : single-layer GRU with attention context

Reference: Bahdanau et al. (2015) "Neural Machine Translation by Jointly
           Learning to Align and Translate." ICLR 2015.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

PAD_IDX = 0


# ══════════════════════════════════════════════════════════════════════════════
#  Bahdanau Attention
# ══════════════════════════════════════════════════════════════════════════════
class BahdanauAttention(nn.Module):
    """
    Additive attention (Bahdanau et al., 2015).

    score(s_t, h_i) = v^T * tanh(W_dec * s_t + W_enc * h_i)

    Args:
        dec_hidden_dim : dimensionality of decoder hidden state
        enc_hidden_dim : dimensionality of encoder hidden state (× 2 if bidirectional)
    """

    def __init__(self, dec_hidden_dim: int, enc_hidden_dim: int):
        super().__init__()
        self.W_dec = nn.Linear(dec_hidden_dim, dec_hidden_dim, bias=False)
        self.W_enc = nn.Linear(enc_hidden_dim, dec_hidden_dim, bias=False)
        self.v = nn.Linear(dec_hidden_dim, 1, bias=False)

    def forward(
        self,
        decoder_hidden: torch.Tensor,  # (batch, dec_hidden)
        encoder_outputs: torch.Tensor, # (batch, src_len, enc_hidden)
        src_mask: Optional[torch.Tensor] = None,  # (batch, src_len) bool, True=pad
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            context : (batch, enc_hidden)  weighted sum of encoder outputs
            weights : (batch, src_len)     attention distribution
        """
        src_len = encoder_outputs.size(1)

        # Expand decoder hidden to (batch, src_len, dec_hidden)
        dec_h = self.W_dec(decoder_hidden).unsqueeze(1).repeat(1, src_len, 1)
        enc_h = self.W_enc(encoder_outputs)  # (batch, src_len, dec_hidden)

        energy = self.v(torch.tanh(dec_h + enc_h)).squeeze(2)  # (batch, src_len)

        if src_mask is not None:
            energy = energy.masked_fill(src_mask, float("-inf"))

        weights = F.softmax(energy, dim=1)                      # (batch, src_len)
        context = torch.bmm(weights.unsqueeze(1), encoder_outputs).squeeze(1)
        return context, weights


# ══════════════════════════════════════════════════════════════════════════════
#  RNN Encoder
# ══════════════════════════════════════════════════════════════════════════════
class RNNEncoder(nn.Module):
    """
    Bidirectional GRU encoder.

    Args:
        vocab_size    : source vocabulary size
        embed_dim     : word embedding dimension
        hidden_dim    : GRU hidden dimension (output is 2×hidden_dim)
        num_layers    : number of GRU layers
        dropout       : dropout between layers (only if num_layers > 1)
        pad_idx       : padding token index
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        dropout: float = 0.3,
        pad_idx: int = PAD_IDX,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.rnn = nn.GRU(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        # Project bidirectional hidden to single hidden for decoder init
        self.fc_hidden = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(
        self, src: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            src : (batch, src_len)
        Returns:
            outputs : (batch, src_len, hidden_dim*2)
            hidden  : (batch, hidden_dim)  — last layer, fwd+bwd combined
        """
        embedded = self.dropout(self.embedding(src))  # (batch, src_len, embed)
        outputs, hidden = self.rnn(embedded)
        # hidden: (num_layers*2, batch, hidden_dim) — take last layer
        # Concatenate forward and backward hidden for last layer
        hidden_fwd = hidden[-2, :, :]  # (batch, hidden_dim)
        hidden_bwd = hidden[-1, :, :]
        hidden_combined = torch.tanh(
            self.fc_hidden(torch.cat([hidden_fwd, hidden_bwd], dim=1))
        )  # (batch, hidden_dim)
        return outputs, hidden_combined


# ══════════════════════════════════════════════════════════════════════════════
#  RNN Decoder
# ══════════════════════════════════════════════════════════════════════════════
class RNNDecoder(nn.Module):
    """
    GRU decoder with Bahdanau attention.

    Args:
        vocab_size    : target vocabulary size
        embed_dim     : word embedding dimension
        hidden_dim    : GRU hidden dimension (must match encoder hidden_dim)
        enc_hidden_dim: encoder output dimension (hidden_dim * 2 for bidir)
        dropout       : dropout rate
        pad_idx       : padding token index
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
        self.rnn = nn.GRU(
            embed_dim + enc_hidden_dim,  # input = embedding + context
            hidden_dim,
            batch_first=True,
        )
        self.fc_out = nn.Linear(hidden_dim + enc_hidden_dim + embed_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        tgt_token: torch.Tensor,       # (batch,) current token
        hidden: torch.Tensor,           # (batch, hidden_dim)
        encoder_outputs: torch.Tensor,  # (batch, src_len, enc_hidden)
        src_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        One decoder step.
        Returns:
            predictions : (batch, vocab_size)  log-probabilities
            hidden      : (batch, hidden_dim)  updated hidden state
            attn_weights: (batch, src_len)
        """
        embedded = self.dropout(self.embedding(tgt_token.unsqueeze(1)))  # (batch,1,embed)
        context, attn_weights = self.attention(hidden, encoder_outputs, src_mask)
        # Combine embedding + context as RNN input
        rnn_input = torch.cat([embedded, context.unsqueeze(1)], dim=2)
        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))
        hidden = hidden.squeeze(0)

        # Final prediction combines all three signals
        prediction = self.fc_out(
            torch.cat([output.squeeze(1), context, embedded.squeeze(1)], dim=1)
        )
        return prediction, hidden, attn_weights


# ══════════════════════════════════════════════════════════════════════════════
#  Seq2Seq Model (ties encoder + decoder)
# ══════════════════════════════════════════════════════════════════════════════
class RNNSeq2Seq(nn.Module):
    """
    Full RNN encoder-decoder with attention for GEC.

    Args:
        src_vocab_size : source vocab size
        tgt_vocab_size : target vocab size
        embed_dim      : embedding dimension
        hidden_dim     : RNN hidden dimension
        enc_layers     : encoder GRU layers
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
        enc_layers: int = 1,
        dropout: float = 0.3,
        sos_idx: int = 1,
        eos_idx: int = 2,
        pad_idx: int = PAD_IDX,
    ):
        super().__init__()
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.pad_idx = pad_idx
        self.tgt_vocab_size = tgt_vocab_size

        self.encoder = RNNEncoder(
            src_vocab_size, embed_dim, hidden_dim, enc_layers, dropout, pad_idx
        )
        enc_hidden_dim = hidden_dim * 2  # bidirectional
        self.decoder = RNNDecoder(
            tgt_vocab_size, embed_dim, hidden_dim, enc_hidden_dim, dropout, pad_idx
        )

    def make_src_mask(self, src: torch.Tensor) -> torch.Tensor:
        """True where src == PAD (positions to ignore in attention)."""
        return src == self.pad_idx  # (batch, src_len)

    def forward(
        self,
        src: torch.Tensor,        # (batch, src_len)
        tgt: torch.Tensor,        # (batch, tgt_len)
        teacher_forcing_ratio: float = 0.5,
    ) -> torch.Tensor:
        """
        Training forward pass with teacher forcing.
        Returns:
            outputs : (batch, tgt_len-1, vocab_size)
        """
        batch_size = src.size(0)
        tgt_len = tgt.size(1)
        device = src.device

        src_mask = self.make_src_mask(src)
        enc_outputs, hidden = self.encoder(src)

        # First decoder input = <sos>
        dec_input = tgt[:, 0]
        outputs = torch.zeros(batch_size, tgt_len - 1, self.tgt_vocab_size, device=device)

        for t in range(1, tgt_len):
            pred, hidden, _ = self.decoder(dec_input, hidden, enc_outputs, src_mask)
            outputs[:, t - 1, :] = pred
            # Teacher forcing
            use_teacher = torch.rand(1).item() < teacher_forcing_ratio
            dec_input = tgt[:, t] if use_teacher else pred.argmax(dim=1)

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
            generated : (batch, max_len) token IDs
        """
        batch_size = src.size(0)
        device = src.device

        src_mask = self.make_src_mask(src)
        enc_outputs, hidden = self.encoder(src)

        dec_input = torch.full((batch_size,), self.sos_idx, dtype=torch.long, device=device)
        generated = []

        for _ in range(max_len):
            pred, hidden, _ = self.decoder(dec_input, hidden, enc_outputs, src_mask)
            dec_input = pred.argmax(dim=1)
            generated.append(dec_input.unsqueeze(1))

        return torch.cat(generated, dim=1)  # (batch, max_len)
