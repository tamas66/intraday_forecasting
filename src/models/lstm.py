# src/models/lstm.py
from __future__ import annotations

from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig


# ======================================================
# CONFIG (MODEL-LEVEL, HYDRA-AGNOSTIC)
# ======================================================

@dataclass
class LSTMConfig:
    input_size_past: int
    input_size_future: int
    hidden_size: int
    num_layers: int
    dropout: float
    quantiles: List[float]
    predict_spike_prob: bool = True


# ======================================================
# MODEL
# ======================================================

class Seq2SeqQuantileLSTM(nn.Module):
    """
    Seq2Seq LSTM with:
      - encoder over past features
      - decoder over known-future features
      - quantile regression output
      - optional spike probability head (per horizon step)
    """

    def __init__(self, cfg: LSTMConfig):
        super().__init__()
        self.cfg = cfg
        self.quantiles = cfg.quantiles
        self.n_q = len(cfg.quantiles)

        # ------------------
        # Encoder
        # ------------------
        self.encoder = nn.LSTM(
            input_size=cfg.input_size_past,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
            batch_first=True,
        )

        # ------------------
        # Decoder
        # ------------------
        self.decoder = nn.LSTM(
            input_size=cfg.input_size_future,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
            batch_first=True,
        )

        # ------------------
        # Heads
        # ------------------
        self.quantile_head = nn.Linear(cfg.hidden_size, self.n_q)

        if cfg.predict_spike_prob:
            self.spike_head = nn.Linear(cfg.hidden_size, 1)
        else:
            self.spike_head = None

    # --------------------------------------------------
    # Forward
    # --------------------------------------------------
    def forward(
        self,
        encoder_x: torch.Tensor,
        decoder_x: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Parameters
        ----------
        encoder_x : (B, L, F_past)
        decoder_x : (B, H, F_future)

        Returns
        -------
        quantiles : (B, H, Q)
        spike_prob : (B, H) or None
        """

        # Encode history
        _, (h, c) = self.encoder(encoder_x)

        # Decode future-known inputs
        dec_out, _ = self.decoder(decoder_x, (h, c))
        # dec_out: (B, H, hidden)

        # Quantiles
        q = self.quantile_head(dec_out)
        # (B, H, Q)

        # Spike probability
        if self.spike_head is not None:
            spike_logits = self.spike_head(dec_out).squeeze(-1)
            spike_prob = torch.sigmoid(spike_logits)
        else:
            spike_prob = None

        return q, spike_prob

    # --------------------------------------------------
    # Inference helper
    # --------------------------------------------------
    @torch.no_grad()
    def predict(
        self,
        encoder_x: torch.Tensor,
        decoder_x: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        self.eval()
        q, spike = self.forward(encoder_x, decoder_x)
        return {
            "quantiles": q,
            "spike_prob": spike,
        }


class Seq2SeqJumpQuantileLSTM(nn.Module):

    def __init__(self, cfg: LSTMConfig):
        super().__init__()
        self.cfg = cfg
        self.quantiles = cfg.quantiles
        self.n_q = len(cfg.quantiles)

        self.encoder = nn.LSTM(
            input_size=cfg.input_size_past,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
            batch_first=True,
        )

        self.decoder = nn.LSTM(
            input_size=cfg.input_size_future,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
            batch_first=True,
        )

        # Smooth quantiles
        self.smooth_head = nn.Linear(cfg.hidden_size, self.n_q)

        # Jump probability
        self.jump_prob_head = nn.Linear(cfg.hidden_size, 1)

        # Jump magnitude (mean)
        self.jump_size_head = nn.Linear(cfg.hidden_size, 1)

    def forward(self, encoder_x, decoder_x):

        _, (h, c) = self.encoder(encoder_x)
        dec_out, _ = self.decoder(decoder_x, (h, c))

        smooth_q = self.smooth_head(dec_out)

        jump_prob = torch.sigmoid(
            self.jump_prob_head(dec_out).squeeze(-1)
        )

        jump_size = self.jump_size_head(dec_out).squeeze(-1)

        return smooth_q, jump_prob, jump_size

# ======================================================
# LOSSES
# ======================================================

def quantile_loss(
    y: torch.Tensor,
    q: torch.Tensor,
    quantiles: List[float],
) -> torch.Tensor:
    """
    Pinball loss for multiple quantiles.

    y: (B, H)
    q: (B, H, Q)
    """
    losses = []
    for i, tau in enumerate(quantiles):
        e = y - q[:, :, i]
        losses.append(torch.max(tau * e, (tau - 1) * e))
    return torch.mean(torch.stack(losses, dim=-1))


def spike_bce_loss(
    spike_prob: torch.Tensor,
    spike_target: torch.Tensor,
) -> torch.Tensor:
    """
    spike_prob: (B, H)
    spike_target: (B, H) binary
    """
    return nn.functional.binary_cross_entropy(spike_prob, spike_target)

def lstm_from_hydra(cfg: DictConfig) -> Seq2SeqQuantileLSTM:
    """
    Build Seq2SeqQuantileLSTM from Hydra config.
    """

    model_cfg = LSTMConfig(
        input_size_past=len(cfg.model.data.past_features),
        input_size_future=len(cfg.model.data.known_future_features),
        hidden_size=cfg.model.architecture.encoder.hidden_size,
        num_layers=cfg.model.architecture.encoder.num_layers,
        dropout=cfg.model.architecture.encoder.dropout,
        quantiles=list(cfg.model.architecture.outputs.quantiles),
        predict_spike_prob=cfg.model.architecture.outputs.spike_probability,
    )

    return Seq2SeqQuantileLSTM(model_cfg)\
    
def jump_lstm_from_hydra(cfg: DictConfig) -> Seq2SeqJumpQuantileLSTM:
    model_cfg = LSTMConfig(
        input_size_past=len(cfg.model.data.past_features),
        input_size_future=len(cfg.model.data.known_future_features),
        hidden_size=cfg.model.architecture.encoder.hidden_size,
        num_layers=cfg.model.architecture.encoder.num_layers,
        dropout=cfg.model.architecture.encoder.dropout,
        quantiles=list(cfg.model.architecture.outputs.quantiles),
        predict_spike_prob=True,  # Always predict jump probability in jump LSTM
    )

    return Seq2SeqJumpQuantileLSTM(model_cfg)