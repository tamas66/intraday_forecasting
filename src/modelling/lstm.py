import torch
import torch.nn as nn
from omegaconf import DictConfig


class LSTMModel(nn.Module):
    """
    Simple univariate LSTM for hourly electricity price forecasting.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()

        lstm_cfg = cfg.model.lstm.architecture

        self.lstm = nn.LSTM(
            input_size=lstm_cfg.input_size,
            hidden_size=lstm_cfg.hidden_size,
            num_layers=lstm_cfg.num_layers,
            dropout=lstm_cfg.dropout if lstm_cfg.num_layers > 1 else 0.0,
            batch_first=True,
        )

        self.fc = nn.Linear(lstm_cfg.hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape (batch_size, sequence_length, input_size)

        Returns
        -------
        torch.Tensor
            Shape (batch_size,)
        """
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        out = self.fc(last_hidden)
        return out.squeeze(-1)
