import torch
import torch.nn as nn
from modelling.model_config import LSTMConfig


class LSTMModel(nn.Module):
    """
    Simple univariate LSTM for hourly electricity price forecasting.
    """

    def __init__(self, config: LSTMConfig):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
            batch_first=True,
        )

        self.fc = nn.Linear(config.hidden_size, 1)

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
