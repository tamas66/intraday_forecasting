import time
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from torch.optim import Adam

from modelling.ann_model import LSTMModel
from data import load_lstm_data
from modelling.model_config import LSTMConfig
from config import MODELS_DIR


def train_lstm(
    config: LSTMConfig,
    target: str = "level",
    save_model: bool = True,
) -> Dict[str, float]:
    """
    Train the LSTM model and return training metadata.

    Returns
    -------
    dict
        Contains training_time (seconds)
    """

    device = torch.device(
        "cuda" if torch.cuda.is_available() and config.device == "cuda" else "cpu"
    )

    train_loader, _, _ = load_lstm_data(config=config, target=target)

    model = LSTMModel(config).to(device)
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=config.learning_rate)

    model.train()
    start_time = time.time()

    for epoch in range(config.num_epochs):
        epoch_loss = 0.0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        epoch_loss /= len(train_loader)

    training_time = time.time() - start_time

    if save_model:
        model_dir = MODELS_DIR / target
        model_dir.mkdir(parents=True, exist_ok=True)

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "config": config,
                "training_time": training_time,
            },
            model_dir / "lstm.pt",
        )

    return {"training_time": training_time}


if __name__ == "__main__":
    # Optional standalone execution
    config = LSTMConfig()
    stats = train_lstm(config=config)
    print(f"LSTM training time: {stats['training_time']:.2f} seconds")
