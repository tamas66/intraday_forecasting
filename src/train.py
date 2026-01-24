import time
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from torch.optim import Adam
import hydra
from omegaconf import DictConfig

from modelling.ann_model import LSTMModel
from data import load_lstm_data


# ======================================================
# TRAINING
# ======================================================

def train_lstm(
    *,
    cfg: DictConfig,
    target: str,
    save_model: bool = True,
) -> Dict[str, float]:
    """
    Train the LSTM model and return training metadata.
    """

    device = torch.device(
        cfg.model.lstm.runtime.device
        if torch.cuda.is_available()
        else "cpu"
    )

    train_loader, _, _, param_length = load_lstm_data(
        cfg=cfg,
        target=target,
        verbose=True,
    )

    model = LSTMModel(cfg).to(device)
    criterion = nn.MSELoss()
    optimizer = Adam(
        model.parameters(),
        lr=cfg.model.lstm.training.learning_rate,
    )

    model.train()
    start_time = time.time()

    for epoch in range(cfg.model.lstm.training.num_epochs):
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

        if epoch == 0 or (epoch + 1) % 5 == 0:
            print(
                f"[LSTM] Epoch {epoch + 1}/{cfg.model.lstm.training.num_epochs} "
                f"loss={epoch_loss:.6f}"
            )

    training_time = time.time() - start_time

    # --------------------------------------------------
    # SAVE MODEL
    # --------------------------------------------------
    if save_model:
        model_dir = Path(cfg.data.paths.models_dir) / target
        model_dir.mkdir(parents=True, exist_ok=True)

        model_name = (
            f"lstm_"
            f"{param_length}_"
            f"{time.strftime('%Y%m%d_%H%M%S', time.localtime(start_time))}.pt"
        )

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "config": cfg.model.lstm,
                "training_time": training_time,
                "param_length": param_length,
                "model_name": model_name,
            },
            model_dir / model_name,
        )

        print(f"[LSTM] Saved model to: {model_dir / model_name}")

    return {"training_time": training_time}


# ======================================================
# ENTRYPOINT
# ======================================================

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    target = cfg.data.dataframe.target_col

    stats = train_lstm(
        cfg=cfg,
        target=target,
        save_model=True,
    )

    print(f"\n[LSTM] Training time: {stats['training_time']:.2f} seconds")


if __name__ == "__main__":
    main()
