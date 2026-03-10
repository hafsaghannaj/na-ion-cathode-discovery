"""Training loops and utilities."""

import time
import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau


def train_gnn(model, train_loader, val_loader, device, epochs=200, lr=1e-3, patience=20):
    """Train GNN model with early stopping and LR scheduling."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=7, min_lr=1e-6)
    criterion = torch.nn.MSELoss()

    history = {"train_loss": [], "val_loss": [], "val_mae": [], "lr": []}
    best_val_mae = float("inf")
    best_state = None
    wait = 0

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # Train
        model.train()
        train_losses = []
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            pred = model(batch)
            loss = criterion(pred, batch.y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

        # Validate
        model.eval()
        val_losses, val_errors = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                pred = model(batch)
                loss = criterion(pred, batch.y)
                val_losses.append(loss.item())
                val_errors.extend(torch.abs(pred - batch.y).cpu().numpy())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        val_mae = np.mean(val_errors)
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_mae"].append(val_mae)
        history["lr"].append(current_lr)

        scheduler.step(val_loss)

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1

        if epoch % 10 == 0:
            dt = time.time() - t0
            print(f"Epoch {epoch:3d} | Train: {train_loss:.4f} | Val MAE: {val_mae:.4f} V | LR: {current_lr:.1e} | {dt:.1f}s")

        if wait >= patience:
            print(f"Early stopping at epoch {epoch}. Best val MAE: {best_val_mae:.4f} V")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history


def evaluate_model(model, loader, device):
    """Evaluate model and return predictions + targets."""
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch)
            preds.extend(pred.cpu().numpy())
            targets.extend(batch.y.cpu().numpy())
    return np.array(preds), np.array(targets)
