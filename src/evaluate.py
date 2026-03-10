"""Evaluation metrics and visualization utilities."""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Consistent plot styling
STYLE = {
    "font.family": "Arial",
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
}
plt.rcParams.update(STYLE)

COLORS = {
    "primary": "#4361ee",
    "secondary": "#7209b7",
    "accent": "#f72585",
    "success": "#06d6a0",
    "warning": "#ffd166",
    "dark": "#1a1a2e",
    "layered_oxide": "#4361ee",
    "phosphate": "#06d6a0",
    "fluorophosphate": "#f72585",
    "sulfate": "#ffd166",
    "prussian_blue": "#7209b7",
    "oxide": "#4895ef",
    "fluoride": "#ff6b6b",
    "silicate": "#90be6d",
    "spinel": "#43aa8b",
    "other": "#adb5bd",
}


def compute_metrics(y_true, y_pred):
    """Compute regression metrics."""
    return {
        "MAE": round(mean_absolute_error(y_true, y_pred), 4),
        "RMSE": round(np.sqrt(mean_squared_error(y_true, y_pred)), 4),
        "R2": round(r2_score(y_true, y_pred), 4),
        "n_samples": len(y_true),
    }


def parity_plot(y_true, y_pred, title="", color=None, ax=None, save_path=None):
    """Create a parity plot (predicted vs actual)."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    c = color or COLORS["primary"]
    metrics = compute_metrics(y_true, y_pred)

    ax.scatter(y_true, y_pred, alpha=0.5, s=20, c=c, edgecolors="none")

    lims = [
        min(min(y_true), min(y_pred)) - 0.2,
        max(max(y_true), max(y_pred)) + 0.2,
    ]
    ax.plot(lims, lims, "k--", lw=1, alpha=0.5)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Actual Voltage (V)")
    ax.set_ylabel("Predicted Voltage (V)")
    ax.set_title(title)
    ax.set_aspect("equal")

    text = f"MAE: {metrics['MAE']:.3f} V\nRMSE: {metrics['RMSE']:.3f} V\nR\u00b2: {metrics['R2']:.3f}"
    ax.text(0.05, 0.95, text, transform=ax.transAxes, va="top",
            fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    if save_path:
        plt.savefig(save_path)
    return ax


def loss_curves(history, title="", save_path=None):
    """Plot training and validation loss curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    epochs = range(1, len(history["train_loss"]) + 1)
    ax1.plot(epochs, history["train_loss"], label="Train", color=COLORS["primary"])
    ax1.plot(epochs, history["val_loss"], label="Validation", color=COLORS["accent"])
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("MSE Loss")
    ax1.set_title(f"{title} Loss")
    ax1.legend()
    ax1.set_yscale("log")

    ax2.plot(epochs, history["val_mae"], color=COLORS["success"])
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Validation MAE (V)")
    ax2.set_title(f"{title} MAE")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    return fig


def benchmark_table(results: dict, save_path=None):
    """Create and optionally save a benchmark comparison table."""
    import pandas as pd
    rows = []
    for name, metrics in results.items():
        rows.append({"Model": name, "MAE (V)": metrics["MAE"],
                      "RMSE (V)": metrics["RMSE"], "R\u00b2": metrics["R2"]})
    df = pd.DataFrame(rows).sort_values("MAE (V)")
    if save_path:
        df.to_csv(save_path, index=False)
    return df
