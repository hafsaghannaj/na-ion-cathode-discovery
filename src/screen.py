"""High-throughput screening pipeline for Na-ion cathode candidates."""

import numpy as np
import pandas as pd
from src.data import classify_cathode_family, compute_theoretical_capacity, compute_element_abundance_score


def screen_candidates(df, voltage_col="predicted_voltage", capacity_col="capacity_mAhg",
                      ehull_col="e_above_hull_eV", voltage_range=(2.0, 4.5),
                      min_capacity=100, max_ehull=0.050):
    """
    Filter candidate materials by voltage, capacity, and stability criteria.

    Returns filtered DataFrame with composite ranking score.
    """
    mask = (
        (df[voltage_col] >= voltage_range[0]) &
        (df[voltage_col] <= voltage_range[1]) &
        (df[capacity_col] >= min_capacity) &
        (df[ehull_col] <= max_ehull)
    )
    filtered = df[mask].copy()

    if len(filtered) == 0:
        return filtered

    # Normalize metrics to [0, 1] for composite scoring
    v_norm = (filtered[voltage_col] - voltage_range[0]) / (voltage_range[1] - voltage_range[0])
    c_norm = (filtered[capacity_col] - min_capacity) / (filtered[capacity_col].max() - min_capacity + 1e-6)
    s_norm = 1.0 - (filtered[ehull_col] / max_ehull)

    # Composite score: weighted combination
    filtered["composite_score"] = 0.35 * v_norm + 0.35 * c_norm + 0.30 * s_norm
    filtered = filtered.sort_values("composite_score", ascending=False)

    return filtered


def rank_candidates(df, top_n=100):
    """Return top N candidates by composite score."""
    ranked = df.nlargest(top_n, "composite_score").copy()
    ranked["rank"] = range(1, len(ranked) + 1)
    return ranked
