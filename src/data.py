"""Data collection and processing utilities for Na-ion cathode screening."""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from pymatgen.core import Structure, Composition


def classify_cathode_family(formula: str, structure: Structure = None) -> str:
    """Classify a Na-containing compound into cathode family based on composition."""
    comp = Composition(formula)
    elements = set(str(e) for e in comp.elements)

    if "P" in elements and "O" in elements:
        if "F" in elements:
            return "fluorophosphate"
        return "phosphate"
    elif "S" in elements and "O" in elements and "P" not in elements:
        return "sulfate"
    elif "Si" in elements and "O" in elements:
        return "silicate"
    elif "C" in elements and "N" in elements:
        return "prussian_blue"
    elif "F" in elements and "O" not in elements:
        return "fluoride"
    elif "O" in elements:
        if structure is not None:
            sg = structure.get_space_group_info()[1]
            if sg in [166, 194]:  # R-3m, P6_3/mmc
                return "layered_oxide"
            elif sg in [227]:  # Fd-3m
                return "spinel"
        return "oxide"
    else:
        return "other"


def compute_theoretical_capacity(formula: str, working_ion: str = "Na",
                                  n_electrons: int = None) -> float:
    """
    Compute gravimetric theoretical capacity in mAh/g.

    Uses the formula: C = (n * F) / (3.6 * M)
    where n = electrons transferred, F = 96485 C/mol, M = molar mass in g/mol.
    """
    comp = Composition(formula)
    molar_mass = comp.weight  # g/mol

    if n_electrons is None:
        # Estimate from Na content per formula unit
        n_electrons = comp[working_ion]

    if n_electrons == 0 or molar_mass == 0:
        return 0.0

    F = 96485.0  # C/mol
    capacity = (n_electrons * F) / (3.6 * molar_mass)
    return round(capacity, 2)


def compute_element_abundance_score(formula: str) -> float:
    """
    Score based on crustal abundance of elements (0 to 1, higher = more abundant).
    Excludes Na and O from scoring.
    """
    abundance = {
        "Fe": 0.95, "Mn": 0.90, "Ti": 0.85, "Al": 0.95, "Si": 0.95,
        "P": 0.80, "Mg": 0.85, "Ca": 0.90, "K": 0.85, "Zn": 0.70,
        "Cu": 0.60, "Cr": 0.65, "Ni": 0.55, "V": 0.50, "Co": 0.30,
        "Li": 0.40, "Nb": 0.35, "Mo": 0.40, "W": 0.35, "Sn": 0.45,
        "Zr": 0.60, "Y": 0.35, "La": 0.30, "Ce": 0.35, "Sc": 0.25,
        "S": 0.75, "F": 0.60, "Cl": 0.70, "B": 0.50, "N": 0.70,
        "C": 0.80, "Sb": 0.25, "Bi": 0.20, "Ge": 0.25, "Ga": 0.30,
        "In": 0.15, "Te": 0.10, "Se": 0.20, "As": 0.30,
    }
    comp = Composition(formula)
    scores = []
    for el in comp.elements:
        sym = str(el)
        if sym in ("Na", "O"):
            continue
        scores.append(abundance.get(sym, 0.20))

    if not scores:
        return 0.5
    return round(np.mean(scores), 3)


def save_dataset(data: dict, path: str):
    """Save dataset as JSON with numpy type conversion."""
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, Structure):
            return obj.as_dict()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    with open(path, "w") as f:
        json.dump(data, f, default=convert, indent=2)


def load_dataset(path: str) -> dict:
    """Load dataset from JSON."""
    with open(path) as f:
        return json.load(f)
