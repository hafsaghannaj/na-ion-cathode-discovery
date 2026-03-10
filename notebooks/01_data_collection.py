# %% [markdown]
# # 01: Data Collection for Na-ion Cathode Screening
#
# This notebook queries the Materials Project database for:
# 1. All Na-containing compounds (structures, stability, band gaps) for screening
# 2. Known Na insertion electrode data (voltages, capacities) for GNN training
#
# **Requirements**: Set your Materials Project API key as an environment variable:
# ```bash
# export MP_API_KEY="your_key_here"
# ```

# %%
import os
import sys
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

sys.path.insert(0, os.path.abspath(".."))
from src.data import classify_cathode_family, compute_theoretical_capacity, compute_element_abundance_score, save_dataset
from src.evaluate import COLORS, STYLE

plt.rcParams.update(STYLE)
print("Libraries loaded.")

# %% [markdown]
# ## 1. Query Na-containing Structures from Materials Project

# %%
from mp_api.client import MPRester

API_KEY = os.environ.get("MP_API_KEY")
if not API_KEY:
    raise ValueError("Set MP_API_KEY environment variable before running this notebook.")

print("Connecting to Materials Project...")

with MPRester(API_KEY) as mpr:
    # Query all Na-containing materials with basic properties
    na_docs = mpr.materials.summary.search(
        elements=["Na"],
        num_elements=(2, 6),  # binary through senary compounds
        fields=[
            "material_id", "formula_pretty", "structure",
            "formation_energy_per_atom", "energy_above_hull",
            "band_gap", "nsites", "symmetry", "volume",
            "is_stable", "decomposition_products"
        ]
    )

print(f"Retrieved {len(na_docs)} Na-containing materials.")

# %%
# Convert to structured records
records = []
for doc in na_docs:
    try:
        formula = doc.formula_pretty
        structure = doc.structure
        family = classify_cathode_family(formula, structure)
        capacity = compute_theoretical_capacity(formula, "Na")
        abundance = compute_element_abundance_score(formula)

        sg_info = doc.symmetry
        sg_symbol = sg_info.symbol if sg_info else "unknown"
        sg_number = sg_info.number if sg_info else 0

        records.append({
            "material_id": str(doc.material_id),
            "formula": formula,
            "family": family,
            "formation_energy_eV": round(doc.formation_energy_per_atom, 4) if doc.formation_energy_per_atom else None,
            "e_above_hull_eV": round(doc.energy_above_hull, 4) if doc.energy_above_hull is not None else None,
            "band_gap_eV": round(doc.band_gap, 3) if doc.band_gap is not None else None,
            "nsites": doc.nsites,
            "spacegroup": sg_symbol,
            "sg_number": sg_number,
            "volume_A3": round(doc.volume, 2) if doc.volume else None,
            "is_stable": doc.is_stable,
            "capacity_mAhg": capacity,
            "abundance_score": abundance,
        })
    except Exception as e:
        continue

df_structures = pd.DataFrame(records)
print(f"Processed {len(df_structures)} structures into DataFrame.")
print(f"Columns: {list(df_structures.columns)}")

# %%
# Save structures DataFrame
df_structures.to_csv("../data/na_structures.csv", index=False)
print(f"Saved to data/na_structures.csv")
df_structures.head(10)

# %% [markdown]
# ## 2. Query Na Insertion Electrode Data (for GNN Training)

# %%
with MPRester(API_KEY) as mpr:
    # Query battery insertion electrode data for Na working ion
    battery_docs = mpr.materials.insertion_electrodes.search(
        working_ion="Na",
        fields=[
            "battery_id", "material_ids", "formula_charge",
            "formula_discharge", "average_voltage", "capacity_grav",
            "energy_grav", "max_voltage", "min_voltage",
            "num_steps", "working_ion"
        ]
    )

print(f"Retrieved {len(battery_docs)} Na insertion electrode entries.")

# %%
battery_records = []
for doc in battery_docs:
    try:
        battery_records.append({
            "battery_id": str(doc.battery_id),
            "material_ids": [str(mid) for mid in doc.material_ids] if doc.material_ids else [],
            "formula_charge": doc.formula_charge,
            "formula_discharge": doc.formula_discharge,
            "average_voltage_V": round(doc.average_voltage, 4) if doc.average_voltage else None,
            "capacity_mAhg": round(doc.capacity_grav, 2) if doc.capacity_grav else None,
            "energy_Whkg": round(doc.energy_grav, 2) if doc.energy_grav else None,
            "max_voltage_V": round(doc.max_voltage, 4) if doc.max_voltage else None,
            "min_voltage_V": round(doc.min_voltage, 4) if doc.min_voltage else None,
            "num_steps": doc.num_steps,
        })
    except Exception as e:
        continue

df_battery = pd.DataFrame(battery_records)
print(f"Processed {len(df_battery)} battery entries.")

# %%
df_battery.to_csv("../data/na_battery_electrodes.csv", index=False)
print(f"Saved to data/na_battery_electrodes.csv")
df_battery.head(10)

# %% [markdown]
# ## 3. Also Collect Li-ion Electrode Data (for Transfer Learning)
#
# If the Na-ion training set is small (< 500 entries), we will pretrain
# the GNN on Li-ion voltages and fine-tune on Na-ion.

# %%
with MPRester(API_KEY) as mpr:
    li_battery_docs = mpr.materials.insertion_electrodes.search(
        working_ion="Li",
        fields=[
            "battery_id", "material_ids", "formula_charge",
            "formula_discharge", "average_voltage", "capacity_grav",
            "energy_grav", "num_steps", "working_ion"
        ]
    )

print(f"Retrieved {len(li_battery_docs)} Li insertion electrode entries.")

li_records = []
for doc in li_battery_docs:
    try:
        li_records.append({
            "battery_id": str(doc.battery_id),
            "material_ids": [str(mid) for mid in doc.material_ids] if doc.material_ids else [],
            "formula_charge": doc.formula_charge,
            "formula_discharge": doc.formula_discharge,
            "average_voltage_V": round(doc.average_voltage, 4) if doc.average_voltage else None,
            "capacity_mAhg": round(doc.capacity_grav, 2) if doc.capacity_grav else None,
            "energy_Whkg": round(doc.energy_grav, 2) if doc.energy_grav else None,
            "num_steps": doc.num_steps,
        })
    except Exception as e:
        continue

df_li_battery = pd.DataFrame(li_records)
df_li_battery.to_csv("../data/li_battery_electrodes.csv", index=False)
print(f"Saved {len(df_li_battery)} Li-ion entries to data/li_battery_electrodes.csv")
print(f"\nTraining data summary:")
print(f"  Na-ion electrodes: {len(df_battery)}")
print(f"  Li-ion electrodes: {len(df_li_battery)}")
print(f"  Transfer learning needed: {'Yes' if len(df_battery) < 500 else 'No'}")

# %% [markdown]
# ## 4. Fetch Host Structures for Battery Entries
#
# For GNN training, we need the crystal structures associated with each
# electrode entry. We fetch the charged (desodiated) host structures.

# %%
# Collect unique material IDs from Na battery data
na_mat_ids = set()
for ids in df_battery["material_ids"]:
    if isinstance(ids, list):
        na_mat_ids.update(ids)
    elif isinstance(ids, str):
        import ast
        na_mat_ids.update(ast.literal_eval(ids))

print(f"Unique material IDs from Na battery data: {len(na_mat_ids)}")

# Fetch structures in batches
na_battery_structures = {}
id_list = list(na_mat_ids)

with MPRester(API_KEY) as mpr:
    for i in range(0, len(id_list), 100):
        batch = id_list[i:i+100]
        docs = mpr.materials.summary.search(
            material_ids=batch,
            fields=["material_id", "structure"]
        )
        for doc in docs:
            na_battery_structures[str(doc.material_id)] = doc.structure
        print(f"  Fetched {len(na_battery_structures)}/{len(id_list)} structures...")

print(f"Total battery host structures retrieved: {len(na_battery_structures)}")

# Save as JSON
struct_data = {mid: s.as_dict() for mid, s in na_battery_structures.items()}
with open("../data/na_battery_structures.json", "w") as f:
    json.dump(struct_data, f)
print("Saved to data/na_battery_structures.json")

# %% [markdown]
# ## 5. Exploratory Data Analysis

# %%
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# 5a. Cathode family distribution
family_counts = df_structures["family"].value_counts()
colors_list = [COLORS.get(f, "#adb5bd") for f in family_counts.index]
axes[0, 0].barh(family_counts.index, family_counts.values, color=colors_list)
axes[0, 0].set_xlabel("Count")
axes[0, 0].set_title("Na Compounds by Family")
axes[0, 0].invert_yaxis()

# 5b. Energy above hull distribution
hull_data = df_structures["e_above_hull_eV"].dropna()
axes[0, 1].hist(hull_data[hull_data < 0.2], bins=50, color=COLORS["primary"], edgecolor="white", alpha=0.8)
axes[0, 1].axvline(0.050, color=COLORS["accent"], linestyle=":", lw=2, label="50 meV/atom cutoff")
axes[0, 1].set_xlabel("Energy Above Hull (eV/atom)")
axes[0, 1].set_ylabel("Count")
axes[0, 1].set_title("Thermodynamic Stability Distribution")
axes[0, 1].legend()

# 5c. Band gap distribution
bg_data = df_structures["band_gap_eV"].dropna()
axes[0, 2].hist(bg_data[bg_data < 8], bins=50, color=COLORS["secondary"], edgecolor="white", alpha=0.8)
axes[0, 2].set_xlabel("Band Gap (eV)")
axes[0, 2].set_ylabel("Count")
axes[0, 2].set_title("Band Gap Distribution")

# 5d. Na-ion voltage distribution (known electrodes)
v_data = df_battery["average_voltage_V"].dropna()
axes[1, 0].hist(v_data, bins=40, color=COLORS["success"], edgecolor="white", alpha=0.8)
axes[1, 0].set_xlabel("Average Voltage (V)")
axes[1, 0].set_ylabel("Count")
axes[1, 0].set_title("Na-ion Electrode Voltages")

# 5e. Capacity vs voltage (known electrodes)
mask = df_battery["average_voltage_V"].notna() & df_battery["capacity_mAhg"].notna()
axes[1, 1].scatter(df_battery.loc[mask, "average_voltage_V"],
                    df_battery.loc[mask, "capacity_mAhg"],
                    s=15, alpha=0.5, c=COLORS["primary"])
axes[1, 1].set_xlabel("Average Voltage (V)")
axes[1, 1].set_ylabel("Capacity (mAh/g)")
axes[1, 1].set_title("Voltage vs Capacity (Known Na Electrodes)")

# 5f. Theoretical capacity distribution (all structures)
cap_data = df_structures["capacity_mAhg"][df_structures["capacity_mAhg"] > 0]
axes[1, 2].hist(cap_data[cap_data < 500], bins=50, color=COLORS["warning"], edgecolor="white", alpha=0.8)
axes[1, 2].axvline(100, color=COLORS["accent"], linestyle=":", lw=2, label="100 mAh/g cutoff")
axes[1, 2].set_xlabel("Theoretical Capacity (mAh/g)")
axes[1, 2].set_ylabel("Count")
axes[1, 2].set_title("Capacity Distribution (All Na Compounds)")
axes[1, 2].legend()

plt.suptitle("Na-ion Cathode Data: Exploratory Analysis", fontsize=16, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("../results/fig01_eda.png")
plt.show()

# %% [markdown]
# ## 6. Summary Statistics

# %%
print("=" * 60)
print("DATA COLLECTION SUMMARY")
print("=" * 60)
print(f"\nNa-containing structures:   {len(df_structures):,}")
print(f"  Stable (on hull):         {df_structures['is_stable'].sum():,}")
print(f"  Near-stable (<50 meV):    {(df_structures['e_above_hull_eV'] < 0.050).sum():,}")
print(f"\nCathode families:")
for fam, count in df_structures["family"].value_counts().items():
    print(f"  {fam:20s} {count:5d}")
print(f"\nNa-ion electrodes (training): {len(df_battery):,}")
print(f"  Voltage range:  {df_battery['average_voltage_V'].min():.2f} to {df_battery['average_voltage_V'].max():.2f} V")
print(f"  Mean voltage:   {df_battery['average_voltage_V'].mean():.2f} V")
print(f"\nLi-ion electrodes (transfer): {len(df_li_battery):,}")
print(f"\nBattery host structures:      {len(na_battery_structures):,}")
print(f"\nFiles saved:")
print(f"  data/na_structures.csv")
print(f"  data/na_battery_electrodes.csv")
print(f"  data/li_battery_electrodes.csv")
print(f"  data/na_battery_structures.json")
print(f"  results/fig01_eda.png")
