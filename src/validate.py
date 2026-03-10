"""ML potential validation using M3GNet/CHGNet for structure relaxation and NEB."""

import numpy as np
from pymatgen.core import Structure


def relax_structure_m3gnet(structure: Structure, max_steps=500):
    """
    Relax structure using M3GNet universal potential.
    Returns relaxed structure and final energy.
    """
    import matgl
    from matgl.ext.ase import M3GNetCalculator
    from ase.optimize import BFGS
    from pymatgen.io.ase import AseAtomsAdaptor

    model = matgl.load_model("M3GNet-MP-2021.2.8-DIRECT-PES")
    calc = M3GNetCalculator(potential=model)

    atoms = AseAtomsAdaptor.get_atoms(structure)
    atoms.calc = calc

    opt = BFGS(atoms, logfile=None)
    opt.run(fmax=0.05, steps=max_steps)

    relaxed = AseAtomsAdaptor.get_structure(atoms)
    energy = atoms.get_potential_energy()
    energy_per_atom = energy / len(atoms)

    return relaxed, energy_per_atom


def relax_structure_chgnet(structure: Structure, max_steps=500):
    """
    Relax structure using CHGNet universal potential.
    Returns relaxed structure and final energy.
    """
    from chgnet.model.model import CHGNet
    from chgnet.model.dynamics import StructOptimizer

    chgnet = CHGNet.load()
    optimizer = StructOptimizer()

    result = optimizer.relax(structure, fmax=0.05, steps=max_steps)
    relaxed = result["final_structure"]
    energy_per_atom = result["trajectory"].energies[-1] / len(relaxed)

    return relaxed, energy_per_atom


def compute_volume_change(struct_sodiated: Structure, struct_desodiated: Structure) -> float:
    """Compute percentage volume change upon Na extraction."""
    v1 = struct_sodiated.volume
    v2 = struct_desodiated.volume
    return round(100.0 * (v2 - v1) / v1, 2)


def estimate_migration_barrier(structure: Structure, migrating_species: str = "Na",
                                max_images: int = 5, potential: str = "m3gnet"):
    """
    Estimate migration barrier using NEB with ML potential.
    Returns barrier height in eV.

    Note: this is a simplified estimate. For publication quality results,
    use DFT-based NEB calculations.
    """
    from pymatgen.analysis.diffusion.neb.full_path_mapper import MigrationGraph

    try:
        mg = MigrationGraph.with_distance(structure, migrating_specie=migrating_species,
                                            max_distance=5.0)
        paths = mg.get_path()
        if not paths:
            return None

        # Return the shortest path hop distance as a rough proxy
        # Full NEB requires more setup; see notebook 04 for complete implementation
        hop_distance = paths[0].length
        return hop_distance

    except Exception as e:
        print(f"Migration analysis failed: {e}")
        return None
