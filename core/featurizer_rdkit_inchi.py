from __future__ import annotations

from functools import lru_cache
from typing import Dict, Any
import math
import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors, AllChem, GraphDescriptors, rdPartialCharges


# --- SMARTS patterns for functional groups ---
_SMARTS: Dict[str, str] = {
    "Fluoro": "[F]",
    "Chloro": "[Cl]",
    "Bromo": "[Br]",
    "Iodo": "[I]",
    "Amino": "[NX3;!$(NC=O)]",
    "Guanidine": "NC(=N)N",
    "Hydroxyl": "[OX2H]",
    "Phenol": "c[OX2H]",
    "Carbonyl": "[CX3]=[OX1]",
    "Carboxyl": "C(=O)[OX2H1]",
    "AminePrimary": "[NX3;H2][CX4]",
    "AmineSecondary": "[NX3;H1]([CX4])[CX4]",
    "AmineTertiary": "[NX3;H0]([CX4])([CX4])[CX4]",
    "Amide": "C(=O)N",
    "Ester": "C(=O)O",
    "Carbamate": "[NX3]C(=O)O[#6]",
    "Thiocarbamate": "[NX3]C(=S)O[#6]",
    "Ether": "[OD2]([#6])[#6]",
    "Thiol": "[SX2H]",
    "Sulfonyl": "S(=O)(=O)",
    "Sulfide": "[SX2]([#6])[#6]",
    "Disulfide": "[SX2][SX2]",
    "Nitro": "[NX3](=O)=O",
    "Nitrile": "[CX2]#N",
    "Alkene": "C=C",
    "Alkyne": "C#C",
    "AromaticRing": "a1aaaaa1",
    "Halogen": "[F,Cl,Br,I]",
    "SulfonicAcid": "S(=O)(=O)[OX2H]",
    "Phosphate": "P(=O)(O)(O)O",
    "Ketone": "[#6][CX3](=O)[#6]",
    "Aldehyde": "[CX3H1](=O)[#6]",
    "Imine": "[CX3]=[NX2]",
    "Cyanate": "OC#N",
    "Isocyanate": "N=C=O",
    "Acetal": "[CX4]([OX2][#6])([OX2][#6])[#6]",
    "Hemiacetal": "[CX4]([OX2H])([OX2][#6])[#6]",
    "Sulfoxide": "S(=O)([#6])[#6]",
    "Thione": "[CX3]=S",
    "Phenyl": "c1ccccc1",
    "Pyrazole": "[nH]1ncc[c,n]1",
    "Pyridine": "n1ccccc1",
    "Pyrimidine": "n1cnccc1",
    "Pyrazine": "n1ccncc1",
    "Ketal": "[CX4]([OX2][#6])([OX2][#6])([#6])[#6]",
}

_SMARTS_MOLS: Dict[str, Chem.Mol] = {k: Chem.MolFromSmarts(v) for k, v in _SMARTS.items()}


def _count_matches(mol: Chem.Mol, patt: Chem.Mol) -> int:
    if patt is None:
        return 0
    return len(mol.GetSubstructMatches(patt))


def _atom_counts(mol: Chem.Mol) -> Dict[str, int]:
    # counts for specific elements
    want = {
        "C", "N", "O", "S", "P", "F", "Cl", "Si", "Br", "Li", "Na", "Zn", "Al", "Sb", "As", "Co", "K", "V",
        "B", "Ca", "Fe", "I", "Cr", "Cu", "Sn", "Se", "Hg", "Ce", "Pb", "Mg", "Mn", "Ni", "Pd", "Mo", "Ti",
        "W", "U",
    }
    out = {e: 0 for e in want}
    for a in mol.GetAtoms():
        sym = a.GetSymbol()
        if sym in out:
            out[sym] += 1
    return out


def _get_coordinate_matrix(mol: Chem.Mol) -> np.ndarray:
    m = Chem.AddHs(Chem.Mol(mol))
    params = AllChem.ETKDGv3()
    params.randomSeed = 0xC0FFEE
    params.useRandomCoords = True
    embed_status = AllChem.EmbedMolecule(m, params)
    if embed_status == 0:
        try:
            AllChem.UFFOptimizeMolecule(m, maxIters=100)
        except Exception:
            pass
        conf = m.GetConformer()
        pts = np.array(
            [[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z] for i in range(m.GetNumAtoms())],
            dtype=float,
        )
        if pts.size:
            return pts

    fallback = Chem.Mol(mol)
    try:
        AllChem.Compute2DCoords(fallback)
        conf = fallback.GetConformer()
        pts = np.array(
            [[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, 0.0] for i in range(fallback.GetNumAtoms())],
            dtype=float,
        )
        if pts.size:
            return pts
    except Exception:
        pass
    return np.zeros((0, 3), dtype=float)


def _compute_pairwise_distance_features(mol: Chem.Mol) -> Dict[str, float]:
    pts = _get_coordinate_matrix(mol)
    if len(pts) < 2:
        return {
            "3D_mean_dist": 0.0,
            "3D_max_dist": 0.0,
            "3D_std_dist": 0.0,
            "3D_pairwise_mean": 0.0,
        }
    distances: list[float] = []
    for i in range(len(pts)):
        delta = pts[i + 1 :] - pts[i]
        if delta.size == 0:
            continue
        distances.extend(np.linalg.norm(delta, axis=1).tolist())
    if not distances:
        return {
            "3D_mean_dist": 0.0,
            "3D_max_dist": 0.0,
            "3D_std_dist": 0.0,
            "3D_pairwise_mean": 0.0,
        }
    arr = np.asarray(distances, dtype=float)
    return {
        "3D_mean_dist": float(arr.mean()),
        "3D_max_dist": float(arr.max()),
        "3D_std_dist": float(arr.std()),
        "3D_pairwise_mean": float(arr.mean()),
    }


def _compute_charge_features(mol: Chem.Mol) -> Dict[str, float]:
    m = Chem.Mol(mol)
    try:
        rdPartialCharges.ComputeGasteigerCharges(m)
    except Exception:
        return {
            "MeanCharge": 0.0,
            "MaxCharge": 0.0,
            "MinCharge": 0.0,
            "ChargeRange": 0.0,
        }
    charges: list[float] = []
    for atom in m.GetAtoms():
        try:
            val = float(atom.GetProp("_GasteigerCharge"))
        except Exception:
            continue
        if math.isfinite(val):
            charges.append(val)
    if not charges:
        return {
            "MeanCharge": 0.0,
            "MaxCharge": 0.0,
            "MinCharge": 0.0,
            "ChargeRange": 0.0,
        }
    return {
        "MeanCharge": float(np.mean(charges)),
        "MaxCharge": float(np.max(charges)),
        "MinCharge": float(np.min(charges)),
        "ChargeRange": float(np.max(charges) - np.min(charges)),
    }


def _compute_2d_geometry(mol: Chem.Mol) -> Dict[str, float]:
    """
    CentroidX/Y and MaxDistance from 2D coordinates.
    If no conformer, creates 2D coords.
    """
    m = Chem.Mol(mol)
    try:
        AllChem.Compute2DCoords(m)
    except Exception:
        # fallback: no coords
        return {"CentroidX": 0.0, "CentroidY": 0.0, "MaxDistance": 0.0, "MaxDist_sq": 0.0}

    conf = m.GetConformer()
    pts = np.array([[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y] for i in range(m.GetNumAtoms())], dtype=float)
    if pts.size == 0:
        return {"CentroidX": 0.0, "CentroidY": 0.0, "MaxDistance": 0.0, "MaxDist_sq": 0.0}

    centroid = pts.mean(axis=0)
    # max pairwise distance (O(n^2) but n is small)
    max_d2 = 0.0
    for i in range(len(pts)):
        d2 = ((pts[i+1:] - pts[i]) ** 2).sum(axis=1) if i+1 < len(pts) else np.array([0.0])
        if d2.size:
            max_d2 = max(max_d2, float(d2.max()))
    max_d = math.sqrt(max_d2) if max_d2 > 0 else 0.0

    return {"CentroidX": float(centroid[0]), "CentroidY": float(centroid[1]), "MaxDistance": float(max_d), "MaxDist_sq": float(max_d2)}


def _compute_feature_row(mol: Chem.Mol) -> Dict[str, Any]:
    row: Dict[str, Any] = {}

    # Core size / rings
    row["MolWeight"] = float(Descriptors.MolWt(mol))
    row["NumAtoms"] = float(mol.GetNumAtoms())
    row["NumBonds"] = float(mol.GetNumBonds())
    row["NumRings"] = float(rdMolDescriptors.CalcNumRings(mol))
    row["NumAromaticRings"] = float(rdMolDescriptors.CalcNumAromaticRings(mol))
    row["BertzCT"] = float(GraphDescriptors.BertzCT(mol))
    row["BalabanJ"] = float(GraphDescriptors.BalabanJ(mol))
    row["HallKierAlpha"] = float(GraphDescriptors.HallKierAlpha(mol))
    row["Kappa1"] = float(GraphDescriptors.Kappa1(mol))
    row["Kappa2"] = float(GraphDescriptors.Kappa2(mol))
    row["Kappa3"] = float(GraphDescriptors.Kappa3(mol))

    # 2D geometry-derived
    row.update(_compute_2d_geometry(mol))
    row.update(_compute_pairwise_distance_features(mol))
    row.update(_compute_charge_features(mol))

    # Functional groups
    for name, patt in _SMARTS_MOLS.items():
        row[name] = float(_count_matches(mol, patt))

    # Element counts
    ac = _atom_counts(mol)
    for k, v in ac.items():
        row[k] = float(v)

    # logP-related (names you used)
    # NOTE: "AlogP" and "XlogP" are often both mapped to MolLogP; keep consistent with your training.
    mol_logp = float(Crippen.MolLogP(mol))
    row["AlogP"] = mol_logp
    row["XlogP"] = mol_logp

    # contribution-like feature (approx; if you used exact contribs in training, скажи — подгоним)
    row["logP_contrib"] = mol_logp

    # Derived ratios/fractions
    num_atoms = max(1.0, row["NumAtoms"])
    row["MW_per_Atom"] = row["MolWeight"] / num_atoms
    row["Ring_per_Atom"] = row["NumRings"] / num_atoms

    c_count = max(1.0, row.get("C", 0.0))
    row["Polar_per_C"] = row.get("O", 0.0) / c_count  # простая прокси полярности

    heavy = float(mol.GetNumHeavyAtoms()) if mol.GetNumHeavyAtoms() else 1.0
    arom_atoms = float(sum(1 for a in mol.GetAtoms() if a.GetIsAromatic()))
    row["AromaticFraction"] = arom_atoms / heavy
    hal = row.get("F", 0.0) + row.get("Cl", 0.0) + row.get("Br", 0.0) + row.get("I", 0.0)
    row["HalogenFraction"] = hal / heavy

    # log transforms
    row["log_MW"] = float(math.log(max(row["MolWeight"], 1e-6)))

    return row


@lru_cache(maxsize=4096)
def _cached_feature_row(smiles: str) -> tuple[tuple[str, Any], ...]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return tuple()
    row = _compute_feature_row(mol)
    return tuple(row.items())


def build_feature_df(mol: Chem.Mol) -> pd.DataFrame:
    try:
        smiles = Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        smiles = ""

    if smiles:
        cached_items = _cached_feature_row(smiles)
        if cached_items:
            return pd.DataFrame([dict(cached_items)])

    return pd.DataFrame([_compute_feature_row(mol)])
