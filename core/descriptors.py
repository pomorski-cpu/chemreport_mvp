from __future__ import annotations
from typing import Dict, Any

from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors, Lipinski

def compute_basic_descriptors(mol) -> Dict[str, Any]:
    return {
        "MolWt": float(Descriptors.MolWt(mol)),
        "ExactMolWt": float(Descriptors.ExactMolWt(mol)),
        "cLogP": float(Crippen.MolLogP(mol)),
        "TPSA": float(rdMolDescriptors.CalcTPSA(mol)),
        "HBD": int(Lipinski.NumHDonors(mol)),
        "HBA": int(Lipinski.NumHAcceptors(mol)),
        "RotBonds": int(Lipinski.NumRotatableBonds(mol)),
        "Rings": int(rdMolDescriptors.CalcNumRings(mol)),
        "HeavyAtoms": int(mol.GetNumHeavyAtoms()),
    }
