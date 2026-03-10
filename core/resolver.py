from __future__ import annotations
from dataclasses import dataclass
from typing import List

from rdkit import Chem
from rdkit.Chem import inchi

@dataclass
class ResolvedMolecule:
    smiles_input: str
    smiles_canonical: str
    inchi: str
    inchikey: str
    mol: Chem.Mol
    warnings: List[str]
    source: str

class ResolveError(Exception):
    pass

def resolve_from_smiles(smiles: str) -> ResolvedMolecule:
    warnings: List[str] = []
    s = (smiles or "").strip()
    if not s:
        raise ResolveError("Пустой ввод.")

    mol = Chem.MolFromSmiles(s, sanitize=True)
    if mol is None:
        raise ResolveError("RDKit не смог разобрать SMILES.")

    smiles_can = Chem.MolToSmiles(mol, canonical=True)

    try:
        inchi_str = inchi.MolToInchi(mol)
        inchikey = inchi.InchiToInchiKey(inchi_str)
    except Exception:
        inchi_str = ""
        inchikey = ""
        warnings.append("Не удалось вычислить InChI/InChIKey. Возможно, сборка RDKit не содержит поддержки InChI.")

    if "." in smiles_can:
        warnings.append("Обнаружено несколько фрагментов (соль/смесь). Надёжность прогнозов может быть снижена.")

    return ResolvedMolecule(
        smiles_input=s,
        smiles_canonical=smiles_can,
        inchi=inchi_str,
        inchikey=inchikey,
        mol=mol,
        warnings=warnings,
        source="smiles",
    )
