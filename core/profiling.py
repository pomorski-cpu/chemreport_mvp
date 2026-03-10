from __future__ import annotations

from typing import Any, Dict, List

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors


_FUNCTIONAL_GROUP_SMARTS: dict[str, str] = {
    "amide": "C(=O)N",
    "ester": "C(=O)O[#6]",
    "carboxyl": "C(=O)[OX2H1]",
    "alcohol": "[OX2H][#6]",
    "phenol": "c[OX2H]",
    "amine": "[NX3;!$(N=*)]",
    "ether": "[OD2]([#6])[#6]",
    "nitro": "[NX3](=O)=O",
    "nitrile": "[CX2]#N",
    "halogen": "[F,Cl,Br,I]",
    "sulfone_like": "S(=O)(=O)[#6]",
}

_ALERT_SMARTS: dict[str, str] = {
    "nitro_alert": "[NX3](=O)=O",
    "isocyanate_alert": "N=C=O",
    "aldehyde_alert": "[CX3H1](=O)[#6]",
    "epoxide_alert": "[OX2r3]1[#6][#6]1",
    "alpha_beta_unsat_carbonyl": "[CX3]=[CX3][CX3](=O)[#6]",
}

_PATTERNS: dict[str, Chem.Mol | None] = {
    key: Chem.MolFromSmarts(smarts)
    for key, smarts in {**_FUNCTIONAL_GROUP_SMARTS, **_ALERT_SMARTS}.items()
}

_GROUP_LABELS_RU: dict[str, str] = {
    "amide": "амидная группа",
    "ester": "эфирная группа",
    "carboxyl": "карбоксильная группа",
    "alcohol": "спиртовая группа",
    "phenol": "фенольная группа",
    "amine": "аминная группа",
    "ether": "простая эфирная связь",
    "nitro": "нитрогруппа",
    "nitrile": "нитрильная группа",
    "halogen": "галогенсодержащий фрагмент",
    "sulfone_like": "сульфонильный фрагмент",
}

_ALERT_LABELS_RU: dict[str, str] = {
    "nitro_alert": "Обнаружен нитросодержащий структурный алерт.",
    "isocyanate_alert": "Обнаружен изоцианатный фрагмент.",
    "aldehyde_alert": "Обнаружена альдегидная функция.",
    "epoxide_alert": "Обнаружен эпоксидный цикл.",
    "alpha_beta_unsat_carbonyl": "Обнаружен альфа-бета-ненасыщенный карбонильный фрагмент.",
}

_ELECTROPHILIC_KEYS = {
    "nitro_alert",
    "isocyanate_alert",
    "aldehyde_alert",
    "alpha_beta_unsat_carbonyl",
}


def _has_match(mol: Chem.Mol, key: str) -> bool:
    patt = _PATTERNS.get(key)
    if patt is None:
        return False
    return mol.HasSubstructMatch(patt)


def _find_keys(mol: Chem.Mol, keys: list[str]) -> list[str]:
    return [key for key in keys if _has_match(mol, key)]


def profile_molecule(mol: Chem.Mol) -> Dict[str, Any]:
    functional_groups = _find_keys(mol, list(_FUNCTIONAL_GROUP_SMARTS))
    alerts = _find_keys(mol, list(_ALERT_SMARTS))

    aromatic = rdMolDescriptors.CalcNumAromaticRings(mol) > 0
    halogenated = any(atom.GetAtomicNum() in {9, 17, 35, 53} for atom in mol.GetAtoms())
    electrophilic = any(alert in _ELECTROPHILIC_KEYS for alert in alerts)

    summary_ru: List[str] = []
    if functional_groups:
        groups_ru = ", ".join(_GROUP_LABELS_RU.get(group, group) for group in functional_groups[:4])
        summary_ru.append(f"Обнаружены функциональные группы: {groups_ru}.")
    else:
        summary_ru.append("Выраженные функциональные группы по базовому профилированию не обнаружены.")

    if aromatic:
        summary_ru.append("Обнаружена ароматическая система.")
    if halogenated:
        summary_ru.append("Молекула содержит атомы галогенов.")
    if electrophilic:
        summary_ru.append("В структуре присутствуют потенциально электрофильные центры.")

    for alert in alerts:
        summary_ru.append(_ALERT_LABELS_RU.get(alert, f"Обнаружен структурный алерт: {alert}."))

    if not alerts:
        summary_ru.append("Явные структурные алерты по базовым правилам не выявлены.")

    return {
        "functional_groups": functional_groups,
        "alerts": alerts,
        "aromatic": aromatic,
        "halogenated": halogenated,
        "electrophilic": electrophilic,
        "summary_ru": summary_ru,
    }
