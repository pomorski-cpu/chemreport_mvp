from __future__ import annotations
import os
import subprocess
import tempfile
from rdkit import Chem

class ChemDrawImportError(RuntimeError):
    pass

def _run_obabel_convert(src_path: str, out_path: str) -> None:
    """
    Convert src_path (cdx/cdxml) -> out_path (mol) using Open Babel CLI.
    Requires 'obabel' available in PATH.
    """
    # obabel -i cdx input.cdx -o mol -O out.mol
    ext = os.path.splitext(src_path)[1].lower().lstrip(".")
    cmd = ["obabel", "-i", ext, src_path, "-o", "mol", "-O", out_path]

    try:
        p = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except FileNotFoundError:
        raise ChemDrawImportError("Open Babel не найден. Установите его и убедитесь, что 'obabel' доступен в PATH.")

    if p.returncode != 0 or not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
        msg = (p.stderr or p.stdout or "").strip()
        raise ChemDrawImportError(f"Преобразование через Open Babel завершилось ошибкой.\n{msg}")

def mol_from_chemdraw(path: str) -> Chem.Mol:
    ext = os.path.splitext(path)[1].lower()
    if ext not in [".cdx", ".cdxml"]:
        raise ChemDrawImportError(f"Неподдерживаемый формат ChemDraw: {ext}")

    with tempfile.TemporaryDirectory() as td:
        out_mol = os.path.join(td, "tmp.mol")
        _run_obabel_convert(path, out_mol)

        mol = Chem.MolFromMolFile(out_mol, sanitize=True, removeHs=True)
        if mol is None:
            # fallback: read as block
            with open(out_mol, "r", encoding="utf-8", errors="ignore") as f:
                block = f.read()
            mol = Chem.MolFromMolBlock(block, sanitize=True, removeHs=True)

        if mol is None:
            raise ChemDrawImportError("Преобразованный MOL-файл не удалось разобрать с помощью RDKit.")
        return mol
