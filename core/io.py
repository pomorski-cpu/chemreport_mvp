from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import pandas as pd
from rdkit import Chem


# ---------------------------
# Reading helpers
# ---------------------------

def read_table(path: str) -> pd.DataFrame:
    """
    Read CSV/XLSX into DataFrame.
    - CSV: tries default encoding, then utf-8-sig
    - XLSX/XLS: uses pandas engine (xlsx needs openpyxl installed)
    """
    ext = os.path.splitext(path)[1].lower()

    if ext == ".csv":
        try:
            return pd.read_csv(path)
        except UnicodeDecodeError:
            return pd.read_csv(path, encoding="utf-8-sig")

    if ext in (".xlsx", ".xls"):
        return pd.read_excel(path)

    raise ValueError(f"Unsupported file type: {ext}")


def detect_input_column(df: pd.DataFrame) -> Optional[str]:
    """
    Detect primary column for input molecules.
    Priority: smiles -> cas -> name
    Returns actual column name in df (preserving original case).
    """
    cols_lower = {c.lower(): c for c in df.columns}

    for key in ("smiles", "cas", "name"):
        if key in cols_lower:
            return cols_lower[key]

    return None


# ---------------------------
# Batch inference
# ---------------------------

def run_batch_smiles(
    df: pd.DataFrame,
    smiles_col: str,
    predictor,
    *,
    keep_invalid: bool = True
) -> pd.DataFrame:
    """
    Batch predict from SMILES column.

    predictor: any object with method predict(mol)->dict, e.g. SVRPredictor
      expected keys (optional): task, value, confidence, ad_score, ad_distance, notes

    Returns: original df + appended result columns.
    """

    results: list[Dict[str, Any]] = []

    for i, raw in enumerate(df[smiles_col].astype(str).fillna("")):
        s = (raw or "").strip()

        if not s:
            results.append(_row_result(status="empty"))
            continue

        mol = Chem.MolFromSmiles(s)
        if mol is None:
            if keep_invalid:
                results.append(_row_result(status="invalid_smiles"))
            else:
                results.append(_row_result(status="skipped_invalid"))
            continue

        try:
            pred = predictor.predict(mol) or {}

            # normalize fields
            results.append({
                "status": "ok",
                "task": pred.get("task", ""),
                "value": pred.get("value", ""),
                "confidence": pred.get("confidence", ""),
                "ad_score": pred.get("ad_score", ""),
                "ad_distance": pred.get("ad_distance", ""),
                "notes": pred.get("notes", ""),
            })
        except Exception as e:
            results.append(_row_result(
                status="predict_error",
                notes=f"{type(e).__name__}: {e}"
            ))

    out = pd.DataFrame(results)

    # keep column order stable and readable
    # attach results at the end
    merged = pd.concat([df.reset_index(drop=True), out], axis=1)
    return merged


def _row_result(
    *,
    status: str,
    task: str = "",
    value: Any = "",
    confidence: str = "",
    ad_score: Any = "",
    ad_distance: Any = "",
    notes: str = ""
) -> Dict[str, Any]:
    return {
        "status": status,
        "task": task,
        "value": value,
        "confidence": confidence,
        "ad_score": ad_score,
        "ad_distance": ad_distance,
        "notes": notes,
    }


# ---------------------------
# Saving helpers
# ---------------------------

def save_table(df: pd.DataFrame, out_path: str) -> str:
    """
    Save results to CSV or XLSX depending on extension.
    Returns resolved saved path.
    """
    ext = os.path.splitext(out_path)[1].lower()

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    if ext == ".csv":
        df.to_csv(out_path, index=False, encoding="utf-8")
        return out_path

    if ext in (".xlsx", ".xls"):
        # requires openpyxl for xlsx
        df.to_excel(out_path, index=False)
        return out_path

    raise ValueError(f"Unsupported output file type: {ext}")


def default_batch_output_path(outputs_dir: str = "outputs", ext: str = ".csv") -> str:
    os.makedirs(outputs_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(outputs_dir, f"batch_results_{ts}{ext}")
