from __future__ import annotations
import re

from pathlib import Path
import sys

CAS_RE = re.compile(r"^\d{2,7}-\d{2}-\d$")

def detect_input_type(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return "empty"
    if CAS_RE.match(t):
        return "cas"
    smiles_chars = set("BCNOFPSIbrclonps[]()=#@+-/\\1234567890.")
    if any(ch in smiles_chars for ch in t) and " " not in t:
        return "smiles"
    return "name"

def resource_path(rel_path: str) -> Path:
    """
    Универсальный путь к ресурсам.
    Работает и в dev, и в PyInstaller (onefile/onedir).
    """
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent.parent))
    return base / rel_path


def app_cache_path(rel_path: str) -> Path:
    """
    Путь для кэша/временных служебных файлов, доступный на запись.
    В dev-режиме кэш хранится в outputs/cache проекта,
    в PyInstaller — в домашней директории пользователя.
    """
    if getattr(sys, "frozen", False):
        base = Path.home() / ".chemreport_mvp" / "cache"
    else:
        base = Path(__file__).resolve().parent.parent / "outputs" / "cache"
    return base / rel_path
