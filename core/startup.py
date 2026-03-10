from __future__ import annotations

import os
import sys


def _append_flag(env_name: str, flag: str) -> None:
    current = os.environ.get(env_name, "").strip()
    if not current:
        os.environ[env_name] = flag
        return
    parts = current.split()
    if flag not in parts:
        os.environ[env_name] = f"{current} {flag}"


def prepare_gui_environment() -> None:
    if sys.platform.startswith("linux") and "QT_QPA_PLATFORM" not in os.environ:
        if os.environ.get("WAYLAND_DISPLAY"):
            os.environ["QT_QPA_PLATFORM"] = "wayland"
        elif os.environ.get("DISPLAY"):
            os.environ["QT_QPA_PLATFORM"] = "xcb"

    disable_webengine_sandbox = False
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        disable_webengine_sandbox = True
    if os.environ.get("CHEMREPORT_DISABLE_QTWEBENGINE_SANDBOX") == "1":
        disable_webengine_sandbox = True

    if disable_webengine_sandbox:
        os.environ.setdefault("QTWEBENGINE_DISABLE_SANDBOX", "1")
        _append_flag("QTWEBENGINE_CHROMIUM_FLAGS", "--no-sandbox")
