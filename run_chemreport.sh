#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_PY=""

find_python() {
  local env_name
  for env_name in ".venv" "venv" ".env" ".venv-linux" ".venv-win"; do
    if [[ -x "$REPO_DIR/$env_name/bin/python" ]]; then
      printf '%s\n' "$REPO_DIR/$env_name/bin/python"
      return 0
    fi
  done

  if command -v python >/dev/null 2>&1; then
    command -v python
    return 0
  fi

  if command -v python3 >/dev/null 2>&1; then
    command -v python3
    return 0
  fi

  return 1
}

if ! VENV_PY="$(find_python)"; then
  echo "Не найден Python для запуска ChemReport. Установите Python и зависимости проекта." >&2
  exit 1
fi

if [[ "$(id -u)" == "0" ]]; then
  export QTWEBENGINE_DISABLE_SANDBOX=1
fi

cd "$SCRIPT_DIR"
exec "$VENV_PY" launch.py
