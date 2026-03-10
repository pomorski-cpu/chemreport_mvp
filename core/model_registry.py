# core/model_registry.py
from __future__ import annotations
import json
from typing import Any, Dict

from core.utils import resource_path

class ModelRegistry:
    def __init__(self, registry_path: str = "models/registry.json"):
        with open(resource_path(registry_path), "r", encoding="utf-8") as f:
            self._reg: Dict[str, Any] = json.load(f)

    def get_task_cfg(self, task_key: str) -> Dict[str, Any]:
        tasks = self._reg.get("tasks", {})
        if task_key not in tasks:
            raise KeyError(f"Task '{task_key}' not found in registry.json")
        return tasks[task_key]

    def resolve(self, task_key: str) -> Dict[str, Any]:
        """
        Возвращает нормализованный конфиг активной версии:
        {
          "type": ...,
          "models_dir": ...,
          ... params of active version ...
        }
        """
        cfg = self.get_task_cfg(task_key)

        active = cfg.get("active")
        versions = cfg.get("versions", {})
        if not active or active not in versions:
            raise KeyError(f"Active version not found for task '{task_key}'")

        resolved = {
            "type": cfg["type"],
            "models_dir": cfg.get("models_dir", "models"),
        }
        resolved.update(versions[active])
        resolved["version"] = active
        return resolved