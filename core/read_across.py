from __future__ import annotations

import csv
import gzip
import json
import math
import pickle
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

from core.logging_utils import configure_logging, get_logger
from core.read_across_sqlite import DEFAULT_CATEGORY_TABLE, DEFAULT_TARGET_TABLES, fetch_table_rows
from core.utils import app_cache_path, resource_path

configure_logging()
logger = get_logger("read_across")


@dataclass(frozen=True)
class ReadAcrossTargetConfig:
    key: str
    label_ru: str
    prediction_task: str
    table: str
    dataset: str
    smiles_col: str
    value_col: str
    mode: str
    enabled: bool
    unit: str
    value_map: dict[str, str]
    exclude_values: tuple[str, ...]


@dataclass(frozen=True)
class ReadAcrossConfig:
    version: str
    database: str
    category_table: str
    category_dataset: str
    category_smiles_col: str
    category_value_col: str
    fingerprint_radius: int
    fingerprint_bits: int
    top_k: int
    min_similarity: float
    fallback_similarity: float
    weight_power: float
    cache_file: str
    targets: list[ReadAcrossTargetConfig]


def _safe_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except Exception:
        return None
    if not math.isfinite(out):
        return None
    return out


def _resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return resource_path(path_str)


def _optional_resolve_path(path_str: str) -> Optional[Path]:
    if not str(path_str).strip():
        return None
    return _resolve_path(path_str)


def _local_override_path(path_str: str) -> Path:
    path = _resolve_path(path_str)
    if path.suffix != ".json":
        return path.with_name(f"{path.name}.local")
    return path.with_name(f"{path.stem}.local{path.suffix}")


def _merge_config(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(merged.get(key), dict) and isinstance(value, dict):
            merged[key] = _merge_config(merged[key], value)
        else:
            merged[key] = value
    return merged


def _canonicalize_smiles(smiles: str) -> Optional[str]:
    text = (smiles or "").replace("\xa0", " ").strip()
    text = re.sub(r"_x([0-9A-Fa-f]{4})_", lambda m: chr(int(m.group(1), 16)), text)
    mol = Chem.MolFromSmiles(text)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


class ReadAcrossService:
    _MEMORY_CACHE: ClassVar[dict[str, dict[str, Any]]] = {}

    def __init__(self, config_path: str = "config/read_across.json"):
        self.config_path = config_path
        self.config = self._load_config(config_path)
        self.cache_file = app_cache_path(self.config.cache_file)

        self._loaded = False
        self._targets: dict[str, dict[str, Any]] = {}
        self._category_map: dict[str, str] = {}
        self._warnings: list[str] = []
        logger.info(
            "ReadAcrossService initialized: config=%s cache=%s targets=%s",
            _resolve_path(self.config_path),
            self.cache_file,
            len(self.config.targets),
        )

    def analyze(self, mol: Chem.Mol, *, meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        meta = meta or {}
        self._ensure_loaded()

        query_smiles = meta.get("smiles") or Chem.MolToSmiles(mol, canonical=True)
        canonical_smiles = _canonicalize_smiles(str(query_smiles))
        logger.info("Read-across analyze started: query=%s canonical=%s", query_smiles, canonical_smiles)
        if canonical_smiles is None:
            warning = "Метод аналогов недоступен: структура не нормализована."
            logger.warning("Read-across aborted: %s", warning)
            return {
                "targets": {},
                "predictions": [],
                "analogues": [],
                "category": {
                    "type": "invalid_query_structure",
                    "members": [],
                    "consistency_score": 0.0,
                    "summary_ru": warning,
                },
                "warnings": list(self._warnings) + [warning],
            }

        query_fp = AllChem.GetMorganFingerprintAsBitVect(
            mol,
            self.config.fingerprint_radius,
            nBits=self.config.fingerprint_bits,
        )

        target_results: dict[str, dict[str, Any]] = {}
        prediction_rows: list[dict[str, Any]] = []

        for target_cfg in self.config.targets:
            if not target_cfg.enabled:
                continue

            bundle = self._targets.get(target_cfg.key) or {}
            entries = bundle.get("entries", [])
            fingerprints = bundle.get("fingerprints", [])
            if not entries or not fingerprints:
                logger.warning(
                    "Read-across target skipped: key=%s entries=%s fingerprints=%s",
                    target_cfg.key,
                    len(entries),
                    len(fingerprints),
                )
                continue

            similarities = DataStructs.BulkTanimotoSimilarity(query_fp, fingerprints)
            hits = self._select_hits(similarities, entries)
            analogues = self._format_analogues(hits, target_cfg)
            category = self._build_category(analogues, canonical_smiles)
            prediction = self._build_prediction(target_cfg, analogues, category)
            logger.info(
                "Read-across target processed: key=%s analogues=%s prediction=%s",
                target_cfg.key,
                len(analogues),
                bool(prediction),
            )

            target_result = {
                "label_ru": target_cfg.label_ru,
                "prediction": prediction,
                "analogues": analogues,
                "category": category,
            }
            target_results[target_cfg.key] = target_result
            if prediction:
                prediction_rows.append(prediction)

        primary_target = self._select_primary_target(target_results)
        primary_analogues: list[dict[str, Any]] = []
        primary_category: dict[str, Any] = {
            "type": "analogue_data_unavailable",
            "members": [],
            "consistency_score": 0.0,
            "summary_ru": "Подходящие аналоги не найдены.",
        }
        if primary_target:
            primary_analogues = list(target_results[primary_target].get("analogues", []) or [])
            primary_category = dict(target_results[primary_target].get("category", {}) or primary_category)
        logger.info(
            "Read-across analyze finished: primary_target=%s primary_analogues=%s warnings=%s",
            primary_target,
            len(primary_analogues),
            len(self._warnings),
        )

        return {
            "targets": target_results,
            "predictions": prediction_rows,
            "analogues": primary_analogues,
            "category": primary_category,
            "warnings": list(self._warnings),
        }

    def clear_cache(self) -> None:
        cache_key = str(_resolve_path(self.config_path).resolve())
        self._MEMORY_CACHE.pop(cache_key, None)
        self._loaded = False
        self._targets = {}
        self._category_map = {}
        self._warnings = []
        if self.cache_file.exists():
            self.cache_file.unlink()

    def cache_info(self) -> dict[str, Any]:
        exists = self.cache_file.exists()
        return {
            "path": str(self.cache_file),
            "exists": exists,
            "size_bytes": self.cache_file.stat().st_size if exists else 0,
        }

    def _load_config(self, config_path: str) -> ReadAcrossConfig:
        path = _resolve_path(config_path)
        with open(path, "r", encoding="utf-8") as fh:
            cfg = json.load(fh)

        local_override = _local_override_path(config_path)
        if local_override.is_file():
            with open(local_override, "r", encoding="utf-8") as fh:
                cfg = _merge_config(cfg, json.load(fh))
            logger.info("Read-across config override loaded: %s", local_override)

        targets = [
            ReadAcrossTargetConfig(
                key=str(item.get("key", "")),
                label_ru=str(item.get("label_ru", item.get("key", ""))),
                prediction_task=str(item.get("prediction_task", "")),
                table=str(item.get("table", DEFAULT_TARGET_TABLES.get(str(item.get("key", "")), ""))),
                dataset=str(item.get("dataset", "")),
                smiles_col=str(item.get("smiles_col", "SMILES")),
                value_col=str(item.get("value_col", "")),
                mode=str(item.get("mode", "regression")),
                enabled=bool(item.get("enabled", True)),
                unit=str(item.get("unit", "")),
                value_map={str(k): str(v) for k, v in (item.get("value_map", {}) or {}).items()},
                exclude_values=tuple(str(x) for x in (item.get("exclude_values", []) or [])),
            )
            for item in cfg.get("targets", [])
            if item.get("key")
        ]

        return ReadAcrossConfig(
            version=str(cfg.get("version", "ra-v2")),
            database=str(cfg.get("database", "")),
            category_table=str(cfg.get("category_table", DEFAULT_CATEGORY_TABLE if cfg.get("database") else "")),
            category_dataset=str(cfg.get("category_dataset", "")),
            category_smiles_col=str(cfg.get("category_smiles_col", "SMILES")),
            category_value_col=str(cfg.get("category_value_col", "Class")),
            fingerprint_radius=int(cfg.get("fingerprint_radius", 2)),
            fingerprint_bits=int(cfg.get("fingerprint_bits", 2048)),
            top_k=int(cfg.get("top_k", 5)),
            min_similarity=float(cfg.get("min_similarity", 0.35)),
            fallback_similarity=float(cfg.get("fallback_similarity", 0.2)),
            weight_power=float(cfg.get("weight_power", 2.0)),
            cache_file=str(cfg.get("cache_file", "read_across_cache.pkl.gz")),
            targets=targets,
        )

    def _ensure_loaded(self) -> None:
        if self._loaded:
            logger.info("Read-across cache already loaded in memory.")
            return

        cache_key = str(_resolve_path(self.config_path).resolve())
        cached = self._MEMORY_CACHE.get(cache_key)
        if cached is not None:
            self._targets = cached["targets"]
            self._category_map = cached["category_map"]
            self._warnings = list(cached["warnings"])
            self._loaded = True
            logger.info("Read-across loaded from process memory cache.")
            return

        metadata = self._build_source_metadata()
        disk_cache = self._load_disk_cache(metadata)
        if disk_cache is not None:
            self._targets = disk_cache["targets"]
            self._category_map = disk_cache["category_map"]
            self._warnings = list(disk_cache["warnings"])
            self._MEMORY_CACHE[cache_key] = disk_cache
            self._loaded = True
            logger.info("Read-across loaded from disk cache: %s", self.cache_file)
            return

        logger.info("Read-across rebuilding cache from source files.")
        warnings: list[str] = []
        category_map = self._load_category_map(warnings)
        targets = self._load_targets(category_map, warnings)

        payload = {
            "metadata": metadata,
            "targets": targets,
            "category_map": category_map,
            "warnings": list(warnings),
        }
        self._save_disk_cache(payload)
        self._MEMORY_CACHE[cache_key] = payload
        self._targets = targets
        self._category_map = category_map
        self._warnings = warnings
        self._loaded = True
        logger.info(
            "Read-across cache rebuilt: category_entries=%s targets=%s warnings=%s",
            len(category_map),
            len(targets),
            len(warnings),
        )

    def _build_source_metadata(self) -> dict[str, Any]:
        files: dict[str, dict[str, Any]] = {}

        def add_file(key: str, path_str: str) -> None:
            if not path_str:
                files[key] = {
                    "path": "",
                    "size": -1,
                    "mtime_ns": -1,
                }
                return
            path = _resolve_path(path_str)
            if path.exists():
                stat = path.stat()
                files[key] = {
                    "path": str(path.resolve()),
                    "size": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                }
            else:
                files[key] = {
                    "path": str(path),
                    "size": -1,
                    "mtime_ns": -1,
                }

        if self.config.database:
            add_file("database", self.config.database)
        else:
            add_file("category_dataset", self.config.category_dataset)
            for target in self.config.targets:
                if not target.enabled:
                    continue
                add_file(f"target:{target.key}", target.dataset)

        return {
            "version": self.config.version,
            "fingerprint_radius": self.config.fingerprint_radius,
            "fingerprint_bits": self.config.fingerprint_bits,
            "database_tables": {
                "category_table": self.config.category_table,
                "targets": {
                    target.key: target.table for target in self.config.targets if target.enabled
                },
            },
            "files": files,
        }

    def _load_disk_cache(self, metadata: dict[str, Any]) -> Optional[dict[str, Any]]:
        if not self.cache_file.exists():
            return None
        try:
            with gzip.open(self.cache_file, "rb") as fh:
                payload = pickle.load(fh)
        except Exception:
            logger.exception("Failed to load read-across disk cache: %s", self.cache_file)
            return None
        if payload.get("metadata") != metadata:
            logger.info("Read-across disk cache invalidated by metadata change: %s", self.cache_file)
            return None
        return payload

    def _save_disk_cache(self, payload: dict[str, Any]) -> None:
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(self.cache_file, "wb") as fh:
            pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("Read-across disk cache saved: %s", self.cache_file)

    def _load_rows_from_database(
        self,
        db_path: Path,
        table_name: str,
        *,
        warning_label: str,
        warnings: list[str],
    ) -> list[dict[str, Any]]:
        if not table_name:
            warnings.append(f"Таблица SQLite для {warning_label} не настроена.")
            return []
        if not db_path.is_file():
            warnings.append(f"Файл SQLite для метода аналогов не найден: {db_path}")
            return []
        try:
            rows = fetch_table_rows(db_path, table_name)
        except Exception:
            logger.exception("Failed to load read-across table: db=%s table=%s", db_path, table_name)
            warnings.append(f"Не удалось загрузить таблицу SQLite «{table_name}» для {warning_label}.")
            return []
        logger.info(
            "Read-across SQLite table loaded: label=%s table=%s rows=%s db=%s",
            warning_label,
            table_name,
            len(rows),
            db_path,
        )
        return rows

    def _load_category_map(self, warnings: list[str]) -> dict[str, str]:
        db_path = _optional_resolve_path(self.config.database)
        if db_path is not None:
            rows = self._load_rows_from_database(
                db_path,
                self.config.category_table,
                warning_label="категорий метода аналогов",
                warnings=warnings,
            )
            if not rows:
                return {}

            class_votes: dict[str, list[str]] = defaultdict(list)
            for row in rows:
                canonical = _canonicalize_smiles(str(row.get(self.config.category_smiles_col, "")))
                class_name = str(row.get(self.config.category_value_col, "")).strip()
                if not canonical or not class_name:
                    continue
                class_votes[canonical].append(class_name)

            out: dict[str, str] = {}
            for canonical, values in class_votes.items():
                out[canonical] = Counter(values).most_common(1)[0][0]
            return out

        path = _optional_resolve_path(self.config.category_dataset)
        if path is None:
            logger.info("Read-across category dataset is not configured.")
            return {}
        if not path.is_file():
            warnings.append(f"Файл категорий для метода аналогов не найден: {path}")
            return {}

        class_votes: dict[str, list[str]] = defaultdict(list)
        with open(path, "r", encoding="utf-8-sig", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                canonical = _canonicalize_smiles(str(row.get(self.config.category_smiles_col, "")))
                class_name = str(row.get(self.config.category_value_col, "")).strip()
                if not canonical or not class_name:
                    continue
                class_votes[canonical].append(class_name)

        out: dict[str, str] = {}
        for canonical, values in class_votes.items():
            out[canonical] = Counter(values).most_common(1)[0][0]
        return out

    def _load_targets(
        self,
        category_map: dict[str, str],
        warnings: list[str],
    ) -> dict[str, dict[str, Any]]:
        bundles: dict[str, dict[str, Any]] = {}
        db_path = _optional_resolve_path(self.config.database)
        for target_cfg in self.config.targets:
            if not target_cfg.enabled:
                continue

            grouped: dict[str, list[Any]] = defaultdict(list)
            source_label = ""
            rows: list[dict[str, Any]]
            if db_path is not None:
                rows = self._load_rows_from_database(
                    db_path,
                    target_cfg.table,
                    warning_label=f"задачи «{target_cfg.label_ru}»",
                    warnings=warnings,
                )
                source_label = str(db_path)
            else:
                path = _optional_resolve_path(target_cfg.dataset)
                if path is None:
                    warnings.append(f"Файл для задачи «{target_cfg.label_ru}» не настроен.")
                    bundles[target_cfg.key] = {"entries": [], "fingerprints": []}
                    logger.warning("Read-across dataset is not configured for %s", target_cfg.key)
                    continue
                if not path.is_file():
                    warnings.append(f"Файл для задачи «{target_cfg.label_ru}» не найден: {path}")
                    bundles[target_cfg.key] = {"entries": [], "fingerprints": []}
                    logger.warning("Read-across dataset missing for %s: %s", target_cfg.key, path)
                    continue
                with open(path, "r", encoding="utf-8-sig", newline="") as fh:
                    rows = list(csv.DictReader(fh))
                source_label = str(path)

            for row in rows:
                canonical = _canonicalize_smiles(str(row.get(target_cfg.smiles_col, "")))
                if not canonical:
                    continue

                if target_cfg.mode == "regression":
                    value = _safe_float(row.get(target_cfg.value_col))
                    if value is None:
                        continue
                else:
                    raw_value = str(row.get(target_cfg.value_col, "")).strip()
                    if not raw_value or raw_value in target_cfg.exclude_values:
                        continue
                    value = target_cfg.value_map.get(raw_value, raw_value)
                grouped[canonical].append(value)

            entries: list[dict[str, Any]] = []
            fingerprints: list[Any] = []
            for canonical, values in grouped.items():
                mol = Chem.MolFromSmiles(canonical)
                if mol is None:
                    continue

                fp = AllChem.GetMorganFingerprintAsBitVect(
                    mol,
                    self.config.fingerprint_radius,
                    nBits=self.config.fingerprint_bits,
                )
                entry = {
                    "smiles": canonical,
                    "class_name": category_map.get(canonical, ""),
                    "n_obs": len(values),
                }
                if target_cfg.mode == "regression":
                    entry["value"] = round(sum(float(v) for v in values) / len(values), 4)
                else:
                    label, label_count = Counter(str(v) for v in values).most_common(1)[0]
                    entry["value"] = label
                    entry["label_support"] = round(label_count / len(values), 3)

                entries.append(entry)
                fingerprints.append(fp)

            bundles[target_cfg.key] = {"entries": entries, "fingerprints": fingerprints}
            logger.info(
                "Read-across dataset loaded: key=%s entries=%s source=%s",
                target_cfg.key,
                len(entries),
                source_label,
            )

        return bundles

    def _select_hits(self, similarities: list[float], entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        for index, similarity in enumerate(similarities):
            if similarity < self.config.min_similarity:
                continue
            candidate = dict(entries[index])
            candidate["similarity"] = float(similarity)
            candidate["match_quality"] = "strong"
            candidates.append(candidate)

        if not candidates:
            fallback: list[dict[str, Any]] = []
            ordered = sorted(enumerate(similarities), key=lambda item: item[1], reverse=True)
            for index, similarity in ordered[: self.config.top_k]:
                if similarity < self.config.fallback_similarity:
                    continue
                candidate = dict(entries[index])
                candidate["similarity"] = float(similarity)
                candidate["match_quality"] = "weak"
                fallback.append(candidate)
            candidates = fallback

        candidates.sort(key=lambda item: (item["similarity"], item.get("n_obs", 1)), reverse=True)
        selected = candidates[: self.config.top_k]
        logger.info(
            "Read-across hit selection: candidates=%s selected=%s",
            len(candidates),
            len(selected),
        )
        return selected

    def _format_analogues(
        self,
        hits: list[dict[str, Any]],
        target_cfg: ReadAcrossTargetConfig,
    ) -> list[dict[str, Any]]:
        analogues: list[dict[str, Any]] = []
        for rank, hit in enumerate(hits, start=1):
            analogue = {
                "rank": rank,
                "smiles": hit["smiles"],
                "similarity": round(float(hit["similarity"]), 3),
                "class_name": hit.get("class_name", ""),
                "n_obs": int(hit.get("n_obs", 1)),
                "target_key": target_cfg.key,
                "target_label_ru": target_cfg.label_ru,
                "match_quality": hit.get("match_quality", "strong"),
            }
            if target_cfg.mode == "regression":
                analogue["value"] = round(float(hit["value"]), 3)
            else:
                analogue["value"] = str(hit["value"])
                analogue["label_support"] = hit.get("label_support")
            analogues.append(analogue)
        return analogues

    def _build_category(self, analogues: list[dict[str, Any]], query_smiles: str) -> dict[str, Any]:
        query_class = self._category_map.get(query_smiles, "")
        class_names = [str(item.get("class_name", "")).strip() for item in analogues if item.get("class_name")]
        if not class_names:
            summary = "Категория по аналогам не определена: у ближайших аналогов нет разметки класса."
            if query_class:
                summary += f" Для исходной структуры найден класс «{query_class}»."
            return {
                "type": "class_data_unavailable",
                "members": [],
                "consistency_score": 0.0,
                "summary_ru": summary,
                "query_class": query_class,
            }

        counts = Counter(class_names)
        dominant_class, dominant_count = counts.most_common(1)[0]
        consistency = dominant_count / max(1, len(class_names))
        members = [name for name, _ in counts.most_common()]
        summary = (
            f"По ближайшим аналогам преобладает класс «{dominant_class}» "
            f"({dominant_count} из {len(class_names)} размеченных аналогов)."
        )
        if query_class:
            summary += f" Для исходной структуры в справочнике указан класс «{query_class}»."

        return {
            "type": "pesticide_class_context",
            "members": members,
            "consistency_score": round(consistency, 3),
            "summary_ru": summary,
            "dominant_class": dominant_class,
            "query_class": query_class,
        }

    def _build_prediction(
        self,
        target_cfg: ReadAcrossTargetConfig,
        analogues: list[dict[str, Any]],
        category: dict[str, Any],
    ) -> Optional[dict[str, Any]]:
        if not analogues:
            return None

        if target_cfg.mode == "regression":
            return self._build_regression_prediction(target_cfg, analogues, category)
        return self._build_classification_prediction(target_cfg, analogues)

    def _build_regression_prediction(
        self,
        target_cfg: ReadAcrossTargetConfig,
        analogues: list[dict[str, Any]],
        category: dict[str, Any],
    ) -> dict[str, Any]:
        weights = [max(item["similarity"], 1e-6) ** self.config.weight_power for item in analogues]
        total_weight = sum(weights)
        weighted_value = sum(weight * float(item["value"]) for weight, item in zip(weights, analogues)) / total_weight
        mean_similarity = sum(item["similarity"] for item in analogues) / len(analogues)
        max_similarity = max(item["similarity"] for item in analogues)
        support_score = min(
            1.0,
            (0.55 * max_similarity + 0.45 * mean_similarity)
            * min(1.0, len(analogues) / max(1, self.config.top_k)),
        )

        if support_score >= 0.75:
            label = "Высокая"
        elif support_score >= 0.5:
            label = "Средняя"
        else:
            label = "Низкая"

        dominant_class = category.get("dominant_class")
        weak_hits = sum(1 for item in analogues if item.get("match_quality") == "weak")
        notes = (
            f"Прогноз по аналогам: использовано {len(analogues)} аналогов; "
            f"максимальная похожесть={max_similarity:.2f}, средняя похожесть={mean_similarity:.2f}"
        )
        if dominant_class:
            notes += f"; преобладающий класс={dominant_class}"
        if weak_hits:
            notes += "; использованы слабые аналоги, интерпретировать осторожно"

        return {
            "task": target_cfg.prediction_task,
            "value": round(weighted_value, 3),
            "confidence": f"{label} (аналогов={len(analogues)}, sim={mean_similarity:.2f})",
            "confidence_score": round(support_score, 3),
            "ad_distance": None,
            "ad_threshold": None,
            "ad_ratio": None,
            "ad_score": None,
            "in_domain": None,
            "notes": notes,
        }

    def _build_classification_prediction(
        self,
        target_cfg: ReadAcrossTargetConfig,
        analogues: list[dict[str, Any]],
    ) -> dict[str, Any]:
        votes: dict[str, float] = defaultdict(float)
        for analogue in analogues:
            label = str(analogue.get("value", "")).strip()
            if not label:
                continue
            votes[label] += max(float(analogue["similarity"]), 1e-6) ** self.config.weight_power

        ordered_votes = sorted(votes.items(), key=lambda item: item[1], reverse=True)
        best_label, best_vote = ordered_votes[0]
        total_vote = sum(votes.values()) or 1.0
        confidence_score = best_vote / total_vote
        mean_similarity = sum(item["similarity"] for item in analogues) / len(analogues)

        if confidence_score >= 0.75:
            label = "Высокая"
        elif confidence_score >= 0.5:
            label = "Средняя"
        else:
            label = "Низкая"

        vote_text = ", ".join(f"{name}: {weight:.2f}" for name, weight in ordered_votes[:3])
        weak_hits = sum(1 for item in analogues if item.get("match_quality") == "weak")
        notes = f"Голоса аналогов: {vote_text}"
        if weak_hits:
            notes += "; использованы слабые аналоги"
        return {
            "task": target_cfg.prediction_task,
            "value": best_label,
            "confidence": f"{label} (аналогов={len(analogues)}, sim={mean_similarity:.2f})",
            "confidence_score": round(confidence_score, 3),
            "ad_distance": None,
            "ad_threshold": None,
            "ad_ratio": None,
            "ad_score": None,
            "in_domain": None,
            "notes": notes,
        }

    def _select_primary_target(self, target_results: dict[str, dict[str, Any]]) -> Optional[str]:
        for key in ["logp", "toxicity", "pesticide_class"]:
            if key in target_results and target_results[key].get("analogues"):
                return key
        for key in ["logp", "toxicity", "pesticide_class"]:
            if key in target_results and target_results[key].get("prediction") is not None:
                return key
        if target_results:
            return next(iter(target_results))
        return None
