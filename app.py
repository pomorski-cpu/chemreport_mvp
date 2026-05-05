from __future__ import annotations
import base64
import io
import logging
import os, sys
from html import escape

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import json
from datetime import datetime
from core.io import read_table, detect_input_column, save_table, default_batch_output_path

from rdkit import Chem
from rdkit.Chem import Draw
from PySide6.QtCore import Qt,QObject, QThread, Signal,Slot,QUrl, qInstallMessageHandler, QtMsgType, QTimer, QEventLoop, QMarginsF
from PySide6.QtWidgets import (
    QSplitter, QFrame,
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLineEdit, QPushButton, QLabel, QComboBox, QTabWidget,
    QTableWidget, QTableWidgetItem, QTextEdit, QFileDialog, QMessageBox,QProgressDialog,
    QHeaderView
)
from PySide6.QtSvgWidgets import QSvgWidget


from core.utils import detect_input_type
from core.resolver import resolve_from_smiles, ResolveError
from core.render2d import mol_to_svg
from core.report import render_report_html, render_report_pdf, render_batch_table_pdf
from core.decision_support import DecisionSupport
from core.logging_utils import configure_logging, get_logger
from core.read_across import ReadAcrossService
from core.startup import prepare_gui_environment
from core.workflow import DSSWorkflow, PredictorSpec


from core.predictor_factory import PredictorFactory
import threading

configure_logging()
logger = get_logger("app")
_qt_message_handler_installed = False

INPUT_MODE_AUTO = "Авто"
INPUT_MODE_SMILES = "SMILES"
INPUT_MODE_CAS = "CAS"
INPUT_MODE_NAME = "Название"


def _localize_task_name(task: str) -> str:
    mapping = {
        "LogP": "LogP",
        "Toxicity": "Токсичность",
        "Pesticide Class": "Класс пестицида",
        "Bioactivity EC50 Invertebrates": "Биоактивность: EC50, водные беспозвоночные",
        "Bioactivity LC50 Fish": "Биоактивность: LC50, рыбы",
        "Bioactivity LD50 Mammals Oral": "Биоактивность: LD50, млекопитающие перорально",
        "Model": "Модель",
    }
    return mapping.get(str(task), str(task))


def _localize_decision_status(status: str) -> str:
    mapping = {
        "approve": "Одобрить",
        "review": "Проверить вручную",
        "reject": "Отклонить",
        "insufficient_data": "Недостаточно данных",
    }
    return mapping.get(str(status), str(status) if status is not None else "-")


def _localize_risk_level(level: str) -> str:
    mapping = {
        "low": "Низкий",
        "medium": "Средний",
        "high": "Высокий",
        "critical": "Критический",
    }
    return mapping.get(str(level), str(level) if level is not None else "-")


def _display_input_mode_to_internal(mode: str) -> str:
    mapping = {
        INPUT_MODE_AUTO: "auto",
        INPUT_MODE_SMILES: "smiles",
        INPUT_MODE_CAS: "cas",
        INPUT_MODE_NAME: "name",
    }
    return mapping.get(mode, str(mode).lower())


def _localize_source_name(source: str) -> str:
    mapping = {
        "smiles": "SMILES",
        "batch_smiles": "Пакетный SMILES",
        "rule_based_dss": "Правила DSS",
        "torch": "PyTorch",
    }
    return mapping.get(str(source), str(source))


def _localize_payload_for_display(data: Any, *, parent_key: str | None = None) -> Any:
    key_map = {
        "generated_at": "сформирован",
        "meta": "метаданные",
        "descriptors": "дескрипторы",
        "predictions": "прогнозы",
        "warnings": "предупреждения",
        "decision": "решение",
        "profile": "профиль",
        "analogues": "аналоги",
        "category": "категория",
        "read_across": "прогноз_по_аналогам",
        "reliability": "надёжность",
        "svg": "svg",
        "input": "ввод",
        "smiles": "SMILES",
        "inchikey": "InChIKey",
        "source": "источник",
        "task": "задача",
        "value": "значение",
        "confidence": "уверенность",
        "notes": "примечания",
        "confidence_score": "оценка_уверенности",
        "ad_distance": "ad_расстояние",
        "ad_threshold": "ad_порог",
        "ad_ratio": "ad_отношение",
        "ad_score": "ad_оценка",
        "in_domain": "в_области_применимости",
        "decision_status": "статус_решения",
        "risk_level": "уровень_риска",
        "score": "сводный_балл",
        "recommendation": "рекомендация",
        "rationale": "обоснование",
        "next_actions": "следующие_действия",
        "rule_version": "версия_правил",
        "final_label": "итоговая_метка",
        "final_score": "итоговый_балл",
        "summary_ru": "сводка",
        "analogue_support": "поддержка_аналогами",
        "category_consistency": "согласованность_категории",
        "model_confidence": "уверенность_моделей",
        "targets": "цели",
        "label_ru": "русская_метка",
        "prediction": "прогноз",
        "rank": "ранг",
        "similarity": "похожесть",
        "logp": "logp",
        "class_name": "класс",
        "members": "элементы",
        "consistency_score": "оценка_согласованности",
        "dominant_class": "преобладающий_класс",
        "prob_toxic": "вероятность_токсичности",
        "threshold": "порог",
        "decision": "решение",
        "toxicity": "токсичность",
        "label": "метка",
        "medium_cutoff": "средний_порог",
        "high_cutoff": "высокий_порог",
        "toxicity_threshold": "порог_токсичности",
        "toxicity_decision": "решение_по_токсичности",
        "type": "тип",
    }

    if isinstance(data, dict):
        localized = {}
        for key, value in data.items():
            localized[key_map.get(key, key)] = _localize_payload_for_display(value, parent_key=key)
        return localized
    if isinstance(data, list):
        return [_localize_payload_for_display(item, parent_key=parent_key) for item in data]
    if parent_key == "task":
        return _localize_task_name(str(data))
    if parent_key == "decision_status":
        return _localize_decision_status(str(data))
    if parent_key == "risk_level":
        return _localize_risk_level(str(data))
    if parent_key == "source":
        return _localize_source_name(str(data))
    if parent_key in {"in_domain", "toxicity_decision"} and isinstance(data, bool):
        return "да" if data else "нет"
    return data


def _qt_message_handler(msg_type, context, message):
    level_map = {
        QtMsgType.QtDebugMsg: logging.DEBUG,
        QtMsgType.QtInfoMsg: logging.INFO,
        QtMsgType.QtWarningMsg: logging.WARNING,
        QtMsgType.QtCriticalMsg: logging.ERROR,
        QtMsgType.QtFatalMsg: logging.CRITICAL,
    }
    level = level_map.get(msg_type, logging.INFO)
    category = getattr(context, "category", "") or "qt"
    logger.log(level, "Qt[%s] %s", category, message)


def install_qt_logging() -> None:
    global _qt_message_handler_installed
    if _qt_message_handler_installed:
        return
    qInstallMessageHandler(_qt_message_handler)
    _qt_message_handler_installed = True


def log_startup_environment() -> None:
    keys = [
        "DISPLAY",
        "WAYLAND_DISPLAY",
        "XDG_SESSION_TYPE",
        "XDG_CURRENT_DESKTOP",
        "DESKTOP_SESSION",
        "QT_QPA_PLATFORM",
        "QT_PLUGIN_PATH",
        "QT_DEBUG_PLUGINS",
        "QTWEBENGINE_DISABLE_SANDBOX",
    ]
    env_state = {key: os.environ.get(key, "") for key in keys}
    logger.info(
        "Startup environment: platform=%s executable=%s cwd=%s env=%s",
        sys.platform,
        sys.executable,
        os.getcwd(),
        env_state,
    )


class SketcherBridge(QObject):
    def __init__(self, on_smiles_cb):
        super().__init__()
        self._on_smiles_cb = on_smiles_cb

    @Slot(str)
    def submitSmiles(self, smiles: str):
        self._on_smiles_cb(smiles)

    @Slot(str)
    def submitError(self, msg: str):
        logger.error("Sketcher error: %s", msg)


class BatchWorker(QObject):
    progress = Signal(int, int)   # current, total
    finished = Signal(object)     # out_df
    failed = Signal(str)          # error text
    canceled = Signal()


    def __init__(
        self,
        df,
        smiles_col: str,
        predictors: dict,
        decision_support: DecisionSupport | None = None,
        read_across: ReadAcrossService | None = None,
    ):
        super().__init__()
        self.df = df
        self.smiles_col = smiles_col
        self.predictors = predictors
        self.decision_support = decision_support
        self.read_across = read_across
        self._cancel_event = threading.Event()
        self._logger = get_logger("batch")
        specs = [
            PredictorSpec(task="LogP", predictor=self.predictors["SVR"], coverage_name="SVR"),
            PredictorSpec(task="Toxicity", predictor=self.predictors["Tox"], coverage_name="Tox"),
            PredictorSpec(task="Pesticide Class", predictor=self.predictors["Class"], coverage_name="Class"),
        ]
        if "BioEC50" in self.predictors:
            specs.append(PredictorSpec(task='Биоактивность: EC50, водные беспозвоночные', predictor=self.predictors["BioEC50"], coverage_name="BioEC50"))
        if "BioLC50" in self.predictors:
            specs.append(PredictorSpec(task='Биоактивность: LC50, рыбы', predictor=self.predictors["BioLC50"], coverage_name="BioLC50"))
        if "BioLD50" in self.predictors:
            specs.append(PredictorSpec(task='Биоактивность: LD50, млекопитающие перорально', predictor=self.predictors["BioLD50"], coverage_name="BioLD50"))
        if "BioEC50Reg" in self.predictors:
            specs.append(PredictorSpec(task='Биоактивность: EC50 регрессия', predictor=self.predictors["BioEC50Reg"], coverage_name="BioEC50Reg"))
        if "BioLD50Reg" in self.predictors:
            specs.append(PredictorSpec(task='Биоактивность: LD50 регрессия', predictor=self.predictors["BioLD50Reg"], coverage_name="BioLD50Reg"))
        self.workflow = DSSWorkflow(
            specs,
            decision_support=self.decision_support,
            read_across=self.read_across,
        )
        self._logger.info(
            "Batch worker initialized: rows=%s smiles_col=%s predictors=%s",
            len(self.df),
            self.smiles_col,
            ",".join(sorted(self.predictors.keys())),
        )



    @Slot()
    def request_cancel(self):
        self._cancel_event.set()

    def _should_cancel(self) -> bool:
        # 1) наш флаг
        if self._cancel_event.is_set():
            return True
        # 2) штатная “прерывалка” QThread
        th = QThread.currentThread()
        return th.isInterruptionRequested() if th is not None else False

    @Slot()
    def run(self):
        try:
            self._logger.info("Batch processing started.")
            out_df = self._run_batch_smiles_progress(self.df, self.smiles_col)
            if self._should_cancel():
                self._logger.warning("Batch processing canceled before finish.")
                self.canceled.emit()
            else:
                self._logger.info("Batch processing finished: rows=%s", len(out_df))
                self.finished.emit(out_df)
        except Exception as e:
            self._logger.exception("Batch processing failed.")
            self.failed.emit(f"{type(e).__name__}: {e}")

    def _run_batch_smiles_progress(self, df, smiles_col: str):
        import pandas as pd
        from rdkit import Chem

        results = []
        total = len(df)

        for i, raw in enumerate(df[self.smiles_col].astype(str).fillna("")):
            if self._should_cancel():
                self._logger.warning("Batch cancel requested at row %s/%s", i, total)
                break
            if i == 0 or (i + 1) == total or (i + 1) % 25 == 0:
                self._logger.info("Batch progress: %s/%s", i + 1, total)

            row = {"Статус": "Успешно"}
            mol = Chem.MolFromSmiles(raw)
            if mol is None:
                self._logger.warning("Invalid SMILES in batch row %s: %s", i + 1, raw)
                row["Статус"] = "Некорректный SMILES"
                row["Статус решения"] = _localize_decision_status("insufficient_data")
                row["Уровень риска"] = _localize_risk_level("high")
                row["Сводный балл решения"] = ""
                row["Метка надёжности"] = "Низкая"
                row["Сводный балл надёжности"] = ""
                row['Балл опасности DSS'] = ""
                row['Балл неопределённости DSS'] = ""
                row['Ключевые причины DSS'] = 'Некорректный SMILES'
                row['Конфликты DSS'] = ""
                row['AD-статус'] = 'Недоступно'
                row['AD-score'] = ''
                row['AD-сводка'] = 'Некорректный SMILES'
                results.append(row)
                self.progress.emit(i + 1, total)
                continue

            analysis = self.workflow.analyze_molecule(
                mol,
                meta={"input": raw, "smiles": raw, "source": "batch_smiles"},
                warnings=[],
            )
            self._logger.info(
                "Batch row analyzed: row=%s warnings=%s analogues=%s",
                i + 1,
                len(analysis.get("warnings", []) or []),
                len(analysis.get("analogues", []) or []),
            )

            prediction_columns = {
                "LogP": "LogP",
                "Toxicity": "Токсичность",
                "Pesticide Class": "Класс пестицида",
        "Bioactivity EC50 Invertebrates": "Биоактивность: EC50, водные беспозвоночные",
        "Bioactivity LC50 Fish": "Биоактивность: LC50, рыбы",
        "Bioactivity LD50 Mammals Oral": "Биоактивность: LD50, млекопитающие перорально",
            }
            for prediction in analysis["predictions"]:
                raw_task = prediction.get("task", "")
                prefix = prediction_columns.get(raw_task, _localize_task_name(prediction.get("task", "Model")))
                row[f"{prefix}: значение"] = prediction.get("value")
                row[f"{prefix}: уверенность"] = prediction.get("confidence")
                row[f"{prefix}: оценка уверенности"] = prediction.get("confidence_score")
                row[f"{prefix}: модельная уверенность"] = prediction.get("model_confidence")
                row[f"{prefix}: модельная оценка уверенности"] = prediction.get("model_confidence_score")
                row[f"{prefix}: калибровка уверенности"] = prediction.get("confidence_calibration")
                row[f"{prefix}: комментарий"] = prediction.get("notes")
                row[f"{prefix}: в области применимости"] = prediction.get("in_domain")
                row[f"{prefix}: AD-статус"] = prediction.get("ad_status_ru") or prediction.get("ad_status")
                row[f"{prefix}: AD-score"] = prediction.get("ad_score")

            ad_summary = analysis.get("applicability_domain", {}) or {}
            row["AD-статус"] = ad_summary.get("overall_status_ru", ad_summary.get("overall_status", ""))
            row["AD-score"] = ad_summary.get("mean_ad_score", "")
            row["AD-сводка"] = ad_summary.get("summary_ru", "")

            decision = analysis.get("decision", {})
            row["Статус решения"] = _localize_decision_status(decision.get("decision_status", ""))
            row["Уровень риска"] = _localize_risk_level(decision.get("risk_level", ""))
            row["Сводный балл решения"] = decision.get("score", "")
            row['Балл опасности DSS'] = decision.get("hazard_score", "")
            row['Балл неопределённости DSS'] = decision.get("uncertainty_score", "")
            evidence = list(decision.get("evidence", []) or [])
            key_evidence = [
                item for item in evidence
                if item.get("category") in {"hazard", "uncertainty"} and float(item.get("score_delta") or 0) > 0
            ]
            key_evidence = sorted(key_evidence, key=lambda item: float(item.get("score_delta") or 0), reverse=True)
            row['Ключевые причины DSS'] = "; ".join(
                str(item.get("label") or item.get("rationale") or item.get("source"))
                for item in key_evidence[:3]
            )
            row['Конфликты DSS'] = "; ".join(
                str(item.get("message") or item.get("code"))
                for item in (decision.get("conflicts", []) or [])
            )
            row["Метка надёжности"] = analysis["reliability"].get("final_label", "")
            row["Сводный балл надёжности"] = analysis["reliability"].get("final_score", "")

            results.append(row)
            self.progress.emit(i + 1, total)


        out = pd.DataFrame(results)
        merged = pd.concat([df.reset_index(drop=True), out], axis=1)
        return merged



class MainWindow(QMainWindow):
    def __init__(self):
        logger.info("MainWindow __init__ start: before QMainWindow super().__init__()")
        super().__init__()
        logger.info("MainWindow __init__ step: after QMainWindow super().__init__()")
        self._logger = get_logger("app.window")
        self.setWindowTitle("Химический отчёт")
        self.resize(1100, 720)
        self._last_payload = None
        self._logger.info("MainWindow __init__ step: basic window properties set")
        self.factory = PredictorFactory("models/registry.json")
        self.svr = None
        self.tox = None
        self.pesticide_class = None
        self.bioactivity_ec50 = None
        self.bioactivity_lc50 = None
        self.bioactivity_ld50 = None
        self.bioactivity_ec50_regression = None
        self.bioactivity_ld50_regression = None
        self._logger.info("MainWindow __init__ step: predictor factory ready")
        self.decision_support = DecisionSupport("config/decision_rules.json")
        self._logger.info("MainWindow __init__ step: decision support ready")
        self.read_across = ReadAcrossService("config/read_across.json")
        self._logger.info("MainWindow __init__ step: read-across service ready")
        self._analogue_image_cache: dict[str, str] = {}
        self.web = None
        self._sketcher_ready = False
        self._draw_tab_index = 0
        self._view_tab_index = 1

        # Root
        root = QWidget()
        self.setCentralWidget(root)
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(12, 12, 12, 12)
        root_layout.setSpacing(12)
        self._logger.info("MainWindow __init__ step: root layout created")


        # Top bar
        top = QHBoxLayout()
        top.setSpacing(10)

        self.input_edit = QLineEdit()
        self.input_edit.setPlaceholderText("Введите SMILES (MVP). Пример: CC(=O)Oc1ccccc1C(=O)O")

        self.type_combo = QComboBox()
        self.type_combo.addItems([INPUT_MODE_AUTO, INPUT_MODE_SMILES, INPUT_MODE_CAS, INPUT_MODE_NAME])

        self.btn_generate = QPushButton("Сформировать отчёт")
        self.btn_import = QPushButton("Импорт CSV/XLSX")
        self.btn_export = QPushButton("Экспорт PDF")
        self.btn_export.setEnabled(False)
        self.btn_clear_ra_cache = QPushButton("Очистить кэш аналогов")

        top.addWidget(QLabel("Ввод:"))
        top.addWidget(self.input_edit, 1)
        top.addWidget(self.type_combo)
        top.addWidget(self.btn_import)
        top.addWidget(self.btn_generate)
        top.addWidget(self.btn_export)
        top.addWidget(self.btn_clear_ra_cache)
        root_layout.addLayout(top)


        # Tabs
        self.tabs = QTabWidget()

        # Properties tab
        self.props_table = QTableWidget(0, 2)
        self.props_table.setHorizontalHeaderLabels(["Свойство", "Значение"])
        self.props_table.horizontalHeader().setStretchLastSection(True)
        self.props_table.setAlternatingRowColors(True)
        self.props_table.setShowGrid(False)
        self.props_table.setWordWrap(True)
        self.props_table.verticalHeader().setVisible(True)
        self.props_table.verticalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.props_table.verticalHeader().setDefaultSectionSize(28)
        self.props_table.verticalHeader().setMinimumSectionSize(20)

        props_wrap = QWidget()
        props_layout = QVBoxLayout(props_wrap)
        props_layout.setContentsMargins(10, 10, 10, 10)
        props_layout.addWidget(self.props_table)
        self.tabs.addTab(props_wrap, "Свойства")

        # Predictions tab
        self.pred_table = QTableWidget(0, 4)
        self.pred_table.setHorizontalHeaderLabels(["Задача", "Значение", "Уверенность", "Примечания"])
        self.pred_table.horizontalHeader().setStretchLastSection(True)
        self.pred_table.setAlternatingRowColors(True)
        self.pred_table.setShowGrid(False)
        self.pred_table.setWordWrap(True)
        self.pred_table.verticalHeader().setVisible(True)
        self.pred_table.verticalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.pred_table.verticalHeader().setDefaultSectionSize(34)
        self.pred_table.verticalHeader().setMinimumSectionSize(20)

        pred_wrap = QWidget()
        pred_layout = QVBoxLayout(pred_wrap)
        pred_layout.setContentsMargins(10, 10, 10, 10)
        pred_layout.addWidget(self.pred_table)
        self.tabs.addTab(pred_wrap, "Прогнозы")

        # Decision support tab
        self.decision_text = QTextEdit()
        self.decision_text.setReadOnly(True)
        self.tabs.addTab(self.decision_text, "Решение DSS")

        self.profile_text = QTextEdit()
        self.profile_text.setReadOnly(True)
        self.tabs.addTab(self.profile_text, "Профиль")

        self.reliability_text = QTextEdit()
        self.reliability_text.setReadOnly(True)
        self.tabs.addTab(self.reliability_text, "Надёжность")

        self.applicability_text = QTextEdit()
        self.applicability_text.setReadOnly(True)
        self.tabs.addTab(self.applicability_text, "Область применимости")

        self.analogues_text = QTextEdit()
        self.analogues_text.setReadOnly(True)
        self.tabs.addTab(self.analogues_text, "Аналоги")

        # Warnings tab
        self.warn_text = QTextEdit()
        self.warn_text.setReadOnly(True)
        self.tabs.addTab(self.warn_text, "Предупреждения")

        # Raw JSON tab
        self.raw_json = QTextEdit()
        self.raw_json.setReadOnly(True)
        self.tabs.addTab(self.raw_json, "JSON")

        # Save batch results tab
        batch_wrap = QWidget()
        self.batch_wrap = batch_wrap
        batch_layout = QVBoxLayout(batch_wrap)

        self.btn_save_batch = QPushButton("Сохранить пакетные результаты")
        self.btn_save_batch.setEnabled(False)
        batch_layout.addWidget(self.btn_save_batch)
        self.btn_export_batch_pdf = QPushButton("Экспорт пакетного PDF-отчёта")
        self.btn_export_batch_pdf.setEnabled(False)
        batch_layout.addWidget(self.btn_export_batch_pdf)

        self.batch_table = QTableWidget()
        self.batch_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.batch_table.setAlternatingRowColors(True)
        self.batch_table.setWordWrap(True)
        self.batch_table.verticalHeader().setVisible(True)
        self.batch_table.verticalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.batch_table.verticalHeader().setDefaultSectionSize(30)
        self.batch_table.verticalHeader().setMinimumSectionSize(20)
        batch_layout.addWidget(self.batch_table, 1)


        self.tabs.addTab(batch_wrap, "Пакетные результаты")
        self.btn_save_batch.clicked.connect(self.on_save_batch)
        self.btn_export_batch_pdf.clicked.connect(self.on_export_batch_pdf)
        self.batch_table.cellDoubleClicked.connect(self.on_batch_row_double_clicked)

        # Splitter
        splitter = QSplitter(Qt.Horizontal)
        root_layout.addWidget(splitter, 1)

        # LEFT: structure + identity
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(10)

        left_layout.addWidget(QLabel("Структура (2D)"))

        self.left_tabs = QTabWidget()
        left_layout.addWidget(self.left_tabs, 1)

        # -------- Draw tab --------
        self.draw_wrap = QWidget()
        self.draw_layout = QVBoxLayout(self.draw_wrap)
        self.draw_layout.setContentsMargins(0, 0, 0, 0)
        self.draw_placeholder = QLabel("Редактор структуры будет загружен при открытии вкладки.")
        self.draw_placeholder.setAlignment(Qt.AlignCenter)
        self.draw_layout.addWidget(self.draw_placeholder)
        self.left_tabs.addTab(self.draw_wrap, "Редактор")

        # -------- View tab --------
        view_wrap = QWidget()
        view_layout = QVBoxLayout(view_wrap)
        view_layout.setContentsMargins(0, 0, 0, 0)

        svg_card = QFrame()
        svg_card.setObjectName("SvgCard")
        svg_card_layout = QVBoxLayout(svg_card)
        svg_card_layout.setContentsMargins(12, 12, 12, 12)

        self.svg_widget = QSvgWidget()
        self.svg_widget.setMinimumSize(520, 420)
        svg_card_layout.addWidget(self.svg_widget)

        view_layout.addWidget(svg_card, 1)
        self.left_tabs.addTab(view_wrap, "Просмотр")
        self.left_tabs.setCurrentIndex(self._draw_tab_index)
        self.left_tabs.currentChanged.connect(self._on_left_tab_changed)
        QTimer.singleShot(250, self._ensure_sketcher_ready)
        # self.devtools = QWebEngineView(self)
        # self.web.page().setDevToolsPage(self.devtools.page())
        # self.devtools.show()


        # -------- Identity --------
        left_layout.addWidget(QLabel("Идентификация"))
        self.identity_label = QLabel("Молекула не загружена.")
        self.identity_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        left_layout.addWidget(self.identity_label)

        # RIGHT: tabs
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addWidget(self.tabs)

        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([520, 580])

        # Signals
        self.btn_generate.clicked.connect(self.on_generate)
        self.btn_export.clicked.connect(self.on_export)
        self.input_edit.returnPressed.connect(self.on_generate)
        self.btn_import.clicked.connect(self.on_import)
        self.btn_clear_ra_cache.clicked.connect(self.on_clear_read_across_cache)
        self._logger.info("MainWindow initialized.")

    def _get_logp_predictor(self):
        if self.svr is None:
            self._logger.info("Loading predictor: svr_logp")
            self.svr = self.factory.create("svr_logp")
            self._logger.info("Predictor ready: svr_logp -> %s", type(self.svr).__name__)
        return self.svr

    def _get_tox_predictor(self):
        if self.tox is None:
            self._logger.info("Loading predictor: toxicity")
            self.tox = self.factory.create("toxicity")
            self._logger.info("Predictor ready: toxicity -> %s", type(self.tox).__name__)
        return self.tox

    def _get_pesticide_class_predictor(self):
        if self.pesticide_class is None:
            self._logger.info("Loading predictor: pesticide_class")
            self.pesticide_class = self.factory.create("pesticide_class")
            self._logger.info("Predictor ready: pesticide_class -> %s", type(self.pesticide_class).__name__)
        return self.pesticide_class

    def _get_bioactivity_ec50_predictor(self):
        if self.bioactivity_ec50 is None:
            self._logger.info("Loading predictor: bioactivity_ec50_aquatic_binary")
            self.bioactivity_ec50 = self.factory.create("bioactivity_ec50_aquatic_binary")
            self._logger.info("Predictor ready: bioactivity_ec50_aquatic_binary -> %s", type(self.bioactivity_ec50).__name__)
        return self.bioactivity_ec50

    def _get_bioactivity_lc50_predictor(self):
        if self.bioactivity_lc50 is None:
            self._logger.info("Loading predictor: bioactivity_lc50_fish_binary")
            self.bioactivity_lc50 = self.factory.create("bioactivity_lc50_fish_binary")
            self._logger.info("Predictor ready: bioactivity_lc50_fish_binary -> %s", type(self.bioactivity_lc50).__name__)
        return self.bioactivity_lc50

    def _get_bioactivity_ld50_predictor(self):
        if self.bioactivity_ld50 is None:
            self._logger.info("Loading predictor: bioactivity_ld50_mammals_oral_binary")
            self.bioactivity_ld50 = self.factory.create("bioactivity_ld50_mammals_oral_binary")
            self._logger.info("Predictor ready: bioactivity_ld50_mammals_oral_binary -> %s", type(self.bioactivity_ld50).__name__)
        return self.bioactivity_ld50

    def _get_bioactivity_ec50_regression_predictor(self):
        if self.bioactivity_ec50_regression is None:
            self._logger.info("Loading predictor: bioactivity_ec50_aquatic_regression")
            self.bioactivity_ec50_regression = self.factory.create("bioactivity_ec50_aquatic_regression")
            self._logger.info("Predictor ready: bioactivity_ec50_aquatic_regression -> %s", type(self.bioactivity_ec50_regression).__name__)
        return self.bioactivity_ec50_regression

    def _get_bioactivity_ld50_regression_predictor(self):
        if self.bioactivity_ld50_regression is None:
            self._logger.info("Loading predictor: bioactivity_ld50_mammals_oral_regression")
            self.bioactivity_ld50_regression = self.factory.create("bioactivity_ld50_mammals_oral_regression")
            self._logger.info("Predictor ready: bioactivity_ld50_mammals_oral_regression -> %s", type(self.bioactivity_ld50_regression).__name__)
        return self.bioactivity_ld50_regression

    def _build_workflow(self) -> DSSWorkflow:
        self._logger.info("Building DSS workflow.")
        return DSSWorkflow(
            [
                PredictorSpec(task="LogP", predictor=self._get_logp_predictor(), coverage_name="SVR"),
                PredictorSpec(task="Toxicity", predictor=self._get_tox_predictor(), coverage_name="Tox"),
                PredictorSpec(
                    task="Pesticide Class",
                    predictor=self._get_pesticide_class_predictor(),
                    coverage_name="Class",
                ),
                PredictorSpec(
                    task="Биоактивность: EC50, водные беспозвоночные",
                    predictor=self._get_bioactivity_ec50_predictor(),
                    coverage_name="BioEC50",
                ),
                PredictorSpec(
                    task="Биоактивность: LC50, рыбы",
                    predictor=self._get_bioactivity_lc50_predictor(),
                    coverage_name="BioLC50",
                ),
                PredictorSpec(
                    task="Биоактивность: LD50, млекопитающие перорально",
                    predictor=self._get_bioactivity_ld50_predictor(),
                    coverage_name="BioLD50",
                ),
                PredictorSpec(
                    task="Биоактивность: EC50 регрессия",
                    predictor=self._get_bioactivity_ec50_regression_predictor(),
                    coverage_name="BioEC50Reg",
                ),
                PredictorSpec(
                    task="Биоактивность: LD50 регрессия",
                    predictor=self._get_bioactivity_ld50_regression_predictor(),
                    coverage_name="BioLD50Reg",
                ),
            ],
            decision_support=self.decision_support,
            read_across=self.read_across,
        )

    def _on_left_tab_changed(self, index: int):
        if index == self._draw_tab_index:
            self._ensure_sketcher_ready()

    def _ensure_sketcher_ready(self):
        if self._sketcher_ready:
            return
        self._logger.info("Initializing structure editor.")

        from PySide6.QtWebChannel import QWebChannel
        from PySide6.QtWebEngineCore import QWebEngineSettings
        from PySide6.QtWebEngineWidgets import QWebEngineView

        self.draw_layout.removeWidget(self.draw_placeholder)
        self.draw_placeholder.deleteLater()
        self.draw_placeholder = None

        self.web = QWebEngineView()
        self.draw_layout.addWidget(self.web)

        settings = self.web.settings()
        settings.setAttribute(QWebEngineSettings.LocalContentCanAccessFileUrls, True)
        settings.setAttribute(QWebEngineSettings.LocalContentCanAccessRemoteUrls, True)

        def _from_draw(smiles: str):
            smiles = (smiles or "").strip()
            if not smiles:
                return
            self.input_edit.setText(smiles)
            self.type_combo.setCurrentText("SMILES")
            self.on_generate()
            self.left_tabs.setCurrentIndex(self._view_tab_index)

        self._sketcher_bridge = SketcherBridge(_from_draw)
        self._sketcher_channel = QWebChannel(self.web.page())
        self._sketcher_channel.registerObject("bridge", self._sketcher_bridge)
        self.web.page().setWebChannel(self._sketcher_channel)

        host = os.path.join(os.path.dirname(__file__), "web", "ketcher_host.html")
        self.web.load(QUrl.fromLocalFile(host))
        self._sketcher_ready = True
        self._logger.info("Structure editor ready: %s", host)


    def on_import(self):
        self._logger.info("Batch import requested.")
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Импорт молекул (CSV/XLSX)",
            "",
            "Файлы данных (*.csv *.xlsx *.xls)"
        )
        if not path:
            return
        self._logger.info("Import selected: %s", path)

        try:
            df = read_table(path)
            col = detect_input_column(df)
            self._batch_input_col = col  # чтобы знать, откуда брать smiles

        except Exception as e:
            QMessageBox.critical(self, "Ошибка импорта", str(e))
            return

        if col is None:
            QMessageBox.warning(self, "Не найден входной столбец", "Не найден столбец 'smiles'/'cas'/'name'.")
            return

        if col.lower() != "smiles":
            QMessageBox.information(self, "Ограничение MVP", f"Найден столбец '{col}', но текущая версия поддерживает только SMILES.")
            return

        # ---- Progress dialog (non-blocking UI) ----
        # Progress dialog
        self._progress = QProgressDialog(
            "Выполняется пакетный расчёт…",
            "Отмена",
            0,
            len(df),
            self
        )
        self._progress.setWindowTitle("Пакетная обработка")
        self._progress.setWindowModality(Qt.WindowModal)
        self._progress.setAutoClose(False)
        self._progress.setAutoReset(False)

        self._progress.setValue(0)

        # Thread/Worker
        self._batch_thread = QThread(self)
        self._batch_worker = BatchWorker(
            df,
            col,
            predictors={
                "SVR": self._get_logp_predictor(),
                "Tox": self._get_tox_predictor(),
                "Class": self._get_pesticide_class_predictor(),
                "BioEC50": self._get_bioactivity_ec50_predictor(),
                "BioLC50": self._get_bioactivity_lc50_predictor(),
                "BioLD50": self._get_bioactivity_ld50_predictor(),
                "BioEC50Reg": self._get_bioactivity_ec50_regression_predictor(),
                "BioLD50Reg": self._get_bioactivity_ld50_regression_predictor(),
            },
            decision_support=self.decision_support,
            read_across=self.read_across,
        )
        self._batch_worker.moveToThread(self._batch_thread)


        self._batch_thread.started.connect(self._batch_worker.run)
        self._batch_worker.progress.connect(self._on_batch_progress)
        self._batch_worker.finished.connect(self._on_batch_finished)
        self._batch_worker.failed.connect(self._on_batch_failed)
        self._batch_worker.canceled.connect(self._on_batch_canceled)

        # Cancel wiring (надёжно)


        # cleanup
        self._batch_worker.finished.connect(self._batch_thread.quit)
        self._batch_worker.failed.connect(self._batch_thread.quit)
        self._batch_worker.canceled.connect(self._batch_thread.quit)
        self._batch_thread.finished.connect(self._batch_worker.deleteLater)
        self._batch_thread.finished.connect(self._batch_thread.deleteLater)

        self._batch_thread.start()
        self._progress.canceled.connect(self._cancel_batch)
        self._logger.info("Batch worker thread started: rows=%s", len(df))

    def _on_batch_progress(self, current: int, total: int):
        if hasattr(self, "_progress") and self._progress:
            self._progress.setMaximum(total)
            self._progress.setValue(current)


    def _on_batch_finished(self, out_df):
        # 1) закрыть прогресс
        if hasattr(self, "_progress") and self._progress:
            self._progress.close()
        self._logger.info("Batch finished in UI: rows=%s columns=%s", len(out_df), len(out_df.columns))

        # 2) сохранить результат в памяти
        self._batch_df = out_df

        # 3) показать в таблице и перейти на вкладку Batch Results
        self._show_df_in_table(self._batch_df, self.batch_table, max_rows=500)
        self.tabs.setCurrentWidget(self.batch_wrap)

        # 4) включить явную кнопку сохранения
        if hasattr(self, "btn_save_batch"):
            self.btn_save_batch.setEnabled(True)
        if hasattr(self, "btn_export_batch_pdf"):
            self.btn_export_batch_pdf.setEnabled(True)

        # 5) (опционально) короткое уведомление без диалога сохранения
        QMessageBox.information(
            self,
            "Пакетная обработка завершена",
            f"Обработано строк: {len(out_df)}.\n"
            "Дважды щёлкните по строке, чтобы открыть подробный отчёт.\n"
            "Для экспорта используйте кнопку «Сохранить пакетные результаты»."
        )



    def _on_batch_failed(self, msg: str):
        if hasattr(self, "_progress") and self._progress:
            self._progress.close()
        self._logger.error("Batch failed in UI: %s", msg)
        QMessageBox.critical(self, "Ошибка пакетной обработки", msg)


    def _on_batch_canceled(self):
        if hasattr(self, "_progress") and self._progress:
            self._progress.close()
        self._logger.warning("Batch canceled in UI.")
        QMessageBox.information(self, "Отменено", "Пакетная обработка отменена.")


    def _set_table(self, table: QTableWidget, rows: list[tuple[str, str]]):
        table.setRowCount(0)
        for r, (k, v) in enumerate(rows):
            table.insertRow(r)

            item_k = QTableWidgetItem(str(k))
            item_k.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)

            item_v = QTableWidgetItem(str(v))
            item_v.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)

            table.setItem(r, 0, item_k)
            table.setItem(r, 1, item_v)


    def _set_pred_table(self, preds: list[dict]):
        self.pred_table.setRowCount(0)
        for r, p in enumerate(preds):
            self.pred_table.insertRow(r)

            for c, key in enumerate(["task", "value", "confidence", "notes"]):
                value = p.get(key, "")
                if key == "task":
                    value = _localize_task_name(str(value))
                item = QTableWidgetItem(str(value))
                item.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)
                self.pred_table.setItem(r, c, item)
        self.pred_table.resizeRowsToContents()

    def _render_decision_text(self, decision: dict, analysis: dict | None = None) -> str:
        if not decision:
            return "Сводка по DSS недоступна."

        def fmt_num(value):
            try:
                return f"{float(value):.3f}"
            except Exception:
                return "-" if value in (None, "") else str(value)

        def status_ru(value):
            return {
                "approve": "Одобрить",
                "review": "Проверить вручную",
                "reject": "Отклонить",
                "insufficient_data": "Недостаточно данных",
            }.get(str(value), str(value) if value is not None else "-")

        def risk_ru(value):
            return {
                "low": "Низкий",
                "medium": "Средний",
                "high": "Высокий",
                "critical": "Критический",
            }.get(str(value), str(value) if value is not None else "-")

        def positive_items(kind: str) -> list[dict]:
            return sorted(
                [
                    item for item in (decision.get("evidence", []) or [])
                    if item.get("category") == kind and float(item.get("score_delta") or 0) > 0
                ],
                key=lambda item: float(item.get("score_delta") or 0),
                reverse=True,
            )

        def render_evidence(title: str, items: list[dict]) -> list[str]:
            lines = [f"{title}:"]
            if not items:
                lines.append("-")
                return lines
            for item in items[:5]:
                label = item.get("label") or item.get("source") or "Фактор"
                source = item.get("source") or "-"
                delta = fmt_num(item.get("score_delta"))
                value = item.get("value")
                value_part = f"; значение={value}" if value not in (None, "") else ""
                rationale = item.get("rationale") or ""
                lines.append(f"- {label} ({source}{value_part}; вклад={delta}). {rationale}".strip())
            return lines

        component_names = {
            "toxicity": "токсичность",
            "bioactivity": "биоактивность EC50/LC50/LD50",
            "physchem": "LogP/TPSA",
            "read_across": "read-across",
            "domain": "область применимости",
            "reliability": "надёжность",
            "conflict": "конфликты",
            "data_completeness": "полнота данных",
        }
        component_lines = [
            f"- {component_names.get(key, key)}: {fmt_num(value)}"
            for key, value in (decision.get("component_scores", {}) or {}).items()
            if value
        ]

        conflict_lines = [
            f"- {item.get('message', item.get('code', 'конфликт'))}"
            for item in (decision.get("conflicts", []) or [])
        ] or ["-"]
        flag_lines = [
            f"- {item.get('message', item.get('code', 'флаг качества'))}"
            for item in (decision.get("data_quality_flags", []) or [])
        ] or ["-"]
        rationale_lines = [f"- {item}" for item in decision.get("rationale", [])] or ["-"]
        action_lines = [f"- {item}" for item in decision.get("next_actions", [])] or ["-"]

        lines = [
            f"Версия правил: {decision.get('rule_version', '-')}",
            "Роль DSS: скрининг и объяснение риска, не финальное регуляторное заключение.",
            f"Статус решения: {status_ru(decision.get('decision_status', '-'))}",
            f"Уровень риска: {risk_ru(decision.get('risk_level', '-'))}",
            f"Сводный балл: {fmt_num(decision.get('score'))}",
            f"Балл опасности: {fmt_num(decision.get('hazard_score'))}",
            f"Балл неопределённости: {fmt_num(decision.get('uncertainty_score'))}",
            "",
            "Вклад компонентов:",
            *(component_lines or ["-"]),
            "",
            *render_evidence("Ключевые факторы риска", positive_items("hazard")),
            "",
            *render_evidence("Факторы неопределённости", positive_items("uncertainty")),
            "",
            "Конфликты:",
            *conflict_lines,
            "",
            "Флаги качества данных:",
            *flag_lines,
            "",
            f"Рекомендация: {decision.get('recommendation', '-')}",
            "",
            "Обоснование:",
            *rationale_lines,
            "",
            "Следующие действия:",
            *action_lines,
        ]
        return "\n".join(lines)

    def _render_profile_text(self, profile: dict) -> str:
        lines = list(profile.get("summary_ru", []) or [])
        if not lines:
            return "Структурный профиль недоступен."
        return "\n".join(f"- {line}" for line in lines)

    def _render_reliability_text(self, reliability: dict) -> str:
        if not reliability:
            return "Оценка надёжности недоступна."
        return (
            f"Метка: {reliability.get('final_label', '-')}\n"
            f"Сводный балл: {reliability.get('final_score', '-')}\n"
            f"AD-оценка: {reliability.get('ad_score', '-')}\n"
            f"Поддержка аналогами: {reliability.get('analogue_support', '-')}\n"
            f"Согласованность категории: {reliability.get('category_consistency', '-')}\n"
            f"Уверенность моделей: {reliability.get('model_confidence', '-')}\n\n"
            f"{reliability.get('summary_ru', '-')}"
        )

    def _render_applicability_domain_text(self, ad: dict) -> str:
        if not ad:
            return "Оценка области применимости недоступна."

        lines = [
            f"Сводный статус: {ad.get('overall_status_ru', '-')}",
            f"Средний AD score: {ad.get('mean_ad_score', '-')}",
            "",
            ad.get("summary_ru", "-"),
            "",
            "Детализация по моделям:",
        ]

        for item in ad.get("items", []) or []:
            lines.extend([
                "",
                f"Модель: {item.get('task', '-')}",
                f"Статус: {item.get('status_ru', item.get('status', '-'))}",
                f"AD score: {item.get('ad_score', '-')}",
                f"Метод: {item.get('method', '-')}",
                f"Причина: {item.get('reason', '-')}",
            ])
            distance = item.get("distance") or {}
            if distance:
                lines.append(
                    "kNN: "
                    f"distance={distance.get('distance', '-')}; "
                    f"threshold={distance.get('threshold', '-')}; "
                    f"ratio={distance.get('ratio', '-')}"
                )
            similarity = item.get("similarity") or {}
            if similarity.get("top_similarity") is not None:
                lines.append(f"Ближайший Tanimoto analogue: {similarity.get('top_similarity')}")
            if "scaffold_in_reference" in item:
                lines.append(f"Scaffold в train: {'да' if item.get('scaffold_in_reference') else 'нет'}")

            descriptor_flags = item.get("descriptor_flags") or []
            if descriptor_flags:
                lines.append("Descriptor flags:")
                for flag in descriptor_flags[:5]:
                    lines.append(f"- {flag.get('message', flag.get('code', '-'))}")

            flags = item.get("flags") or []
            if flags:
                lines.append("Molecular flags:")
                for flag in flags[:5]:
                    lines.append(f"- {flag.get('message', flag.get('code', '-'))}")

            nearest = item.get("nearest") or []
            if nearest:
                lines.append("Ближайшие обучающие молекулы:")
                for analogue in nearest[:3]:
                    lines.append(
                        f"- sim={analogue.get('similarity', '-')}; "
                        f"label={analogue.get('label', '-')}; "
                        f"SMILES={analogue.get('smiles', '-')}"
                    )

        return "\n".join(str(line) for line in lines)

    def _render_analogues_html(self, category: dict, read_across: dict) -> str:
        summary = escape((category or {}).get("summary_ru", "Сводка по аналогам недоступна."))
        targets = (read_across or {}).get("targets", {}) or {}
        self._logger.info(
            "Rendering analogues tab: targets=%s summary=%s",
            len(targets),
            (category or {}).get("type", ""),
        )
        blocks = [
            "<html><body style='font-family: Segoe UI, Arial, sans-serif; color: #E6E6E6;'>",
            f"<p><b>Сводка:</b> {summary}</p>",
        ]

        if not targets:
            blocks.append("<p>Подходящие аналоги для метода аналогов не найдены.</p></body></html>")
            return "".join(blocks)

        for target_key, target_data in targets.items():
            prediction = target_data.get("prediction") or {}
            analogues = target_data.get("analogues", []) or []
            self._logger.info(
                "Rendering analogues for target=%s label=%s analogues=%s prediction=%s",
                target_key,
                target_data.get("label_ru", target_key),
                len(analogues),
                bool(prediction),
            )
            blocks.append(
                "<div style='margin: 14px 0; padding: 12px; border: 1px solid #2A2F36; border-radius: 10px;'>"
                f"<h3 style='margin: 0 0 8px 0; color: #FFFFFF;'>{escape(str(target_data.get('label_ru', target_key)))}</h3>"
            )
            if prediction:
                blocks.append(
                    f"<p style='margin: 0 0 10px 0;'><b>Прогноз:</b> {escape(str(prediction.get('value', '-')))}"
                    f" &nbsp;|&nbsp; <b>Уверенность:</b> {escape(str(prediction.get('confidence', '-')))}</p>"
                )
            else:
                blocks.append("<p style='margin: 0 0 10px 0;'><b>Прогноз:</b> не построен</p>")

            if not analogues:
                blocks.append("<p style='margin: 0;'>Аналоги для этой задачи не найдены.</p></div>")
                continue

            for analogue in analogues[:4]:
                image_html = "<div style='color:#B0B7C3;'>Структура недоступна</div>"
                try:
                    analogue_smiles = str(analogue.get("smiles", ""))
                    encoded = self._analogue_image_cache.get(analogue_smiles)
                    if encoded is None:
                        analogue_mol = Chem.MolFromSmiles(analogue_smiles)
                        if analogue_mol is not None:
                            image = Draw.MolToImage(analogue_mol, size=(220, 140))
                            buf = io.BytesIO()
                            image.save(buf, format="PNG")
                            encoded = base64.b64encode(buf.getvalue()).decode("ascii")
                            self._analogue_image_cache[analogue_smiles] = encoded
                        else:
                            self._logger.warning(
                                "Analogue image skipped: invalid SMILES for target=%s smiles=%s",
                                target_key,
                                analogue_smiles,
                            )
                    if encoded:
                        image_html = f"<img width='220' height='140' src='data:image/png;base64,{encoded}'/>"
                except Exception:
                    self._logger.exception(
                        "Analogue image rendering failed for target=%s smiles=%s",
                        target_key,
                        analogue.get("smiles", ""),
                    )

                class_name = escape(str(analogue.get("class_name") or "без класса"))
                quality = "слабый" if analogue.get("match_quality") == "weak" else "сильный"
                blocks.append(
                    "<div style='display:flex; gap:12px; align-items:flex-start; margin:10px 0; "
                    "padding:10px; background:#14171A; border-radius:8px;'>"
                    f"<div style='min-width:220px; background:#FFFFFF; border-radius:6px; padding:4px;'>{image_html}</div>"
                    "<div>"
                    f"<div><b>#{analogue.get('rank', '-')}</b> | sim={analogue.get('similarity', '-')}</div>"
                    f"<div><b>Значение:</b> {escape(str(analogue.get('value', '-')))}</div>"
                    f"<div><b>Тип:</b> {quality}</div>"
                    f"<div><b>Класс:</b> {class_name}</div>"
                    f"<div><b>SMILES:</b> <span style='font-family: monospace;'>{escape(str(analogue.get('smiles', '-')))}</span></div>"
                    "</div></div>"
                )
            blocks.append("</div>")

        blocks.append("</body></html>")
        return "".join(blocks)

    def on_clear_read_across_cache(self):
        info_before = self.read_across.cache_info()
        self.read_across.clear_cache()
        self._logger.info("Read-across cache cleared: %s", info_before)
        QMessageBox.information(
            self,
            "Кэш аналогов очищен",
            f"Путь: {info_before.get('path')}\n"
            f"Размер до очистки: {info_before.get('size_bytes', 0)} байт",
        )


    def on_generate(self):
        text = self.input_edit.text().strip()
        mode = self.type_combo.currentText()
        self._logger.info("Generate requested: mode=%s input=%s", mode, text[:120])

        if not text:
            QMessageBox.warning(self, "Ошибка ввода", "Введите значение.")
            return

        # MVP: поддерживаем только SMILES
        input_type = detect_input_type(text) if mode == INPUT_MODE_AUTO else _display_input_mode_to_internal(mode)
        if input_type != "smiles":
            QMessageBox.information(
                self,
                "Ограничение MVP",
                "Текущая версия поддерживает только ввод SMILES. Поддержку CAS и названий можно добавить через локальный registry.csv."
            )
            return

        # 1) resolve
        try:
            resolved = resolve_from_smiles(text)
        except ResolveError as e:
            self._logger.error("Resolve error: %s", e)
            QMessageBox.critical(self, "Ошибка разбора", str(e))
            return
        self._logger.info(
            "Resolve complete: canonical_smiles=%s inchikey=%s",
            resolved.smiles_canonical,
            resolved.inchikey,
        )

        try:
            svg = mol_to_svg(resolved.mol, width=520, height=420)
        except TypeError:
            svg = mol_to_svg(resolved.mol, size=(520, 420))

        meta = {
            "input": resolved.smiles_input,
            "smiles": resolved.smiles_canonical,
            "inchikey": resolved.inchikey,
            "source": resolved.source,
        }
        analysis = self._build_workflow().analyze_molecule(
            resolved.mol,
            meta=meta,
            warnings=list(resolved.warnings),
            svg=svg,
        )
        self._logger.info(
            "Analysis complete: predictions=%s warnings=%s analogues=%s",
            len(analysis["predictions"]),
            len(analysis["warnings"]),
            len(analysis.get("analogues", []) or []),
        )
        self._last_payload = analysis["payload"]

        self.svg_widget.load(bytearray(svg.encode("utf-8")))
        self.identity_label.setText(
            f"Канонический SMILES: {resolved.smiles_canonical}\n"
            f"InChIKey: {resolved.inchikey or 'нет'}"
        )

        rows = []
        for k, v in analysis["descriptors"].items():
            rows.append((k, f"{v:.4f}" if isinstance(v, float) else str(v)))
        self._set_table(self.props_table, rows)

        self._set_pred_table(analysis["predictions"])
        self.decision_text.setPlainText(self._render_decision_text(analysis["decision"], analysis))
        self.profile_text.setPlainText(self._render_profile_text(analysis["profile"]))
        self.reliability_text.setPlainText(self._render_reliability_text(analysis["reliability"]))
        self.applicability_text.setPlainText(self._render_applicability_domain_text(analysis.get("applicability_domain", {})))
        self.analogues_text.setHtml(
            self._render_analogues_html(
                analysis["category"],
                analysis.get("read_across", {}),
            )
        )
        self.warn_text.setPlainText(
            "\n".join(analysis["warnings"]) if analysis["warnings"] else "Предупреждения отсутствуют."
        )
        self.raw_json.setPlainText(
            json.dumps(_localize_payload_for_display(self._last_payload), ensure_ascii=False, indent=2)
        )

        self.btn_export.setEnabled(True)
        self.tabs.setCurrentIndex(0)


    def _export_html_pdf_fallback(self, html: str, out_path: str) -> None:
        """Render PDF fallback through Qt WebEngine.

        QTextDocument keeps only a small subset of HTML/CSS and drops SVG in
        practice, which made reports look like plain text. WebEngine uses the
        same renderer as the sketcher tab, so tables, pastel highlights and
        inline SVG structures survive even when reportlab is not installed.
        """
        from PySide6.QtGui import QPageLayout, QPageSize
        from PySide6.QtWebEngineCore import QWebEnginePage

        page = QWebEnginePage(self)
        loop = QEventLoop()
        state = {"done": False, "ok": False, "error": ""}

        is_batch = ("приоритизационный" in html) or ("Пакет" in html) or ("Batch" in html)
        orientation = QPageLayout.Landscape if is_batch else QPageLayout.Portrait
        layout = QPageLayout(
            QPageSize(QPageSize.A4),
            orientation,
            QMarginsF(10.0, 10.0, 10.0, 10.0),
            QPageLayout.Millimeter,
        )

        def finish(ok: bool, error: str = ""):
            if state["done"]:
                return
            state["done"] = True
            state["ok"] = bool(ok)
            state["error"] = error
            loop.quit()

        def on_pdf_finished(path: str, success: bool):
            finish(success, "" if success else f"Qt WebEngine не смог записать PDF: {path}")

        def on_loaded(ok: bool):
            if not ok:
                finish(False, "Qt WebEngine не смог загрузить HTML отчёта.")
                return
            page.pdfPrintingFinished.connect(on_pdf_finished)
            page.printToPdf(out_path, layout)

        page.loadFinished.connect(on_loaded)
        QTimer.singleShot(60000, lambda: finish(False, "Таймаут PDF-экспорта через Qt WebEngine."))
        page.setHtml(html, QUrl.fromLocalFile(BASE_DIR + os.sep))
        loop.exec()
        page.deleteLater()

        if not state["ok"]:
            raise RuntimeError(state["error"] or "Неизвестная ошибка PDF-экспорта через Qt WebEngine.")

    def _batch_html_for_pdf_fallback(self, df, title: str) -> str:
        def esc_html(value):
            return escape("" if value is None else str(value))

        def find_col(candidates):
            raw = {str(col).strip().lower(): col for col in df.columns}
            for candidate in candidates:
                key = candidate.strip().lower()
                if key in raw:
                    return raw[key]
            for col in df.columns:
                name = str(col).strip().lower()
                for candidate in candidates:
                    if candidate.strip().lower() in name:
                        return col
            return None

        def status_rank(value):
            text = str(value or "").strip().lower()
            if "approve" in text or "одобр" in text:
                return 0
            if "review" in text or "провер" in text:
                return 1
            if "reject" in text or "отклон" in text:
                return 2
            if "insufficient" in text or "недостат" in text:
                return 3
            return 4

        def badge_class(value):
            text = str(value or "").strip().lower()
            if "approve" in text or "одобр" in text or "низ" in text:
                return "ok"
            if "review" in text or "провер" in text or "сред" in text:
                return "warn"
            if "reject" in text or "отклон" in text or "выс" in text or "крит" in text:
                return "bad"
            if "insufficient" in text or "недостат" in text:
                return "info"
            return "neutral"

        status_col = find_col(["Статус решения", "decision_status", "DSS"])
        risk_col = find_col(["Уровень риска", "risk_level", "Риск"])
        hazard_col = find_col(["Балл опасности DSS", "hazard_score", "Опасность"])
        uncertainty_col = find_col(["Балл неопределённости DSS", "uncertainty_score", "Неопределённость"])
        reason_col = find_col(["Ключевые причины DSS", "Главная причина", "reason"])
        ad_status_col = find_col(["AD-статус", "AD status", "applicability domain"])
        ad_score_col = find_col(["AD-score", "AD score", "mean_ad_score"])
        reliability_col = find_col(["Метка надёжности", "reliability", "Надёжность"])
        smiles_col = find_col(["SMILES", "smiles", "input", "Молекула"])

        work = df.copy()
        if status_col:
            work["_sort_status"] = work[status_col].map(status_rank)
            work["_sort_unc"] = work[uncertainty_col].map(lambda x: float(x) if isinstance(x, (int, float)) else 999.0) if uncertainty_col else 999.0
            work["_sort_haz"] = work[hazard_col].map(lambda x: float(x) if isinstance(x, (int, float)) else 999.0) if hazard_col else 999.0
            work = work.sort_values(["_sort_status", "_sort_unc", "_sort_haz"], kind="stable").drop(columns=["_sort_status", "_sort_unc", "_sort_haz"], errors="ignore")

        max_rows = min(len(work), 120)
        counts = {}
        if status_col:
            for value in work[status_col]:
                counts[str(value)] = counts.get(str(value), 0) + 1

        cards = "".join(
            f"<div class='card'><div class='muted'>{esc_html(label)}</div><b>{count}</b></div>"
            for label, count in counts.items()
        ) or "<div class='muted'>Статусы DSS недоступны.</div>"

        rows = []
        for _, row in work.head(max_rows).iterrows():
            status = row[status_col] if status_col else ""
            risk = row[risk_col] if risk_col else ""
            rows.append(
                "<tr>"
                f"<td class='mono'>{esc_html(row[smiles_col] if smiles_col else '')}</td>"
                f"<td><span class='badge {badge_class(status)}'>{esc_html(status)}</span></td>"
                f"<td><span class='badge {badge_class(risk)}'>{esc_html(risk)}</span></td>"
                f"<td>{esc_html(row[hazard_col] if hazard_col else '')}</td>"
                f"<td>{esc_html(row[uncertainty_col] if uncertainty_col else '')}</td>"
                f"<td><span class='badge {badge_class(row[ad_status_col] if ad_status_col else '')}'>{esc_html(row[ad_status_col] if ad_status_col else '')}</span></td>"
                f"<td>{esc_html(row[ad_score_col] if ad_score_col else '')}</td>"
                f"<td>{esc_html(row[reason_col] if reason_col else '')}</td>"
                f"<td>{esc_html(row[reliability_col] if reliability_col else '')}</td>"
                "</tr>"
            )
        note = "" if len(work) <= max_rows else f"<p class='muted'>В PDF показаны первые {max_rows} строк после сортировки. Полная таблица остаётся в CSV/XLSX.</p>"
        return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<style>
body {{ font-family: Segoe UI, Arial, sans-serif; font-size: 9pt; color: #20242A; }}
h1 {{ font-size: 17pt; margin-bottom: 2px; }}
.muted {{ color: #68717C; font-size: 8.5pt; }}
.summary {{ display: flex; gap: 8px; margin: 10px 0; }}
.card {{ border: 1px solid #DDE3EA; background: #F7F9FB; padding: 7px 9px; border-radius: 6px; }}
table {{ width: 100%; border-collapse: collapse; }}
th, td {{ border: 1px solid #D0D7DE; padding: 4px; vertical-align: top; }}
th {{ background: #EEF3F7; text-align: left; }}
.badge {{ border-radius: 5px; padding: 3px 6px; font-weight: 600; }}
.ok {{ background: #DDEFD8; color: #2F5C3B; }}
.warn {{ background: #F7E7BE; color: #755B16; }}
.bad {{ background: #F3D2D2; color: #7A3030; }}
.info {{ background: #DCE7F3; color: #2C4D6B; }}
.neutral {{ background: #ECEFF3; color: #3E4852; }}
.mono {{ font-family: Consolas, monospace; overflow-wrap: anywhere; }}
</style>
</head>
<body>
<h1>{esc_html(title)}</h1>
<p class="muted">Сформирован: {datetime.now().isoformat(timespec='seconds')} | Молекул: {len(work)}</p>
<p class="muted">Сортировка: Approve → Review → Reject → Insufficient data; внутри Approve выше молекулы с меньшей неопределённостью.</p>
<div class="summary">{cards}</div>
{note}
<table>
<thead><tr><th>Молекула</th><th>DSS</th><th>Риск</th><th>Опасность</th><th>Неопределённость</th><th>AD</th><th>AD score</th><th>Главная причина</th><th>Надёжность</th></tr></thead>
<tbody>{''.join(rows)}</tbody>
</table>
</body>
</html>"""

    def on_export(self):
        if not self._last_payload:
            QMessageBox.warning(self, "Нет данных для экспорта", "Сначала сформируйте отчёт.")
            return

        default_name = f"chem_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        out_path, _ = QFileDialog.getSaveFileName(self, "Сохранить отчёт в PDF", default_name, "PDF (*.pdf)")
        if not out_path:
            return

        try:
            render_report_pdf(self._last_payload, out_path)
        except ModuleNotFoundError as e:
            if e.name != "reportlab":
                self._logger.exception("PDF export failed: %s", out_path)
                QMessageBox.critical(self, "Ошибка экспорта", f"{type(e).__name__}: {e}")
                return
            self._logger.warning("reportlab is unavailable; using Qt HTML PDF fallback.")
            try:
                self._export_html_pdf_fallback(render_report_html(self._last_payload), out_path)
            except Exception as fallback_exc:
                self._logger.exception("PDF fallback export failed: %s", out_path)
                QMessageBox.critical(self, "Ошибка экспорта", f"{type(fallback_exc).__name__}: {fallback_exc}")
                return
        except Exception as e:
            self._logger.exception("PDF export failed: %s", out_path)
            QMessageBox.critical(self, "Ошибка экспорта", f"{type(e).__name__}: {e}")
            return
        self._logger.info("PDF export completed: %s", out_path)
        QMessageBox.information(self, "Экспорт завершён", f"Сохранено:\n{out_path}")

    def _show_df_in_table(self, df, table: QTableWidget, max_rows: int = 500):
        # ограничим количество строк, чтобы UI не тормозил
        n_rows = min(len(df), max_rows)
        cols = list(df.columns)

        table.clear()
        table.setColumnCount(len(cols))
        table.setRowCount(n_rows)
        table.setHorizontalHeaderLabels([str(c) for c in cols])
        table.horizontalHeader().setStretchLastSection(True)

        for r in range(n_rows):
            for c, col in enumerate(cols):
                val = df.iloc[r, c]
                item = QTableWidgetItem("" if val is None else str(val))
                item.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)
                table.setItem(r, c, item)

        # optional: показать пользователю, что не все строки отображены
        if len(df) > max_rows:
            table.setToolTip(f"Показаны первые {max_rows} из {len(df)} строк.")

    def on_save_batch(self):
        if getattr(self, "_batch_df", None) is None:
            QMessageBox.information(self, "Нет данных для сохранения", "Пакетные результаты ещё не загружены.")
            return

        default_path = default_batch_output_path("outputs", ext=".csv")
        out_path, _ = QFileDialog.getSaveFileName(self, "Сохранить пакетные результаты", default_path, "CSV (*.csv);;Excel (*.xlsx)")
        if not out_path:
            return

        try:
            save_table(self._batch_df, out_path)
        except Exception as e:
            self._logger.exception("Batch save failed: %s", out_path)
            QMessageBox.critical(self, "Ошибка сохранения", str(e))
            return
        self._logger.info("Batch results saved: %s", out_path)

        QMessageBox.information(self, "Сохранено", f"Сохранено:\n{out_path}")

    def on_export_batch_pdf(self):
        if getattr(self, "_batch_df", None) is None:
            QMessageBox.information(self, "Нет данных для экспорта", "Пакетные результаты ещё не загружены.")
            return

        default_name = f"batch_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        out_path, _ = QFileDialog.getSaveFileName(self, "Сохранить пакетный отчёт в PDF", default_name, "PDF (*.pdf)")
        if not out_path:
            return

        batch_title = "Пакетный химический отчёт"
        try:
            render_batch_table_pdf(self._batch_df, out_path, title=batch_title)
        except ModuleNotFoundError as e:
            if e.name != "reportlab":
                self._logger.exception("Batch PDF export failed: %s", out_path)
                QMessageBox.critical(self, "Ошибка экспорта", f"{type(e).__name__}: {e}")
                return
            self._logger.warning("reportlab is unavailable; using Qt HTML batch PDF fallback.")
            try:
                self._export_html_pdf_fallback(self._batch_html_for_pdf_fallback(self._batch_df, batch_title), out_path)
            except Exception as fallback_exc:
                self._logger.exception("Batch PDF fallback export failed: %s", out_path)
                QMessageBox.critical(self, "Ошибка экспорта", f"{type(fallback_exc).__name__}: {fallback_exc}")
                return
        except Exception as e:
            self._logger.exception("Batch PDF export failed: %s", out_path)
            QMessageBox.critical(self, "Ошибка экспорта", f"{type(e).__name__}: {e}")
            return
        self._logger.info("Batch PDF export completed: %s", out_path)
        QMessageBox.information(self, "Экспорт завершён", f"Сохранено:\n{out_path}")
    def on_batch_row_double_clicked(self, row: int, col: int):
        if getattr(self, "_batch_df", None) is None:
            return

        smiles_col = getattr(self, "_batch_input_col", None)
        if not smiles_col or smiles_col not in self._batch_df.columns:
            QMessageBox.warning(self, "Не найден столбец SMILES", "Не удалось найти столбец SMILES в пакетных данных.")
            return

        # Важно: если ты показываешь только первые N строк в таблице — row совпадает
        # только для первых N. Это ок для MVP.
        try:
            smiles = str(self._batch_df.iloc[row][smiles_col]).strip()
        except Exception as e:
            QMessageBox.warning(self, "Ошибка строки", str(e))
            return

        if not smiles or smiles.lower() == "nan":
            QMessageBox.information(self, "Пустой SMILES", "В выбранной строке отсутствует SMILES.")
            return

        # Заполняем input и запускаем тот же pipeline, что и при ручном вводе
        self.input_edit.setText(smiles)
        self.type_combo.setCurrentText("SMILES")
        self.on_generate()
        self.left_tabs.setCurrentIndex(1)
    def _cancel_batch(self):
            # визуальная обратная связь
            self._progress.setLabelText("Отмена… завершается обработка текущей молекулы.")
            self._progress.setCancelButtonText("Отмена…")
            self._logger.warning("Batch cancel requested by user.")

            # запрос на отмену
            if hasattr(self, "_batch_worker"):
                self._batch_worker.request_cancel()

            if hasattr(self, "_batch_thread"):
                self._batch_thread.requestInterruption()

def main():
    prepare_gui_environment()
    logger.info("Запуск приложения начат.")
    install_qt_logging()
    log_startup_environment()

    logger.info("Этап запуска: перед QApplication()")
    app = QApplication(sys.argv)
    logger.info("Этап запуска: QApplication() создан")

    logger.info("Этап запуска: перед применением стилей")
    app.setStyleSheet("""
    QMainWindow { background: #111315; }
    QLabel { color: #E6E6E6; font-size: 12px; }

    QLineEdit, QComboBox {
      background: #1A1D21; color: #E6E6E6;
      border: 1px solid #2A2F36; border-radius: 8px;
      padding: 8px;
    }

    QPushButton {
      background: #1F6FEB; color: white;
      border: 0px; border-radius: 8px;
      padding: 8px 12px;
    }
    QPushButton:disabled { background: #2A2F36; color: #8B8F97; }

    QTabWidget::pane { border: 1px solid #2A2F36; border-radius: 10px; }
    QTabBar::tab {
      background: #1A1D21; color: #CFCFCF;
      padding: 8px 12px; border-top-left-radius: 8px; border-top-right-radius: 8px;
      margin-right: 6px;
    }
    QTabBar::tab:selected { background: #20242A; color: #FFFFFF; }

    QTableWidget {
      background: #14171A; color: #E6E6E6;
      border: 1px solid #2A2F36; border-radius: 10px;
      gridline-color: #2A2F36;
    }
    QHeaderView::section {
      background: #1A1D21; color: #BFC7D5;
      border: 0px; padding: 8px;
    }

    QTextEdit {
      background: #14171A; color: #E6E6E6;
      border: 1px solid #2A2F36; border-radius: 10px;
      padding: 8px;
    }

    QFrame#SvgCard {
      background: #FFFFFF;
      border: 1px solid #E6E6E6;
      border-radius: 12px;
    }
    """)
    logger.info("Этап запуска: стили применены")


    logger.info("Этап запуска: перед MainWindow()")
    w = MainWindow()
    logger.info("Этап запуска: MainWindow() создан")
    w.show()
    logger.info("Этап запуска: MainWindow() показан")
    app.exec()
    logger.info("Цикл событий приложения завершён.")

    # app = QApplication([])
    # w = MainWindow()
    # w.show()
    # app.exec()


if __name__ == "__main__":
    main()



