from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from app import MainWindow, install_qt_logging, log_startup_environment, logger
from core.startup import prepare_gui_environment


STYLE_SHEET = """
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
"""


def main() -> int:
    prepare_gui_environment()
    logger.info("Launcher started.")
    install_qt_logging()
    log_startup_environment()

    logger.info("Launcher: before QApplication")
    app = QApplication(sys.argv)
    logger.info("Launcher: after QApplication")

    app.setStyleSheet(STYLE_SHEET)
    logger.info("Launcher: stylesheet applied")

    window = MainWindow()
    window.show()
    logger.info("Launcher: main window shown")

    exit_code = app.exec()
    logger.info("Launcher finished: exit_code=%s", exit_code)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
