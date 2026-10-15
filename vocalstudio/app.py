"""Точка входа."""
import os
import sys
import traceback


def _install_excepthook():
    """В windowed-exe нет консоли: пишем ошибки в лог и показываем окно."""
    from .session import config_dir

    def hook(exc_type, exc, tb):
        text = "".join(traceback.format_exception(exc_type, exc, tb))
        try:
            with open(os.path.join(config_dir(), "error.log"), "a", encoding="utf-8") as f:
                f.write(text + "\n")
        except Exception:
            pass
        try:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.critical(None, "Vocal Studio — ошибка",
                                 f"{exc_type.__name__}: {exc}\n\nПодробности в {os.path.join(config_dir(), 'error.log')}")
        except Exception:
            pass

    sys.excepthook = hook


def main():
    from PyQt6.QtGui import QFont
    from PyQt6.QtWidgets import QApplication
    from .gui import theme
    from .gui.main_window import MainWindow, APP_NAME
    from .icon import make_icon

    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setStyle("Fusion")
    app.setStyleSheet(theme.QSS)
    app.setFont(QFont("Segoe UI", 9))
    app.setWindowIcon(make_icon())
    _install_excepthook()
    w = MainWindow()
    w.setWindowIcon(make_icon())
    w.show()
    if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]):
        w.load_vocal(sys.argv[1])
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
