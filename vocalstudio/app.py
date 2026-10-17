"""Точка входа."""
import os
import sys


def _install_excepthook():
    """В windowed-exe нет консоли: пишем ошибки в журнал и показываем окно (с подавлением повторов)."""
    from .gui import safety

    def hook(exc_type, exc, tb):
        safety.report(exc_type, exc, tb, where="uncaught")

    sys.excepthook = hook
    try:
        import threading

        def thook(args):
            safety.log_exception(args.exc_type, args.exc_value, args.exc_traceback, where="thread")

        threading.excepthook = thook
    except Exception:
        pass


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
        w.open_path(sys.argv[1])
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
