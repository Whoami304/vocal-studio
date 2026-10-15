"""Диалоги: настройки аудио, экспорт, Auto-Mix, приветствие, обучение."""
from __future__ import annotations

import os
import subprocess
import sys

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (QButtonGroup, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFormLayout,
                             QFrame, QHBoxLayout, QLabel, QProgressBar, QPushButton, QRadioButton, QStackedWidget,
                             QTextEdit, QVBoxLayout, QWidget)

from .. import analysis, audio_io
from . import theme


def open_folder(path: str):
    try:
        if sys.platform.startswith("win"):
            os.startfile(path)  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", path])
        else:
            subprocess.Popen(["xdg-open", path])
    except Exception:
        pass


class SettingsDialog(QDialog):
    def __init__(self, settings: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Настройки аудио")
        self.setMinimumWidth(540)
        self.settings = dict(settings)
        ins, outs, d_in, d_out = audio_io.list_devices()
        form = QFormLayout()
        self.cb_in = QComboBox()
        self.cb_out = QComboBox()
        self.cb_in.addItem("Системное по умолчанию", None)
        self.cb_out.addItem("Системное по умолчанию", None)
        for i, n in ins:
            self.cb_in.addItem(n, n)
        for i, n in outs:
            self.cb_out.addItem(n, n)
        self._ins = dict((n, i) for i, n in ins)
        self._outs = dict((n, i) for i, n in outs)
        for cb, key in ((self.cb_in, "in_dev"), (self.cb_out, "out_dev")):
            name = settings.get(key)
            if name:
                idx = cb.findData(name)
                if idx >= 0:
                    cb.setCurrentIndex(idx)
        form.addRow("Микрофон (вход):", self.cb_in)
        form.addRow("Наушники / колонки (выход):", self.cb_out)
        self.sp_lat = QDoubleSpinBox()
        self.sp_lat.setRange(-300.0, 500.0)
        self.sp_lat.setSuffix(" мс")
        self.sp_lat.setDecimals(1)
        self.sp_lat.setValue(float(settings.get("latency_ms", 0.0)))
        self.sp_lat.setToolTip("Если вокал после записи звучит позже музыки — увеличьте значение.")
        form.addRow("Подстройка задержки записи:", self.cb_lat_label())
        form.addRow("", self.sp_lat)
        self.cb_mon = QCheckBox("Слышать себя во время записи (мониторинг без эффектов)")
        self.cb_mon.setChecked(bool(settings.get("monitor", False)))
        form.addRow("", self.cb_mon)
        note = QLabel("Используйте наушники при включённом мониторинге, иначе возникнет обратная связь (свист).\n"
                      "Эффекты применяются после записи — сразу при воспроизведении.")
        note.setObjectName("muted")
        note.setWordWrap(True)
        lay = QVBoxLayout(self)
        lay.addLayout(form)
        lay.addWidget(note)
        row = QHBoxLayout()
        b = QPushButton("Открыть папку с записями")
        b.clicked.connect(lambda: open_folder(audio_io.takes_dir()))
        row.addWidget(b)
        row.addStretch(1)
        lay.addLayout(row)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

    def cb_lat_label(self):
        l = QLabel("Если вокал звучит позже/раньше музыки при записи с битом — подстройте (мс):")
        l.setObjectName("muted")
        l.setWordWrap(True)
        return l

    def result_settings(self) -> dict:
        s = dict(self.settings)
        s["in_dev"] = self.cb_in.currentData()
        s["out_dev"] = self.cb_out.currentData()
        s["latency_ms"] = self.sp_lat.value()
        s["monitor"] = self.cb_mon.isChecked()
        return s


class ExportDialog(QDialog):
    def __init__(self, n_tracks: int, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Экспорт")
        self.setMinimumWidth(420)
        lay = QVBoxLayout(self)
        form = QFormLayout()
        self.cb_fmt = QComboBox()
        self.cb_fmt.addItems(list(audio_io.EXPORT_FORMATS.keys()))
        self.cb_fmt.setToolTip("WAV 24-bit — для дальнейшей обработки, MP3 — чтобы отправить другу")
        form.addRow("Формат:", self.cb_fmt)
        lay.addLayout(form)
        self.rb_mix = QRadioButton("Весь проект одним файлом (готовый микс)")
        self.rb_stems = QRadioButton("Каждую дорожку отдельным файлом (после обработки)")
        self.rb_mix.setChecked(True)
        self.rb_stems.setEnabled(n_tracks > 1)
        lay.addWidget(self.rb_mix)
        lay.addWidget(self.rb_stems)
        self.chk_norm = QCheckBox("Нормализовать пик до −1 дБFS")
        lay.addWidget(self.chk_norm)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)


class AutoMixDialog(QDialog):
    """Мастер Auto-Mix: выбор стиля → анализ → отчёт с возможностью отмены."""

    def __init__(self, session, tid, parent=None):
        super().__init__(parent)
        self.s = session
        self.tid = tid
        self.undo = None
        self.setWindowTitle("✨ Auto-Mix — умная настройка вокала")
        self.setMinimumWidth(640)
        self.setMinimumHeight(560)
        t = session.track(tid)
        root = QVBoxLayout(self)
        root.setSpacing(10)
        h = QLabel(f"Дорожка: <b>{t.name if t else ''}</b>")
        root.addWidget(h)
        intro = QLabel("Программа послушает ваш голос и подложку, определит тональность, тембр, динамику и резкость «с», "
                       "затем сама соберёт цепочку плагинов и выставит громкость. Всё можно потом подправить вручную "
                       "или отменить.")
        intro.setWordWrap(True)
        intro.setObjectName("muted")
        root.addWidget(intro)

        self.pages = QStackedWidget()
        root.addWidget(self.pages, 1)

        # --- страница 1: настройки
        p1 = QWidget()
        l1 = QVBoxLayout(p1)
        l1.setContentsMargins(0, 0, 0, 0)
        cap = QLabel("СТИЛЬ ЗВУЧАНИЯ")
        cap.setObjectName("title")
        l1.addWidget(cap)
        self.style_group = QButtonGroup(self)
        self.style_btns = {}
        for name in analysis.STYLE_ORDER:
            st = analysis.STYLES[name]
            rb = QRadioButton(f"{st['label']}  —  {st['desc']}")
            rb.setStyleSheet("padding: 3px;")
            self.style_group.addButton(rb)
            self.style_btns[name] = rb
            l1.addWidget(rb)
        self.style_btns["Pop"].setChecked(True)
        cap2 = QLabel("ЧТО СДЕЛАТЬ")
        cap2.setObjectName("title")
        l1.addSpacing(6)
        l1.addWidget(cap2)
        self.opts = {}
        for key, text in (("tune", "Исправить фальшь (автотюн, тональность подберу сам)"),
                          ("gate", "Убрать шум и гул (Noise Gate, срез низов)"),
                          ("eq", "Подправить тембр (EQ)"),
                          ("dynamics", "Выровнять громкость голоса (компрессор)"),
                          ("deess", "Смягчить резкие «с» и «ш» (De-esser)"),
                          ("space", "Добавить пространство (реверберация, эхо, тепло)"),
                          ("levels", "Выставить баланс громкости голоса и подложки")):
            cb = QCheckBox(text)
            cb.setChecked(True)
            self.opts[key] = cb
            l1.addWidget(cb)
        l1.addStretch(1)
        self.pages.addWidget(p1)

        # --- страница 2: прогресс
        p2 = QWidget()
        l2 = QVBoxLayout(p2)
        l2.addStretch(1)
        self.lbl_prog = QLabel("Анализирую…")
        self.lbl_prog.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_prog.setStyleSheet("font-size: 15px;")
        l2.addWidget(self.lbl_prog)
        bar = QProgressBar()
        bar.setRange(0, 0)
        bar.setFixedHeight(10)
        l2.addWidget(bar)
        l2.addStretch(1)
        self.pages.addWidget(p2)

        # --- страница 3: отчёт
        p3 = QWidget()
        l3 = QVBoxLayout(p3)
        l3.setContentsMargins(0, 0, 0, 0)
        self.lbl_done = QLabel("")
        self.lbl_done.setWordWrap(True)
        self.lbl_done.setStyleSheet("font-size: 14px; font-weight: 600;")
        l3.addWidget(self.lbl_done)
        self.txt = QTextEdit()
        self.txt.setReadOnly(True)
        l3.addWidget(self.txt, 1)
        self.pages.addWidget(p3)

        # --- кнопки
        row = QHBoxLayout()
        self.btn_run = QPushButton("✨  Анализировать и настроить")
        self.btn_run.setObjectName("magic")
        self.btn_run.setMinimumHeight(36)
        self.btn_undo = QPushButton("Отменить Auto-Mix")
        self.btn_close = QPushButton("Закрыть")
        row.addWidget(self.btn_undo)
        row.addStretch(1)
        row.addWidget(self.btn_close)
        row.addWidget(self.btn_run)
        root.addLayout(row)
        self.btn_undo.hide()
        self.btn_run.clicked.connect(self._run)
        self.btn_close.clicked.connect(self.accept)
        self.btn_undo.clicked.connect(self._undo)
        session.automix_progress.connect(self._progress)
        session.automix_done.connect(self._done)

    def done(self, r):
        try:
            self.s.automix_progress.disconnect(self._progress)
            self.s.automix_done.disconnect(self._done)
        except Exception:
            pass
        super().done(r)

    def _style(self):
        for name, rb in self.style_btns.items():
            if rb.isChecked():
                return name
        return "Pop"

    def _run(self):
        self.pages.setCurrentIndex(1)
        self.btn_run.setEnabled(False)
        opts = {k: cb.isChecked() for k, cb in self.opts.items()}
        self.s.run_automix(self.tid, self._style(), opts, apply_levels=opts.pop("levels"))

    def _progress(self, text):
        self.lbl_prog.setText(text)

    def _done(self, res):
        self.btn_run.setEnabled(True)
        if "error" in res:
            self.pages.setCurrentIndex(0)
            self.lbl_prog.setText("")
            self.btn_run.setText("✨  Попробовать снова")
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.information(self, "Auto-Mix", res["error"])
            return
        self.undo = self.s.apply_automix(res)
        va = res["analysis"]
        lines = []
        info = [f"Диапазон голоса: {_note(va.f0_min)} – {_note(va.f0_max)} ({va.voice_type})"]
        if va.key:
            info.append(f"Тональность по пению: {va.key.name} (уверенность {va.key.confidence_label})")
        if va.backing_key:
            info.append(f"Тональность подложки: {va.backing_key.name} (уверенность {va.backing_key.confidence_label})")
        if va.backing_tempo:
            info.append(f"Темп подложки: ≈{va.backing_tempo[0]:.0f} BPM")
        html = "<h4>Что я услышал</h4><ul>" + "".join(f"<li>{x}</li>" for x in info) + "</ul>"
        html += "<h4>Что сделано</h4><ul>" + "".join(f"<li>{x}</li>" for x in res["report"]) + "</ul>"
        if res.get("vol_db") is not None:
            html += f"<p>Громкость дорожки: <b>{res['vol_db']:+.1f} дБ</b></p>"
        self.txt.setHtml(html)
        self.lbl_done.setText("✓ Готово! Нажмите ▶ Play, чтобы послушать. Плагины можно подправить в нижней части окна.")
        self.pages.setCurrentIndex(2)
        self.btn_run.hide()
        self.btn_undo.show()

    def _undo(self):
        if self.undo is not None:
            self.s.undo_automix(self.undo)
            self.undo = None
            self.btn_undo.setEnabled(False)
            self.lbl_done.setText("Auto-Mix отменён. Плагины и громкость возвращены как были.")


def _note(f):
    import math
    if f <= 0:
        return "—"
    m = int(round(69 + 12 * math.log2(f / 440.0)))
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    return f"{names[m % 12]}{m // 12 - 1}"


class WelcomeDialog(QDialog):
    """Приветствие при первом запуске."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Добро пожаловать в Vocal Studio")
        self.setMinimumWidth(680)
        self.choice = None
        lay = QVBoxLayout(self)
        lay.setSpacing(12)
        t = QLabel("Добро пожаловать в <span style='color:#2fd0c0'>Vocal Studio</span>")
        t.setStyleSheet("font-size: 22px; font-weight: 700;")
        lay.addWidget(t)
        s = QLabel("Запишите голос, исправьте фальшь автотюном и доведите звук до студийного — всё на вашем компьютере. "
                   "Не нужно ничего знать о звукорежиссуре: за четыре шага получится готовый трек.")
        s.setWordWrap(True)
        s.setObjectName("muted")
        lay.addWidget(s)
        steps = QLabel(
            "<ol style='line-height:150%; margin-left:-18px'>"
            "<li><b>Добавьте звук.</b> Запишите голос (кнопка ● REC) или откройте готовый файл.</li>"
            "<li><b>Нажмите ✨ Auto-Mix.</b> Программа сама определит тональность и настроит все эффекты. "
            "Либо добавьте плагины сами.</li>"
            "<li><b>Слушайте.</b> Кнопка ▶ Play (или пробел). «Original» — сравнить с необработанным звуком.</li>"
            "<li><b>Сохраните.</b> Кнопка Export — готовый WAV или MP3.</li></ol>")
        steps.setWordWrap(True)
        lay.addWidget(steps)
        row = QHBoxLayout()
        for text, key in (("●  Записать мой голос", "record"), ("Открыть вокал из файла", "vocal"),
                          ("Загрузить бит / инструментал", "backing")):
            b = QPushButton(text)
            b.setMinimumHeight(44)
            b.clicked.connect(lambda _=False, k=key: self._pick(k))
            if key == "record":
                b.setObjectName("rec")
            row.addWidget(b)
        lay.addLayout(row)
        row2 = QHBoxLayout()
        self.chk = QCheckBox("Больше не показывать")
        row2.addWidget(self.chk)
        row2.addStretch(1)
        tour = QPushButton("Быстрый тур по программе")
        tour.clicked.connect(lambda: self._pick("tour"))
        skip = QPushButton("Пропустить")
        skip.clicked.connect(lambda: self._pick(None))
        row2.addWidget(tour)
        row2.addWidget(skip)
        lay.addLayout(row2)

    def _pick(self, k):
        self.choice = k
        self.accept()


class TourDialog(QDialog):
    PAGES = [
        ("Дорожки", "Верхняя часть окна — <b>дорожки</b>. Каждая дорожка — это отдельный голос или музыка.<br><br>"
                    "<b>●</b> — записывать на эту дорожку, <b>M</b> — выключить звук, <b>S</b> — слушать только её. "
                    "Ползунки — громкость и панорама (левее/правее).<br><br>"
                    "Добавьте столько дорожек, сколько нужно: основной вокал, бэки, дубль, бит."),
        ("Плагины", "Нижняя часть — <b>плагины выбранной дорожки</b>. Это эффекты: автотюн, эквалайзер, компрессор, "
                    "реверберация, дилей…<br><br>"
                    "Кнопка «＋ Добавить плагин» добавляет эффект в цепочку. Звук идёт сверху вниз. "
                    "Нажмите на плагин — справа откроются его настройки. Кружок слева включает/выключает плагин."),
        ("Auto-Mix", "Не хотите разбираться? Нажмите <b>✨ Auto-Mix</b>. Программа проанализирует голос: определит "
                     "тональность и темп, найдёт шум, резкие «с», выровняет громкость — и сама настроит плагины."),
        ("Автотюн", "Плагин <b>Auto-Tune</b> исправляет фальшивые ноты. Кнопка «🎯 Определить тональность» сама найдёт ключ. "
                    "<b>Retune Speed</b>: 0 — эффект «робот», 20–40 — как в современной музыке, 80+ — почти незаметно.<br><br>"
                    "На графике можно перетащить любую ноту вручную."),
        ("Слушать и сохранить", "<b>Пробел</b> — воспроизведение. Кнопка <b>Original (A/B)</b> — сравнить с необработанным звуком.<br><br>"
                                "<b>Export</b> — сохранить готовый микс (или каждую дорожку отдельно). "
                                "<b>Файл → Сохранить проект</b> — чтобы продолжить позже.<br><br>"
                                "Подсказка: наведите курсор на любую ручку — внизу окна появится пояснение."),
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Быстрый тур")
        self.setMinimumSize(560, 340)
        lay = QVBoxLayout(self)
        self.stack = QStackedWidget()
        for i, (title, text) in enumerate(self.PAGES):
            w = QWidget()
            wl = QVBoxLayout(w)
            t = QLabel(f"{i + 1}/{len(self.PAGES)}  ·  {title}")
            t.setStyleSheet("font-size: 18px; font-weight: 700; color: #2fd0c0;")
            wl.addWidget(t)
            b = QLabel(text)
            b.setWordWrap(True)
            b.setAlignment(Qt.AlignmentFlag.AlignTop)
            b.setStyleSheet("font-size: 13px; line-height: 150%;")
            wl.addWidget(b, 1)
            self.stack.addWidget(w)
        lay.addWidget(self.stack, 1)
        row = QHBoxLayout()
        self.b_prev = QPushButton("← Назад")
        self.b_next = QPushButton("Далее →")
        self.b_next.setObjectName("primary2")
        row.addWidget(self.b_prev)
        row.addStretch(1)
        row.addWidget(self.b_next)
        lay.addLayout(row)
        self.b_prev.clicked.connect(lambda: self._go(-1))
        self.b_next.clicked.connect(lambda: self._go(1))
        self._upd()

    def _go(self, d):
        i = self.stack.currentIndex() + d
        if i >= self.stack.count():
            self.accept()
            return
        self.stack.setCurrentIndex(max(0, i))
        self._upd()

    def _upd(self):
        i = self.stack.currentIndex()
        self.b_prev.setEnabled(i > 0)
        self.b_next.setText("Готово" if i == self.stack.count() - 1 else "Далее →")


class ShortcutsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Горячие клавиши")
        lay = QVBoxLayout(self)
        rows = [("Пробел", "Воспроизведение / пауза"), ("R", "Запись на выбранную (armed) дорожку"),
                ("Home", "В начало"), ("Ctrl+T", "Новая вокальная дорожка"), ("Ctrl+I", "Импорт аудиофайла на новую дорожку"),
                ("Ctrl+M", "Auto-Mix для выбранной дорожки"), ("Ctrl+B", "Original (A/B): послушать без обработки"),
                ("Ctrl+Z", "Отменить последнюю запись на дорожке"), ("Ctrl+S", "Сохранить проект"),
                ("Ctrl+O", "Открыть проект"), ("Ctrl+E", "Экспорт"),
                ("Ctrl+колесо", "Масштаб дорожек по времени"), ("Shift+колесо", "Прокрутка по времени"),
                ("Двойной клик по ручке", "Сбросить значение")]
        txt = "<table cellpadding='4'>" + "".join(f"<tr><td><b>{a}</b></td><td>{b}</td></tr>" for a, b in rows) + "</table>"
        l = QLabel(txt)
        lay.addWidget(l)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        bb.rejected.connect(self.reject)
        bb.accepted.connect(self.accept)
        bb.button(QDialogButtonBox.StandardButton.Close).clicked.connect(self.accept)
        lay.addWidget(bb)
