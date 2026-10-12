"""MathSnap — сфотографировал пример, получил пошаговое решение.

Экраны: снимок -> распознанный пример (его можно поправить руками) ->
решение со шагами -> история решённого. Всё офлайн: своя модель
распознавания assets/model.tflite и решатель на SymPy. Интернет не нужен.

История и снимки лежат в приватной папке приложения (user_data_dir).
"""

from __future__ import annotations

import json
import os
import shutil
import threading
from datetime import datetime
from pathlib import Path

from kivy.app import App
from kivy.clock import Clock, mainthread
from kivy.lang import Builder
from kivy.metrics import dp
from kivy.properties import StringProperty
from kivy.uix.button import Button
from kivy.uix.screenmanager import Screen, ScreenManager, SlideTransition
from kivy.utils import platform

from ocr.recognize import read_expression
from solver.solve import solve

REQUEST_CAMERA = 0x51
REQUEST_GALLERY = 0x52
IS_ANDROID = platform == "android"
MODEL_PATH = "assets/model.tflite"
HISTORY_LIMIT = 200

KV = """
#:import dp kivy.metrics.dp

<HomeScreen>:
    name: "home"
    BoxLayout:
        orientation: "vertical"
        padding: dp(14)
        spacing: dp(10)
        Label:
            text: "MathSnap"
            font_size: dp(28)
            bold: True
            size_hint_y: None
            height: dp(42)
            color: 0.13, 0.35, 0.68, 1
        Image:
            source: root.preview_path
            size_hint_y: None
            height: dp(150) if root.preview_path else dp(0)
        Label:
            text: root.status
            font_size: dp(13)
            size_hint_y: None
            height: dp(22)
            color: 0.36, 0.41, 0.52, 1
        TextInput:
            id: expression_input
            text: root.expression
            multiline: False
            font_size: dp(24)
            size_hint_y: None
            height: dp(56)
            hint_text: "например: 3/4 + sqrt(25)"
            background_color: 1, 1, 1, 1
            foreground_color: 0.12, 0.15, 0.22, 1
        Label:
            text: root.note
            font_size: dp(12)
            size_hint_y: None
            height: dp(20)
            color: 0.45, 0.51, 0.62, 1
        BoxLayout:
            size_hint_y: None
            height: dp(52)
            spacing: dp(10)
            Button:
                text: "Сфотографировать"
                background_color: 0.13, 0.35, 0.68, 1
                on_release: root.take_photo()
            Button:
                text: "Из галереи"
                on_release: root.open_gallery()
        BoxLayout:
            size_hint_y: None
            height: dp(52)
            spacing: dp(10)
            Button:
                text: "Решить"
                background_color: 0.16, 0.54, 0.43, 1
                on_release: root.solve_current()
            Button:
                text: "История"
                on_release: root.show_history()
        Widget:

<ResultScreen>:
    name: "result"
    BoxLayout:
        orientation: "vertical"
        padding: dp(14)
        spacing: dp(8)
        Label:
            text: root.expression
            font_size: dp(20)
            size_hint_y: None
            height: dp(34)
            color: 0.25, 0.29, 0.38, 1
        Label:
            text: root.answer
            font_size: dp(26)
            bold: True
            size_hint_y: None
            height: dp(42)
            color: 0.13, 0.35, 0.68, 1
        ScrollView:
            Label:
                text: root.steps_text
                size_hint_y: None
                height: self.texture_size[1]
                text_size: self.width - dp(10), None
                halign: "left"
                valign: "top"
                font_size: dp(14)
                color: 0.18, 0.21, 0.28, 1
        BoxLayout:
            size_hint_y: None
            height: dp(50)
            spacing: dp(10)
            Button:
                text: "Назад"
                on_release: root.go_back()
            Button:
                text: "История"
                on_release: root.go_history()

<HistoryScreen>:
    name: "history"
    BoxLayout:
        orientation: "vertical"
        padding: dp(14)
        spacing: dp(8)
        Label:
            text: "История"
            font_size: dp(24)
            bold: True
            size_hint_y: None
            height: dp(38)
            color: 0.13, 0.35, 0.68, 1
        ScrollView:
            GridLayout:
                id: history_box
                cols: 1
                size_hint_y: None
                height: self.minimum_height
                spacing: dp(6)
        BoxLayout:
            size_hint_y: None
            height: dp(50)
            spacing: dp(10)
            Button:
                text: "Назад"
                on_release: root.go_back()
            Button:
                text: "Очистить"
                background_color: 0.75, 0.29, 0.29, 1
                on_release: root.clear_history()
"""


class HomeScreen(Screen):
    """Главный экран: снимок, распознанный пример, кнопка «Решить»."""

    status = StringProperty("Сфотографируй пример или набери его руками")
    expression = StringProperty("")
    preview_path = StringProperty("")
    note = StringProperty("")

    def take_photo(self):
        App.get_running_app().take_photo()

    def open_gallery(self):
        App.get_running_app().open_gallery()

    def solve_current(self):
        App.get_running_app().solve_text(self.ids.expression_input.text)

    def show_history(self):
        self.manager.current = "history"


class ResultScreen(Screen):
    """Ответ и шаги решения."""

    expression = StringProperty("")
    answer = StringProperty("")
    steps_text = StringProperty("")

    def go_back(self):
        self.manager.current = "home"

    def go_history(self):
        self.manager.current = "history"


class HistoryScreen(Screen):
    """Список решённых примеров, тап открывает решение заново."""

    def on_pre_enter(self, *args):
        app = App.get_running_app()
        box = self.ids.history_box
        box.clear_widgets()

        if not app.history:
            box.add_widget(Button(text="Пока пусто", size_hint_y=None, height=dp(46),
                                  disabled=True))
            return

        for item in reversed(app.history[-HISTORY_LIMIT:]):
            label = f"{item['when']}   {item['expression']} = {item['answer']}"
            button = Button(text=label, size_hint_y=None, height=dp(52),
                            halign="left", valign="middle", font_size=dp(13))
            button.bind(size=lambda widget, size: setattr(widget, "text_size",
                                                          (size[0] - dp(12), None)))
            button.bind(on_release=lambda widget, entry=item:
                        App.get_running_app().open_history_item(entry))
            box.add_widget(button)

    def clear_history(self):
        App.get_running_app().clear_history()
        self.on_pre_enter()

    def go_back(self):
        self.manager.current = "home"


class MathSnapApp(App):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_path = Path(MODEL_PATH)
        self.history = []
        self.shots_dir = None
        self.pending_uri = ""
        self.manager = None

    def build(self):
        Builder.load_string(KV)
        manager = ScreenManager(transition=SlideTransition())
        manager.add_widget(HomeScreen())
        manager.add_widget(ResultScreen())
        manager.add_widget(HistoryScreen())
        self.manager = manager
        Clock.schedule_once(self._after_start, 0.2)
        return manager

    def _after_start(self, *_):
        self.shots_dir = Path(self.user_data_dir) / "shots"
        self.shots_dir.mkdir(parents=True, exist_ok=True)
        self._load_history()

        if not self.model_path.exists():
            self._set_status("Модель не найдена: assets/model.tflite")
        if IS_ANDROID:
            try:
                from android import activity
                activity.bind(on_activity_result=self._on_activity_result)
            except Exception as error:                      # noqa: BLE001
                self._set_status(f"Связь с системой не удалась: {error}")

    # ---------- вспомогательное ----------

    def _set_status(self, text):
        home = self.manager.get_screen("home")
        home.status = text

    def _history_file(self) -> Path:
        return Path(self.user_data_dir) / "history.json"

    def _load_history(self):
        try:
            self.history = json.loads(self._history_file().read_text(encoding="utf-8"))
        except Exception:
            self.history = []

    def _save_history(self):
        try:
            self._history_file().write_text(
                json.dumps(self.history, ensure_ascii=False, indent=1), encoding="utf-8")
        except Exception:
            pass

    def clear_history(self):
        self.history = []
        self._save_history()

    def open_history_item(self, item):
        screen = self.manager.get_screen("result")
        screen.expression = item["expression"]
        screen.answer = item["answer"]
        screen.steps_text = item.get("steps", "")
        self.manager.current = "result"

    # ---------- съёмка и выбор фото ----------

    def take_photo(self):
        if not IS_ANDROID:
            self._set_status("Съёмка доступна на телефоне — на компьютере выбери «Из галереи»")
            return
        try:
            from jnius import autoclass

            PythonActivity = autoclass("org.kivy.android.PythonActivity")
            Intent = autoclass("android.content.Intent")
            MediaStore = autoclass("android.provider.MediaStore")
            ContentValues = autoclass("android.content.ContentValues")

            activity = PythonActivity.mActivity
            values = ContentValues()
            values.put("title", "MathSnap")
            uri = activity.getContentResolver().insert(
                MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values)
            if uri is None:
                self._set_status("Камера не дала место для снимка")
                return

            self.pending_uri = str(uri)
            intent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
            intent.putExtra(MediaStore.EXTRA_OUTPUT, uri)
            activity.startActivityForResult(intent, REQUEST_CAMERA)
            self._set_status("Открываю камеру…")
        except Exception as error:                          # noqa: BLE001
            self._set_status(f"Камера недоступна: {error}")

    def open_gallery(self):
        if not IS_ANDROID:
            self._set_status("На компьютере положи фото рядом и набери пример руками")
            return
        try:
            from jnius import autoclass

            PythonActivity = autoclass("org.kivy.android.PythonActivity")
            Intent = autoclass("android.content.Intent")

            activity = PythonActivity.mActivity
            intent = Intent(Intent.ACTION_GET_CONTENT)
            intent.setType("image/*")
            intent.addCategory(Intent.CATEGORY_OPENABLE)
            activity.startActivityForResult(
                Intent.createChooser(intent, "Выбери фото с примером"), REQUEST_GALLERY)
            self._set_status("Открываю галерею…")
        except Exception as error:                          # noqa: BLE001
            self._set_status(f"Галерея недоступна: {error}")

    def _copy_uri(self, uri: str, target: Path) -> bool:
        """Копирует картинку из content:// в папку приложения.

        Дескриптор файла забираем у системы и открываем обычным Python-файлом:
        так фото на 2–3 МБ копируется мгновенно, а не по байту через JNI.
        """
        try:
            from jnius import autoclass

            PythonActivity = autoclass("org.kivy.android.PythonActivity")
            Uri = autoclass("android.net.Uri")

            resolver = PythonActivity.mActivity.getContentResolver()
            descriptor = resolver.openFileDescriptor(Uri.parse(uri), "r")
            if descriptor is None:
                return False

            handle = descriptor.detachFd()
            with os.fdopen(handle, "rb") as source, open(target, "wb") as destination:
                shutil.copyfileobj(source, destination, 1 << 16)
            return target.exists() and target.stat().st_size > 0
        except Exception as error:                          # noqa: BLE001
            self._set_status(f"Фото не сохранилось: {error}")
            return False

    def _on_activity_result(self, request_code, result_code, intent):
        if result_code != -1:            # RESULT_OK
            self._set_status("Отменено")
            return

        uri = ""
        if request_code == REQUEST_CAMERA:
            uri = self.pending_uri
        elif request_code == REQUEST_GALLERY and intent is not None:
            data = intent.getData()
            if data is not None:
                uri = str(data)
        self.pending_uri = ""

        if not uri:
            self._set_status("Фото не получено")
            return

        target = (self.shots_dir or Path(self.user_data_dir)) / \
            f"shot_{datetime.now():%Y%m%d_%H%M%S}.jpg"
        if self._copy_uri(uri, target):
            self._recognize(target)
        else:
            self._set_status("Не удалось сохранить фото")

    # ---------- распознавание ----------

    def _recognize(self, path: Path):
        home = self.manager.get_screen("home")
        home.preview_path = str(path)
        self._set_status("Распознаю символы…")

        def worker():
            try:
                expression, details = read_expression(path, self.model_path)
            except Exception as error:                      # noqa: BLE001
                message = f"Распознавание не удалось: {error}"
                Clock.schedule_once(lambda *_: self._set_status(message), 0)
                return
            Clock.schedule_once(
                lambda *_: self._show_expression(expression, details), 0)

        threading.Thread(target=worker, daemon=True).start()

    @mainthread
    def _show_expression(self, expression, details):
        home = self.manager.get_screen("home")
        home.expression = expression
        home.ids.expression_input.text = expression

        count = details.get("count", 0)
        note = f"Распознано символов: {count}"
        weak = [item for item in details.get("symbols", []) if item["score"] < 0.6]
        if weak:
            shown = ", ".join(f"{item['symbol']} — {item['score']:.2f}" for item in weak[:6])
            note += f"\nСомнительные: {shown}"
        home.note = note

        if count:
            self._set_status("Проверь распознанное и нажми «Решить»")
        else:
            self._set_status("Символы не нашлись: снимай ровно, на светлом фоне")

    # ---------- решение ----------

    def solve_text(self, text: str):
        text = (text or "").strip()
        if not text:
            self._set_status("Сначала введи или сфотографируй пример")
            return

        result = solve(text)
        screen = self.manager.get_screen("result")
        screen.expression = text

        if result["error"] and not result["answer"]:
            screen.answer = "Не решено"
            screen.steps_text = result["error"]
        else:
            screen.answer = result["answer"]
            lines = []
            for index, (title, detail) in enumerate(result["steps"], start=1):
                lines.append(f"{index}. {title}")
                if detail:
                    lines.append(f"      {detail}")
            screen.steps_text = "\n".join(lines)
            self._add_history(text, result["answer"], screen.steps_text)

        self.manager.current = "result"

    def _add_history(self, expression, answer, steps):
        self.history.append({
            "when": datetime.now().strftime("%d.%m %H:%M"),
            "expression": expression,
            "answer": answer,
            "steps": steps,
        })
        self.history = self.history[-HISTORY_LIMIT:]
        self._save_history()


if __name__ == "__main__":
    MathSnapApp().run()
