"""Распознавание примера на фото: нарезка символов и своя CNN.

Пайплайн: фото -> оттенки серого -> бинаризация по Otsu -> строки ->
символы -> подготовка каждого символа ровно как при обучении ->
model.tflite -> склейка символов в выражение для SymPy.

Препроцессинг повторяет train/train.py: символ обрезается по границам,
вписывается в 3/4 кадра IMG_SIZE и подаётся белым по чёрному фону.
Если менять что-то здесь, надо менять и там: сеть обучилась на таком
виде картинок, и любое расхождение бьёт по точности на телефоне.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from PIL import Image

from ocr.labels import CLASSES, IMG_SIZE

MAX_SIDE = 1400      # фото крупнее не нужно: символы и так большие, а время растёт
MIN_INK = 12         # меньше пикселей — это шум, а не символ
COL_GAP = 2          # разрыв до 2 колонок внутри символа не считается границей
ROW_GAP = 3          # то же по строкам
DIGITS = set("0123456789")
ATOM = r"(?:\d+(?:\.\d+)?|[xy])"
OPERATORS = set("+-=()/")


def _load_gray(path) -> np.ndarray:
    """Читает фото в оттенках серого и уменьшает длинную сторону."""
    image = Image.open(path).convert("L")
    if max(image.size) > MAX_SIDE:
        scale = MAX_SIDE / max(image.size)
        image = image.resize((max(1, int(image.width * scale)),
                              max(1, int(image.height * scale))), Image.BILINEAR)
    return np.asarray(image, dtype=np.uint8)


def otsu_threshold(values: np.ndarray) -> int:
    """Порог Оцу: делит гистограмму там, где части яркости различаются сильнее."""
    hist = np.bincount(values.ravel(), minlength=256).astype(np.float64)
    total = values.size
    levels = np.arange(256)
    weight_low = np.cumsum(hist)
    weight_high = total - weight_low
    sum_low = np.cumsum(levels * hist)
    mean_low = sum_low / np.maximum(weight_low, 1)
    mean_high = (sum_low[-1] - sum_low) / np.maximum(weight_high, 1)
    between = weight_low * weight_high * (mean_low - mean_high) ** 2
    return int(np.argmax(between))


def ink_mask(gray: np.ndarray) -> np.ndarray:
    """True там, где чернила. Работает и для светлого фона, и для тёмного."""
    mask = gray <= otsu_threshold(gray)
    if mask.mean() > 0.5:      # фон темнее символов — фото как негатив
        mask = ~mask
    return mask


def _runs(flags: np.ndarray, gap: int):
    """Участки True с допуском на разрыв в gap пикселей."""
    index = np.flatnonzero(flags)
    if index.size == 0:
        return []
    parts = []
    start = previous = int(index[0])
    for value in index[1:]:
        value = int(value)
        if value - previous > gap + 1:
            parts.append((start, previous + 1))
            start = value
        previous = value
    parts.append((start, previous + 1))
    return parts


def prepare_glyph(patch: np.ndarray, size: int = IMG_SIZE):
    """Готовит один символ к подаче в сеть — тот же вид, что при обучении.

    Обрезка по границам -> инверсия (символ белым) -> вписывание
    в 3/4 кадра с сохранением пропорций -> нормализация 0..1.
    """
    inner = max(8, int(size * 0.75))

    mask = patch < 160
    if not mask.any():
        mask = patch <= otsu_threshold(patch)   # тусклое фото: берём свой порог
    if not mask.any():
        mask = patch > 96
    if not mask.any():
        return None

    rows, cols = np.where(mask)
    crop = patch[rows.min():rows.max() + 1, cols.min():cols.max() + 1]

    glyph = 255 - crop
    height, width = glyph.shape
    scale = inner / max(height, width)
    new_h = max(1, int(round(height * scale)))
    new_w = max(1, int(round(width * scale)))

    resized = Image.fromarray(glyph).resize((new_w, new_h), Image.LANCZOS)
    canvas = Image.new("L", (size, size), 0)
    canvas.paste(resized, ((size - new_w) // 2, (size - new_h) // 2))
    return np.asarray(canvas, dtype=np.float32) / 255.0


def segment(path, size: int = IMG_SIZE):
    """Находит на фото отдельные символы.

    Возвращает список (картинка символа, рамка) в порядке чтения:
    строки сверху вниз, внутри строки слева направо.
    """
    gray = _load_gray(path)
    mask = ink_mask(gray)

    found = []
    for line_top, line_bottom in _runs(mask.any(axis=1), ROW_GAP):
        line = mask[line_top:line_bottom]
        if line.sum() < MIN_INK:
            continue
        for left, right in _runs(line.any(axis=0), COL_GAP):
            column = line[:, left:right]
            if column.sum() < MIN_INK:
                continue
            rows = np.flatnonzero(column.any(axis=1))
            top = line_top + int(rows[0])
            bottom = line_top + int(rows[-1]) + 1
            patch = gray[top:bottom, left:right]
            glyph = prepare_glyph(patch, size)
            if glyph is None:
                continue
            found.append((glyph, (int(left), int(top), int(right), int(bottom))))
    return found


class SymbolReader:
    """Инференс своей CNN. Модель ленивая: грузится при первом символе."""

    def __init__(self, model_path):
        self.model_path = Path(model_path)
        self._interpreter = None
        self._input = None
        self._output = None

    def _ensure_loaded(self):
        if self._interpreter is not None:
            return
        try:
            from tflite_runtime.interpreter import Interpreter
        except ImportError:                      # на компьютере tflite-runtime обычно нет
            import tensorflow as tf
            Interpreter = tf.lite.Interpreter
        interpreter = Interpreter(model_path=str(self.model_path))
        interpreter.allocate_tensors()
        self._interpreter = interpreter
        self._input = interpreter.get_input_details()[0]
        self._output = interpreter.get_output_details()[0]

    def predict(self, glyph: np.ndarray):
        """Возвращает (символ, уверенность) для одной картинки."""
        self._ensure_loaded()
        batch = glyph.reshape(1, IMG_SIZE, IMG_SIZE, 1).astype(np.float32)
        self._interpreter.set_tensor(self._input["index"], batch)
        self._interpreter.invoke()
        scores = self._interpreter.get_tensor(self._output["index"])[0]
        best = int(np.argmax(scores))
        return CLASSES[best], float(scores[best])

    def predict_many(self, glyphs):
        """То же для списка символов — модель загружается один раз."""
        result = []
        for glyph in glyphs:
            result.append(self.predict(glyph))
        return result


def _merge_tokens(symbols):
    """Склеивает соседние цифры в числа, а l + 0 + g — в log."""
    tokens = []
    index = 0
    while index < len(symbols):
        symbol = symbols[index]

        # логические буквы l и g рядом с кружком читаются как log
        if (symbol == "l" and index + 2 < len(symbols)
                and symbols[index + 1] in ("0", "o")
                and symbols[index + 2] == "g"):
            tokens.append(("func", "log"))
            index += 3
            continue

        if symbol in DIGITS or symbol == ".":
            number = symbol
            index += 1
            while index < len(symbols) and (symbols[index] in DIGITS or symbols[index] == "."):
                number += symbols[index]
                index += 1
            tokens.append(("num", number))
            continue

        if symbol == "sqrt":
            tokens.append(("func", "sqrt"))
        elif symbol == "^":
            tokens.append(("pow", "^"))
        elif symbol in ("x", "y"):
            tokens.append(("var", symbol))
        else:
            tokens.append(("op", symbol))
        index += 1
    return tokens


def tokens_to_expression(symbols):
    """Собирает распознанные символы в строку, понятную SymPy.

    Правила: «2x» -> «2*x», «sqrt 25» -> «sqrt(25)», «2^3» -> «2**(3)»,
    «l 0 g 100» -> «log(100)». Слева направо, без перестановок.
    """
    tokens = _merge_tokens(symbols)

    parts = []
    previous = None
    for index, (kind, token) in enumerate(tokens):
        if kind in ("num", "var"):
            if previous in ("num", "var", "close"):
                parts.append("*")
        elif kind == "op" and token == "(" and previous in ("num", "var", "close"):
            parts.append("*")
        parts.append(token if kind != "pow" else "**")

        if kind == "num":
            previous = "num"
        elif kind == "var":
            previous = "var"
        elif kind in ("func", "pow"):
            previous = kind
        elif token == ")":
            previous = "close"
        else:
            previous = None

    text = "".join(parts)
    text = re.sub(r"sqrt\s*(" + ATOM + r")", r"sqrt(\1)", text)
    text = re.sub(r"\*\*\s*(" + ATOM + r")", r"**(\1)", text)
    return text


def read_expression(image_path, model_path, size: int = IMG_SIZE):
    """Главная функция: фото -> (выражение, подробности распознавания)."""
    glyphs = segment(image_path, size)
    if not glyphs:
        return "", {"symbols": [], "count": 0}

    reader = SymbolReader(model_path)
    details = []
    symbols = []
    for glyph, box in glyphs:
        symbol, score = reader.predict(glyph)
        symbols.append(symbol)
        details.append({"symbol": symbol, "score": round(score, 3), "box": box})

    expression = tokens_to_expression(symbols)
    return expression, {"symbols": details, "count": len(symbols)}
