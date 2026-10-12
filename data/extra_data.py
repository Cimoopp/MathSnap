"""Дополнительные источники данных для обучения MathSnap: MNIST и EMNIST.

Зачем они нужны. В HASYv2 своих примеров по цифрам всего 60-130 на класс,
а по буквам x, y, l и g — единицы или ноль. Сеть на таком объёме не учится
ничему: она заучивает выборку, а на проверке сваливается к угадыванию
одного частого класса. Здесь берём два открытых набора:

  * MNIST — 70 000 рукописных цифр (0..9);
  * EMNIST letters — рукописные строчные латинские буквы, из них нам нужны
    x, y, l и g (для log и переменных).

Оба набора скачиваются через keras.datasets и кэшируются в ~/.keras/datasets,
поэтому повторный прогон берёт их с диска и не тратит время на загрузку.

Все картинки приводятся ровно к тому же виду, что и HASYv2, и что получится
в приложении после съёмки камерой: IMG_SIZE x IMG_SIZE, оттенки серого,
символ белым по чёрному, обрезан по границам и вписан в 75 % кадра.
"""

import numpy as np
import tensorflow as tf
from PIL import Image

from ocr.labels import CLASS_TO_INDEX, IMG_SIZE

# Буквы, которые берём из EMNIST letters
EMNIST_LETTERS = ("x", "y", "l", "g")


def fit_glyph(image, size: int = IMG_SIZE):
    """Приводит картинку символа к нашему виду.

    Шаги: обрезка по границам символа -> вписывание в квадрат с сохранением
    пропорций -> центрирование на чёрном фоне. Возвращает массив float32
    в диапазоне 0..1 или None, если в картинке нет символа.

    Тот же самый вид должен получаться в приложении: иначе сеть будет видеть
    на телефоне не то, на чём училась.
    """
    arr = np.asarray(image, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[..., 0]
    if arr.size == 0:
        return None
    if arr.max() <= 1.0:
        arr = arr * 255.0

    mask = arr > 64  # MNIST и EMNIST хранят символ светлым на тёмном фоне
    if not mask.any():
        return None

    rows, cols = np.where(mask)
    crop = arr[rows.min():rows.max() + 1, cols.min():cols.max() + 1]

    inner = max(16, int(size * 0.75))
    height, width = crop.shape
    scale = inner / max(height, width)
    new_h = max(1, int(round(height * scale)))
    new_w = max(1, int(round(width * scale)))

    resized = Image.fromarray(crop.astype(np.uint8)).resize(
        (new_w, new_h), Image.LANCZOS)
    canvas = Image.new("L", (size, size), 0)
    canvas.paste(resized, ((size - new_w) // 2, (size - new_h) // 2))

    return np.asarray(canvas, dtype=np.float32) / 255.0


def _collect(images, labels, to_class, per_class_cap, name):
    """Готовит картинки набора: переводит метки в наши классы и держит предел.

    to_class возвращает имя класса из ocr/labels.py или None, если символ
    нам не нужен. per_class_cap ограничивает число примеров на класс, чтобы
    частые символы не перевесили редкие.
    """
    features, classes, per_class = [], [], {}
    unusable = 0

    for image, label in zip(images, labels):
        symbol = to_class(label)
        if symbol is None or symbol not in CLASS_TO_INDEX:
            continue
        if per_class.get(symbol, 0) >= per_class_cap:
            continue

        glyph = fit_glyph(image)
        if glyph is None:
            unusable += 1
            continue

        features.append(glyph)
        classes.append(CLASS_TO_INDEX[symbol])
        per_class[symbol] = per_class.get(symbol, 0) + 1

    if features:
        x = np.stack(features).astype(np.float32)
        y = np.asarray(classes, dtype=np.int64)
    else:
        x = np.zeros((0, IMG_SIZE, IMG_SIZE), dtype=np.float32)
        y = np.zeros((0,), dtype=np.int64)

    breakdown = ", ".join(f"{key}:{per_class[key]}"
                          for key in sorted(per_class))
    print(f"[i] {name}: взято {len(features)} символов ({breakdown})")
    if unusable:
        print(f"[i] {name}: пустых картинок пропущено: {unusable}")
    return x, y


def load_mnist(per_class_cap: int = 800):
    """Цифры 0..9 из MNIST — по per_class_cap примеров на цифру."""
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    images = np.concatenate([x_train, x_test])
    labels = np.concatenate([y_train, y_test])
    return _collect(images, labels, lambda label: str(int(label)),
                    per_class_cap, "MNIST (цифры)")


def load_emnist_letters(per_class_cap: int = 800,
                        letters=EMNIST_LETTERS):
    """Буквы x, y, l, g из EMNIST letters — по per_class_cap на букву.

    Две тонкости EMNIST:
      1) в keras-версии картинки лежат транспонированными, поэтому их нужно
         переставить осями, иначе буквы окажутся повёрнутыми на 90 градусов;
      2) метки бывают сдвинуты на единицу (a = 1), поэтому сдвиг определяется
         по минимальной метке, а не задаётся жёстко.

    Про класс «l» отдельно: в EMNIST буква l идёт вместе с написаниями L и 1,
    так что её примеры шумнее остальных. Поэтому она и подписана в интерфейсе
    как «буква l (для log)», а решает по ней всё равно сборка слова целиком.
    """
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.emnist.load_data(
        "letters")
    images = np.transpose(np.concatenate([x_train, x_test]), (0, 2, 1))
    labels = np.concatenate([y_train, y_test])

    letters = {name for name in letters if name in CLASS_TO_INDEX}
    first = int(labels.min())
    alphabet = {first + index: chr(ord("a") + index) for index in range(26)}

    def to_class(value):
        name = alphabet.get(int(value))
        return name if name in letters else None

    return _collect(images, labels, to_class, per_class_cap,
                    f"EMNIST (буквы {'/'.join(sorted(letters))})")


if __name__ == "__main__":
    # Быстрая проверка: сколько символов даёт каждый набор и как выглядит вид картинки
    for loader in (load_mnist, load_emnist_letters):
        x, y = loader(per_class_cap=50)
        print(f"{loader.__name__}: x={x.shape}, y={y.shape}, "
              f"минимум={x.min():.2f}, максимум={x.max():.2f}")
