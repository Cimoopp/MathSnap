"""Дополнительные источники данных для обучения MathSnap: MNIST и EMNIST.

Зачем они нужны. В HASYv2 своих примеров по цифрам всего 60-130 на класс,
а по буквам x, y, l и g — единицы или ноль. Сеть на таком объёме не учится
ничему: она заучивает выборку, а на проверке сваливается к угадыванию
одного частого класса. Здесь берём два открытых набора:

  * MNIST — 70 000 рукописных цифр (0..9), качается через keras.datasets;
  * EMNIST letters — рукописные строчные латинские буквы, из них нужны
    x, y, l и g.

Про EMNIST отдельно. В keras.datasets этого набора нет — там только MNIST,
Fashion-MNIST и CIFAR. Поэтому прошлый прогон получал по буквам ровно ноль:
вызов tf.keras.datasets.emnist падал с AttributeError, а ошибку молча
съедала обёртка safe_source. Здесь файлы качаются самостоятельно, в формате
idx, с нескольких зеркал, и читаются напрямую через numpy.

Про неполные зеркала. Набор EMNIST состоит из четырёх файлов: картинки
и метки отдельно для train- и test-части. Зеркала отдают их неровно —
например, test-labels на трёх проверенных зеркалах отсутствует, вместо
архива приходит страница-заглушка. Поэтому части теперь независимы:
train- и test-часть загружаются по отдельности, недоступная пропускается
с предупреждением, а не отбрасывает весь набор. Train-части достаточно:
это 88 800 картинок, тогда как раньше из-за одного файла классы x, y, l, g
оставались с 56 примерами из HASYv2.

Все картинки приводятся ровно к тому же виду, что и HASYv2, и что получится
в приложении после съёмки камерой: IMG_SIZE x IMG_SIZE, оттенки серого,
символ белым по чёрному, обрезан по границам и вписан в 75 % кадра.
"""

import gzip
import struct
import urllib.request
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

from ocr.labels import CLASS_TO_INDEX, IMG_SIZE

# Буквы, которые берём из EMNIST letters
EMNIST_LETTERS = ("x", "y", "l", "g")

# Куда складывать скачанные файлы: папка кэшируется в Actions
CACHE_DIR = Path("data/keras")

# Файлы EMNIST letters в формате idx (том же, что у MNIST)
EMNIST_FILES = {
    "train_images": "emnist-letters-train-images-idx3-ubyte.gz",
    "train_labels": "emnist-letters-train-labels-idx1-ubyte.gz",
    "test_images": "emnist-letters-test-images-idx3-ubyte.gz",
    "test_labels": "emnist-letters-test-labels-idx1-ubyte.gz",
}

# Пары «картинки + метки»: train- и test-часть загружаются независимо,
# поэтому отсутствие одной пары не лишает нас второй.
EMNIST_PARTS = (
    ("train_images", "train_labels"),
    ("test_images", "test_labels"),
)

# Зеркала с файлами EMNIST. Официальный источник (NIST) отдаёт набор одним
# архивом gzip.zip на 500 МБ — для прогона это лишние 10-15 минут, поэтому
# берём отдельные файлы. Первое зеркало — Hugging Face, дальше два зеркала
# на GitHub с теми же самыми файлами.
EMNIST_MIRRORS = (
    "https://huggingface.co/datasets/Royc30ne/emnist-letters/resolve/main/{file}",
    "https://raw.githubusercontent.com/aurelienduarte/emnist/master/gzip/{file}",
    "https://raw.githubusercontent.com/akshaybahadur21/"
    "Alphabet-Recognition-EMNIST/master/gzip/{file}",
)


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
    """Цифры 0..9 из MNIST — по per_class_cap примеров на цифру.

    MNIST есть в keras.datasets, поэтому здесь всё просто: набор лежит
    в ~/.keras/datasets и повторно не скачивается.
    """
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    images = np.concatenate([x_train, x_test])
    labels = np.concatenate([y_train, y_test])
    return _collect(images, labels, lambda label: str(int(label)),
                    per_class_cap, "MNIST (цифры)")


def _download(urls, target: Path, attempts: int = 3) -> Path:
    """Скачивает файл с первого ответившего зеркала.

    Качаем в файл с расширением .part и переименовываем только после
    проверки: иначе недокачанный файл сошёл бы за готовый и при следующем
    прогоне набор бы просто не появился.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size > 0:
        print(f"[i] EMNIST: {target.name} уже готов "
              f"({target.stat().st_size // 1024} КБ)")
        return target

    part = target.with_name(target.name + ".part")
    last_error = None

    for url in urls:
        for attempt in range(1, attempts + 1):
            try:
                print(f"[i] EMNIST: качаю {url} — попытка {attempt}")
                with urllib.request.urlopen(url, timeout=180) as response:
                    with open(part, "wb") as out:
                        while True:
                            chunk = response.read(1 << 16)
                            if not chunk:
                                break
                            out.write(chunk)

                payload = part.read_bytes()
                if len(payload) < 1024 or payload[:2] != b"\x1f\x8b":
                    raise ValueError("файл не похож на gzip: обрыв загрузки")

                part.replace(target)
                print(f"[i] EMNIST: {target.name} готов ({len(payload) // 1024} КБ)")
                return target
            except Exception as error:  # noqa: BLE001 — пробуем следующее зеркало
                last_error = error
                print(f"[!] EMNIST: не вышло ({type(error).__name__}: {error})")
                part.unlink(missing_ok=True)

    raise RuntimeError(f"ни одно зеркало не отдало {target.name}: {last_error}")


def _emnist_bytes(key: str) -> bytes:
    """Распакованные байты одного файла EMNIST — из кэша или из сети."""
    name = EMNIST_FILES[key]
    urls = [mirror.format(file=name) for mirror in EMNIST_MIRRORS]
    path = _download(urls, CACHE_DIR / name)
    return gzip.decompress(path.read_bytes())


def _optional_emnist_bytes(key: str):
    """Байты файла EMNIST или None, если файла нет ни на одном зеркале.

    Файлы набора лежат на зеркалах неровно: test-labels, например, отдаёт
    страницу-заглушку вместо архива. Раньше из-за одного такого файла
    отбрасывался весь набор и буквы учились на 56 примерах из HASYv2 —
    отсюда путаница «1» с «l» и точность 75 % по l и g.

    Теперь отсутствие файла лишь выключает соответствующую пару
    «картинки + метки», остальные части набора работают.
    """
    try:
        return _emnist_bytes(key)
    except Exception as error:  # noqa: BLE001 — часть набора не критична
        print(f"[!] EMNIST: {EMNIST_FILES[key]} недоступен "
              f"({type(error).__name__}: {error}) — пропускаю эту часть")
        return None


def _idx_images(payload: bytes):
    """Картинки из idx-файла: заголовок 16 байт, дальше байт на пиксель."""
    magic, count, rows, cols = struct.unpack(">IIII", payload[:16])
    if magic != 2051:
        raise ValueError(f"это не idx-файл картинок: магия {magic}")
    images = np.frombuffer(payload, dtype=np.uint8, offset=16)
    return images.reshape(count, rows, cols)


def _idx_labels(payload: bytes):
    """Метки из idx-файла: заголовок 8 байт."""
    magic, count = struct.unpack(">II", payload[:8])
    if magic != 2049:
        raise ValueError(f"это не idx-файл меток: магия {magic}")
    labels = np.frombuffer(payload, dtype=np.uint8, offset=8)
    return labels.reshape(count)


def save_preview(x, y, path: Path, cols: int = 8, rows: int = 3):
    """Сохраняет сетку случайных образцов набора отдельной картинкой.

    Нужна для проверки глазами: EMNIST отдаёт изображения повёрнутыми
    на 90 градусов, и ошибку в перестановке осей иначе не заметить —
    сеть обучится и на кривых буквах, просто заметно хуже.
    """
    if len(x) == 0:
        return

    tile = IMG_SIZE * 4
    canvas = Image.new("L", (cols * tile, rows * tile), 40)
    rng = np.random.default_rng(3)
    picks = rng.choice(len(x), size=min(rows * cols, len(x)), replace=False)

    for position, index in enumerate(picks):
        sample = Image.fromarray((x[index] * 255).astype(np.uint8))
        sample = sample.resize((tile, tile), Image.NEAREST)
        canvas.paste(sample, ((position % cols) * tile,
                              (position // cols) * tile))

    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
    print(f"[i] EMNIST: образцы для проверки — {path}")


def load_emnist_letters(per_class_cap: int = 800,
                        letters=EMNIST_LETTERS,
                        preview_path="train/out/emnist_preview.png"):
    """Буквы x, y, l, g из EMNIST letters — по per_class_cap на букву.

    Три тонкости EMNIST:
      1) картинки хранятся повёрнутыми и отражёнными, поэтому оси нужно
         переставить, иначе буквы окажутся повёрнутыми на 90 градусов;
      2) метки начинаются с единицы (a = 1), поэтому сдвиг определяется
         по минимальной метке, а не задаётся жёстко;
      3) файлы на зеркалах лежат неровно, поэтому train- и test-часть
         загружаются по отдельности: чего нет — то пропускается.

    Про класс «l» отдельно: в EMNIST буква l идёт вместе с написаниями L и 1,
    так что её примеры шумнее остальных. Поэтому она и подписана в интерфейсе
    как «буква l (для log)», а решает по ней всё равно сборка слова целиком.
    """
    image_parts, label_parts = [], []

    for images_key, labels_key in EMNIST_PARTS:
        images_payload = _optional_emnist_bytes(images_key)
        labels_payload = _optional_emnist_bytes(labels_key)
        if images_payload is None or labels_payload is None:
            continue

        part_images = _idx_images(images_payload)
        part_labels = _idx_labels(labels_payload)
        usable = min(len(part_images), len(part_labels))
        image_parts.append(part_images[:usable])
        label_parts.append(part_labels[:usable])
        print(f"[i] EMNIST: часть «{images_key}» — {usable} картинок")

    if not image_parts:
        raise RuntimeError("ни одна часть EMNIST letters не скачалась")

    images = np.concatenate(image_parts)
    labels = np.concatenate(label_parts)
    images = np.transpose(images, (0, 2, 1))

    letters = {name for name in letters if name in CLASS_TO_INDEX}
    first = int(labels.min())
    alphabet = {first + index: chr(ord("a") + index) for index in range(26)}

    def to_class(value):
        name = alphabet.get(int(value))
        return name if name in letters else None

    x, y = _collect(images, labels, to_class, per_class_cap,
                    f"EMNIST (буквы {'/'.join(sorted(letters))})")

    if preview_path and len(x):
        save_preview(x, y, Path(preview_path))

    return x, y


if __name__ == "__main__":
    # Быстрая проверка: сколько символов даёт каждый набор и как выглядит вид картинки
    for loader in (load_mnist, load_emnist_letters):
        x, y = loader(per_class_cap=50)
        print(f"{loader.__name__}: x={x.shape}, y={y.shape}, "
              f"минимум={x.min():.2f}, максимум={x.max():.2f}")
