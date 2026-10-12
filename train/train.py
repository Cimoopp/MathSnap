"""Обучение своей CNN для распознавания рукописных символов MathSnap.

Источники данных:
  * HASYv2 — реальные рукописные символы (качает data/download_data.py);
  * MNIST — рукописные цифры 0..9 (data/extra_data.py);
  * EMNIST letters — рукописные x, y, l, g (data/extra_data.py);
  * отрисовка шрифтом — знаки +, -, =, (, ), ., /, ^, sqrt, по которым
    рукописных примеров почти нет (data/render_glyphs.py);
  * свои фото из data/handwritten/ — папка на каждый класс.

Выход: assets/model.tflite (лежит внутри APK) и отчёт с метриками в train/out.

Почему предыдущий прогон дал 15,94 %. В обучение попало 2598 символов,
из них 1200 — три самых частых класса, срезанных пределом 400, а пять
классов (=, (, ), ., ^) остались вообще без примеров. При таком объёме
сеть заучила выборку: обучающая точность выросла до 82 %, проверочная
упала с 16,4 % до 3,2 %, лучшей оказалась первая же эпоха. Что изменено:

  1) подключены MNIST, EMNIST и отрисовка знаков — пустых классов больше нет;
  2) предел на класс поднят 400 -> 1200 и применяется ко всем источникам
     сразу, а не по отдельности;
  3) редкие классы получают вес, обратный частоте (sample_weights в tf.data);
  4) деление на обучение/проверку/тест стратифицированное: каждый класс
     попадает во все три части, проверка больше не зависит от везения;
  5) терпение ранней остановки 6 -> 12 эпох;
  6) в отчёт добавлены источники, примеры по классам и точность по классам.

Запуск локально:
    pip install -r requirements.txt
    python data/download_data.py --out data/hasy
    python data/inspect_dataset.py --data data
    python train/train.py --epochs 60 --per-class-cap 1200
В облаке это делает .github/workflows/train.yml
"""

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

# Чтобы import ocr.* и data.* работал и при запуске из корня, и из папки train
sys.path.append(str(Path(__file__).resolve().parents[1]))

from data.extra_data import load_emnist_letters, load_mnist  # noqa: E402
from data.render_glyphs import render_dataset  # noqa: E402
from ocr.labels import (  # noqa: E402
    CLASS_TO_INDEX,
    CLASSES,
    HUMAN_NAMES,
    IMG_SIZE,
    NUM_CLASSES,
)
from ocr.model import build_model  # noqa: E402

# Как метки LaTeX из HASYv2 переводятся в наши классы.
# Всё, чего здесь нет, при обучении отбрасывается.
LATEX_MAP = {
    "0": "0", "1": "1", "2": "2", "3": "3", "4": "4",
    "5": "5", "6": "6", "7": "7", "8": "8", "9": "9",
    "+": "+", "-": "-", "=": "=", "(": "(", ")": ")",
    "x": "x", "y": "y", ".": ".",
    "/": "/", r"\cdot": "x", r"\times": "x", r"\div": "/",
    r"\sqrt": "sqrt", r"\sqrt{}": "sqrt", "^": "^",
    "l": "l", "g": "g",
}

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")

# Имена колонок, которые встречаются в разных выгрузках датасета
PATH_EXACT = ("path", "filepath", "file_name", "filename", "file",
              "image_path", "img_path", "png", "image")
PATH_PARTIAL = ("path", "file")
ID_EXACT = ("image_id", "imageid", "img_id", "imgid", "id")
LABEL_EXACT = ("symbol_id", "symbolid", "symbol", "label",
               "class", "class_id", "label_id", "target")

# Ниже этого числа примеров класс считается редким и попадает в предупреждения
LOW_COUNT = 200


def empty_batch():
    """Пустая порция данных — общий ответ для недоступного источника."""
    return (np.zeros((0, IMG_SIZE, IMG_SIZE), dtype=np.float32),
            np.zeros((0,), dtype=np.int64))


def norm_row(row) -> dict:
    """Приводит имена колонок строки к нижнему регистру и убирает пробелы."""
    result = {}
    for key, value in (row or {}).items():
        if key is None:
            continue
        result[key.strip().lower()] = (value or "").strip()
    return result


def read_header(path: Path):
    """Возвращает нормализованные имена колонок таблицы."""
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        return [h.strip().lower() for h in next(csv.reader(f), [])]


def as_int(value):
    """Число из строки или None, если это не число."""
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def pick_exact(row: dict, names):
    for name in names:
        value = row.get(name)
        if value:
            return value
    return ""


def pick_contains(row: dict, fragments):
    for key, value in row.items():
        if value and any(fragment in key for fragment in fragments):
            return value
    return ""


def latex_to_class(latex):
    """Переводит latex-метку в наш класс или None, если символ не нужен."""
    if not latex:
        return None
    key = latex.strip().strip("$")
    for junk in (r"\left", r"\right", r"\,"):
        key = key.replace(junk, "")
    return LATEX_MAP.get(key.strip())


def read_symbols_map(path: Path) -> dict:
    """Читает таблицу символов: номер символа -> LaTeX."""
    table = {}
    header = read_header(path)
    latex_key = next((name for name in header if "latex" in name), None)
    id_key = next((name for name in header if name in
                   ("symbol_id", "symbolid", "symbol", "id", "class_id")), None)
    if latex_key is None or id_key is None:
        return table

    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            row = norm_row(row)
            number = as_int(row.get(id_key, ""))
            latex = row.get(latex_key, "")
            if number is not None and latex:
                table[number] = latex
    return table


def build_label_getter(header, symbols_map):
    """Собирает функцию «строка таблицы -> LaTeX-метка».

    Метка может лежать в колонке latex, либо в колонке с номером символа
    (тогда номер расшифровывается через symbols_map).
    """
    latex_key = next((name for name in header if "latex" in name), None)
    label_key = next((name for name in header if name in LABEL_EXACT), None)
    if label_key is None:
        label_key = next((name for name in header
                          if any(f in name for f in ("symbol", "label", "class", "target"))
                          and "latex" not in name), None)
    if latex_key is None and label_key is None:
        return None

    def getter(row: dict):
        for key in (latex_key, label_key):
            if not key:
                continue
            value = row.get(key, "")
            if not value:
                continue
            number = as_int(value)
            if number is None:
                return value
            mapped = symbols_map.get(number)
            if mapped:
                return mapped
        return None

    return getter


def pick_label_csv(hasy_dir: Path):
    """Выбирает таблицу с метками: берёт ту, где совпало больше всего символов.

    Такой выбор не зависит от порядка файлов в архиве и от того, как
    названы колонки. Служебные разбивки (classification-task) обычно
    содержат меньше наших символов и проигрывают полной таблице.
    """
    csv_files = sorted(hasy_dir.rglob("*.csv"))
    if not csv_files:
        return None, {}

    # Сначала ищем таблицу-расшифровку «номер символа -> LaTeX»
    symbols_map = {}
    for path in csv_files:
        header = read_header(path)
        is_plain_table = (not any(name in PATH_EXACT or any(f in name for f in PATH_PARTIAL)
                                  for name in header))
        if is_plain_table and any("latex" in name for name in header):
            table = read_symbols_map(path)
            if len(table) > len(symbols_map):
                symbols_map = table
                print(f"[i] Расшифровка символов: {path.name} — {len(table)} записей")

    best_path, best_matched = None, 0
    for path in csv_files:
        header = read_header(path)
        if not header:
            continue

        has_path = any(name in PATH_EXACT or any(f in name for f in PATH_PARTIAL)
                       for name in header)
        has_id = any(name in ID_EXACT for name in header)
        if not (has_path or has_id):
            print(f"[i] {path.relative_to(hasy_dir)} — не про картинки, пропускаю")
            continue

        getter = build_label_getter(header, symbols_map)
        if getter is None:
            print(f"[i] {path.relative_to(hasy_dir)} — колонки с метками нет, пропускаю")
            continue

        total = matched = 0
        with open(path, newline="", encoding="utf-8", errors="replace") as f:
            for row in csv.DictReader(f):
                total += 1
                if latex_to_class(getter(norm_row(row))) is not None:
                    matched += 1

        print(f"[i] {path.relative_to(hasy_dir)}: строк {total}, наших символов {matched}")
        if matched > best_matched:
            best_path, best_matched = path, matched

    return best_path, symbols_map


def build_image_indexes(hasy_dir: Path):
    """Индексы картинок: по имени файла и по числу из имени (v2-00042 -> 42)."""
    by_name = {}
    by_digits = {}
    for path in hasy_dir.rglob("*"):
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        by_name.setdefault(path.name, path)
        digits = re.sub(r"\D", "", path.stem)
        if digits:
            by_digits.setdefault(int(digits), path)
    return by_name, by_digits


def resolve_image(row: dict, hasy_dir: Path, indexes):
    """Ищет файл картинки для одной строки таблицы."""
    by_name, by_digits = indexes

    raw = pick_exact(row, PATH_EXACT) or pick_contains(row, PATH_PARTIAL)
    if raw:
        name = Path(raw.replace("\\", "/")).name
        hit = by_name.get(name)
        if hit is not None:
            return hit
        candidate = hasy_dir / raw
        if candidate.exists():
            return candidate

    number = as_int(pick_exact(row, ID_EXACT))
    if number is not None:
        return by_digits.get(number)
    return None


def load_glyph(path: Path, size: int = IMG_SIZE):
    """Готовит одно изображение символа к подаче в сеть.

    Шаги: оттенки серого -> бинаризация -> обрезка по границам символа ->
    вписывание в квадрат с сохранением пропорций -> инверсия
    (символ белым по чёрному) -> нормализация 0..1.
    Ровно тот же вид должен получаться в приложении после съёмки камерой.
    """
    # Символ занимает 3/4 кадра: так у сети остаётся запас на неточную обрезку
    inner = max(16, int(size * 0.75))

    image = Image.open(path).convert("L")
    arr = np.asarray(image)

    mask = arr < 160  # символ темнее фона
    if not mask.any():
        mask = arr > 96  # на всякий случай: символ светлее фона
    if not mask.any():
        return None

    rows, cols = np.where(mask)
    crop = arr[rows.min():rows.max() + 1, cols.min():cols.max() + 1]

    glyph = 255 - crop  # инверсия: символ белый
    height, width = glyph.shape
    scale = inner / max(height, width)
    new_h = max(1, int(round(height * scale)))
    new_w = max(1, int(round(width * scale)))

    resized = Image.fromarray(glyph).resize((new_w, new_h), Image.LANCZOS)
    canvas = Image.new("L", (size, size), 0)
    canvas.paste(resized, ((size - new_w) // 2, (size - new_h) // 2))

    return np.asarray(canvas, dtype=np.float32) / 255.0


def load_hasy(csv_path: Path, hasy_dir: Path, symbols_map: dict, per_class_cap: int):
    """Читает датасет: таблица с метками + папка с картинками."""
    indexes = build_image_indexes(hasy_dir)
    print(f"[i] Картинок в датасете: {len(indexes[0])}")

    getter = build_label_getter(read_header(csv_path), symbols_map)
    if getter is None:
        print("[X] В выбранной таблице не нашлось колонки с метками.")
        return (np.zeros((0, IMG_SIZE, IMG_SIZE), dtype=np.float32),
                np.zeros((0,), dtype=np.int64), {})

    features, labels, per_class = [], [], {}
    no_image = 0

    with open(csv_path, newline="", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            row = norm_row(row)

            symbol = latex_to_class(getter(row))
            if symbol is None:
                continue
            if per_class.get(symbol, 0) >= per_class_cap:
                continue

            path = resolve_image(row, hasy_dir, indexes)
            if path is None:
                no_image += 1
                continue

            glyph = load_glyph(path)
            if glyph is None:
                continue

            features.append(glyph)
            labels.append(CLASS_TO_INDEX[symbol])
            per_class[symbol] = per_class.get(symbol, 0) + 1

    if no_image:
        print(f"[!] Для {no_image} строк с нашей меткой не нашлась картинка")

    if features:
        x = np.stack(features)
        y = np.asarray(labels, dtype=np.int64)
    else:
        x, y = empty_batch()
    return x, y, per_class


def load_hasy_cached(csv_path: Path, hasy_dir: Path, symbols_map: dict,
                     per_class_cap: int, cache_path: Path):
    """То же, что load_hasy, но с кэшем на диск: препроцессинг идёт один раз."""
    if cache_path.exists():
        try:
            packed = np.load(cache_path, allow_pickle=False)
            x = packed["x"]
            y = packed["y"]
            per_class = json.loads(str(packed["per_class"]))
            print(f"[i] Готовые символы из кэша: {len(x)} штук ({cache_path.name})")
            return x, y, per_class
        except Exception as error:  # noqa: BLE001 — кэш не критичен, готовим заново
            print(f"[!] Кэш не прочитался ({error}) — готовлю символы заново")

    x, y, per_class = load_hasy(csv_path, hasy_dir, symbols_map, per_class_cap)

    if len(x):
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, x=x, y=y,
                            per_class=np.asarray(json.dumps(per_class)))
        print(f"[i] Символы сохранены в кэш: {cache_path}")
    return x, y, per_class


def load_or_build(cache_path: Path, builder, name: str):
    """Берёт готовый набор из кэша, а если кэша нет — собирает и сохраняет.

    Пустой результат в кэш не пишется: иначе недоступный источник
    закрепился бы и следующий прогон снова получил бы ноль.
    """
    if cache_path.exists():
        try:
            packed = np.load(cache_path, allow_pickle=False)
            x, y = packed["x"], packed["y"]
            print(f"[i] {name}: {len(x)} символов из кэша ({cache_path.name})")
            return x, y
        except Exception as error:  # noqa: BLE001 — кэш не критичен
            print(f"[!] Кэш «{name}» не прочитался ({error}) — готовлю заново")

    x, y = builder()
    if len(x):
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, x=x, y=y)
        print(f"[i] {name}: {len(x)} символов сохранено в кэш ({cache_path.name})")
    return x, y


def safe_source(name: str, builder):
    """Берёт данные источника, не роняя прогон при его недоступности.

    Внешний набор может не скачаться, шрифта может не найтись — из-за этого
    терять весь прогон незачем: работаем на том, что есть, и печатаем причину.
    """
    try:
        return builder()
    except Exception as error:  # noqa: BLE001 — источник не критичен
        print(f"[!] {name}: данные недоступны ({type(error).__name__}: {error})")
        print(f"    Продолжаю обучение без источника «{name}».")
        return empty_batch()


def load_handwritten(root: Path):
    """Свои фото: папка на каждый класс, имя папки = класс из ocr/labels.py."""
    features, labels = [], []

    if not root.exists():
        print(f"[i] Папки со своими фото нет ({root}) — учусь только на наборах.")
        return empty_batch()

    for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        symbol = class_dir.name
        if symbol not in CLASS_TO_INDEX:
            print(f"[!] Папка «{symbol}» не совпадает ни с одним классом — пропускаю.")
            continue
        for path in sorted(class_dir.iterdir()):
            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            glyph = load_glyph(path)
            if glyph is None:
                continue
            features.append(glyph)
            labels.append(CLASS_TO_INDEX[symbol])

    if features:
        x = np.stack(features)
        y = np.asarray(labels, dtype=np.int64)
        print(f"[i] Свои фото: {len(features)} штук")
    else:
        x, y = empty_batch()
    return x, y


def limit_per_class(x, y, cap: int, seed: int = 21):
    """Оставляет не больше cap примеров на класс — по всем источникам сразу.

    Предел применяется здесь, а не внутри каждого набора, иначе по одному
    классу можно было бы набрать 400 из HASYv2 плюс 1200 из MNIST
    и перекосить обучение.
    """
    if cap <= 0 or len(y) == 0:
        return x, y

    rng = np.random.default_rng(seed)
    chosen = []
    for index in range(NUM_CLASSES):
        positions = np.where(y == index)[0]
        if len(positions) > cap:
            positions = rng.choice(positions, size=cap, replace=False)
        chosen.append(positions)

    keep = np.concatenate(chosen)
    keep = rng.permutation(keep)
    return x[keep], y[keep]


def stratified_split(y, val_fraction=0.12, test_fraction=0.08, seed=13):
    """Делит данные на обучение/проверку/тест с сохранением классов.

    Обычный случайный сплит может отдать редкий класс целиком в проверку,
    и тогда точность по нему окажется нулевой при живом обучении.
    Здесь каждый класс делится отдельно, а совсем крошечные классы
    (меньше четырёх примеров) целиком остаются в обучении.
    """
    rng = np.random.default_rng(seed)
    train_parts, val_parts, test_parts = [], [], []

    for index in range(NUM_CLASSES):
        positions = np.where(y == index)[0]
        if len(positions) == 0:
            continue
        rng.shuffle(positions)

        if len(positions) < 4:
            train_parts.append(positions)
            continue

        n_test = min(max(1, int(round(len(positions) * test_fraction))),
                     len(positions) - 3)
        n_val = min(max(1, int(round(len(positions) * val_fraction))),
                     len(positions) - n_test - 2)

        test_parts.append(positions[:n_test])
        val_parts.append(positions[n_test:n_test + n_val])
        train_parts.append(positions[n_test + n_val:])

    def join(parts):
        if not parts:
            return np.zeros((0,), dtype=np.int64)
        return np.sort(np.concatenate(parts))

    return join(train_parts), join(val_parts), join(test_parts)


def class_weights(y, num_classes: int = NUM_CLASSES):
    """Веса классов, обратные частоте: редкий символ весит больше частого.

    Без этого двадцать три класса конкурируют «по числу картинок»: цифры
    из MNIST задавили бы корень, степень и дробную черту.
    """
    counts = np.bincount(y, minlength=num_classes).astype(np.float64)
    present = counts > 0
    weights = np.ones(num_classes, dtype=np.float32)
    if present.any():
        weights[present] = (counts[present].sum() /
                            (present.sum() * counts[present])).astype(np.float32)
    return weights


def make_dataset(x, y, weights=None, batch_size: int = 128,
                 training: bool = False, seed: int = 1):
    """tf.data-пайплайн.

    Аугментация сделана на дешёвых операциях tf.image (яркость, контраст,
    сдвиг на пару пикселей) — на процессоре это в разы быстрее, чем
    поворот и масштабирование слоями Keras при том же эффекте.
    """
    if weights is None:
        weights = np.ones(len(y), dtype=np.float32)

    dataset = tf.data.Dataset.from_tensor_slices((x, y, weights))

    if training:
        dataset = dataset.shuffle(min(len(x), 20000), seed=seed,
                                  reshuffle_each_iteration=True)

        def augment(image, label, weight):
            image = image[..., tf.newaxis]
            image = tf.image.random_brightness(image, max_delta=0.10)
            image = tf.image.random_contrast(image, 0.85, 1.15)
            image = tf.clip_by_value(image, 0.0, 1.0)
            shift = max(2, IMG_SIZE // 8)
            image = tf.pad(image, [[shift, shift], [shift, shift], [0, 0]],
                           constant_values=0.0)
            image = tf.image.random_crop(image, [IMG_SIZE, IMG_SIZE, 1])
            return tf.squeeze(image, -1), label, weight

        dataset = dataset.map(augment, num_parallel_calls=tf.data.AUTOTUNE)

    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def export_tflite(model, out_path: Path) -> int:
    """Конвертирует модель в TensorFlow Lite с квантованием весов."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    data = converter.convert()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(data)
    return len(data)


def main() -> int:
    parser = argparse.ArgumentParser(description="Обучение модели распознавания MathSnap")
    parser.add_argument("--data", default="data", help="папка с данными")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--per-class-cap", type=int, default=1200,
                        help="предел примеров на класс — чтобы частые символы не перевесили")
    parser.add_argument("--patience", type=int, default=12,
                        help="сколько эпох без улучшения ждать до остановки")
    parser.add_argument("--no-extra", action="store_true",
                        help="не подключать MNIST, EMNIST и отрисовку знаков")
    parser.add_argument("--out", default="assets/model.tflite")
    parser.add_argument("--report-dir", default="train/out")
    parser.add_argument("--no-cache", action="store_true",
                        help="игнорировать кэш подготовленных картинок")
    args = parser.parse_args()

    tf.keras.utils.set_random_seed(42)

    data_dir = Path(args.data)
    hasy_dir = data_dir / "hasy"
    cache_dir = data_dir / "cache"
    cap = args.per_class_cap

    if not hasy_dir.exists():
        print(f"[X] Не нашёл папку датасета {hasy_dir}.")
        print("    Сначала: python data/download_data.py --out data/hasy")
        return 1

    label_csv, symbols_map = pick_label_csv(hasy_dir)
    if label_csv is None:
        print("[X] Ни одна таблица не подошла под наши метки. Вот что есть:")
        for path in sorted(hasy_dir.rglob("*.csv")):
            print(f"    - {path.relative_to(hasy_dir)}: {read_header(path)}")
        return 1
    print(f"[i] Таблица меток: {label_csv.relative_to(hasy_dir)}")

    cache_path = cache_dir / f"glyphs_{IMG_SIZE}_{cap}_{label_csv.stem}.npz"
    if args.no_cache and cache_path.exists():
        cache_path.unlink()

    x_hasy, y_hasy, _ = load_hasy_cached(
        label_csv, hasy_dir, symbols_map, cap, cache_path)
    print(f"[i] Из датасета взято {len(x_hasy)} символов")

    sources = {"hasyv2": int(len(x_hasy))}
    parts_x, parts_y = [x_hasy], [y_hasy]

    # ── дополнительные источники ────────────────────────────────────────────
    if not args.no_extra:
        def mnist_builder():
            return load_mnist(per_class_cap=cap)

        x_mnist, y_mnist = load_or_build(
            cache_dir / f"mnist_{IMG_SIZE}_{cap}.npz",
            lambda: safe_source("MNIST (цифры)", mnist_builder),
            "MNIST (цифры)")
        sources["mnist"] = int(len(x_mnist))

        def emnist_builder():
            return load_emnist_letters(per_class_cap=cap)

        x_emnist, y_emnist = load_or_build(
            cache_dir / f"emnist_{IMG_SIZE}_{cap}.npz",
            lambda: safe_source("EMNIST (буквы x/y/l/g)", emnist_builder),
            "EMNIST (буквы x/y/l/g)")
        sources["emnist"] = int(len(x_emnist))

        def rendered_builder():
            x_rendered, y_rendered, _ = render_dataset(per_class_cap=cap)
            return x_rendered, y_rendered

        x_rendered, y_rendered = load_or_build(
            cache_dir / f"rendered_{IMG_SIZE}_{cap}.npz",
            lambda: safe_source("Отрисовка шрифтом (знаки)", rendered_builder),
            "Отрисовка шрифтом (знаки)")
        sources["rendered"] = int(len(x_rendered))

        parts_x += [x_mnist, x_emnist, x_rendered]
        parts_y += [y_mnist, y_emnist, y_rendered]

    x_sets = np.concatenate([part for part in parts_x if len(part)])
    y_sets = np.concatenate([part for part in parts_y if len(part)])
    print(f"[i] Наборы вместе: {len(x_sets)} символов")

    x_sets, y_sets = limit_per_class(x_sets, y_sets, cap)
    print(f"[i] После предела {cap} на класс: {len(x_sets)} символов")

    if len(x_sets) == 0:
        print("[X] Ни один символ не сопоставился с метками.")
        print("    Смотри шаг «Диагностика датасета» в логе прогона.")
        return 1

    # Свои фото добавляются после предела и идут только в обучение:
    # проверять на них бессмысленно (их единицы), а в весах они ценны.
    x_mine, y_mine = load_handwritten(data_dir / "handwritten")
    sources["handwritten"] = int(len(x_mine))

    n_sets = len(x_sets)
    train_idx, val_idx, test_idx = stratified_split(y_sets)
    train_idx = np.concatenate([train_idx,
                               np.arange(n_sets, n_sets + len(x_mine))])

    x = np.concatenate([x_sets, x_mine])
    y = np.concatenate([y_sets, y_mine])

    if len(x) < 500:
        print(f"[X] Данных слишком мало: {len(x)}. Проверь распаковку датасета.")
        return 1

    counts = np.bincount(y, minlength=NUM_CLASSES)
    empty_classes = [symbol for index, symbol in enumerate(CLASSES)
                     if counts[index] == 0]
    rare_classes = [(symbol, int(counts[index]))
                    for index, symbol in enumerate(CLASSES)
                    if 0 < counts[index] < LOW_COUNT]

    steps_per_epoch = max(1, len(train_idx) // args.batch)
    print(f"[i] Всего символов: {len(x)} → обучение {len(train_idx)}, "
          f"проверка {len(val_idx)}, тест {len(test_idx)}")
    print(f"[i] Шагов на эпоху: {steps_per_epoch} (батч {args.batch}, "
          f"вход {IMG_SIZE}x{IMG_SIZE})")

    if empty_classes:
        print(f"[!] Классы без единого примера: {', '.join(empty_classes)}")
    for symbol, count in rare_classes:
        pretty = HUMAN_NAMES.get(symbol, symbol)
        print(f"[!] Мало примеров для «{pretty}»: {count}. "
              f"Добавь свои фото в data/handwritten/{symbol}/")

    weights = class_weights(y[train_idx])
    train_ds = make_dataset(x[train_idx], y[train_idx],
                            weights=weights[y[train_idx]],
                            batch_size=args.batch, training=True)
    val_ds = make_dataset(x[val_idx], y[val_idx], batch_size=args.batch)
    print("[i] Веса редких классов: " + ", ".join(
        f"{HUMAN_NAMES.get(symbol, symbol)}={weights[CLASS_TO_INDEX[symbol]]:.1f}"
        for symbol, count in rare_classes[:6] if count > 0) or "нет")

    model = build_model()
    print(f"[i] Параметров в модели: {model.count_params()}")

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = report_dir / "best.keras"

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(str(checkpoint), monitor="val_accuracy",
                                           save_best_only=True, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy",
                                         patience=args.patience,
                                         restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                             patience=4, min_lr=1e-5, verbose=1),
    ]

    started = time.time()
    history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs,
                        callbacks=callbacks, verbose=2)
    minutes = (time.time() - started) / 60

    # Итоговая проверка на отложенной части
    eval_ds = make_dataset(x[test_idx], y[test_idx], batch_size=args.batch)
    loss, accuracy = model.evaluate(eval_ds, verbose=0)
    print(f"\nТест: точность {accuracy * 100:.2f} %, ошибка {loss:.3f}")
    print(f"Время обучения: {minutes:.1f} мин")

    # Какие символы путаются чаще всего и как обстоит дело по каждому классу
    predictions = model.predict(eval_ds, verbose=0).argmax(axis=1)
    true_labels = y[test_idx]
    mistakes = {}
    for true_index, predicted_index in zip(true_labels, predictions):
        if true_index != predicted_index:
            pair = (CLASSES[true_index], CLASSES[predicted_index])
            mistakes[pair] = mistakes.get(pair, 0) + 1
    top_mistakes = sorted(mistakes.items(), key=lambda item: -item[1])[:12]

    per_class_accuracy = {}
    for index, symbol in enumerate(CLASSES):
        mask = true_labels == index
        if not mask.any():
            continue
        per_class_accuracy[symbol] = {
            "accuracy": round(float((predictions[mask] == index).mean()), 4),
            "samples": int(mask.sum()),
        }

    def pretty(symbol):
        return HUMAN_NAMES.get(symbol, symbol)

    lines = [
        "Отчёт обучения MathSnap",
        "",
        "Источники данных:",
    ]
    lines += [f"  {name}: {count}" for name, count in sources.items()]
    lines += [
        "",
        f"Таблица меток: {label_csv.relative_to(hasy_dir)}",
        f"Предел на класс: {cap}",
        f"Символов всего: {len(x)} → обучение {len(train_idx)}, "
        f"проверка {len(val_idx)}, тест {len(test_idx)}",
        f"Классов: {len(CLASSES)}, без примеров: "
        f"{len(empty_classes)}{' (' + ', '.join(empty_classes) + ')' if empty_classes else ''}",
        f"Вход: {IMG_SIZE}x{IMG_SIZE}, параметров: {model.count_params()}",
        f"Эпох пройдено: {len(history.history.get('loss', []))} "
        f"(терпение {args.patience})",
        f"Время обучения: {minutes:.1f} мин",
        f"Точность на тесте: {accuracy * 100:.2f} %",
        "",
        "Примеров по классам:",
    ]
    lines += [f"  {symbol}: {int(counts[CLASS_TO_INDEX[symbol]])}"
              for symbol in CLASSES]
    lines += ["", "Точность по классам:"]
    if per_class_accuracy:
        lines += [f"  {pretty(symbol)}: "
                  f"{value['accuracy'] * 100:.1f} % (примеров {value['samples']})"
                  for symbol, value in per_class_accuracy.items()]
    else:
        lines.append("  нет данных")
    lines += ["", "Частые ошибки (что приняли за что):"]
    lines += [f"  {pretty(a)} → {pretty(b)}: {count} раз"
              for (a, b), count in top_mistakes] or ["  нет"]

    report_text = "\n".join(lines)
    (report_dir / "report.txt").write_text(report_text, encoding="utf-8")

    model.save(report_dir / "model.keras")
    size = export_tflite(model, Path(args.out))

    metrics = {
        "accuracy": round(float(accuracy), 4),
        "loss": round(float(loss), 4),
        "samples": int(len(x)),
        "samples_train": int(len(train_idx)),
        "samples_val": int(len(val_idx)),
        "samples_test": int(len(test_idx)),
        "sources": sources,
        "class_counts": {symbol: int(counts[CLASS_TO_INDEX[symbol]])
                         for symbol in CLASSES},
        "per_class_accuracy": per_class_accuracy,
        "empty_classes": empty_classes,
        "classes": len(CLASSES),
        "img_size": IMG_SIZE,
        "per_class_cap": cap,
        "patience": args.patience,
        "params": int(model.count_params()),
        "epochs_done": len(history.history.get("loss", [])),
        "minutes": round(minutes, 1),
        "tflite_bytes": size,
        "label_csv": str(label_csv.relative_to(hasy_dir)),
        "top_mistakes": [{"from": a, "to": b, "count": c}
                         for (a, b), c in top_mistakes],
    }
    (report_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print(report_text)
    print(f"\nВеса: {size / 1024:.1f} КБ → {args.out}")
    print(f"Отчёт: {report_dir / 'report.txt'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
