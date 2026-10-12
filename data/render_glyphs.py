"""Отрисовка знаков шрифтом — для символов, которых нет в рукописных наборах.

Почему так. В HASYv2 по знакам =, (, ), ., / и ^ либо единицы примеров,
либо ноль: рукописных «знаков равенства» люди почти не пишут отдельно.
Эти классы всё равно нужны, потому что без них сеть сваливает ошибки
в похожие символы — именно поэтому в отчёте половина ошибок уходила в sqrt.

Пока своих фото нет, знаки рисуем шрифтом DejaVu с искажениями, чтобы
сеть увидела разнообразие начертаний: поворот, сдвиг, разная толщина штриха,
размер, шум и лёгкое размытие. Позже, когда появятся свои фото, они просто
добавятся в обучение, а отрисовка останется как подстраховка от пустых классов.

Важно: вид картинки на выходе ровно такой же, как у HASYv2, MNIST и EMNIST —
символ белым по чёрному, обрезан и вписан в 75 % кадра IMG_SIZE x IMG_SIZE.
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from data.extra_data import fit_glyph
from ocr.labels import CLASS_TO_INDEX, IMG_SIZE

# Что рисуем: наш класс -> символ для вывода шрифтом.
# Корень берём знаком U+221A: в шрифте он без верхней черты, зато форма узнаваемая.
RENDER_CHARS = {
    "+": "+",
    "-": "-",
    "=": "=",
    "(": "(",
    ")": ")",
    ".": ".",
    "/": "/",
    "^": "^",
    "sqrt": "\u221a",
}

# Шрифты ищем по порядку: DejaVu есть в ubuntu-раннере после установки
# пакета fonts-dejavu-core, остальные — запасные варианты.
FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
)

# Рисуем на увеличенном холсте: после поворота и сдвига края не обрезаются
CANVAS = 112


def find_font_path():
    """Возвращает путь к первому доступному шрифту или None."""
    for path in FONT_CANDIDATES:
        try:
            ImageFont.truetype(path, 40)
            return path
        except OSError:
            continue
    return None


def render_one(char: str, font_path: str, rng):
    """Рисует один знак со случайными искажениями. Возвращает вид 0..1 или None."""
    font_size = int(rng.integers(70, 104))
    font = ImageFont.truetype(font_path, font_size)

    image = Image.new("L", (CANVAS, CANVAS), 0)
    draw = ImageDraw.Draw(image)
    stroke = int(rng.integers(0, 3))  # толщина штриха: тонкий, обычный, жирный
    draw.text((CANVAS // 2, CANVAS // 2), char, font=font, fill=255,
              anchor="mm", stroke_width=stroke, stroke_fill=255)

    # Наклон: у рукописных знаков почти всегда есть наклон
    angle = float(rng.uniform(-14.0, 14.0))
    image = image.rotate(angle, resample=Image.BICUBIC, fillcolor=0)

    # Сдвиг: сеть не должна требовать идеального центра
    dx = int(rng.integers(-8, 9))
    dy = int(rng.integers(-8, 9))
    image = image.transform(image.size, Image.AFFINE,
                            (1, 0, -dx, 0, 1, -dy), fillcolor=0)

    arr = np.asarray(image, dtype=np.float32)
    # Шум и лёгкое размытие: без них шрифт слишком «печатный» и сеть
    # привяжется к идеальным линиям, а на фото будет теряться
    arr = np.clip(arr + rng.normal(0.0, 6.0, arr.shape), 0.0, 255.0)
    if rng.random() < 0.5:
        blurred = Image.fromarray(arr.astype(np.uint8)).filter(
            ImageFilter.GaussianBlur(float(rng.uniform(0.3, 0.9))))
        arr = np.asarray(blurred, dtype=np.float32)

    return fit_glyph(arr)


def render_dataset(per_class_cap: int = 400, seed: int = 7):
    """Готовит отрисованные знаки. Возвращает x, y и счётчик по классам."""
    font_path = find_font_path()
    if font_path is None:
        print("[!] Шрифтов не нашлось — отрисованные знаки пропускаю.")
        print("    Проверь пакет fonts-dejavu-core в раннере.")
        return (np.zeros((0, IMG_SIZE, IMG_SIZE), dtype=np.float32),
                np.zeros((0,), dtype=np.int64), {})

    rng = np.random.default_rng(seed)
    features, classes, per_class = [], [], {}

    for symbol, char in RENDER_CHARS.items():
        if symbol not in CLASS_TO_INDEX:
            continue
        for _ in range(per_class_cap):
            glyph = render_one(char, font_path, rng)
            if glyph is None:
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

    breakdown = ", ".join(f"{key}:{per_class[key]}" for key in sorted(per_class))
    print(f"[i] Отрисовка шрифтом: {len(features)} символов ({breakdown})")
    return x, y, per_class


if __name__ == "__main__":
    x, y, stats = render_dataset(per_class_cap=5)
    print(f"Проверка: x={x.shape}, классы={sorted(stats)}")
