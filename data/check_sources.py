"""Быстрая проверка источников данных перед долгим обучением.

Зачем отдельный файл. Обучение идёт 30-60 минут, и узнать, что EMNIST
не скачался, на тридцатой минуте обидно. Этот скрипт берёт по несколько
примеров из каждого источника, печатает счётчик и причину отказа.
Занимает 1-2 минуты и запускается в workflow до обучения.

Запуск:
    python data/check_sources.py
"""

import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from data.extra_data import load_emnist_letters, load_mnist  # noqa: E402
from data.render_glyphs import find_font_path, render_dataset  # noqa: E402
from ocr.labels import CLASSES, IMG_SIZE  # noqa: E402


def probe(name, builder):
    """Берёт по несколько примеров источника и печатает итог."""
    started = time.time()
    try:
        x, y = builder()
    except Exception as error:  # noqa: BLE001 — показываем причину как есть
        print(f"[X] {name}: не получилось ({type(error).__name__}: {error})")
        return None

    seconds = time.time() - started
    if len(x) == 0:
        print(f"[X] {name}: 0 символов за {seconds:.1f} с")
        return None

    counts = np.bincount(y, minlength=len(CLASSES))
    filled = int((counts > 0).sum())
    print(f"[OK] {name}: {len(x)} символов, классов заполнено "
          f"{filled} из {len(CLASSES)}, за {seconds:.1f} с")

    missing = [symbol for index, symbol in enumerate(CLASSES)
               if counts[index] == 0]
    if missing:
        print(f"     без примеров: {', '.join(missing)}")
    return x, y


def main() -> int:
    print("Проверка источников данных MathSnap")
    print(f"Вход модели: {IMG_SIZE}x{IMG_SIZE}, классов: {len(CLASSES)}")

    font = find_font_path()
    print(f"[{'OK' if font else 'X'}] Шрифт: "
          f"{font or 'не найден — отрисовка знаков пропустится'}")

    hasy = Path("data/hasy")
    print(f"[{'OK' if hasy.exists() else 'X'}] HASYv2: "
          f"{'папка на месте' if hasy.exists() else 'папки нет, шаг датасета не сработал'}")

    probe("MNIST (цифры)", lambda: load_mnist(per_class_cap=5))
    probe("EMNIST (буквы x/y/l/g)",
          lambda: load_emnist_letters(per_class_cap=5, preview_path=None))
    probe("Отрисовка шрифтом (знаки)",
          lambda: render_dataset(per_class_cap=3)[:2])

    print("\nГотово: если у всех источников стоит [OK], можно обучать.")
    print("Если у MNIST или EMNIST стоит [X] — это не ошибка прогона,")
    print("обучение пройдёт на остальных данных, но редкие классы")
    print("останутся слабыми: смотри строки «без примеров» выше.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
