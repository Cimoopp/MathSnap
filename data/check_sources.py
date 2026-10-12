"""Быстрая проверка источников данных перед долгим обучением.

Зачем отдельный файл. Обучение идёт 30-60 минут, и узнать, что EMNIST
не скачался, на тридцатой минуте обидно. Этот скрипт берёт по несколько
примеров из каждого источника, печатает счётчик и причину отказа.
Занимает 1-2 минуты и запускается в workflow до обучения.

Что именно печатается по EMNIST, если он не подключился. Причина отказа
сохраняется в DATA/extra_data.py и попадает в train/out/emnist_diag.txt:
отдельной строкой видно, какое зеркало ответило, и чем именно ответ
не устроил — Git LFS-указатель, веб-страница вместо архива или обрыв
загрузки. Раньше в отчёте был просто ноль, и причину приходилось искать
наугад.

Запуск:
    python data/check_sources.py
"""

import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from data.extra_data import (  # noqa: E402
    EMNIST_MIRRORS,
    emnist_notes,
    load_emnist_letters,
    load_mnist,
    write_emnist_diag,
)
from data.render_glyphs import find_font_path, render_dataset  # noqa: E402
from ocr.labels import CLASSES, IMG_SIZE  # noqa: E402

# Куда записывается отчёт проверки: файл коммитится в репозиторий
CHECK_REPORT = Path("train/out/sources_check.txt")


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


def report_emnist():
    """Печатает состояние EMNIST и сохраняет причины отказа в файл.

    Отдельная функция нужна потому, что этот источник — самый хрупкий:
    его файлы лежат на одном зеркале, а прежние адреса отдавали
    Git LFS-указатель и веб-страницы. Молчаливый ноль по EMNIST уже один раз
    стоил прогона: буквы y, l и g учились на 56 примерах, и сеть путала
    «1» с «l».
    """
    print("\nEMNIST letters:")
    for mirror in EMNIST_MIRRORS:
        print(f"  зеркало: {mirror}")
    notes = emnist_notes()
    if notes:
        print(f"  причины отказа: {notes}")
    else:
        print("  отказов не зафиксировано")

    try:
        write_emnist_diag()
        print("  подробности сохранены в train/out/emnist_diag.txt")
    except Exception as error:  # noqa: BLE001 — файл не критичен для проверки
        print(f"  [!] диагностику записать не удалось: {error}")


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

    report_emnist()

    print("\nГотово: если у всех источников стоит [OK], можно обучать.")
    print("Если у MNIST или EMNIST стоит [X] — это не ошибка прогона,")
    print("обучение пройдёт на остальных данных, но редкие классы")
    print("останутся слабыми: смотри строки «без примеров» выше.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
