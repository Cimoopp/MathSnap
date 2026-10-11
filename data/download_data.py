"""Скачивание датасета рукописных символов для обучения MathSnap.

Основа — HASYv2: реальные рукописные символы математики, собранные
с живых людей, а не отрисованные шрифтом.
Страница датасета: https://zenodo.org/records/259444

Датасет научный (research use). Если проект пойдёт в магазин приложений,
свою модель лучше дообучить на собственных фото — тогда вопрос лицензии
датасета снимается.

Свои фото кладём в data/handwritten/<класс>/ , например:
    data/handwritten/sqrt/*.png
    data/handwritten/7/*.jpg
Имена папок — ровно как в ocr/labels.py (0..9, +, -, =, (, ), x, y, ., /, sqrt, ^, l, g).
"""

import argparse
import sys
import tarfile
import urllib.request
from pathlib import Path

ARCHIVE_NAME = "HASYv2.tar.bz2"

# Первое зеркало — актуальное, второе — запасное написание адреса у Zenodo.
URLS = [
    "https://zenodo.org/records/259444/files/HASYv2.tar.bz2?download=1",
    "https://zenodo.org/record/259444/files/HASYv2.tar.bz2?download=1",
]


def download(url: str, dest: Path) -> None:
    print(f"Скачиваю: {url}")
    with urllib.request.urlopen(url, timeout=180) as response:
        total = int(response.headers.get("Content-Length") or 0)
        done = 0
        with open(dest, "wb") as f:
            while True:
                chunk = response.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r  {done / 1e6:6.1f} / {total / 1e6:6.1f} МБ", end="")
    print()
    size_mb = dest.stat().st_size / 1e6
    print(f"  скачано {size_mb:.1f} МБ")


def extract(archive: Path, out_dir: Path) -> None:
    print("Распаковываю архив…")
    with tarfile.open(archive, "r:bz2") as tar:
        tar.extractall(out_dir)
    print("  готово")


def main() -> int:
    parser = argparse.ArgumentParser(description="Загрузка датасета рукописных символов")
    parser.add_argument("--out", default="data/hasy", help="куда распаковать датасет")
    parser.add_argument("--archive", default=None,
                        help="путь к уже скачанному HASYv2.tar.bz2 (если качать не нужно)")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    archive = Path(args.archive) if args.archive else out_dir / ARCHIVE_NAME

    if not archive.exists():
        for url in URLS:
            try:
                download(url, archive)
                break
            except Exception as error:  # сеть, лимиты, блокировки
                print(f"  не получилось: {error}")
        else:
            print("\nСкачать не удалось. Что можно сделать:")
            print(f"  1) скачать HASYv2.tar.bz2 вручную: {URLS[0]}")
            print(f"  2) положить файл сюда: {archive}")
            print("  3) повторить: python data/download_data.py --out data/hasy")
            return 1
    else:
        print(f"Архив уже на месте: {archive}")

    extract(archive, out_dir)

    # Ищем учебный CSV и картинки, чтобы показать, что всё на месте.
    csv_files = sorted(out_dir.rglob("*.csv"))
    png_files = list(out_dir.rglob("*.png"))
    print(f"\nCSV-файлов: {len(csv_files)}")
    print(f"PNG-изображений: {len(png_files)}")
    if csv_files:
        print(f"Основной CSV: {csv_files[0]}")
    if png_files and not csv_files:
        print("[!] Картинки есть, а CSV с метками нет — проверь распаковку архива.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
