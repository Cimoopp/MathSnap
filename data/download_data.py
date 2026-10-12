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

Про устойчивость: сеть в облаке рвётся, и раньше это ломало прогоны
насмерть — частично скачанный файл считался готовым, распаковка падала,
а кэш закреплял мусор, из-за чего следующий прогон падал по той же причине.
Теперь файл сначала качается в .part, потом проверяется как архив, и только
потом занимает своё место. Битый архив удаляется и качается заново.
"""

import argparse
import shutil
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

ARCHIVE_NAME = "HASYv2.tar.bz2"

# Меньше этого числа картинок — значит распаковалось не всё: у HASYv2 их около 168 тысяч.
MIN_IMAGES = 20000

# Первое зеркало — актуальное, второе — запасное написание адреса у Zenodo.
URLS = [
    "https://zenodo.org/records/259444/files/HASYv2.tar.bz2?download=1",
    "https://zenodo.org/record/259444/files/HASYv2.tar.bz2?download=1",
]

ATTEMPTS_PER_URL = 3
MIN_ARCHIVE_BYTES = 1_000_000

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


def is_usable_archive(path: Path) -> bool:
    """Проверяет, что файл — целый архив, а не обрывок скачивания."""
    try:
        if not path.exists() or path.stat().st_size < MIN_ARCHIVE_BYTES:
            return False
        return tarfile.is_tarfile(path)
    except (OSError, tarfile.TarError):
        return False


def download_once(url: str, dest: Path) -> None:
    """Одна попытка скачивания. Пишет в .part, готовый файл подменяет в конце."""
    print(f"Скачиваю: {url}")
    part = dest.with_name(dest.name + ".part")

    with urllib.request.urlopen(url, timeout=180) as response:
        total = int(response.headers.get("Content-Length") or 0)
        done = 0
        with open(part, "wb") as f:
            while True:
                chunk = response.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r  {done / 1e6:6.1f} / {total / 1e6:6.1f} МБ", end="")
    print()

    if not is_usable_archive(part):
        part.unlink(missing_ok=True)
        raise ValueError("скачанный файл не читается как архив")
    if total and done < total:
        part.unlink(missing_ok=True)
        raise ValueError(f"скачано {done} из {total} байт — обрыв")

    part.replace(dest)
    print(f"  скачано {dest.stat().st_size / 1e6:.1f} МБ")


def fetch_archive(dest: Path) -> bool:
    """Качает архив, перебирая зеркала и повторы. True — архив на месте и целый."""
    for url in URLS:
        for attempt in range(1, ATTEMPTS_PER_URL + 1):
            try:
                download_once(url, dest)
                return True
            except Exception as error:  # сеть, лимиты, блокировки, обрыв
                print(f"  попытка {attempt} не удалась: {error}")
                if attempt < ATTEMPTS_PER_URL:
                    time.sleep(5 * attempt)
    return False


def extract(archive: Path, out_dir: Path) -> int:
    """Распаковывает архив и возвращает число картинок на выходе."""
    print("Распаковываю архив…")
    try:
        with tarfile.open(archive, "r:bz2") as tar:
            tar.extractall(out_dir)
    except (tarfile.TarError, OSError, EOFError) as error:
        print(f"[X] Распаковка не удалась: {error}")
        print("    Архив снова удаляю, чтобы следующий прогон скачал его заново.")
        archive.unlink(missing_ok=True)
        return -1

    images = [p for p in out_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS]
    print(f"  готово, картинок: {len(images)}")
    return len(images)


def main() -> int:
    parser = argparse.ArgumentParser(description="Загрузка датасета рукописных символов")
    parser.add_argument("--out", default="data/hasy", help="куда распаковать датасет")
    parser.add_argument("--archive", default=None,
                        help="путь к уже скачанному HASYv2.tar.bz2 (если качать не нужно)")
    parser.add_argument("--force", action="store_true",
                        help="перекачать архив и распаковать заново")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    archive = Path(args.archive) if args.archive else out_dir / ARCHIVE_NAME

    if args.force:
        archive.unlink(missing_ok=True)
        archive.with_name(archive.name + ".part").unlink(missing_ok=True)

    if archive.exists() and not is_usable_archive(archive):
        print(f"[!] Архив {archive} битый или недокачанный — удаляю и качаю заново.")
        archive.unlink(missing_ok=True)

    if archive.exists():
        print(f"Архив уже на месте: {archive}")
    elif not fetch_archive(archive):
        print("\nСкачать не удалось. Что можно сделать:")
        print(f"  1) скачать HASYv2.tar.bz2 вручную: {URLS[0]}")
        print(f"  2) положить файл сюда: {archive}")
        print("  3) повторить: python data/download_data.py --out data/hasy")
        return 1

    images = extract(archive, out_dir)
    if images < 0:
        return 1

    if images < MIN_IMAGES:
        print(f"[X] Картинок всего {images}, ожидалось около 168000.")
        print("    Распаковка неполная — чищу папку, чтобы кэш не закрепил мусор")
        print("    и следующий прогон скачал датасет заново.")
        for item in out_dir.iterdir():
            if item.name == ARCHIVE_NAME:
                continue
            if item.is_dir():
                shutil.rmtree(item, ignore_errors=True)
            else:
                item.unlink(missing_ok=True)
        return 1

    csv_files = sorted(out_dir.rglob("*.csv"))
    print(f"\nCSV-файлов: {len(csv_files)}")
    print(f"PNG-изображений: {images}")
    if csv_files:
        print(f"Основной CSV: {csv_files[0]}")
    if not csv_files:
        print("[!] Картинки есть, а CSV с метками нет — проверь распаковку архива.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
