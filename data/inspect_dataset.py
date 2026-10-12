"""Быстрая диагностика датасета перед обучением.

Печатает структуру распакованного архива, все таблицы CSV с колонками и
первыми строками, число картинок. Запускается отдельным шагом workflow до
обучения: если что-то сломается, всё нужное уже лежит в логе прогона.
"""

import argparse
import csv
from pathlib import Path

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


def read_header(path: Path):
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        return [h.strip() for h in next(csv.reader(f), [])]


def head_rows(path: Path, count: int = 2):
    rows = []
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        next(reader, None)
        for _ in range(count):
            row = next(reader, None)
            if row is None:
                break
            rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Диагностика датасета символов")
    parser.add_argument("--data", default="data", help="папка с данными")
    args = parser.parse_args()

    hasy_dir = Path(args.data) / "hasy"
    if not hasy_dir.exists():
        print(f"[X] Папки {hasy_dir} нет — сначала запусти data/download_data.py")
        return 1

    print(f"Датасет: {hasy_dir}")

    print("\nВерхний уровень:")
    for path in sorted(hasy_dir.iterdir()):
        if path.is_dir():
            files = sum(1 for item in path.rglob("*") if item.is_file())
            print(f"  {path.name}/ — файлов: {files}")
        else:
            size_mb = path.stat().st_size / 1e6
            print(f"  {path.name} — {size_mb:.1f} МБ")

    images = [p for p in hasy_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS]
    print(f"\nИзображений: {len(images)}")
    for path in images[:3]:
        print(f"  пример: {path.relative_to(hasy_dir)}")
    if 0 < len(images) < 20000:
        print("[!] Изображений подозрительно мало — архив распаковался не полностью.")

    csv_files = sorted(hasy_dir.rglob("*.csv"))
    print(f"\nТаблиц CSV: {len(csv_files)}")
    for path in csv_files:
        print(f"\n  {path.relative_to(hasy_dir)}")
        print(f"    колонки: {read_header(path)}")
        for row in head_rows(path):
            print(f"    строка:  {row}")

    print("\nГотово. Если в таблице нет колонки с LaTeX-метками,")
    print("train.py возьмёт колонку symbol_id и расшифрует её через symbols.csv.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
