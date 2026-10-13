"""Переводит обученную модель в веса для numpy-инференса на телефоне.

Зачем это нужно. В python-for-android нет рабочего рецепта tflite-runtime,
и buildozer падает прямо на разборе requirements — до сборки APK дело не
доходит. Но модель у нас крошечная (173 807 параметров), и считать её
можно обычным numpy: это простой набор свёрток и умножений матриц.

Что делает скрипт:
  * читает train/out/model.keras (его сохраняет train/train.py);
  * проходит слои по порядку и раскладывает веса в плоский словарь массивов;
  * пишет assets/model.npz — тот же порядок слоёв, что в ocr/net_numpy.py.

Порядок слоёв — часть контракта. Если менять ocr/model.py, надо править
и этот файл, и ocr/net_numpy.py, иначе числа разъедутся.

Запуск:
    python tools/export_npz.py --model train/out/model.keras --out assets/model.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def die(message: str, code: int = 1) -> int:
    print(f"[X] {message}")
    return code


def export(model_path: Path, out_path: Path, verify: bool = True) -> int:
    """Раскладывает веса Keras-модели в npz-архив."""
    if not model_path.exists():
        return die(f"Нет файла модели: {model_path}. "
                   f"Сначала запусти обучение (train/train.py).")

    # TensorFlow нужен только здесь, на сборке. В APK он не попадает.
    try:
        import tensorflow as tf
    except ImportError:
        return die("TensorFlow не установлен. Он нужен только для этого шага.")

    model = tf.keras.models.load_model(str(model_path), compile=False)
    print(f"[i] Модель загружена: {model_path}")
    print(f"[i] Параметров: {model.count_params()}")

    packed: dict[str, np.ndarray] = {}
    conv_index = 0
    dense_index = 0

    # Идём по слоям в том же порядке, в каком они собираются в ocr/model.py:
    # Conv2D -> BatchNormalization -> (ReLU без весов) — три раза по два раза,
    # затем GlobalAveragePooling (без весов), Dropout (без весов),
    # Dense(96) и Dense(23).
    for layer in model.layers:
        kind = layer.__class__.__name__
        weights = layer.get_weights()

        if kind == "Conv2D":
            # use_bias=False, поэтому весов ровно один тензор
            if len(weights) != 1:
                return die(f"У свёртки {layer.name} {len(weights)} тензоров, "
                           f"ожидался один — модель изменилась.")
            packed[f"conv{conv_index}_w"] = weights[0].astype(np.float32)
            print(f"[i] conv{conv_index}_w: {weights[0].shape}")
            conv_index += 1

        elif kind == "BatchNormalization":
            if len(weights) != 4:
                return die(f"У BatchNorm {layer.name} {len(weights)} тензоров, "
                           f"ожидалось четыре (гамма, бета, среднее, дисперсия).")
            gamma, beta, mean, variance = weights
            packed[f"bn{conv_index - 1}_gamma"] = gamma.astype(np.float32)
            packed[f"bn{conv_index - 1}_beta"] = beta.astype(np.float32)
            packed[f"bn{conv_index - 1}_mean"] = mean.astype(np.float32)
            packed[f"bn{conv_index - 1}_var"] = variance.astype(np.float32)
            print(f"[i] bn{conv_index - 1}: {gamma.shape}")

        elif kind == "Dense":
            bias = weights[1] if len(weights) > 1 else np.zeros_like(weights[0][0])
            packed[f"dense{dense_index}_w"] = weights[0].astype(np.float32)
            packed[f"dense{dense_index}_b"] = np.asarray(bias, dtype=np.float32)
            print(f"[i] dense{dense_index}: {weights[0].shape}")
            dense_index += 1

        # ReLU, MaxPooling2D, GlobalAveragePooling2D, Dropout и InputLayer
        # весов не имеют — они пропускаются молча.

    if conv_index != 6:
        return die(f"Нашлось свёрток: {conv_index}, ожидалось шесть. "
                   f"Проверь ocr/model.py — структура сети изменилась.")
    if dense_index != 2:
        return die(f"Нашлось полносвязных слоёв: {dense_index}, ожидалось два.")

    # Постоянные для BatchNorm: считаем их один раз здесь, чтобы на телефоне
    # инференс был проще (не тянуть дисперсию и epsilon в горячий цикл).
    eps = 1e-3
    for index in range(conv_index):
        gamma = packed[f"bn{index}_gamma"]
        beta = packed[f"bn{index}_beta"]
        mean = packed[f"bn{index}_mean"]
        variance = packed[f"bn{index}_var"]
        scale = gamma / np.sqrt(variance + eps)
        packed[f"bn{index}_scale"] = scale.astype(np.float32)
        packed[f"bn{index}_shift"] = (beta - mean * scale).astype(np.float32)

    # Индекс слоя, где заканчиваются свёртки, читается из числа каналов
    packed["img_size"] = np.asarray([model.input_shape[1]], dtype=np.int32)
    packed["num_classes"] = np.asarray([model.output_shape[-1]], dtype=np.int32)
    packed["stages"] = np.asarray(
        [packed[f"conv{i}_w"].shape[-1] for i in range(conv_index)], dtype=np.int32)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **packed)

    size = out_path.stat().st_size
    print(f"\n[i] Веса сохранены: {out_path} ({size / 1024:.1f} КБ)")

    if verify:
        code = verify_against_keras(model, packed, int(packed["img_size"][0]))
        if code != 0:
            return code

    return 0


def verify_against_keras(model, packed: dict, img_size: int) -> int:
    """Прогоняет numpy-инференс и сравнивает его с ответом Keras.

    Это главная проверка: если расхождений больше одной картинки из ста,
    значит веса разложены неверно, и на телефоне распознавание поедет.
    """
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    try:
        from ocr.net_numpy import NetNumpy
    except ImportError as error:
        print(f"[!] Проверка пропущена: {error}")
        return 0

    rng = np.random.default_rng(7)
    samples = rng.random((120, img_size, img_size), dtype=np.float32)
    # чуть сдвинем значения к «чернилам», как на настоящих картинках
    samples = np.clip(samples * 1.4 - 0.2, 0.0, 1.0)

    net = NetNumpy(packed)
    mine = np.stack([net.predict(image) for image in samples])
    theirs = model.predict(samples[..., np.newaxis], verbose=0)

    my_classes = mine.argmax(axis=1)
    tf_classes = theirs.argmax(axis=1)
    different = int((my_classes != tf_classes).sum())
    worst = float(np.abs(mine - theirs).max())

    print(f"[i] Сверка numpy и Keras на {len(samples)} картинках: "
          f"расхождений {different}, наибольшая разница {worst:.4f}")

    if different > len(samples) // 100:
        return die(f"numpy и Keras расходятся на {different} картинках. "
                   f"Веса разложены неверно — на телефоне будет путаница.")
    if worst > 0.35:
        return die(f"Числа сильно расходятся (разница {worst:.4f}) — "
                   f"проверь порядок слоёв и BatchNorm.")

    print("[OK] numpy-инференс совпадает с моделью.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Веса обученной модели → assets/model.npz для numpy-инференса")
    parser.add_argument("--model", default="train/out/model.keras")
    parser.add_argument("--out", default="assets/model.npz")
    parser.add_argument("--no-verify", action="store_true",
                        help="не сверять numpy с Keras")
    args = parser.parse_args()

    return export(Path(args.model), Path(args.out), verify=not args.no_verify)


if __name__ == "__main__":
    sys.exit(main())
