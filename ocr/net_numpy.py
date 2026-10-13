"""Инференс своей CNN на чистом numpy — без TensorFlow и tflite-runtime.

Почему так. В python-for-android нет рабочего рецепта tflite-runtime:
buildozer падает на разборе requirements ещё до сборки NDK. Тянуть в APK
целый TensorFlow ради сети на 173 807 параметров бессмысленно — это
десятки мегабайт. Свёртки 3x3 и два полносвязных слоя считаются numpy
за миллисекунды даже на слабом телефоне.

Порядок слоёв здесь ровно такой, как в ocr/model.py:
    Conv 24 -> BN -> ReLU -> Conv 24 -> BN -> ReLU -> MaxPool
    Conv 48 -> BN -> ReLU -> Conv 48 -> BN -> ReLU -> MaxPool
    Conv 96 -> BN -> ReLU -> Conv 96 -> BN -> ReLU -> MaxPool
    GlobalAveragePooling -> Dropout (в инференсе выключен)
    Dense 96 -> ReLU -> Dropout (выключен) -> Dense 23 -> softmax

Сверка с Keras выполняется при упаковке весов в tools/export_npz.py.
"""

from __future__ import annotations

import numpy as np

from ocr.labels import CLASSES, IMG_SIZE

# Тот же epsilon, что у BatchNorm в Keras по умолчанию
BN_EPS = 1e-3


def conv2d_same(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Свёртка 3x3 с дополнением нулями (padding="same").

    image:  (высота, ширина, каналы)
    kernel: (3, 3, каналы входа, каналы выхода)
    """
    height, width, channels = image.shape
    out_channels = kernel.shape[3]
    padded = np.pad(image, ((1, 1), (1, 1), (0, 0)))

    result = np.zeros((height, width, out_channels), dtype=np.float32)
    # Собираем свёртку как сумму девяти сдвинутых умножений: так numpy
    # работает векторно по всем каналам сразу, без циклов по пикселям.
    for dy in range(3):
        for dx in range(3):
            patch = padded[dy:dy + height, dx:dx + width, :]        # (h, w, cin)
            weights = kernel[dy, dx]                                # (cin, cout)
            result += patch @ weights
    return result


def max_pool2x2(image: np.ndarray) -> np.ndarray:
    """Уменьшение вдвое по максимуму (2x2, шаг 2), как MaxPooling2D()."""
    height, width, channels = image.shape
    # Обрезаем нечётный край, чтобы форма делилась на два
    if height % 2 or width % 2:
        image = image[:height - height % 2, :width - width % 2]
        height, width, _ = image.shape
    blocks = image.reshape(height // 2, 2, width // 2, 2, channels)
    return blocks.max(axis=(1, 3))


def relu(values: np.ndarray) -> np.ndarray:
    return np.maximum(values, 0.0, dtype=np.float32)


def softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - values.max()
    exponent = np.exp(shifted)
    return exponent / max(float(exponent.sum()), 1e-12)


class NetNumpy:
    """Считает ответ сети по весам из assets/model.npz.

    Веса приходят словарём: conv0_w..conv5_w, bn0_scale/bn0_shift..,
    dense0_w/dense0_b, dense1_w/dense1_b. Постоянные BatchNorm
    (scale и shift) уже посчитаны при упаковке.
    """

    def __init__(self, weights: dict, img_size: int | None = None):
        self.w = weights
        if img_size is None:
            raw = weights.get("img_size")
            img_size = int(np.asarray(raw)[0]) if raw is not None else IMG_SIZE
        self.img_size = img_size
        self.stages = [int(v) for v in np.asarray(
            weights.get("stages", [24, 48, 96]))]

    # ---------- слои ----------

    def _bn(self, image: np.ndarray, index: int) -> np.ndarray:
        """BatchNorm в режиме инференса: готовые scale и shift."""
        scale = self.w[f"bn{index}_scale"]
        shift = self.w[f"bn{index}_shift"]
        return image * scale + shift

    def _conv_bn_relu(self, image: np.ndarray, index: int) -> np.ndarray:
        image = conv2d_same(image, self.w[f"conv{index}_w"])
        image = self._bn(image, index)
        return relu(image)

    def _features(self, image: np.ndarray) -> np.ndarray:
        """Свёрточная часть: три блока по два Conv-BN-ReLU плюс MaxPool."""
        layer = 0
        for _ in self.stages:
            image = self._conv_bn_relu(image, layer)
            layer += 1
            image = self._conv_bn_relu(image, layer)
            layer += 1
            image = max_pool2x2(image)
        return image

    # ---------- ответ ----------

    def probabilities(self, glyph: np.ndarray) -> np.ndarray:
        """Вероятности всех 23 классов для одной картинки."""
        image = np.asarray(glyph, dtype=np.float32)
        if image.ndim == 2:
            image = image[..., np.newaxis]

        features = self._features(image)

        # GlobalAveragePooling2D
        pooled = features.mean(axis=(0, 1))

        # Dense 96 + ReLU (Dropout в инференсе выключен)
        hidden = relu(pooled @ self.w["dense0_w"] + self.w["dense0_b"])

        # Dense 23 + softmax
        logits = hidden @ self.w["dense1_w"] + self.w["dense1_b"]
        return softmax(logits)

    def predict(self, glyph: np.ndarray):
        """Возвращает (символ, уверенность) — как прежний читатель tflite."""
        scores = self.probabilities(glyph)
        best = int(np.argmax(scores))
        return CLASSES[best], float(scores[best])


def load_weights(path) -> dict:
    """Читает assets/model.npz в словарь массивов float32."""
    packed = np.load(path, allow_pickle=False)
    weights = {name: packed[name].astype(np.float32) for name in packed.files
               if name not in ("img_size", "num_classes", "stages")}

    # Постоянные, которые не нужно переводить в float32
    for name in ("img_size", "num_classes", "stages"):
        if name in packed.files:
            weights[name] = packed[name]
    return weights
