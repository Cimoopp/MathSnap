"""Своя свёрточная сеть MathSnap для распознавания рукописных символов.

Модель намеренно маленькая: она должна работать на телефоне в офлайне
и весить меньше мегабайта после конвертации в TensorFlow Lite.
"""

import tensorflow as tf

from ocr.labels import IMG_SIZE, NUM_CLASSES


def conv_block(x, filters: int, pool: bool = True):
    """Два свёрточных слоя 3x3 + BatchNorm + ReLU, затем уменьшение вдвое."""
    for _ in range(2):
        x = tf.keras.layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)
    if pool:
        x = tf.keras.layers.MaxPooling2D()(x)
    return x


def build_model(num_classes: int = NUM_CLASSES, img_size: int = IMG_SIZE,
                learning_rate: float = 1e-3) -> tf.keras.Model:
    """Собирает и компилирует модель распознавания одного символа.

    Вход: изображение img_size x img_size, оттенки серого,
    символ белым по чёрному фону, значения в диапазоне 0..1.
    Выход: softmax на num_classes классов из ocr/labels.py.
    """
    inputs = tf.keras.Input(shape=(img_size, img_size, 1), name="glyph")

    x = inputs
    for filters in (32, 64, 128):
        x = conv_block(x, filters)          # 64 -> 32 -> 16 -> 8

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax",
                                    name="classes")(x)

    model = tf.keras.Model(inputs, outputs, name="mathsnap_ocr")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def augment_layers():
    """Слои аугментации для обучения.

    Применяются только в пайплайне обучения (training=True), в саму модель
    не входят, поэтому в .tflite они не попадут и инференс останется быстрым.
    """
    return [
        tf.keras.layers.RandomRotation(0.07, fill_mode="constant"),
        tf.keras.layers.RandomTranslation(0.06, 0.06, fill_mode="constant"),
        tf.keras.layers.RandomZoom(0.08, fill_mode="constant"),
    ]


if __name__ == "__main__":
    # Быстрая проверка: собрать модель, посчитать параметры и размер весов.
    m = build_model()
    m.summary()
    params = m.count_params()
    print(f"\nПараметров: {params}")
    print(f"Веса в float32: {params * 4 / 1024 / 1024:.2f} МБ")
