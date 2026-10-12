[app]

title = MathSnap
package.name = mathsnap
package.domain = org.cimoopp

source.dir = .
source.include_exts = py,png,jpg,jpeg,webp,tflite,json,atlas,ttf,so

version = 0.1.0

# tflite-runtime есть в python-for-android готовым рецептом, поэтому
# инференс идёт своим движком TensorFlow Lite, без тяжёлого TensorFlow.
# sympy и kivy тянутся как обычно.
requirements = python3,kivy,numpy,pillow,sympy,tflite-runtime

orientation = portrait
fullscreen = 0

# Камера нужна для съёмки примера. Доступ к интернету приложению не нужен:
# распознавание и решение работают офлайн.
android.permissions = CAMERA

android.api = 34
android.minapi = 24
android.ndk_api = 24

# Только arm64: так первая сборка вдвое короче, а 32-битные телефоны
# с Android 7+ практически не встречаются. Понадобится armeabi-v7a —
# дописать через запятую, сборка станет длиннее.
android.archs = arm64-v8a

android.accept_sdk_license = True
android.allow_backup = True
android.enable_androidx = True

p4a.bootstrap = sdl2

[buildozer]

log_level = 2
warn_on_root = 1
