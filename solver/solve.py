"""Пошаговое решение школьных примеров на SymPy.

Что умеет: арифметика, дроби, степени, корни, логарифмы, линейные
и квадратные уравнения. Возвращает не только ответ, но и шаги
человеческим языком — приложение показывает разбор, а не просто число.

Всё считается на месте, офлайн: интернет не нужен.
"""

from __future__ import annotations

import sympy as sp
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

X = sp.Symbol("x")
Y = sp.Symbol("y")

TRANSFORMS = standard_transformations + (convert_xor, implicit_multiplication_application)

LOCAL_DICT = {
    "x": X,
    "y": Y,
    "sqrt": sp.sqrt,
    "log": sp.log,       # натуральный логарифм
    "ln": sp.log,
    "lg": lambda value: sp.log(value, 10),   # десятичный
    "pi": sp.pi,
    "e": sp.E,
    "sin": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,
}


def parse(text: str):
    """Переводит запись примера в выражение SymPy."""
    cleaned = (text or "").strip().replace(",", ".").replace("**", "^")
    if not cleaned:
        raise ValueError("Пусто: нечего решать")
    return parse_expr(cleaned, local_dict=LOCAL_DICT,
                      transformations=TRANSFORMS, evaluate=True)


def _pretty(value) -> str:
    return sp.sstr(value)


def _numeric(expression, digits: int = 8) -> str:
    try:
        value = sp.N(expression, digits)
    except Exception:
        return ""
    return _pretty(value) if value.is_real else ""


def _evaluate(text: str, digits: int = 8) -> dict:
    """Арифметика: вычислить и упростить."""
    expression = parse(text)
    steps = [("Записываю пример", text.strip())]

    expanded = sp.expand(expression)
    if _pretty(expanded) != _pretty(expression):
        steps.append(("Раскрываю скобки", _pretty(expanded)))

    simplified = sp.simplify(expanded)
    if _pretty(simplified) != _pretty(expanded):
        steps.append(("Привожу к простому виду", _pretty(simplified)))

    exact = sp.nsimplify(simplified, rational=True)
    if _pretty(exact) != _pretty(simplified):
        steps.append(("Считаю дроби точно, без округления", _pretty(exact)))
    exact = sp.simplify(exact)

    answer = _pretty(exact)
    numeric = _numeric(exact, digits)
    if numeric and numeric != answer:
        steps.append(("Проверяю приближённо", numeric))
        answer = f"{answer} ≈ {numeric}" if not exact.is_Number else answer

    steps.append(("Ответ", _pretty(exact)))
    return {"kind": "expression", "steps": steps, "answer": _pretty(exact),
            "error": ""}


def _pick_unknown(expression):
    """Какая буква неизвестна: x, потом y, иначе первая найденная."""
    for symbol in (X, Y):
        if symbol in expression.free_symbols:
            return symbol
    symbols = sorted(expression.free_symbols, key=lambda item: item.name)
    return symbols[0] if symbols else X


def _linear_steps(poly, unknown) -> list:
    """Уравнение a*x + b = 0: перенос свободного члена и деление на a."""
    a = poly.coeff_monomial(unknown)
    b = poly.coeff_monomial(1)
    steps = [
        (f"Переношу свободный член: {_pretty(a)}·{unknown} = {_pretty(-b)}", "")
        ,
        (f"Делю обе части на {_pretty(a)}", f"{unknown} = {_pretty(-b / a)}"),
    ]
    return steps


def _quadratic_steps(poly, unknown) -> list:
    """Уравнение a*x^2 + b*x + c = 0: через дискриминант, как в школе."""
    a = poly.coeff_monomial(unknown ** 2)
    b = poly.coeff_monomial(unknown)
    c = poly.coeff_monomial(1)
    discriminant = sp.simplify(b ** 2 - 4 * a * c)

    steps = [
        ("Нахожу коэффициенты", f"a = {_pretty(a)}, b = {_pretty(b)}, c = {_pretty(c)}"),
        ("Считаю дискриминант: D = b² − 4ac", f"D = {_pretty(discriminant)}"),
    ]

    value = _numeric(discriminant)
    if value:
        steps.append(("D числом", value))

    steps.append(("Подставляю в формулу корней x = (−b ± √D) / (2a)", ""))
    roots = sp.solve(poly.as_expr(), unknown)
    for index, root in enumerate(roots, start=1):
        steps.append((f"Корень {index}", f"{unknown} = {_pretty(root)}"))
    return steps


def _solve_equation(left_text: str, right_text: str, digits: int = 8) -> dict:
    """Уравнение: переносим всё влево и решаем."""
    left = parse(left_text)
    right = parse(right_text)
    step = sp.expand(left - right)
    unknown = _pick_unknown(step)

    steps = [
        ("Переношу всё в левую часть", f"{_pretty(left)} − ({_pretty(right)}) = 0"),
        ("Раскрываю скобки и привожу подобные", f"{_pretty(step)} = 0"),
    ]

    if len(step.free_symbols) > 1:
        return {"kind": "equation", "steps": steps, "answer": "",
                "error": "Неизвестных больше одной: пока решаю уравнения с одной буквой."}

    degree = None
    try:
        degree = sp.degree(sp.Poly(step, unknown))
    except Exception:
        degree = None

    if degree == 1:
        steps += _linear_steps(sp.Poly(step, unknown), unknown)
    elif degree == 2:
        steps += _quadratic_steps(sp.Poly(step, unknown), unknown)
    else:
        roots = sp.solve(step, unknown)
        if not roots:
            return {"kind": "equation", "steps": steps, "answer": "",
                    "error": "Не нашёл решения этого уравнения."}
        steps.append(("Решаю уравнение", _pretty(sp.Eq(step, 0))))
        for index, root in enumerate(roots, start=1):
            steps.append((f"Корень {index}", f"{unknown} = {_pretty(root)}"))

    roots = sp.solve(step, unknown)
    if not roots:
        return {"kind": "equation", "steps": steps, "answer": "",
                "error": "Корней нет."}

    # Проверка подстановкой: подставляем каждый корень в исходную разность
    for root in roots:
        check = sp.simplify(step.subs(unknown, root))
        steps.append((f"Проверка: подставляю {unknown} = {_pretty(root)}",
                      f"получаю {_pretty(check)} — верно" if check == 0
                      else f"получаю {_pretty(check)} — не ноль"))

    answer = "; ".join(f"{unknown} = {_pretty(root)}" for root in roots)
    steps.append(("Ответ", answer))
    return {"kind": "equation", "steps": steps, "answer": answer, "error": ""}


def solve(text: str, digits: int = 8) -> dict:
    """Точка входа: принимает строку примера, возвращает ответ и шаги.

    Результат: {"kind": ..., "steps": [(заголовок, пояснение), ...],
                "answer": ..., "error": ...}
    """
    cleaned = (text or "").strip()
    if not cleaned:
        return {"kind": "error", "steps": [], "answer": "",
                "error": "Поле пустое: напиши пример или сфотографируй его."}

    try:
        if "=" in cleaned:
            left, right = cleaned.split("=", 1)
            return _solve_equation(left, right, digits)
        return _evaluate(cleaned, digits)
    except Exception as error:                      # noqa: BLE001 — показываем причину
        return {"kind": "error", "steps": [], "answer": "",
                "error": f"Не получилось разобрать запись: {error}"}
