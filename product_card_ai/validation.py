"""Разбор и проверка входных данных.

Входной JSON приходит от контент-менеджера и может быть каким угодно:
пустым, с опечатками в типах, без половины полей. Задача модуля —
привести его к `ProductInput` и честно сказать, чего не хватает.
"""

from __future__ import annotations

import json
from typing import Any

from .models import NO_DATA, InputReport, ProductInput

# Поля, без которых карточку делать бессмысленно.
REQUIRED_FIELDS = ("name",)

# Поля, без которых карточка получится, но будет неполной.
RECOMMENDED_FIELDS = ("manufacturer", "specs", "description")

# Характеристики, которые для КИП почти всегда нужны в карточке B2B-магазина.
EXPECTED_SPEC_HINTS = (
    "диапазон измерений",
    "класс точности",
    "присоединение",
    "питание",
    "тип корпуса",
    "интерфейсы",
    "степень защиты",
)


class InvalidInputError(ValueError):
    """Входные данные невозможно интерпретировать как товар."""


def load_json(text: str) -> Any:
    """Разобрать JSON с понятным сообщением об ошибке."""
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise InvalidInputError(
            f"Некорректный JSON: {exc.msg} (строка {exc.lineno}, символ {exc.colno})"
        ) from exc


def _clean_str(value: Any) -> str | None:
    """Привести значение к непустой строке или вернуть None."""
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return str(value)
    if not isinstance(value, str):
        return None
    cleaned = " ".join(value.split())
    if not cleaned:
        return None
    # Контент-менеджер мог уже проставить пометку вручную — не считаем её данными.
    if cleaned.lower() in {NO_DATA, "нет данных", "n/a", "na", "-", "—"}:
        return None
    return cleaned


def _clean_specs(value: Any) -> tuple[dict[str, str], list[str]]:
    """Нормализовать словарь характеристик.

    Возвращает очищенные характеристики и список замечаний.
    """
    notes: list[str] = []
    specs: dict[str, str] = {}

    if value is None:
        return specs, notes

    if isinstance(value, list):
        # Допускаем формат [{"label": ..., "value": ...}, ...].
        converted: dict[str, Any] = {}
        for entry in value:
            if isinstance(entry, dict):
                label = _clean_str(entry.get("label") or entry.get("name"))
                raw_value = entry.get("value")
                if label:
                    converted[label] = raw_value
            else:
                notes.append("Элемент списка specs пропущен: ожидался объект с label/value.")
        value = converted

    if not isinstance(value, dict):
        notes.append("Поле specs проигнорировано: ожидался объект \"характеристика\": \"значение\".")
        return specs, notes

    for raw_key, raw_value in value.items():
        key = _clean_str(raw_key)
        if not key:
            continue
        cleaned_value = _clean_str(raw_value)
        if cleaned_value is None:
            notes.append(f"Характеристика «{key}» указана без значения.")
            continue
        specs[key] = cleaned_value

    return specs, notes


def parse_product(data: Any) -> tuple[ProductInput, InputReport]:
    """Превратить произвольный объект в `ProductInput` и отчёт о пробелах."""
    if not isinstance(data, dict):
        raise InvalidInputError(
            "Ожидался JSON-объект с полями name, manufacturer, specs, description."
        )

    name = _clean_str(data.get("name"))
    if not name:
        raise InvalidInputError("Поле name обязательно: без названия товара карточку не собрать.")

    manufacturer = _clean_str(data.get("manufacturer"))
    description = _clean_str(data.get("description"))
    specs, spec_notes = _clean_specs(data.get("specs"))

    report = InputReport()
    report.needs_clarification.extend(spec_notes)

    if manufacturer is None:
        report.missing_fields.append("manufacturer (производитель)")
    if not specs:
        report.missing_fields.append("specs (технические характеристики)")
    if description is None:
        report.missing_fields.append("description (описание или фрагмент техпаспорта)")

    if specs:
        known = " ".join(specs.keys()).lower()
        absent = [hint for hint in EXPECTED_SPEC_HINTS if hint not in known]
        if absent:
            report.needs_clarification.append(
                "Нет типовых характеристик КИП: " + ", ".join(absent) + "."
            )

    if description and len(description) < 80:
        report.needs_clarification.append(
            "Описание очень короткое — SEO-текст будет опираться почти только на характеристики."
        )

    product = ProductInput(
        name=name,
        manufacturer=manufacturer,
        specs=specs,
        description=description,
    )
    return product, report


def parse_products(data: Any) -> list[tuple[ProductInput, InputReport]]:
    """Разобрать один товар или список товаров.

    Поддерживаются три формы: объект товара, массив товаров,
    объект вида {"products": [...]}.
    """
    if isinstance(data, dict) and isinstance(data.get("products"), list):
        data = data["products"]

    if isinstance(data, list):
        if not data:
            raise InvalidInputError("Передан пустой список товаров.")
        return [parse_product(entry) for entry in data]

    return [parse_product(data)]
