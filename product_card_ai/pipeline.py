"""Конвейер: входные данные -> LLM -> проверенная карточка товара.

Разбор ответа модели и контроль качества вынесены сюда, чтобы они работали
одинаково для любого провайдера. Ключевая проверка — поиск чисел, которых
нет во входных данных: это основной способ поймать выдуманные характеристики.
"""

from __future__ import annotations

import json
import re
from typing import Any

from .models import (
    NO_DATA,
    InputReport,
    ProductCard,
    ProductInput,
    QualityReport,
    SpecItem,
)
from .prompts import (
    DESCRIPTION_MAX,
    DESCRIPTION_MIN,
    QUERIES_MAX,
    QUERIES_MIN,
    TITLE_MAX,
)
from .providers import Provider, ProviderError

# Числа вида 1,6 / 0-1,6 / 100 / G1/2. Ищем именно числовые токены:
# текстовые формулировки модель имеет право перефразировать, числа — нет.
_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)?")


class ModelOutputError(RuntimeError):
    """Модель вернула ответ, который не удалось разобрать."""


def _strip_code_fence(text: str) -> str:
    """Убрать ```json ... ``` вокруг ответа, если модель их всё-таки добавила."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def parse_model_output(raw: str) -> dict[str, Any]:
    """Достать JSON-объект из ответа модели."""
    if not raw or not raw.strip():
        raise ModelOutputError("Модель вернула пустой ответ.")

    cleaned = _strip_code_fence(raw)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        # Запасной путь: вырезать первый сбалансированный объект из текста.
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end <= start:
            raise ModelOutputError(
                "Ответ модели не содержит JSON-объект. Проверьте промпт или модель."
            ) from None
        try:
            data = json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError as exc:
            raise ModelOutputError(f"Не удалось разобрать JSON от модели: {exc.msg}") from exc

    if not isinstance(data, dict):
        raise ModelOutputError("Модель вернула JSON, но не объект.")
    return data


def _as_text(value: Any, fallback: str = NO_DATA) -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()
    if isinstance(value, (int, float)):
        return str(value)
    return fallback


def _as_specs(value: Any) -> list[SpecItem]:
    items: list[SpecItem] = []
    if isinstance(value, dict):
        value = [{"label": k, "value": v} for k, v in value.items()]
    if not isinstance(value, list):
        return items
    for entry in value:
        if isinstance(entry, dict):
            label = _as_text(entry.get("label") or entry.get("name"), "")
            raw_value = _as_text(entry.get("value"))
            if label:
                items.append(SpecItem(label=label, value=raw_value))
    return items


def _as_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if not isinstance(value, list):
        return []
    return [item.strip() for item in value if isinstance(item, str) and item.strip()]


def _numbers(text: str) -> set[str]:
    """Нормализованные числовые токены строки."""
    return {match.group(0).replace(",", ".").rstrip("0").rstrip(".") or "0"
            for match in _NUMBER_RE.finditer(text)}


def check_quality(card: ProductCard, product: ProductInput) -> QualityReport:
    """Проверить карточку на соответствие ТЗ и на выдуманные значения."""
    report = QualityReport(
        title_length=len(card.seo_title),
        description_length=len(card.seo_description),
    )

    if report.title_length > TITLE_MAX:
        report.issues.append(
            f"SEO-заголовок длиннее {TITLE_MAX} символов ({report.title_length})."
        )
    if report.title_length == 0:
        report.issues.append("SEO-заголовок пустой.")

    if not DESCRIPTION_MIN <= report.description_length <= DESCRIPTION_MAX:
        report.issues.append(
            f"SEO-описание вне диапазона {DESCRIPTION_MIN}–{DESCRIPTION_MAX} символов "
            f"({report.description_length})."
        )

    queries_count = len(card.search_queries)
    if not QUERIES_MIN <= queries_count <= QUERIES_MAX:
        report.issues.append(
            f"Поисковых запросов должно быть {QUERIES_MIN}–{QUERIES_MAX}, получено {queries_count}."
        )

    if not card.key_specs:
        report.issues.append("Список ключевых характеристик пуст.")

    # Главная проверка: любое число в генерации должно встречаться во входных данных.
    allowed = _numbers(product.source_text())
    generated_text = " ".join(
        [card.seo_title, card.seo_description]
        + [f"{item.label} {item.value}" for item in card.key_specs]
        + card.search_queries
        + [card.commercial_audience, card.commercial_tasks]
    )
    for number in sorted(_numbers(generated_text)):
        if number not in allowed:
            report.unsupported_values.append(number)

    if report.unsupported_values:
        report.issues.append(
            "В тексте есть числа, которых нет во входных данных: "
            + ", ".join(report.unsupported_values)
            + ". Проверьте вручную — возможна выдуманная характеристика."
        )

    return report


def build_card(
    data: dict[str, Any],
    product: ProductInput,
    report: InputReport,
    provider_name: str,
    model_name: str,
) -> ProductCard:
    """Собрать `ProductCard` из разобранного ответа модели."""
    commercial = data.get("commercial_block")
    if not isinstance(commercial, dict):
        commercial = {}

    missing = _as_list(data.get("missing_data"))
    # Пробелы, найденные валидатором, добавляем всегда: модель могла их пропустить.
    for item in report.missing_fields:
        if item not in missing:
            missing.append(item)

    card = ProductCard(
        product_name=product.name,
        seo_title=_as_text(data.get("seo_title"), ""),
        seo_description=_as_text(data.get("seo_description"), ""),
        key_specs=_as_specs(data.get("key_specs")),
        search_queries=_as_list(data.get("search_queries")),
        commercial_audience=_as_text(commercial.get("audience")),
        commercial_tasks=_as_text(commercial.get("tasks")),
        missing_data=missing,
        provider=provider_name,
        model=model_name,
    )
    card.quality = check_quality(card, product)
    return card


def generate_card(
    product: ProductInput,
    report: InputReport,
    provider: Provider,
) -> ProductCard:
    """Полный цикл для одного товара."""
    try:
        raw = provider.generate(product, report)
    except ProviderError:
        raise
    data = parse_model_output(raw)
    return build_card(data, product, report, provider.name, provider.model)
