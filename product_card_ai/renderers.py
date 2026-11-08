"""Отрисовка карточки в форматы, удобные для копирования.

Контент-менеджер работает с Markdown, разработчик — с JSON.
Оба формата строятся из одного и того же `ProductCard`.
"""

from __future__ import annotations

import json

from .models import NO_DATA, InputReport, ProductCard


def to_json(card: ProductCard, *, indent: int = 2) -> str:
    return json.dumps(card.to_dict(), ensure_ascii=False, indent=indent)


def to_markdown(card: ProductCard, report: InputReport | None = None) -> str:
    lines: list[str] = []
    lines.append(f"# {card.product_name}")
    lines.append("")
    lines.append(f"**SEO-заголовок** ({len(card.seo_title)}/70 символов)")
    lines.append("")
    lines.append(card.seo_title or NO_DATA)
    lines.append("")
    lines.append(f"**SEO-описание** ({len(card.seo_description)} символов)")
    lines.append("")
    lines.append(card.seo_description or NO_DATA)
    lines.append("")

    lines.append("## Ключевые характеристики")
    lines.append("")
    if card.key_specs:
        lines.append("| Характеристика | Значение |")
        lines.append("| --- | --- |")
        for item in card.key_specs:
            lines.append(f"| {item.label} | {item.value} |")
    else:
        lines.append(NO_DATA)
    lines.append("")

    lines.append("## Поисковые запросы")
    lines.append("")
    if card.search_queries:
        lines.extend(f"- {query}" for query in card.search_queries)
    else:
        lines.append(NO_DATA)
    lines.append("")

    lines.append("## Для менеджера")
    lines.append("")
    lines.append(f"**Кому подходит.** {card.commercial_audience}")
    lines.append("")
    lines.append(f"**Какие задачи решает.** {card.commercial_tasks}")
    lines.append("")

    if card.missing_data:
        lines.append("## Чего не хватает во входных данных")
        lines.append("")
        lines.extend(f"- {item}" for item in card.missing_data)
        lines.append("")

    if report and report.needs_clarification:
        lines.append("## Требует уточнения")
        lines.append("")
        lines.extend(f"- {item}" for item in report.needs_clarification)
        lines.append("")

    lines.append("## Автопроверка")
    lines.append("")
    if card.quality.ok:
        lines.append("Замечаний нет: длины в норме, чисел вне входных данных не найдено.")
    else:
        lines.extend(f"- {issue}" for issue in card.quality.issues)
    lines.append("")
    lines.append(f"_Провайдер: {card.provider} / {card.model}_")

    return "\n".join(lines)


def to_text(card: ProductCard, report: InputReport | None = None) -> str:
    """Плоский текст: когда карточку просто вставляют в CMS."""
    blocks = [
        f"SEO-ЗАГОЛОВОК: {card.seo_title}",
        "",
        "SEO-ОПИСАНИЕ:",
        card.seo_description,
        "",
        "ХАРАКТЕРИСТИКИ:",
    ]
    blocks.extend(f"- {item.label}: {item.value}" for item in card.key_specs)
    blocks.append("")
    blocks.append("ПОИСКОВЫЕ ЗАПРОСЫ:")
    blocks.extend(f"- {query}" for query in card.search_queries)
    blocks.append("")
    blocks.append("ДЛЯ МЕНЕДЖЕРА:")
    blocks.append(f"Кому подходит: {card.commercial_audience}")
    blocks.append(f"Задачи: {card.commercial_tasks}")
    if card.missing_data:
        blocks.append("")
        blocks.append("НЕ ХВАТАЕТ ДАННЫХ:")
        blocks.extend(f"- {item}" for item in card.missing_data)
    if report and report.needs_clarification:
        blocks.append("")
        blocks.append("ТРЕБУЕТ УТОЧНЕНИЯ:")
        blocks.extend(f"- {item}" for item in report.needs_clarification)
    blocks.append("")
    blocks.append("АВТОПРОВЕРКА:")
    if card.quality.ok:
        blocks.append("Замечаний нет: длины в норме, чисел вне входных данных не найдено.")
    else:
        blocks.extend(f"- {issue}" for issue in card.quality.issues)
    blocks.append("")
    blocks.append(f"Провайдер: {card.provider} / {card.model}")
    return "\n".join(blocks)


def render(card: ProductCard, fmt: str, report: InputReport | None = None) -> str:
    if fmt == "json":
        return to_json(card)
    if fmt == "text":
        return to_text(card, report)
    return to_markdown(card, report)
