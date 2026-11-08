"""Доменные модели прототипа.

Здесь описаны только структуры данных: вход (данные о приборе),
выход (черновик карточки товара) и отчёты о качестве. Никакой логики
обращения к LLM тут нет.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Единая пометка для отсутствующих данных. Используется и в промпте,
# и в постобработке, и в отрисовке результата.
NO_DATA = "данных нет"


@dataclass(frozen=True)
class SpecItem:
    """Одна характеристика прибора."""

    label: str
    value: str

    @property
    def is_known(self) -> bool:
        return self.value.strip().lower() != NO_DATA

    def to_dict(self) -> dict[str, str]:
        return {"label": self.label, "value": self.value}


@dataclass(frozen=True)
class ProductInput:
    """Нормализованные входные данные по одному товару."""

    name: str
    manufacturer: str | None = None
    specs: dict[str, str] = field(default_factory=dict)
    description: str | None = None

    @property
    def manufacturer_or_marker(self) -> str:
        return self.manufacturer or NO_DATA

    @property
    def description_or_marker(self) -> str:
        return self.description or NO_DATA

    def spec_items(self) -> list[SpecItem]:
        return [SpecItem(label=k, value=v) for k, v in self.specs.items()]

    def source_text(self) -> str:
        """Весь исходный текст одной строкой.

        Нужен для проверки того, что модель не выдумала числовые значения:
        любое число из результата должно встречаться здесь.
        """
        parts: list[str] = [self.name]
        if self.manufacturer:
            parts.append(self.manufacturer)
        for key, value in self.specs.items():
            parts.append(f"{key}: {value}")
        if self.description:
            parts.append(self.description)
        return "\n".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "manufacturer": self.manufacturer,
            "specs": dict(self.specs),
            "description": self.description,
        }


@dataclass
class InputReport:
    """Что известно о полноте входных данных."""

    missing_fields: list[str] = field(default_factory=list)
    needs_clarification: list[str] = field(default_factory=list)

    @property
    def has_gaps(self) -> bool:
        return bool(self.missing_fields or self.needs_clarification)

    def to_dict(self) -> dict[str, Any]:
        return {
            "missing_fields": list(self.missing_fields),
            "needs_clarification": list(self.needs_clarification),
        }


@dataclass
class QualityReport:
    """Результат автоматической проверки сгенерированной карточки."""

    title_length: int = 0
    description_length: int = 0
    issues: list[str] = field(default_factory=list)
    unsupported_values: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.issues and not self.unsupported_values

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "title_length": self.title_length,
            "description_length": self.description_length,
            "issues": list(self.issues),
            "unsupported_values": list(self.unsupported_values),
        }


@dataclass
class ProductCard:
    """Черновик карточки товара — то, что получает контент-менеджер."""

    product_name: str
    seo_title: str
    seo_description: str
    key_specs: list[SpecItem] = field(default_factory=list)
    search_queries: list[str] = field(default_factory=list)
    commercial_audience: str = NO_DATA
    commercial_tasks: str = NO_DATA
    missing_data: list[str] = field(default_factory=list)
    provider: str = ""
    model: str = ""
    quality: QualityReport = field(default_factory=QualityReport)

    def to_dict(self) -> dict[str, Any]:
        return {
            "product_name": self.product_name,
            "seo_title": self.seo_title,
            "seo_description": self.seo_description,
            "key_specs": [item.to_dict() for item in self.key_specs],
            "search_queries": list(self.search_queries),
            "commercial_block": {
                "audience": self.commercial_audience,
                "tasks": self.commercial_tasks,
            },
            "missing_data": list(self.missing_data),
            "meta": {"provider": self.provider, "model": self.model},
            "quality": self.quality.to_dict(),
        }
