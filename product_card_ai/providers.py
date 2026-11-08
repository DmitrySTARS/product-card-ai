"""Провайдеры LLM и оффлайн-заглушка.

Каждый провайдер получает системный и пользовательский промпт и возвращает
сырую строку ответа. Разбор JSON и проверка качества живут в pipeline.py,
чтобы логика не дублировалась между провайдерами.

HTTP выполняется через urllib из стандартной библиотеки: прототипу не нужны
внешние SDK, а так его проще запустить по инструкции.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Protocol

from .config import Settings
from .models import NO_DATA, InputReport, ProductInput
from .prompts import (
    DESCRIPTION_MIN,
    QUERIES_MAX,
    SYSTEM_PROMPT,
    TITLE_MAX,
    build_user_prompt,
)
from .validation import EXPECTED_SPEC_HINTS


class ProviderError(RuntimeError):
    """Обращение к LLM завершилось ошибкой."""


class Provider(Protocol):
    name: str
    model: str

    def generate(self, product: ProductInput, report: InputReport) -> str:
        """Вернуть сырой JSON-ответ модели строкой."""


def _post_json(url: str, payload: dict, headers: dict[str, str], timeout: int) -> dict:
    """POST с JSON-телом и понятными сообщениями об ошибках."""
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise ProviderError(f"LLM вернула HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise ProviderError(f"Не удалось связаться с LLM: {exc.reason}") from exc
    except TimeoutError as exc:
        raise ProviderError("Таймаут запроса к LLM. Увеличьте PCA_TIMEOUT.") from exc

    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ProviderError("LLM вернула не-JSON ответ на уровне транспорта.") from exc


class OpenAIProvider:
    """Chat Completions API. Ответ принудительно в JSON-режиме."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self.name = "openai"
        self.model = settings.model

    def generate(self, product: ProductInput, report: InputReport) -> str:
        api_key = self._settings.require_api_key()
        payload = {
            "model": self.model,
            "temperature": 0.2,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(product, report)},
            ],
        }
        data = _post_json(
            f"{self._settings.base_url}/chat/completions",
            payload,
            {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            self._settings.timeout,
        )
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError("Неожиданная структура ответа OpenAI.") from exc


class AnthropicProvider:
    """Messages API. JSON-режима нет, поэтому формат держится промптом."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self.name = "anthropic"
        self.model = settings.model

    def generate(self, product: ProductInput, report: InputReport) -> str:
        api_key = self._settings.require_api_key()
        payload = {
            "model": self.model,
            "max_tokens": 2000,
            "temperature": 0.2,
            "system": SYSTEM_PROMPT,
            "messages": [
                {"role": "user", "content": build_user_prompt(product, report)},
                # Префилл ответа ассистента сокращает риск markdown-обвязки.
                {"role": "assistant", "content": "{"},
            ],
        }
        data = _post_json(
            f"{self._settings.base_url}/v1/messages",
            payload,
            {
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            self._settings.timeout,
        )
        try:
            text = "".join(
                block.get("text", "")
                for block in data["content"]
                if block.get("type") == "text"
            )
        except (KeyError, TypeError) as exc:
            raise ProviderError("Неожиданная структура ответа Anthropic.") from exc
        # Возвращаем префилл на место.
        return "{" + text if not text.lstrip().startswith("{") else text


class OfflineProvider:
    """Детерминированная сборка карточки без обращения к LLM.

    Нужна для двух вещей: демонстрации прототипа без API-ключа и для тестов.
    Провайдер принципиально не умеет выдумывать: он только переставляет
    факты из входных данных и расставляет пометки «данных нет».
    Текст получается суше, чем у настоящей модели, — это осознанный размен.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.name = "offline"
        self.model = "offline-rule-based"
        self._settings = settings

    def generate(self, product: ProductInput, report: InputReport) -> str:
        manufacturer = product.manufacturer
        title = product.name if not manufacturer else f"{product.name} — {manufacturer}"
        if len(title) > TITLE_MAX:
            title = product.name[:TITLE_MAX].rstrip(" ,-—")

        key_specs = [{"label": k, "value": v} for k, v in product.specs.items()]
        known_labels = " ".join(product.specs.keys()).lower()
        for hint in EXPECTED_SPEC_HINTS:
            if hint not in known_labels:
                key_specs.append({"label": hint, "value": NO_DATA})

        description = self._build_description(product)
        queries = self._build_queries(product)

        audience = (
            "Снабженцы и инженеры КИПиА промышленных предприятий, монтажные "
            "и сервисные организации, закупающие измерительные приборы."
        )
        if manufacturer:
            audience += f" Подходит тем, кто работает с оборудованием производителя «{manufacturer}»."

        if product.specs:
            listed = ", ".join(f"{k}: {v}" for k, v in list(product.specs.items())[:4])
            tasks = (
                "Прибор применяется в рамках заявленных в паспорте параметров "
                f"({listed}). Уточняйте условия применения по техпаспорту."
            )
        else:
            tasks = (
                f"{NO_DATA}: технические характеристики не переданы, назначение "
                "нужно подтвердить по техпаспорту."
            )

        missing = list(report.missing_fields)
        missing.extend(
            f"{item['label']}: {NO_DATA}" for item in key_specs if item["value"] == NO_DATA
        )

        result = {
            "seo_title": title,
            "seo_description": description,
            "key_specs": key_specs,
            "search_queries": queries,
            "commercial_block": {"audience": audience, "tasks": tasks},
            "missing_data": missing,
        }
        return json.dumps(result, ensure_ascii=False)

    def _build_description(self, product: ProductInput) -> str:
        """Собрать SEO-описание только из переданных фактов.

        Добиваем длину до нижней границы нейтральными формулировками о порядке
        работы с карточкой, а не выдуманными параметрами.
        """
        parts: list[str] = []
        if product.manufacturer:
            parts.append(f"{product.name} производства «{product.manufacturer}».")
        else:
            parts.append(f"{product.name}. Производитель: {NO_DATA}.")

        if product.description:
            parts.append(product.description)
        else:
            parts.append(f"Описание из техпаспорта: {NO_DATA}.")

        if product.specs:
            parts.append(
                "Заявленные характеристики: "
                + "; ".join(f"{k} — {v}" for k, v in product.specs.items())
                + "."
            )
        else:
            parts.append(f"Технические характеристики: {NO_DATA}.")

        parts.append(
            "Перед заказом сверьте параметры с техническим паспортом и условиями "
            "эксплуатации на объекте: тип измеряемой среды, рабочее давление и "
            "температуру, способ монтажа и требования к присоединению. "
            "Значения, не подтверждённые паспортом, в карточке не указываются."
        )
        parts.append(
            "Поставка выполняется по запросу: уточняйте у менеджера наличие, "
            "комплектацию, наличие поверки и сопроводительных документов. "
            "Этот текст — черновик для контент-менеджера: проверьте фактические "
            "данные и дополните недостающие поля перед публикацией."
        )

        text = " ".join(parts)
        if len(text) < DESCRIPTION_MIN:
            text += (
                " Недостающие сведения отмечены пометкой «"
                + NO_DATA
                + "»: их нужно запросить у поставщика или взять из технической "
                "документации, после чего карточку можно публиковать. "
                "Автоматическая генерация намеренно не достраивает параметры, "
                "которых нет во входных данных, чтобы в каталог не попали "
                "недостоверные значения."
            )
        return text

    def _build_queries(self, product: ProductInput) -> list[str]:
        name = product.name.lower()
        queries = [name, f"{name} купить", f"{name} цена"]
        if product.manufacturer:
            queries.append(f"{name} {product.manufacturer.lower()}")
        if product.specs:
            first_label, first_value = next(iter(product.specs.items()))
            queries.append(f"{name} {first_label.lower()} {first_value.lower()}")
        # Дедупликация с сохранением порядка.
        unique: list[str] = []
        for query in queries:
            if query not in unique:
                unique.append(query)
        return unique[:QUERIES_MAX]


def build_provider(settings: Settings) -> Provider:
    if settings.provider == "openai":
        return OpenAIProvider(settings)
    if settings.provider == "anthropic":
        return AnthropicProvider(settings)
    return OfflineProvider(settings)