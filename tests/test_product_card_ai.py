"""Тесты прототипа.

Запуск: python -m unittest discover -s tests -v

Проверяем главное из ТЗ: разбор некорректного ввода, пометки «данных нет»
и детектор выдуманных числовых характеристик.
"""

from __future__ import annotations

import json
import unittest

from product_card_ai.config import ConfigError, Settings
from product_card_ai.models import NO_DATA, InputReport, ProductInput
from product_card_ai.pipeline import (
    ModelOutputError,
    build_card,
    generate_card,
    parse_model_output,
)
from product_card_ai.providers import OfflineProvider
from product_card_ai.renderers import to_json, to_markdown
from product_card_ai.validation import (
    InvalidInputError,
    load_json,
    parse_product,
    parse_products,
)

FULL_PRODUCT = {
    "name": "Манометр показывающий ДМ 02",
    "manufacturer": "Манотомь",
    "specs": {
        "диапазон измерений": "0–1,6 МПа",
        "класс точности": "1,5",
        "присоединение": "G1/2",
        "диаметр корпуса": "100 мм",
    },
    "description": "Прибор предназначен для измерения избыточного давления "
    "неагрессивных к медным сплавам жидкостей, газов и пара в системах отопления.",
}

SPARSE_PRODUCT = {
    "name": "Датчик давления СДВ-И",
    "specs": {"выходной сигнал": "4–20 мА"},
}


class ValidationTests(unittest.TestCase):
    def test_broken_json_reports_position(self):
        with self.assertRaises(InvalidInputError) as ctx:
            load_json('{"name": "Манометр",}')
        self.assertIn("Некорректный JSON", str(ctx.exception))

    def test_missing_name_is_fatal(self):
        with self.assertRaises(InvalidInputError):
            parse_product({"manufacturer": "Манотомь"})

    def test_non_object_input_is_fatal(self):
        with self.assertRaises(InvalidInputError):
            parse_product("просто строка")

    def test_missing_fields_are_reported(self):
        _, report = parse_product(SPARSE_PRODUCT)
        joined = " ".join(report.missing_fields)
        self.assertIn("manufacturer", joined)
        self.assertIn("description", joined)

    def test_empty_spec_value_becomes_clarification(self):
        _, report = parse_product(
            {"name": "Прибор", "specs": {"класс точности": "   "}}
        )
        self.assertTrue(
            any("класс точности" in note for note in report.needs_clarification)
        )

    def test_manual_no_data_marker_is_not_a_value(self):
        product, report = parse_product(
            {"name": "Прибор", "manufacturer": "данных нет"}
        )
        self.assertIsNone(product.manufacturer)
        self.assertTrue(any("manufacturer" in f for f in report.missing_fields))

    def test_specs_as_list_of_objects(self):
        product, _ = parse_product(
            {
                "name": "Прибор",
                "specs": [{"label": "класс точности", "value": "1,5"}],
            }
        )
        self.assertEqual(product.specs["класс точности"], "1,5")

    def test_parse_products_accepts_wrapper_and_list(self):
        self.assertEqual(len(parse_products({"products": [FULL_PRODUCT, SPARSE_PRODUCT]})), 2)
        self.assertEqual(len(parse_products([FULL_PRODUCT])), 1)
        self.assertEqual(len(parse_products(FULL_PRODUCT)), 1)

    def test_empty_product_list_is_fatal(self):
        with self.assertRaises(InvalidInputError):
            parse_products([])


class ModelOutputTests(unittest.TestCase):
    def test_code_fence_is_stripped(self):
        data = parse_model_output('```json\n{"seo_title": "Тест"}\n```')
        self.assertEqual(data["seo_title"], "Тест")

    def test_surrounding_prose_is_tolerated(self):
        data = parse_model_output('Вот результат: {"seo_title": "Тест"} Готово.')
        self.assertEqual(data["seo_title"], "Тест")

    def test_empty_response_raises(self):
        with self.assertRaises(ModelOutputError):
            parse_model_output("   ")

    def test_non_json_response_raises(self):
        with self.assertRaises(ModelOutputError):
            parse_model_output("Извините, не могу помочь.")


class QualityTests(unittest.TestCase):
    """Главная проверка ТЗ: выдуманные характеристики должны ловиться."""

    def setUp(self):
        self.product, self.report = parse_product(FULL_PRODUCT)

    def _card_from(self, payload: dict):
        return build_card(payload, self.product, self.report, "test", "test")

    def test_fabricated_number_is_flagged(self):
        card = self._card_from(
            {
                "seo_title": "Манометр ДМ 02",
                "seo_description": "Степень защиты IP65, гарантия 24 месяца." * 20,
                "key_specs": [{"label": "степень защиты", "value": "IP65"}],
                "search_queries": ["манометр", "манометр купить", "манометр цена"],
                "commercial_block": {"audience": "снабженцы", "tasks": "измерение"},
            }
        )
        self.assertFalse(card.quality.ok)
        self.assertIn("65", card.quality.unsupported_values)
        self.assertIn("24", card.quality.unsupported_values)

    def test_values_from_input_are_not_flagged(self):
        card = self._card_from(
            {
                "seo_title": "Манометр ДМ 02, 0–1,6 МПа",
                "seo_description": "Класс точности 1,5, присоединение G1/2.",
                "key_specs": [{"label": "класс точности", "value": "1,5"}],
                "search_queries": ["манометр", "манометр купить", "манометр цена"],
                "commercial_block": {"audience": "снабженцы", "tasks": "измерение"},
            }
        )
        self.assertEqual(card.quality.unsupported_values, [])

    def test_long_title_is_flagged(self):
        card = self._card_from({"seo_title": "Манометр " * 20, "seo_description": "текст"})
        self.assertTrue(any("заголовок" in i.lower() for i in card.quality.issues))

    def test_description_length_is_checked(self):
        card = self._card_from({"seo_title": "Манометр", "seo_description": "коротко"})
        self.assertTrue(any("описание" in i.lower() for i in card.quality.issues))

    def test_query_count_is_checked(self):
        card = self._card_from(
            {"seo_title": "Манометр", "seo_description": "a", "search_queries": ["один"]}
        )
        self.assertTrue(any("апрос" in i for i in card.quality.issues))

    def test_validator_gaps_are_merged_into_missing_data(self):
        product, report = parse_product(SPARSE_PRODUCT)
        card = build_card({"seo_title": "Датчик"}, product, report, "test", "test")
        self.assertTrue(any("manufacturer" in item for item in card.missing_data))


class OfflineProviderTests(unittest.TestCase):
    def setUp(self):
        self.provider = OfflineProvider()

    def test_full_product_passes_all_checks(self):
        product, report = parse_product(FULL_PRODUCT)
        card = generate_card(product, report, self.provider)
        self.assertTrue(card.quality.ok, card.quality.issues)
        self.assertLessEqual(len(card.seo_title), 70)
        self.assertGreaterEqual(len(card.seo_description), 800)
        self.assertLessEqual(len(card.seo_description), 1200)
        self.assertGreaterEqual(len(card.search_queries), 3)
        self.assertLessEqual(len(card.search_queries), 5)

    def test_sparse_product_marks_gaps_and_stays_clean(self):
        product, report = parse_product(SPARSE_PRODUCT)
        card = generate_card(product, report, self.provider)
        self.assertTrue(card.quality.ok, card.quality.issues)
        self.assertTrue(any(item.value == NO_DATA for item in card.key_specs))
        self.assertTrue(card.missing_data)

    def test_offline_never_invents_numbers(self):
        for payload in (FULL_PRODUCT, SPARSE_PRODUCT, {"name": "Прибор без данных"}):
            product, report = parse_product(payload)
            card = generate_card(product, report, self.provider)
            self.assertEqual(
                card.quality.unsupported_values, [], f"выдуманные числа для {payload['name']}"
            )

    def test_output_is_valid_json(self):
        product, report = parse_product(FULL_PRODUCT)
        raw = self.provider.generate(product, report)
        self.assertIsInstance(json.loads(raw), dict)


class RendererTests(unittest.TestCase):
    def setUp(self):
        product, report = parse_product(FULL_PRODUCT)
        self.report = report
        self.card = generate_card(product, report, OfflineProvider())

    def test_markdown_contains_sections(self):
        text = to_markdown(self.card, self.report)
        for heading in ("Ключевые характеристики", "Поисковые запросы", "Для менеджера"):
            self.assertIn(heading, text)

    def test_json_round_trips(self):
        data = json.loads(to_json(self.card))
        self.assertEqual(data["product_name"], self.card.product_name)
        self.assertIn("quality", data)


class ConfigTests(unittest.TestCase):
    def test_unknown_provider_rejected(self):
        with self.assertRaises(ConfigError):
            Settings.from_env("gigachat-3000")

    def test_missing_key_message_is_actionable(self):
        settings = Settings(provider="openai", api_key=None, model="gpt-4o-mini")
        with self.assertRaises(ConfigError) as ctx:
            settings.require_api_key()
        self.assertIn("OPENAI_API_KEY", str(ctx.exception))

    def test_offline_needs_no_key(self):
        self.assertEqual(Settings.from_env("offline").provider, "offline")


class ProductInputTests(unittest.TestCase):
    def test_source_text_includes_all_facts(self):
        product = ProductInput(
            name="Прибор",
            manufacturer="Завод",
            specs={"класс точности": "1,5"},
            description="Описание",
        )
        text = product.source_text()
        for fragment in ("Прибор", "Завод", "1,5", "Описание"):
            self.assertIn(fragment, text)

    def test_markers_for_absent_fields(self):
        product = ProductInput(name="Прибор")
        self.assertEqual(product.manufacturer_or_marker, NO_DATA)
        self.assertEqual(product.description_or_marker, NO_DATA)

    def test_empty_report_has_no_gaps(self):
        self.assertFalse(InputReport().has_gaps)


if __name__ == "__main__":
    unittest.main()
