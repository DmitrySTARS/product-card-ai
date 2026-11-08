"""CLI прототипа.

Пример:
    python -m product_card_ai.cli --input samples/products.json --format text
    type samples\\products.json | python -m product_card_ai.cli --provider offline
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import ConfigError, Settings, load_dotenv
from .pipeline import ModelOutputError, generate_card
from .providers import ProviderError, build_provider
from .renderers import render
from .validation import InvalidInputError, load_json, parse_products

EXIT_OK = 0
EXIT_INPUT_ERROR = 2
EXIT_PROVIDER_ERROR = 3
EXIT_CONFIG_ERROR = 4


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="product-card-ai",
        description=(
            "Черновик карточки товара для B2B-магазина КИП по входному JSON. "
            "Характеристики берутся только из входных данных."
        ),
    )
    parser.add_argument(
        "--input",
        "-i",
        help="Путь к JSON-файлу с товаром или списком товаров. По умолчанию stdin.",
    )
    parser.add_argument(
        "--provider",
        choices=("offline", "openai", "anthropic"),
        help="Провайдер LLM. По умолчанию берётся из PCA_PROVIDER (offline).",
    )
    parser.add_argument(
        "--format",
        "-f",
        choices=("text", "markdown", "json"),
        default="text",
        help="Формат вывода. По умолчанию text (обычный текст).",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Файл для записи результата. По умолчанию stdout.",
    )
    parser.add_argument(
        "--env",
        default=".env",
        help="Путь к файлу с переменными окружения. По умолчанию .env.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Завершаться с ненулевым кодом, если автопроверка нашла замечания.",
    )
    return parser


def read_input(path: str | None) -> str:
    if path:
        file_path = Path(path)
        if not file_path.is_file():
            raise InvalidInputError(f"Файл не найден: {path}")
        return file_path.read_text(encoding="utf-8")
    data = sys.stdin.read()
    if not data.strip():
        raise InvalidInputError(
            "Пустой ввод. Передайте JSON через --input или на стандартный вход."
        )
    return data


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    load_dotenv(args.env)

    try:
        raw_input_text = read_input(args.input)
        parsed = parse_products(load_json(raw_input_text))
    except InvalidInputError as exc:
        print(f"Ошибка входных данных: {exc}", file=sys.stderr)
        return EXIT_INPUT_ERROR

    try:
        settings = Settings.from_env(args.provider)
        provider = build_provider(settings)
    except ConfigError as exc:
        print(f"Ошибка конфигурации: {exc}", file=sys.stderr)
        return EXIT_CONFIG_ERROR

    chunks: list[str] = []
    has_warnings = False

    for product, report in parsed:
        try:
            card = generate_card(product, report, provider)
        except ConfigError as exc:
            print(f"Ошибка конфигурации: {exc}", file=sys.stderr)
            return EXIT_CONFIG_ERROR
        except (ProviderError, ModelOutputError) as exc:
            print(f"Ошибка LLM для «{product.name}»: {exc}", file=sys.stderr)
            return EXIT_PROVIDER_ERROR

        if not card.quality.ok:
            has_warnings = True
            for issue in card.quality.issues:
                print(f"Предупреждение [{product.name}]: {issue}", file=sys.stderr)

        chunks.append(render(card, args.format, report))

    if args.format == "markdown":
        separator = "\n\n---\n\n"
    elif args.format == "text":
        separator = "\n\n" + "=" * 70 + "\n\n"
    else:
        separator = "\n\n"
    result = separator.join(chunks)

    if args.output:
        Path(args.output).write_text(result, encoding="utf-8")
        print(f"Результат записан в {args.output}", file=sys.stderr)
    else:
        sys.stdout.write(result + "\n")

    if args.strict and has_warnings:
        return 1
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
