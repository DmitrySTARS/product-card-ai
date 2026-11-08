"""Простая веб-форма.

Запуск: python -m product_card_ai.web
Открыть: http://127.0.0.1:8000

Намеренно на http.server из стандартной библиотеки — прототипу не нужен
веб-фреймворк, а запуск остаётся одной командой без установки зависимостей.
Сервер рассчитан на локальное использование одним человеком.
"""

from __future__ import annotations

import argparse
import html
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs

from .config import ConfigError, Settings, load_dotenv
from .pipeline import ModelOutputError, generate_card
from .providers import ProviderError, build_provider
from .renderers import render
from .validation import InvalidInputError, load_json, parse_products

SAMPLE = json.dumps(
    {
        "name": "Манометр показывающий ДМ 02",
        "manufacturer": "Манотомь",
        "specs": {
            "диапазон измерений": "0–1,6 МПа",
            "класс точности": "1,5",
            "присоединение": "G1/2",
            "диаметр корпуса": "100 мм",
        },
        "description": "Прибор предназначен для измерения избыточного давления...",
    },
    ensure_ascii=False,
    indent=2,
)

PAGE = """<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Черновик карточки товара</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{ font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
         max-width: 900px; margin: 0 auto; padding: 24px; line-height: 1.5; }}
  h1 {{ font-size: 1.4rem; }}
  textarea {{ width: 100%; min-height: 260px; font-family: ui-monospace, Consolas, monospace;
              font-size: 13px; padding: 10px; box-sizing: border-box; }}
  .row {{ display: flex; gap: 12px; align-items: center; margin: 12px 0; flex-wrap: wrap; }}
  button {{ padding: 8px 18px; font-size: 14px; cursor: pointer; }}
  pre {{ background: rgba(127,127,127,.12); padding: 14px; overflow-x: auto;
         white-space: pre-wrap; word-wrap: break-word; }}
  .error {{ border-left: 4px solid #c0392b; padding: 10px 14px; background: rgba(192,57,43,.1); }}
  .warn {{ border-left: 4px solid #d68910; padding: 10px 14px; background: rgba(214,137,16,.1); }}
  .hint {{ color: #666; font-size: 13px; }}
</style>
</head>
<body>
<h1>Черновик карточки товара для B2B-магазина КИП</h1>
<p class="hint">Вставьте JSON с данными о приборе. Характеристики берутся только из
входных данных: то, чего нет, помечается «данных нет».</p>
<form method="post" action="/">
  <textarea name="payload" spellcheck="false">{payload}</textarea>
  <div class="row">
    <label>Провайдер:
      <select name="provider">
        <option value="offline"{sel_offline}>offline (без ключа)</option>
        <option value="openai"{sel_openai}>openai</option>
        <option value="anthropic"{sel_anthropic}>anthropic</option>
      </select>
    </label>
    <label>Формат:
      <select name="format">
        <option value="text"{sel_text}>Текст</option>
        <option value="markdown"{sel_md}>Markdown</option>
        <option value="json"{sel_json}>JSON</option>
      </select>
    </label>
    <button type="submit">Сформировать</button>
  </div>
</form>
{result}
</body>
</html>
"""


def _render_page(
    payload: str,
    provider: str,
    fmt: str,
    result_html: str = "",
) -> str:
    return PAGE.format(
        payload=html.escape(payload),
        result=result_html,
        sel_offline=" selected" if provider == "offline" else "",
        sel_openai=" selected" if provider == "openai" else "",
        sel_anthropic=" selected" if provider == "anthropic" else "",
        sel_text=" selected" if fmt == "text" else "",
        sel_md=" selected" if fmt == "markdown" else "",
        sel_json=" selected" if fmt == "json" else "",
    )


def _process(payload: str, provider_name: str, fmt: str) -> str:
    """Вернуть HTML-блок с результатом или описанием ошибки."""
    try:
        parsed = parse_products(load_json(payload))
    except InvalidInputError as exc:
        return f'<div class="error"><b>Ошибка входных данных.</b> {html.escape(str(exc))}</div>'

    try:
        settings = Settings.from_env(provider_name)
        provider = build_provider(settings)
    except ConfigError as exc:
        return f'<div class="error"><b>Ошибка конфигурации.</b> {html.escape(str(exc))}</div>'

    blocks: list[str] = []
    for product, report in parsed:
        try:
            card = generate_card(product, report, provider)
        except ConfigError as exc:
            return f'<div class="error"><b>Ошибка конфигурации.</b> {html.escape(str(exc))}</div>'
        except (ProviderError, ModelOutputError) as exc:
            return (
                '<div class="error"><b>Ошибка LLM для «'
                + html.escape(product.name)
                + "».</b> "
                + html.escape(str(exc))
                + "</div>"
            )

        if not card.quality.ok:
            issues = "".join(f"<li>{html.escape(i)}</li>" for i in card.quality.issues)
            blocks.append(f'<div class="warn"><b>Автопроверка.</b><ul>{issues}</ul></div>')

        text = render(card, fmt if fmt in ("text", "markdown", "json") else "text", report)
        blocks.append(f"<pre>{html.escape(text)}</pre>")

    return "<h2>Результат</h2>" + "".join(blocks)


class Handler(BaseHTTPRequestHandler):
    server_version = "product-card-ai/0.1"

    def _send(self, body: str, status: int = 200) -> None:
        encoded = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self) -> None:  # noqa: N802 - имя задано базовым классом
        if self.path not in ("/", "/index.html"):
            self._send("<h1>404</h1>", 404)
            return
        self._send(_render_page(SAMPLE, "offline", "text"))

    def do_POST(self) -> None:  # noqa: N802 - имя задано базовым классом
        length = int(self.headers.get("Content-Length") or 0)
        if length > 1_000_000:
            self._send('<div class="error">Слишком большой запрос.</div>', 413)
            return
        raw = self.rfile.read(length).decode("utf-8", errors="replace")
        form = parse_qs(raw, keep_blank_values=True)
        payload = (form.get("payload") or [""])[0]
        provider = (form.get("provider") or ["offline"])[0]
        fmt = (form.get("format") or ["text"])[0]

        if not payload.strip():
            result = '<div class="error">Поле ввода пустое.</div>'
        else:
            result = _process(payload, provider, fmt)
        self._send(_render_page(payload, provider, fmt, result))

    def log_message(self, fmt: str, *args) -> None:
        # Тише стандартного логирования, но запросы видно.
        print(f"[web] {fmt % args}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="product-card-ai-web")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--env", default=".env")
    args = parser.parse_args(argv)

    load_dotenv(args.env)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Веб-форма доступна на http://{args.host}:{args.port} (Ctrl+C — остановить)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nОстановлено.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
