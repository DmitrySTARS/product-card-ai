"""Конфигурация прототипа.

Ключ API читается только из переменных окружения или из файла .env,
который добавлен в .gitignore. В коде ключей нет.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_PROVIDER = "offline"
KNOWN_PROVIDERS = ("offline", "openai", "anthropic")


class ConfigError(RuntimeError):
    """Конфигурация не позволяет выполнить запрос."""


def load_dotenv(path: str | Path = ".env") -> None:
    """Подхватить переменные из .env, не перетирая уже заданные в окружении.

    Намеренно простой разбор: KEY=VALUE, строки с # игнорируются.
    Внешние зависимости прототипу не нужны.
    """
    env_path = Path(path)
    if not env_path.is_file():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


@dataclass(frozen=True)
class Settings:
    provider: str = DEFAULT_PROVIDER
    api_key: str | None = None
    model: str = ""
    base_url: str = ""
    timeout: int = 60

    @classmethod
    def from_env(cls, provider: str | None = None) -> "Settings":
        name = (provider or os.getenv("PCA_PROVIDER") or DEFAULT_PROVIDER).strip().lower()
        if name not in KNOWN_PROVIDERS:
            raise ConfigError(
                f"Неизвестный провайдер «{name}». Доступны: {', '.join(KNOWN_PROVIDERS)}."
            )

        try:
            timeout = int(os.getenv("PCA_TIMEOUT", "60"))
        except ValueError:
            timeout = 60

        if name == "openai":
            return cls(
                provider=name,
                api_key=os.getenv("OPENAI_API_KEY") or None,
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/"),
                timeout=timeout,
            )
        if name == "anthropic":
            return cls(
                provider=name,
                api_key=os.getenv("ANTHROPIC_API_KEY") or None,
                model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5"),
                base_url=os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com").rstrip("/"),
                timeout=timeout,
            )
        return cls(provider="offline", model="offline-rule-based", timeout=timeout)

    def require_api_key(self) -> str:
        if not self.api_key:
            env_name = "OPENAI_API_KEY" if self.provider == "openai" else "ANTHROPIC_API_KEY"
            raise ConfigError(
                f"Не задан {env_name}. Укажите его в .env или в переменных окружения, "
                "либо запустите прототип с --provider offline."
            )
        return self.api_key
