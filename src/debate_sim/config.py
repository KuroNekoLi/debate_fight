from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None


@dataclass(frozen=True)
class AppConfig:
    provider: str
    model: str
    openai_api_key: str | None
    gemini_api_key: str | None
    temperature: float
    max_tokens: int
    request_timeout: float
    max_retries: int
    retry_initial_delay: float
    retry_max_delay: float
    max_workers: int
    seed: int
    data_path: Path
    results_dir: Path


    @staticmethod
    def from_env() -> "AppConfig":
        if load_dotenv is not None:
            load_dotenv()
        else:
            AppConfig._load_dotenv_fallback(Path(".env"))
        provider = os.getenv("DEBATE_LLM_PROVIDER", "mock").lower()
        model = os.getenv("DEBATE_MODEL", "gpt-4o-mini")
        openai_api_key = os.getenv("OPENAI_API_KEY")
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        temperature = float(os.getenv("DEBATE_TEMPERATURE", "0.4"))
        max_tokens = int(os.getenv("DEBATE_MAX_TOKENS", "1200"))
        request_timeout = float(os.getenv("DEBATE_REQUEST_TIMEOUT", "60"))
        max_retries = int(os.getenv("DEBATE_MAX_RETRIES", "5"))
        retry_initial_delay = float(os.getenv("DEBATE_RETRY_INITIAL_DELAY", "1"))
        retry_max_delay = float(os.getenv("DEBATE_RETRY_MAX_DELAY", "30"))
        max_workers = int(os.getenv("DEBATE_MAX_WORKERS", "1"))
        seed = int(os.getenv("DEBATE_SEED", "42"))
        data_path = Path(os.getenv("DEBATE_ARGUMENTS_PATH", "data/arguments.json"))
        results_dir = Path(os.getenv("DEBATE_RESULTS_DIR", "results"))
        return AppConfig(
            provider=provider,
            model=model,
            openai_api_key=openai_api_key,
            gemini_api_key=gemini_api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            request_timeout=request_timeout,
            max_retries=max_retries,
            retry_initial_delay=retry_initial_delay,
            retry_max_delay=retry_max_delay,
            max_workers=max_workers,
            seed=seed,
            data_path=data_path,
            results_dir=results_dir,
        )

    @staticmethod
    def _load_dotenv_fallback(path: Path) -> None:
        if not path.exists():
            return
        for raw_line in path.read_text().splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            os.environ.setdefault(key, value)
