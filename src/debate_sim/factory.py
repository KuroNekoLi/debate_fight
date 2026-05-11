from __future__ import annotations

from debate_sim.application.engine import DebateSimulationEngine
from debate_sim.config import AppConfig
from debate_sim.infrastructure.llm import GeminiClient, MockLLMClient, OpenAIClient


def build_engine(config: AppConfig) -> DebateSimulationEngine:
    provider = config.provider
    if provider == "openai":
        if not config.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required when DEBATE_LLM_PROVIDER=openai")
        llm = OpenAIClient(
            api_key=config.openai_api_key,
            timeout=config.request_timeout,
            max_retries=config.max_retries,
            retry_initial_delay=config.retry_initial_delay,
            retry_max_delay=config.retry_max_delay,
        )
    elif provider == "gemini":
        if not config.gemini_api_key:
            raise ValueError("GEMINI_API_KEY is required when DEBATE_LLM_PROVIDER=gemini")
        llm = GeminiClient(api_key=config.gemini_api_key, timeout=config.request_timeout)
    else:
        llm = MockLLMClient(seed=config.seed)

    return DebateSimulationEngine(config=config, llm=llm)
