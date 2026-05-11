from __future__ import annotations

import json
import random
import re
import time
from dataclasses import dataclass
from typing import Any

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None


@dataclass
class LLMRequest:
    system_prompt: str
    user_prompt: str
    model: str
    temperature: float
    max_tokens: int


class LLMClient:
    def generate_json(self, req: LLMRequest) -> dict[str, Any]:
        raise NotImplementedError


class OpenAIClient(LLMClient):
    def __init__(
        self,
        api_key: str,
        timeout: float = 60,
        max_retries: int = 5,
        retry_initial_delay: float = 1.0,
        retry_max_delay: float = 30.0,
    ):
        if OpenAI is None:
            raise RuntimeError("openai package is not installed")
        self.client = OpenAI(api_key=api_key, timeout=timeout)
        self.max_retries = max(0, max_retries)
        self.retry_initial_delay = max(0.1, retry_initial_delay)
        self.retry_max_delay = max(self.retry_initial_delay, retry_max_delay)

    def generate_json(self, req: LLMRequest) -> dict[str, Any]:
        for attempt in range(self.max_retries + 1):
            try:
                completion = self.client.chat.completions.create(
                    model=req.model,
                    temperature=req.temperature,
                    max_tokens=req.max_tokens,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": req.system_prompt},
                        {"role": "user", "content": req.user_prompt},
                    ],
                )
                content = completion.choices[0].message.content or "{}"
                return _parse_json_object(content)
            except Exception as exc:
                if attempt >= self.max_retries or not _is_retryable_openai_error(exc):
                    raise
                delay = self._retry_delay(exc, attempt)
                time.sleep(delay)
        raise RuntimeError("unreachable retry state")

    def _retry_delay(self, exc: Exception, attempt: int) -> float:
        retry_after = getattr(exc, "retry_after", None)
        if retry_after is not None:
            try:
                return min(float(retry_after), self.retry_max_delay)
            except (TypeError, ValueError):
                pass
        jitter = random.uniform(0, self.retry_initial_delay)
        exponential = self.retry_initial_delay * (2**attempt)
        return min(exponential + jitter, self.retry_max_delay)


class GeminiClient(LLMClient):
    def __init__(self, api_key: str, timeout: float = 60):
        try:
            import google.generativeai as genai
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("google-generativeai package is not installed") from exc
        self.genai = genai
        self.timeout = timeout
        if self.genai is None:
            raise RuntimeError("google-generativeai package is not installed")
        self.genai.configure(api_key=api_key)

    def generate_json(self, req: LLMRequest) -> dict[str, Any]:
        model = self.genai.GenerativeModel(
            req.model,
            system_instruction=req.system_prompt,
        )
        response = model.generate_content(
            req.user_prompt + "\n\n請只輸出有效 JSON，不要加 markdown code fence。",
            generation_config={
                "temperature": req.temperature,
                "max_output_tokens": req.max_tokens,
                "response_mime_type": "application/json",
            },
            request_options={"timeout": self.timeout},
        )
        return _parse_json_object(response.text or "{}")


def _parse_json_object(content: str) -> dict[str, Any]:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, flags=re.S)
        if not match:
            raise
        return json.loads(match.group(0))


def _is_retryable_openai_error(exc: Exception) -> bool:
    name = exc.__class__.__name__
    if name in {"RateLimitError", "APITimeoutError", "APIConnectionError", "InternalServerError"}:
        return True
    status_code = getattr(exc, "status_code", None)
    return isinstance(status_code, int) and status_code in {408, 409, 429, 500, 502, 503, 504}


class MockLLMClient(LLMClient):
    """Deterministic mock for local simulation without API calls."""

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def generate_json(self, req: LLMRequest) -> dict[str, Any]:
        user = req.user_prompt
        if "task=argument_selection" in user:
            # selection task: parse candidate ids from JSON payload in prompt
            ids = re.findall(r'"id"\s*:\s*"([^"]+)"', user)
            unique = []
            for arg_id in ids:
                if arg_id not in unique:
                    unique.append(arg_id)
            if len(unique) < 5:
                unique += [f"MOCK_{100+i}" for i in range(5 - len(unique))]
            scored = []
            for arg_id in unique:
                score_match = re.search(
                    rf'"id"\s*:\s*"{re.escape(arg_id)}".*?"strength_score"\s*:\s*([0-9.]+)',
                    user,
                    flags=re.S,
                )
                score = float(score_match.group(1)) if score_match else self.rng.uniform(6.0, 9.0)
                scored.append((arg_id, score + self.rng.uniform(-0.35, 0.35)))
            scored.sort(key=lambda item: item[1], reverse=True)
            unique = [arg_id for arg_id, _ in scored]
            return {
                "main_argument_ids": unique[:3],
                "defense_argument_ids": unique[3:5],
                "reason": "依強度分數、可防守性與攻擊對手能力排序，前三個作主軸，後兩個作防守。",
            }

        if "task=judge" in user:
            pro_args = re.findall(r"P_U\d+", user)
            con_args = re.findall(r"C_U\d+", user)
            pro_base = 300 + len(set(pro_args)) * 8
            con_base = 300 + len(set(con_args)) * 8
            pro = pro_base + self.rng.randint(0, 95)
            con = con_base + self.rng.randint(0, 95)
            winner = "pro" if pro >= con else "con"
            best_pool = pro_args if winner == "pro" and pro_args else con_args
            worst_pool = con_args if winner == "pro" and con_args else pro_args
            return {
                "speaker_scores": {
                    "P1": self.rng.randint(70, 90),
                    "P2": self.rng.randint(70, 90),
                    "P3": self.rng.randint(70, 90),
                    "C1": self.rng.randint(70, 90),
                    "C2": self.rng.randint(70, 90),
                    "C3": self.rng.randint(70, 90),
                },
                "team_scores": {"pro": self.rng.randint(5, 10), "con": self.rng.randint(5, 10)},
                "final_total": {"pro": pro, "con": con},
                "winner": winner,
                "key_clash": self.rng.choice([
                    "就業入口 vs 最低保障",
                    "市場效率 vs 國家底線",
                    "補貼精準度 vs 工資底線",
                    "正式保障 vs 灰色勞動",
                ]),
                "turning_point": "質詢把對方因果鏈中最容易被攻擊的一段凸顯出來。",
                "best_argument": self.rng.choice(best_pool or ["P_U01", "C_U40"]),
                "worst_argument": self.rng.choice(worst_pool or ["P_U07", "C_U30"]),
                "judge_reason": "勝方比較清楚說明 Claim、Reason、Example、Impact，也把關鍵衝突拉回本場已選論點。",
            }

        # speech task
        allowed = re.findall(r'"id"\s*:\s*"([^"]+)"', user)
        picked = ""
        if allowed:
            picked = self.rng.choice(allowed)
        picked = picked or "P_U01"
        if "mode=cross_exam" in user:
            return {
                "content": (
                    f"質詢者：你方 {picked} 的因果鏈是不是假設雇主和勞工都沒有其他選擇？\n"
                    "答辯者：不是，我方承認現場有不同反應，但核心是制度會改變誘因。\n"
                    "質詢者：如果同一結果也可能由景氣、缺工或產業轉型造成，你方怎麼證明是最低工資？\n"
                    "答辯者：我方會回到本場已選論點，說明最低工資至少是可被政策控制的直接因素。\n"
                    "質詢者：所以你方其實不能排除其他原因，對嗎？\n"
                    "答辯者：不能排除全部原因，但可以比較哪個制度工具更能處理主要風險。"
                ),
                "used_argument_ids": [picked],
            }
        return {
            "content": (
                f"Claim: 我方本段使用 {picked} 作為核心攻防。"
                "Reason: 這個論點的因果鏈能解釋制度如何影響勞工與雇主的實際選擇。"
                "Example: 以便利商店排班、地方小店聘人或弱勢勞工求職情境來看，規則改變會直接影響工作機會與保障方式。"
                "Impact: 因此評審應把勝負放在這個論點能否真正處理辯題傷害，而不是只看口號。"
            ),
            "used_argument_ids": [picked],
        }
