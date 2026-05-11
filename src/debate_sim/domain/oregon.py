from __future__ import annotations

from .models import Side, SpeechTurn


def oregon_flow() -> list[SpeechTurn]:
    """Return Oregon debate sequence with statement, cross-exam, and closing stages."""
    return [
        SpeechTurn(step=1, label="正一申論", speaker="P1", side=Side.PRO, mode="speech"),
        SpeechTurn(step=1, label="反二質詢", speaker="C2", side=Side.CON, mode="cross_exam"),
        SpeechTurn(step=2, label="反一申論", speaker="C1", side=Side.CON, mode="speech"),
        SpeechTurn(step=2, label="正三質詢", speaker="P3", side=Side.PRO, mode="cross_exam"),
        SpeechTurn(step=3, label="正二申論", speaker="P2", side=Side.PRO, mode="speech"),
        SpeechTurn(step=3, label="反三質詢", speaker="C3", side=Side.CON, mode="cross_exam"),
        SpeechTurn(step=4, label="反二申論", speaker="C2", side=Side.CON, mode="speech"),
        SpeechTurn(step=4, label="正一質詢", speaker="P1", side=Side.PRO, mode="cross_exam"),
        SpeechTurn(step=5, label="正三申論", speaker="P3", side=Side.PRO, mode="speech"),
        SpeechTurn(step=5, label="反一質詢", speaker="C1", side=Side.CON, mode="cross_exam"),
        SpeechTurn(step=6, label="反三申論", speaker="C3", side=Side.CON, mode="speech"),
        SpeechTurn(step=6, label="正二質詢", speaker="P2", side=Side.PRO, mode="cross_exam"),
        SpeechTurn(step=7, label="正方結辯", speaker="P3", side=Side.PRO, mode="closing"),
        SpeechTurn(step=7, label="反方結辯", speaker="C3", side=Side.CON, mode="closing"),
    ]


def role_prompt_name(speaker: str) -> str:
    role_map = {
        "P1": "pro_1",
        "P2": "pro_2",
        "P3": "pro_3",
        "C1": "con_1",
        "C2": "con_2",
        "C3": "con_3",
    }
    return role_map[speaker]
