from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Side(str, Enum):
    PRO = "pro"
    CON = "con"


class JudgeType(str, Enum):
    LOGIC = "logic"
    PERSUASION = "persuasion"
    STRUCTURE = "structure"


@dataclass(frozen=True)
class ArgumentUnit:
    id: str
    side: Side
    claim: str
    warrant: str
    impact: str
    attack_points: list[str]
    strength_score: float
    detail: str = ""


@dataclass(frozen=True)
class SelectedArguments:
    side: Side
    main_argument_ids: list[str]
    defense_argument_ids: list[str]
    reason: str


@dataclass(frozen=True)
class SelectionPlan:
    mode: str
    pro_required_ids: list[str] = field(default_factory=list)
    con_required_ids: list[str] = field(default_factory=list)
    pro_required_main_ids: list[str] = field(default_factory=list)
    con_required_main_ids: list[str] = field(default_factory=list)
    note: str = ""


@dataclass(frozen=True)
class SpeechTurn:
    step: int
    label: str
    speaker: str
    side: Side
    mode: str


@dataclass(frozen=True)
class Utterance:
    turn: SpeechTurn
    content: str
    used_argument_ids: list[str]


@dataclass(frozen=True)
class JudgeScore:
    judge_type: JudgeType
    speaker_scores: dict[str, int]
    team_scores: dict[str, int]
    final_total: dict[str, int]
    winner: Side
    key_clash: str
    turning_point: str
    best_argument: str
    worst_argument: str
    judge_reason: str
    best_speech: str = ""
    best_cross_exam: str = ""
    best_closing: str = ""


@dataclass(frozen=True)
class DebateResult:
    run_id: int
    topic: str
    selection_plan: SelectionPlan
    pro_selection: SelectedArguments
    con_selection: SelectedArguments
    utterances: list[Utterance]
    judge_scores: list[JudgeScore]
    final_winner: Side
    key_clash: str
    turning_point: str
    best_argument: str
    worst_argument: str


@dataclass
class SimulationReport:
    topic: str
    num_runs: int
    wins: dict[str, int] = field(default_factory=lambda: {"pro": 0, "con": 0})
    argument_usage: dict[str, int] = field(default_factory=dict)
    argument_wins: dict[str, int] = field(default_factory=dict)
    argument_losses: dict[str, int] = field(default_factory=dict)
    key_clashes: dict[str, int] = field(default_factory=dict)
    combo_wins: dict[str, int] = field(default_factory=dict)
    judge_comments: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        def ratio(win: int, total: int) -> float:
            return round((win / total) if total else 0.0, 4)

        argument_win_rate = {}
        beaten_rate = {}
        for arg_id, used in self.argument_usage.items():
            wins = self.argument_wins.get(arg_id, 0)
            losses = self.argument_losses.get(arg_id, 0)
            argument_win_rate[arg_id] = {
                "wins": wins,
                "uses": used,
                "win_rate": ratio(wins, used),
            }
            beaten_rate[arg_id] = {
                "losses": losses,
                "uses": used,
                "beaten_rate": ratio(losses, used),
            }

        def top_by_side(prefix: str, limit: int) -> list[dict[str, Any]]:
            records = []
            for arg_id, metrics in argument_win_rate.items():
                if arg_id.lower().startswith(prefix):
                    records.append({"argument_id": arg_id, **metrics})
            records.sort(key=lambda x: (x["win_rate"], x["wins"], x["uses"]), reverse=True)
            return records[:limit]

        combo_sorted = sorted(
            self.combo_wins.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:5]

        clash_sorted = sorted(
            self.key_clashes.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:5]

        return {
            "topic": self.topic,
            "num_runs": self.num_runs,
            "wins": self.wins,
            "win_rate": {
                "pro": ratio(self.wins["pro"], self.num_runs),
                "con": ratio(self.wins["con"], self.num_runs),
            },
            "argument_win_rate": argument_win_rate,
            "argument_beaten_rate": beaten_rate,
            "top3_pro_arguments": top_by_side("p", 3),
            "top3_con_arguments": top_by_side("c", 3),
            "top10_pro_arguments": top_by_side("p", 10),
            "top10_con_arguments": top_by_side("c", 10),
            "top_key_clashes": [{"clash": k, "count": v} for k, v in clash_sorted],
            "best_argument_combinations": [
                {"combo": combo, "wins": wins} for combo, wins in combo_sorted
            ],
            "judge_comments": self.judge_comments,
        }
