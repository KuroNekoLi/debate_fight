from __future__ import annotations

import json
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
from dataclasses import asdict
from typing import Any

from debate_sim.config import AppConfig
from debate_sim.domain.models import (
    ArgumentUnit,
    DebateResult,
    JudgeScore,
    JudgeType,
    SelectionPlan,
    SelectedArguments,
    Side,
    SimulationReport,
    Utterance,
)
from debate_sim.domain.oregon import oregon_flow, role_prompt_name
from debate_sim.infrastructure.llm import LLMClient, LLMRequest
from debate_sim.infrastructure.repository import ArgumentUnitRepository
from debate_sim.prompts.templates import (
    ARGUMENT_SELECTION_SYSTEM,
    JUDGE_SYSTEM,
    ROLE_PROMPTS,
    SPEECH_SYSTEM,
)


class DebateSimulationEngine:
    def __init__(self, config: AppConfig, llm: LLMClient):
        self.config = config
        self.llm = llm
        self.repo = ArgumentUnitRepository(config.data_path)

    def run_simulation(self, topic: str, num_runs: int = 100) -> dict[str, Any]:
        units = self.repo.load()
        self.config.results_dir.mkdir(parents=True, exist_ok=True)

        report = SimulationReport(topic=topic, num_runs=num_runs)
        all_results: list[DebateResult] = []
        failed_runs: list[dict[str, Any]] = []
        selection_plans = self._build_selection_plans(units, num_runs)

        max_workers = max(1, self.config.max_workers)
        if max_workers == 1:
            for run_id in range(1, num_runs + 1):
                try:
                    result = self._run_single(
                        topic=topic,
                        run_id=run_id,
                        units=units,
                        selection_plan=selection_plans[run_id - 1],
                    )
                except Exception as exc:
                    failure = self._serialize_failure(run_id, selection_plans[run_id - 1], exc)
                    failed_runs.append(failure)
                    print(f"[run {run_id}/{num_runs}] failed={type(exc).__name__}: {exc}", flush=True)
                    self._write_partial_report(
                        report,
                        all_results,
                        failed_runs,
                        completed=len(all_results),
                        failed=len(failed_runs),
                        total=num_runs,
                    )
                    continue
                all_results.append(result)
                self._accumulate(report, result)
                self._write_partial_report(
                    report,
                    all_results,
                    failed_runs,
                    completed=len(all_results),
                    failed=len(failed_runs),
                    total=num_runs,
                )
                print(f"[run {run_id}/{num_runs}] winner={result.final_winner.value}", flush=True)
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(
                        self._run_single,
                        topic,
                        run_id,
                        units,
                        selection_plans[run_id - 1],
                    ): run_id
                    for run_id in range(1, num_runs + 1)
                }
                for future in as_completed(futures):
                    run_id = futures[future]
                    try:
                        result = future.result()
                    except Exception as exc:
                        failure = self._serialize_failure(run_id, selection_plans[run_id - 1], exc)
                        failed_runs.append(failure)
                        print(
                            f"[run {run_id}/{num_runs}] completed={len(all_results)}/{num_runs} "
                            f"failed={len(failed_runs)} error={type(exc).__name__}: {exc}",
                            flush=True,
                        )
                        self._write_partial_report(
                            report,
                            all_results,
                            failed_runs,
                            completed=len(all_results),
                            failed=len(failed_runs),
                            total=num_runs,
                        )
                        continue
                    all_results.append(result)
                    self._accumulate(report, result)
                    self._write_partial_report(
                        report,
                        all_results,
                        failed_runs,
                        completed=len(all_results),
                        failed=len(failed_runs),
                        total=num_runs,
                    )
                    print(
                        f"[run {run_id}/{num_runs}] completed={len(all_results)}/{num_runs} "
                        f"failed={len(failed_runs)} winner={result.final_winner.value}",
                        flush=True,
                    )

        final_report = report.to_dict()
        final_report["sample_judge_comments"] = report.judge_comments[: min(20, len(report.judge_comments))]
        final_report["completed_runs"] = len(all_results)
        final_report["failed_runs_count"] = len(failed_runs)
        final_report["failed_runs"] = failed_runs
        completed_denominator = max(1, len(all_results))
        final_report["completed_win_rate"] = {
            "pro": round(report.wins["pro"] / completed_denominator, 4),
            "con": round(report.wins["con"] / completed_denominator, 4),
        }
        final_report["debate_runs"] = [
            self._serialize_result(r) for r in sorted(all_results, key=lambda item: item.run_id)
        ]
        units_by_id = {u.id: u for u in units}
        final_report["best_performances"] = self._best_performances(all_results)
        final_report["selection_mode_summary"] = self._selection_mode_summary(all_results)
        final_report["detailed_stats"] = self._build_detailed_stats(all_results, failed_runs)
        final_report["analysis"] = self._build_analysis_summary(final_report, units_by_id)

        output_file = self.config.results_dir / "simulation_report.json"
        output_file.write_text(json.dumps(final_report, ensure_ascii=False, indent=2))
        self._write_markdown_report(final_report, self.config.results_dir / "simulation_report.md")
        return final_report

    def _run_single(self, topic: str, run_id: int, units, selection_plan: SelectionPlan):
        self._progress(run_id, f"start mode={selection_plan.mode}")
        self._units_by_id = {u.id: u for u in units}
        pro_units = [u for u in units if u.side == Side.PRO]
        con_units = [u for u in units if u.side == Side.CON]

        self._progress(run_id, "select pro arguments")
        pro_selection = self._select_arguments(
            topic,
            Side.PRO,
            pro_units,
            con_units,
            selection_mode=selection_plan.mode,
            required_ids=selection_plan.pro_required_ids,
            required_main_ids=selection_plan.pro_required_main_ids,
        )
        self._progress(run_id, "select con arguments")
        con_selection = self._select_arguments(
            topic,
            Side.CON,
            con_units,
            pro_units,
            selection_mode=selection_plan.mode,
            required_ids=selection_plan.con_required_ids,
            required_main_ids=selection_plan.con_required_main_ids,
        )

        allowed = {
            Side.PRO: set(pro_selection.main_argument_ids + pro_selection.defense_argument_ids),
            Side.CON: set(con_selection.main_argument_ids + con_selection.defense_argument_ids),
        }

        utterances: list[Utterance] = []
        for turn in oregon_flow():
            self._progress(run_id, f"generate {turn.label}")
            u = self._generate_turn(
                topic,
                turn,
                pro_selection,
                con_selection,
                allowed[turn.side],
                self._units_by_id,
            )
            utterances.append(u)

        self._progress(run_id, "judge debate")
        judge_scores = self._evaluate_debate(topic, utterances, pro_selection, con_selection)
        final_winner = self._majority_winner(judge_scores)

        key_clash = self._majority_field(judge_scores, "key_clash")
        turning_point = self._majority_field(judge_scores, "turning_point")
        best_argument = self._majority_field(judge_scores, "best_argument")
        worst_argument = self._majority_field(judge_scores, "worst_argument")

        return DebateResult(
            run_id=run_id,
            topic=topic,
            selection_plan=selection_plan,
            pro_selection=pro_selection,
            con_selection=con_selection,
            utterances=utterances,
            judge_scores=judge_scores,
            final_winner=final_winner,
            key_clash=key_clash,
            turning_point=turning_point,
            best_argument=best_argument,
            worst_argument=worst_argument,
        )

    @staticmethod
    def _progress(run_id: int, stage: str) -> None:
        print(f"[run {run_id}] {stage}", flush=True)

    def _select_arguments(
        self,
        topic,
        side,
        own_units,
        opp_units,
        selection_mode: str = "free",
        required_ids: list[str] | None = None,
        required_main_ids: list[str] | None = None,
    ) -> SelectedArguments:
        required_ids = required_ids or []
        required_main_ids = required_main_ids or []
        candidate_summary = [
            {
                "id": u.id,
                "claim": u.claim,
                "warrant": u.warrant,
                "impact": u.impact,
                "strength_score": u.strength_score,
                "attack_points": u.attack_points,
            }
            for u in own_units
        ]
        user_prompt = (
            "task=argument_selection\n"
            f"topic={topic}\n"
            f"side={side.value}\n"
            f"selection_mode={selection_mode}\n"
            f"required_argument_ids={json.dumps(required_ids, ensure_ascii=False)}\n"
            f"required_main_argument_ids={json.dumps(required_main_ids, ensure_ascii=False)}\n"
            f"candidates={json.dumps(candidate_summary, ensure_ascii=False)}\n"
            f"opponent_top={json.dumps([{ 'id':u.id, 'claim':u.claim, 'warrant':u.warrant, 'impact':u.impact} for u in opp_units[:8]], ensure_ascii=False)}\n"
            "若 required_argument_ids 不為空，必須放入 main 或 defense；若 required_main_argument_ids 不為空，必須放入 main。"
            "請輸出包含 main_argument_ids / defense_argument_ids / reason 的 JSON。"
        )
        data = self.llm.generate_json(
            LLMRequest(
                system_prompt=ARGUMENT_SELECTION_SYSTEM,
                user_prompt=user_prompt,
                model=self.config.model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
        )
        own_ids = {u.id for u in own_units}
        main_ids = self._valid_unique_ids(data.get("main_argument_ids", []), own_ids, limit=3)
        defense_ids = self._valid_unique_ids(
            data.get("defense_argument_ids", []),
            own_ids - set(main_ids),
            limit=2,
        )
        if len(main_ids) < 3 or len(defense_ids) < 2:
            fallback = [u.id for u in sorted(own_units, key=lambda x: x.strength_score, reverse=True)]
            for arg_id in fallback:
                if len(main_ids) < 3 and arg_id not in main_ids and arg_id not in defense_ids:
                    main_ids.append(arg_id)
                elif len(defense_ids) < 2 and arg_id not in main_ids and arg_id not in defense_ids:
                    defense_ids.append(arg_id)
                if len(main_ids) == 3 and len(defense_ids) == 2:
                    break
        defense_ids = [arg_id for arg_id in defense_ids if arg_id not in main_ids]
        if len(defense_ids) < 2:
            for unit in sorted(own_units, key=lambda x: x.strength_score, reverse=True):
                if unit.id not in main_ids and unit.id not in defense_ids:
                    defense_ids.append(unit.id)
                if len(defense_ids) == 2:
                    break
        main_ids, defense_ids = self._apply_selection_constraints(
            main_ids=main_ids,
            defense_ids=defense_ids,
            own_units=own_units,
            required_ids=required_ids,
            required_main_ids=required_main_ids,
        )

        return SelectedArguments(
            side=side,
            main_argument_ids=main_ids,
            defense_argument_ids=defense_ids,
            reason=str(data.get("reason", "")),
        )

    @staticmethod
    def _build_selection_plans(units: list[ArgumentUnit], num_runs: int) -> list[SelectionPlan]:
        pro_ids = sorted(u.id for u in units if u.side == Side.PRO)
        con_ids = sorted(u.id for u in units if u.side == Side.CON)
        free_count = round(num_runs * 0.3)
        coverage_count = round(num_runs * 0.5)
        stress_count = max(0, num_runs - free_count - coverage_count)

        plans: list[SelectionPlan] = []
        for _ in range(free_count):
            plans.append(SelectionPlan(mode="free", note="AI 自由選 3 主軸 + 2 防守。"))

        for idx in range(coverage_count):
            pro_required = [pro_ids[idx % len(pro_ids)]] if pro_ids else []
            con_required = [con_ids[idx % len(con_ids)]] if con_ids else []
            plans.append(
                SelectionPlan(
                    mode="coverage",
                    pro_required_ids=pro_required,
                    con_required_ids=con_required,
                    note="覆蓋探索：每場強制正反各一個論點進入五論點組合，其餘由 AI 補齊。",
                )
            )

        pro_combos = [
            ["P_U51", "P_U05", "P_U11"],
            ["P_U01", "P_U23", "P_U35"],
            ["P_U19", "P_U05", "P_U51"],
            ["P_U37", "P_U41", "P_U49"],
            ["P_U11", "P_U17", "P_U51"],
        ]
        con_combos = [
            ["C_U02", "C_U04", "C_U40"],
            ["C_U02", "C_U08", "C_U40"],
            ["C_U04", "C_U22", "C_U24"],
            ["C_U32", "C_U38", "C_U40"],
            ["C_U34", "C_U52", "C_U54"],
        ]
        pro_set = set(pro_ids)
        con_set = set(con_ids)
        for idx in range(stress_count):
            pro_required_main = [arg_id for arg_id in pro_combos[idx % len(pro_combos)] if arg_id in pro_set]
            con_required_main = [arg_id for arg_id in con_combos[idx % len(con_combos)] if arg_id in con_set]
            plans.append(
                SelectionPlan(
                    mode="stress_combo",
                    pro_required_main_ids=pro_required_main[:3],
                    con_required_main_ids=con_required_main[:3],
                    note="組合壓測：指定 2-3 個主軸測化學反應，其餘由 AI 補齊。",
                )
            )

        return plans[:num_runs]

    def _generate_turn(self, topic, turn, pro_selection, con_selection, allowed_ids, units_by_id=None) -> Utterance:
        units_by_id = units_by_id or getattr(self, "_units_by_id", {})
        own_selection = pro_selection if turn.side == Side.PRO else con_selection
        opp_selection = con_selection if turn.side == Side.PRO else pro_selection

        own_units = self._selected_unit_payload(own_selection, units_by_id)
        opp_units = self._selected_unit_payload(opp_selection, units_by_id)
        role_prompt = ROLE_PROMPTS[role_prompt_name(turn.speaker)]
        answerer = self._answerer_for_cross_exam(turn.label)
        answerer_selection = None
        if answerer:
            answerer_side = Side.PRO if answerer.startswith("P") else Side.CON
            answerer_selection = pro_selection if answerer_side == Side.PRO else con_selection
        answerer_units = self._selected_unit_payload(answerer_selection, units_by_id) if answerer_selection else []
        user_prompt = (
            "task=speech\n"
            f"topic={topic}\n"
            f"turn={turn.label} ({turn.mode})\n"
            f"mode={turn.mode}\n"
            f"role={role_prompt}\n"
            f"speaker={turn.speaker}\n"
            f"questioner={turn.speaker if turn.mode == 'cross_exam' else ''}\n"
            f"answerer={answerer or ''}\n"
            f"own_arguments={json.dumps(own_units, ensure_ascii=False)}\n"
            f"opponent_arguments={json.dumps(opp_units, ensure_ascii=False)}\n"
            f"answerer_arguments={json.dumps(answerer_units, ensure_ascii=False)}\n"
            f"allowed_argument_ids={sorted(allowed_ids)}\n"
            "請輸出 JSON：content 與 used_argument_ids。"
        )
        if turn.mode == "cross_exam":
            user_prompt += (
                "\n\n這一回合是質詢，不是申論。content 必須第一行就以「質詢者：」開始，"
                "且必須包含至少三行「答辯者：」。不可寫成演講稿。"
            )

        data = self.llm.generate_json(
            LLMRequest(
                system_prompt=SPEECH_SYSTEM,
                user_prompt=user_prompt,
                model=self.config.model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
        )

        used_ids = [arg_id for arg_id in data.get("used_argument_ids", []) if arg_id in allowed_ids]
        if not used_ids:
            fallback = list(allowed_ids)[:1]
            used_ids = fallback

        content = str(data.get("content", "")).strip()
        if turn.mode == "cross_exam" and not self._looks_like_cross_exam_dialogue(content):
            content = self._fallback_cross_exam_dialogue(turn.speaker, answerer or "答辯者", own_units, opp_units)
        content = self._enforce_pro_existing_admin_network(topic, turn, content)

        return Utterance(
            turn=turn,
            content=content,
            used_argument_ids=used_ids,
        )

    def _evaluate_debate(self, topic, utterances, pro_selection, con_selection) -> list[JudgeScore]:
        transcript = [
            {
                "turn": u.turn.label,
                "speaker": u.turn.speaker,
                "side": u.turn.side.value,
                "mode": u.turn.mode,
                "content": self._compact_text(u.content, limit=700),
                "used_argument_ids": u.used_argument_ids,
            }
            for u in utterances
        ]

        judge_scores: list[JudgeScore] = []
        for jt in (JudgeType.LOGIC, JudgeType.PERSUASION, JudgeType.STRUCTURE):
            user_prompt = (
                "task=judge\n"
                f"topic={topic}\n"
                f"judge_type={jt.value}\n"
                f"pro_selection={asdict(pro_selection)}\n"
                f"con_selection={asdict(con_selection)}\n"
                f"transcript={json.dumps(transcript, ensure_ascii=False)}\n"
                "請依評分框架輸出 JSON。"
            )
            data = self.llm.generate_json(
                LLMRequest(
                    system_prompt=JUDGE_SYSTEM,
                    user_prompt=user_prompt,
                    model=self.config.model,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                )
            )

            final_total = {k: int(v) for k, v in data.get("final_total", {}).items()}
            winner = self._parse_winner(data.get("winner"), final_total)
            judge_scores.append(
                JudgeScore(
                    judge_type=jt,
                    speaker_scores={k: int(v) for k, v in data.get("speaker_scores", {}).items()},
                    team_scores={k: int(v) for k, v in data.get("team_scores", {}).items()},
                    final_total=final_total,
                    winner=winner,
                    key_clash=str(data.get("key_clash", "")),
                    turning_point=str(data.get("turning_point", "")),
                    best_argument=str(data.get("best_argument", "")),
                    worst_argument=str(data.get("worst_argument", "")),
                    best_speech=str(data.get("best_speech", "")),
                    best_cross_exam=str(data.get("best_cross_exam", "")),
                    best_closing=str(data.get("best_closing", "")),
                    judge_reason=str(data.get("judge_reason", "")),
                )
            )

        return judge_scores

    @staticmethod
    def _compact_text(text: str, limit: int) -> str:
        clean = " ".join(str(text).split())
        if len(clean) <= limit:
            return clean
        return clean[: limit - 1] + "…"

    @staticmethod
    def _parse_winner(raw_winner: Any, final_total: dict[str, int]) -> Side:
        winner = str(raw_winner or "").strip().lower()
        if winner in {"pro", "正方"}:
            return Side.PRO
        if winner in {"con", "反方"}:
            return Side.CON
        pro_score = final_total.get("pro", 0)
        con_score = final_total.get("con", 0)
        return Side.PRO if pro_score >= con_score else Side.CON

    @staticmethod
    def _majority_winner(scores: list[JudgeScore]) -> Side:
        pro_votes = sum(1 for s in scores if s.winner == Side.PRO)
        return Side.PRO if pro_votes >= 2 else Side.CON

    @staticmethod
    def _majority_field(scores: list[JudgeScore], field_name: str) -> str:
        counter: dict[str, int] = {}
        for s in scores:
            key = getattr(s, field_name)
            counter[key] = counter.get(key, 0) + 1
        return max(counter.items(), key=lambda x: x[1])[0]

    def _accumulate(self, report: SimulationReport, result: DebateResult) -> None:
        winner = result.final_winner.value
        loser = Side.CON.value if winner == Side.PRO.value else Side.PRO.value
        report.wins[winner] += 1

        pro_ids = result.pro_selection.main_argument_ids + result.pro_selection.defense_argument_ids
        con_ids = result.con_selection.main_argument_ids + result.con_selection.defense_argument_ids

        for arg_id in pro_ids + con_ids:
            report.argument_usage[arg_id] = report.argument_usage.get(arg_id, 0) + 1

        winner_ids = pro_ids if winner == Side.PRO.value else con_ids
        loser_ids = con_ids if winner == Side.PRO.value else pro_ids

        for arg_id in winner_ids:
            report.argument_wins[arg_id] = report.argument_wins.get(arg_id, 0) + 1

        for arg_id in loser_ids:
            report.argument_losses[arg_id] = report.argument_losses.get(arg_id, 0) + 1

        report.key_clashes[result.key_clash] = report.key_clashes.get(result.key_clash, 0) + 1

        combo = (
            f"{winner}:"
            + "+".join(
                sorted(
                    (pro_ids if winner == Side.PRO.value else con_ids),
                    key=lambda x: x,
                )
            )
        )
        report.combo_wins[combo] = report.combo_wins.get(combo, 0) + 1

        for j in result.judge_scores:
            report.judge_comments.append(
                {
                    "run_id": result.run_id,
                    "judge": j.judge_type.value,
                    "winner": j.winner.value,
                    "key_clash": j.key_clash,
                    "turning_point": j.turning_point,
                    "best_argument": j.best_argument,
                    "worst_argument": j.worst_argument,
                    "best_speech": j.best_speech,
                    "best_cross_exam": j.best_cross_exam,
                    "best_closing": j.best_closing,
                    "judge_reason": j.judge_reason,
                }
            )

    @staticmethod
    def _serialize_result(result: DebateResult) -> dict[str, Any]:
        return {
            "run_id": result.run_id,
            "topic": result.topic,
            "selection_plan": asdict(result.selection_plan),
            "pro_selection": asdict(result.pro_selection),
            "con_selection": asdict(result.con_selection),
            "final_winner": result.final_winner.value,
            "key_clash": result.key_clash,
            "turning_point": result.turning_point,
            "best_argument": result.best_argument,
            "worst_argument": result.worst_argument,
            "judge_scores": [
                {
                    "judge_type": j.judge_type.value,
                    "scores": {
                        "speaker_scores": j.speaker_scores,
                        "team_scores": j.team_scores,
                        "final_total": j.final_total,
                    },
                    "winner": j.winner.value,
                    "speaker_scores": j.speaker_scores,
                    "team_scores": j.team_scores,
                    "final_total": j.final_total,
                    "key_clash": j.key_clash,
                    "turning_point": j.turning_point,
                    "best_argument": j.best_argument,
                    "worst_argument": j.worst_argument,
                    "best_speech": j.best_speech,
                    "best_cross_exam": j.best_cross_exam,
                    "best_closing": j.best_closing,
                    "judge_reason": j.judge_reason,
                }
                for j in result.judge_scores
            ],
            "transcript": [
                {
                    "turn": u.turn.label,
                    "speaker": u.turn.speaker,
                    "side": u.turn.side.value,
                    "mode": u.turn.mode,
                    "content": u.content,
                    "used_argument_ids": u.used_argument_ids,
                }
                for u in result.utterances
            ],
        }

    @staticmethod
    def _selection_mode_summary(results: list[DebateResult]) -> dict[str, dict[str, Any]]:
        summary: dict[str, dict[str, Any]] = {}
        for result in results:
            mode = result.selection_plan.mode
            bucket = summary.setdefault(
                mode,
                {
                    "runs": 0,
                    "wins": {"pro": 0, "con": 0},
                    "pro_required_usage": Counter(),
                    "con_required_usage": Counter(),
                },
            )
            bucket["runs"] += 1
            bucket["wins"][result.final_winner.value] += 1
            bucket["pro_required_usage"].update(result.selection_plan.pro_required_ids)
            bucket["pro_required_usage"].update(result.selection_plan.pro_required_main_ids)
            bucket["con_required_usage"].update(result.selection_plan.con_required_ids)
            bucket["con_required_usage"].update(result.selection_plan.con_required_main_ids)

        for bucket in summary.values():
            runs = max(1, bucket["runs"])
            bucket["win_rate"] = {
                "pro": round(bucket["wins"]["pro"] / runs, 4),
                "con": round(bucket["wins"]["con"] / runs, 4),
            }
            bucket["pro_required_usage"] = dict(bucket["pro_required_usage"])
            bucket["con_required_usage"] = dict(bucket["con_required_usage"])
        return summary

    @staticmethod
    def _build_detailed_stats(
        results: list[DebateResult],
        failed_runs: list[dict[str, Any]],
    ) -> dict[str, Any]:
        def ratio(count: int, total: int) -> float:
            return round((count / total) if total else 0.0, 4)

        mode_argument_stats: dict[str, dict[str, dict[str, int]]] = {}
        mode_combo_stats: dict[str, dict[str, dict[str, int]]] = {}
        required_argument_results: list[dict[str, Any]] = []
        run_index: list[dict[str, Any]] = []
        judge_vote_patterns: Counter[str] = Counter()
        judge_type_votes: dict[str, Counter[str]] = {
            JudgeType.LOGIC.value: Counter(),
            JudgeType.PERSUASION.value: Counter(),
            JudgeType.STRUCTURE.value: Counter(),
        }

        for result in sorted(results, key=lambda item: item.run_id):
            mode = result.selection_plan.mode
            pro_ids = result.pro_selection.main_argument_ids + result.pro_selection.defense_argument_ids
            con_ids = result.con_selection.main_argument_ids + result.con_selection.defense_argument_ids
            winner = result.final_winner.value
            mode_argument_stats.setdefault(mode, {})
            mode_combo_stats.setdefault(mode, {})

            for side, ids in ((Side.PRO.value, pro_ids), (Side.CON.value, con_ids)):
                side_won = winner == side
                for arg_id in ids:
                    stats = mode_argument_stats[mode].setdefault(
                        arg_id,
                        {"uses": 0, "wins": 0, "losses": 0},
                    )
                    stats["uses"] += 1
                    stats["wins" if side_won else "losses"] += 1

                combo_key = f"{side}:" + "+".join(sorted(ids))
                combo_stats = mode_combo_stats[mode].setdefault(
                    combo_key,
                    {"uses": 0, "wins": 0, "losses": 0},
                )
                combo_stats["uses"] += 1
                combo_stats["wins" if side_won else "losses"] += 1

            pro_votes = sum(1 for judge in result.judge_scores if judge.winner == Side.PRO)
            con_votes = len(result.judge_scores) - pro_votes
            vote_pattern = f"pro{pro_votes}-con{con_votes}"
            judge_vote_patterns[vote_pattern] += 1
            for judge in result.judge_scores:
                judge_type_votes[judge.judge_type.value][judge.winner.value] += 1

            run_index.append(
                {
                    "run_id": result.run_id,
                    "mode": mode,
                    "winner": winner,
                    "judge_votes": {"pro": pro_votes, "con": con_votes},
                    "pro_main": result.pro_selection.main_argument_ids,
                    "pro_defense": result.pro_selection.defense_argument_ids,
                    "con_main": result.con_selection.main_argument_ids,
                    "con_defense": result.con_selection.defense_argument_ids,
                    "key_clash": result.key_clash,
                    "turning_point": result.turning_point,
                    "best_argument": result.best_argument,
                    "worst_argument": result.worst_argument,
                }
            )

            required_specs = [
                (Side.PRO.value, result.selection_plan.pro_required_ids, result.selection_plan.pro_required_main_ids, pro_ids, result.pro_selection.main_argument_ids),
                (Side.CON.value, result.selection_plan.con_required_ids, result.selection_plan.con_required_main_ids, con_ids, result.con_selection.main_argument_ids),
            ]
            for side, required_any, required_main, selected_ids, selected_main in required_specs:
                for arg_id in required_any + required_main:
                    if not arg_id:
                        continue
                    required_argument_results.append(
                        {
                            "run_id": result.run_id,
                            "mode": mode,
                            "side": side,
                            "argument_id": arg_id,
                            "required_as": "main" if arg_id in required_main else "any",
                            "selected": arg_id in selected_ids,
                            "selected_as": "main" if arg_id in selected_main else "defense" if arg_id in selected_ids else "missing",
                            "winner": winner,
                            "won": winner == side,
                        }
                    )

        for stats_by_arg in mode_argument_stats.values():
            for stats in stats_by_arg.values():
                stats["win_rate"] = ratio(stats["wins"], stats["uses"])
                stats["loss_rate"] = ratio(stats["losses"], stats["uses"])

        for combos in mode_combo_stats.values():
            for stats in combos.values():
                stats["win_rate"] = ratio(stats["wins"], stats["uses"])

        required_summary: dict[str, dict[str, Any]] = {}
        for item in required_argument_results:
            key = f"{item['side']}:{item['argument_id']}"
            bucket = required_summary.setdefault(
                key,
                {
                    "side": item["side"],
                    "argument_id": item["argument_id"],
                    "required_uses": 0,
                    "selected": 0,
                    "wins": 0,
                    "missing": 0,
                    "modes": Counter(),
                },
            )
            bucket["required_uses"] += 1
            bucket["selected"] += 1 if item["selected"] else 0
            bucket["wins"] += 1 if item["won"] else 0
            bucket["missing"] += 0 if item["selected"] else 1
            bucket["modes"].update([item["mode"]])
        for bucket in required_summary.values():
            bucket["win_rate"] = ratio(bucket["wins"], bucket["required_uses"])
            bucket["selected_rate"] = ratio(bucket["selected"], bucket["required_uses"])
            bucket["modes"] = dict(bucket["modes"])

        failed_by_type = Counter(item.get("error_type", "UnknownError") for item in failed_runs)

        return {
            "mode_argument_stats": mode_argument_stats,
            "mode_combo_stats": mode_combo_stats,
            "required_argument_results": required_argument_results,
            "required_argument_summary": sorted(
                required_summary.values(),
                key=lambda item: (item["side"], item["argument_id"]),
            ),
            "judge_vote_patterns": dict(judge_vote_patterns),
            "judge_type_votes": {
                judge_type: dict(counter) for judge_type, counter in judge_type_votes.items()
            },
            "run_index": run_index,
            "failed_by_type": dict(failed_by_type),
        }

    @staticmethod
    def _serialize_failure(run_id: int, selection_plan: SelectionPlan, exc: Exception) -> dict[str, Any]:
        return {
            "run_id": run_id,
            "selection_plan": asdict(selection_plan),
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "traceback": traceback.format_exc(limit=8),
        }

    def _write_partial_report(
        self,
        report: SimulationReport,
        all_results: list[DebateResult],
        failed_runs: list[dict[str, Any]],
        completed: int,
        failed: int,
        total: int,
    ) -> None:
        partial = report.to_dict()
        partial["completed_runs"] = completed
        partial["failed_runs_count"] = failed
        partial["failed_runs"] = failed_runs
        partial["total_requested_runs"] = total
        partial["selection_mode_summary"] = self._selection_mode_summary(all_results)
        partial["detailed_stats"] = self._build_detailed_stats(all_results, failed_runs)
        partial["debate_runs"] = [
            self._serialize_result(r) for r in sorted(all_results, key=lambda item: item.run_id)
        ]
        output_file = self.config.results_dir / "simulation_report.partial.json"
        output_file.write_text(json.dumps(partial, ensure_ascii=False, indent=2))

    def _best_performances(self, results: list[DebateResult]) -> dict[str, dict[str, list[dict[str, Any]]]]:
        buckets: dict[str, dict[str, list[dict[str, Any]]]] = {
            "pro": {"speech": [], "cross_exam": [], "closing": []},
            "con": {"speech": [], "cross_exam": [], "closing": []},
        }
        for result in results:
            avg_speaker_scores = self._average_speaker_scores(result.judge_scores)
            for utterance in result.utterances:
                side = utterance.turn.side.value
                mode = utterance.turn.mode
                category = "closing" if mode == "closing" else "cross_exam" if mode == "cross_exam" else "speech"
                score = avg_speaker_scores.get(utterance.turn.speaker, 0.0)
                if result.final_winner == utterance.turn.side:
                    score += 5.0
                if utterance.used_argument_ids and utterance.used_argument_ids[0] == result.best_argument:
                    score += 3.0
                buckets[side][category].append(
                    {
                        "run_id": result.run_id,
                        "turn": utterance.turn.label,
                        "speaker": utterance.turn.speaker,
                        "side": side,
                        "score_estimate": round(score, 2),
                        "used_argument_ids": utterance.used_argument_ids,
                        "content": utterance.content,
                    }
                )
        for side_buckets in buckets.values():
            for category, items in side_buckets.items():
                items.sort(key=lambda item: (item["score_estimate"], -item["run_id"]), reverse=True)
                side_buckets[category] = items[:5]
        return buckets

    @staticmethod
    def _average_speaker_scores(judge_scores: list[JudgeScore]) -> dict[str, float]:
        totals: dict[str, list[int]] = {}
        for judge in judge_scores:
            for speaker, score in judge.speaker_scores.items():
                totals.setdefault(speaker, []).append(score)
        return {
            speaker: sum(scores) / len(scores)
            for speaker, scores in totals.items()
            if scores
        }

    def _build_analysis_summary(
        self,
        final_report: dict[str, Any],
        units_by_id: dict[str, ArgumentUnit],
    ) -> dict[str, Any]:
        wins = final_report["wins"]
        total = max(1, final_report.get("completed_runs", final_report["num_runs"]))
        pro_rate = round(wins["pro"] / total, 4)
        con_rate = round(wins["con"] / total, 4)
        con_gap = con_rate - pro_rate
        if abs(con_gap) >= 0.3:
            balance_note = (
                "勝率差距很大，這不應直接解讀成辯題本身已經定案；更可能代表目前論點池、裁判 prompt、"
                "模型既有價值偏好與選點策略共同造成一方更容易被判勝。"
            )
        elif abs(con_gap) >= 0.15:
            balance_note = "勝率有明顯傾斜，建議檢查弱勢方論點是否過度重疊、裁判是否給予另一方較高的預設可信度。"
        else:
            balance_note = "勝率接近，表示目前 prompt 與論點池沒有呈現極端單邊傾斜。"

        repeated_clashes = final_report.get("top_key_clashes", [])[:5]
        mode_summary = final_report.get("selection_mode_summary", {})
        mode_notes = []
        for mode, item in mode_summary.items():
            mode_notes.append(
                f"{mode}: {item['runs']} 場，正方勝率 {item['win_rate']['pro']:.2%}，反方勝率 {item['win_rate']['con']:.2%}。"
            )
        return {
            "balance_note": balance_note,
            "pro_win_rate": pro_rate,
            "con_win_rate": con_rate,
            "mode_notes": mode_notes,
            "failure_note": (
                f"本次有 {final_report.get('failed_runs_count', 0)} 場失敗；勝率與論點勝率只根據成功完成評分的場次計算。"
                if final_report.get("failed_runs_count", 0)
                else "本次沒有失敗場次。"
            ),
            "interpretation": [
                "Top 論點勝率需要同時看 uses；使用次數太低的 100% 勝率，不一定比高使用率且穩定勝出的論點更可靠。",
                "最佳組合代表在模擬中最常共同導向勝利的五個論點，不等於每個單點都最強。",
                "混合模式下，free 代表 AI 自然策略，coverage 代表單點覆蓋探索，stress_combo 代表指定組合壓測；三者應分開解讀，不宜只看總勝率。",
                "coverage 模式最適合看冷門論點和其他論點組合後是否有潛力；stress_combo 模式最適合看特定主軸是否產生化學反應。",
                "若某方長期大勝，應檢查裁判是否把價值底線、國家責任、可訴權利等語彙視為較高可信度。",
            ],
            "top_key_clashes_plain": [item["clash"] for item in repeated_clashes],
            "argument_labels": {
                arg_id: units_by_id[arg_id].claim
                for arg_id in units_by_id
                if arg_id in final_report.get("argument_win_rate", {})
            },
        }

    def _write_markdown_report(self, final_report: dict[str, Any], output_file) -> None:
        completed_win_rate = final_report.get("completed_win_rate", final_report.get("win_rate", {"pro": 0, "con": 0}))
        lines = [
            f"# 辯論模擬解讀報告",
            "",
            f"- 辯題：{final_report['topic']}",
            f"- 場次：{final_report['num_runs']}",
            f"- 完成：{final_report.get('completed_runs', final_report['num_runs'])}",
            f"- 失敗：{final_report.get('failed_runs_count', 0)}",
            f"- 勝場：正方 {final_report['wins']['pro']} / 反方 {final_report['wins']['con']}",
            f"- 完成場次勝率：正方 {completed_win_rate['pro']:.2%} / 反方 {completed_win_rate['con']:.2%}",
            "",
            "## 一、結果解讀",
            "",
            final_report.get("analysis", {}).get("balance_note", ""),
            "",
            final_report.get("analysis", {}).get("failure_note", ""),
            "",
            "## 二、選點模式摘要",
            "",
        ]
        for note in final_report.get("analysis", {}).get("mode_notes", []):
            lines.append(f"- {note}")
        if final_report.get("analysis", {}).get("interpretation"):
            lines.extend(["", "### 解讀提醒", ""])
            for item in final_report["analysis"]["interpretation"]:
                lines.append(f"- {item}")
        lines.extend([""])
        lines.extend(self._markdown_selection_mode_summary(final_report.get("selection_mode_summary", {})))
        lines.extend([
            "",
            "## 三、失敗場次摘要",
            "",
        ])
        lines.extend(self._markdown_failure_summary(final_report))
        lines.extend(["", "## 四、裁判票型", ""])
        lines.extend(self._markdown_judge_summary(final_report.get("detailed_stats", {})))
        lines.extend(["", "## 五、覆蓋探索論點表", ""])
        lines.extend(self._markdown_required_summary(final_report.get("detailed_stats", {})))
        lines.extend(["", "## 六、各模式論點表", ""])
        lines.extend(self._markdown_mode_argument_stats(final_report.get("detailed_stats", {})))
        lines.extend(["", "## 七、各模式組合表", ""])
        lines.extend(self._markdown_mode_combo_stats(final_report.get("detailed_stats", {})))
        lines.extend([
            "",
            "## 八、正方 Top 10 論點",
            "",
        ])
        lines.extend(self._markdown_argument_table(final_report.get("top10_pro_arguments", [])))
        lines.extend(["", "## 九、反方 Top 10 論點", ""])
        lines.extend(self._markdown_argument_table(final_report.get("top10_con_arguments", [])))
        lines.extend(["", "## 十、最佳論點組合 Top 5", ""])
        lines.extend([
            "| 排名 | 組合 | 勝場 |",
            "|---:|---|---:|",
        ])
        for idx, item in enumerate(final_report.get("best_argument_combinations", [])[:5], 1):
            lines.append(f"| {idx} | `{item['combo']}` | {item['wins']} |")
        lines.extend(["", "## 十一、關鍵衝突", ""])
        for idx, item in enumerate(final_report.get("top_key_clashes", [])[:5], 1):
            lines.append(f"{idx}. {item['clash']}（{item['count']} 次）")
        lines.extend(["", "## 十二、逐場索引", ""])
        lines.extend(self._markdown_run_index(final_report.get("detailed_stats", {})))
        lines.extend(["", "## 十三、最佳稿件摘錄", ""])
        lines.extend(self._markdown_best_performances(final_report.get("best_performances", {})))
        output_file.write_text("\n".join(lines), encoding="utf-8")

    @staticmethod
    def _markdown_selection_mode_summary(summary: dict[str, dict[str, Any]]) -> list[str]:
        if not summary:
            return ["尚無選點模式資料。"]
        lines = [
            "| 模式 | 場次 | 正方勝 | 反方勝 | 正方勝率 | 反方勝率 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for mode, item in summary.items():
            lines.append(
                f"| `{mode}` | {item['runs']} | {item['wins']['pro']} | {item['wins']['con']} | "
                f"{item['win_rate']['pro']:.2%} | {item['win_rate']['con']:.2%} |"
            )
        return lines

    @staticmethod
    def _markdown_failure_summary(final_report: dict[str, Any]) -> list[str]:
        failed = final_report.get("failed_runs", [])
        failed_by_type = final_report.get("detailed_stats", {}).get("failed_by_type", {})
        if not failed:
            return ["本次沒有失敗場次。"]
        lines = ["### 失敗類型", "", "| 錯誤類型 | 次數 |", "|---|---:|"]
        for error_type, count in sorted(failed_by_type.items(), key=lambda item: item[1], reverse=True):
            lines.append(f"| `{error_type}` | {count} |")
        lines.extend(["", "### 失敗場次", "", "| Run | 模式 | 錯誤類型 | 訊息 |", "|---:|---|---|---|"])
        for item in failed:
            message = str(item.get("error_message", "")).replace("|", "\\|")
            lines.append(
                f"| {item.get('run_id')} | `{item.get('selection_plan', {}).get('mode', '')}` | "
                f"`{item.get('error_type', '')}` | {message} |"
            )
        return lines

    @staticmethod
    def _markdown_judge_summary(stats: dict[str, Any]) -> list[str]:
        lines: list[str] = []
        patterns = stats.get("judge_vote_patterns", {})
        lines.extend(["### 票型分布", "", "| 票型 | 場次 |", "|---|---:|"])
        if patterns:
            for pattern, count in sorted(patterns.items(), key=lambda item: item[1], reverse=True):
                lines.append(f"| `{pattern}` | {count} |")
        else:
            lines.append("| 無 | 0 |")
        lines.extend(["", "### 裁判類型投票", "", "| 裁判 | 正方票 | 反方票 |", "|---|---:|---:|"])
        for judge_type, votes in stats.get("judge_type_votes", {}).items():
            lines.append(f"| `{judge_type}` | {votes.get('pro', 0)} | {votes.get('con', 0)} |")
        return lines

    @staticmethod
    def _markdown_required_summary(stats: dict[str, Any]) -> list[str]:
        items = stats.get("required_argument_summary", [])
        if not items:
            return ["本次沒有 coverage / stress_combo 指定論點資料。"]
        lines = [
            "| 立場 | 論點 | 指定次數 | 成功進場 | 進場率 | 勝場 | 勝率 | 模式 |",
            "|---|---|---:|---:|---:|---:|---:|---|",
        ]
        for item in items:
            modes = ", ".join(f"{mode}:{count}" for mode, count in item.get("modes", {}).items())
            lines.append(
                f"| `{item['side']}` | `{item['argument_id']}` | {item['required_uses']} | {item['selected']} | "
                f"{item['selected_rate']:.2%} | {item['wins']} | {item['win_rate']:.2%} | {modes} |"
            )
        return lines

    @staticmethod
    def _markdown_mode_argument_stats(stats: dict[str, Any]) -> list[str]:
        mode_stats = stats.get("mode_argument_stats", {})
        if not mode_stats:
            return ["尚無各模式論點統計。"]
        lines: list[str] = []
        for mode, arg_stats in mode_stats.items():
            lines.extend([f"### `{mode}`", "", "| 論點 | 使用 | 勝場 | 敗場 | 勝率 | 敗率 |", "|---|---:|---:|---:|---:|---:|"])
            sorted_items = sorted(
                arg_stats.items(),
                key=lambda item: (item[1].get("win_rate", 0), item[1].get("wins", 0), item[1].get("uses", 0)),
                reverse=True,
            )
            for arg_id, item in sorted_items:
                lines.append(
                    f"| `{arg_id}` | {item['uses']} | {item['wins']} | {item['losses']} | "
                    f"{item['win_rate']:.2%} | {item['loss_rate']:.2%} |"
                )
            lines.append("")
        return lines

    @staticmethod
    def _markdown_mode_combo_stats(stats: dict[str, Any]) -> list[str]:
        mode_stats = stats.get("mode_combo_stats", {})
        if not mode_stats:
            return ["尚無各模式組合統計。"]
        lines: list[str] = []
        for mode, combo_stats in mode_stats.items():
            lines.extend([f"### `{mode}`", "", "| 組合 | 使用 | 勝場 | 敗場 | 勝率 |", "|---|---:|---:|---:|---:|"])
            sorted_items = sorted(
                combo_stats.items(),
                key=lambda item: (item[1].get("win_rate", 0), item[1].get("wins", 0), item[1].get("uses", 0)),
                reverse=True,
            )
            for combo, item in sorted_items[:25]:
                lines.append(
                    f"| `{combo}` | {item['uses']} | {item['wins']} | {item['losses']} | {item['win_rate']:.2%} |"
                )
            lines.append("")
        return lines

    @staticmethod
    def _markdown_run_index(stats: dict[str, Any]) -> list[str]:
        runs = stats.get("run_index", [])
        if not runs:
            return ["尚無逐場資料。"]
        lines = [
            "| Run | 模式 | 勝方 | 票型 | 正方論點 | 反方論點 | Key Clash |",
            "|---:|---|---|---|---|---|---|",
        ]
        for run in runs:
            pro_ids = "+".join(run.get("pro_main", []) + run.get("pro_defense", []))
            con_ids = "+".join(run.get("con_main", []) + run.get("con_defense", []))
            votes = run.get("judge_votes", {})
            clash = str(run.get("key_clash", "")).replace("|", "\\|")
            lines.append(
                f"| {run.get('run_id')} | `{run.get('mode')}` | `{run.get('winner')}` | "
                f"{votes.get('pro', 0)}:{votes.get('con', 0)} | `{pro_ids}` | `{con_ids}` | {clash} |"
            )
        return lines

    @staticmethod
    def _markdown_argument_table(items: list[dict[str, Any]]) -> list[str]:
        lines = [
            "| 排名 | 論點 | 勝場 | 使用 | 勝率 |",
            "|---:|---|---:|---:|---:|",
        ]
        for idx, item in enumerate(items, 1):
            lines.append(
                f"| {idx} | `{item['argument_id']}` | {item['wins']} | {item['uses']} | {item['win_rate']:.2%} |"
            )
        return lines

    @staticmethod
    def _markdown_best_performances(best: dict[str, dict[str, list[dict[str, Any]]]]) -> list[str]:
        label_map = {
            "pro": "正方",
            "con": "反方",
            "speech": "最佳申論",
            "cross_exam": "最佳質詢答辯",
            "closing": "最佳結辯",
        }
        lines: list[str] = []
        for side in ("pro", "con"):
            lines.append(f"### {label_map[side]}")
            lines.append("")
            for category in ("speech", "cross_exam", "closing"):
                lines.append(f"#### {label_map[category]}")
                lines.append("")
                for idx, item in enumerate(best.get(side, {}).get(category, [])[:5], 1):
                    lines.append(
                        f"{idx}. Run {item['run_id']} / {item['turn']} / {item['speaker']} / "
                        f"估計分 {item['score_estimate']} / 論點 {', '.join(item['used_argument_ids'])}"
                    )
                    lines.append("")
                    lines.append("```text")
                    lines.append(item["content"].strip())
                    lines.append("```")
                    lines.append("")
        return lines

    @staticmethod
    def _answerer_for_cross_exam(label: str) -> str | None:
        mapping = {
            "反二質詢": "P1",
            "正三質詢": "C1",
            "反三質詢": "P2",
            "正一質詢": "C2",
            "反一質詢": "P3",
            "正二質詢": "C3",
        }
        return mapping.get(label)

    @staticmethod
    def _looks_like_cross_exam_dialogue(content: str) -> bool:
        return content.count("質詢者：") >= 2 and content.count("答辯者：") >= 2

    @staticmethod
    def _apply_selection_constraints(
        main_ids: list[str],
        defense_ids: list[str],
        own_units: list[ArgumentUnit],
        required_ids: list[str],
        required_main_ids: list[str],
    ) -> tuple[list[str], list[str]]:
        own_ids = {u.id for u in own_units}
        fallback = [u.id for u in sorted(own_units, key=lambda x: x.strength_score, reverse=True)]

        required_main = []
        for arg_id in required_main_ids:
            if arg_id in own_ids and arg_id not in required_main:
                required_main.append(arg_id)
        required_main = required_main[:3]

        required_any = []
        for arg_id in required_ids:
            if arg_id in own_ids and arg_id not in required_any and arg_id not in required_main:
                required_any.append(arg_id)

        main = []
        for arg_id in required_main + main_ids:
            if arg_id in own_ids and arg_id not in main:
                main.append(arg_id)
            if len(main) == 3:
                break

        defense = []
        for arg_id in defense_ids + required_any:
            if arg_id in own_ids and arg_id not in main and arg_id not in defense:
                defense.append(arg_id)
            if len(defense) == 2:
                break

        for arg_id in required_any:
            if arg_id in main or arg_id in defense:
                continue
            if len(defense) < 2:
                defense.append(arg_id)
            else:
                replace_idx = next(
                    (
                        idx
                        for idx, current in enumerate(defense)
                        if current not in required_any and current not in required_main
                    ),
                    len(defense) - 1,
                )
                defense[replace_idx] = arg_id

        for arg_id in fallback:
            if len(main) < 3 and arg_id not in main and arg_id not in defense:
                main.append(arg_id)
            if len(defense) < 2 and arg_id not in main and arg_id not in defense:
                defense.append(arg_id)
            if len(main) == 3 and len(defense) == 2:
                break

        return main[:3], defense[:2]

    @staticmethod
    def _fallback_cross_exam_dialogue(
        questioner: str,
        answerer: str,
        own_units: list[dict[str, Any]],
        opp_units: list[dict[str, Any]],
    ) -> str:
        target = opp_units[0] if opp_units else {}
        defense = own_units[0] if own_units else {}
        target_id = target.get("id", "對方核心論點")
        defense_id = defense.get("id", "本方核心論點")
        target_claim = target.get("claim", "你方主張")
        target_warrant = target.get("warrant", "你方因果鏈")
        defense_claim = defense.get("claim", "我方主張")
        return (
            f"質詢者：你方使用 {target_id}，主張「{target_claim}」。請問你方能不能先承認，"
            f"這個論點成立的關鍵是「{target_warrant}」必須穩定發生？\n"
            f"答辯者：我承認這是重要機制，但我方不是說它在每個個案都一樣發生，而是說它在制度層次上具有足夠風險。\n"
            f"質詢者：如果它只是風險，而不是必然結果，那評審為什麼不能選擇對就業入口傷害較小的制度？\n"
            f"答辯者：因為制度不能只看入口，也要看已經工作的人是否有最低保障；這正是我方 {target_id} 要保護的地方。\n"
            f"質詢者：但你方是否同意，如果制度把一部分最弱的人推到保障之外，這個保護就不是完整保護？\n"
            f"答辯者：我同意需要補強外部人問題，但我方會用 {defense_id} 回應：{defense_claim}\n"
            f"質詢者：所以本回合的分歧很清楚：你方能證明底線保障大於排除傷害，才守得住這個論點。"
        )

    @staticmethod
    def _enforce_pro_existing_admin_network(topic: str, turn: SpeechTurn, content: str) -> str:
        if turn.side != Side.PRO or "最低工資" not in topic:
            return content
        if (
            "現成行政網" in content
            and "現成薪資線" in content
            and "責任底線" in content
            and "被動自動" in content
            and "資料主動" in content
        ):
            return content

        if turn.mode == "cross_exam":
            addition = (
                "\n質詢者：最後確認一點，反方保留的是一條現成薪資線；正方接起的是勞保投保級距、薪資明細、"
                "所得稅資料、工時紀錄、地方勞工局申訴勞檢、就業服務站與職訓系統這張現成行政網。"
                "如果制度可以主動用資料找到低薪、低工時、未投保和假承攬的人，請問反方為什麼仍說只有最低工資才叫自動保障？\n"
                "答辯者：我方仍認為薪資線最簡單明確，但會承認正方提出的是現有行政資料的接軌，而不是完全空白的新制度。\n"
                "質詢者：所以本場其實不是有沒有法律底線，而是價格底線和責任底線的比較；反方是被動自動，正方是資料主動，對嗎？\n"
                "答辯者：這會是雙方真正的比較。"
            )
            return content.rstrip() + addition

        if turn.mode == "closing":
            addition = (
                "\n\n正方最後用法律底線的三大反轉收束本場。反方今天最強的是國家責任、低薪競逐、自動保障，"
                "但這三件事正方都做得更完整。第一，國家責任。反方把國家責任縮成一條時薪線；正方把國家責任落實成找得到人、"
                "補得到錢、查得到雇主、接得到入口。反方是國家宣示責任；正方是國家執行責任。第二，低薪競逐。"
                "反方只防價格向下競逐；正方防更危險的責任向下競逐。正方設三條硬底線：不得無薪資明細、不得無工時紀錄、"
                "不得無投保或假承攬逃責。這比一條時薪線更能防止雇主把弱勢推出正式制度。第三，自動保障。"
                "反方是被動自動，勞工要先站進正式雇傭，最低工資才適用；正方是資料主動，用勞保投保級距、薪資明細、"
                "所得稅資料、工時紀錄和地方勞工局資料主動找到低薪、低工時、未投保和假承攬的人。"
                "所以本場不是有沒有底線，而是價格底線對責任底線。反方保留一條現成薪資線；正方接起一張現成行政網。"
                "反方守一小時價格；正方守人不能從正式制度消失。若裁判真正重視弱勢保障，正方才是更完整的法律底線。"
            )
            return content.rstrip() + addition

        addition = (
            "\n\n這裡正方要把可行性講硬：我們不是拿一套未來想像取代最低工資，而是把現有行政系統重新接線。"
            "勞保投保級距、薪資明細、所得稅資料、工時紀錄，本來就能辨識誰是低薪、低工時、未投保或收入不穩；"
            "地方勞工局本來就有申訴、調解和勞檢系統；就業服務站與職訓系統本來就能接門外者。"
            "反方保留一條現成薪資線；正方接起一張現成行政網。裁判要比的不是哪個口號比較簡單，而是誰能用現有資料找到人、"
            "補到錢、查到雇主、接到入口。更重要的是，反方守的是價格底線，正方守的是責任底線：不得無薪資明細、"
            "不得無工時紀錄、不得無投保或假承攬逃責。反方是國家宣示責任，正方是國家執行責任；反方是被動自動，"
            "正方是資料主動。"
        )
        return content.rstrip() + addition

    @staticmethod
    def _valid_unique_ids(raw_ids: Any, allowed_ids: set[str], limit: int) -> list[str]:
        selected: list[str] = []
        if not isinstance(raw_ids, list):
            return selected
        for raw_id in raw_ids:
            arg_id = str(raw_id)
            if arg_id in allowed_ids and arg_id not in selected:
                selected.append(arg_id)
            if len(selected) == limit:
                break
        return selected

    @staticmethod
    def _selected_unit_payload(selection: SelectedArguments, units_by_id: dict[str, ArgumentUnit]) -> list[dict[str, Any]]:
        payload = []
        for arg_id in selection.main_argument_ids + selection.defense_argument_ids:
            unit = units_by_id.get(arg_id)
            if not unit:
                continue
            payload.append(
                {
                    "id": unit.id,
                    "side": unit.side.value,
                    "claim": unit.claim,
                    "warrant": unit.warrant,
                    "impact": unit.impact,
                    "attack_points": unit.attack_points,
                    "strength_score": unit.strength_score,
                    "detail": unit.detail,
                    "role": "main" if arg_id in selection.main_argument_ids else "defense",
                }
            )
        return payload
