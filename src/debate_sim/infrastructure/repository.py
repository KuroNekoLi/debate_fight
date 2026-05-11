from __future__ import annotations

import json
from pathlib import Path

from debate_sim.domain.models import ArgumentUnit, Side


class ArgumentUnitRepository:
    def __init__(self, path: Path):
        self.path = path

    def load(self) -> list[ArgumentUnit]:
        data = json.loads(self.path.read_text())
        units: list[ArgumentUnit] = []
        for row in data:
            units.append(
                ArgumentUnit(
                    id=row["id"],
                    side=Side(row["side"]),
                    claim=row["claim"],
                    warrant=row["warrant"],
                    impact=row["impact"],
                    attack_points=row.get("attack_points", []),
                    strength_score=float(row.get("strength_score", 0.0)),
                    detail=row.get("detail", ""),
                )
            )
        return units
