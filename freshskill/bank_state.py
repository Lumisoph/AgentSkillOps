"""``BankState``: which skills are active vs retired, evolved across cycles.

The Maintenance layer (``freshskill.lifecycle``) produces per-cycle KEEP/RETIRE
decisions; ``BankState`` records the resulting bank contents.  This is the
"bank 装什么" half of RQ4, complementary to Freshness (which decides "bank
怎么用").

JSON schema (design doc section 5)::

    {
      "gen_001": {"status": "active",  "last_eval_step": 40, "pending": false},
      "gen_002": {"status": "retired", "retired_at": 40, "retired_utility": -0.029}
    }
"""

import json
import os

from .lifecycle import KEEP, RETIRE


class BankState:
    def __init__(self, skills=None):
        self.skills = skills or {}

    def set(self, skill_id, status="active", **fields):
        e = dict(fields)
        e["status"] = status
        self.skills[skill_id] = e

    def apply_actions(self, actions, current_step, timeline):
        """Apply a ``{skill_id: KEEP|RETIRE}`` decision taken at ``current_step``.

        RETIRE -> ``status=retired`` with ``retired_at`` / ``retired_utility``.
        KEEP   -> ``status=active`` with ``last_eval_step`` updated.
        """
        for sid, action in actions.items():
            u = timeline.utility(sid, current_step)
            if action == RETIRE:
                self.set(sid, status="retired", retired_at=current_step,
                         retired_utility=u)
            else:
                e = self.skills.setdefault(sid, {})
                e["status"] = "active"
                e["last_eval_step"] = current_step
                self.skills[sid] = e

    def record_pending(self, skill_ids, current_step, timeline, theta_retire):
        """Mark ``pending`` = (U_i(current_step) < theta_retire) for each skill."""
        for sid in skill_ids:
            e = self.skills.setdefault(sid, {"status": "active"})
            e["pending"] = timeline.utility(sid, current_step) < theta_retire
            self.skills[sid] = e

    def active_skill_ids(self):
        return sorted(sid for sid, e in self.skills.items()
                      if e.get("status", "active") != "retired")

    def retired_skill_ids(self):
        return sorted(sid for sid, e in self.skills.items()
                      if e.get("status", "active") == "retired")

    def status(self, skill_id):
        return self.skills.get(skill_id, {}).get("status", "active")

    # ---- persistence ----------------------------------------------------
    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.skills, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls(json.load(f))

    def __repr__(self):
        return (f"BankState({len(self.active_skill_ids())} active, "
                f"{len(self.retired_skill_ids())} retired)")
