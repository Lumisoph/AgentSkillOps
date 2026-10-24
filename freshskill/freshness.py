"""``FreshnessScorer``: time-decayed skill selection (Stage 5.5 / RQ4).

    Score_i = U_i * F_i,   F_i = e^{-lambda * A_i},   A_i = current_step - t_i

where ``t_i`` is the step at which skill i's utility was last measured.  This is
a *confidence prior*, not a sign-flip corrector: ``F_i`` in (0, 1] shrinks a
stale estimate toward 0 but never changes its sign.  Its observable effect is
magnitude demotion — it only changes selection when skills have *heterogeneous*
ages and the truncation (top-k / threshold) is aggressive enough to cross the
boundary.  See docs/stage55_freshness_design.md section 2.

The ``steps`` override map is the heterogeneous-age lever: it records, per
skill, the step the SkillBank *thinks* that skill was last measured at (which
may differ from the freshest measurement actually available in the timeline —
e.g. a skill not re-measured since SFT).  A ``"*"`` entry supplies the default
for any skill without an explicit entry.
"""

import math


class FreshnessScorer:
    def __init__(self, timeline, current_step, lambda_, steps=None):
        self.timeline = timeline
        self.current_step = current_step
        self.lambda_ = lambda_
        self.steps = steps or {}   # {skill_id: measured_step}, "*" = default

    def measured_step(self, skill_id):
        if skill_id in self.steps:
            return self.steps[skill_id]
        if "*" in self.steps:
            return self.steps["*"]
        return self.timeline.latest_le(skill_id, self.current_step)

    def freshness(self, skill_id):
        """F_i = e^{-lambda * max(0, current_step - measured_step)}."""
        t = self.measured_step(skill_id)
        if t is None:
            return 1.0
        return math.exp(-self.lambda_ * max(0.0, self.current_step - t))

    def score(self, skill_id):
        """Score_i = U_i(t_i) * F_i."""
        t = self.measured_step(skill_id)
        u = self.timeline.utility(skill_id, t) if t is not None else 0.0
        return u * self.freshness(skill_id)

    def select(self, skills, mode="rank", threshold=0.0, top_k=None):
        """Apply filter and/or rank on ``score``.  ``mode`` is one of
        ``"filter"`` / ``"rank"`` / ``"filter_rank"`` (same contract as
        ``freshskill.scorer.select``)."""
        if mode not in ("filter", "rank", "filter_rank"):
            raise ValueError(f"unknown mode: {mode}")

        def key(s):
            return self.score(s.get("skill_id", ""))

        out = list(skills)
        if mode in ("filter", "filter_rank"):
            out = [s for s in out if key(s) >= threshold]
        if mode in ("rank", "filter_rank"):
            out = sorted(out, key=key, reverse=True)
            if top_k is not None:
                out = out[:top_k]
        return out
