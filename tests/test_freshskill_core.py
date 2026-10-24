"""FreshSkill 核心不变量的轻量单元测试。"""

import unittest

from freshskill.bank_state import BankState
from freshskill.lifecycle import KEEP, RETIRE, LifecyclePolicy
from freshskill.measurement import utility, utility_variance
from freshskill.retirement import alpha_from_costs, costs_from_alpha
from freshskill.timeline import UtilityTimeline


COUNTS = {"help": 1, "harm": 0, "neutral": 9, "usage": 10}


class MeasurementTest(unittest.TestCase):
    def test_utility_matches_paired_outcome_mean(self):
        self.assertAlmostEqual(utility(6, 10, 124), -4 / 140)

    def test_variance_is_non_negative_and_handles_empty_sample(self):
        self.assertGreater(utility_variance(6, 10, 124), 0)
        self.assertEqual(utility_variance(0, 0, 0), 0)


class TimelineAndLifecycleTest(unittest.TestCase):
    def setUp(self):
        self.timeline = UtilityTimeline()
        self.timeline.set("skill_a", 20, -0.05, COUNTS)
        self.timeline.set("skill_a", 40, -0.03, COUNTS)
        self.timeline.set("skill_b", 40, 0.04, COUNTS)

    def test_timeline_returns_latest_measurement_not_after_target(self):
        self.assertEqual(self.timeline.latest_le("skill_a", 35), 20.0)
        self.assertIsNone(self.timeline.latest_le("skill_a", 10))

    def test_retirement_requires_two_consecutive_harmful_measurements(self):
        policy = LifecyclePolicy(theta_retire=-0.02)
        first = policy.maintain(self.timeline, current_step=20, prev_step=None)
        second = policy.maintain(self.timeline, current_step=40, prev_step=20)
        self.assertEqual(first["skill_a"], KEEP)
        self.assertEqual(second["skill_a"], RETIRE)

    def test_unmeasured_skill_is_not_retired(self):
        policy = LifecyclePolicy(theta_retire=-0.02)
        actions = policy.maintain(
            self.timeline,
            current_step=40,
            prev_step=20,
            skill_ids=["missing"],
        )
        self.assertEqual(actions["missing"], KEEP)

    def test_bank_state_exposes_only_active_skills(self):
        state = BankState()
        state.apply_actions(
            {"skill_a": RETIRE, "skill_b": KEEP},
            current_step=40,
            timeline=self.timeline,
        )
        self.assertEqual(state.active_skill_ids(), ["skill_b"])
        self.assertEqual(state.retired_skill_ids(), ["skill_a"])


class DecisionCostTest(unittest.TestCase):
    def test_cost_ratio_and_alpha_round_trip(self):
        false_retire, missed_harm = costs_from_alpha(0.05)
        self.assertAlmostEqual(false_retire, 19.0)
        self.assertAlmostEqual(alpha_from_costs(false_retire, missed_harm), 0.05)


if __name__ == "__main__":
    unittest.main()
