import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from lerobot.scripts.nero_ros2_deploy_act_real import (
    ObservationFreshnessWatchdog,
    configure_temporal_ensemble,
    parse_args,
    start_errors,
    validate_start_state,
)


class RealDeploymentSafetyTest(unittest.TestCase):
    def test_original_cli_options_remain_compatible(self):
        with patch(
            "sys.argv",
            [
                "nero_ros2_deploy_act_real.py",
                "--n-action-steps",
                "1",
                "--allow-ood-start",
                "--dry-run",
            ],
        ):
            args = parse_args()

        self.assertEqual(args.n_action_steps, 1)
        self.assertTrue(args.allow_start_mismatch)
        self.assertTrue(args.dry_run)

    def test_configure_temporal_ensemble_forces_single_step_inference(self):
        config = SimpleNamespace(type="act", n_action_steps=5, temporal_ensemble_coeff=None)

        config.chunk_size = 100
        configure_temporal_ensemble(config, 0.01, requested_n_action_steps=5)

        self.assertEqual(config.n_action_steps, 1)
        self.assertEqual(config.temporal_ensemble_coeff, 0.01)

    def test_start_errors_separate_arm_and_gripper(self):
        state = np.asarray([0.0, -1.48, 0.0, 2.14, 0.0, 0.0, 1.57, 0.09])
        target = np.asarray([0.0, -1.50, 0.0, 2.14, 0.0, 0.0, 1.57])

        arm_error, gripper_error = start_errors(state, target, 0.096)

        self.assertAlmostEqual(arm_error, 0.02)
        self.assertAlmostEqual(gripper_error, 0.006)

    def test_validate_start_state_rejects_mismatch_without_override(self):
        state = np.asarray([0.0, -1.70, 0.0, 2.14, 0.0, 0.0, 1.57, 0.096])
        target = np.asarray([0.0, -1.50, 0.0, 2.14, 0.0, 0.0, 1.57])

        with self.assertRaisesRegex(RuntimeError, "偏离采集起始姿态"):
            validate_start_state(state, target, 0.096, 0.03, 0.005, allow_mismatch=False)

    def test_freshness_watchdog_accepts_advancing_streams(self):
        watchdog = ObservationFreshnessWatchdog({"joint_state": 1, "color": 1}, now=10.0)

        watchdog.check({"joint_state": 2, "color": 2}, now=10.6, timeout_s=0.5)

    def test_freshness_watchdog_rejects_stale_stream(self):
        watchdog = ObservationFreshnessWatchdog({"joint_state": 1, "color": 1}, now=10.0)

        with self.assertRaisesRegex(RuntimeError, "color"):
            watchdog.check({"joint_state": 2, "color": 1}, now=10.6, timeout_s=0.5)


if __name__ == "__main__":
    unittest.main()
