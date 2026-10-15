"""Offline CLI exit regressions; workflow failures below are deliberately injected."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

RUNNER = Path(__file__).resolve().parent / "test_vision_analysis_mock.py"
STEPS = (
    "test_rating_extraction",
    "test_agreement_calculation",
    "test_result_caching",
    "test_visualization_data_prep",
)
CHILD = """
import runpy
import sys

runner, fault = sys.argv[1:]

def no_network(event, args):
    if event.startswith("socket.") or event in ("subprocess.Popen", "os.system", "os.exec", "os.posix_spawn"):
        raise RuntimeError("REGRESSION_NETWORK_BLOCKED")

sys.addaudithook(no_network)

def injected_failure(*args, **kwargs):
    raise RuntimeError("INJECTED_WORKFLOW_FAILURE: " + fault)

def inject(frame, event, arg):
    if event == "call" and frame.f_code.co_filename == runner and frame.f_code.co_name == "main":
        frame.f_globals[fault] = injected_failure
    return inject

if fault:
    sys.settrace(inject)
runpy.run_path(runner, run_name="__main__")
"""


class MockRunnerExitTests(unittest.TestCase):
    def run_cli(self, fault=""):
        # Each caller owns its cache directory; no parent credentials are inherited.
        with tempfile.TemporaryDirectory(prefix="mock-runner-exit-") as directory:
            cwd = Path(directory)
            env = {"TEMP": directory, "TMP": directory}
            if sys.platform == "win32":
                env.update(SystemRoot=r"C:\Windows", WINDIR=r"C:\Windows")
            result = subprocess.run(
                [sys.executable, "-I", "-B", "-X", "utf8", "-c", CHILD, str(RUNNER), fault],
                cwd=cwd, env=env, capture_output=True, text=True, encoding="utf-8",
                timeout=30, check=False,
            )
            if fault:
                self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
                self.assertIn("TEST FAILED: INJECTED_WORKFLOW_FAILURE: " + fault, result.stdout)
                self.assertIn("Traceback (most recent call last):", result.stderr)
                self.assertIn("RuntimeError: INJECTED_WORKFLOW_FAILURE: " + fault, result.stderr)
                self.assertNotIn("ALL MOCK TESTS PASSED!", result.stdout)
            else:
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn("ALL MOCK TESTS PASSED!", result.stdout)
                self.assertNotIn("TEST FAILED:", result.stdout)
                self.assertNotIn("Traceback (most recent call last):", result.stderr)
            self.assertNotIn("REGRESSION_NETWORK_BLOCKED", result.stdout + result.stderr)
            caches = list((cwd / "test_output" / "visual_llm_cache").glob("mock_test_*.json"))
            expected_cache = not fault or fault == STEPS[-1]
            self.assertEqual(len(caches), int(expected_cache))
            if caches:
                payload = json.loads(caches[0].read_text(encoding="utf-8"))
                self.assertEqual(payload["test_type"], "mock_vr_avatar_validation")
                self.assertEqual(sorted(payload["results"]), ["avatar_001", "avatar_002", "avatar_006"])
                self.assertEqual(sum(len(row["analyses"]) for row in payload["results"].values()), 6)

    def test_native_normal_main(self):
        self.run_cli()

    def test_injected_failure_at_each_workflow_step(self):
        for step in STEPS:
            with self.subTest(injected_step=step):
                self.run_cli(step)

    def test_bounded_concurrent_callers_keep_separate_caches(self):
        with ThreadPoolExecutor(max_workers=2) as callers:
            list(callers.map(self.run_cli, ("", STEPS[0], "", STEPS[-1])))

    def test_repeated_failure_then_normal_recovers_in_fresh_directories(self):
        for attempt in range(3):
            with self.subTest(recovery_pair=attempt):
                self.run_cli(STEPS[-1])
                self.run_cli()


if __name__ == "__main__":
    unittest.main()
