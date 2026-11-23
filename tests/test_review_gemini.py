"""Direct local reviews through the Google Gemini API (`pi --provider google`) adapter."""
import argparse
import datetime
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from review_fixture import ReviewFixture

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools/project"))
sys.path.append(str(Path(__file__).resolve().parents[1] / "tools/review"))
from sphereceti.archive_store import snapshot, validate  # noqa: E402
from sphereceti.review_cost_bound import RequestLimits, installed_price_windows, round_cost_bound  # noqa: E402
from sphereceti.worker_cli import add_parser as add_worker_parser  # noqa: E402
from sphereceti.worker_execution import requested_model, review_arguments  # noqa: E402
from runner import costs  # noqa: E402
from runner.pricing import usage_totals  # noqa: E402

# The fake pi turn: 20k fresh input + 80k cache reads = 100k total input, 1k output (400 reasoning).
NORMALIZED = {"input_tokens": 100000, "fresh_input_tokens": 20000, "cached_input_tokens": 80000,
              "cache_creation_input_tokens": 0, "output_tokens": 1000, "reasoning_output_tokens": 400}


def flag_value(argv, flag):
    return argv[argv.index(flag) + 1] if flag in argv else None


class GeminiReviewTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="sphereceti-gemini-test-")
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.fixture = ReviewFixture(self.root)

    def review(self, *extra, output="result"):
        process = self.fixture.run(output=output, extra=("--provider", "gemini", "--rubrics", "correctness", *extra))
        self.assertEqual(process.returncode, 0, process.stderr + process.stdout)
        return json.loads(process.stdout)

    def archived_runs(self):
        journal = self.root / "state/thefundamentaltheor3m__SphereCeti/archive/journal"
        return [(record, run) for record in validate(snapshot(journal), "thefundamentaltheor3m/SphereCeti")
                for run in record["runs"]]

    def spent_today(self):
        return self.fixture.ledger()["days"][str(datetime.date.today())]

    def test_default_gemini_review_uses_google_provider_readonly_tools_and_isolated_key(self):
        result = self.review()
        self.assertEqual(result["verdicts"]["correctness"], "approve")
        [call] = self.fixture.pi_calls()
        argv = call["argv"]
        self.assertEqual(flag_value(argv, "--provider"), "google")
        self.assertEqual(flag_value(argv, "--model"), "gemini-4-carbon")
        self.assertEqual(flag_value(argv, "--mode"), "json")
        self.assertIn("--print", argv)
        for flag in ("--no-session", "--no-context-files", "--no-skills",
                     "--no-extensions", "--no-prompt-templates"):
            self.assertIn(flag, argv)
        tools = set(flag_value(argv, "--tools").split(","))
        self.assertEqual(tools, {"read", "grep", "ls"})
        self.assertFalse(tools & {"bash", "edit", "write"})
        self.assertEqual(call["gemini_key"], "test-gemini-key")
        self.assertEqual(call["leaked"], [])
        self.assertTrue(call["prompt_ok"])
        rev = self.fixture.home / ".tauceti-rev"
        self.assertTrue(Path(call["home"]).is_relative_to(rev))
        self.assertEqual(list(rev.iterdir()), [], "per-attempt throwaway HOME is removed")
        # Priced from prices.json (ignoring pi's self-reported $9.99):
        # (20k fresh x 2.00 + 80k cached x 0.10 + 1k out x 10.00) / 1M = $0.058.
        self.assertAlmostEqual(self.spent_today(), 0.058)
        [(record, run)] = self.archived_runs()
        self.assertEqual((run["provider"], run["model"]), ("gemini", "gemini-4-carbon"))
        self.assertTrue(run["cost_estimated"])
        self.assertAlmostEqual(run["cost_usd"], 0.058)
        self.assertEqual(run["usage"], {"input_tokens": 20000, "cached_input_tokens": 80000,
                                        "output_tokens": 1000, "reasoning_output_tokens": 400})
        [state] = self.fixture.ledger()["prs"].values()
        self.assertEqual(state["rounds"][-1]["tokens"], NORMALIZED)
        # Analytics pricing from the normalized tokens agrees with the live engine estimate.
        totals = usage_totals("gemini", run["usage"])
        self.assertEqual(totals, NORMALIZED)
        win = costs.price_window(installed_price_windows(), "gemini-4-carbon", "2026-11-23")
        self.assertAlmostEqual(costs.cost_from_window(win, totals["input_tokens"],
                                                      totals["cached_input_tokens"],
                                                      totals["output_tokens"]), 0.058)
        bound = round_cost_bound(installed_price_windows(), ("gemini-4-carbon",),
                                 RequestLimits(100_000, 1_000, 1), attempts=1, rubrics=1)
        self.assertGreaterEqual(bound.per_request, 0.058)

    def test_explicit_gemini_models_are_dispatched_and_billed_at_their_rates(self):
        cases = (
            ("gemini-4-argon", 0.058),
            ("gemini-4-neon", 0.087),
            ("gemini-4-hydrogen", 0.0435),
            ("gemini-3.8-flash", 0.02475),
        )
        expected_total = 0.0
        for index, (model, cost) in enumerate(cases):
            result = self.review("--mode", "manual", "--gemini-model", model, output=f"model-{index}")
            self.assertEqual(result["verdicts"]["correctness"], "approve")
            expected_total += cost
            self.assertAlmostEqual(self.spent_today(), expected_total, places=6)
        self.assertEqual([flag_value(c["argv"], "--model") for c in self.fixture.pi_calls()],
                         [model for model, _ in cases])

    def test_default_auth_mode_loads_saved_operator_config_and_passes_gemini_key(self):
        saved = self.fixture.home / ".config/sphereceti/operator.toml"
        saved.parent.mkdir(parents=True)
        saved.write_text(f'provider = "gemini"\nbudget_usd = 4\nstorage = {json.dumps(str(self.root / "saved"))}\n')
        cmd = self.fixture.command(extra=("--rubrics", "correctness"))
        for flag in ("--operator-config", "--auth"):
            pos = cmd.index(flag)
            del cmd[pos:pos + 2]
        result = subprocess.run(cmd, cwd=self.fixture.foreign, env=self.fixture.env,
                                capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        report = json.loads(result.stdout)
        self.assertEqual(report["verdicts"]["correctness"], "approve")
        self.assertEqual(report["evaluation_request"]["daily_budget_usd"], 4)
        [call] = self.fixture.pi_calls()
        self.assertEqual(call["gemini_key"], "test-gemini-key")

    def test_missing_gemini_api_key_fails_fast_before_invoking_pi(self):
        self.fixture.env.pop("GEMINI_API_KEY", None)
        process = self.fixture.run(extra=("--provider", "gemini", "--rubrics", "correctness"))
        self.assertEqual(process.returncode, 2)
        self.assertIn("GEMINI_API_KEY", process.stderr)
        self.assertEqual(self.fixture.pi_calls(), [])

    def test_unpriced_gemini_model_is_rejected_before_inference(self):
        process = self.fixture.run(extra=("--provider", "gemini", "--rubrics", "correctness",
                                          "--gemini-model", "gemini-99-unknown"))
        self.assertEqual(process.returncode, 1, process.stderr)
        self.assertEqual(json.loads(process.stdout)["completion"], "error")
        self.assertEqual(self.fixture.pi_calls(), [])

    def test_attempt_deadline_stops_pi_and_records_unknown_usage(self):
        (self.fixture.bin / "pi-hang").write_text("1")
        result = self.review("--attempt-timeout", "2")
        [pids] = self.fixture.pi_pids()
        for pid in (pids["leader"], pids["child"]):
            path = Path(f"/proc/{pid}/stat")
            self.assertTrue(not path.exists() or path.read_text().rsplit(")", 1)[1].split()[0] == "Z", pid)
        self.assertEqual(len(self.fixture.pi_calls()), 1)
        self.assertEqual((result["stop_reason"], result["usage_unknown_rubrics"]),
                         ("usage_unknown", ["correctness"]))
        [(record, run)] = self.archived_runs()
        self.assertEqual(run["usage_status"], "unknown")
        self.assertNotIn("cost_usd", run)

    def test_nonzero_pi_exit_records_unknown_usage_and_rejects_verdict(self):
        (self.fixture.bin / "pi-exit").write_text("1")
        result = self.review()
        self.assertEqual(result["verdicts"]["correctness"], "error")
        self.assertEqual((result["stop_reason"], result["usage_unknown_rubrics"]),
                         ("usage_unknown", ["correctness"]))
        self.assertEqual(len(self.fixture.pi_calls()), 1)

    def test_pi_api_error_is_error_never_approval(self):
        (self.fixture.bin / "pi-error").write_text("API key not valid. Please pass a valid API key (API_KEY_INVALID).")
        process = self.fixture.run(extra=("--provider", "gemini", "--rubrics", "correctness"))
        self.assertEqual(process.returncode, 1, process.stderr)
        result = json.loads(process.stdout)
        self.assertEqual(result["completion"], "error")
        self.assertNotEqual(result["verdicts"]["correctness"], "approve")

    def test_worker_cli_forwards_gemini_model_and_records_it(self):
        parser = argparse.ArgumentParser()
        add_worker_parser(parser.add_subparsers(dest="command"))
        args = parser.parse_args(["worker", "run", "--execute", "--pr", "1",
                                  "--provider", "gemini", "--gemini-model", "gemini-4-neon"])
        review = review_arguments(args, 1)
        self.assertEqual(review.provider, "gemini")
        self.assertEqual(review.gemini_model, "gemini-4-neon")
        self.assertEqual(requested_model(args), {"gemini_model": "gemini-4-neon"})


if __name__ == "__main__":
    unittest.main()
