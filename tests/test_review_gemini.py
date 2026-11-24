# SPDX-License-Identifier: Apache-2.0
"""Direct local reviews through SphereCeti's built-in read-only Google Gemini API runner."""
import argparse
import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import threading
import time
import unittest

from review_fixture import ReviewFixture

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools/project"))
sys.path.append(str(Path(__file__).resolve().parents[1] / "tools/review"))
from sphereceti import gemini_agent  # noqa: E402
from sphereceti.archive_store import snapshot, validate  # noqa: E402
from sphereceti.review_cost_bound import RequestLimits, installed_price_windows, round_cost_bound  # noqa: E402
from sphereceti.review_summary import receipt_runs  # noqa: E402
from sphereceti.worker_cli import add_parser as add_worker_parser  # noqa: E402
from sphereceti.worker_execution import requested_model, review_arguments  # noqa: E402
from runner import costs  # noqa: E402
from runner.pricing import usage_totals  # noqa: E402

# Two turns totaling 20k fresh input + 80k cached = 100k total input, 1k output (400 thoughts).
NORMALIZED = {"input_tokens": 100000, "fresh_input_tokens": 20000, "cached_input_tokens": 80000,
              "cache_creation_input_tokens": 0, "output_tokens": 1000, "reasoning_output_tokens": 400}


class MockGeminiServer:
    """Loopback mock of Google's Gemini REST API (`POST /v1beta/models/<model>:generateContent`)."""

    def __init__(self, mode="two_turn"):
        self.mode = mode
        self.requests = []
        self.lock = threading.Lock()
        server_ref = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                pass

            def do_POST(self):
                length = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(length).decode("utf-8"))
                record = {
                    "path": self.path,
                    "api_key": self.headers.get("x-goog-api-key"),
                    "body": body,
                }
                with server_ref.lock:
                    server_ref.requests.append(record)
                    mode = server_ref.mode
                if mode == "hang":
                    time.sleep(30)
                    return
                if mode == "auth_error":
                    payload = {"error": {"code": 401, "status": "UNAUTHENTICATED",
                                         "message": "API key not valid. Please pass a valid API key (API_KEY_INVALID)."}}
                    self._send_json(401, payload)
                    return
                if mode == "rate_limit":
                    payload = {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED",
                                         "message": "Quota exceeded for model."}}
                    self._send_json(429, payload)
                    return
                contents = body["contents"]
                prompt = contents[0]["parts"][0]["text"]
                marker = re.findall(r"TAUCETI-VERDICT-[a-z0-9]+", prompt)[-1]
                if mode == "two_turn" and len(contents) == 1:
                    # Turn 1: thinking + function calls carrying thoughtSignature and call ids.
                    self._send_json(200, {
                        "candidates": [{
                            "content": {
                                "role": "model",
                                "parts": [
                                    {"thought": True, "text": "Inspecting approved and candidate files."},
                                    {
                                        "thoughtSignature": "opaque-sig-turn-1",
                                        "functionCall": {
                                            "id": "call-read-1",
                                            "name": "read_file",
                                            "args": {"path": "approved/README.md", "offset": 1, "limit": 20},
                                        },
                                    },
                                    {
                                        "functionCall": {
                                            "id": "call-grep-2",
                                            "name": "grep",
                                            "args": {"pattern": "PROPOSED SENTINEL", "path": "code"},
                                        },
                                    },
                                ],
                            },
                            "finishReason": "STOP",
                        }],
                        "usageMetadata": {
                            "promptTokenCount": 50000,
                            "cachedContentTokenCount": 40000,
                            "candidatesTokenCount": 200,
                            "thoughtsTokenCount": 150,
                        },
                    })
                    return
                # Final turn: visible verdict text + hidden thought block that must be excluded.
                verdict = {"verdict": "approve", "summary": "Mock Gemini verified workspace.", "findings": []}
                self._send_json(200, {
                    "candidates": [{
                        "content": {
                            "role": "model",
                            "parts": [
                                {"thought": True, "text": "Internal thought that must not leak into text."},
                                {"text": f"Checked files.\n{marker}\n" + json.dumps(verdict)},
                            ],
                        },
                        "finishReason": "STOP",
                    }],
                    "usageMetadata": {
                        "promptTokenCount": 50000,
                        "cachedContentTokenCount": 40000,
                        "candidatesTokenCount": 400,
                        "thoughtsTokenCount": 250,
                    },
                })

            def _send_json(self, status, payload):
                raw = json.dumps(payload).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                try:
                    self.wfile.write(raw)
                except BrokenPipeError:
                    pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.httpd.daemon_threads = True
        self.port = self.httpd.server_address[1]
        self.base_url = f"http://127.0.0.1:{self.port}/v1beta"
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


class GeminiAgentToolsTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="sphereceti-gemini-tools-")
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.ws = self.root / "workspace"
        (self.ws / "code").mkdir(parents=True)
        (self.ws / "code/Main.lean").write_text("import Mathlib\ntheorem foo : True := trivial\n")
        (self.ws / "code/link_inside.lean").symlink_to("Main.lean")
        (self.root / "secret.txt").write_text("OUTSIDE SECRET\n")
        (self.ws / "escape_link.txt").symlink_to(str(self.root / "secret.txt"))
        self.root_real = str(self.ws.resolve())

    def test_read_grep_list_find_and_path_jailing(self):
        out, target, ok = gemini_agent.dispatch_tool(self.root_real, "read_file", {"path": "code/Main.lean"})
        self.assertTrue(ok)
        self.assertEqual(target, "code/Main.lean")
        self.assertIn("2: theorem foo : True := trivial", out)

        out, target, ok = gemini_agent.dispatch_tool(self.root_real, "read_file", {"path": "code/link_inside.lean"})
        self.assertTrue(ok)
        self.assertIn("theorem foo", out)

        out, target, ok = gemini_agent.dispatch_tool(self.root_real, "grep", {"pattern": "theorem foo", "glob": "*.lean"})
        self.assertTrue(ok)
        self.assertIn("code/Main.lean:2:theorem foo", out)

        out, target, ok = gemini_agent.dispatch_tool(self.root_real, "list_dir", {"path": "code"})
        self.assertTrue(ok)
        self.assertIn("Main.lean", out)

        out, target, ok = gemini_agent.dispatch_tool(self.root_real, "find_files", {"pattern": "*.lean"})
        self.assertTrue(ok)
        self.assertIn("code/Main.lean", out)

        for bad_path in ("../secret.txt", "/etc/passwd", "escape_link.txt"):
            out, target, ok = gemini_agent.dispatch_tool(self.root_real, "read_file", {"path": bad_path})
            self.assertFalse(ok, bad_path)
            self.assertEqual(target, gemini_agent.OUTSIDE, bad_path)
            self.assertNotIn("OUTSIDE SECRET", out)

        out, target, ok = gemini_agent.dispatch_tool(self.root_real, "bash", {"command": "id"})
        self.assertFalse(ok)
        self.assertIn("unknown or disallowed tool", out)

    def test_custom_api_base_rejects_non_loopback_urls(self):
        self.assertEqual(gemini_agent.resolve_api_base(None), gemini_agent.DEFAULT_API_BASE)
        self.assertEqual(gemini_agent.resolve_api_base("http://127.0.0.1:8080/v1beta"),
                         "http://127.0.0.1:8080/v1beta")
        for bad in ("https://evil.example/v1beta", "http://127.0.0.2:8080/v1beta", "http://localhost:8080/v1beta"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                gemini_agent.resolve_api_base(bad)


class GeminiReviewTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="sphereceti-gemini-test-")
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.fixture = ReviewFixture(self.root)
        self.server = MockGeminiServer()
        self.addCleanup(self.server.close)
        self.fixture.env["SPHERECETI_GEMINI_API_BASE"] = self.server.base_url

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

    def test_end_to_end_multi_turn_gemini_review_preserves_thought_signature_and_accounts_cost(self):
        result = self.review()
        self.assertEqual(result["verdicts"]["correctness"], "approve")
        self.assertIn("gemini gemini-4-carbon", result["plan"]["line"])
        self.assertEqual(len(self.server.requests), 2)
        req1, req2 = self.server.requests
        self.assertEqual(req1["path"], "/v1beta/models/gemini-4-carbon:generateContent")
        self.assertNotIn("test-gemini-key", req1["path"])
        self.assertEqual(req1["api_key"], "test-gemini-key")
        declared = {d["name"] for d in req1["body"]["tools"][0]["functionDeclarations"]}
        self.assertEqual(declared, {"read_file", "grep", "list_dir", "find_files"})
        # Turn 2 must echo turn 1's model parts verbatim (including thoughtSignature) + functionResponses.
        model_parts = req2["body"]["contents"][1]["parts"]
        self.assertEqual(model_parts[1]["thoughtSignature"], "opaque-sig-turn-1")
        responses = [p["functionResponse"] for p in req2["body"]["contents"][2]["parts"]]
        self.assertEqual([r["id"] for r in responses], ["call-read-1", "call-grep-2"])
        self.assertIn("APPROVED SENTINEL README.md", responses[0]["response"]["output"])
        self.assertIn("code/README.md:1:PROPOSED SENTINEL", responses[1]["response"]["output"])
        # Per-attempt throwaway HOME must be cleaned up.
        rev = self.fixture.home / ".tauceti-rev"
        self.assertEqual(list(rev.iterdir()), [])
        # Priced from prices.json: (20k fresh x 2.00 + 80k cached x 0.10 + 1k out x 10.00) / 1M = $0.058.
        self.assertAlmostEqual(self.spent_today(), 0.058)
        [(record, run)] = self.archived_runs()
        self.assertEqual((run["provider"], run["model"]), ("gemini", "gemini-4-carbon"))
        self.assertTrue(run["cost_estimated"])
        self.assertAlmostEqual(run["cost_usd"], 0.058)
        self.assertEqual(run["usage"], {"input_tokens": 20000, "cached_input_tokens": 80000,
                                        "output_tokens": 1000, "reasoning_output_tokens": 400})
        [receipt] = receipt_runs(result)
        self.assertEqual(receipt["tool_trace"], [
            {"tool": "read_file", "ok": True, "target": "approved/README.md"},
            {"tool": "grep", "ok": True, "target": "code"},
        ])
        self.assertEqual(receipt["tool_trace_meta"], {"total_calls": 2, "trace_truncated": False})
        [state] = self.fixture.ledger()["prs"].values()
        self.assertEqual(state["rounds"][-1]["tokens"], NORMALIZED)
        totals = usage_totals("gemini", run["usage"])
        self.assertEqual(totals, NORMALIZED)
        win = costs.price_window(installed_price_windows(), "gemini-4-carbon", "2026-11-24")
        self.assertAlmostEqual(costs.cost_from_window(win, totals["input_tokens"],
                                                      totals["cached_input_tokens"],
                                                      totals["output_tokens"]), 0.058)
        bound = round_cost_bound(installed_price_windows(), ("gemini-4-carbon",),
                                 RequestLimits(100_000, 1_000, 1), attempts=1, rubrics=1)
        self.assertGreaterEqual(bound.per_request, 0.058)

    def test_explicit_gemini_models_hit_exact_endpoint_and_bill_their_rates(self):
        self.server.mode = "single_turn"
        models = ("gemini-4-argon", "gemini-4-neon", "gemini-4-hydrogen", "gemini-3.8-flash")
        expected_total = 0.0
        for index, model in enumerate(models):
            result = self.review("--mode", "manual", "--gemini-model", model, output=f"model-{index}")
            self.assertEqual(result["verdicts"]["correctness"], "approve")
            self.assertIn(f"gemini {model}", result["plan"]["line"])
            win = costs.price_window(installed_price_windows(), model, "2026-11-24")
            expected_total += round(
                (10_000 * win["input"] + 40_000 * win["cache_read"] + 650 * win["output"]) / 1e6, 6
            )
            self.assertAlmostEqual(self.spent_today(), expected_total, places=6)
        self.assertEqual([r["path"] for r in self.server.requests],
                         [f"/v1beta/models/{model}:generateContent" for model in models])

    def test_default_auth_mode_loads_saved_operator_config_and_uses_gemini_key(self):
        self.server.mode = "single_turn"
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
        self.assertEqual(self.server.requests[0]["api_key"], "test-gemini-key")

    def test_missing_gemini_api_key_fails_fast_before_network(self):
        self.fixture.env.pop("GEMINI_API_KEY", None)
        process = self.fixture.run(extra=("--provider", "gemini", "--rubrics", "correctness"))
        self.assertEqual(process.returncode, 2)
        self.assertIn("GEMINI_API_KEY", process.stderr)
        self.assertEqual(self.server.requests, [])

    def test_unpriced_gemini_model_is_rejected_before_inference(self):
        process = self.fixture.run(extra=("--provider", "gemini", "--rubrics", "correctness",
                                          "--gemini-model", "gemini-99-unknown"))
        self.assertEqual(process.returncode, 1, process.stderr)
        self.assertEqual(json.loads(process.stdout)["completion"], "error")
        self.assertEqual(self.server.requests, [])

    def test_attempt_deadline_stops_gemini_runner_and_records_unknown_usage(self):
        self.server.mode = "hang"
        result = self.review("--attempt-timeout", "2")
        self.assertEqual(len(self.server.requests), 1)
        self.assertEqual((result["stop_reason"], result["usage_unknown_rubrics"]),
                         ("usage_unknown", ["correctness"]))
        [(record, run)] = self.archived_runs()
        self.assertEqual(run["usage_status"], "unknown")
        self.assertNotIn("cost_usd", run)

    def test_api_auth_and_rate_limit_errors_trip_provider_down_and_never_approve(self):
        for index, (mode, kind) in enumerate((("auth_error", "not_authenticated"),
                                              ("rate_limit", "rate_limited"))):
            self.server.mode = mode
            self.server.requests.clear()
            process = self.fixture.run(output=f"err-{index}",
                                       extra=("--provider", "gemini", "--rubrics", "correctness"))
            self.assertEqual(process.returncode, 1, process.stderr)
            result = json.loads(process.stdout)
            self.assertEqual(result["completion"], "error")
            self.assertEqual(result["engine_exit_code"], 3)  # PROVIDER_DOWN_EXIT
            self.assertEqual(len(self.server.requests), 2)   # confirmed twice, then aborted
            self.assertIn(kind.replace("_", " "), result["summary"]["line"])

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
