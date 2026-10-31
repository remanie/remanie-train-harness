import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from make_trace_fixture import MODEL, Endpoint

ROOT = Path(__file__).resolve().parents[1]
RUNNER = Path(os.environ.get("REMANIE_RUNNER", ROOT / "target/debug/remanie-train-harness"))
COMMIT = json.loads((ROOT / "upstream.json").read_text())["commit"]
FAKE_GROK = r'''
import atexit, json, os, re, signal, subprocess, sys, time, urllib.error, urllib.request
if sys.argv[1:] == ["--version"]:
    print(os.environ.get("FAKE_VERSION", "grok 1.0.45 (" + os.environ["FAKE_COMMIT"][:12] + ")"))
    sys.exit(0)
args = sys.argv[1:]
home = os.environ["GROK_HOME"]
helpers = [subprocess.Popen(["sleep", "600"]), subprocess.Popen(["sleep", "600"], start_new_session=True)]
atexit.register(lambda: [helper.kill() for helper in helpers])
with open(os.path.join(home, "pids"), "w") as pids:
    pids.write(" ".join(str(pid) for pid in [os.getpid(), *(helper.pid for helper in helpers)]) + "\n")
mode = os.environ["FAKE_GROK"]
if mode == "stubborn":
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
base = re.search(r'^base_url = "([^"]+)"$', open(os.path.join(home, "config.toml")).read(), re.M).group(1)

def post(body):
    request = urllib.request.Request(base + "/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json", "Authorization": "Bearer placeholder"})
    while True:
        try:
            with urllib.request.urlopen(request) as response:
                return response.read().decode()
        except urllib.error.HTTPError as error:
            if mode != "retry":
                return error.read().decode()
        time.sleep(0.2)

def turn():
    query = next(arg.removeprefix("-p=") for arg in args if arg.startswith("-p="))
    return post({"model": args[args.index("-m") + 1], "messages": [{"role": "user", "content": query}], "temperature": 1.0, "max_tokens": 512, "tools": [{"type": "function", "function": {"name": "read_file"}}], "stream": True})

if mode in ("hang", "stubborn"):
    time.sleep(600)
if mode == "error_first":
    print(json.dumps({"type": "error", "message": "Couldn't start session"}))
    sys.exit(1)
post({"model": "m", "messages": [{"role": "system", "content": "title"}], "tools": [{"type": "function", "function": {"name": "session_title"}}], "stream": True})
answer = turn()
if mode == "turn_then_hang":
    open(os.path.join(home, "answered"), "w").close()
    time.sleep(600)
session = os.path.join(home, "sessions", "%2Fworkspace", args[args.index("--session-id") + 1])
os.makedirs(session)
if mode == "error_after_reply":
    print(json.dumps({"type": "error", "message": "Model returned an error"}))
    sys.exit(1)
print(json.dumps({"text": "done after " + str(answer.count("data:")) + " events", "stopReason": "end_turn", "sessionId": "s"}, indent=2))
sys.exit(1 if mode == "max_turns" else 0)
'''
REPLIES = {"turn": [{"message": {"content": "Done."}, "finish_reason": "stop", "usage": [100, 5, 40]}]}


@unittest.skipUnless(RUNNER.is_file(), "build the runner or set REMANIE_RUNNER")
class WrapperTest(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        (self.root / "workspace").mkdir()
        (self.root / "grok").write_text(f"#!{sys.executable}\n{FAKE_GROK}")
        (self.root / "grok").chmod(0o755)
        self.endpoint = Endpoint(REPLIES)
        self.addCleanup(self.endpoint.server.shutdown)

    def job(self, **overrides):
        return {
            "base_url": self.endpoint.url, "model": MODEL, "api_key_env": "TEST_KEY", "user_query": "-v: say done.",
            "workspace": str(self.root / "workspace"), "session_dir": str(self.root / "session"),
            "sampling_params": {"max_tokens": 512, "temperature": 1.0, "provider": {"sort": "price"}},
            "max_turns": 4, "deadline_secs": 60, "context_window_tokens": 32768, **overrides,
        }

    def start(self, mode, job=None, **env):
        environment = {"PATH": os.environ["PATH"], "HOME": str(self.root), "TEST_KEY": "secret-key", "REMANIE_GROK": str(self.root / "grok"), "FAKE_GROK": mode, "FAKE_COMMIT": COMMIT, **env}
        process = subprocess.Popen([str(RUNNER)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env=environment)
        process.stdin.write(json.dumps(job or self.job()).encode())
        process.stdin.close()
        return process

    def finish(self, process):
        try:
            process.wait(timeout=90)
        except subprocess.TimeoutExpired:
            process.kill()
            raise
        result = json.loads(process.stdout.read())
        self.assertEqual(json.loads((self.root / "session/result.json").read_text()), result)
        return process.returncode, result

    def run_mode(self, mode, job=None, **env):
        return self.finish(self.start(mode, job, **env))

    def events(self):
        return [json.loads(line) for line in (self.root / "session/trace.jsonl").read_text().splitlines()]

    def grok_pids(self):
        grok, helper, detached = (int(pid) for pid in (self.root / "session/grok-home/pids").read_text().split())
        self.addCleanup(lambda: subprocess.run(["kill", "-9", str(detached)], capture_output=True))
        return [grok, helper, detached] if sys.platform == "linux" else [grok, helper]

    def assert_gone(self, pids):
        for pid in pids:
            for _ in range(50):
                state = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()
                if not state or state.startswith("Z"):
                    break
                time.sleep(0.1)
            else:
                self.fail(f"process {pid} outlived the runner")

    def test_a_completed_run_traces_every_call_and_injects_missing_sampling_params(self):
        code, result = self.run_mode("complete")
        self.assertEqual(code, 0)
        self.assertEqual((result["outcome"], result["gradable"], result["final_answer"]), ("completed", True, "done after 4 events"))
        self.assertEqual((result["turns"], result["compactions"], result["model_calls"]), (1, 0, 1))
        self.assertEqual(result["usage"], {"prompt_tokens": 100, "completion_tokens": 5, "cached_tokens": 40, "complete": True})
        self.assertEqual(result["grok_exit"], {"code": 0})
        self.assertTrue(result["grok_session_dir"].startswith("grok-home/sessions/"))
        forwarded = json.loads(self.endpoint.requests[0][1])
        self.assertEqual(forwarded["provider"], {"sort": "price"})
        self.assertEqual((forwarded["temperature"], forwarded["max_tokens"]), (1.0, 512))
        self.assertEqual(forwarded["messages"], [{"role": "user", "content": "-v: say done."}])
        self.assertEqual(self.endpoint.authorizations, ["Bearer secret-key"])
        self.assertEqual(self.endpoint.titles, [])
        events = self.events()
        self.assertEqual([event["event"] for event in events], ["header", "request", "response", "request", "response", "result"])
        self.assertEqual(events[0]["grok_version"], f"grok 1.0.45 ({COMMIT[:12]})")
        self.assertEqual([(event["call"], event["kind"]) for event in events if event["event"] == "request"], [(1, "title"), (2, "turn")])
        self.assertEqual(events[3]["injected"], ["provider"])
        self.assertEqual(events[3]["body"], forwarded)
        self.assertEqual(events[4]["body"]["choices"][0]["message"]["content"], "Done.")
        self.assertIn("data: [DONE]", events[4]["raw_sse"])
        self.assertEqual(events[-1]["outcome"], "completed")

    def test_the_deadline_stops_grok_and_its_process_group(self):
        code, result = self.run_mode("hang", self.job(deadline_secs=1))
        self.assertEqual((code, result["outcome"], result["cause"]), (0, "cancelled", {"kind": "deadline"}))
        self.assertEqual(result["grok_exit"], {"signal": signal.SIGTERM})
        self.assert_gone(self.grok_pids())

    def test_grok_that_ignores_sigterm_is_killed_five_seconds_later(self):
        started = time.monotonic()
        code, result = self.run_mode("stubborn", self.job(deadline_secs=1))
        self.assertEqual((code, result["outcome"]), (0, "cancelled"))
        self.assertEqual(result["grok_exit"], {"signal": signal.SIGKILL})
        self.assertGreaterEqual(time.monotonic() - started, 6)
        self.assert_gone(self.grok_pids())

    def test_sigterm_to_the_runner_cancels_the_attempt(self):
        process = self.start("turn_then_hang")
        for _ in range(100):
            if (self.root / "session/grok-home/answered").exists():
                break
            time.sleep(0.1)
        process.send_signal(signal.SIGTERM)
        code, result = self.finish(process)
        self.assertEqual((code, result["outcome"], result["cause"]), (0, "cancelled", {"kind": "signal", "signal": "SIGTERM"}))
        self.assertEqual(result["turns"], 1)
        self.assert_gone(self.grok_pids())

    def test_an_unreachable_endpoint_at_the_first_request_is_infrastructure(self):
        started = time.monotonic()
        code, result = self.run_mode("retry", self.job(base_url="http://127.0.0.1:9/v1"))
        self.assertEqual((code, result["outcome"], result["cause"]["kind"], result["gradable"]), (70, "endpoint_unavailable", "unreachable", False))
        self.assertLess(time.monotonic() - started, 30)
        self.assert_gone(self.grok_pids())

    def test_grok_exits_map_to_outcomes(self):
        for mode, code, outcome in (
            ("max_turns", 0, "max_turns"),
            ("error_after_reply", 0, "error"),
            ("error_first", 70, "infrastructure"),
        ):
            with self.subTest(mode):
                job = self.job(session_dir=str(self.root / f"session-{mode}"))
                process = self.start(mode, job)
                process.wait(timeout=90)
                result = json.loads(process.stdout.read())
                self.assertEqual((process.returncode, result["outcome"]), (code, outcome))
        self.assertEqual(result["cause"], {"kind": "grok_startup", "message": "Couldn't start session"})

    def test_a_grok_binary_at_another_commit_is_refused(self):
        code, result = self.run_mode("complete", FAKE_VERSION="grok 1.0.46 (0123456789ab)")
        self.assertEqual((code, result["outcome"], result["cause"]["kind"]), (70, "infrastructure", "grok_binary"))
        self.assertFalse((self.root / "session/trace.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
