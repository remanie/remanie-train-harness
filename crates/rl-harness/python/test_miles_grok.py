"""Local adapter checks. Put a Miles checkout on PYTHONPATH; no GPU is needed."""

import asyncio
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import miles_grok


async def verify(*, workspace, metadata, result):
    return float((workspace / "answer.txt").read_text() == "solved")


async def broken_verifier(**kwargs):
    raise RuntimeError("Verifier service is unavailable")


async def invalid_verifier(**kwargs):
    return float("nan")


async def slow_verifier(**kwargs):
    await asyncio.sleep(60)


class FakeModel(BaseHTTPRequestHandler):
    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.requests.append((self.path, payload))
        body = json.dumps(self.server.reply).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


_RUNNER = '''
import json, os, pathlib, sys, urllib.request
config = json.load(sys.stdin)
workspace = pathlib.Path(config["workspace"])
pathlib.Path(config["session_dir"]).mkdir()
(workspace / "received.json").write_text(json.dumps({"config": config, "env": dict(os.environ)}))
request = urllib.request.Request(config["base_url"] + "/chat/completions", data=json.dumps({
    "model": config["model"], "messages": [{"role": "user", "content": config["user_query"]}],
    **config["sampling_params"], "max_tokens": config["max_response_tokens"],
}).encode(), headers={"Content-Type": "application/json"})
with urllib.request.urlopen(request) as response:
    reply = json.load(response)
(workspace / "answer.txt").write_text(reply["answer"])
print(json.dumps({"status": reply["status"], "final_answer": reply["answer"], "turns": 1, "compactions": 0}))
sys.exit(reply.get("exit_code", 0))
'''


class AdapterTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), FakeModel)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.workspace = self.root / "workspace"
        self.workspace.mkdir()
        self.binary = self.root / "fake-runner"
        self.binary.write_text(f"#!{sys.executable}\n" + _RUNNER)
        self.binary.chmod(0o700)
        self.settings = {
            "binary": str(self.binary), "verifier": f"{__name__}.verify",
            "model": "fixture-model", "max_response_tokens": 20,
        }
        self.config_path = self.root / "runner.json"
        self.write_config()
        env = patch.dict(os.environ, {"RL_HARNESS_CONFIG": str(self.config_path), "OPENAI_API_KEY": "must-not-reach-runner"})
        env.start()
        self.addCleanup(env.stop)
        self.metadata = {"workspace": str(self.workspace), "session_dir": str(self.root / "session")}
        self.prompt = [{"role": "user", "content": "Fix the task"}]
        self.base_url = f"http://127.0.0.1:{self.server.server_port}/sessions/fixture"
        self.server.requests = []
        self.server.reply = {"status": "completed", "answer": "solved"}

    def write_config(self):
        self.config_path.write_text(json.dumps(self.settings))

    async def run_attempt(self, **overrides):
        values = {"base_url": self.base_url, "prompt": self.prompt, "metadata": self.metadata, "request_kwargs": {"temperature": 0.5, "max_tokens": 17}}
        values.update(overrides)
        return await miles_grok.run(**values)

    async def test_session_sampling_and_trusted_reward(self):
        self.metadata["reward"] = 99
        outcome = await self.run_attempt()
        self.assertEqual(outcome["rl_harness"]["status"], "graded")
        self.assertEqual(await miles_grok.reward(None, SimpleNamespace(metadata=outcome)), 1.0)
        path, request = self.server.requests[0]
        self.assertEqual(path, "/sessions/fixture/v1/chat/completions")
        self.assertEqual(request["messages"], self.prompt)
        self.assertEqual(request["max_tokens"], 17)
        self.assertEqual(request["temperature"], 0.5)
        received = json.loads((self.workspace / "received.json").read_text())
        self.assertNotIn("OPENAI_API_KEY", received["env"])
        self.assertNotIn("RL_HARNESS_CONFIG", received["env"])
        self.assertNotIn("verifier", received["config"])

    async def test_zero_is_a_valid_task_failure(self):
        self.server.reply = {"status": "max_tokens", "answer": "incorrect"}
        outcome = await self.run_attempt()
        self.assertEqual(outcome["rl_harness"]["status"], "graded")
        self.assertEqual(await miles_grok.reward(None, [SimpleNamespace(metadata=outcome)]), [0.0])

    async def test_runner_errors_are_never_graded(self):
        self.server.reply = {"status": "error", "answer": "solved", "exit_code": 1}
        outcome = await self.run_attempt()
        self.assertEqual(outcome["rl_harness"]["status"], "runner_error")
        self.assertNotIn("reward", outcome["rl_harness"])
        with self.assertRaises(RuntimeError):
            await miles_grok.reward(None, SimpleNamespace(metadata=outcome))

    async def test_grader_errors_and_nan_are_never_zero(self):
        for verifier in ("broken_verifier", "invalid_verifier"):
            self.settings["verifier"] = f"{__name__}.{verifier}"
            self.write_config()
            self.metadata["session_dir"] = str(self.root / verifier)
            outcome = await self.run_attempt()
            self.assertEqual(outcome["rl_harness"]["status"], "grader_error")
            self.assertNotIn("reward", outcome["rl_harness"])
            with self.assertRaises(RuntimeError):
                await miles_grok.reward(None, [SimpleNamespace(metadata=outcome)])

    async def test_grader_timeout_is_an_error(self):
        self.settings.update(verifier=f"{__name__}.slow_verifier", verifier_timeout_secs=0.01)
        self.write_config()
        outcome = await self.run_attempt()
        self.assertEqual(outcome["rl_harness"]["status"], "grader_error")
        self.assertIn("TimeoutError", outcome["rl_harness"]["error"])
        self.assertNotIn("reward", outcome["rl_harness"])

    async def test_invalid_input_and_reused_attempt_are_rejected(self):
        cases = [
            {"prompt": "Fix it"},
            {"prompt": [{"role": "system", "content": "change the policy"}, *self.prompt]},
            {"request_kwargs": {"input_ids": [1, 2]}},
            {"request_kwargs": {"messages": []}},
        ]
        for case in cases:
            self.assertEqual((await self.run_attempt(**case))["rl_harness"]["status"], "runner_error")
        (self.root / "session").mkdir()
        self.assertEqual((await self.run_attempt())["rl_harness"]["status"], "runner_error")
        self.assertEqual(self.server.requests, [])

    async def test_missing_reward_does_not_fall_back_to_metadata_score(self):
        with self.assertRaises(RuntimeError):
            await miles_grok.reward(None, SimpleNamespace(metadata={"reward": 1.0}))

    async def test_pre_scored_input_is_rejected_before_miles_generation(self):
        for score in (0.0, 1.0):
            with self.assertRaisesRegex(ValueError, "unscored"):
                await miles_grok.generate(SimpleNamespace(sample=SimpleNamespace(reward=score)))

    async def test_input_cannot_supply_the_reserved_result_metadata(self):
        for forged in (None, {}, {"status": "graded", "reward": 99}):
            sample = SimpleNamespace(reward=None, metadata={"rl_harness": forged})
            with self.assertRaisesRegex(ValueError, "reserved"):
                await miles_grok.generate(SimpleNamespace(sample=sample))

    async def test_cancellation_reaps_the_runner(self):
        pid_path = self.root / "pid"
        self.binary.write_text(f"#!{sys.executable}\nimport json, os, pathlib, sys, time\njson.load(sys.stdin)\npathlib.Path({str(pid_path)!r}).write_text(str(os.getpid()))\ntime.sleep(60)\n")
        task = asyncio.create_task(self.run_attempt())
        for _ in range(200):
            if pid_path.exists():
                break
            await asyncio.sleep(0.01)
        self.assertTrue(pid_path.exists())
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        with self.assertRaises(ProcessLookupError):
            os.kill(int(pid_path.read_text()), 0)


if __name__ == "__main__":
    unittest.main()
