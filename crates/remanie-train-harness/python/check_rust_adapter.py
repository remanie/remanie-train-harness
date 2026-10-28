"""Exercise the built Rust binary through the Miles adapter and a local model fixture."""

import asyncio
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import miles_grok


async def verify(*, workspace, metadata, result):
    return float((workspace / "answer.txt").read_text() == "solved")


class Model(BaseHTTPRequestHandler):
    def do_POST(self):
        request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.requests.append(request)
        turn = len(self.server.requests)
        if turn == 1:
            message = {
                "role": "assistant", "content": None, "reasoning_content": "original reasoning",
                "tool_calls": [{"id": "fixture-call", "type": "function", "function": {
                    "name": "run_terminal_cmd",
                    "arguments": json.dumps({"command": self.server.command, "description": "Run the fixture command", "block_until_ms": self.server.block_ms}),
                }}],
            }
            finish = "tool_calls"
            self.server.assistant = message
        else:
            self.server.continued.set()
            self.server.release.wait(10)
            message, finish = {"role": "assistant", "content": "done"}, "stop"
        body = json.dumps({"choices": [{"message": message, "finish_reason": finish}], "usage": {"total_tokens": 100}}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except BrokenPipeError:
            pass

    def log_message(self, *args):
        pass


class RustAdapterTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.binary = Path(os.environ["RL_HARNESS_TEST_BINARY"]).resolve(strict=True)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.workspace = self.root / "workspace"
        self.workspace.mkdir()
        config = self.root / "config.json"
        config.write_text(json.dumps({"binary": str(self.binary), "verifier": f"{__name__}.verify", "model": "fixture-model"}))
        env = patch.dict(os.environ, {"RL_HARNESS_CONFIG": str(config)})
        env.start()
        self.addCleanup(env.stop)
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Model)
        self.server.requests = []
        self.server.command = "printf solved > answer.txt"
        self.server.block_ms = 1000
        self.server.continued = threading.Event()
        self.server.release = threading.Event()
        self.server.release.set()
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.release.set()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    async def attempt(self):
        return await miles_grok.run(
            f"http://127.0.0.1:{self.server.server_port}/sessions/fixture",
            [{"role": "user", "content": "Write solved to answer.txt."}],
            {"temperature": 0.5, "max_tokens": 64},
            {"workspace": str(self.workspace), "session_dir": str(self.root / "session")},
        )

    async def test_real_rust_tools_and_verifier(self):
        outcome = await self.attempt()
        self.assertEqual(outcome["rl_harness"]["status"], "graded", outcome)
        self.assertEqual(outcome["rl_harness"]["result"]["status"], "completed")
        self.assertEqual(await miles_grok.reward(None, SimpleNamespace(metadata=outcome)), 1.0)
        first, second = self.server.requests
        self.assertEqual(first["messages"], [
            {"role": "system", "content": "You are an autonomous coding agent. Your main goal is to complete the user's request denoted with the <user_query> tag."},
            {"role": "user", "content": "<user_query>\nWrite solved to answer.txt.\n</user_query>"},
        ])
        self.assertEqual(second["messages"][2], self.server.assistant)
        self.assertEqual(second["messages"][3]["tool_call_id"], "fixture-call")
        self.assertEqual(first["temperature"], 0.5)
        self.assertEqual(first["max_tokens"], 64)
        self.assertEqual({tool["function"]["name"] for tool in first["tools"]}, {
            "run_terminal_cmd", "get_terminal_command_output", "kill_terminal_command",
            "read_file", "search_replace", "list_dir", "grep", "todo_write",
        })

    async def test_cancellation_stops_native_background_command(self):
        self.server.command = "echo $$ > command.pid; sleep 60"
        self.server.block_ms = 0
        self.server.release.clear()
        task = asyncio.create_task(self.attempt())
        pid_path = self.workspace / "command.pid"
        try:
            for _ in range(500):
                if pid_path.exists() and self.server.continued.is_set():
                    break
                await asyncio.sleep(0.01)
            self.assertTrue(pid_path.exists())
            self.assertTrue(self.server.continued.is_set())
            pid = int(pid_path.read_text())
            os.kill(pid, 0)
        finally:
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        with self.assertRaises(ProcessLookupError):
            os.kill(pid, 0)


if __name__ == "__main__":
    unittest.main()
