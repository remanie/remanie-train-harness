"""Run real Harbor/Docker trials against a local scripted model; no paid API calls."""

import argparse
import asyncio
import json
import os
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

CRATE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CRATE / "harbor"))

from harbor.models.trial.config import (
    AgentConfig,
    EnvironmentConfig,
    TaskConfig,
    TrialConfig,
    VerifierConfig,
)
from harbor.trial.trial import Trial
from remanie_agent import RemanieAgent

FAKE_KEY = "remanie-local-fixture-key"
CATALOG = json.loads((CRATE / "assets/tools.json").read_text())


class UnconfirmedShutdownAgent(RemanieAgent):
    async def _stop_runtime(self, environment, process, execution):
        await super()._stop_runtime(environment, None, execution)


def call(name, arguments, id):
    return {
        "id": id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


def shell(command, id, wait=1000):
    return call("run_terminal_cmd", {
        "command": command,
        "description": "Exercise the local integration fixture",
        "block_until_ms": wait,
    }, id)


def reply(content=None, calls=None, tokens=100):
    message = {"role": "assistant", "content": content}
    if calls:
        message["tool_calls"] = calls
    return {
        "choices": [{"message": message, "finish_reason": "tool_calls" if calls else "stop"}],
        "usage": {"prompt_tokens": tokens, "completion_tokens": 10, "total_tokens": tokens + 10, "cost": 0},
    }


class ScriptedModel:
    def __init__(self, mode):
        self.mode = mode
        self.requests = []
        self.errors = []
        self.turn = 0
        self.compactions = 0
        self.stop = threading.Event()

    def sample(self, request):
        self.requests.append(request)
        assert request["model"] == "x-ai/grok-4.7"
        assert request["reasoning"] == {"effort": "high"}
        advertised = {tool["function"]["name"]: {
            "description": tool["function"]["description"],
            "parameters": tool["function"]["parameters"],
        } for tool in request["tools"]}
        assert advertised == CATALOG
        assert FAKE_KEY not in json.dumps(request)
        messages = request["messages"]
        assert messages[:2] == self.requests[0]["messages"][:2]
        if request["tool_choice"] == "none":
            assert self.mode == "completed" and self.turn == 1
            self.compactions += 1
            return reply("<summary>Created result.txt containing before. The verification todo is in progress and a background task remains running.</summary>")

        self.turn += 1
        if self.mode == "crash":
            assert self.turn == 1
            return reply(calls=[shell("printf '%s' $$ > crash.pid; kill -KILL \"$PPID\"; exec sleep 300", "crash-process", wait=30000)])
        if self.mode == "timeout":
            if self.turn == 1:
                return reply(calls=[shell("printf '1\\n' > /logs/agent/runtime.pid; printf '%s' $$ > timeout.pid; exec sleep 300", "timeout-process")])
            assert self.turn == 2
            self.stop.wait(60)
            return reply("The outer timeout should already have cancelled this request.")

        if self.turn == 1:
            return reply(calls=[
                call("todo_write", {"todos": [{"id": "verify", "content": "Verify result.txt", "status": "in_progress"}]}, "todo-start"),
                shell("printf 'before\\n' > result.txt; env > child-env.txt; cat /proc/$PPID/environ > parent-env.bin", "initialize"),
                shell("printf '%s' $$ > background.pid; printf background-ready; exec sleep 300", "background"),
            ], tokens=30_000)
        if self.turn == 2:
            assert self.compactions == 1 and len(messages) == 3
            state = json.loads(messages[2]["content"].split("Current tool state:\n", 1)[1])
            todo = state["todos"]["todos"]["verify"]
            assert todo["content"] == "Verify result.txt" and todo["status"] == "in_progress"
            tasks = state["background_tasks"]
            assert len(tasks) == 1 and not tasks[0]["completed"]
            task_id = tasks[0]["task_id"]
            return reply(calls=[
                call("get_terminal_command_output", {"task_ids": [task_id], "timeout_ms": 0}, "poll"),
                call("read_file", {"target_file": "result.txt"}, "read"),
                call("search_replace", {"file_path": "result.txt", "old_string": "before", "new_string": "after"}, "edit"),
                call("grep", {"pattern": "after", "path": "result.txt"}, "search"),
                call("list_dir", {"target_directory": "."}, "list"),
                call("kill_terminal_command", {"task_id": task_id}, "kill"),
                call("todo_write", {"todos": [{"id": "verify", "status": "completed"}]}, "todo-finish"),
                shell("printf '%s' $$ > cleanup.pid; exec sleep 300", "cleanup-process"),
            ])
        assert self.turn == 3
        outputs = {m["tool_call_id"]: m["content"] for m in messages if m["role"] == "tool"}
        assert len(outputs) == 8 and not any(value.startswith("Tool error:") for value in outputs.values())
        assert "before" in outputs["read"] and "after" in outputs["search"]
        assert "result.txt" in outputs["list"]
        assert "completed" in outputs["todo-finish"] and "Verify result.txt" in outputs["todo-finish"]
        return reply("Completed the local integration fixture.")


def model_server(model):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):
            try:
                assert self.path == "/v1/chat/completions"
                assert self.headers.get("Authorization") == f"Bearer {FAKE_KEY}"
                size = int(self.headers["Content-Length"])
                assert 0 < size <= 16 * 1024 * 1024
                response = model.sample(json.loads(self.rfile.read(size)))
                status = 200
            except Exception as error:
                model.errors.append(repr(error))
                response, status = {"error": "scripted fixture assertion failed"}, 500
            body = json.dumps(response).encode()
            try:
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


async def check(mode, runtime_dir, output_dir):
    model = ScriptedModel("completed" if mode == "cleanup_error" else mode)
    server, thread = model_server(model)
    try:
        config = TrialConfig(
            task=TaskConfig(path=CRATE / "checks/fixture"),
            trial_name=mode,
            trials_dir=output_dir,
            agent=AgentConfig(
                import_path="check_harbor:UnconfirmedShutdownAgent" if mode == "cleanup_error" else "remanie_agent:RemanieAgent",
                model_name="openrouter/x-ai/grok-4.7",
                base_url=f"http://127.0.0.1:{server.server_port}/v1",
                env={"OPENROUTER_API_KEY": FAKE_KEY},
                override_timeout_sec=8 if mode == "timeout" else 60,
                override_setup_timeout_sec=120,
                kwargs={
                    "runtime_dir": str(runtime_dir), "max_turns": 4,
                    "max_response_tokens": 2048, "deadline_secs": 120,
                    "compaction_threshold_tokens": 20_000,
                    "sampling_params": {"reasoning": {"effort": "high"}},
                },
            ),
            environment=EnvironmentConfig(type="docker", delete=True),
            verifier=VerifierConfig(env={"REMANIE_CHECK_MODE": mode}),
        )
        trial = await Trial.create(config)
        result = await trial.run()
    finally:
        model.stop.set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
        (output_dir / f"{mode}-model-requests.json").write_text(json.dumps(model.requests, indent=2) + "\n")
    assert not model.errors, model.errors
    if mode in ("cleanup_error", "crash"):
        assert result.exception_info and result.exception_info.exception_type == "RuntimeError"
        assert result.verifier_result is None and result.verifier is None
        assert not (output_dir / mode / "verifier/reward.txt").exists()
        return {"mode": mode, "reward": None, "verification_skipped": True}
    assert result.verifier_result and result.verifier_result.rewards == {"reward": 1.0}, (
        f"Harbor verifier did not pass; inspect {output_dir / mode}. "
        f"Error: {result.exception_info.exception_type if result.exception_info else 'none'}"
    )
    if mode == "timeout":
        assert result.exception_info and result.exception_info.exception_type == "AgentTimeoutError"
        assert model.turn == 2 and model.compactions == 0
    else:
        assert result.exception_info is None, result.exception_info
        assert model.turn == 3 and model.compactions == 1
    log_dir = output_dir / mode / "agent"
    runtime_result = json.loads((log_dir / "result.json").read_text())
    assert runtime_result["status"] == ("cancelled" if mode == "timeout" else "completed"), runtime_result
    assert runtime_result["compactions"] == model.compactions
    trace = (log_dir / "remanie/trace.jsonl").read_text()
    assert FAKE_KEY not in trace and FAKE_KEY not in (log_dir / "input.json").read_text()
    return {"mode": mode, "reward": 1.0, "status": runtime_result["status"], "compactions": model.compactions, "model_requests": len(model.requests)}


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-dir", type=Path, default=CRATE / "runtime")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output = (args.output_dir or Path(tempfile.mkdtemp(prefix="remanie-harbor-check-"))).resolve()
    output.mkdir(parents=True, exist_ok=True)
    modes = ("completed", "timeout", "cleanup_error", "crash")
    assert not any((output / mode).exists() for mode in modes), "Use a fresh output directory"
    os.environ["OPENROUTER_API_KEY"] = FAKE_KEY
    print(f"Harbor integration logs: {output}", flush=True)
    checks = [await check(mode, args.runtime_dir.resolve(), output) for mode in modes]
    summary = {"fixture_only": True, "paid_api_calls": 0, "checks": checks}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
