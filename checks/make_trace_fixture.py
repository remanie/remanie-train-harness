"""Run the runner once against a scripted endpoint and save its trace and the request bodies it sent."""

import argparse
import json
import os
import sys
import shutil
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

FIXTURES = Path(__file__).resolve().parents[1] / "checks/fixtures"


if sys.platform != "linux":
    sys.exit("Record the fixture on Linux, where CI checks it: the first user message names the OS. On a Mac: docker run --rm -v \"$PWD\":/w -w /w rust:1.94-slim-bookworm bash -c 'apt-get update -qq && apt-get install -y -qq ripgrep python3 >/dev/null && cargo build -q && python3 checks/make_trace_fixture.py target/debug/remanie-train-harness'")

def reply(message, finish, prompt_tokens, completion_tokens, cached_tokens):
    return {
        "id": "chatcmpl-fixture",
        "object": "chat.completion",
        "model": "fixture-model",
        "choices": [{"index": 0, "message": {"role": "assistant", **message}, "finish_reason": finish}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "prompt_tokens_details": {"cached_tokens": cached_tokens},
        },
    }


def tool_call(id, name, arguments):
    return {"id": id, "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}


SCRIPT = [
    (503, {"error": "scripted overload"}),
    (200, reply({
        "content": None,
        "reasoning_content": "Write the file and check that the shell counts.",
        "tool_calls": [
            tool_call("call-write", "search_replace", {"file_path": "hello.txt", "old_string": "", "new_string": "hello\n"}),
            tool_call("call-count", "run_terminal_cmd", {"command": "seq 1 500", "description": "Check that the shell runs."}),
        ],
    }, "tool_calls", 9000, 40, 0)),
    (200, reply({
        "content": None,
        "reasoning_content": "Read the file back.",
        "tool_calls": [tool_call("call-read", "read_file", {"target_file": "hello.txt"})],
    }, "tool_calls", 18000, 40, 0)),
    (200, reply({"content": "<summary>Wrote hello.txt and read it back.</summary>"}, "stop", 18300, 20, 8000)),
    (200, reply({
        "content": None,
        "reasoning_content": "Print two long logs.",
        "tool_calls": [
            tool_call("call-log-1", "run_terminal_cmd", {"command": "seq 1 3800", "description": "Print the first log."}),
            tool_call("call-log-2", "run_terminal_cmd", {"command": "seq 3801 7000", "description": "Print the second log."}),
        ],
    }, "tool_calls", 8000, 40, 0)),
    (200, reply({"content": "<summary>Wrote hello.txt, read it back, and printed two logs.</summary>"}, "stop", 9100, 20, 0)),
    (200, reply({"content": "hello.txt contains hello."}, "stop", 600, 10, 0)),
]


def serve(requests):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):
            requests.append(self.rfile.read(int(self.headers["Content-Length"])))
            status, body = SCRIPT[len(requests) - 1] if len(requests) <= len(SCRIPT) else (500, {"error": "script exhausted"})
            payload = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            if status == 503:
                self.send_header("Retry-After", "0")
            self.end_headers()
            self.wfile.write(payload)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runner", type=Path)
    parser.add_argument("--scratch", default="/tmp/remanie-trace-fixture", help="emptied, used for the run, then deleted")
    args = parser.parse_args()
    requests: list[bytes] = []
    server = serve(requests)
    root = Path(args.scratch)
    shutil.rmtree(root, ignore_errors=True)
    (root / "workspace").mkdir(parents=True)
    try:
        job = {
            "base_url": f"http://127.0.0.1:{server.server_port}/v1",
            "model": "fixture-model",
            "user_query": "Create hello.txt containing hello and check it.",
            "workspace": str(root / "workspace"),
            "session_dir": str(root / "session"),
            "sampling_params": {"max_tokens": 512, "temperature": 1.0},
            "max_turns": 4,
            "deadline_secs": 60,
            "context_window_tokens": 20000,
            "compaction": {"max_tokens": 256},
        }
        run = subprocess.run([str(args.runner.resolve())], input=json.dumps(job).encode(), capture_output=True, timeout=120, env={**os.environ, "SOURCE_DATE_EPOCH": "0", "SHELL": "/bin/bash"})
        server.shutdown()
        result = json.loads(run.stdout)
        assert run.returncode == 0 and result["outcome"] == "completed", (run.returncode, run.stdout, run.stderr)
        assert result["compactions"] == 2 and result["tool_calls"]["ok"] == 5, result
        assert len(requests) == len(SCRIPT), len(requests)
        events = [json.loads(line) for line in (root / "session/trace.jsonl").read_bytes().splitlines()]
        inputs = [event["input"] for event in events if event["event"] == "compaction"]
        assert inputs == ["fitted", "lossy"], inputs
        FIXTURES.mkdir(exist_ok=True)
        (FIXTURES / "trace.jsonl").write_bytes((root / "session/trace.jsonl").read_bytes())
        (FIXTURES / "requests.jsonl").write_bytes(b"".join(body + b"\n" for body in requests))
    finally:
        shutil.rmtree(root, ignore_errors=True)
    print(f"Wrote {FIXTURES / 'trace.jsonl'} and {FIXTURES / 'requests.jsonl'}")


if __name__ == "__main__":
    main()
