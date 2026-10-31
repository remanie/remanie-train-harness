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
    (200, reply({"content": "hello.txt contains hello."}, "stop", 600, 10, 0)),
]


def chunks(completion):
    choice = completion["choices"][0]
    head = {"id": completion["id"], "object": "chat.completion.chunk", "created": 0, "model": completion["model"]}
    deltas = [
        {"tool_calls": [{"index": index, **call} for index, call in enumerate(value)]} if key == "tool_calls" else {key: value}
        for key, value in choice["message"].items()
        if key != "role" and value is not None
    ] or [{}]
    deltas[0] = {"role": "assistant", **deltas[0]}
    return [
        *({**head, "choices": [{"index": 0, "delta": delta, "finish_reason": None}]} for delta in deltas),
        {**head, "choices": [{"index": 0, "delta": {}, "finish_reason": choice["finish_reason"]}]},
        {**head, "choices": [], "usage": completion["usage"]},
    ]


def serve(answer):
    """Serve on a free port. answer(method, path, body) returns (status, payload, headers); a 200 to a request that sets stream goes out as chat.completion.chunk events."""
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_GET(self):
            self.respond(b"")

        def do_POST(self):
            self.respond(self.rfile.read(int(self.headers["Content-Length"])))

        def respond(self, body):
            with lock:
                status, payload, headers = answer(self.command, self.path, body)
            if status == 200 and body and json.loads(body).get("stream") is True:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                self.wfile.write(b"".join(b"data: %s\n\n" % json.dumps(chunk).encode() for chunk in chunks(payload)) + b"data: [DONE]\n\n")
                return
            data = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            for name, value in headers.items():
                self.send_header(name, value)
            self.end_headers()
            self.wfile.write(data)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def main():
    if sys.platform != "linux":
        sys.exit("Record the fixture on Linux, where CI checks it: the first user message names the OS. On a Mac: docker run --rm -v \"$PWD\":/w -w /w rust:1.94-slim-bookworm bash -c 'apt-get update -qq && apt-get install -y -qq ripgrep python3 >/dev/null && cargo build -q && python3 checks/make_trace_fixture.py target/debug/remanie-train-harness'")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runner", type=Path)
    parser.add_argument("--scratch", default="/tmp/remanie-trace-fixture", help="emptied, used for the run, then deleted")
    args = parser.parse_args()
    requests: list[bytes] = []

    def answer(_method, _path, body):
        requests.append(body)
        status, payload = SCRIPT[len(requests) - 1] if len(requests) <= len(SCRIPT) else (500, {"error": "script exhausted"})
        return status, payload, {"Retry-After": "0"} if status == 503 else {}

    server = serve(answer)
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
        assert result["compactions"] == 1 and result["tool_calls"]["ok"] == 3, result
        assert len(requests) == len(SCRIPT), len(requests)
        events = [json.loads(line) for line in (root / "session/trace.jsonl").read_bytes().splitlines()]
        compaction = next(event for event in events if event["event"] == "compaction")
        assert compaction["input"] == "fitted", compaction
        FIXTURES.mkdir(exist_ok=True)
        (FIXTURES / "trace.jsonl").write_bytes((root / "session/trace.jsonl").read_bytes())
        (FIXTURES / "requests.jsonl").write_bytes(b"".join(body + b"\n" for body in requests))
    finally:
        shutil.rmtree(root, ignore_errors=True)
    print(f"Wrote {FIXTURES / 'trace.jsonl'} and {FIXTURES / 'requests.jsonl'}")


if __name__ == "__main__":
    main()
