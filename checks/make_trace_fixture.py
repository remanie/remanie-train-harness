"""Run the trial runner once on Grok Build against a scripted endpoint and save its trace and the request bodies the endpoint received."""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

FIXTURES = Path(__file__).resolve().parents[1] / "checks/fixtures"
MODEL = "fixture-model"
SUMMARY_PROMPT = "Your task is to produce a faithful, concise summary of the conversation so far"
FIXED_DATE = "2026-01-01"
DATE = re.compile(r"(?<=Today's date: )\d{4}-\d{2}-\d{2}")


def reply(message, finish, prompt_tokens, completion_tokens, cached_tokens):
    return {
        "id": "chatcmpl-fixture",
        "object": "chat.completion",
        "model": MODEL,
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
                status, payload, headers = answer(self.command, self.path, body, self.headers)
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


def kind(body):
    """The rule the runner's proxy also uses to tell turn, compaction and title requests apart."""
    if any(tool.get("function", {}).get("name") == "session_title" for tool in body.get("tools", [])):
        return "title"
    last = body["messages"][-1]
    return "compaction" if last["role"] == "user" and SUMMARY_PROMPT in (last.get("content") or "") else "turn"


def scripted(entry):
    if "status" in entry:
        return entry["status"], entry["body"], entry.get("headers", {})
    message = dict(entry["message"])
    if "tool_calls" in message:
        message["tool_calls"] = [tool_call(call["id"], call["name"], call["arguments"]) for call in message["tool_calls"]]
    return 200, reply(message, entry["finish_reason"], *entry["usage"]), {}


TITLE_REPLY = (200, reply({"content": None, "tool_calls": [tool_call("call-title", "session_title", {"session_title": "Scripted title"})]}, "tool_calls", 50, 10, 0), {})


class Endpoint:
    """Answers each kind of chat-completions request from its own queue, repeating the last entry once the queue runs down."""

    def __init__(self, replies, window=None):
        self.queues = {name: [scripted(entry) for entry in entries] for name, entries in replies.items()}
        self.window = window
        self.requests = []
        self.titles = []
        self.authorizations = []
        self.server = serve(self.answer)
        self.url = f"http://127.0.0.1:{self.server.server_port}/v1"

    def answer(self, method, path, body, headers):
        if method == "GET":
            if path == "/v1/models":
                model = {"id": MODEL, "object": "model", "created": 0, "owned_by": "fixture"}
                if self.window:
                    model["context_length"] = self.window
                return 200, {"object": "list", "data": [model]}, {}
            return 404, {"error": "not found"}, {}
        if path != "/v1/chat/completions":
            self.requests.append(("other " + path, body))
            return 404, {"error": "not found"}, {}
        request_kind = kind(json.loads(body))
        if request_kind == "title":
            self.titles.append(body)
            return TITLE_REPLY
        self.requests.append((request_kind, body))
        self.authorizations.append(headers.get("Authorization"))
        queue = self.queues.get(request_kind, [])
        if not queue:
            return 500, {"error": f"no {request_kind} reply is scripted"}, {}
        return queue.pop(0) if len(queue) > 1 else queue[0]


SUMMARY = "<summary>\n1. Primary Request and Intent: The user asked to create hello.txt containing hello and then check it.\n2. Key Technical Concepts: shell commands through run_terminal_cmd and file reads through read_file, in the workspace.\n3. Files and Code Sections: hello.txt now holds one line, hello, written with printf.\n4. Errors and Fixes: None. The first request was refused once with a 503 and succeeded on the retry.\n5. Problem Solving: The file was written, its line count checked with wc, and its text read back.\n6. All User Messages: \"Create hello.txt containing hello and check it.\"\n7. Pending Tasks: Report the result to the user.\n8. Current Work: Both checks finished; nothing else is running.\n9. Optional Next Step: Tell the user that hello.txt contains hello.\n</summary>"
REPLIES = {
    "turn": [
        {"status": 503, "body": {"error": "scripted overload"}, "headers": {"Retry-After": "0"}},
        {"message": {"content": None, "reasoning_content": "Write the file first.", "tool_calls": [
            {"id": "call-write", "name": "run_terminal_cmd", "arguments": {"command": "printf 'hello\\n' > hello.txt && seq 1 3", "description": "Write hello.txt."}},
        ]}, "finish_reason": "tool_calls", "usage": [60000, 40, 0]},
        {"message": {"content": None, "reasoning_content": "Check it two ways at once.", "tool_calls": [
            {"id": "call-count", "name": "run_terminal_cmd", "arguments": {"command": "sleep 1; wc -l < hello.txt", "description": "Count the lines."}},
            {"id": "call-read", "name": "read_file", "arguments": {"target_file": "hello.txt"}},
        ]}, "finish_reason": "tool_calls", "usage": [120000, 60, 59000]},
        {"message": {"content": "hello.txt contains hello."}, "finish_reason": "stop", "usage": [3000, 10, 0]},
    ],
    "compaction": [
        {"message": {"content": SUMMARY}, "finish_reason": "stop", "usage": [119000, 160, 0]},
    ],
}
JOB = {
    "model": MODEL,
    "user_query": "Create hello.txt containing hello and check it.",
    "sampling_params": {"max_tokens": 512, "temperature": 1.0, "provider": {"order": ["fixture"]}},
    "max_turns": 6,
    "deadline_secs": 120,
    "context_window_tokens": 131072,
}


def fixed_date(text):
    return DATE.sub(FIXED_DATE, text)


def main():
    if sys.platform != "linux":
        sys.exit("Record the fixture on Linux, where CI checks it: the first user message names the OS. checks/README.txt has the Docker recipe for a Mac.")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runner", type=Path)
    parser.add_argument("--grok", type=Path, required=True, help="xai-grok-pager built at upstream.json's commit")
    parser.add_argument("--scratch", default="/tmp/remanie-trace-fixture", help="emptied, used for the run, then deleted")
    args = parser.parse_args()
    endpoint = Endpoint(REPLIES, JOB["context_window_tokens"])
    root = Path(args.scratch)
    shutil.rmtree(root, ignore_errors=True)
    (root / "workspace").mkdir(parents=True)
    try:
        job = {**JOB, "base_url": endpoint.url, "api_key_env": "FIXTURE_KEY", "workspace": str(root / "workspace"), "session_dir": str(root / "session")}
        env = {**os.environ, "SHELL": "/bin/bash", "FIXTURE_KEY": "fixture-key", "REMANIE_GROK": str(args.grok.resolve())}
        run = subprocess.run([str(args.runner.resolve())], input=json.dumps(job).encode(), capture_output=True, timeout=300, env=env)
        endpoint.server.shutdown()
        result = json.loads(run.stdout)
        kinds = [request_kind for request_kind, _ in endpoint.requests]
        assert run.returncode == 0 and result["outcome"] == "completed", (run.returncode, run.stdout, run.stderr[-2000:])
        assert kinds.count("compaction") >= 1 and kinds[:2] == ["turn", "turn"] and not endpoint.titles, kinds
        assert set(endpoint.authorizations) == {"Bearer fixture-key"}, endpoint.authorizations
        trace = (root / "session/trace.jsonl").read_text()
        events = [json.loads(line) for line in trace.splitlines()]
        assert [event["kind"] for event in events if event["event"] == "request"].count("title") == 1, "the title request is traced"
        FIXTURES.mkdir(exist_ok=True)
        (FIXTURES / "trace.jsonl").write_text(fixed_date(trace))
        (FIXTURES / "requests.jsonl").write_text("".join(fixed_date(body.decode()) + "\n" for _, body in endpoint.requests))
    finally:
        shutil.rmtree(root, ignore_errors=True)
    print(f"Wrote {FIXTURES / 'trace.jsonl'} and {FIXTURES / 'requests.jsonl'}: {', '.join(kinds)}")


if __name__ == "__main__":
    main()
