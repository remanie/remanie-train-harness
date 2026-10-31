"""Run Grok Build and the runner on each scenario against a scripted endpoint, and diff the request bodies they send."""

import argparse
import difflib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import uuid
from pathlib import Path

ORACLE = Path(__file__).resolve().parent
REPO = ORACLE.parents[1]
sys.path.insert(0, str(REPO / "checks"))
from make_trace_fixture import reply, serve, tool_call

MODEL = "fixture-model"
KEY_ENV = "ORACLE_API_KEY"
KEY = "oracle-fake-key"
RUNNER_COMPACT_AT_PERCENT = 85
SUMMARY_PROMPT = "Your task is to produce a faithful, concise summary of the conversation so far"
TITLE_REPLY = reply({"content": None, "tool_calls": [tool_call("call-title", "session_title", {"session_title": "Oracle scenario"})]}, "tool_calls", 50, 10, 0)
GROK_SAMPLING = {"max_tokens": "max_completion_tokens", "temperature": "temperature", "top_p": "top_p"}
GROK_URLS = ["GROK_CLI_CHAT_PROXY_BASE_URL", "GROK_XAI_API_BASE_URL", "GROK_MODELS_BASE_URL", "GROK_FEEDBACK_BASE_URL", "GROK_TRACE_UPLOAD_URL", "GROK_MANAGED_CONFIG_URL", "GROK_CODE_WEB_URL", "GROK_CONVERSATIONS_BASE_URL"]
GROK_ENV = {
    "GROK_TELEMETRY_ENABLED": "false",
    "GROK_TELEMETRY_MIXPANEL_ENABLED": "false",
    "GROK_TELEMETRY_MIXPANEL_TOKEN": "",
    "GROK_TELEMETRY_EVENTS_URL": "",
    "GROK_TELEMETRY_EVENTS_API_KEY": "",
    "GROK_TELEMETRY_TRACE_UPLOAD": "false",
    "GROK_FEEDBACK_ENABLED": "false",
    "GROK_TRACE_UPLOAD": "false",
    "GROK_INSTRUMENTATION": "disabled",
    "OTEL_SDK_DISABLED": "true",
    "DISABLE_TELEMETRY": "1",
    "DISABLE_FEEDBACK_COMMAND": "1",
    "GROK_DISABLE_AUTOUPDATER": "1",
    "GROK_PROMPT_SUGGESTIONS": "false",
    "GROK_AGENT_ID": "oracle",
    "GROK_TURN_SUMMARY": "0",
    "GROK_MEMORY": "0",
    "GROK_WORKFLOWS": "0",
    "NO_PROXY": "127.0.0.1,localhost",
    "no_proxy": "127.0.0.1,localhost",
    "RUST_LOG": "info",
}
DATE = re.compile(r"(?<=Today's date: )\d{4}-\d{2}-\d{2}")
RULES = re.compile(r"\n\n<rules>\n.*\n</rules>\Z", re.S)
REMINDER = r"<system-reminder>\n.*?\n</system-reminder>"
WHOLE_REMINDER = re.compile(rf"\A\s*{REMINDER}\s*\Z", re.S)
EDGE_REMINDER = re.compile(rf"\A\s*{REMINDER}\n+|\n+{REMINDER}\s*\Z", re.S)
MISSING = object()


def kind(body):
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


class Endpoint:
    def __init__(self, scenario):
        self.queues = {name: [scripted(entry) for entry in entries] for name, entries in scenario["replies"].items()}
        self.window = scenario["job"].get("context_window_tokens")
        self.requests = []
        self.titles = 0
        self.server = serve(self.answer)
        self.url = f"http://127.0.0.1:{self.server.server_port}/v1"

    def answer(self, method, path, body):
        if method == "GET":
            if path == "/v1/models":
                model = {"id": MODEL, "object": "model", "created": 0, "owned_by": "oracle"}
                if self.window:
                    model["context_length"] = self.window
                return 200, {"object": "list", "data": [model]}, {}
            return 404, {"error": "not found"}, {}
        if path != "/v1/chat/completions":
            self.requests.append(("other " + path, body.decode()))
            return 404, {"error": "not found"}, {}
        request_kind = kind(json.loads(body))
        if request_kind == "title":
            self.titles += 1
            return 200, TITLE_REPLY, {}
        self.requests.append((request_kind, body.decode()))
        queue = self.queues.get(request_kind, [])
        if not queue:
            return 500, {"error": f"no {request_kind} reply is scripted"}, {}
        return queue.pop(0) if len(queue) > 1 else queue[0]


def file_text(spec):
    if isinstance(spec, str):
        return spec
    return "".join(spec["line"].format(n=n) for n in range(1, spec["count"] + 1))


def base_env(live):
    return {"PATH": os.environ["PATH"], "HOME": str(live / "home"), "TMPDIR": str(live / "tmp"), "SHELL": "/bin/bash", "USER": "oracle", "LANG": "en_US.UTF-8", "TZ": "UTC"}


def execute(command, cwd, env, stdin, timeout):
    process = subprocess.Popen(command, cwd=cwd, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
    try:
        stdout, stderr = process.communicate(stdin, timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        stdout, stderr = process.communicate()
        return f"timed out after {timeout} s", stdout, stderr
    return f"exit {process.returncode}", stdout, stderr


def grok_config(scenario, url):
    job = scenario["job"]
    model = {"name": MODEL, "model": MODEL, "base_url": url, "env_key": KEY_ENV, "api_backend": "chat_completions"}
    if "context_window_tokens" in job:
        model["context_window"] = job["context_window_tokens"]
    for key, value in job["sampling_params"].items():
        if key not in GROK_SAMPLING:
            sys.exit(f"{scenario['name']}: sampling_params.{key} has no Grok Build setting in the oracle")
        model[GROK_SAMPLING[key]] = value
    if "threshold_tokens" in (job.get("compaction") or {}):
        sys.exit(f"{scenario['name']}: compaction.threshold_tokens has no Grok Build setting in the oracle")
    lines = [
        "[cli]", "auto_update = false", "",
        "[features]", "telemetry = false", "",
        "[harness]", "disable_codebase_upload = true", "",
        "[session]", f"auto_compact_threshold_percent = {RUNNER_COMPACT_AT_PERCENT}", "",
        "[models]", *(f"{role} = {json.dumps(MODEL)}" for role in ("default", "session_summary", "image_description", "web_search")), "",
        f"[model.{json.dumps(MODEL)}]", *(f"{key} = {json.dumps(value)}" for key, value in model.items()),
    ]
    return "\n".join(lines) + "\n"


def grok_agent():
    tools = json.loads((REPO / "assets/tools.json").read_text())
    lines = ["---", "name: oracle", "description: Grok Build with the runner's tools.", "toolConfig:", "  tools:"]
    lines += [f"    - id: GrokBuild:{name}" for name in tools]
    lines += ["injectDefaultTools: false", "agentsMd: false", "discoverSkills: false", "inheritSkills: false", "---", ""]
    return "\n".join(lines)


def run_grok(grok, scenario, endpoint, live):
    (live / "home/.grok").mkdir(parents=True)
    (live / "home/.grok/config.toml").write_text(grok_config(scenario, endpoint.url))
    (live / "oracle-agent.md").write_text(grok_agent())
    job = scenario["job"]
    workspace = live / "workspace"
    command = [
        str(grok), "-p", scenario["user_query"], "-m", MODEL, "--cwd", str(workspace),
        "--output-format", "json", "--always-approve", "--no-auto-update",
        "--agent", str(live / "oracle-agent.md"), "--max-turns", str(job["max_turns"]),
        "--session-id", str(uuid.uuid5(uuid.NAMESPACE_URL, "remanie-oracle/" + scenario["name"])),
    ]
    env = {
        **base_env(live), **dict.fromkeys(GROK_URLS, endpoint.url), **GROK_ENV,
        "GROK_HOME": str(live / "home/.grok"), "GROK_LOG_FILE": str(live / "grok.log"), KEY_ENV: KEY, "XAI_API_KEY": KEY,
    }
    code, stdout, _ = execute(command, workspace, env, None, job["deadline_secs"] + 60)
    try:
        result = json.loads(stdout)
    except ValueError:
        result = {}
    detail = result.get("stopReason") or " ".join(result.get("message", "no JSON on stdout").split())[:300]
    return f"{code}, {detail}"


def run_runner(runner, scenario, endpoint, live):
    workspace = live / "workspace"
    job = {"base_url": endpoint.url, "model": MODEL, "api_key_env": KEY_ENV, "user_query": scenario["user_query"], "workspace": str(workspace), "session_dir": str(live / "session"), **scenario["job"]}
    code, stdout, _ = execute([str(runner)], workspace, {**base_env(live), KEY_ENV: KEY}, json.dumps(job).encode(), job["deadline_secs"] + 60)
    try:
        result = json.loads(stdout)
    except ValueError:
        return f"{code}, no JSON on stdout"
    trace = live / "session/trace.jsonl"
    events = [json.loads(line) for line in trace.read_text().splitlines()] if trace.exists() else []
    inputs = [event["input"] for event in events if event["event"] == "compaction"]
    cause = f" ({result['cause']['kind']})" if result.get("cause", {}).get("kind") else ""
    return f"{code}, {result['outcome']}{cause}" + (f", compaction input {', '.join(inputs)}" if inputs else "")


AGENTS = {"grok": run_grok, "runner": run_runner}


def run(agent, binary, scenario, scratch):
    live = scratch / "live"
    shutil.rmtree(live, ignore_errors=True)
    for directory in ("workspace", "home", "tmp"):
        (live / directory).mkdir(parents=True)
    for name, spec in scenario.get("files", {}).items():
        (live / "workspace" / name).write_text(file_text(spec))
    endpoint = Endpoint(scenario)
    try:
        outcome = AGENTS[agent](binary, scenario, endpoint, live)
    finally:
        endpoint.server.shutdown()
    kept = scratch / "runs" / scenario["name"] / agent
    kept.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(live, kept)
    return outcome, endpoint


def strip_injected(body):
    notes = []
    messages = []
    for index, message in enumerate(body["messages"]):
        content = message.get("content")
        if isinstance(content, str):
            if message.get("role") == "user" and WHOLE_REMINDER.match(content):
                notes.append(f"dropped message {index}, a whole <system-reminder> ({len(content)} bytes)")
                continue
            if message.get("role") == "user" and RULES.search(content):
                notes.append(f"cut the <rules> block from message {index} ({len(RULES.search(content).group())} bytes)")
                content = RULES.sub("", content)
            while EDGE_REMINDER.search(content):
                notes.append(f"cut a <system-reminder> from message {index} ({len(EDGE_REMINDER.search(content).group())} bytes)")
                content = EDGE_REMINDER.sub("", content, count=1)
            if re.search(REMINDER, content, re.S):
                notes.append(f"left a <system-reminder> inside message {index}")
            message = {**message, "content": content}
        messages.append(message)
    return {**body, "messages": messages}, notes


def unname(value):
    if isinstance(value, dict):
        return {key: unname(item) for key, item in value.items()}
    if isinstance(value, list):
        return [unname(item) for item in value]
    return "<model>" if value == MODEL else value


def normalize(agent, raw):
    body = unname(json.loads(raw))
    if "tools" in body:
        body["tools"] = "<tools>"
    body["messages"] = [{**message, "content": DATE.sub("<date>", message["content"])} if isinstance(message.get("content"), str) else message for message in body["messages"]]
    return strip_injected(body) if agent == "grok" else (body, [])


def flat(value, path=""):
    if isinstance(value, dict):
        if not value:
            yield f"{path}: {{}}"
        for key in sorted(value):
            yield from flat(value[key], f"{path}.{key}" if path else key)
    elif isinstance(value, list):
        if not value:
            yield f"{path}: []"
        for index, item in enumerate(value):
            yield from flat(item, f"{path}[{index}]")
    elif isinstance(value, str) and "\n" in value:
        yield f"{path}: |"
        yield from ("  " + line for line in value.split("\n"))
    else:
        yield f"{path}: {json.dumps(value, ensure_ascii=False)}"


def key_order(a, b, path=""):
    if isinstance(a, dict) and isinstance(b, dict):
        shared = [key for key in a if key in b]
        if shared != [key for key in b if key in a]:
            yield re.sub(r"\[\d+\]", "[*]", path) or "(top level)"
        for key in shared:
            yield from key_order(a[key], b[key], f"{path}.{key}" if path else key)
    elif isinstance(a, list) and isinstance(b, list):
        for index, (x, y) in enumerate(zip(a, b)):
            yield from key_order(x, y, f"{path}[{index}]")


def spans(indices):
    out = []
    for index in indices:
        if out and out[-1][1] == index - 1:
            out[-1][1] = index
        else:
            out.append([index, index])
    return ", ".join(str(a) if a == b else f"{a}-{b}" for a, b in out)


def differs_in(a, b):
    parts = []
    for key in sorted(a.keys() | b.keys()):
        if a.get(key, MISSING) == b.get(key, MISSING):
            continue
        if key == "messages" and isinstance(a.get(key), list) and isinstance(b.get(key), list):
            ma, mb = a["messages"], b["messages"]
            changed = [i for i in range(max(len(ma), len(mb))) if i >= len(ma) or i >= len(mb) or ma[i] != mb[i]]
            parts.append(f"messages [{spans(changed)}] of {len(ma)} and {len(mb)}")
        else:
            parts.append(key)
    return "; ".join(parts)


def compare(name, outcomes, pairs):
    grok, runner = pairs["grok"], pairs["runner"]
    rows, diffs, first = [], [], None
    for index in range(max(len(grok), len(runner))):
        g = grok[index] if index < len(grok) else None
        r = runner[index] if index < len(runner) else None
        same = g is not None and r is not None and g[0] == r[0] and g[1] == r[1]
        if same:
            rows.append(f"{index + 1:>4}  {g[0]:<11} {r[0]:<11} same")
            continue
        detail = differs_in(g[1], r[1]) if g and r else ("runner sent no request" if g else "grok sent no request")
        rows.append(f"{index + 1:>4}  {g[0] if g else '-':<11} {r[0] if r else '-':<11} {detail}")
        diff = difflib.unified_diff(
            list(flat(g[1])) if g else [], list(flat(r[1])) if r else [],
            f"grok request {index + 1} ({g[0]})" if g else "grok (none)", f"runner request {index + 1} ({r[0]})" if r else "runner (none)", lineterm="")
        diffs.append("\n".join(diff))
        first = first or (index + 1, g[0] if g else r[0])
    orders = sorted({path for g, r in zip(grok, runner) for path in key_order(g[1], r[1])})
    notes = sorted({note for _, _, request_notes in grok for note in request_notes})
    verdict = "identical" if first is None else f"first difference at request {first[0]} ({first[1]})"
    lines = [f"== {name}: {verdict}"]
    lines += [f"{agent}: {outcomes[agent]}; {len(pairs[agent])} requests compared" for agent in AGENTS]
    lines += [f"normalized grok: {note}" for note in notes]
    if orders:
        lines.append("key order differs at: " + ", ".join(orders))
    lines += ["", "   #  grok        runner      differs in", *rows]
    report = "\n".join(lines)
    return verdict, report, diffs


def grok_version(grok, scratch):
    live = scratch / "version"
    for directory in ("home/.grok", "tmp"):
        (live / directory).mkdir(parents=True, exist_ok=True)
    code, stdout, _ = execute([str(grok), "--version"], scratch, {**base_env(live), **GROK_ENV, "GROK_HOME": str(live / "home/.grok")}, None, 60)
    return stdout.decode().strip() if code == "exit 0" else f"unknown ({code})"


def runner_source():
    out = subprocess.run(["git", "log", "-1", "--format=%h %s", "--", "src", "assets", "Cargo.toml", "Cargo.lock"], cwd=REPO, capture_output=True, text=True)
    return out.stdout.strip() or "unknown"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grok", type=Path, required=True, help="the Grok Build binary (xai-grok-pager) built at NOTICE's commit")
    parser.add_argument("--runner", type=Path, required=True, help="the runner binary")
    parser.add_argument("--scenario", action="append", help="run only this scenario; repeatable")
    parser.add_argument("--out", type=Path, help="write report.txt and each scenario's normalized requests and diffs here")
    parser.add_argument("--scratch", type=Path, default=Path("/tmp/remanie-oracle"), help="emptied, then holds each run under runs/<scenario>/<agent>")
    args = parser.parse_args()
    names = args.scenario or sorted(path.stem for path in (ORACLE / "scenarios").glob("*.json"))
    scratch = args.scratch.resolve()
    shutil.rmtree(scratch, ignore_errors=True)
    scratch.mkdir(parents=True)
    binaries = {"grok": args.grok.resolve(), "runner": args.runner.resolve()}
    version = grok_version(binaries["grok"], scratch)
    pinned = re.search(r"\b[0-9a-f]{40}\b", (REPO / "NOTICE").read_text()).group()
    if pinned[:12] not in version:
        sys.exit(f"{binaries['grok']} reports {version!r}, not NOTICE's commit {pinned}")
    header = [
        f"grok: {version}",
        f"runner source: {runner_source()}",
        f"host: {sys.platform}",
        "compared: parsed JSON values of each chat-completions body, title calls excluded; key order is listed separately",
        "normalized on both sides: the model name wherever it is a whole string, tools, the date in <user_info>",
        "normalized on Grok's side: its <rules> block and its <system-reminder> blocks",
    ]
    reports, verdicts, all_diffs = [], {}, {}
    for name in names:
        scenario = {**json.loads((ORACLE / "scenarios" / f"{name}.json").read_text()), "name": name}
        outcomes, pairs = {}, {}
        for agent in AGENTS:
            outcomes[agent], endpoint = run(agent, binaries[agent], scenario, scratch)
            pairs[agent] = [(request_kind, *normalize(agent, raw)) for request_kind, raw in endpoint.requests]
            if endpoint.titles:
                outcomes[agent] += f", title calls not compared: {endpoint.titles}"
        verdicts[name], report, all_diffs[name] = compare(name, outcomes, pairs)
        reports.append(report + ("\n\n" + all_diffs[name][0] if all_diffs[name] else ""))
        if args.out:
            directory = args.out / name
            directory.mkdir(parents=True, exist_ok=True)
            for agent in AGENTS:
                lines = [json.dumps({"kind": request_kind, "body": body}, ensure_ascii=False) for request_kind, body, _ in pairs[agent]]
                (directory / f"{agent}.jsonl").write_text("".join(line + "\n" for line in lines))
            (directory / "diff.txt").write_text("\n\n".join(all_diffs[name]) + ("\n" if all_diffs[name] else ""))
    width = max(len(name) for name in names)
    summary = ["== summary", *(f"{name:<{width}}  {verdicts[name]}" for name in names)]
    text = "\n".join(header) + "\n\n" + "\n\n".join(reports) + "\n\n" + "\n".join(summary) + "\n"
    print(text, end="")
    if args.out:
        (args.out / "report.txt").write_text(text)
    sys.exit(0 if all(verdict == "identical" for verdict in verdicts.values()) else 1)


if __name__ == "__main__":
    main()
