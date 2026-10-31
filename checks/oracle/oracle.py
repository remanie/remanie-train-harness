"""Run each scenario through Grok Build's own binary and through the trial runner wrapping that binary, and check that the endpoint received the same request bodies."""

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
from make_trace_fixture import MODEL, Endpoint

KEY_ENV = "ORACLE_API_KEY"
KEY = "oracle-key"
GROK_KEY_ENV = "REMANIE_TRAIN_KEY"
GROK_KEY = "replaced-by-the-trial-runner-proxy"
GROK_SAMPLING = {"max_tokens": "max_completion_tokens", "temperature": "temperature"}
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
    "GROK_AGENT_ID": "remanie-train-harness",
    "GROK_TURN_SUMMARY": "0",
    "GROK_MEMORY": "0",
    "GROK_WORKFLOWS": "0",
}
DATE = re.compile(r"(?<=Today's date: )\d{4}-\d{2}-\d{2}")


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


def job(scenario, live, url):
    return {"base_url": url, "model": MODEL, "api_key_env": KEY_ENV, "user_query": scenario["user_query"], "workspace": str(live / "workspace"), "session_dir": str(live / "session"), **scenario["job"]}


def grok_config(scenario, url):
    """The config.toml the runner writes, rebuilt here independently so the oracle checks the runner's config writer too. Grok Build appends to it while it runs, so both sides are compared after their runs."""
    settings = scenario["job"]
    q = json.dumps
    model = {"name": MODEL, "model": MODEL, "base_url": url, "env_key": GROK_KEY_ENV, "api_backend": "chat_completions", "context_window": settings["context_window_tokens"]}
    model.update({setting: settings["sampling_params"][key] for key, setting in GROK_SAMPLING.items() if key in settings["sampling_params"]})
    threshold = settings.get("compaction", {}).get("threshold_percent", 85)
    lines = [
        "[cli]", "auto_update = false", "",
        "[features]", "telemetry = false", "",
        "[harness]", "disable_codebase_upload = true", "",
        "[session]", f"auto_compact_threshold_percent = {threshold}", "",
        "[models]", *(f"{role} = {q(MODEL)}" for role in ("default", "session_summary", "image_description", "web_search")), "",
        f"[model.{q(MODEL)}]", *(f"{key} = {float(value) if key == 'temperature' else q(value)}" for key, value in model.items()),
    ]
    return "\n".join(lines) + "\n"


def run_grok(grok, scenario, endpoint, live):
    """Grok Build as the runner launches it, with the scripted endpoint in place of the runner's proxy."""
    session_dir = live / "session"
    home = session_dir / "grok-home"
    home.mkdir(parents=True)
    (home / "config.toml").write_text(grok_config(scenario, endpoint.url))
    (home / "agent.md").write_text((REPO / "assets/agent.md").read_text())
    settings = scenario["job"]
    workspace = live / "workspace"
    command = [
        str(grok), "-p=" + scenario["user_query"], "-m", MODEL, "--cwd", str(workspace),
        "--output-format", "json", "--always-approve", "--no-auto-update",
        "--agent", str(home / "agent.md"), "--max-turns", str(settings["max_turns"]),
        "--session-id", str(uuid.uuid5(uuid.NAMESPACE_URL, str(session_dir))),
    ]
    env = {
        **base_env(live), "GROK_HOME": str(home), "GROK_LOG_FILE": str(session_dir / "grok.log"),
        **dict.fromkeys(GROK_URLS, endpoint.url), **GROK_ENV,
        GROK_KEY_ENV: GROK_KEY, "XAI_API_KEY": GROK_KEY, "NO_PROXY": "127.0.0.1,localhost", "no_proxy": "127.0.0.1,localhost",
    }
    code, stdout, _ = execute(command, workspace, env, None, settings["deadline_secs"] + 60)
    try:
        result = json.loads(stdout)
    except ValueError:
        result = {}
    detail = result.get("stopReason") or " ".join(result.get("message", "no JSON on stdout").split())[:300]
    return {"summary": f"{code}, {detail}", "config": (home / "config.toml").read_text()}


def run_runner(runner, grok, scenario, endpoint, live):
    workspace = live / "workspace"
    env = {**base_env(live), KEY_ENV: KEY, "REMANIE_GROK": str(grok)}
    code, stdout, stderr = execute([str(runner)], workspace, env, json.dumps(job(scenario, live, endpoint.url)).encode(), scenario["job"]["deadline_secs"] + 60)
    try:
        result = json.loads(stdout)
    except ValueError:
        return {"summary": f"{code}, no JSON on stdout: {stderr.decode()[-300:]}", "events": [], "config": None}
    trace = live / "session/trace.jsonl"
    events = [json.loads(line) for line in trace.read_text().splitlines()] if trace.exists() else []
    cause = f" ({result['cause']['kind']})" if result.get("cause", {}).get("kind") else ""
    config = live / "session/grok-home/config.toml"
    return {
        "summary": f"{code}, {result['outcome']}{cause}, turns {result['turns']}, compactions {result['compactions']}, model calls {result['model_calls']}",
        "events": events,
        "config": config.read_text() if config.exists() else None,
        "result": result,
    }


def run(agent, scenario, scratch, launch):
    live = scratch / "live"
    shutil.rmtree(live, ignore_errors=True)
    for directory in ("workspace", "home", "tmp"):
        (live / directory).mkdir(parents=True)
    for name, spec in scenario.get("files", {}).items():
        (live / "workspace" / name).write_text(file_text(spec))
    endpoint = Endpoint(scenario["replies"], scenario["job"].get("context_window_tokens"))
    try:
        outcome = launch(endpoint, live)
    finally:
        endpoint.server.shutdown()
    kept = scratch / "runs" / scenario["name"] / agent
    kept.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(live, kept)
    return outcome, endpoint


def dated(body):
    return json.loads(DATE.sub("<date>", body.decode()))


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


def compare(scenario, grok, runner, grok_endpoint, runner_endpoint):
    """Returns the problems that make the scenario differ, the per-request table, and the diffs."""
    allowed = scenario.get("injected", [])
    sampling = scenario["job"]["sampling_params"]
    problems, rows, diffs = [], [], []
    expected_config = grok["config"].replace(grok_endpoint.url, proxy_url(runner["config"]))
    if runner["config"] != expected_config:
        problems.append("the runner's config.toml differs from the oracle's")
        diffs.append("\n".join(difflib.unified_diff(expected_config.splitlines(), (runner["config"] or "").splitlines(), "oracle config.toml", "runner config.toml", lineterm="")))
    traced = [event for event in runner["events"] if event["event"] == "request"]
    titles = [event for event in traced if event["kind"] == "title"]
    forwarded = [event for event in traced if event["kind"] != "title"]
    if runner_endpoint.titles or len(titles) != len(grok_endpoint.titles):
        problems.append(f"title calls: grok sent {len(grok_endpoint.titles)} to the endpoint; the runner traced {len(titles)} and forwarded {len(runner_endpoint.titles)}")
    if [kind for kind, _ in runner_endpoint.requests] != [event["kind"] for event in forwarded]:
        problems.append("the trace's request kinds differ from what the endpoint received")
    if any(authorization != f"Bearer {KEY}" for authorization in runner_endpoint.authorizations):
        problems.append("a forwarded request did not carry the endpoint key")
    for index in range(max(len(grok_endpoint.requests), len(runner_endpoint.requests))):
        g = grok_endpoint.requests[index] if index < len(grok_endpoint.requests) else None
        r = runner_endpoint.requests[index] if index < len(runner_endpoint.requests) else None
        if g is None or r is None:
            rows.append(f"{index + 1:>4}  {g[0] if g else '-':<11} {r[0] if r else '-':<11} {'runner' if g else 'grok'} sent no request")
            problems.append(f"request {index + 1}: only one side sent it")
            continue
        injected = forwarded[index]["injected"] if index < len(forwarded) else []
        g_body, r_body = json.loads(g[1]), json.loads(r[1])
        if index < len(forwarded) and r_body != forwarded[index]["body"]:
            problems.append(f"request {index + 1}: the trace's body differs from what the endpoint received")
        values = g[0] == r[0] and {key: value for key, value in r_body.items() if key not in injected} == g_body
        injection = set(injected) == set(allowed) and all(r_body[key] == sampling[key] for key in injected)
        exact = r[1] == g[1] if not injected else r[1].startswith(g[1][:-1]) and r[1].endswith(b"}")
        same = values and injection and exact
        note = "same bytes" + (", injected " + ", ".join(injected) if injected else "") if same else "values differ" if not values else "injection differs" if not injection else "same values, different bytes"
        rows.append(f"{index + 1:>4}  {g[0]:<11} {r[0]:<11} {note}")
        if not same:
            problems.append(f"request {index + 1} ({g[0]}): {note}")
            right = {key: value for key, value in dated(r[1]).items() if key not in injected}
            diffs.append("\n".join(difflib.unified_diff(list(flat(dated(g[1]))), list(flat(right)), f"grok request {index + 1} ({g[0]})", f"runner request {index + 1} ({r[0]})", lineterm="")))
    return problems, rows, diffs


def proxy_url(config):
    match = re.search(r'^base_url = "([^"]+)"$', config or "", re.M)
    return match.group(1) if match else "<missing>"


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
    parser.add_argument("--grok", type=Path, required=True, help="xai-grok-pager built at upstream.json's commit")
    parser.add_argument("--runner", type=Path, required=True, help="the trial runner binary")
    parser.add_argument("--scenario", action="append", help="run only this scenario; repeatable")
    parser.add_argument("--out", type=Path, help="write report.txt and, per scenario, the request bodies, Grok's title calls and the diffs here")
    parser.add_argument("--scratch", type=Path, default=Path("/tmp/remanie-oracle"), help="emptied, then holds each run under runs/<scenario>/<side>")
    args = parser.parse_args()
    names = args.scenario or sorted(path.stem for path in (ORACLE / "scenarios").glob("*.json"))
    scratch = args.scratch.resolve()
    shutil.rmtree(scratch, ignore_errors=True)
    scratch.mkdir(parents=True)
    grok, runner = args.grok.resolve(), args.runner.resolve()
    version = grok_version(grok, scratch)
    pinned = json.loads((REPO / "upstream.json").read_text())["commit"]
    if pinned[:12] not in version:
        sys.exit(f"{grok} reports {version!r}, not upstream.json's commit {pinned}")
    header = [
        f"grok: {version}",
        f"runner source: {runner_source()}",
        f"host: {sys.platform}",
        "compared: the bytes of every chat-completions body the endpoint received; the runner's may add only the scenario's injected keys before the closing brace",
        "also checked: the runner's config.toml against the oracle's, its trace against what the endpoint received, the endpoint key, and that title calls stay in the runner",
    ]
    reports, verdicts = [], {}
    for name in names:
        scenario = {**json.loads((ORACLE / "scenarios" / f"{name}.json").read_text()), "name": name}
        grok_side, grok_endpoint = run("grok", scenario, scratch, lambda endpoint, live: run_grok(grok, scenario, endpoint, live))
        runner_side, runner_endpoint = run("runner", scenario, scratch, lambda endpoint, live: run_runner(runner, grok, scenario, endpoint, live))
        problems, rows, diffs = compare(scenario, grok_side, runner_side, grok_endpoint, runner_endpoint)
        verdicts[name] = "identical" if not problems else problems[0]
        lines = [
            f"== {name}: {verdicts[name]}",
            f"grok: {grok_side['summary']}; {len(grok_endpoint.requests)} requests, {len(grok_endpoint.titles)} title call",
            f"runner: {runner_side['summary']}; {len(runner_endpoint.requests)} requests forwarded, title calls answered by the proxy",
            *(f"problem: {problem}" for problem in problems[1:]),
            "", "   #  grok        runner      result", *rows,
        ]
        reports.append("\n".join(lines) + ("\n\n" + diffs[0] if diffs else ""))
        if args.out:
            directory = args.out / name
            directory.mkdir(parents=True, exist_ok=True)
            for side, endpoint in (("grok", grok_endpoint), ("runner", runner_endpoint)):
                (directory / f"{side}.jsonl").write_text("".join(json.dumps({"kind": request_kind, "body": dated(body)}, ensure_ascii=False) + "\n" for request_kind, body in endpoint.requests))
            (directory / "grok.titles.jsonl").write_text("".join(json.dumps(dated(body), ensure_ascii=False) + "\n" for body in grok_endpoint.titles))
            (directory / "diff.txt").write_text("\n\n".join(diffs) + ("\n" if diffs else ""))
    width = max(len(name) for name in names)
    summary = ["== summary", *(f"{name:<{width}}  {verdicts[name]}" for name in names)]
    text = "\n".join(header) + "\n\n" + "\n\n".join(reports) + "\n\n" + "\n".join(summary) + "\n"
    print(text, end="")
    if args.out:
        (args.out / "report.txt").write_text(text)
    sys.exit(0 if all(verdict == "identical" for verdict in verdicts.values()) else 1)


if __name__ == "__main__":
    main()
