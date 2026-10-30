# remanie-train-harness

`remanie-train-harness` is the agent runtime a model trains through on Harbor tasks. One process runs one attempt inside the task sandbox. It reads one JSON object on stdin, sends every model call to one OpenAI-compatible `/v1/chat/completions` endpoint, runs the tool calls, and exits with a code that tells Harbor whether to grade the sandbox. In training the endpoint is Miles' session server, which records token ids and log-probs. The runner never sees them and never computes a reward.

Each request carries the two-sentence system prompt from `assets/system_prompt.jinja2`, the first user message from `assets/user_query.jinja2` (Grok Build's `<user_info>` block with the OS, the shell, the workspace path and the date, then the task wrapped once in `<user_query>` tags), and the eight tool definitions from `assets/tools.json`. Turn requests send `tool_choice: "auto"`, and compaction requests send `tool_choice: "none"`. The endpoint applies the chat template and parses the tool calls. Until a compaction, the history only grows. The runner appends each assistant message exactly as the endpoint returned it, runs its tool calls at the same time, and appends one tool message per call in the order the calls are listed. Grok Build appends them in the order they finish, but Qwen3's chat template drops `tool_call_id`, so the model can tell which result answers which call only by position. Calls in one reply that name the same file in `file_path`, `path`, or `target_file` run one at a time, in the order listed, if any of them writes. Compaction runs only when the input sets `compaction`. The runner records every message, model call, and tool call in `trace.jsonl` during the attempt.

It has no interactive mode, no permission prompts, and no workspace confinement. The sandbox isolates the task.

## Input

`assets/contract.json` holds the input schema. The runner embeds it, and the Harbor adapter checks job settings against it on the host. The runner reads at most 1 MiB from stdin and refuses unknown fields.

| Field | Required | Meaning |
|---|---|---|
| `base_url` | yes | `http` or `https` URL whose path ends in `/v1`, with no credentials, query, or fragment. |
| `model` | yes | Model name sent with every request. |
| `api_key_env` | no | Environment variable that holds the key. The runner reads it and removes it before any tool starts. When it is null or absent, requests carry no `Authorization` header. |
| `user_query` | yes | The task text. |
| `workspace` | yes | Existing absolute directory. Commands start here. |
| `session_dir` | yes | Absolute path that must not exist yet. Its parent must exist. It receives `trace.jsonl` and `result.json`. |
| `sampling_params` | yes | Forwarded verbatim. Must contain `max_tokens`, 1 to 4294967295. Must not set `model`, `messages`, `tools`, `tool_choice`, or `stream`. |
| `max_turns` | yes | Turn requests allowed, 1 to 512. Compaction requests do not count. |
| `deadline_secs` | yes | Wall-clock limit, 1 to 86400. Set it below Harbor's agent timeout. |
| `context_window_tokens` | no | The model's context size. After the first reply, the runner ends the attempt as `context_overflow` before sending a turn request that cannot fit. Before the first reply, such a request ends the attempt as `infrastructure` with cause `config`. A compaction request is sized to fit instead, as described below. |
| `compaction` | no | `{max_tokens, threshold_tokens}`. `threshold_tokens` is optional, and without it `context_window_tokens` is required. When `compaction` is null or absent, the runner never compacts. |

The runner estimates a turn request's prompt tokens as the endpoint's last `usage.total_tokens` plus a quarter of the bytes of the tool messages appended since. The request fits when the estimate plus `sampling_params.max_tokens` is at most `context_window_tokens`. With compaction on, the runner compacts before a turn request once the estimate reaches `threshold_tokens`, or 85% of `context_window_tokens` when `threshold_tokens` is absent. It also compacts once the estimate plus `sampling_params.max_tokens` exceeds `context_window_tokens`, or once the estimate plus the summary prompt's estimate plus `compaction.max_tokens` does. When the endpoint refuses a turn request for its length and the segment holds a reply, the runner compacts and sends the turn again, as Grok Build does.

For the compaction request, the runner appends Grok Build's summary prompt from `assets/compaction.jinja2` as a user message and sends `compaction.max_tokens` in place of `sampling_params.max_tokens`. With `context_window_tokens` set, the request must fit 95% of the window. When the history, the summary prompt, and `compaction.max_tokens` do not fit, the runner drops the oldest turns from the request until they do. A turn is an assistant message and the tool messages that answer it. A compaction request always keeps the messages before the first reply (the system prompt, the task, and in later segments the continuation message) and the last turn. When the kept messages leave less room than `compaction.max_tokens`, the runner caps `max_tokens` at the room left. When no room is left, the attempt ends as `context_overflow` with cause `context_window`, and the summary prompt is neither appended nor sent. When the endpoint refuses the request for its length, the runner resends it once with the oldest turns dropped until the history and the summary prompt fit 70% of the window. It resends only when that drops more turns than the refused request did. Otherwise, or when the endpoint refuses the resend too, the attempt ends as `context_overflow` with cause `endpoint_rejected`. The runner's history keeps every message, and the `model_call` event names the ones its request dropped. The `compaction` event's `input` is `verbatim` when the accepted request dropped nothing, `fitted` when it dropped turns to fit 95% of the window, and `lossy` when it was the resend.

The reply's `<summary>` block becomes a `Summary:` heading. A summary cut off at `max_tokens` is kept, as Grok Build keeps it. The next segment holds the system prompt, the task, and one user message from `assets/continuation.jinja2`. That message holds Grok Build's preamble saying the session continues, followed by the summary. When background tasks are still running or todos are open, the message ends with a `<system-reminder>` that lists them. No earlier turn is kept. The estimate then restarts at a quarter of the new segment's bytes.

The `verbatim`, `fitted`, and `lossy` requests follow the shape of Grok Build's compaction input ladder. Grok Build's `fit_conversation_to_budget` keeps the system message but can drop the task. When the last turn alone does not fit, it shortens that turn's text instead of capping `max_tokens`. Grok Build's lossy step also strips tool calls, tool results, and reasoning, where the runner sends the kept turns intact. Grok Build subtracts the tool definitions' tokens from its fitted and lossy budgets; the runner's estimate already counts them through `usage.total_tokens`.

## Exit codes and result

| Exit | Meaning |
|---|---|
| 0 | Gradable. The sandbox state is the policy's product. |
| 70 | Infrastructure failure. Do not grade. |
| anything else | The process died, for example a panic (101) or a signal. Do not grade. |

`contract.json` lists each outcome and the `cause.kind` values it can carry.

| `outcome` | Exit | When |
|---|---|---|
| `completed` | 0 | The model replied without tool calls. |
| `max_turns` | 0 | The runner sent `max_turns` turn requests. The last turn's tools ran. |
| `max_tokens` | 0 | The reply ended with `finish_reason: "length"`. |
| `context_overflow` | 0 | After the first reply, the next request does not fit `context_window_tokens` (`context_window`) or exceeds 16 MiB (`request_size`), or the endpoint refuses it for its length (`endpoint_rejected`). |
| `invalid_response` | 0 | `empty_turn`, `malformed_message`, `malformed_tool_call`, `duplicate_call_id` within one reply, `unexpected_finish`, or `invalid_summary`. |
| `deadline` | 0 | `deadline_secs` passed. `cause.request_in_flight` says whether a model request was abandoned. |
| `cancelled` | 0 | The runner received SIGTERM or SIGINT. |
| `infrastructure` | 70 | `config`, `endpoint_unavailable`, `endpoint_status`, `connection_lost`, `endpoint_response`, `tool_runtime`, or `internal`. |

Before the first reply, a request that does not fit ends the attempt as `infrastructure` with cause `config`, because the job settings are wrong. A reply with `finish_reason: "abort"` or `"error"` ends the attempt as `infrastructure` with cause `endpoint_response`, and the runner does not resend the request.

The runner resends a request with identical bytes only when the endpoint did not process it. It keeps resending while the next wait, at most 30 s, ends before `deadline_secs`.

Once the runner has created `session_dir`, it writes `<session_dir>/result.json` atomically on every exit it controls. It prints the same object on stdout. Its fields are `contract`, `outcome`, `cause`, `gradable`, `final_answer`, `turns`, `compactions`, `model_calls`, `usage`, `tool_calls`, `background_tasks_left_running`, `trace_truncated`, and `wall_time_ms`. `usage` holds `prompt_tokens`, `completion_tokens`, `cached_tokens`, and `complete`, which is false when any response lacked usage. `tool_calls` holds `total`, `ok`, `failed`, `invalid_arguments`, and `unknown_tool`. Each `background_tasks_left_running` entry holds `task_id`, `pid`, `command`, and `output_file`.

On exit 0 background tasks keep running so the verifier can reach them. On exit 70 the runner kills them. When writing `result.json` fails after a gradable outcome, the runner exits 70 with them still running.

## Trace

`<session_dir>/trace.jsonl` is append-only, one JSON object per line, and each line carries `t_ms` since the start. Each message appears once per segment, and a `model_call` names a message range instead of copying the history. Under the new segment, a compaction writes the system prompt, the task, and the continuation message. `checks/fixtures/` holds a trace recorded from one run against a scripted endpoint and the request bodies it sent; CI re-records it and fails when the bodies differ. remanie-tooling's `research/agents/remanie_train/remanie_trace.py` reads the file and rebuilds the body of any accepted request from the `header`, a message range, and the omitted ranges.

| `event` | Written | Fields |
|---|---|---|
| `header` | once, first | `contract`, `started_at_unix_ms`, `config` (the input object), `tools` |
| `message` | when a message is appended | `segment`, `index`, `message` |
| `tool_call` | after each tool message | `call_id`, `name`, `class` (`ok`, `unknown_tool`, `invalid_arguments`, or `failed`), `task_id`, `latency_ms` |
| `model_call` | per accepted response | `call`, `turn`, `kind` (`turn` or `compaction`), `prompt` (`segment`, `end`, and, when a compaction request dropped messages, `omitted`, their half-open index ranges), `max_tokens_override` (compaction calls only), `attempts`, `latency_ms`, `response` verbatim |
| `endpoint_error` | per failed attempt | `call`, `turn`, `attempt`, `failure` (`connect`, `lost`, `status`, or `response`) with that failure's fields, `retry_in_ms` |
| `compaction` | per compaction | `from_segment`, `to_segment`, `summary_call`, `input` (`verbatim`, `fitted`, or `lossy`) |
| `result` | once, last | the `result.json` object |

When a write fails or the trace would pass 256 MiB, the runner stops tracing, finishes the attempt, and sets `trace_truncated`.

## Tools

`assets/tools.json` holds Grok Build's tool descriptions without their usage notes. Its parameter names, types, descriptions, required lists, defaults, and bounds are Grok Build's too, except that `read_file`'s `format` accepts only `image` or `text` and defaults to `image`. An unknown tool name, arguments that are not JSON or fail the schema, a bound the runner checks, and a tool's own error go back to the model as the tool message. They do not end the attempt.

| Tool | Enforced limits |
|---|---|
| `run_terminal_cmd` | Runs `/bin/bash --noprofile --norc -c` in `workspace` with the runner's environment, after setting 13 variables such as `TERM=dumb` and `PAGER=cat`. `block_until_ms` defaults to 30000 and is capped at 36,000,000. A command still running after `block_until_ms` becomes a background task with an id the runner assigns, such as `task-1`. If `deadline_secs` passes while the tool waits, the attempt ends as `deadline` and the runner kills the command. An unwaited `&` is refused. To run a command in the background, pass `block_until_ms` 0. The reply shows at most 20,000 bytes of output, taken from its start and end. Each command logs to `.remanie-terminal/<pid>-<n>/task-N.log` under the sandbox's temp dir, or under `/var/tmp` when `workspace` is or holds the temp dir. The log keeps every byte with no size cap, so a command that never stops writing can fill the sandbox's disk. A command runs at most 36,000 s. |
| `get_terminal_command_output` | 1 to 20 task ids. `timeout_ms` at most 3,600,000, and 0 or no `timeout_ms` returns a snapshot. Up to 40,000 bytes per task. |
| `kill_terminal_command` | Sends SIGTERM to every process in one task's session, then SIGKILL after 1 s, and waits up to 5 s for the task to exit. |
| `read_file` | At most 1000 lines and about 100,000 bytes per call. When a full 1000-line window leaves lines unread, the output ends with a note naming the next `offset`. A window over about 100,000 bytes is a tool error. A read stops after 30 s. A PDF read returns `Vision is not available in this environment.` unless `format` is `text`. A read of a PNG, JPEG, GIF, or WebP file, recognized by its first bytes as in Grok Build, returns the same text as a successful read. When `format` is `text`, the tool reads a PDF of at most 50 MiB through Poppler and stops after 60 s. A PDF over 10 pages needs `pages`, at most 20 per call. |
| `search_replace` | `old_string` must match exactly once, overlapping matches included, unless `replace_all`. The tool edits UTF-8 files up to 50 MiB, writes in place, and keeps the file's line endings. |
| `list_dir` | Hides ignored and dot files. Scans up to 100,000 entries and stops the listing at about 10,000 bytes. |
| `grep` | Runs `rg`. `head_limit` defaults to 200 lines, at most 2000. The tool keeps at most 40,000 bytes of `rg` output and then at most `head_limit` lines, and a result cut at `head_limit` ends with an "at least" count of the lines `rg` printed. `rg` cuts lines at 1000 columns and skips files over 5 MiB. A search stops after 20 s. When `rg` cannot read a file but finds matches, the tool returns the matches followed by the first three of `rg`'s error messages and a count of the rest. |
| `todo_write` | `merge` defaults to `true` and updates by `id`. Each `id` may appear once per call. The list survives a compaction, and the continuation message shows the open todos. |

Paths may be absolute and may point outside `workspace`. `~` expands to the sandbox's `$HOME`.

## Build and test

`rust-toolchain.toml` pins Rust 1.94.0 with rustfmt and clippy. Install ripgrep. The tools run `rg` from `PATH`, or from the path in `RG_BIN_PATH` when it is set.

```sh
cargo build --release --locked
cargo test --locked
cargo clippy --all-targets --locked -- -D warnings
```

The Rust tests run on Linux and macOS. The runtime bundle is Linux-only; on macOS, build it in a Debian container. [checks/README.txt](checks/README.txt) lists every check.

## Build the bundle

On Debian or Ubuntu, with the CPU architecture of the machines that will run it, run from the repository root:

```sh
cargo build --release --locked
python3 package_linux.py target/release/remanie-train-harness runtime
```

Pushing a `v*` tag runs `.github/workflows/release.yml`, which publishes the x86_64 bundle as the release asset `remanie-train-harness-<tag>-linux-x86_64.tar.gz` beside its `.sha256`. The archive holds `runtime/`.

The output directory must be new. It holds the runner, `rg`, their Linux loader and shared libraries, the license notices, `contract.json`, and `manifest.json` with the architecture and each file's sha256.

## Running under Harbor

The runner knows nothing about Harbor. The Harbor agent that installs a bundle in a task sandbox, runs one attempt and converts the trace to an ATIF trajectory is `research/agents/remanie_train/` in remanie-tooling, which pins the release it runs.

## What the sandbox must provide

- Linux on the bundle's CPU architecture.
- `/bin/bash` and `setsid`.
- `rg`, which the Harbor bundle carries.
- Network access to `base_url`.
- `pdfinfo` and `pdftotext` from Poppler, only for reading a PDF as text.

## Known limits

- Code in the sandbox can read the key from the runner's process through `/proc`. Use a key with a spending cap, or an endpoint that needs none.
- The file tools and commands can reach any path the sandbox user can.
- Harbor's local Docker environment does not enforce a task's disk quota.
- A model response over 16 MiB ends the attempt with exit 70.

This project derives from Grok Build. [NOTICE](NOTICE) names the source and the retained code.
