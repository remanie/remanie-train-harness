# remanie-train-harness

`remanie-train-harness` is the agent runtime a model trains through on Harbor tasks. One process runs one attempt inside the task sandbox. It reads one JSON object on stdin, sends every model call to one OpenAI-compatible `/v1/chat/completions` endpoint, runs the tool calls, and exits with a code that tells Harbor whether to grade the sandbox. In training the endpoint is Miles' session server, which records token ids and log-probs. The runner never sees them and never computes a reward.

Each request carries the two-sentence system prompt from `assets/system_prompt.jinja2`, the task wrapped once in `<user_query>` tags by `assets/user_query.jinja2`, and the eight tool definitions from `assets/tools.json` with `tool_choice: "auto"`. The endpoint owns the chat template and the tool-call parser. The history only grows. The runner appends each assistant message exactly as the endpoint returned it, runs its tool calls one at a time in the order listed, and appends one tool message per call. Compaction runs only when the input sets `compaction`. The runner writes a trace as it goes.

It is a training runtime, not a coding assistant. It has no interactive mode, no permission prompts, and no workspace confinement. The sandbox is what isolates the task.

## Input

`assets/contract.json` holds the input schema. The runner embeds it, and the Harbor adapter checks job settings against it on the host. The runner reads at most 1 MiB from stdin and refuses unknown fields.

| Field | Required | Meaning |
|---|---|---|
| `base_url` | yes | `http` or `https` URL whose path ends in `/v1`, with no credentials, query, or fragment. |
| `model` | yes | Model name sent with every request. |
| `api_key_env` | no | Environment variable that holds the key. The runner reads it and removes it before any tool starts. When it is null or absent, requests carry no `Authorization` header. |
| `user_query` | yes | The task text. |
| `workspace` | yes | Existing absolute directory. Commands start here. |
| `session_dir` | yes | Absolute path that must not exist yet. Its parent must exist. It receives `trace.jsonl`, `result.json`, and the command logs. |
| `sampling_params` | yes | Forwarded verbatim. Must contain `max_tokens`, 1 to 4294967295. Must not set `model`, `messages`, `tools`, `tool_choice`, or `stream`. |
| `max_turns` | yes | Model calls allowed, 1 to 512. |
| `deadline_secs` | yes | Wall-clock limit, 1 to 86400. Set it below Harbor's agent timeout. |
| `context_window_tokens` | no | The model's context size. The runner ends the attempt as `context_overflow` before sending a request that cannot fit. |
| `compaction` | no | `{threshold_tokens, max_tokens}`. When it is null or absent, the runner never compacts. |

With compaction on, the runner asks for a summary once the last reported prompt size plus the bytes of new tool output plus the response reserve reaches `threshold_tokens`. The summary request sends `tool_choice: "none"` and `max_tokens` from `compaction`. The next segment starts with the system prompt, the task, and a continuation message that carries the summary and the todo list, followed by the last two assistant turns and their tool results.

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
| `max_turns` | 0 | The runner made `max_turns` model calls. The last turn's tools ran. |
| `max_tokens` | 0 | The reply ended with `finish_reason: "length"`. |
| `context_overflow` | 0 | The next request would not fit `context_window_tokens` (`context_window`), exceeds 16 MiB (`request_size`), or the endpoint refused it for its length (`endpoint_rejected`). |
| `invalid_response` | 0 | `empty_turn`, `malformed_message`, `malformed_tool_call`, `duplicate_call_id` within one reply, `unexpected_finish`, or `invalid_summary`. |
| `deadline` | 0 | `deadline_secs` passed. `cause.request_in_flight` says whether a model request was abandoned. |
| `cancelled` | 0 | The runner received SIGTERM or SIGINT. |
| `infrastructure` | 70 | `config`, `endpoint_unavailable`, `endpoint_status`, `connection_lost`, `endpoint_response`, `tool_runtime`, or `internal`. |

The runner resends a request with identical bytes only when the endpoint did not process it.

The runner writes `<session_dir>/result.json` atomically on every exit it controls and prints the same object on stdout. Its fields are `contract`, `outcome`, `cause`, `gradable`, `final_answer`, `turns`, `compactions`, `model_calls`, `usage` (`prompt_tokens`, `completion_tokens`, `cached_tokens`, and `complete`, false when any response lacked usage), `tool_calls` (`total`, `ok`, `failed`, `invalid_arguments`, `unknown_tool`), `background_tasks_left_running` (`task_id`, `pid`, `command`), `trace_truncated`, and `wall_time_ms`.

On exit 0 background tasks keep running so the verifier can reach them. On exit 70 the runner kills them.

## Trace

`<session_dir>/trace.jsonl` is append-only, one JSON object per line, and each line carries `t_ms` since the start. Each message appears once, so the file grows linearly with the conversation. `harbor/remanie_trace.py` reads it and rebuilds the body of any request from the `header` and a message range.

| `event` | Written | Fields |
|---|---|---|
| `header` | once, first | `contract`, `started_at_unix_ms`, `input` without secrets, `tools` |
| `message` | when a message is appended | `segment`, `index`, `message`, and for a tool result `tool` (`name`, `error`, `task_id`, `latency_ms`) |
| `model_call` | per accepted response | `call`, `turn`, `kind` (`turn` or `compaction`), `prompt` (`segment`, `end`), `tool_choice`, `max_tokens`, `attempts`, `latency_ms`, `response` verbatim |
| `endpoint_error` | per failed attempt | `call`, `turn`, `attempt`, `failure`, `retry_in_ms` |
| `compaction` | per compaction | `from_segment`, `to_segment`, `summary_call`, `kept_messages` |
| `result` | once, last | the `result.json` object |

When a write fails or the trace would pass 256 MiB, the runner stops tracing, finishes the attempt, and sets `trace_truncated`.

## Tools

An unknown tool name, arguments that are not JSON or fail the schema, and a tool's own error go back to the model as the tool message. They do not end the attempt.

| Tool | Enforced limits |
|---|---|
| `run_terminal_cmd` | Runs `/bin/bash --noprofile --norc -c` in `workspace` with the sandbox's environment. `block_until_ms` defaults to 30000, at most 300000. A command still running then becomes a background task whose id is the tool-call id. An unwaited `&` is refused unless `block_until_ms` is 0. The reply shows 20,000 bytes of output. The log stops growing at 16 MiB while the command keeps running. A command runs at most 36,000 s. |
| `get_terminal_command_output` | 1 to 20 task ids. `timeout_ms` at most 300000, and 0 returns a snapshot. Up to 40,000 bytes per task. |
| `kill_terminal_command` | Stops one task's process group and waits up to 5 s for it to exit. |
| `read_file` | At most 1000 lines and about 100,000 bytes per call. A partial window ends with a note naming the next `offset`. Reading stops after 30 s. A PDF needs `format: "text"` and Poppler and may be at most 50 MiB. A PDF over 10 pages needs `pages`, at most 20 per call. |
| `search_replace` | `old_string` must match exactly once, overlapping matches included, unless `replace_all`. Files up to 50 MiB. The tool keeps untouched bytes and line endings. |
| `list_dir` | Hides ignored and dot files. Scans up to 100,000 entries and returns at most 10,000 bytes. |
| `grep` | Runs `rg`. `head_limit` defaults to 200 lines, at most 2000. `rg` cuts lines at 1000 columns and skips files over 5 MiB. A search stops after 20 s. |
| `todo_write` | `merge` defaults to `true` and updates by `id`. The list survives compaction. |

Paths may be absolute and may point outside `workspace`. `~` expands to the sandbox's `$HOME`.

## Build and test

`rust-toolchain.toml` pins Rust 1.94.0 with rustfmt and clippy. Install ripgrep. The tools find `rg` on `PATH`, or at the absolute path in `RG_BIN_PATH`.

```sh
cargo build --release --locked
cargo test --locked
cargo clippy --all-targets --locked -- -D warnings
```

The Rust tests run on Linux and macOS. The runtime bundle and the Harbor trials are Linux-only. On macOS, build the bundle in a Debian container and run the Harbor check against Docker. [checks/README.txt](checks/README.txt) lists every check, and [harbor/README.md](harbor/README.md) covers the runtime bundle and Harbor jobs.

## What the sandbox must provide

- Linux on the bundle's CPU architecture.
- `/bin/bash` and `setsid`.
- `rg`, which the Harbor bundle carries.
- Network access to `base_url`.
- `pdfinfo` and `pdftotext` from Poppler, only for reading PDFs.

## Known limits

- Code in the sandbox can read the key from the runner's process through `/proc`. Use a key with a spending cap, or an endpoint that needs none.
- The file tools and commands can reach any path the sandbox user can.
- Harbor's local Docker environment does not enforce a task's disk quota.
- A model response over 16 MiB ends the attempt with exit 70.

This project derives from Grok Build. [NOTICE](NOTICE) names the source and the retained code.
