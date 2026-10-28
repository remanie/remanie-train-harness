# remanie-train-harness

`remanie-train-harness` is the agent runtime a model trains through on Harbor tasks. One process runs one attempt inside the task sandbox. It reads one JSON object on stdin, sends every model call to one OpenAI-compatible `/v1/chat/completions` endpoint, runs the tool calls, and exits with a code that tells Harbor whether to grade the sandbox. In training the endpoint is Miles' session server, which records token ids and log-probs. The runner never sees them and never computes a reward.

Each request carries the two-sentence system prompt from `assets/system_prompt.jinja2`, the task wrapped once in `<user_query>` tags by `assets/user_query.jinja2`, and the eight tool definitions from `assets/tools.json` with `tool_choice: "auto"`. The endpoint owns the chat template and the tool-call parser. The history only grows. The runner appends each assistant message exactly as the endpoint returned it, runs its tool calls one at a time in the order listed, and appends one tool message per call. Compaction runs only when the input sets `compaction`. The runner writes a trace as it goes.

It is not a product harness. It has no interactive mode, no permission prompts and no workspace confinement. The sandbox is what isolates the task.

## Input

`assets/contract.json` holds the input schema. The runner embeds it, and the Harbor adapter checks job settings against it on the host. Stdin is at most 1 MiB, and unknown fields are refused.

| Field | Required | Meaning |
|---|---|---|
| `base_url` | yes | `http` or `https` URL whose path ends in `/v1`, with no credentials, query or fragment. |
| `model` | yes | Model name sent with every request. |
| `api_key_env` | no | Environment variable that holds the key. The runner reads it and removes it before any tool starts. Without it, requests carry no `Authorization` header. |
| `user_query` | yes | The task text. |
| `workspace` | yes | Existing absolute directory. Commands start here. |
| `session_dir` | yes | Absolute path that must not exist yet. Its parent must exist. It receives `trace.jsonl`, `result.json` and the command logs. |
| `sampling_params` | yes | Forwarded verbatim. Must contain `max_tokens`. Must not set `model`, `messages`, `tools`, `tool_choice`, `stream`, `stream_options`, `n`, `logprobs` or `input_ids`. |
| `max_turns` | yes | Model calls allowed, 1 to 512. |
| `deadline_secs` | yes | Wall-clock limit, 1 to 86400. Set it below Harbor's agent timeout. |
| `compaction` | no | `{threshold_tokens, max_tokens}`. Without it the runner never compacts. |
| `retry` | no | `{attempts, base_backoff_ms, max_backoff_ms}`, default 6, 2000 and 30000. |
| `keep_background_tasks` | no | Default `true`. |

With compaction on, the runner asks for a summary once the last reported prompt size plus the bytes of new tool output plus the response reserve reaches `threshold_tokens`. The summary request sends `tool_choice: "none"` and `max_tokens` from `compaction`. The next request starts a new history of the system prompt, the task, and a continuation message with the summary, the todo list and the background tasks.

## Exit codes and result

| Exit | Meaning |
|---|---|
| 0 | Gradable. The sandbox state is the policy's product. |
| 70 | Infrastructure failure. Do not grade. |
| anything else | The process died, for example a panic (101) or a signal. Do not grade. |

Exit 0 ends with one of these outcomes:

| `outcome` | When |
|---|---|
| `completed` | The model replied without tool calls. |
| `max_turns` | `max_turns` model calls were made. The last turn's tools ran. |
| `max_tokens` | The reply ended with `finish_reason: "length"`. |
| `context_overflow` | The endpoint returned 400 with a body that names the context length. |
| `invalid_response` | An empty reply, a tool call that is not a function, two tool calls in one reply sharing an id, or an unusable summary. |
| `deadline` | `deadline_secs` passed. |
| `cancelled` | The runner received SIGTERM or SIGINT. |

Exit 70 covers an endpoint that still fails after the retries, an endpoint status the runner does not retry, a response the runner cannot parse, a configuration error, a tool runtime failure and a missing runtime dependency. The runner resends a request with identical bytes only when the endpoint did not process it: a connection failure before the request was sent, 429 and 503. Backoff starts at 2 s, doubles with jitter up to 30 s, honors `Retry-After`, and stops at the deadline.

The runner writes `<session_dir>/result.json` atomically on every exit it controls and prints the same object on stdout. Its fields are `schema`, `outcome`, `cause` (exit 70 only, with up to 64 KiB of the endpoint's error body), `gradable`, `final_answer`, `turns`, `compactions`, `model_calls`, `usage` (`prompt_tokens`, `completion_tokens`, `cached_tokens`, and `complete`, false when any response lacked usage), `tool_calls` (`total`, `ok`, `tool_error`, `invalid_arguments`, `unknown_tool`), `background_tasks_left_running` (`task_id`, `pid`, `command`), `trace` (`path`, `events`, `truncated`) and `wall_time`.

On exit 0 background tasks keep running so the verifier can reach them. On exit 70 the runner kills them.

## Trace

`<session_dir>/trace.jsonl` is append-only, one JSON object per line. Each message appears once, so its size grows linearly with the conversation. The body of every request can be rebuilt from the `header` and a message range. `harbor/remanie_trace.py` reads it.

| `event` | Written | Fields |
|---|---|---|
| `header` | once, first | `config` without secrets, `tools`, `prompts_sha256`, `bundle_sha256` |
| `message` | when a message is appended | `segment`, `index`, `message` |
| `model_call` | per accepted response | `call`, `kind` (`turn` or `compaction`), `prompt` (`segment`, `end`), `max_tokens_override`, `request_sha256`, `response` verbatim, `attempts`, `latency_ms` |
| `endpoint_error` | per failed attempt | `call`, `attempt`, `status`, `body_excerpt` (up to 64 KiB), `verdict`, `retry_after_ms` |
| `tool_call` | per tool call | `call_id`, `name`, `class` (`ok`, `tool_error`, `invalid_arguments`, `unknown_tool`), `latency_ms`, `output_bytes` |
| `compaction` | per compaction | `from`, `to`, `summary_call`, `estimated_tokens` |
| `result` | once, last | the `result.json` object |

When a write fails or the trace reaches 256 MiB, the runner stops tracing, finishes the attempt and sets `trace.truncated`.

## Tools

An unknown tool name, arguments that are not JSON or fail the schema, and a tool's own error go back to the model as the tool message. They do not end the attempt.

| Tool | Enforced limits |
|---|---|
| `run_terminal_cmd` | Runs `/bin/bash --noprofile --norc -c` in `workspace` with the sandbox's environment. `block_until_ms` defaults to 30000, at most 300000. A command still running then becomes a background task whose id is the tool-call id. An unwaited `&` is refused unless `block_until_ms` is 0. The reply shows 20,000 bytes of output. The log stops growing at 16 MiB while the command keeps running. A command runs at most 36,000 s. |
| `get_terminal_command_output` | 1 to 20 task ids. `timeout_ms` at most 300000, and 0 returns a snapshot. Up to 40,000 bytes per task. |
| `kill_terminal_command` | Stops one task's process group and waits up to 5 s for it to exit. |
| `read_file` | At most 1000 lines and about 100,000 bytes per call. A partial window ends with a note naming the next `offset`. Reading stops after 30 s. A PDF needs `format: "text"` and Poppler and may be at most 50 MiB. A PDF over 10 pages needs `pages`, at most 20 per call. |
| `search_replace` | `old_string` must match exactly once, overlapping matches included, unless `replace_all`. Files up to 50 MiB. Untouched bytes and line endings are kept. |
| `list_dir` | Hides ignored and dot files. Scans up to 100,000 entries and returns at most 10,000 bytes. |
| `grep` | Runs `rg`. `head_limit` defaults to 200 lines, at most 2000. Lines are cut at 1000 columns, files over 5 MiB are skipped, and a search stops after 20 s. |
| `todo_write` | `merge` defaults to `true` and updates by `id`. The list survives compaction. |

Paths may be absolute and are not confined to `workspace`. `~` is the sandbox's `$HOME`.

## Build and test

`rust-toolchain.toml` pins Rust 1.94.0 with rustfmt and clippy. Install ripgrep. The tools find `rg` on `PATH`, or at the absolute path in `RG_BIN_PATH`.

```sh
cargo build --release --locked
cargo test --locked
cargo clippy --all-targets --locked -- -D warnings
```

The tests run on Linux and macOS. Packaging and the Harbor integration check need Linux. On macOS, run the Harbor check against Docker. [checks/README.txt](checks/README.txt) lists every check, and [harbor/README.md](harbor/README.md) covers the runtime bundle and Harbor jobs.

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
- A model request or response over 16 MiB ends the attempt with exit 70.

This project derives from Grok Build. [NOTICE](NOTICE) names the source and the retained code.
