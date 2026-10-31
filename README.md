# remanie-train-harness

`remanie-train-harness` runs one Harbor trial attempt on Grok Build. It starts Grok Build's own binary, `xai-grok-pager` built from the commit `upstream.json` pins, in headless mode with the agent file `assets/agent.md`, and puts a local proxy between Grok Build and the model endpoint. It reads one JSON object on stdin, writes `trace.jsonl` and `result.json` in `session_dir`, and exits with a code that tells Harbor whether to grade the sandbox. In training the endpoint is Miles' session server, which records token ids and log-probs. The runner never sees them and never computes a reward.

## What differs from Grok Build

The agent loop, the system prompt, the tools' code, compaction, retries and background tasks are Grok Build's. Three things differ.

- **The eight tools.** `assets/agent.md` gives Grok Build exactly eight tools and no injected defaults: `run_terminal_cmd`, `read_file` and `search_replace` from the `grok-build-concise` agent, `list_dir`, `grep` and `todo_write` from Grok Build's shared set, and `kill_terminal_command` and `get_terminal_command_output` from its `grok-computer` set, in the `grok-build-concise` order.
- **No outside context.** The agent file turns off AGENTS.md, skill discovery and inherited skills. `GROK_MEMORY=0`, `GROK_WORKFLOWS=0` and `GROK_TURN_SUMMARY=0` turn off memory, workflows, turn summaries and title refreshes. Grok Build has no setting for its other system reminders, so the reminders about finished background commands and a changed date still reach the model. The `<rules>` block in the first user message, the post-compaction reminder and the `compaction/segment_*.md` files are Grok Build's own.
- **Standalone input and output.** The runner takes the JSON input below instead of Grok Build's CLI and config, keeps Grok Build's state in `session_dir/grok-home`, and writes the trace and result below. Grok Build sends a session-title request at the start of every session and has no setting that turns it off. The proxy answers it itself, never forwards it, and traces it as kind `title`. The proxy also answers `GET /v1/models` with a one-model list and every other request that is not a chat completion with 404, so nothing but Grok Build's model calls reaches the endpoint.

## Input

`assets/contract.json` holds the input schema. The runner embeds it, and the Harbor adapter checks job settings against it on the host. The runner reads at most 1 MiB from stdin and refuses unknown fields.

| Field | Required | Meaning |
|---|---|---|
| `base_url` | yes | `http` or `https` URL whose path ends in `/v1`, with no credentials, query, or fragment. The proxy sends Grok Build's chat completions to `<base_url>/chat/completions`. |
| `model` | yes | Model name. The runner configures it for every Grok Build model role, and Grok Build sends it with every request. |
| `api_key_env` | no | Environment variable that holds the endpoint key. The proxy sends `Authorization: Bearer <key>` on every forwarded request, and Grok Build never sees the key. When it is null or absent, requests carry no `Authorization` header. |
| `user_query` | yes | The task. |
| `workspace` | yes | Existing absolute directory. Grok Build works there. |
| `session_dir` | yes | Absolute path that must not exist yet. Its parent must exist. It receives `trace.jsonl`, `result.json`, Grok Build's log `grok.log`, and Grok Build's home `grok-home/`. |
| `sampling_params` | yes | Must contain `max_tokens`, 1 to 4294967295, which becomes Grok Build's `max_completion_tokens`. An optional `temperature` becomes Grok Build's `temperature`. The proxy appends every key that a forwarded body lacks, such as OpenRouter's `provider` and `reasoning`, and never replaces a key Grok Build set. Must not set `model`, `messages`, `tools`, `tool_choice`, `stream`, `stream_options`, or `max_completion_tokens`. |
| `max_turns` | yes | Grok Build's `--max-turns`, 1 to 512. |
| `deadline_secs` | yes | Wall-clock limit from the runner's start, 1 to 86400. Set it below Harbor's agent timeout. |
| `context_window_tokens` | yes | Grok Build's `context_window` for the model. |
| `compaction` | no | `{threshold_percent}`, 1 to 100, which becomes Grok Build's `auto_compact_threshold_percent`. It is 85 when absent. Grok Build has no setting that turns auto-compaction off. |

## How the runner starts Grok Build

The runner binds the proxy on `127.0.0.1`, writes `grok-home/config.toml` and `grok-home/agent.md`, checks that `grok --version` names the pinned commit, and runs, in `workspace`:

```sh
grok -p=<user_query> -m <model> --cwd <workspace> --output-format json --always-approve --no-auto-update \
  --agent <session_dir>/grok-home/agent.md --max-turns <max_turns> --session-id <session id>
```

The attached `-p=` form keeps a task that starts with a dash from parsing as a flag. The session id is the version 5 UUID of `session_dir` in the URL namespace. `config.toml` names the model for every role with the proxy as `base_url`, `api_backend = "chat_completions"`, the context window, `max_completion_tokens`, `temperature`, and the compaction threshold, and turns off auto-update, telemetry and codebase upload.

Grok Build inherits the runner's environment without the key, without any `GROK_*` or `XAI_*` variable, and with these set: `GROK_HOME`, `GROK_LOG_FILE`, Grok Build's eight service URLs pointing at the proxy, the switches in `src/grok.rs` that turn off telemetry, feedback, trace upload, auto-update, prompt suggestions, turn summaries, memory and workflows, a placeholder key, and `127.0.0.1,localhost` added to `NO_PROXY`. Grok Build's stdin is `/dev/null`, and its stderr is the runner's.

At the deadline, on SIGTERM or SIGINT, or when the endpoint gives no HTTP response to the first request, the runner sends SIGTERM to Grok Build's process group and SIGKILL five seconds later. Grok Build's commands run in sessions of their own, so the runner is their subreaper and kills every process left under it once Grok Build is gone. On a run that ends on its own, the runner kills nothing. Grok Build waits up to 600 s for its background commands after the last turn and kills those still running when it exits.

## Exit codes and result

| Exit | Meaning |
|---|---|
| 0 | Gradable. The sandbox state is the policy's product. |
| 70 | The attempt must not be graded. |
| anything else | The process died, for example a panic (101) or a signal. Do not grade. |

`contract.json` lists each outcome and the `cause.kind` values it can carry.

| `outcome` | Exit | When |
|---|---|---|
| `completed` | 0 | Grok Build exited 0 with its JSON result. |
| `max_turns` | 0 | Grok Build stopped at `--max-turns`. It printed its JSON result and exited 1. |
| `cancelled` | 0 | The runner stopped Grok Build at the deadline (`deadline`) or on SIGTERM or SIGINT (`signal`, with `signal`). |
| `error` | 0 | Grok Build failed after the endpoint answered a model request. `cause.message` is Grok Build's own error text, or says how it exited without one. |
| `endpoint_unavailable` | 70 | The endpoint gave no HTTP response to the first request (`unreachable`, with the proxy's `detail`), or Grok Build failed before any request got a 2xx answer (`refused`, with the last `status` and Grok Build's `message`). |
| `infrastructure` | 70 | `config` (the input or the key's variable), `grok_binary` (`REMANIE_GROK` unset, not runnable, or not the pinned commit), `grok_startup` (Grok Build failed before sending a model request), `proxy_bind`, or `internal`. |

Once the runner has created `session_dir`, it writes `<session_dir>/result.json` atomically and prints the same object on stdout. Its fields:

| Field | Meaning |
|---|---|
| `contract` | 2. |
| `outcome`, `cause`, `gradable` | As above. Only `completed` and `max_turns` have no `cause`. |
| `final_answer` | The `text` of Grok Build's JSON result, which joins all the assistant text Grok Build streamed in the run. Empty without a result. |
| `turns`, `compactions` | Turn and compaction requests the endpoint answered with a 2xx status and a complete body. |
| `model_calls` | Requests the proxy forwarded, retries included. The title request is not one. |
| `usage` | `prompt_tokens`, `completion_tokens` and `cached_tokens` summed over the 2xx responses, and `complete`, false when any of them lacked usage or was cut off. |
| `wall_time_ms` | From the runner's start. |
| `grok_exit` | `{"code": n}` or `{"signal": n}`, or null when Grok Build never ran. |
| `grok_session_dir` | Grok Build's session directory relative to `session_dir`, such as `grok-home/sessions/<cwd>/<session id>`, or null. It holds Grok Build's `chat_history.jsonl`, `events.jsonl` and `compaction/` segments. |

## Trace

`<session_dir>/trace.jsonl` is append-only, one JSON object per line, and each line carries `t_ms` since the runner's start. The runner stops tracing after a failed write, so a trace without a closing `result` line is incomplete. `checks/fixtures/` holds a trace recorded on Linux from one run against a scripted endpoint and the request bodies that endpoint received, with the date set to 2026-01-01. CI re-records it and fails when the bodies differ. remanie-tooling's trace reader tests read both files.

| `event` | Written | Fields |
|---|---|---|
| `header` | once, first | `contract`, `started_at_unix_ms`, `config` (the input object), `agent_file` (the text of `assets/agent.md`), `grok_version` |
| `request` | per chat-completions request Grok Build sends | `call` (1, 2, ... in arrival order), `kind` (`turn`, `compaction` or `title`), `body` (the JSON object forwarded, injected keys included), `injected` (the keys the proxy appended) |
| `response` | per request, once its answer is complete | `call`, `status` (null when the endpoint gave no HTTP response), `body`, `raw_sse`, `error` |
| `result` | once, last | the `result.json` object |

A request whose tools include `session_title` is a `title`, one whose last message is a user message holding Grok Build's summary prompt is a `compaction`, and any other is a `turn`. The oracle uses the same rule. For a 2xx event stream, `body` is the chat completion rebuilt from the chunks, with text joined, tool calls merged by index and the last `usage`, and `raw_sse` holds the stream as received. For a 2xx JSON answer, `body` is that object. For any other status, `body` is the response text. `error` appears when the endpoint gave no HTTP response, the stream broke, or Grok Build closed the connection first. A title request's response is the proxy's own.

## Bundle

On Debian or Ubuntu, with the CPU architecture of the machines that will run it, run from the repository root:

```sh
cargo build --release --locked
python3 build_grok.py
python3 package_linux.py target/release/remanie-train-harness target/grok-build/out runtime
```

`build_grok.py` fetches `upstream.json`'s commit of Grok Build into `target/grok-build/src` and runs `cargo build --release --locked -p xai-grok-pager-bin` there, so upstream's `Cargo.lock` and `rust-toolchain.toml` apply. It fetches the protoc that upstream's DotSlash file `bin/protoc` pins and the ripgrep 15.0.0 release that upstream's build scripts embed, checks each against its sha256, and passes them as `PROTOC`, `GROK_TOOLS_BUNDLE_RG_PATH` and `GROK_SHELL_BUNDLE_RG_PATH`. It needs `git`, a C compiler and network access. On aarch64, set `RUSTFLAGS="-C force-unwind-tables=yes"`, because upstream's `.cargo/config.toml` targets Neoverse V2 there. Pushing a `v*` tag runs `.github/workflows/release.yml`, which publishes the x86_64 bundle as `remanie-train-harness-<tag>-linux-x86_64.tar.gz` beside its `.sha256`.

The output directory must be new. It holds `remanie-train-harness`, `grok` and `rg`, each a launcher that runs the binary in `bin/` with the bundle's own loader and libraries from `lib/`, the license notices in `licenses/` (Grok Build's in `licenses/grok-build/`), `contract.json`, and `manifest.json` with the architecture and each file's sha256. The `remanie-train-harness` launcher sets `REMANIE_GROK` to the bundle's `grok`. Outside a bundle, set `REMANIE_GROK` to a Grok Build binary built at the pinned commit.

## Checks

`checks/oracle/oracle.py` runs each scenario in `checks/oracle/scenarios/` twice against a scripted endpoint, once through Grok Build run directly with the oracle's own `config.toml` and `assets/agent.md`, and once through the runner. It fails a scenario unless every chat-completions body the endpoint received is byte-identical apart from the scenario's injected `sampling_params`, the runner's `config.toml` matches the oracle's own, the trace holds what the endpoint received, every forwarded request carries the endpoint key, and the title request stayed in the runner. `checks/oracle/baseline/` holds its last run. [checks/README.txt](checks/README.txt) lists every check and how to run it.

```sh
cargo test --locked
cargo clippy --all-targets --locked -- -D warnings
```

`rust-toolchain.toml` pins Rust 1.94.0 with rustfmt and clippy.

## What the sandbox must provide

- Linux on the bundle's CPU architecture.
- `/bin/bash` and `setsid`.
- Network access to `base_url`.

## Known limits

- Code in the sandbox can read the key from the runner's process through `/proc`, and can send requests through the proxy, which adds the key. Use a key with a spending cap, or an endpoint that needs none.
- Grok Build's tools reach any path the sandbox user can.
- Harbor's local Docker environment does not enforce a task's disk quota.
- On macOS the runner cannot be a subreaper, so processes Grok Build's commands start can outlive a stopped attempt there.

[NOTICE](NOTICE) names Grok Build's license and the pinned commit.
