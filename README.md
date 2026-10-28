# remanie-train-harness

A small Rust program that lets a model work on a coding task using eight tools. It can summarize long conversations and saves an execution trace. Harbor runs each task in a disposable environment and grades the result.

The tools are `run_terminal_cmd`, `get_terminal_command_output`, `kill_terminal_command`, `read_file`, `search_replace`, `list_dir`, `grep`, and `todo_write`. Their names and argument definitions match the agreed Grok Build tool set. Each tool description keeps its original first sentence.

## Build and check

Install Rust 1.94, Bash and ripgrep. Linux and macOS are supported.

```sh
cargo build --release --locked
cargo test --locked
cargo clippy --all-targets --locked -- -D warnings
```

`rg` must be on PATH, or `RG_BIN_PATH` must name its absolute location. PDF text extraction additionally needs `pdfinfo` and `pdftotext`, usually supplied by Poppler. Image output, rendered documents and PPTX extraction are not supported.

See [Harbor setup](harbor/README.md) for a Linux runtime bundle and the pinned Terminal-Bench configuration. [The integration check](checks/README.txt) tests all eight tools, compaction and timeout cleanup without paid inference.

## One attempt

The executable reads one JSON object from stdin and writes one JSON result to stdout. Diagnostics go to stderr. The workspace must exist; the session directory must be new and its parent must exist.

```json
{
  "base_url": "https://openrouter.ai/api/v1",
  "model": "x-ai/grok-4.7",
  "api_key_env": "OPENROUTER_API_KEY",
  "user_query": "Fix the failing tests in this repository.",
  "workspace": "/workspace/task",
  "session_dir": "/attempts/unique-attempt",
  "sampling_params": {"reasoning": {"effort": "high"}},
  "max_turns": 100,
  "max_response_tokens": 4096,
  "deadline_secs": 600,
  "compaction_threshold_tokens": 0,
  "compaction_max_tokens": 2048
}
```

The system prompt is exactly:

```text
You are an autonomous coding agent. Your main goal is to complete the user's request denoted with the <user_query> tag.
```

The runner wraps the task once in `<user_query>` tags and sends the eight definitions from `assets/tools.json` as native function tools. The endpoint owns the model's chat template and tool parser. The runner preserves assistant JSON when replaying conversation history.

`run_terminal_cmd` starts a fresh Bash shell in the task workspace. Its environment is explicit, with a private HOME and TMPDIR. A command that outlasts `block_until_ms` becomes a background task; use the output and kill tools to manage it. Shutdown terminates managed process groups. The launcher must still destroy the sandbox after crashes or forced termination.

The file tools accept absolute paths and do not restrict access to the workspace. Both file tools and terminal commands expand `~` to the attempt's private home directory.

`search_replace` refuses an ambiguous match and cannot overwrite a nonempty file with an empty `old_string`. It preserves untouched bytes, including line endings. Atomic replacement prevents partial files but does not lock out other processes.

Todo state and background task metadata survive compaction and reset for each attempt.

## Compaction and traces

A positive `compaction_threshold_tokens` enables compaction. Zero disables it. The estimate uses endpoint token usage, new tool-result bytes and a response reserve. Set it below the model's context limit. A context overflow ends the attempt; the runner does not silently discard messages.

Compaction makes one summary request through the same endpoint. A finished, nonempty summary replaces conversation history together with the original system prompt, task and current tool state. Earlier requests remain in `trace.jsonl`. Malformed summaries stop as `invalid_response`; endpoint failures stop as `error`. There are no automatic model retries.

The trace records request and response JSON, tool results, summaries and the outcome. A training backend must supply token IDs, log probabilities and loss masks, associate them with a trusted reward, and update the model. This runner does not perform those steps.

The result includes `status`, `final_answer`, `turns` and `compactions`, plus `error` on infrastructure failure. Harbor independently runs the verifier. The model's final answer does not set the reward.

## Isolation and limits

Run one attempt per disposable sandbox. Keep private grading material outside the task. The Harbor adapter passes the selected model credential into the runner's environment. The runner removes it before starting tools, but code inside the sandbox may still recover it from process memory or operating-system interfaces. Use a restricted evaluation credential or a reachable gateway with a limited token. Credential and spending limits belong to the provider or gateway.

Each model request/response is capped at 16 MiB, each command log at 16 MiB, and the trace at 256 MiB. The launcher owns CPU, memory, disk and network limits. Harbor's selected environment provider must enforce those limits. The local Docker backend does not enforce per-task disk quotas.

This project derives from Grok Build. [NOTICE](NOTICE) identifies the original source and retained code. The source tree contains one Rust package, the Harbor adapter, runtime packaging and their checks.

See [validation](reviews/validation.txt) for the extraction checks. The [earlier smoke report](reviews/terminal-bench-2.1-smoke-2026-10-27.txt) tested the larger predecessor, not this extracted implementation.
