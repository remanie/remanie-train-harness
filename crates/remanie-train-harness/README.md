# remanie-train-harness

This Rust program runs one coding attempt against a Chat Completions endpoint. It uses Grok Build's native tools and compaction helper. Harbor can run it inside a benchmark task environment and grade the result. The evaluation configuration selects Grok 4.7 through OpenRouter; it does not require Miles.

See the [Terminal-Bench evaluation guide](harbor/README.md) for setup, the pinned three-task configuration, and validation limits. The existing Miles adapter remains available separately in `python/`.

The supported workload is one sequential agent per disposable sandbox, with many sandboxes running independently. The program does not create a sandbox. Native shell and file tools can access the environment in which the program runs, including absolute paths. The launcher must keep trainer credentials, private tests, and the verifier outside that environment and destroy it after every attempt, including crashes or forced termination.

## Build and test

Use the repository's Rust 1.94 toolchain and install DotSlash as described in README.upstream.md. Debug builds also require `rg` on PATH, or an absolute `RG_BIN_PATH`. Release builds bundle ripgrep. When using the Python adapter, put `rg` on PATH. Run these commands from the repository root:

```sh
cargo build -p remanie-train-harness
cargo test -p remanie-train-harness
PYTHONPATH=/path/to/miles:$PWD/crates/remanie-train-harness/python python3 -m unittest discover -s crates/remanie-train-harness/python -p 'test_miles_grok.py' -v
```

The tests use temporary workspaces and a local fake model endpoint. They run real file and process tools. They do not run an optimizer update or establish model learning quality.

## One attempt

The executable reads one JSON object from standard input and writes one JSON result to standard output. Diagnostics go to standard error. The workspace must already exist. The session directory must not exist, and its parent must exist. Reusing a session directory fails instead of overwriting previous evidence.

```json
{
  "base_url": "https://openrouter.ai/api/v1",
  "model": "x-ai/grok-4.7",
  "api_key_env": "OPENROUTER_API_KEY",
  "user_query": "Fix the failing tests in this repository.",
  "workspace": "/workspace/task",
  "session_dir": "/attempts/unique-attempt",
  "sampling_params": {"temperature": 1.0},
  "max_turns": 100,
  "max_response_tokens": 4096,
  "deadline_secs": 600,
  "compaction_threshold_tokens": 0,
  "compaction_max_tokens": 2048
}
```

For a direct run, `api_key_env` names the environment variable containing the bearer credential. It is optional for endpoints that need no authentication. The key is excluded from request JSON and traces, but environment removal is not an operating-system security boundary. The Harbor adapter keeps the provider key on the host and gives the task a temporary, limited proxy credential.

The endpoint supplies the model chat template and tool parser. The runner rejects unexpected configuration fields. `sampling_params` accepts `reasoning` and `reasoning_effort` as well as its existing sampling fields.

`user_query` is raw task text. The program wraps it once with the stored template. The system prompt is exactly:

```text
You are an autonomous coding agent. Your main goal is to complete the user's request denoted with the <user_query> tag.
```

The eight tools are `run_terminal_cmd`, `get_terminal_command_output`, `kill_terminal_command`, `read_file`, `search_replace`, `list_dir`, `grep`, and `todo_write`. `assets/tools.json` is the agreed name-keyed dictionary. The program converts it to the API's function-tool array. Main descriptions contain the original first sentence. Argument descriptions remain unchanged. The dictionary also enforces its stated argument types and bounds.

The profile disables skills, subagents, memory, IDE instructions, automatic instruction-file reads, and tool-result reminders. It enables the native empty-string overwrite guard. Commands receive an explicit environment and private HOME and TMPDIR. Background commands preserve their native wait, poll, and cancellation behavior. Waiting is separate from killing a process. Todo state and background task IDs remain alive across compaction and are discarded between attempts.

This first training path supports text results. The native `read_file` implementation remains in use, but the adapter returns a tool error for image outputs. PDF text reads can use `format="text"`. Image and video training require another model path.

## Compaction and training records

Set a positive `compaction_threshold_tokens` to enable automatic compaction. Zero disables it. Compaction works with the evaluation endpoint directly. If training through Miles later, use its experimental v2 session implementation; v1 cannot preserve arbitrary replacement histories for training.

The trigger starts with the model endpoint's `usage.total_tokens`. It adds the serialized bytes of new tool results and a response-token reserve. Byte counts deliberately overestimate tokens for common byte-based tokenizers, but this is not an exact count for every chat template. Set the threshold below the model's context limit with room for tool results and the summary request. A server context overflow ends the attempt with an error. The program never silently discards old messages to force a request to fit.

Compaction makes one summary request through the same endpoint, model, and sampling settings. It accepts a finished, nonempty text summary, cleans it with Grok's helper, then starts a new conversation containing the original system prompt, original task, summary, and a snapshot of todo items and background task metadata. The tool executor stays alive. A malformed, empty, or truncated policy summary ends the attempt as a gradable `invalid_response` without replacing the old context. Transport and server failures remain ungraded errors. A short valid summary is accepted. Automatic retries are disabled because retries affect the training trajectory.

The summary generation is intentionally a training target under the attempt's final reward. Its later copy is prompt context. Miles v2 retains earlier segments and masks shared generated prefixes so they are not trained twice. This behavior still needs validation with the selected model's tokenizer and parser before a real training run.

The local `trace.jsonl` records exact request and response JSON, tool results, compacted contexts, and the outcome. It is an execution journal. Miles owns the authoritative token IDs, log probabilities, and loss masks. Export those records through Miles when retaining a training dataset.

Each request and response is limited to 16 MiB, each native process log to 16 MiB, and the local journal to 256 MiB. The connection timeout is 30 seconds. Model requests share the attempt deadline, which defaults to ten minutes for direct runs. The Harbor adapter leaves the task deadline to Harbor. `max_turns` is at most 512 to fit Miles v2's 1024-node limit even when each turn triggers compaction. The outer sandbox must also enforce CPU, memory, disk, process, and network limits.

## Miles adapter and grading

See `python/README.txt` for the trusted adapter configuration. Use `miles_grok.generate` as the custom generate entry point and `miles_grok.reward` as the reward hook. The entry point rejects samples that already carry a reward, because Miles would otherwise skip the verifier's reward hook. It also rejects the reserved `rl_harness` metadata key so an old or forged outcome cannot survive a failed agent process. The adapter accepts a single text user message, launches the Rust executable, and invokes a configured trusted verifier after an ordinary task outcome. It never trusts a reward written by the model or passed in task metadata.

The result includes `status`, `final_answer`, `turns`, and `compactions`, with `error` on infrastructure failure. `completed`, `max_turns`, `max_tokens`, `invalid_response`, and `deadline` can be graded. `error` and `cancelled` are not assigned fabricated rewards. The reward hook fails the batch on an ungraded attempt.

The adapter requires an already-provisioned, unique workspace and a fresh session path for every sample. Repeated samples cannot share a fixed dataset workspace. This local Miles adapter does not use the separate Harbor evaluation adapter. Connecting training to a provisioner remains separate work. Normal exits and signals clean up native process groups; sandbox deletion is still required after a crash or SIGKILL.

## Source and scope

This checkout starts from Grok Build commit `2bdd1d6a6369de0e8c68132ea4539e9abd9e14a8`. The first version retains the upstream Cargo dependency graph. It omits the production application from the execution path. It does not extract or rewrite the native tools.

The Miles adapter was checked against the PoC pin `203a8d279d80dc7d8d4691f442dc7fd6d6f64e2e` and the local v0.1.3 tag. The relevant agent wrapper and session sources match between those revisions. Upstream license and third-party notices remain in this checkout.

The two review cycles and test results are recorded in `reviews/`. The original runner results are in `reviews/validation.txt`. The rename and Harbor checks are in `reviews/harbor-validation.txt`. The [live Grok 4.7 smoke report](reviews/terminal-bench-2.1-smoke-2026-10-27.txt) records two initial passes, one HTTP 502 interruption, and a passing fresh retry.
