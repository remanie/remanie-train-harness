# Harbor adapter

`remanie_agent.py` defines `RemanieAgent`, a Harbor installed agent with import path `remanie_agent:RemanieAgent`. It installs the runtime bundle in the task sandbox, runs one attempt, and uses the runner's exit code to tell Harbor whether to run the verifier. `requirements.txt` pins Harbor 0.25.0.

## Build the bundle

On Debian or Ubuntu, with the CPU architecture of the Harbor workers, run from the repository root:

```sh
cargo build --release --locked
python3 package_linux.py target/release/remanie-train-harness runtime
```

The output directory must be new. It holds the runner, `rg`, their Linux loader and shared libraries, the license notices, `contract.json`, and `manifest.json` with the architecture and each file's sha256.

## Job settings

| Setting | Meaning |
|---|---|
| `model_name` | Required. The adapter refuses a job without one. |
| `base_url` | Optional. Without it, Harbor's model connection supplies OpenRouter's URL and requires `OPENROUTER_API_KEY`. The URL must be reachable from the sandbox. |
| `model_api_key_name` | Optional. Names the host variable that holds the key. With a custom `base_url` and no key name, the runner sends no key. |
| `kwargs.runtime_dir` | The bundle directory. |
| other `kwargs` | Runner settings from [the input table](../README.md#input), such as `max_turns`, `deadline_secs`, `sampling_params`, and `compaction`. |

Harbor supplies `model`, `base_url`, `api_key_env`, `user_query`, `workspace`, and `session_dir`, so `kwargs` must not set them. Before any sandbox starts, the adapter refuses the job in these cases:

- The runner settings fail the input schema in the bundle's `contract.json`.
- The job sets a prompt template.
- The bundle lacks `remanie-train-harness`, `rg`, `manifest.json`, or `contract.json`.

When the selected key has no value on the host, `run` raises before it executes anything in the sandbox.

`terminal-bench-2.1-smoke.json` is a complete job. `harbor/test_remanie_agent.py` checks every job file in the repository against `assets/contract.json`.

## Install and launch

`install` uploads the bundle to `/opt/remanie-train-harness` and fails when the sandbox's `uname -m` differs from the manifest's architecture. It links nothing into `/usr/local/bin`, so the task's own `rg` and `PATH` stay as they were. The runner finds the bundled `rg` through `RG_BIN_PATH`.

`run` writes `input.json`. When a key is selected, it writes the key to a 0600 file outside `/logs`, which the launch script reads into the runner's environment and deletes. The key never appears in a command line or in the job file. The adapter starts the runner with `setsid --wait bash -c` through `environment.exec`, so the provider's default shell does not matter.

## What Harbor grades

- Exit 0. The adapter returns and Harbor runs the verifier.
- Exit 70. The adapter raises `RemanieInfrastructureError` and Harbor skips the verifier.
- Any other exit. The runner died. The adapter raises an error with the exit code and Harbor skips the verifier. Commands the model runs share the runner's user and PID namespace, so a command can kill the runner. The adapter handles that exit the same way.
- Harbor's agent timeout. The adapter sends TERM to the runner's process group, waits up to 5 s, and then sends KILL if the runner is still running. If the runner exits 0 in that time, the adapter re-raises the cancellation, and Harbor records `AgentTimeoutError` and still runs the verifier. Otherwise the adapter raises an error, and Harbor skips the verifier.

On every path the adapter sets `context.metadata["remanie"]` to the runner's `exit_code`, the bundle's `bundle_sha256`, and `result`, the parsed `result.json`. `result` is null when the file is missing, empty, or invalid, and for an invalid file the adapter also sets `result_invalid` to true. When `usage.complete` is true the adapter also sets `n_input_tokens`, `n_output_tokens`, and `n_cache_tokens`.

Each trial directory holds `agent/input.json`, `agent/result.json` (the runner's stdout), `agent/remanie/trace.jsonl`, `agent/remanie/result.json`, and `agent/runtime-stderr.txt`. The stderr file is empty unless the runner panicked or failed to start.

## Train with Miles

Miles' Harbor connector runs the agent inside the trial and creates one session per attempt. Set `base_url` to that session, `http://<session-server>/sessions/<id>/v1`. The session server does not check a key, so leave `model_api_key_name` unset. A variable holding any dummy value also works. Pass the launcher's sampling parameters as `sampling_params`. The server rejects a `temperature` that differs from the training temperature.

Leave `compaction` unset under the v1 linear session server. A compaction opens a new history, which v1 rejects with 400 `rollback failed`. Turn it on only with `--use-session-server v2`.

## Read the trace

`remanie_trace.py` uses only the standard library and needs Python 3.10 or newer, so a trainer can copy the one file.

```python
import remanie_trace

trace = remanie_trace.read("eval-results/<job>/<trial>/agent/remanie")
trace.outcome                            # "completed", ..., or None when the trace has no result event
body = trace.request_body(trace.calls[0])  # the chat-completions body of the first call
trace.usage()                            # token totals, with complete False when any are missing
```

`request_body()` equals the sent body as JSON, not byte for byte. Python's `json.dumps` writes the float `0.00001` as `1e-05`, for example, so do not compare hashes of rebuilt bodies.

## Run the smoke job

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r harbor/requirements.txt
export PYTHONPATH="$PWD/harbor"
.venv/bin/harbor run --config harbor/terminal-bench-2.1-smoke.json
```

Keep `OPENROUTER_API_KEY` in the host environment. Add `--dry-run --skip-credential-checks` to check the job without model calls. The job runs one attempt each of three Terminal-Bench 2.1 tasks, one at a time with no retries, from the dataset pinned by digest in the job file. Its reward checks the integration and is not a benchmark score.
