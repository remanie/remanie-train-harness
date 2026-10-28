# Harbor adapter

`remanie_agent.py` defines `RemanieAgent`, a Harbor installed agent with import path `remanie_agent:RemanieAgent`. It installs the runtime bundle in the task sandbox, runs one attempt, and turns the runner's exit code into Harbor's grade-or-skip decision. `requirements.txt` pins Harbor 0.25.0.

## Build the bundle

On Debian or Ubuntu, with the CPU architecture of the Harbor workers, run from the repository root:

```sh
cargo build --release --locked
python3 package_linux.py target/release/remanie-train-harness runtime
```

The output directory must be new. It holds the runner, `rg`, their Linux loader and shared libraries, the license notices, and `manifest.json` with the architecture and each file's sha256.

## Job settings

| Setting | Meaning |
|---|---|
| `model_name` | Required. The adapter refuses a job without one. |
| `base_url` | Optional. Defaults to Harbor's model connection. It must be reachable from the sandbox. |
| `model_api_key_name` | Optional. Names the host variable that holds the key. When it is unset, the runner sends no key. |
| `kwargs.runtime_dir` | The bundle directory. |
| other `kwargs` | Runner options from [the input table](../README.md#input): `max_turns`, `deadline_secs`, `sampling_params`, `compaction`, `retry`, `keep_background_tasks`. |

Before any sandbox starts, the adapter refuses the job when the runner options fail `assets/contract.json`, when a key name is set and the host has no value for it, or when the bundle lacks `manifest.json` or an executable. `terminal-bench-2.1-smoke.json` is a complete job.

## Install and launch

`install` uploads the bundle to `/opt/remanie-train-harness` and fails when the sandbox's `uname -m` differs from the manifest's architecture. It links nothing into `/usr/local/bin`, so the task's own `rg` and `PATH` stay as they were. The runner finds the bundled `rg` through `RG_BIN_PATH`.

`run` writes `input.json`. When a key is selected, it writes the key to a 0600 file outside `/logs`, which the launch script reads into the runner's environment and deletes. The key never appears in a command line or in the job file. The adapter starts the runner with `setsid --wait bash -c` through `environment.exec`, so the provider's default shell does not matter.

## What Harbor grades

- Exit 0. The adapter returns and Harbor runs the verifier.
- Exit 70. The adapter raises `RemanieInfrastructureError` and Harbor skips the verifier.
- Any other exit. The adapter sends TERM to the runner's process group, then KILL after 5 s, and raises `RemanieInfrastructureError`.
- Harbor's agent timeout. The adapter stops the process group the same way and re-raises the cancellation. Harbor records `AgentTimeoutError` and still runs the verifier.

On every path the adapter copies `result.json` into `context.metadata["remanie"]`. When `usage.complete` is true it also sets `n_input_tokens`, `n_output_tokens`, and `n_cache_tokens`.

Each trial directory holds `agent/result.json`, `agent/remanie/trace.jsonl`, and `agent/runtime-stderr.txt`. The stderr file is for diagnostics and has no fixed format.

## Train with Miles

Miles' Harbor connector runs the agent inside the trial and creates one session per attempt. Set `base_url` to that session, `http://<session-server>/sessions/<id>/v1`. The session server does not check a key, so leave `model_api_key_name` unset. A variable holding any dummy value also works. Pass the launcher's sampling parameters as `sampling_params`. The server rejects a `temperature` that differs from the training temperature.

Leave `compaction` unset under the v1 linear session server. A compaction opens a new history, which v1 rejects with 400 `rollback failed`. Turn it on only with `--use-session-server v2`.

## Read the trace

`remanie_trace.py` uses only the standard library, so a trainer can copy the one file.

```python
import remanie_trace

trace = remanie_trace.read("eval-results/<job>/<trial>/agent/remanie")
trace.outcome                    # the result object, or None when the runner died
body = trace.request_body(0)     # the exact chat-completions body of call 0
trace.usage()                    # token totals, with complete=False if any response lacked usage
```

## Run the smoke job

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r harbor/requirements.txt
export PYTHONPATH="$PWD/harbor"
.venv/bin/harbor run --config harbor/terminal-bench-2.1-smoke.json
```

Keep `OPENROUTER_API_KEY` in the host environment. Add `--dry-run --skip-credential-checks` to check the job without model calls. The job runs one attempt each of three Terminal-Bench 2.1 tasks, one at a time with no retries, from the dataset pinned by digest in the job file. It checks the integration and is not a benchmark score.
