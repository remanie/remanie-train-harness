# Terminal-Bench with remanie-train-harness

This adapter runs our Rust executable inside Harbor's task environment. Harbor supplies the task, enforces its deadline, runs its verifier, and saves the result. The evaluation uses OpenRouter's `x-ai/grok-4.7`. Miles is not involved.

The eight tools and their prompt assets stay unchanged. The adapter passes the task instruction to the runner without adding another system prompt.

## Prepare the runtime

Build on Linux with Rust 1.94 and the prerequisites in the [upstream guide](../../../README.upstream.md). Install ripgrep, Python 3, and binutils. From the repository root:

```sh
cargo build -p remanie-train-harness
python3 crates/remanie-train-harness/package_linux.py \
  target/debug/remanie-train-harness \
  crates/remanie-train-harness/runtime
```

The output directory must be new. Packaging copies the executable, ripgrep, their Linux loader and shared libraries, and available Debian license notices. It records their hashes in `manifest.json`. Harbor installs this bundle under `/opt/remanie-train-harness`. The bundled loader avoids requiring a particular glibc version in each task image.

Use an AMD64 build and worker for comparable benchmark results. Local validation on this Mac uses a native ARM64 runner and emulated AMD64 task commands. This tests the integration but changes task execution speed. The runner still uses the upstream development build profile.

## Install Harbor and run

Run these commands from the repository root:

```sh
uv venv --python 3.12 crates/remanie-train-harness/.venv
uv pip install --python crates/remanie-train-harness/.venv/bin/python \
  -r crates/remanie-train-harness/harbor/requirements.txt
export PYTHONPATH="$PWD/crates/remanie-train-harness/harbor"
```

Set `OPENROUTER_API_KEY` in the host environment using your credential manager. Do not put it in the job JSON or pass its value on the command line.

```sh
crates/remanie-train-harness/.venv/bin/harbor run \
  --config crates/remanie-train-harness/harbor/terminal-bench-2.1-smoke.json
```

For a preflight that makes no model calls:

```sh
crates/remanie-train-harness/.venv/bin/harbor run \
  --config crates/remanie-train-harness/harbor/terminal-bench-2.1-smoke.json \
  --dry-run --skip-credential-checks
```

The adapter keeps the provider key in the host process. A temporary proxy gives each task a separate credential, restricts it to the configured model and token limits, and closes when the attempt ends. Timeout cleanup uses the process identity captured by the host before the model runs. If shutdown cannot be confirmed, Harbor skips grading and tears down the task. The task receives no OpenRouter key. This adapter currently targets local Docker; the container must be able to reach `host.docker.internal`.

## What the smoke configuration measures

The dataset is pinned to revision 6, digest `sha256:7d7bdc1cbedad549fc1140404bd4dc45e5fd0ea7c4186773687d177ad3a0699a`. It contains 89 tasks. The smoke configuration runs one attempt each on:

- `terminal-bench/openssl-selfsigned-cert`
- `terminal-bench/log-summary-date-ranges`
- `terminal-bench/cancel-async-tasks`

It runs one task at a time, with no automatic retries or task resource overrides. Each selected task allows 900 seconds for the agent. Harness settings are 100 turns, 32,768 output tokens per response, high reasoning effort, and compaction at an estimated 400,000 tokens with a 4,096-token summary limit.

These three attempts check whether the integration works. They are not a full Terminal-Bench score. For a full run, copy the configuration, remove `task_names`, and choose the attempt count deliberately. The benchmark repository recommends at least five attempts per task. Keep the task-specific time limits; some are much longer than 900 seconds.

Results go to `crates/remanie-train-harness/eval-results/`. Each trial saves Harbor's verdict, `agent/result.json`, `agent/runtime-stderr.txt`, and `agent/remanie/trace.jsonl`. The adapter sums token usage and cost reported by the endpoint, including summary calls. If usage is missing, it leaves the corresponding total unknown. The model's final text does not set the reward.

OpenRouter charges for inference, including reasoning and repeated input tokens. A dry run checks configuration and dataset metadata, not access to the model or available credits. Use a capped evaluation key for a paid run.

## Limits

The local Docker backend enforces CPU and memory limits but does not enforce the dataset's per-task disk quota. Task-created files consume the Docker VM's shared disk. Harbor controls task verification; this adapter does not change a benchmark's shared or separate verifier mode.

The task can modify its own files and logs. Treat logs as evidence to inspect, not trusted grading authority. Container deletion remains necessary after crashes or forced termination.

Sources: [Terminal-Bench 2.1](https://hub.harborframework.com/datasets/terminal-bench/terminal-bench-2-1), [Harbor custom agents](https://docs.harborframework.com/agents/custom-agents), [OpenRouter Grok 4.7](https://openrouter.ai/x-ai/grok-4.7).

Local verification results are in [harbor-validation.txt](../reviews/harbor-validation.txt). The reproducible integration check is in [checks/](../checks/README.txt). No live model result is included in those checks.
