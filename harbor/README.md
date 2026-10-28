# Terminal-Bench with remanie-train-harness

This adapter runs our Rust executable inside Harbor's task environment. Harbor supplies the task, enforces its deadline, runs its verifier, and saves the result. The evaluation uses OpenRouter's `x-ai/grok-4.7`.

The eight tool definitions and prompt assets stay unchanged. Their handlers live in this standalone runner. The adapter passes the task instruction without adding another system prompt.

## Prepare the runtime

Build on Debian or Ubuntu with Rust 1.94. Install Bash, ripgrep, Python 3, and binutils. PDF text extraction optionally needs Poppler in the task environment. From the repository root:

```sh
cargo build --release --locked
python3 package_linux.py \
  target/release/remanie-train-harness \
  runtime
```

The output directory must be new. Packaging copies the executable, ripgrep, their Linux loader and shared libraries. It requires the Debian copyright notices and any common license texts they reference. It records the bundled files' hashes in `manifest.json`. Harbor installs this bundle under `/opt/remanie-train-harness`. The bundled loader avoids requiring a particular glibc version in each task image.

Use an AMD64 build and worker for comparable benchmark results. The local integration fixture uses a native ARM64 runner and Debian image on this Mac. Match the runner and task-image architectures. Mixing architectures under emulation can change the reported process identity and prevent confirmed shutdown.

## Install Harbor and run

Run these commands from the repository root:

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python \
  -r harbor/requirements.txt
export PYTHONPATH="$PWD/harbor"
```

Set `OPENROUTER_API_KEY` in the host environment using your credential manager. Do not put it in the job JSON or pass its value on the command line.

```sh
.venv/bin/harbor run \
  --config harbor/terminal-bench-2.1-smoke.json
```

For a preflight that makes no model calls:

```sh
.venv/bin/harbor run \
  --config harbor/terminal-bench-2.1-smoke.json \
  --dry-run --skip-credential-checks
```

The adapter uses Harbor's resolved model connection. It puts the selected endpoint in the runner's JSON configuration and passes the selected credential through the execution environment. Harbor's `base_url` and `model_api_key_name` agent settings select a custom endpoint and credential. The endpoint must support OpenAI Chat Completions under a `/v1` URL and be reachable from the task sandbox.

The credential is available inside the sandbox. The runner removes it from its environment before starting tools, but this does not protect it from code in the same sandbox. Use a restricted evaluation credential or a reachable gateway with a limited token. The adapter no longer runs a local proxy or enforces its former request and credential limits.

Cleanup uses the process identity emitted before the runner starts. The host must also confirm that the runner exited with code zero. A crash or unconfirmed shutdown makes Harbor skip grading and tear down the task.

Choose Docker or your Azure environment adapter in Harbor's environment configuration. Remanie uses Harbor's file-transfer and command-execution methods without checking the provider name. The environment must run Linux, provide Bash and `setsid`, and match the runtime bundle's CPU architecture. The model endpoint must be permitted by the task's network policy.

Remanie reads the process identity from live output when available, or from the completed command result. Graceful timeout cleanup needs live output and a second command to stop the process. Without a captured identity, an interrupted run skips grading and requires sandbox teardown.

Unit tests cover live and buffered output through a mocked Harbor environment, including the Azure provider name. The integration fixtures run on real Docker. Live Azure execution has not been verified.

## What the smoke configuration measures

The dataset is pinned to revision 6, digest `sha256:7d7bdc1cbedad549fc1140404bd4dc45e5fd0ea7c4186773687d177ad3a0699a`. It contains 89 tasks. The smoke configuration runs one attempt each on:

- `terminal-bench/openssl-selfsigned-cert`
- `terminal-bench/log-summary-date-ranges`
- `terminal-bench/cancel-async-tasks`

It runs one task at a time, with no automatic retries or task resource overrides. Each selected task allows 900 seconds for the agent. Harness settings are 100 turns, 32,768 output tokens per response, high reasoning effort, and compaction at an estimated 400,000 tokens with a 4,096-token summary limit.

These three attempts check whether the integration works. They are not a full Terminal-Bench score. For a full run, copy the configuration, remove `task_names`, and choose the attempt count deliberately. The benchmark repository recommends at least five attempts per task. Keep the task-specific time limits; some are much longer than 900 seconds.

Results go to `eval-results/`. Each trial saves Harbor's verdict, `agent/result.json`, `agent/runtime-stderr.txt`, and `agent/remanie/trace.jsonl`. The adapter sums token usage and cost reported by the endpoint, including summary calls. A request without a recorded response leaves the totals unknown. Missing usage or cost also leaves the corresponding total unknown. The model's final text does not set the reward.

OpenRouter charges for inference, including reasoning and repeated input tokens. A dry run checks configuration and dataset metadata, not access to the model or available credits. Use a capped evaluation key for a paid run.

## Limits

The local Docker backend enforces CPU and memory limits but does not enforce the dataset's per-task disk quota. Task-created files consume the Docker VM's shared disk. Harbor controls task verification; this adapter does not change a benchmark's shared or separate verifier mode.

The task can modify its own files and logs. Treat logs as evidence to inspect, not trusted grading authority. Container deletion remains necessary after crashes or forced termination.

Sources: [Terminal-Bench 2.1](https://hub.harborframework.com/datasets/terminal-bench/terminal-bench-2-1), [Harbor custom agents](https://docs.harborframework.com/agents/custom-agents), [OpenRouter Grok 4.7](https://openrouter.ai/x-ai/grok-4.7).

Local verification results are in [validation.txt](../reviews/validation.txt). The reproducible integration check is in [checks/](../checks/README.txt). The [earlier Grok 4.7 smoke report](../reviews/terminal-bench-2.1-smoke-2026-10-27.txt) tested the larger predecessor, before extraction.
