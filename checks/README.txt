Checks

Run each command from the repository root. .venv is the Python environment
from harbor/README.md with pytest also installed.

cargo fmt --check
  Rust formatting against rustfmt.toml.

cargo clippy --locked --all-targets -- -D warnings
  Rust lints, with every warning an error.

cargo test --locked
  Rust unit tests. Needs rg on PATH or RG_BIN_PATH.

python3 checks/licenses.py --check
  Fails when THIRD-PARTY-NOTICES differs from the notices rebuilt from
  Cargo.lock. It runs cargo --offline, so run cargo fetch --locked first.
  Without --check it rewrites the file.

.venv/bin/python -m pytest harbor/ checks/ -q
  Adapter, trace reader, and packaging unit tests. Every job file and the
  check_harbor.py settings must pass assets/contract.json. Set REMANIE_BUNDLE
  to a bundle directory to also check that the bundle carries contract.json
  and a working rg. That part needs Linux.

python3 checks/make_trace_fixture.py target/debug/remanie-train-harness
  Rewrites harbor/fixtures/trace.jsonl and harbor/fixtures/requests.jsonl
  from one run against a scripted endpoint: a 503 retry, two tool calls, and
  a compaction. The trace reader tests read both files. Rerun it after any
  change to the trace or the request body, and commit the result.

.venv/bin/python checks/check_harbor.py
  Real Harbor trials in Docker against a scripted local endpoint, with no paid
  model calls. Needs Docker and the bundle in runtime/ built for the Docker
  worker's CPU architecture. --runtime-dir picks another bundle.
  --output-dir picks a new directory for the trial output, which otherwise
  goes under the system temp folder. Harbor bind-mounts each trial's log
  directory, so on Colima or Docker Desktop the output directory must be
  under a path the VM shares, such as $HOME. --model-host is the host's
  address as a container sees it. It defaults to host.docker.internal. On
  Linux Docker, pass the docker0 bridge address, usually 172.17.0.1.
  It runs five trials:
    completed       all eight tools and one compaction, graded with reward 1
    timeout         Harbor's timeout cancels the runner, still graded
    cleanup_error   the stop is never confirmed, verifier skipped
    crash           the runner is killed, verifier skipped
    infrastructure  the endpoint returns 401, exit 70, verifier skipped
  In the graded trials the verifier also checks that a killed background
  task is gone and a running one is still alive. A passing reward means the
  integration works. It is not a Terminal-Bench score.

CI

.github/workflows/checks.yml runs on every pull request and every push to
main. It runs cargo fmt --check, cargo clippy, cargo test, and licenses.py
--check, then builds and packages the bundle, runs pytest with
REMANIE_BUNDLE set, and compiles every tracked Python file. It does not run
check_harbor.py, so run that by hand after a change to the adapter, the
launch or stop scripts, the exit codes, or the background-task rules.
