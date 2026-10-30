Checks

Run each command from the repository root. .venv is a Python environment
with pytest installed.

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

.venv/bin/python -m pytest checks/ -q
  Packaging unit tests. Set REMANIE_BUNDLE to a bundle directory to also
  check that the bundle carries contract.json and a working rg. That part
  needs Linux.

python3 checks/make_trace_fixture.py target/debug/remanie-train-harness   (on Linux; the script prints the Docker one-liner for a Mac)
  Rewrites checks/fixtures/trace.jsonl and checks/fixtures/requests.jsonl
  from one run against a scripted endpoint: a 503 retry, three tool calls,
  and a compaction whose request drops the first turn to fit the window. The
  trace reader tests in remanie-tooling read the trace.
  Rerun it after any change to the trace or the request body, commit the
  result, and copy trace.jsonl to remanie-tooling's
  research/agents/remanie_train/fixtures/.

CI

.github/workflows/checks.yml runs on every pull request and every push to
main. It runs cargo fmt --check, cargo clippy, cargo test, and licenses.py
--check, then builds and packages the bundle. Next it rerecords the trace
fixture and fails when checks/fixtures/requests.jsonl differs, runs pytest
with REMANIE_BUNDLE set, and compiles every tracked Python file.
