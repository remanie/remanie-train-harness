Checks

Run each command from the repository root. .venv is a Python environment
with pytest installed. GROK is xai-grok-pager built at upstream.json's
commit. On Linux, python3 build_grok.py builds it into
target/grok-build/out/grok. On a Mac, for the oracle only:
    git clone https://github.com/xai-org/grok-build && cd grok-build
    git checkout 2bdd1d6a6369de0e8c68132ea4539e9abd9e14a8
    cargo build -p xai-grok-pager-bin    (needs dotslash on PATH for protoc)

cargo fmt --check
  Rust formatting against rustfmt.toml.

cargo clippy --locked --all-targets -- -D warnings
  Rust lints, with every warning an error. The orphan sweep in src/grok.rs
  compiles only on Linux, so run it there too.

cargo test --locked
  Rust unit tests: the contract's outcomes, input parsing, request kinds,
  sampling_params injection, chunk-stream reassembly, and the mapping from
  Grok Build's exit and output to an outcome.

python3 checks/licenses.py --check
  Fails when THIRD-PARTY-NOTICES differs from the notices rebuilt from
  Cargo.lock. It runs cargo --offline, so run cargo fetch --locked first.
  Without --check it rewrites the file.

.venv/bin/python -m pytest checks/ -q
  Packaging tests, and test_wrapper.py, which runs the built runner
  (REMANIE_RUNNER, default target/debug/remanie-train-harness) with a fake
  Grok Build to check the deadline, SIGTERM, the SIGKILL after five seconds,
  an unreachable endpoint, the exit mapping, the version check, injection
  and the trace. Set REMANIE_BUNDLE to a bundle directory to also check that
  the bundle carries contract.json, a working rg and a grok that names the
  pinned commit. That part, and the check that no process outlives a
  stopped attempt, need Linux.

python3 checks/make_trace_fixture.py target/release/remanie-train-harness --grok GROK   (on Linux)
  Rewrites checks/fixtures/trace.jsonl and checks/fixtures/requests.jsonl
  from one run against a scripted endpoint: a 503 retry, three tool calls
  (two at once), a two-pass compaction, an injected provider key, and the
  title request the proxy keeps. Both files have the date set to
  2026-01-01. The trace reader tests in remanie-tooling read both files.
  Rerun it after any change to the trace or the request body, commit the
  result, and copy both files to remanie-tooling's
  research/agents/remanie_train/fixtures/. On a Mac, run it in a Linux
  container with the build in a volume that outlives the run:
    docker run --rm -v "$PWD":/w -v remanie-grok:/w/target -w /w rust:1.94-bookworm bash -c \
      'apt-get update -qq && apt-get install -y -qq python3 >/dev/null &&
       RUSTFLAGS="-C force-unwind-tables=yes" python3 build_grok.py && cargo build --release --locked &&
       python3 checks/make_trace_fixture.py target/release/remanie-train-harness --grok target/grok-build/out/grok'

python3 checks/oracle/oracle.py --grok GROK --runner target/debug/remanie-train-harness [--out DIR]
  Runs each scenario in checks/oracle/scenarios/ through GROK directly and
  through the runner wrapping GROK, each against a fresh scripted endpoint
  that reuses make_trace_fixture.py's, and exits 1 unless every scenario is
  identical: every chat-completions body the endpoint receives through the
  runner equals, byte for byte, the one Grok Build sends directly, apart
  from the keys the scenario lists under "injected". It also checks the
  runner's config.toml against the one the oracle writes itself, the
  trace's bodies against what the endpoint received, the endpoint key on
  every forwarded request, and that the title request stayed in the runner.
  It refuses a GROK whose --version does not name upstream.json's commit.
  checks/oracle/baseline/ holds its last run. Rerun with --out
  checks/oracle/baseline and read git diff to compare. CI does not run it.

CI

.github/workflows/checks.yml runs on every pull request and every push to
main. It runs cargo fmt --check, cargo clippy, cargo test, and licenses.py
--check, builds the runner, restores Grok Build's build from the cache keyed
on upstream.json and build_grok.py or runs build_grok.py, and packages the
bundle. Next it rerecords the trace fixture and fails when
checks/fixtures/requests.jsonl differs, runs pytest with REMANIE_BUNDLE and
REMANIE_RUNNER set, and compiles every tracked Python file.
