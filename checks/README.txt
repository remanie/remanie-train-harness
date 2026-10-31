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

python3 checks/vendor.py [--refresh GROK_BUILD_CLONE]
  Fails when a file under vendor/grok-shell/src/session/ differs from the
  Grok Build file it copies at NOTICE's commit, by git blob sha
  (vendor/UPSTREAM.txt). With --refresh it copies the files again from a
  clone and rewrites the manifest.

python3 checks/lockdiff.py Cargo.lock GROK_BUILD_CLONE/Cargo.lock   (Python 3.11 or later)
  Fails when Cargo.lock holds a crate at a version Grok Build's own lock
  does not. A git dependency inherits upstream's manifests but not its
  lock, so seed Cargo.lock from upstream's instead of running cargo update.

python3 checks/make_trace_fixture.py target/debug/remanie-train-harness   (on Linux; the script prints the Docker one-liner for a Mac)
  Rewrites checks/fixtures/trace.jsonl and checks/fixtures/requests.jsonl
  from one run against a scripted endpoint: a 503 retry, three tool calls,
  and a compaction whose request drops the first turn to fit the window. The
  trace reader tests in remanie-tooling read both files.
  Rerun it after any change to the trace or the request body, commit the
  result, and copy both files to remanie-tooling's
  research/agents/remanie_train/fixtures/.

python3 checks/oracle/oracle.py --grok GROK --runner target/debug/remanie-train-harness [--out DIR]
  Runs each scenario in checks/oracle/scenarios/ through Grok Build's own
  binary and through the runner, each against a fresh scripted endpoint
  that reuses make_trace_fixture.py's, and diffs the chat-completions
  bodies they send. It prints, per scenario, every request side by side,
  the first one that differs, and a unified diff of the two bodies, and
  exits 1 while any scenario differs. It normalizes only the sanctioned
  differences: the tool definitions, the model name, the date, and the
  <rules> and <system-reminder> blocks Grok Build injects. Grok Build gets
  the runner's eight tools through an agent file, and its AGENTS.md,
  skills, memory and workflows are turned off. GROK is xai-grok-pager
  built at NOTICE's commit, which the oracle checks:
    git clone https://github.com/xai-org/grok-build && cd grok-build
    git checkout 2bdd1d6a6369de0e8c68132ea4539e9abd9e14a8
    cargo build -p xai-grok-pager-bin    (needs dotslash on PATH for protoc)
  checks/oracle/baseline/ holds a run with the runner at main, before the
  fidelity rewrite. Rerun with --out checks/oracle/baseline and read
  git diff to see old against new. CI does not run it.

CI

.github/workflows/checks.yml runs on every pull request and every push to
main. It runs cargo fmt --check, cargo clippy, cargo test, and licenses.py
--check, then builds and packages the bundle. Next it rerecords the trace
fixture and fails when checks/fixtures/requests.jsonl differs, runs pytest
with REMANIE_BUNDLE set, and compiles every tracked Python file.
