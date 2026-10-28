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
  Adapter, trace reader and packaging unit tests. Set REMANIE_BUNDLE to a
  bundle directory to also check that the bundle carries contract.json and a
  working rg.

.venv/bin/python checks/check_harbor.py
  Real Harbor trials in Docker against a scripted local endpoint, with no paid
  model calls. Needs Docker, host.docker.internal resolving to the host, and
  the bundle in runtime/ built for the Docker worker's CPU architecture.
  --runtime-dir picks another bundle. --output-dir picks a new directory for
  the trial output, which otherwise goes under the system temp folder.
  It runs five trials:
    completed       all eight tools and one compaction, graded with reward 1
    timeout         Harbor's timeout cancels the runner, still graded
    cleanup_error   the stop is never confirmed, verifier skipped
    crash           the runner is killed, verifier skipped
    infrastructure  the endpoint returns 401, exit 70, verifier skipped
  A passing reward says the integration works. It is not a Terminal-Bench
  score.

CI

.github/workflows/checks.yml runs on every pull request and every push to
main. It runs cargo fmt --check, cargo clippy, cargo test, licenses.py
--check, builds and packages the bundle, runs pytest with REMANIE_BUNDLE set,
and compiles every tracked Python file. It does not run check_harbor.py.
