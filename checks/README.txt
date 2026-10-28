Checks

Run each command from the repository root. .venv is the Python environment
with harbor/requirements.txt installed, as in harbor/README.md.

cargo test --locked
  Rust unit tests, including the runner against the fake session server in
  checks/session_fixture.py.

cargo clippy --all-targets --locked -- -D warnings
  Rust lints, with every warning an error.

cargo fmt --check
  Rust formatting against rustfmt.toml.

python3 checks/licenses.py --check
  Fails when THIRD-PARTY-NOTICES differs from the notices rebuilt from
  Cargo.lock. Uses cargo --offline, so fetch the crates first. Without --check
  it rewrites the file.

PYTHONPATH=. python3 checks/test_package_linux.py
  package_linux.py refuses to bundle a library whose Debian license text is
  missing.

.venv/bin/python harbor/test_remanie_agent.py
  Adapter unit tests against a mocked Harbor environment.

.venv/bin/python checks/check_harbor.py
  Real Harbor trials in Docker against a scripted local endpoint. No paid
  model calls. Needs Docker, host.docker.internal resolving to the host, and
  the bundle in runtime/ built for the Docker worker's CPU architecture.
  --runtime-dir picks another bundle. --output-dir picks a new directory for
  the trial output, which otherwise goes under the system temp folder. Docker
  may pull the pinned Debian image. A passing reward says the integration
  works. It is not a Terminal-Bench score.

CI

.github/workflows/checks.yml runs cargo test, cargo clippy, cargo fmt --check,
checks/licenses.py --check and both Python unit tests on every pull request.
It does not run checks/check_harbor.py.
