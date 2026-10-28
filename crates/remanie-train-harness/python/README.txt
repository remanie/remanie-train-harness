Miles adapter

This module uses Miles' custom-agent hook and existing session server. It does
not collect tokens, build loss masks, or create training samples. Miles owns
those operations. The source contract was checked at Miles commit
203a8d279d80dc7d8d4691f442dc7fd6d6f64e2e and the corresponding files at
74f47d37c8d741fef467be558cb0ea2caced60bc.

Add this directory to PYTHONPATH in the Miles rollout environment. Configure:

  --custom-generate-function-path miles_grok.generate
  --custom-agent-function-path miles_grok.run
  --custom-rm-path miles_grok.reward
  --use-session-server v2

The reward hook is required. A runner or verifier error raises from that hook
and fails the rollout batch. It is not a failed task and receives no score.
A verifier may return 0 for a task that really failed. No reward is read from
model output, a workspace grading file, or a sample's arbitrary reward field.
The generate entry point rejects input Samples whose reward is not None before
delegating to Miles. Miles otherwise skips reward hooks for pre-scored samples.
It also rejects the reserved rl_harness key in input metadata. Only trusted
adapter output may supply that key; other task metadata remains unchanged.

Do not set --apply-chat-template. Each prompt must contain exactly one text
user message with the raw task. Rust owns the fixed system message and wraps
the task in <user_query> once. Rust receives the session URL with /v1 appended;
it must not call an untraced model endpoint. Select and verify the Miles TITO
model implementation and SGLang tool parser for the policy model you train.

Set RL_HARNESS_CONFIG to a trusted JSON file, for example:

  {
    "binary": "/opt/remanie/remanie-train-harness",
    "model": "policy",
    "verifier": "company_verifier.grade",
    "verifier_timeout_secs": 60,
    "max_turns": 100,
    "max_response_tokens": 4096,
    "deadline_secs": 600,
    "compaction_threshold_tokens": 0,
    "compaction_max_tokens": 2048
  }

binary must be an absolute executable path. verifier is a trusted async Python
callable selected by this configuration, not by the task. Its contract is:

  async def grade(*, workspace: pathlib.Path, metadata: dict, result: dict) -> float:
      ...

Return a finite numeric reward only after verification completes. Raise on
verifier infrastructure failures. The callback must perform asynchronous I/O;
the timeout cannot interrupt blocking Python or an external job by itself.
Use the existing protected verifier service or sandbox teardown mechanism for
those jobs. The default verifier timeout is 60 seconds.

For each invocation, the trusted environment owner must provide metadata with
an existing absolute workspace and a new absolute session_dir whose parent
exists. Both paths belong to one attempt. Provision fresh paths for repeated
samples and concurrent rollouts. A static dataset entry reused across attempts
is insufficient. Rust exclusively creates session_dir and rejects collisions.
The adapter does not copy repositories, create containers, reset workspaces,
or dispose of attempts. It preserves other metadata for the trusted verifier
and does not send it to Rust or the model.

The adapter forwards supported Chat Completions sampling settings. Miles'
request_kwargs.max_tokens overrides max_response_tokens. Unknown settings and
TITO control fields are rejected. The Rust process gets PATH, TMPDIR and TLS
certificate paths when set, without the caller's API keys or verifier config.
Rust writes one JSON result on stdout and diagnostics on inherited stderr.

completed, max_turns, max_tokens, invalid_response and deadline are task
outcomes that the verifier can grade. error and cancelled skip verification.
The returned metadata is {"rl_harness": {"status": "graded", "result": ...,
"reward": ...}}, or runner_error/grader_error with no reward. The reward hook
supports both one Miles Sample and the list used by multi-root v2 sessions.

This adapter is not a sandbox. Its subprocess runs on the current host. Before
running model-written commands, the environment owner must isolate files,
network access, processes and resources, and keep grading authority outside
the agent's access. Use an outer sandbox that removes every descendant when
an attempt ends. The adapter sends SIGTERM on cancellation, then SIGKILL after
three seconds. Native shell tasks use separate process groups; they depend on
Rust's normal cleanup or the outer sandbox if Rust crashes or gets SIGKILL.

Local fixture checks

  PYTHONPATH=/path/to/miles:/path/to/this/python \
    python3 -m unittest discover -s /path/to/this/python -p 'test_*.py' -v

These tests import the real Miles session URL helper and exercise a fake Rust
executable plus a local HTTP model fixture. They check the adapter contract,
reward failure handling, secret filtering and process cancellation. They do
not establish real model TITO parity, GPU training, Harbor integration, or
the isolation of a production attempt environment.

Real Miles CPU trace checks

  PYTHONPATH=/path/to/miles:/path/to/this/python \
    python3 /path/to/this/python/check_miles_trace.py -v

Run this separate check inside an environment with Miles' Python dependencies.
It uses the real v2 tree, record assembler, default picker and postprocessor,
wire codec, and rollout reward normalization. Six tests check exact recorded token IDs
and logprobs, strict assistant replay, shared-token loss masks, and compaction
into two roots. The summary generation trains once, its copied context has
zero loss, and the two samples share one rollout reward during normalization.
The sampling allowlist is checked against SGLang's actual request schema.
The entry-point guard also checks the real Miles input/output classes and CLI
registration contract; only its delegate call is intercepted for that test.
The codec regression proves that forged result metadata would survive an agent
failure without the entry-point guard, and that the guard rejects it.

These checks passed on CPU with Miles 74f47d3, SGLang 0.5.22, Transformers
5.19.0, PyTorch 2.14.1, NumPy 2.5.4, Pydantic 2.14.1, tokenizers 0.23.3 and
Ray 2.59.0. The records contain synthetic token IDs. A local WordLevel decoder
produces display strings only. These tests do not run model generation or
check a policy model's chat-template/tokenizer parity.

Built Rust integration checks

  RL_HARNESS_TEST_BINARY=/absolute/path/to/remanie-train-harness \
    PYTHONPATH=/path/to/miles:/path/to/this/python \
    python3 /path/to/this/python/check_rust_adapter.py -v

This separate check runs the built binary against a local HTTP model fixture.
It verifies a real native tool call through the adapter and trusted verifier,
then checks that cancellation stops a native background command. Its commands
write into temporary attempt directories and do not contact a model service.
