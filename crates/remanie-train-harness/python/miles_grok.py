"""Miles agent and reward hooks for the Rust coding rollout process."""

from __future__ import annotations

import asyncio
import importlib
import json
import math
import os
from pathlib import Path
import signal


_LIMITS = {
    "max_turns", "max_response_tokens", "deadline_secs",
    "compaction_threshold_tokens", "compaction_max_tokens",
}
_SAMPLING = {
    "temperature", "top_p", "top_k", "min_p", "presence_penalty",
    "frequency_penalty", "repetition_penalty", "seed", "stop",
    "chat_template_kwargs", "stop_token_ids", "skip_special_tokens",
    "no_stop_trim",
}
_STATUSES = {"completed", "max_turns", "max_tokens", "invalid_response", "deadline", "error", "cancelled"}


async def generate(input):
    if input.sample.reward is not None:
        raise ValueError("RL harness requires an unscored input Sample")
    if "rl_harness" in input.sample.metadata:
        raise ValueError("rl_harness is reserved for trusted adapter output metadata")
    from miles.rollout.generate_hub.agentic_tool_call import generate as miles_generate

    return await miles_generate(input)


def _add_arguments(parser):
    from miles.rollout.generate_hub.agentic_tool_call import generate as miles_generate

    return miles_generate.add_arguments(parser)


generate.add_arguments = _add_arguments


def _score(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("Verifier reward must be a finite number")
    return float(value)


async def _stop(proc):
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        await asyncio.wait_for(proc.wait(), timeout=3)
    except asyncio.TimeoutError:
        pass
    finally:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        await proc.wait()


async def run(base_url, prompt, request_kwargs=None, metadata=None, **kwargs):
    """Run one already-provisioned attempt and grade it with trusted Python code."""
    phase = "runner_error"
    result = None
    try:
        from miles.rollout.agentic.session import openai_session_url

        settings = json.loads(Path(os.environ["RL_HARNESS_CONFIG"]).read_text())
        if not isinstance(settings, dict) or settings.keys() - (_LIMITS | {"binary", "verifier", "verifier_timeout_secs", "model"}):
            raise ValueError("Unexpected runner configuration fields")
        binary = Path(settings.pop("binary"))
        verifier_path = settings.pop("verifier")
        verifier_timeout = settings.pop("verifier_timeout_secs", 60)
        if isinstance(verifier_timeout, bool) or not isinstance(verifier_timeout, (int, float)) or not math.isfinite(verifier_timeout) or verifier_timeout <= 0:
            raise ValueError("verifier_timeout_secs must be a positive finite number")
        if not binary.is_absolute() or not binary.is_file():
            raise ValueError("binary must name an existing absolute executable path")
        module, _, name = verifier_path.rpartition(".")
        verifier = getattr(importlib.import_module(module), name)
        if not callable(verifier):
            raise ValueError("verifier must name a trusted async callable")
        if (
            not isinstance(prompt, list) or len(prompt) != 1
            or not isinstance(prompt[0], dict) or set(prompt[0]) != {"role", "content"}
            or prompt[0]["role"] != "user" or not isinstance(prompt[0]["content"], str)
        ):
            raise ValueError("prompt must contain exactly one plain text user message")
        if not isinstance(metadata, dict):
            raise ValueError("metadata must supply workspace and session_dir")
        workspace = Path(metadata["workspace"])
        session_dir = Path(metadata["session_dir"])
        if not workspace.is_absolute() or not workspace.is_dir():
            raise ValueError("workspace must name an existing absolute attempt directory")
        if not session_dir.is_absolute() or session_dir.exists() or not session_dir.parent.is_dir():
            raise ValueError("session_dir must be a new absolute path with an existing parent")
        sampling = {key: value for key, value in (request_kwargs or {}).items() if value is not None}
        if sampling.keys() - (_SAMPLING | {"max_tokens"}):
            raise ValueError("Unsupported sampling fields")
        if "max_tokens" in sampling:
            settings["max_response_tokens"] = sampling.pop("max_tokens")
        deadline = settings.get("deadline_secs", 600)
        if isinstance(deadline, bool) or not isinstance(deadline, (int, float)) or not math.isfinite(deadline) or deadline <= 0:
            raise ValueError("deadline_secs must be a positive finite number")
        config = {
            **settings,
            "base_url": openai_session_url(base_url.rstrip("/")),
            "user_query": prompt[0]["content"],
            "workspace": str(workspace),
            "session_dir": str(session_dir),
            "sampling_params": sampling,
        }
        env = {key: os.environ[key] for key in ("PATH", "TMPDIR", "SSL_CERT_FILE", "SSL_CERT_DIR") if key in os.environ}
        payload = json.dumps(config, allow_nan=False).encode()
        proc = await asyncio.create_subprocess_exec(
            str(binary), stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            env=env, start_new_session=True,
        )
        io = asyncio.create_task(proc.communicate(payload))
        try:
            stdout, _ = await asyncio.wait_for(asyncio.shield(io), timeout=deadline + 10)
        except BaseException:
            await _stop(proc)
            await io
            raise
        result = json.loads(stdout)
        if (
            not isinstance(result, dict) or result.get("status") not in _STATUSES
            or not isinstance(result.get("final_answer"), str)
            or any(type(result.get(key)) is not int or result[key] < 0 for key in ("turns", "compactions"))
        ):
            raise ValueError("Runner returned an invalid result")
        if proc.returncode or result["status"] in {"error", "cancelled"}:
            return {"rl_harness": {"status": "runner_error", "result": result}}
        phase = "grader_error"
        score = _score(await asyncio.wait_for(
            verifier(workspace=workspace, metadata=metadata, result=result), timeout=verifier_timeout,
        ))
        return {"rl_harness": {"status": "graded", "result": result, "reward": score}}
    except Exception as exc:
        return {"rl_harness": {"status": phase, "result": result, "error": f"{type(exc).__name__}: {exc}"}}


async def reward(args, samples, **kwargs):
    """Fail the rollout batch on runner/grader errors instead of inventing a reward."""
    def extract(sample):
        outcome = sample.metadata.get("rl_harness", {})
        if outcome.get("status") != "graded":
            raise RuntimeError(f"RL harness attempt was not graded: {outcome.get('status', 'missing')}")
        return _score(outcome["reward"])

    if isinstance(samples, list):
        return [extract(sample) for sample in samples]
    return extract(samples)
