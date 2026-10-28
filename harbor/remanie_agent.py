import asyncio
import json
import math
import os
import re
import shlex
import stat
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import Any

from pydantic import Field

from harbor.agents.agent_auth_resolution import ModelConnectionSpec, ProviderAccess, require_model_api_key
from harbor.agents.capabilities import AgentCapabilities
from harbor.agents.installed.base import BaseInstalledAgent
from harbor.agents.options import InstalledAgentOptions
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from harbor.models.model_api_format import ModelApiFormat

MAX_BODY = 16 * 1024 * 1024


@contextmanager
def bounded_log(root: Path, relative: str, limit: int):
    directory = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        parts = Path(relative).parts
        for part in parts[:-1]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
            os.close(directory)
            directory = child
        descriptor = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
        with os.fdopen(descriptor, "rb") as file:
            info = os.fstat(file.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
                raise ValueError("Log must be a bounded regular file")
            yield file
    finally:
        os.close(directory)


class RemanieOptions(InstalledAgentOptions):
    runtime_dir: Path
    max_turns: int = Field(default=100, ge=1, le=512)
    max_response_tokens: int = Field(default=4096, gt=0)
    deadline_secs: int = Field(default=86400, ge=1, le=86400)
    compaction_threshold_tokens: int = Field(default=0, ge=0)
    compaction_max_tokens: int = Field(default=2048, gt=0)
    sampling_params: dict[str, Any] = Field(default_factory=dict)


class RemanieAgent(BaseInstalledAgent):
    options_model = RemanieOptions
    capabilities = AgentCapabilities(base_url=True)
    MODEL_CONNECTION = ModelConnectionSpec(
        default_provider="openrouter",
        agent_provider_access=ProviderAccess(
            api_key_candidate_names=["OPENROUTER_API_KEY"],
        ),
        supported_api_formats=[ModelApiFormat.OPENAI_CHAT_COMPLETIONS],
        api_key_destination_names={"default": "OPENROUTER_API_KEY"},
        base_url_destination_names={"default": None},
        inject_default_base_url=True,
        default_api_format_by_provider={
            "default": ModelApiFormat.OPENAI_CHAT_COMPLETIONS,
        },
    )
    runtime_path = PurePosixPath("/opt/remanie-train-harness")

    def __init__(self, *args, model_name="openrouter/x-ai/grok-4.7", **kwargs):
        super().__init__(*args, model_name=model_name, **kwargs)
        if self._prompt_template_path:
            raise ValueError("Remanie passes the task instruction unchanged")
        self.runtime_dir = self.options.runtime_dir.expanduser().resolve()
        for name in ("remanie-train-harness", "rg"):
            executable = self.runtime_dir / name
            if not executable.is_file() or not os.access(executable, os.X_OK):
                raise ValueError(f"runtime_dir must contain an executable {name}")
        self.run_dir = self.environment_logs_dir / "remanie"

    @staticmethod
    def name() -> str:
        return "remanie-train-harness"

    async def install(self, environment: BaseEnvironment) -> None:
        await self.exec_as_root(
            environment,
            "mkdir -p /opt /usr/local/bin && command -v bash >/dev/null && command -v setsid >/dev/null",
        )
        await environment.upload_dir(self.runtime_dir, str(self.runtime_path))
        await self.exec_as_root(
            environment,
            f"chmod -R a+rX {self.runtime_path} && "
            f"ln -sf {self.runtime_path}/remanie-train-harness /usr/local/bin/remanie-train-harness && "
            f"ln -sf {self.runtime_path}/rg /usr/local/bin/rg && "
            "/usr/local/bin/rg --version >/dev/null",
        )

    async def run(
        self, instruction: str, environment: BaseEnvironment, context: AgentContext
    ) -> None:
        connection = self.model_connection
        require_model_api_key(connection, self.name())
        if not connection.execution_base_url:
            raise ValueError("Set Harbor's base_url to an OpenAI-compatible /v1 endpoint")
        cwd = await self.exec_as_agent(environment, "pwd -P")
        workspace = (cwd.stdout or "").rstrip("\n")
        if not PurePosixPath(workspace).is_absolute():
            raise RuntimeError("Task environment did not provide an absolute working directory")
        await self.exec_as_agent(
            environment, f"mkdir -p {shlex.quote(str(self.environment_logs_dir))}"
        )
        config = {
            "model": self._parsed_model_name,
            "base_url": connection.execution_base_url,
            "api_key_env": connection.api_key_destination_name,
            "user_query": instruction,
            "workspace": workspace,
            "session_dir": str(self.run_dir),
            **self.options.model_dump(include={
                "max_turns", "max_response_tokens", "deadline_secs",
                "compaction_threshold_tokens", "compaction_max_tokens", "sampling_params",
            }),
        }
        config_path = self.environment_logs_dir / "input.json"
        result_path = self.environment_logs_dir / "result.json"
        stderr_path = self.environment_logs_dir / "runtime-stderr.txt"
        script = (
            'read -r process_stat < /proc/$$/stat; set -- ${process_stat##*) }; '
            'printf "REMANIE_PROCESS=%s:%s\\n" "$$" "${20}"; '
            f"exec {self.runtime_path}/remanie-train-harness "
            f"< {shlex.quote(str(config_path))} "
            f"> {shlex.quote(str(result_path))} "
            f"2> {shlex.quote(str(stderr_path))}"
        )
        await self._upload_config_text(
            environment,
            content=json.dumps(config, allow_nan=False),
            remote_path=str(config_path),
            filename="input.json",
        )
        process = None

        async def capture_process(text, stream):
            nonlocal process
            match = re.fullmatch(r"REMANIE_PROCESS=([1-9][0-9]*):([0-9]+)\n?", text)
            if process is None and stream == "stdout" and match:
                process = match.groups()

        try:
            with environment.scoped_output_callback(capture_process):
                execution = asyncio.create_task(self.exec_as_agent(
                    environment,
                    f"exec setsid --wait bash -c {shlex.quote(script)}",
                    env=dict(connection.env),
                ))
                result = await asyncio.shield(execution)
                if process is None:
                    await capture_process(result.stdout or "", "stdout")
        finally:
            cleanup = asyncio.create_task(self._stop_runtime(environment, process, execution))
            try:
                await asyncio.shield(cleanup)
            except asyncio.CancelledError:
                await cleanup
                raise
            finally:
                if not execution.done():
                    execution.cancel()
                await asyncio.gather(execution, return_exceptions=True)

    async def _stop_runtime(self, environment, process, execution):
        if process is None:
            raise RuntimeError("Runtime identity was not captured; container teardown required")
        pid, started = process
        script = f"""
pid={pid}
alive() {{
    [ -e "/proc/$pid/stat" ] || return 1
    IFS= read -r status < "/proc/$pid/stat" || return 1
    set -- ${{status##*) }}
    [ "${{20}}" = {started} ] || exit 1
    [ "$1" != Z ] && [ "$1" != X ] || return 1
    [ "$3" = "$pid" ] && [ "$4" = "$pid" ] || exit 1
}}
if ! alive; then exit 0; fi
kill -TERM -- "-$pid" 2>/dev/null || true
for i in {{1..50}}; do alive || exit 0; sleep 0.1; done
kill -KILL -- "-$pid" 2>/dev/null || true
exit 1
"""
        try:
            await self.exec_as_agent(environment, script, timeout_sec=10)
            await asyncio.wait_for(asyncio.shield(execution), timeout=10)
        except Exception as exc:
            raise RuntimeError("Runtime shutdown was not confirmed; container teardown required") from exc

    def populate_context_post_run(self, context: AgentContext) -> None:
        metadata: dict[str, Any] = {}
        try:
            with bounded_log(self.logs_dir, "result.json", MAX_BODY) as file:
                raw = file.read(MAX_BODY + 1)
                if len(raw) > MAX_BODY:
                    raise ValueError("Result exceeds 16 MiB")
                result = json.loads(raw)
                if not isinstance(result, dict):
                    raise ValueError("result must be an object")
                metadata.update({
                    key: result[key]
                    for key in ("status", "turns", "compactions", "error")
                    if key in result
                })
        except FileNotFoundError:
            pass
        except (OSError, ValueError, TypeError, RecursionError):
            metadata["result_invalid"] = True

        totals = {"prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0}
        requests = 0
        responses = 0
        complete = True
        cost_count = 0
        total_cost = 0.0
        try:
            with bounded_log(self.logs_dir, "remanie/trace.jsonl", 256 * 1024 * 1024) as trace:
                total_bytes = 0
                while line := trace.readline(MAX_BODY + 1025):
                    total_bytes += len(line)
                    if len(line) > MAX_BODY + 1024 or total_bytes > 256 * 1024 * 1024:
                        raise ValueError("Trace exceeds read limits")
                    try:
                        event = json.loads(line)
                    except (ValueError, RecursionError):
                        metadata["trace_incomplete"] = True
                        complete = False
                        continue
                    if not isinstance(event, dict):
                        continue
                    if event.get("event") in ("model_request", "compaction_request"):
                        requests += 1
                        continue
                    if event.get("event") not in ("model_response", "compaction_response"):
                        continue
                    responses += 1
                    body = event.get("body")
                    usage = body.get("usage") if isinstance(body, dict) else None
                    if not isinstance(usage, dict):
                        complete = False
                        continue
                    details = usage.get("prompt_tokens_details")
                    counts = {
                        "prompt_tokens": usage.get("prompt_tokens"),
                        "completion_tokens": usage.get("completion_tokens"),
                        "cached_tokens": details.get("cached_tokens", 0) if isinstance(details, dict) else 0,
                    }
                    if all(type(n) is int and n >= 0 for n in counts.values()):
                        for key, value in counts.items():
                            totals[key] += value
                    else:
                        complete = False
                    cost = usage.get("cost")
                    if type(cost) in (int, float) and 0 <= cost <= 1e9 and math.isfinite(cost):
                        cost_count += 1
                        total_cost += cost
        except FileNotFoundError:
            pass
        except (OSError, ValueError):
            metadata["trace_incomplete"] = True
            complete = False
        complete = complete and requests == responses
        if responses and complete:
            context.n_input_tokens = totals["prompt_tokens"]
            context.n_output_tokens = totals["completion_tokens"]
            context.n_cache_tokens = totals["cached_tokens"]
        if responses and requests == responses and cost_count == responses and not metadata.get("trace_incomplete"):
            context.cost_usd = total_cost
        metadata.update(request_count=requests, response_count=responses, usage_complete=bool(responses and complete))
        context.metadata = {**(context.metadata or {}), "remanie": metadata}
