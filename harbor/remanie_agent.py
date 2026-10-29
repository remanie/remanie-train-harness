import asyncio
import hashlib
import json
import os
import re
import shlex
import stat
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import jsonschema
from pydantic import ConfigDict

from harbor.agents.agent_auth_resolution import ModelConnectionSpec, ProviderAccess, require_model_api_key
from harbor.agents.capabilities import AgentCapabilities
from harbor.agents.installed.base import BaseInstalledAgent
from harbor.agents.options import InstalledAgentOptions
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from harbor.models.model_api_format import ModelApiFormat

RUNTIME = PurePosixPath("/opt/remanie-train-harness")
MAX_RESULT_BYTES = 16 * 1024 * 1024
IDENTITY = re.compile(r"^REMANIE_PROCESS=([1-9][0-9]*):([0-9]+)$", re.M)
TRIAL_FIELDS = ("user_query", "workspace", "session_dir")
CONNECTION_FIELDS = ("model", "base_url", "api_key_env")


class RemanieInfrastructureError(RuntimeError):
    """Harbor skips the verifier for this error, unlike for a NonZeroAgentExitCodeError."""


@dataclass(frozen=True)
class Bundle:
    root: Path
    architecture: str
    sha256: str
    input_schema: dict[str, Any]
    exit_gradable: int
    exit_infrastructure: int

    @classmethod
    def load(cls, root: Path) -> "Bundle":
        root = root.expanduser().resolve()
        for name in ("remanie-train-harness", "rg"):
            executable = root / name
            if not executable.is_file() or not os.access(executable, os.X_OK):
                raise ValueError(f"runtime_dir must contain an executable {name}")
        try:
            manifest_bytes = (root / "manifest.json").read_bytes()
            architecture = json.loads(manifest_bytes)["architecture"]
            contract = json.loads((root / "contract.json").read_text())
            codes = contract["exit_codes"]
            return cls(
                root=root,
                architecture=architecture,
                sha256=hashlib.sha256(manifest_bytes).hexdigest(),
                input_schema=contract["input"],
                exit_gradable=codes["gradable"],
                exit_infrastructure=codes["infrastructure"],
            )
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise ValueError(f"runtime_dir is not a packaged bundle: {error!r}") from None

    def validate_job(self, job: dict[str, Any]) -> None:
        required = [name for name in self.input_schema.get("required", []) if name not in TRIAL_FIELDS]
        try:
            jsonschema.validate(job, {**self.input_schema, "required": required})
        except jsonschema.ValidationError as error:
            raise ValueError(f"Invalid runner setting at {error.json_path}: {error.message}") from None


class RemanieOptions(InstalledAgentOptions):
    model_config = ConfigDict(extra="allow")
    runtime_dir: Path


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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self._prompt_template_path:
            raise ValueError("Remove the prompt template. Remanie passes the task instruction unchanged")
        if not self._parsed_model_name:
            raise ValueError("Set model_name. Remanie has no default model")
        self.bundle = Bundle.load(self.options.runtime_dir)
        settings = dict(self.options.model_extra or {})
        reserved = sorted(settings.keys() & {*TRIAL_FIELDS, *CONNECTION_FIELDS})
        if reserved:
            raise ValueError(f"Harbor supplies {', '.join(reserved)}. Remove them from kwargs")
        base_url = self.model_connection.execution_base_url
        if not base_url:
            raise ValueError("Set Harbor's base_url to an OpenAI-compatible /v1 endpoint")
        self.job = {**settings, "model": self._parsed_model_name, "base_url": base_url}
        self.bundle.validate_job(self.job)
        self.exit_code: int | None = None

    @staticmethod
    def name() -> str:
        return "remanie_train"

    async def install(self, environment: BaseEnvironment) -> None:
        probe = await self.exec_as_root(
            environment,
            "mkdir -p /opt && command -v bash >/dev/null && command -v setsid >/dev/null && uname -m",
        )
        machine = (probe.stdout or "").strip()
        if machine != self.bundle.architecture:
            raise RuntimeError(f"Bundle is built for {self.bundle.architecture}, but the container runs {machine}")
        await environment.upload_dir(self.bundle.root, str(RUNTIME))
        await self.exec_as_root(environment, f"chmod -R a+rX {RUNTIME} && {RUNTIME}/rg --version >/dev/null")

    def _credential(self) -> tuple[str, str] | None:
        connection = self.model_connection
        if not (self._model_api_key_name or connection.base_url_is_default):
            return None
        require_model_api_key(connection, self.name())
        return connection.api_key_destination_name, connection.api_key

    async def _upload_private(self, environment: BaseEnvironment, content: str, remote_path: str) -> None:
        with tempfile.TemporaryDirectory(prefix="remanie-") as directory:
            local = Path(directory) / "credential"
            local.touch(mode=0o600)
            local.write_text(content)
            await environment.upload_file(local, remote_path)
        target = shlex.quote(remote_path)
        owner = "" if environment.default_user is None else f"chown {shlex.quote(str(environment.default_user))} {target} && "
        await self.exec_as_root(environment, f"{owner}chmod 600 {target}")

    async def run(
        self, instruction: str, environment: BaseEnvironment, context: AgentContext
    ) -> None:
        credential = self._credential()
        cwd = await self.exec_as_agent(environment, "pwd -P")
        workspace = (cwd.stdout or "").rstrip("\n")
        if not PurePosixPath(workspace).is_absolute():
            raise RuntimeError("Task environment did not provide an absolute working directory")
        logs = self.environment_logs_dir
        await self.exec_as_agent(environment, f"mkdir -p {shlex.quote(str(logs))}")
        config = {
            **self.job,
            "api_key_env": credential[0] if credential else None,
            "user_query": instruction,
            "workspace": workspace,
            "session_dir": str(logs / "remanie"),
        }
        input_path = logs / "input.json"
        await self._upload_config_text(
            environment,
            content=json.dumps(config, allow_nan=False),
            remote_path=str(input_path),
            filename="input.json",
        )
        exports = f"RG_BIN_PATH={RUNTIME}/rg"
        read_key = ""
        if credential:
            # Keep the key outside the logs directory, which Docker bind-mounts from the host.
            key_path = f"/tmp/remanie-{uuid.uuid4().hex}.key"
            await self._upload_private(environment, credential[1] + "\n", key_path)
            read_key = f"IFS= read -r key < {key_path}; rm -f -- {key_path}; "
            exports = f'{credential[0]}="$key" {exports}'
        script = (
            'read -r process_stat < /proc/$$/stat; set -- ${process_stat##*) }; '
            'printf "REMANIE_PROCESS=%s:%s\\n" "$$" "${20}"; '
            f"{read_key}{exports} exec {RUNTIME}/remanie-train-harness "
            f"< {shlex.quote(str(input_path))} "
            f"> {shlex.quote(str(logs / 'result.json'))} "
            f"2> {shlex.quote(str(logs / 'runtime-stderr.txt'))}"
        )
        identity = None

        async def capture_identity(text, stream):
            nonlocal identity
            if identity is None and stream == "stdout" and (match := IDENTITY.search(text)):
                identity = match.groups()

        # asyncio copies the context, callback included, when it creates the task.
        with environment.scoped_output_callback(capture_identity):
            execution = asyncio.create_task(environment.exec(
                command=f"exec setsid --wait bash -c {shlex.quote(script)}"
            ))
        try:
            result = await asyncio.shield(execution)
        except BaseException:
            await self._stop_after_interruption(environment, identity, execution)
            raise
        self.exit_code = result.return_code
        if self.exit_code == self.bundle.exit_infrastructure:
            raise RemanieInfrastructureError(
                f"Runner reported an infrastructure failure (exit {self.exit_code}). See metadata remanie.result"
            )
        if self.exit_code != self.bundle.exit_gradable:
            raise RuntimeError(f"Runner died with exit {self.exit_code}. Tear down the container")

    async def _stop_after_interruption(self, environment, identity, execution):
        cleanup = asyncio.create_task(self._stop_runtime(environment, identity, execution))
        try:
            await asyncio.shield(cleanup)
        except asyncio.CancelledError:
            await cleanup
            raise
        finally:
            if not execution.done():
                execution.cancel()
            await asyncio.gather(execution, return_exceptions=True)

    async def _stop_runtime(self, environment, identity, execution):
        if identity is None:
            raise RuntimeError(
                "The launch script's REMANIE_PROCESS line did not arrive, so the adapter cannot stop the runner. "
                "Tear down the container"
            )
        pid, started = identity
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
            await self.exec_as_agent(environment, f"bash -c {shlex.quote(script)}", timeout_sec=10)
            result = await asyncio.wait_for(asyncio.shield(execution), timeout=10)
        except Exception as exc:
            raise RuntimeError("The stop script could not confirm that the runner exited. Tear down the container") from exc
        self.exit_code = result.return_code
        if self.exit_code != self.bundle.exit_gradable:
            raise RuntimeError(f"Runner exited {self.exit_code} after the stop signal. Tear down the container")

    def populate_context_post_run(self, context: AgentContext) -> None:
        metadata: dict[str, Any] = {"bundle_sha256": self.bundle.sha256, "exit_code": self.exit_code, "result": None}
        try:
            descriptor = os.open(self.logs_dir / "result.json", os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
            with os.fdopen(descriptor, "rb") as file:
                info = os.fstat(file.fileno())
                if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_RESULT_BYTES:
                    raise ValueError("result.json must be a regular file of at most 16 MiB")
                if info.st_size == 0:
                    raise FileNotFoundError("result.json is empty")
                result = json.loads(file.read(MAX_RESULT_BYTES + 1))
            if not isinstance(result, dict):
                raise ValueError("result.json must hold an object")
            metadata["result"] = result
        except FileNotFoundError:
            pass
        except (OSError, ValueError, RecursionError):
            metadata["result_invalid"] = True
        usage = (metadata["result"] or {}).get("usage")
        if isinstance(usage, dict) and usage.get("complete") is True:
            counts = [usage.get(name) for name in ("prompt_tokens", "completion_tokens", "cached_tokens")]
            if all(type(count) is int and count >= 0 for count in counts):
                context.n_input_tokens, context.n_output_tokens, context.n_cache_tokens = counts
        context.metadata = {**(context.metadata or {}), "remanie": metadata}
