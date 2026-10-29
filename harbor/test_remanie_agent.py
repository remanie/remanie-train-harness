import asyncio
import contextvars
import hashlib
import importlib.util
import json
import os
import shlex
import shutil
import subprocess
import tempfile
import unittest
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from harbor.agents.factory import AgentFactory
from harbor.agents.installed.base import NonZeroAgentExitCodeError
from harbor.environments.base import BaseEnvironment, ExecResult
from harbor.environments.docker.docker import MAIN_SERVICE_NAME, DockerEnvironment
from harbor.models.agent.context import AgentContext
from harbor.models.trial.config import AgentConfig

from remanie_agent import RemanieAgent, RemanieInfrastructureError

SECRET = "test-only-secret"
REPO = Path(__file__).resolve().parents[1]
JOB = {"max_turns": 7, "deadline_secs": 120, "sampling_params": {"max_tokens": 4096, "reasoning": {"effort": "high"}}}


async def compose_argv(call):
    run = AsyncMock(return_value=ExecResult(return_code=0))
    docker = SimpleNamespace(
        _platform=SimpleNamespace(exec_shell_args=lambda command: ["bash", "-c", command]),
        _run_docker_compose_command=run,
        _output_callback=lambda: None,
    )
    await DockerEnvironment._compose_exec(
        docker, call["command"], service=MAIN_SERVICE_NAME, cwd=call.get("cwd"),
        env=call.get("env"), timeout_sec=call.get("timeout_sec"), user=call.get("user"),
    )
    return run.await_args.args[0]


class RemanieAgentTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.runtime = self.root / "runtime"
        self.runtime.mkdir()
        for name in ("remanie-train-harness", "rg"):
            executable = self.runtime / name
            executable.write_text("#!/bin/sh\nexit 0\n")
            executable.chmod(0o755)
        manifest = json.dumps({"architecture": "x86_64", "files": {}}).encode()
        (self.runtime / "manifest.json").write_bytes(manifest)
        self.bundle_sha256 = hashlib.sha256(manifest).hexdigest()
        shutil.copyfile(REPO / "assets/contract.json", self.runtime / "contract.json")
        self.logs = self.root / "logs"
        self.logs.mkdir()
        self.env = patch.dict(os.environ, {"OPENROUTER_API_KEY": SECRET}, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)

    def agent(self, *, model_name="openrouter/x-ai/grok-4.7", base_url=None, model_api_key_name=None, **kwargs):
        settings = {key: value for key, value in {**JOB, **kwargs}.items() if value is not None}
        return AgentFactory.create_agent_from_config(
            AgentConfig(
                import_path="remanie_agent:RemanieAgent",
                model_name=model_name,
                base_url=base_url,
                model_api_key_name=model_api_key_name,
                kwargs={"runtime_dir": str(self.runtime), **settings},
            ),
            logs_dir=self.logs,
        )

    def environment(self, *, launch=None, machine="x86_64"):
        environment = AsyncMock(spec=BaseEnvironment)
        environment.default_user = None
        environment.calls = []
        environment.uploads = {}
        environment._output_callbacks = contextvars.ContextVar("output_callbacks", default=())

        async def execute(**kwargs):
            environment.calls.append(kwargs)
            command = kwargs["command"]
            if command.startswith("exec setsid") and launch:
                environment.output_callback = BaseEnvironment._output_callback(environment)
                return await launch(environment)
            if "uname -m" in command:
                return ExecResult(stdout=f"{machine}\n", return_code=0)
            return ExecResult(stdout="/task with spaces\n", return_code=0)

        async def upload(source, target):
            environment.uploads[target] = Path(source).read_text()

        environment.scoped_output_callback = partial(BaseEnvironment.scoped_output_callback, environment)
        environment.exec.side_effect = execute
        environment.upload_file.side_effect = upload
        return environment

    def launch_command(self, environment):
        return next(call["command"] for call in environment.calls if call["command"].startswith("exec setsid"))

    def metadata(self, agent):
        context = AgentContext()
        agent.populate_context_post_run(context)
        return context.metadata["remanie"]

    async def test_runner_input_is_the_job_plus_the_trial(self):
        agent = self.agent()
        environment = self.environment()
        await agent.setup(environment)
        environment.upload_dir.assert_awaited_once_with(self.runtime.resolve(), "/opt/remanie-train-harness")
        instruction = "Fix /task/input.txt. Preserve the user's exact instruction."
        await agent.run(instruction, environment, AgentContext())
        self.assertEqual(json.loads(environment.uploads["/logs/agent/input.json"]), {
            **JOB,
            "model": "x-ai/grok-4.7",
            "base_url": "https://openrouter.ai/api/v1",
            "api_key_env": "OPENROUTER_API_KEY",
            "user_query": instruction,
            "workspace": "/task with spaces",
            "session_dir": "/logs/agent/remanie",
        })
        launch = self.launch_command(environment)
        self.assertIn("RG_BIN_PATH=/opt/remanie-train-harness/rg", launch)
        self.assertIn("exec /opt/remanie-train-harness/remanie-train-harness", launch)
        for call in environment.calls:
            self.assertNotIn("/usr/local/bin", call["command"])

    async def test_exit_code_decides_whether_harbor_grades(self):
        cases = {
            0: None,
            70: RemanieInfrastructureError,
            1: RuntimeError,
            137: RuntimeError,
        }
        (self.logs / "result.json").write_text(json.dumps({"outcome": "completed", "usage": {}}))
        for code, raised in cases.items():
            with self.subTest(exit=code):
                agent = self.agent()

                async def launch(environment, code=code):
                    await environment.output_callback("REMANIE_PROCESS=123:456\n", "stdout")
                    return ExecResult(return_code=code)

                run = agent.run("task", self.environment(launch=launch), AgentContext())
                if raised is None:
                    await run
                else:
                    with self.assertRaises(raised) as error:
                        await run
                    self.assertNotIsInstance(error.exception, NonZeroAgentExitCodeError)
                    if code != 70:
                        self.assertNotIsInstance(error.exception, RemanieInfrastructureError)
                metadata = self.metadata(agent)
                self.assertEqual(metadata["exit_code"], code)
                self.assertEqual(metadata["bundle_sha256"], self.bundle_sha256)
                self.assertEqual(metadata["result"]["outcome"], "completed")

    def test_job_errors_fail_on_the_host(self):
        for options in (
            {"model_name": None},
            {"max_turns": 0},
            {"max_turns": "7"},
            {"sampling_params": {"max_tokens": 4096, "stream": True}},
            {"deadline_secs": 0},
            {"deadline_secs": None},
            {"sampling_params": {"max_tokens": 4294967296}},
            {"sampling_params": {}},
            {"workspace": "/elsewhere"},
            {"unknown": True},
            {"prompt_template_path": "wrapper.jinja2"},
            {"runtime_dir": str(self.root)},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.agent(**options)
        (self.runtime / "contract.json").unlink()
        with self.assertRaisesRegex(ValueError, "packaged bundle"):
            self.agent()

    def test_every_job_in_the_repository_passes_the_real_contract(self):
        spec = importlib.util.spec_from_file_location("check_harbor", REPO / "checks/check_harbor.py")
        check_harbor = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(check_harbor)
        listed = subprocess.run(["git", "ls-files", "*.json"], cwd=REPO, capture_output=True, text=True, check=True)
        jobs = [
            (path, agent["model_name"], agent["kwargs"])
            for path in listed.stdout.split()
            for agent in json.loads((REPO / path).read_text()).get("agents", [])
            if agent.get("import_path") == "remanie_agent:RemanieAgent"
        ]
        self.assertIn("harbor/terminal-bench-2.1-smoke.json", [path for path, _, _ in jobs])
        jobs.append(("checks/check_harbor.py", "openrouter/x-ai/grok-4.7", check_harbor.KWARGS))
        for path, model_name, kwargs in jobs:
            with self.subTest(job=path):
                AgentFactory.create_agent_from_config(
                    AgentConfig(
                        import_path="remanie_agent:RemanieAgent",
                        model_name=model_name,
                        kwargs={**kwargs, "runtime_dir": str(self.runtime)},
                    ),
                    logs_dir=self.logs,
                )

    async def test_credential_reaches_the_runner_only_through_a_private_file(self):
        agent = self.agent()
        environment = self.environment()
        await agent.run("task", environment, AgentContext())
        key_path = next(path for path, content in environment.uploads.items() if content == SECRET + "\n")
        self.assertFalse(key_path.startswith("/logs"))
        self.assertTrue(any(call["command"].endswith(f"chmod 600 {key_path}") for call in environment.calls))
        launch = self.launch_command(environment)
        script = shlex.split(launch)[-1]
        self.assertLess(script.index(f"< {key_path}"), script.index(f"rm -f -- {key_path}"))
        self.assertLess(script.index(f"rm -f -- {key_path}"), script.index('OPENROUTER_API_KEY="$key"'))
        self.assertNotIn(SECRET, environment.uploads["/logs/agent/input.json"])
        for call in environment.calls:
            self.assertNotIn(SECRET, json.dumps(await compose_argv(call)))

    async def test_a_custom_endpoint_gets_no_key_unless_one_is_selected(self):
        for host_key in (SECRET, None):
            with self.subTest(host_key=host_key), patch.dict(os.environ, clear=True):
                if host_key:
                    os.environ["OPENROUTER_API_KEY"] = host_key
                agent = self.agent(base_url="http://127.0.0.1:30000/v1")
                environment = self.environment()
                await agent.run("task", environment, AgentContext())
                config = json.loads(environment.uploads["/logs/agent/input.json"])
                self.assertIsNone(config["api_key_env"])
                self.assertEqual(config["base_url"], "http://127.0.0.1:30000/v1")
                self.assertEqual(set(environment.uploads), {"/logs/agent/input.json"})
                for call in environment.calls:
                    self.assertNotIn("OPENROUTER_API_KEY", call["command"])

    async def test_a_selected_key_reaches_a_custom_endpoint(self):
        agent = self.agent(base_url="http://127.0.0.1:30000/v1", model_api_key_name="OPENROUTER_API_KEY")
        environment = self.environment()
        await agent.run("task", environment, AgentContext())
        config = json.loads(environment.uploads["/logs/agent/input.json"])
        self.assertEqual(config["api_key_env"], "OPENROUTER_API_KEY")
        self.assertIn(SECRET + "\n", environment.uploads.values())

    async def test_a_selected_or_default_key_is_required_before_any_exec(self):
        os.environ.pop("OPENROUTER_API_KEY")
        for options in ({}, {"base_url": "http://127.0.0.1:30000/v1", "model_api_key_name": "TRIAL_MODEL_KEY"}):
            with self.subTest(options=options):
                environment = self.environment()
                with self.assertRaises(ValueError):
                    await self.agent(**options).run("task", environment, AgentContext())
                self.assertEqual(environment.calls, [])

    async def test_finished_run_is_kept_without_a_parsed_identity(self):
        for stdout in ("profile banner\nREMANIE_PROCESS=123:456\n", "", "REMANIE_PROCESS=garbled\n"):
            with self.subTest(stdout=stdout):
                async def launch(environment, stdout=stdout):
                    return ExecResult(stdout=stdout, return_code=0)

                environment = self.environment(launch=launch)
                await self.agent().run("task", environment, AgentContext())
                self.assertFalse(any("kill -TERM" in call["command"] for call in environment.calls))

    async def cancel_running(self, chunk, after_stop=0):
        stopped = asyncio.Event()
        started = asyncio.Event()

        async def launch(environment):
            if chunk:
                await environment.output_callback(chunk, "stdout")
            started.set()
            await stopped.wait()
            return ExecResult(return_code=after_stop)

        environment = self.environment(launch=launch)
        original = environment.exec.side_effect

        async def execute(**kwargs):
            if "kill -TERM" in kwargs["command"]:
                stopped.set()
            return await original(**kwargs)

        environment.exec.side_effect = execute
        task = asyncio.create_task(self.agent().run("task", environment, AgentContext()))
        await asyncio.wait_for(started.wait(), 5)
        task.cancel()
        return environment, task

    async def test_cancellation_stops_the_runner_found_in_mixed_output(self):
        environment, task = await self.cancel_running("banner\nREMANIE_PROCESS=123:456\nmore\n")
        with self.assertRaises(asyncio.CancelledError):
            await task
        stop = environment.calls[-1]
        self.assertIn("bash -c ", stop["command"])
        self.assertIn("pid=123", stop["command"])
        self.assertIn('"${20}" = 456', stop["command"])
        self.assertEqual(stop["timeout_sec"], 10)

    async def test_cancellation_without_identity_requires_teardown(self):
        _, task = await self.cancel_running(None)
        with self.assertRaisesRegex(RuntimeError, "REMANIE_PROCESS line did not arrive"):
            await task

    async def test_cancelled_runner_that_dies_badly_requires_teardown(self):
        _, task = await self.cancel_running("REMANIE_PROCESS=123:456\n", after_stop=137)
        with self.assertRaisesRegex(RuntimeError, "Tear down the container"):
            await task

    async def test_stop_script_checks_process_identity(self):
        agent = self.agent()
        stat_file = self.root / "process.stat"
        execution = asyncio.get_running_loop().create_future()
        execution.set_result(ExecResult(return_code=0))
        environment = self.environment()

        async def execute(**kwargs):
            command = kwargs["command"].replace('"/proc/$pid/stat"', shlex.quote(str(stat_file)))
            process = await asyncio.create_subprocess_exec(
                "bash", "-c", command,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await process.communicate()
            return ExecResult(stdout=stdout.decode(), stderr=stderr.decode(), return_code=process.returncode)

        environment.exec.side_effect = execute
        stat_file.write_text("")
        await agent._stop_runtime(environment, ("123", "456"), execution)
        fields = ["S", "1", "123", "123", *(["0"] * 15), "789"]
        stat_file.write_text("123 (runtime) " + " ".join(fields) + "\n")
        with self.assertRaisesRegex(RuntimeError, "could not confirm that the runner exited"):
            await agent._stop_runtime(environment, ("123", "456"), execution)

    async def test_install_refuses_a_bundle_for_another_architecture(self):
        environment = self.environment(machine="aarch64")
        with self.assertRaisesRegex(RuntimeError, "x86_64"):
            await self.agent().setup(environment)
        environment.upload_dir.assert_not_awaited()

    def test_usage_comes_from_result_json_only_when_complete(self):
        agent = self.agent()
        result = self.logs / "result.json"
        for usage, expected in (
            ({"prompt_tokens": 30, "completion_tokens": 8, "cached_tokens": 4, "complete": True}, (30, 8, 4)),
            ({"prompt_tokens": 30, "completion_tokens": 8, "cached_tokens": 4, "complete": False}, (None, None, None)),
            ({"prompt_tokens": -1, "completion_tokens": 8, "cached_tokens": 4, "complete": True}, (None, None, None)),
        ):
            with self.subTest(usage=usage):
                result.write_text(json.dumps({"outcome": "max_turns", "usage": usage}))
                context = AgentContext()
                agent.populate_context_post_run(context)
                self.assertEqual((context.n_input_tokens, context.n_output_tokens, context.n_cache_tokens), expected)
                self.assertEqual(context.metadata["remanie"]["result"]["usage"], usage)

    def test_unsafe_or_missing_result_is_recorded_not_read(self):
        agent = self.agent()
        self.assertIsNone(self.metadata(agent)["result"])
        result = self.logs / "result.json"
        result.write_bytes(b"")
        metadata = self.metadata(agent)
        self.assertIsNone(metadata["result"])
        self.assertNotIn("result_invalid", metadata)
        with result.open("wb") as file:
            file.truncate(16 * 1024 * 1024 + 1)
        self.assertTrue(self.metadata(agent)["result_invalid"])
        result.unlink()
        secret = self.root / "secret"
        secret.write_text(json.dumps({"host": "secret-must-not-be-read"}))
        result.symlink_to(secret)
        metadata = self.metadata(agent)
        self.assertTrue(metadata["result_invalid"])
        self.assertNotIn("secret-must-not-be-read", json.dumps(metadata))


if __name__ == "__main__":
    unittest.main()
