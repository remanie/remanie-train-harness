import asyncio
import json
import os
import shlex
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import AsyncMock, patch

from harbor.agents.factory import AgentFactory
from harbor.agents.installed.base import NonZeroAgentExitCodeError
from harbor.environments.base import BaseEnvironment, ExecResult
from harbor.models.agent.context import AgentContext
from harbor.models.trial.config import AgentConfig

from remanie_agent import RemanieAgent


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
        self.logs = self.root / "logs"
        self.logs.mkdir()
        self.env = patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-only-secret"})
        self.env.start()
        self.addCleanup(self.env.stop)

    def agent(self, **kwargs):
        return AgentFactory.create_agent_from_config(
            AgentConfig(
                import_path="remanie_agent:RemanieAgent",
                model_name="openrouter/x-ai/grok-4.7",
                kwargs={"runtime_dir": str(self.runtime), **kwargs},
            ),
            logs_dir=self.logs,
        )

    def environment(self):
        environment = AsyncMock(spec=BaseEnvironment)
        environment.default_user = None
        environment.type.return_value = "docker"

        @contextmanager
        def output_callback(callback):
            environment.output_callback = callback
            yield

        environment.scoped_output_callback = output_callback
        return environment

    async def test_real_harbor_factory_connection_and_task_execution(self):
        agent = self.agent(sampling_params={"reasoning": {"effort": "high"}})
        self.assertIsInstance(agent, RemanieAgent)
        self.assertEqual(agent.model_connection.candidate_base_url, "https://openrouter.ai/api/v1")
        self.assertEqual(agent.to_agent_info().model_info.name, "x-ai/grok-4.7")
        environment = self.environment()

        async def execute(**kwargs):
            if "exec setsid" in kwargs["command"]:
                await environment.output_callback("REMANIE_PROCESS=123:456\n", "stdout")
                await environment.output_callback("REMANIE_PROCESS=999:999\n", "stdout")
            return ExecResult(stdout="/task with spaces\n", return_code=0)

        environment.exec.side_effect = execute
        uploaded = {}

        async def upload(source, target):
            uploaded[target] = Path(source).read_text()

        environment.upload_file.side_effect = upload
        await agent.setup(environment)
        environment.upload_dir.assert_awaited_once_with(self.runtime.resolve(), "/opt/remanie-train-harness")
        context = AgentContext()
        instruction = "Fix /task/input.txt. Preserve the user's exact instruction."
        await agent.run(instruction, environment, context)
        config = json.loads(uploaded["/logs/agent/input.json"])
        self.assertEqual(config["user_query"], instruction)
        self.assertEqual(config["workspace"], "/task with spaces")
        self.assertEqual(config["session_dir"], "/logs/agent/remanie")
        self.assertEqual(config["api_key_env"], "OPENROUTER_API_KEY")
        self.assertEqual(config["model"], "x-ai/grok-4.7")
        self.assertEqual(config["sampling_params"], {"reasoning": {"effort": "high"}})
        self.assertEqual(config["deadline_secs"], 86400)
        self.assertTrue(context.is_empty())
        self.assertTrue(config["base_url"].startswith("http://host.docker.internal:"))
        run_call = next(call.kwargs for call in environment.exec.await_args_list if call.kwargs.get("env"))
        self.assertNotEqual(run_call["env"]["OPENROUTER_API_KEY"], "test-only-secret")
        self.assertIn("exec /opt/remanie-train-harness/remanie-train-harness", run_call["command"])
        self.assertNotIn("test-only-secret", json.dumps(uploaded))
        for call in environment.exec.await_args_list:
            self.assertNotIn("test-only-secret", call.kwargs["command"])
            self.assertIsNone(call.kwargs.get("cwd"))
        cleanup = environment.exec.await_args_list[-1].kwargs["command"]
        self.assertIn("pid=123", cleanup)
        self.assertIn('"${20}" = 456', cleanup)
        self.assertNotIn("runtime.pid", cleanup)

    async def test_missing_key_fails_before_execution(self):
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": ""}):
            agent = self.agent()
            environment = AsyncMock(spec=BaseEnvironment)
            with self.assertRaisesRegex(ValueError, "OPENROUTER_API_KEY"):
                await agent.run("task", environment, AgentContext())
            environment.exec.assert_not_awaited()

    def test_scoped_credential_precedes_host_and_options_are_validated(self):
        agent = RemanieAgent(
            logs_dir=self.logs,
            runtime_dir=str(self.runtime),
            extra_env={"OPENROUTER_API_KEY": "scoped-test-secret"},
        )
        self.assertEqual(agent.model_connection.env["OPENROUTER_API_KEY"], "scoped-test-secret")
        self.assertEqual(agent.extra_env, {})
        for options in ({"max_turns": 0}, {"runtime_dir": str(self.root)}, {"prompt_template_path": "wrapper.jinja2"}, {"unknown": True}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.agent(**options)

    async def test_cancellation_stops_runtime_before_proxy_closes(self):
        agent = self.agent()
        environment = self.environment()
        started = asyncio.Event()
        stopped = asyncio.Event()

        async def execute(**kwargs):
            if "exec setsid" in kwargs["command"]:
                await environment.output_callback("REMANIE_PROCESS=123:456\n", "stdout")
                started.set()
                await stopped.wait()
            elif 'kill -TERM -- "-$pid"' in kwargs["command"]:
                stopped.set()
            return ExecResult(stdout="/task\n", return_code=0)

        environment.exec.side_effect = execute
        task = asyncio.create_task(agent.run("task", environment, AgentContext()))
        await asyncio.wait_for(started.wait(), 5)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        cleanup = environment.exec.await_args_list[-1].kwargs
        self.assertIn('kill -TERM -- "-$pid"', cleanup["command"])
        self.assertIn('kill -KILL -- "-$pid"', cleanup["command"])
        self.assertEqual(cleanup["timeout_sec"], 10)

    async def test_crashed_runtime_cannot_be_confirmed_by_a_missing_process(self):
        agent = self.agent()
        environment = self.environment()

        async def execute(**kwargs):
            if "exec setsid" in kwargs["command"]:
                await environment.output_callback("REMANIE_PROCESS=123:456\n", "stdout")
                return ExecResult(stderr="runtime was killed", return_code=137)
            return ExecResult(stdout="/task\n", return_code=0)

        environment.exec.side_effect = execute
        with self.assertRaisesRegex(RuntimeError, "shutdown was not confirmed") as raised:
            await agent.run("task", environment, AgentContext())
        self.assertNotIsInstance(raised.exception, NonZeroAgentExitCodeError)

    async def test_cancelled_attempt_still_checks_the_runtime_exit_status(self):
        agent = self.agent()
        environment = self.environment()
        started = asyncio.Event()
        stopped = asyncio.Event()

        async def execute(**kwargs):
            if "exec setsid" in kwargs["command"]:
                await environment.output_callback("REMANIE_PROCESS=123:456\n", "stdout")
                started.set()
                await stopped.wait()
                return ExecResult(stderr="runtime was already killed", return_code=137)
            if 'kill -TERM -- "-$pid"' in kwargs["command"]:
                stopped.set()
            return ExecResult(stdout="/task\n", return_code=0)

        environment.exec.side_effect = execute
        task = asyncio.create_task(agent.run("task", environment, AgentContext()))
        await asyncio.wait_for(started.wait(), 5)
        task.cancel()
        with self.assertRaisesRegex(RuntimeError, "shutdown was not confirmed"):
            await task

    async def test_unconfirmed_shutdown_fails_closed(self):
        agent = self.agent()
        environment = self.environment()
        execution = asyncio.get_running_loop().create_future()
        execution.set_result(None)
        with self.assertRaisesRegex(RuntimeError, "identity was not captured"):
            await agent._stop_runtime(environment, None, execution)
        environment.exec.assert_not_awaited()
        environment.exec.side_effect = NonZeroAgentExitCodeError("cleanup failed")
        with self.assertRaisesRegex(RuntimeError, "shutdown was not confirmed") as raised:
            await agent._stop_runtime(environment, ("123", "456"), execution)
        self.assertNotIsInstance(raised.exception, NonZeroAgentExitCodeError)

    async def test_proc_read_eof_allows_confirmed_exit_but_identity_mismatch_fails(self):
        agent = self.agent()
        environment = self.environment()
        stat_file = self.root / "process.stat"
        execution = asyncio.get_running_loop().create_future()
        execution.set_result(ExecResult(return_code=0))

        async def execute(**kwargs):
            command = kwargs["command"].replace('"/proc/$pid/stat"', shlex.quote(str(stat_file)))
            process = await asyncio.create_subprocess_exec(
                "/bin/bash", "-c", command,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await process.communicate()
            return ExecResult(stdout=stdout.decode(), stderr=stderr.decode(), return_code=process.returncode)

        environment.exec.side_effect = execute
        stat_file.write_text("")
        await agent._stop_runtime(environment, ("123", "456"), execution)
        fields = ["S", "1", "123", "123", *(["0"] * 15), "789"]
        stat_file.write_text("123 (runtime) " + " ".join(fields) + "\n")
        with self.assertRaisesRegex(RuntimeError, "shutdown was not confirmed"):
            await agent._stop_runtime(environment, ("123", "456"), execution)

    def test_oversized_or_symlink_logs_are_not_read(self):
        agent = self.agent()
        result = self.logs / "result.json"
        with result.open("wb") as file:
            file.truncate(16 * 1024 * 1024 + 1)
        trace_dir = self.logs / "remanie"
        trace_dir.mkdir()
        trace = trace_dir / "trace.jsonl"
        with trace.open("wb") as file:
            file.truncate(256 * 1024 * 1024 + 1)
        context = AgentContext()
        agent.populate_context_post_run(context)
        self.assertTrue(context.metadata["remanie"]["result_invalid"])
        self.assertTrue(context.metadata["remanie"]["trace_incomplete"])
        result.unlink()
        trace.unlink()
        secret = self.root / "secret"
        secret.write_text("host-secret-must-not-be-read")
        result.symlink_to(secret)
        trace.symlink_to(secret)
        context = AgentContext()
        agent.populate_context_post_run(context)
        self.assertTrue(context.metadata["remanie"]["result_invalid"])
        self.assertTrue(context.metadata["remanie"]["trace_incomplete"])
        self.assertNotIn(secret.read_text(), context.model_dump_json())

    def test_long_trace_line_is_bounded(self):
        agent = self.agent()
        trace_dir = self.logs / "remanie"
        trace_dir.mkdir()
        with (trace_dir / "trace.jsonl").open("wb") as file:
            file.truncate(16 * 1024 * 1024 + 1025)
        context = AgentContext()
        agent.populate_context_post_run(context)
        self.assertTrue(context.metadata["remanie"]["trace_incomplete"])

    def test_post_run_records_status_and_usage_including_compaction(self):
        agent = self.agent()
        (self.logs / "result.json").write_text(json.dumps({
            "status": "max_turns", "turns": 2, "compactions": 1, "final_answer": "",
        }))
        trace = self.logs / "remanie"
        trace.mkdir()
        events = [
            {"event": "model_request", "body": {}},
            {"event": "model_response", "body": {"usage": {
                "prompt_tokens": 10, "completion_tokens": 3,
                "prompt_tokens_details": {"cached_tokens": 4}, "cost": 0.01,
            }}},
            {"event": "compaction_request", "body": {}},
            {"event": "compaction_response", "body": {"usage": {
                "prompt_tokens": 20, "completion_tokens": 5, "cost": 0.02,
            }}},
        ]
        (trace / "trace.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
        context = AgentContext()
        agent.populate_context_post_run(context)
        self.assertEqual((context.n_input_tokens, context.n_output_tokens, context.n_cache_tokens), (30, 8, 4))
        self.assertAlmostEqual(context.cost_usd, 0.03)
        self.assertEqual(context.metadata["remanie"]["status"], "max_turns")
        with (trace / "trace.jsonl").open("a") as file:
            file.write('{"event":')
        context = AgentContext()
        agent.populate_context_post_run(context)
        self.assertTrue(context.metadata["remanie"]["trace_incomplete"])
        self.assertIsNone(context.n_input_tokens)
        self.assertIsNone(context.cost_usd)

    def test_missing_response_keeps_cost_and_usage_totals_unknown(self):
        agent = self.agent()
        trace = self.logs / "remanie"
        trace.mkdir()
        events = [
            {"event": "model_request", "body": {}},
            {"event": "model_response", "body": {"usage": {
                "prompt_tokens": 10, "completion_tokens": 3, "cost": 0.01,
            }}},
            {"event": "model_request", "body": {}},
        ]
        (trace / "trace.jsonl").write_text("".join(json.dumps(event) + "\n" for event in events))
        context = AgentContext()
        agent.populate_context_post_run(context)
        self.assertIsNone(context.n_input_tokens)
        self.assertIsNone(context.cost_usd)
        self.assertEqual(context.metadata["remanie"]["request_count"], 2)
        self.assertFalse(context.metadata["remanie"]["usage_complete"])

    def test_partial_usage_is_not_reported_as_complete(self):
        agent = self.agent()
        trace = self.logs / "remanie"
        trace.mkdir()
        (trace / "trace.jsonl").write_text("\n".join(json.dumps(event) for event in [
            {"event": "model_response", "body": {"usage": {"prompt_tokens": 3}}},
            {"event": "model_response", "body": {"usage": []}},
        ]) + "\n")
        context = AgentContext()
        agent.populate_context_post_run(context)
        self.assertIsNone(context.n_input_tokens)
        self.assertIsNone(context.cost_usd)
        self.assertFalse(context.metadata["remanie"]["usage_complete"])


if __name__ == "__main__":
    unittest.main()
