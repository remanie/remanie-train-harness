import asyncio
import unittest
from unittest.mock import patch

from aiohttp import ClientError, ClientSession, web

from model_proxy import model_proxy


class ModelProxyTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.requests = []
        self.reply_status = 200
        self.reply_body = b'{"choices":[]}'
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.release.set()

        async def upstream(request):
            self.requests.append((request.headers.get("Authorization"), await request.json()))
            self.entered.set()
            await self.release.wait()
            return web.Response(
                status=self.reply_status, body=self.reply_body,
                headers={"Location": f"{self.origin}/should-not-follow"},
            )

        app = web.Application()
        app.router.add_post("/v1/chat/completions", upstream)
        self.runner = web.AppRunner(app, access_log=None)
        await self.runner.setup()
        await web.TCPSite(self.runner, "127.0.0.1", 0).start()
        self.origin = f"http://127.0.0.1:{self.runner.addresses[0][1]}"
        self.client = ClientSession()

    async def asyncTearDown(self):
        await self.client.close()
        await self.runner.cleanup()

    async def test_authentication_route_model_and_cost_limits(self):
        tools = [{"type": "function", "function": {"name": "read_file"}}]
        async with model_proxy(f"{self.origin}/v1", "host-key", "x-ai/grok-4.7", 20, 2, 10, {"reasoning": {"effort": "high"}}, tools) as (url, token):
            url = url.replace("host.docker.internal", "127.0.0.1")
            payload = {"model": "x-ai/grok-4.7", "max_tokens": 20, "messages": [], "stream": False,
                       "reasoning": {"effort": "high"}, "tools": tools, "tool_choice": "auto"}
            headers = {"Authorization": f"Bearer {token}"}
            async with self.client.post(f"{url}/chat/completions", json=payload) as response:
                self.assertEqual(response.status, 401)
            async with self.client.post(f"{url}/other", json=payload, headers=headers) as response:
                self.assertEqual(response.status, 404)
            for change in (
                {"model": "other"}, {"max_tokens": 21}, {"max_tokens": True},
                {"max_completion_tokens": 1000}, {"stream": True}, {"models": ["other"]},
                {"plugins": [{"id": "web"}]}, {"provider": {"order": ["other"]}},
                {"tools": [{"type": "web_search"}]}, {"tool_choice": "required"},
                {"reasoning": {"effort": "low"}},
            ):
                async with self.client.post(f"{url}/chat/completions", json={**payload, **change}, headers=headers) as response:
                    self.assertEqual(response.status, 400)
            self.assertFalse(self.requests)
            for _ in range(2):
                async with self.client.post(f"{url}/chat/completions", json=payload, headers=headers) as response:
                    self.assertEqual(response.status, 200)
                    self.assertNotIn("host-key", await response.text())
            async with self.client.post(f"{url}/chat/completions", json=payload, headers=headers) as response:
                self.assertEqual(response.status, 429)
            self.assertEqual(self.requests, [("Bearer host-key", payload)] * 2)
            self.assertNotEqual(token, "host-key")
        with self.assertRaises(ClientError):
            await self.client.post(f"{url}/chat/completions", json=payload, headers=headers)

    async def test_response_limit_and_redirect_are_not_forwarded(self):
        with patch("model_proxy.MAX_BODY", 1024):
            async with model_proxy(f"{self.origin}/v1", "host-key", "model", 20, 3, 10, {}, []) as (url, token):
                endpoint = url.replace("host.docker.internal", "127.0.0.1") + "/chat/completions"
                payload = {"model": "model", "max_tokens": 20, "tools": [], "tool_choice": "none"}
                headers = {"Authorization": f"Bearer {token}"}
                self.reply_body = b"x" * 1025
                async with self.client.post(endpoint, json=payload, headers=headers) as response:
                    self.assertEqual(response.status, 502)
                self.reply_status = 302
                self.reply_body = b"redirect"
                async with self.client.post(endpoint, json=payload, headers=headers) as response:
                    self.assertEqual(response.status, 302)
                    self.assertNotIn("Location", response.headers)
                async with self.client.post(endpoint, data=b"x" * 1025, headers=headers) as response:
                    self.assertEqual(response.status, 413)
                self.assertEqual(len(self.requests), 2)

    async def test_parallel_requests_do_not_reach_upstream(self):
        self.release.clear()
        async with model_proxy(f"{self.origin}/v1", "host-key", "model", 20, 3, 10, {}, []) as (url, token):
            endpoint = url.replace("host.docker.internal", "127.0.0.1") + "/chat/completions"
            payload = {"model": "model", "max_tokens": 20, "tools": [], "tool_choice": "none"}
            headers = {"Authorization": f"Bearer {token}"}
            first = asyncio.create_task(self.client.post(endpoint, json=payload, headers=headers))
            try:
                await asyncio.wait_for(self.entered.wait(), 5)
                async with self.client.post(endpoint, json=payload, headers=headers) as response:
                    self.assertEqual(response.status, 429)
                self.assertEqual(len(self.requests), 1)
            finally:
                self.release.set()
                response = await first
                response.release()


if __name__ == "__main__":
    unittest.main()
