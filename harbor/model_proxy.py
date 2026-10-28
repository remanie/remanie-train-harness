import asyncio
import json
import secrets
from contextlib import asynccontextmanager
from urllib.parse import urlsplit

from aiohttp import ClientError, ClientSession, ClientTimeout, web

MAX_BODY = 16 * 1024 * 1024


@asynccontextmanager
async def model_proxy(base_url, api_key, model, max_tokens, max_requests, deadline_secs, sampling_params, tools):
    parsed = urlsplit(base_url)
    if (
        parsed.scheme not in {"http", "https"} or not parsed.hostname
        or parsed.username or parsed.password or parsed.query or parsed.fragment
        or not parsed.path.rstrip("/").endswith("/v1")
    ):
        raise ValueError("Upstream must be an HTTP(S) /v1 URL without credentials")
    token = secrets.token_urlsafe(32)
    expected_auth = f"Bearer {token}".encode()
    allowed_fields = {"model", "messages", "tools", "tool_choice", "max_tokens", "stream"} | sampling_params.keys()
    requests = 0
    lock = asyncio.Lock()
    timeout = ClientTimeout(total=deadline_secs, connect=30)
    async with ClientSession(timeout=timeout) as client:
        async def forward(request):
            if not secrets.compare_digest(request.headers.get("Authorization", "").encode(), expected_auth):
                raise web.HTTPUnauthorized()
            if lock.locked():
                raise web.HTTPTooManyRequests(text="Only one request may be active per attempt")
            async with lock:
                return await forward_authorized(request)

        async def forward_authorized(request):
            nonlocal requests
            try:
                payload = json.loads(await request.read())
            except (ValueError, RecursionError):
                raise web.HTTPBadRequest(text="Invalid JSON") from None
            if (
                not isinstance(payload, dict) or payload.keys() - allowed_fields
                or payload.get("model") != model
                or payload.get("tools") != tools
                or payload.get("tool_choice") not in ("auto", "none")
                or any(payload.get(key) != value for key, value in sampling_params.items())
                or type(payload.get("max_tokens")) is not int
                or not 0 < payload["max_tokens"] <= max_tokens
                or payload.get("stream", False) is not False
            ):
                raise web.HTTPBadRequest(text="Request exceeds this attempt's model limits")
            if requests >= max_requests:
                raise web.HTTPTooManyRequests(text="Attempt request limit reached")
            requests += 1
            try:
                async with client.post(
                    f"{base_url.rstrip('/')}/chat/completions",
                    json=payload,
                    headers={"Authorization": f"Bearer {api_key}"},
                    allow_redirects=False,
                ) as response:
                    body = bytearray()
                    async for chunk in response.content.iter_chunked(65536):
                        if len(body) + len(chunk) > MAX_BODY:
                            raise web.HTTPBadGateway(text="Upstream response exceeds 16 MiB")
                        body.extend(chunk)
                    return web.Response(status=response.status, body=body, content_type="application/json")
            except (ClientError, asyncio.TimeoutError):
                raise web.HTTPBadGateway(text="Upstream request failed") from None

        app = web.Application(client_max_size=MAX_BODY)
        app.router.add_post("/v1/chat/completions", forward)
        runner = web.AppRunner(app, access_log=None, shutdown_timeout=1)
        await runner.setup()
        try:
            site = web.TCPSite(runner, "0.0.0.0", 0)
            await site.start()
            port = runner.addresses[0][1]
            yield f"http://host.docker.internal:{port}/v1", token
        finally:
            await runner.cleanup()
