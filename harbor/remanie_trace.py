import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

MAX_LINE_BYTES = 17 * 1024 * 1024


@dataclass(frozen=True)
class Call:
    index: int
    kind: str
    segment: int
    end: int
    max_tokens_override: int | None
    response: dict[str, Any]


@dataclass(frozen=True)
class ToolResult:
    call_id: str
    name: str
    status: str
    content: str | None


@dataclass(frozen=True)
class Usage:
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int
    complete: bool


@dataclass(frozen=True)
class Trace:
    header: dict[str, Any]
    segments: tuple[tuple[dict[str, Any], ...], ...]
    calls: tuple[Call, ...]
    tool_results: tuple[ToolResult, ...]
    endpoint_errors: tuple[dict[str, Any], ...]
    result: dict[str, Any] | None
    complete: bool

    @property
    def outcome(self) -> dict[str, Any] | None:
        return self.result["outcome"] if self.result else None

    def request_body(self, call: Call) -> dict[str, Any]:
        config = self.header["config"]
        body = {
            **config["sampling_params"],
            "model": config["model"],
            "messages": list(self.segments[call.segment][:call.end]),
            "tools": self.header["tools"],
            "tool_choice": "none" if call.kind == "compaction" else "auto",
            "stream": False,
        }
        if call.max_tokens_override is not None:
            body["max_tokens"] = call.max_tokens_override
        return body

    def usage(self) -> Usage:
        totals = [0, 0, 0]
        complete = self.complete
        for call in self.calls:
            usage = call.response.get("usage")
            if not isinstance(usage, dict):
                complete = False
                continue
            details = usage.get("prompt_tokens_details")
            counts = (
                usage.get("prompt_tokens"),
                usage.get("completion_tokens"),
                details.get("cached_tokens", 0) if isinstance(details, dict) else 0,
            )
            if all(type(count) is int and count >= 0 for count in counts):
                totals = [total + count for total, count in zip(totals, counts)]
            else:
                complete = False
        return Usage(*totals, complete=complete)


def fold(events: Iterable[dict[str, Any]], *, torn: bool = False) -> Trace:
    header = None
    segments: list[list[dict[str, Any]]] = []
    calls: list[Call] = []
    tool_results: list[ToolResult] = []
    endpoint_errors: list[dict[str, Any]] = []
    result = None
    last_message: dict[str, Any] | None = None
    for event in events:
        kind = event.get("event")
        if kind == "header":
            header = event
        elif kind == "message":
            segment, index = event["segment"], event["index"]
            if segment == len(segments):
                segments.append([])
            if segment >= len(segments) or index != len(segments[segment]):
                raise ValueError(f"message {segment}:{index} is out of order")
            last_message = event["message"]
            segments[segment].append(last_message)
        elif kind == "model_call":
            prompt = event["prompt"]
            if prompt["segment"] >= len(segments) or prompt["end"] > len(segments[prompt["segment"]]):
                raise ValueError(f"call {event['call']} names messages the trace does not hold")
            calls.append(Call(
                index=event["call"],
                kind=event["kind"],
                segment=prompt["segment"],
                end=prompt["end"],
                max_tokens_override=event.get("max_tokens_override"),
                response=event["response"],
            ))
        elif kind == "tool_call":
            replied = (
                last_message is not None
                and last_message.get("role") == "tool"
                and last_message.get("tool_call_id") == event["call_id"]
            )
            tool_results.append(ToolResult(
                call_id=event["call_id"],
                name=event["name"],
                status=event["class"],
                content=last_message.get("content") if replied else None,
            ))
        elif kind == "endpoint_error":
            endpoint_errors.append(event)
        elif kind == "result":
            result = {key: value for key, value in event.items() if key != "event"}
    if header is None:
        raise ValueError("trace has no header")
    return Trace(
        header=header,
        segments=tuple(tuple(segment) for segment in segments),
        calls=tuple(calls),
        tool_results=tuple(tool_results),
        endpoint_errors=tuple(endpoint_errors),
        result=result,
        complete=result is not None and not torn,
    )


def read(session_dir: str | Path) -> Trace:
    descriptor = os.open(Path(session_dir) / "trace.jsonl", os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as file:
        if not stat.S_ISREG(os.fstat(file.fileno()).st_mode):
            raise ValueError("trace.jsonl must be a regular file")
        events = []
        torn = False
        while line := file.readline(MAX_LINE_BYTES + 1):
            if len(line) > MAX_LINE_BYTES:
                raise ValueError("trace line exceeds the read limit")
            if not line.endswith(b"\n"):
                torn = True
                break
            events.append(json.loads(line))
    return fold(events, torn=torn)
