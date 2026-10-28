import json
import tempfile
import unittest
from pathlib import Path

import remanie_trace

TOOLS = [{"type": "function", "function": {"name": "read_file", "description": "Read", "parameters": {"type": "object"}}}]
SYSTEM = {"role": "system", "content": "You are a coding agent."}
QUERY = {"role": "user", "content": "Fix the failing test."}
CALL = {"id": "call-1", "type": "function", "function": {"name": "read_file", "arguments": "{\"target_file\": \"a.py\"}"}}
TURN_1 = {"role": "assistant", "content": None, "reasoning_content": "Look first.", "tool_calls": [CALL]}
TOOL_1 = {"role": "tool", "tool_call_id": "call-1", "content": "1|print('a')"}
SUMMARIZE = {"role": "user", "content": "Summarize the work so far."}
SUMMARY = {"role": "assistant", "content": "<summary>Read a.py.</summary>"}
CONTINUE = {"role": "user", "content": "Continue from the summary: Read a.py."}
DONE = {"role": "assistant", "content": "Fixed."}
RESULT = {"outcome": {"kind": "completed", "final_answer": "Fixed."}, "turns": 2, "compactions": 1}


def response(message, prompt_tokens, cached=None):
    usage = {"prompt_tokens": prompt_tokens, "completion_tokens": 5}
    if cached is not None:
        usage["prompt_tokens_details"] = {"cached_tokens": cached}
    return {"choices": [{"message": message, "finish_reason": "stop"}], "usage": usage}


def message(segment, index, body):
    return {"event": "message", "segment": segment, "index": index, "message": body}


def model_call(call, kind, segment, end, reply, override=None):
    return {"event": "model_call", "call": call, "kind": kind, "prompt": {"segment": segment, "end": end},
            "max_tokens_override": override, "request_sha256": "0" * 64, "response": reply, "attempts": 1}


EVENTS = [
    {"event": "header", "config": {"model": "qwen", "sampling_params": {"max_tokens": 4096, "temperature": 1.0}},
     "tools": TOOLS, "prompts_sha256": "1" * 64, "bundle_sha256": None},
    message(0, 0, SYSTEM),
    message(0, 1, QUERY),
    {"event": "endpoint_error", "call": 0, "attempt": 1, "status": 503, "body_excerpt": "abort", "verdict": "retry"},
    model_call(0, "turn", 0, 2, response(TURN_1, 100, cached=60)),
    message(0, 2, TURN_1),
    message(0, 3, TOOL_1),
    {"event": "tool_call", "call_id": "call-1", "name": "read_file", "class": "ok", "latency_ms": 3, "output_bytes": 12},
    message(0, 4, SUMMARIZE),
    model_call(1, "compaction", 0, 5, response(SUMMARY, 150), override=512),
    message(0, 5, SUMMARY),
    {"event": "compaction", "from": 0, "to": 1, "summary_call": 1, "estimated_tokens": 150},
    message(1, 0, SYSTEM),
    message(1, 1, QUERY),
    message(1, 2, CONTINUE),
    model_call(2, "turn", 1, 3, response(DONE, 40)),
    message(1, 3, DONE),
    {"event": "result", **RESULT},
]


class TraceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.session = Path(self.temp.name)

    def write(self, events, tail=""):
        (self.session / "trace.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events) + tail)
        return remanie_trace.read(self.session)

    def test_requests_are_rebuilt_from_the_message_log(self):
        trace = self.write(EVENTS)
        self.assertEqual([call.kind for call in trace.calls], ["turn", "compaction", "turn"])
        base = {"max_tokens": 4096, "temperature": 1.0, "model": "qwen", "tools": TOOLS, "stream": False}
        self.assertEqual(trace.request_body(trace.calls[0]), {**base, "messages": [SYSTEM, QUERY], "tool_choice": "auto"})
        self.assertEqual(trace.request_body(trace.calls[1]), {
            **base, "max_tokens": 512, "tool_choice": "none",
            "messages": [SYSTEM, QUERY, TURN_1, TOOL_1, SUMMARIZE],
        })
        self.assertEqual(trace.request_body(trace.calls[2]), {
            **base, "tool_choice": "auto", "messages": [SYSTEM, QUERY, CONTINUE],
        })

    def test_tool_results_errors_and_outcome(self):
        trace = self.write(EVENTS)
        self.assertEqual(trace.tool_results, (remanie_trace.ToolResult("call-1", "read_file", "ok", "1|print('a')"),))
        self.assertEqual([error["status"] for error in trace.endpoint_errors], [503])
        self.assertEqual(trace.outcome, {"kind": "completed", "final_answer": "Fixed."})
        self.assertEqual(trace.result, RESULT)
        self.assertTrue(trace.complete)
        self.assertEqual(trace.usage(), remanie_trace.Usage(290, 15, 60, complete=True))

    def test_a_torn_or_unfinished_trace_is_incomplete(self):
        for events, tail in ((EVENTS, '{"event": "mess'), (EVENTS[:-1], "")):
            with self.subTest(tail=tail, events=len(events)):
                trace = self.write(events, tail)
                self.assertFalse(trace.complete)
                self.assertFalse(trace.usage().complete)
                self.assertEqual(len(trace.calls), 3)

    def test_missing_usage_makes_totals_incomplete(self):
        events = [dict(event) for event in EVENTS]
        events[4] = {**events[4], "response": {"choices": []}}
        self.assertFalse(self.write(events).usage().complete)

    def test_inconsistent_traces_are_refused(self):
        for events in (EVENTS[1:], EVENTS[:2] + EVENTS[3:], EVENTS[:1] + [model_call(0, "turn", 0, 2, {})]):
            with self.subTest(first=events[1]["event"]), self.assertRaises(ValueError):
                self.write(events)

    def test_symlinked_trace_is_not_followed(self):
        target = self.session / "elsewhere.jsonl"
        target.write_text(json.dumps(EVENTS[0]) + "\n")
        (self.session / "trace.jsonl").symlink_to(target)
        with self.assertRaises(OSError):
            remanie_trace.read(self.session)


if __name__ == "__main__":
    unittest.main()
