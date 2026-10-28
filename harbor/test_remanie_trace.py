import json
import tempfile
import unittest
from itertools import groupby
from pathlib import Path

import remanie_trace

FIXTURES = Path(__file__).resolve().parent / "fixtures"
LINES = (FIXTURES / "trace.jsonl").read_text().splitlines(keepends=True)
WIRE = (FIXTURES / "requests.jsonl").read_bytes().splitlines()


class TraceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.session = Path(self.temp.name)

    def write(self, lines):
        (self.session / "trace.jsonl").write_text("".join(lines))
        return remanie_trace.read(self.session)

    def test_every_request_is_rebuilt_from_the_trace(self):
        trace = self.write(LINES)
        self.assertEqual([call.kind for call in trace.calls], ["turn", "compaction", "turn"])
        self.assertEqual(WIRE[0], WIRE[1], "a retry resends the same bytes")
        sent = [json.loads(body) for body, _ in groupby(WIRE)]
        self.assertEqual([trace.request_body(call) for call in trace.calls], sent)

    def test_tool_results_errors_and_outcome(self):
        trace = self.write(LINES)
        self.assertEqual(
            [(result.call_id, result.name, result.status) for result in trace.tool_results],
            [("call-write", "run_terminal_cmd", "ok"), ("call-read", "read_file", "ok")],
        )
        self.assertIn("hello", trace.tool_results[1].content)
        self.assertEqual([error["status"] for error in trace.endpoint_errors], [503])
        self.assertEqual(trace.outcome, "completed")
        self.assertEqual(trace.result["compactions"], 1)
        self.assertTrue(trace.complete)
        self.assertEqual(trace.usage(), remanie_trace.Usage(**trace.result["usage"]))

    def test_a_torn_or_unfinished_trace_is_incomplete(self):
        for lines in (LINES[:-1] + [LINES[-1][:20]], LINES[:-1]):
            with self.subTest(last=lines[-1][:20]):
                trace = self.write(lines)
                self.assertFalse(trace.complete)
                self.assertFalse(trace.usage().complete)
                self.assertEqual(len(trace.calls), 3)

    def test_missing_usage_makes_totals_incomplete(self):
        events = [json.loads(line) for line in LINES]
        call = next(event for event in events if event["event"] == "model_call")
        del call["response"]["usage"]
        self.assertFalse(self.write(json.dumps(event) + "\n" for event in events).usage().complete)

    def test_inconsistent_traces_are_refused(self):
        kinds = [json.loads(line)["event"] for line in LINES]
        first_call = kinds.index("model_call")
        for lines in (LINES[1:], LINES[:1] + LINES[2:], LINES[:1] + [LINES[first_call]]):
            with self.subTest(second=json.loads(lines[1])["event"]), self.assertRaises(ValueError):
                self.write(lines)

    def test_symlinked_trace_is_not_followed(self):
        target = self.session / "elsewhere.jsonl"
        target.write_text(LINES[0])
        (self.session / "trace.jsonl").symlink_to(target)
        with self.assertRaises(OSError):
            remanie_trace.read(self.session)


if __name__ == "__main__":
    unittest.main()
