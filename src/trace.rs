use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::Path;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use serde::Serialize;
use serde_json::Value;

use crate::model::Failure;
use crate::outcome::RunResult;
use crate::runner::ToolErrorKind;

pub const MAX_TRACE_BYTES: u64 = 256 * 1024 * 1024;

/// A request's `messages` are the first `end` messages of segment `segment`.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct MessageRange {
    pub segment: u32,
    pub end: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CallKind {
    Turn,
    Compaction,
}

impl CallKind {
    pub fn tool_choice(self) -> &'static str {
        match self {
            CallKind::Turn => "auto",
            CallKind::Compaction => "none",
        }
    }
}

#[derive(Debug, Serialize)]
pub struct ToolMeta<'a> {
    pub name: &'a str,
    pub error: Option<ToolErrorKind>,
    pub task_id: Option<&'a str>,
    pub latency_ms: u64,
}

/// One line of `trace.jsonl`. Request `k` is `header.input.sampling_params` with `max_tokens`
/// replaced, then `model`, `messages` (the call's `prompt` range), `header.tools`,
/// `tool_choice` and `stream: false`, serialized in that order.
#[derive(Debug, Serialize)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum Event<'a> {
    Header {
        contract: u32,
        started_at_unix_ms: u128,
        input: &'a Value,
        tools: &'a [Value],
    },
    Message {
        segment: u32,
        index: usize,
        message: &'a Value,
        #[serde(skip_serializing_if = "Option::is_none")]
        tool: Option<ToolMeta<'a>>,
    },
    ModelCall {
        call: u32,
        turn: u32,
        kind: CallKind,
        prompt: MessageRange,
        tool_choice: &'a str,
        max_tokens: u64,
        attempts: u32,
        latency_ms: u64,
        response: &'a Value,
    },
    EndpointError {
        call: u32,
        turn: u32,
        attempt: u32,
        #[serde(flatten)]
        failure: &'a Failure,
        retry_in_ms: Option<u64>,
    },
    Compaction {
        from_segment: u32,
        to_segment: u32,
        summary_call: u32,
        kept_messages: usize,
    },
    Result(&'a RunResult),
}

#[derive(Serialize)]
struct Line<'a> {
    #[serde(flatten)]
    event: &'a Event<'a>,
    t_ms: u128,
}

/// Append-only, one synchronous write per line, so a cancelled future never leaves half a line.
/// A failed write or the size cap stops tracing and marks it truncated; the attempt goes on,
/// because the session server, not this file, is the token record.
pub struct Trace {
    file: Option<File>,
    written: u64,
    cap: u64,
    started: Instant,
}

impl Trace {
    pub fn create(session_dir: &Path, cap: u64) -> std::io::Result<Self> {
        let file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(session_dir.join("trace.jsonl"))?;
        Ok(Self {
            file: Some(file),
            written: 0,
            cap,
            started: Instant::now(),
        })
    }

    pub fn header(&mut self, input: &Value, tools: &[Value]) {
        let started_at_unix_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis();
        self.record(&Event::Header {
            contract: crate::outcome::CONTRACT_VERSION,
            started_at_unix_ms,
            input,
            tools,
        });
    }

    pub fn record(&mut self, event: &Event<'_>) {
        let Some(file) = &mut self.file else { return };
        let line = Line {
            event,
            t_ms: self.started.elapsed().as_millis(),
        };
        let written = serde_json::to_vec(&line).ok().and_then(|mut bytes| {
            bytes.push(b'\n');
            let total = self.written + bytes.len() as u64;
            (total <= self.cap && file.write_all(&bytes).is_ok()).then_some(total)
        });
        match written {
            Some(total) => self.written = total,
            None => self.file = None,
        }
    }

    pub fn truncated(&self) -> bool {
        self.file.is_none()
    }
}

/// One linear history as the session server sees it. Messages are only appended, each is traced
/// once when it enters, and compaction starts a new segment instead of rewriting this one.
pub struct Segment {
    index: u32,
    messages: Vec<Value>,
}

impl Segment {
    pub fn new(index: u32, messages: Vec<Value>, trace: &mut Trace) -> Self {
        let mut segment = Self {
            index,
            messages: Vec::with_capacity(messages.len()),
        };
        for message in messages {
            segment.push(message, None, trace);
        }
        segment
    }

    pub fn push(&mut self, message: Value, tool: Option<ToolMeta<'_>>, trace: &mut Trace) {
        trace.record(&Event::Message {
            segment: self.index,
            index: self.messages.len(),
            message: &message,
            tool,
        });
        self.messages.push(message);
    }

    pub fn index(&self) -> u32 {
        self.index
    }

    pub fn messages(&self) -> &[Value] {
        &self.messages
    }

    pub fn range(&self) -> MessageRange {
        MessageRange {
            segment: self.index,
            end: self.messages.len(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn the_size_cap_truncates_the_trace_without_failing() {
        let dir = tempfile::tempdir().unwrap();
        let mut trace = Trace::create(dir.path(), 300).unwrap();
        let mut segment = Segment::new(
            0,
            vec![json!({"role": "system", "content": "s"})],
            &mut trace,
        );
        assert!(!trace.truncated());
        for _ in 0..10 {
            segment.push(
                json!({"role": "user", "content": "x".repeat(40)}),
                None,
                &mut trace,
            );
        }
        assert!(trace.truncated());
        assert_eq!(segment.messages().len(), 11);
        let text = std::fs::read_to_string(dir.path().join("trace.jsonl")).unwrap();
        assert!(text.len() <= 300);
        assert!(text.ends_with('\n'));
        for line in text.lines() {
            serde_json::from_str::<Value>(line).unwrap();
        }
    }

    #[test]
    fn an_existing_trace_is_never_overwritten() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("trace.jsonl"), "evidence").unwrap();
        assert!(Trace::create(dir.path(), MAX_TRACE_BYTES).is_err());
        assert_eq!(
            std::fs::read_to_string(dir.path().join("trace.jsonl")).unwrap(),
            "evidence"
        );
    }
}
