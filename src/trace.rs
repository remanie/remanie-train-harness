use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::Path;
use std::sync::Mutex;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use serde::Serialize;
use serde_json::Value;

use crate::outcome::RunResult;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Kind {
    Turn,
    Compaction,
    Title,
}

#[derive(Debug, Serialize)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum Event<'a> {
    Header {
        contract: u32,
        started_at_unix_ms: u128,
        config: &'a Value,
        agent_file: &'a str,
        grok_version: &'a str,
    },
    Request {
        call: u32,
        kind: Kind,
        body: &'a Value,
        injected: &'a [String],
    },
    Response {
        call: u32,
        status: Option<u16>,
        body: &'a Value,
        #[serde(skip_serializing_if = "Option::is_none")]
        raw_sse: Option<&'a str>,
        #[serde(skip_serializing_if = "Option::is_none")]
        error: Option<&'a str>,
    },
    Result(&'a RunResult),
}

#[derive(Serialize)]
struct Line<'a> {
    #[serde(flatten)]
    event: &'a Event<'a>,
    t_ms: u128,
}

/// Appends one JSON line per event. After a failed write it stops, so a trace without a closing
/// `result` line is known to be incomplete.
pub struct Trace {
    file: Mutex<Option<File>>,
    started: Instant,
}

impl Trace {
    pub fn create(session_dir: &Path, started: Instant) -> std::io::Result<Self> {
        let file = OpenOptions::new()
            .append(true)
            .create_new(true)
            .open(session_dir.join("trace.jsonl"))?;
        Ok(Self {
            file: Mutex::new(Some(file)),
            started,
        })
    }

    pub fn header(&self, config: &Value, agent_file: &str, grok_version: &str) {
        let started_at_unix_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis()
            - self.started.elapsed().as_millis();
        self.record(&Event::Header {
            contract: crate::outcome::CONTRACT_VERSION,
            started_at_unix_ms,
            config,
            agent_file,
            grok_version,
        });
    }

    pub fn record(&self, event: &Event<'_>) {
        let line = Line {
            event,
            t_ms: self.started.elapsed().as_millis(),
        };
        let Ok(mut bytes) = serde_json::to_vec(&line) else {
            return;
        };
        bytes.push(b'\n');
        let mut file = self
            .file
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(open) = file.as_mut()
            && open.write_all(&bytes).is_err()
        {
            *file = None;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_existing_trace_is_never_overwritten() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("trace.jsonl"), "evidence").unwrap();
        assert!(Trace::create(dir.path(), Instant::now()).is_err());
        assert_eq!(
            std::fs::read_to_string(dir.path().join("trace.jsonl")).unwrap(),
            "evidence"
        );
    }
}
