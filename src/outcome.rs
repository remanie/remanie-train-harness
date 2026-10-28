use std::path::Path;
use std::time::Duration;

use serde::Serialize;

use crate::tools::ToolErrorKind;

pub const CONTRACT: &str = include_str!("../assets/contract.json");
pub const CONTRACT_VERSION: u32 = 1;
// `setsid --wait` reports a signal death as 1..=64 or 129..=192, so 70 always means the runner
// finished its own shutdown and reported infrastructure.
const EXIT_INFRASTRUCTURE: u8 = 70;

#[derive(Debug, Serialize)]
#[serde(tag = "outcome", content = "cause", rename_all = "snake_case")]
pub enum Outcome {
    Completed,
    MaxTurns,
    MaxTokens,
    ContextOverflow(Overflow),
    InvalidResponse(Misbehaviour),
    Deadline { request_in_flight: bool },
    Cancelled,
    Infrastructure(Cause),
}

#[derive(Debug, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Overflow {
    ContextWindow {
        estimated_prompt_tokens: u64,
        max_tokens: u64,
        context_window_tokens: u64,
    },
    RequestSize {
        bytes: usize,
    },
    EndpointRejected {
        status: u16,
        body_excerpt: String,
    },
}

#[derive(Debug, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Misbehaviour {
    EmptyTurn,
    MalformedMessage,
    MalformedToolCall { index: usize },
    DuplicateCallId { id: String },
    UnexpectedFinish { finish_reason: String },
    InvalidSummary,
}

#[derive(Debug, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Cause {
    Config {
        detail: String,
    },
    EndpointUnavailable {
        attempts: u32,
        status: Option<u16>,
        detail: String,
    },
    EndpointStatus {
        status: u16,
        body_excerpt: String,
    },
    ConnectionLost {
        detail: String,
    },
    EndpointResponse {
        detail: String,
    },
    ToolRuntime {
        tool: String,
        detail: String,
    },
    Internal {
        detail: String,
    },
}

impl Outcome {
    pub fn gradable(&self) -> bool {
        !matches!(self, Outcome::Infrastructure(_))
    }

    pub fn exit_code(&self) -> u8 {
        if self.gradable() {
            0
        } else {
            EXIT_INFRASTRUCTURE
        }
    }

    pub fn keeps_background_tasks(&self) -> bool {
        self.gradable()
    }
}

#[derive(Debug, Clone, Copy, Serialize)]
pub struct Usage {
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub cached_tokens: u64,
    pub complete: bool,
}

impl Default for Usage {
    fn default() -> Self {
        Self {
            prompt_tokens: 0,
            completion_tokens: 0,
            cached_tokens: 0,
            complete: true,
        }
    }
}

impl Usage {
    pub fn add(&mut self, other: Option<Usage>) {
        match other {
            Some(other) => {
                self.prompt_tokens += other.prompt_tokens;
                self.completion_tokens += other.completion_tokens;
                self.cached_tokens += other.cached_tokens;
            }
            None => self.complete = false,
        }
    }
}

#[derive(Debug, Default, Serialize)]
pub struct ToolCallCounts {
    pub total: u32,
    pub ok: u32,
    pub failed: u32,
    pub invalid_arguments: u32,
    pub unknown_tool: u32,
}

impl ToolCallCounts {
    pub fn count(&mut self, error: Option<ToolErrorKind>) {
        self.total += 1;
        *match error {
            None => &mut self.ok,
            Some(ToolErrorKind::Failed) => &mut self.failed,
            Some(ToolErrorKind::InvalidArguments) => &mut self.invalid_arguments,
            Some(ToolErrorKind::UnknownTool) => &mut self.unknown_tool,
        } += 1;
    }
}

#[derive(Debug, Default, Serialize)]
pub struct Tally {
    pub final_answer: String,
    pub turns: u32,
    pub compactions: u32,
    pub model_calls: u32,
    pub usage: Usage,
    pub tool_calls: ToolCallCounts,
    pub background_tasks_left_running: Vec<crate::tools::LeftRunning>,
}

impl Tally {
    pub fn count_response(&mut self, usage: Option<Usage>) {
        self.model_calls += 1;
        self.usage.add(usage);
    }
}

#[derive(Debug, Serialize)]
pub struct RunResult {
    pub contract: u32,
    #[serde(flatten)]
    pub outcome: Outcome,
    pub gradable: bool,
    #[serde(flatten)]
    pub tally: Tally,
    pub trace_truncated: bool,
    pub wall_time_ms: u64,
}

impl RunResult {
    pub fn new(outcome: Outcome, tally: Tally, trace_truncated: bool, wall_time: Duration) -> Self {
        Self {
            contract: CONTRACT_VERSION,
            gradable: outcome.gradable(),
            outcome,
            tally,
            trace_truncated,
            wall_time_ms: wall_time.as_millis() as u64,
        }
    }

    pub fn write_atomically(&self, session_dir: &Path) -> std::io::Result<()> {
        let temp = session_dir.join("result.json.tmp");
        std::fs::write(&temp, serde_json::to_vec(self)?)?;
        std::fs::rename(temp, session_dir.join("result.json"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{Value, json};
    use std::collections::BTreeSet;

    fn every_outcome() -> Vec<Outcome> {
        let text = || "x".to_string();
        let outcomes = vec![
            Outcome::Completed,
            Outcome::MaxTurns,
            Outcome::MaxTokens,
            Outcome::ContextOverflow(Overflow::ContextWindow {
                estimated_prompt_tokens: 1,
                max_tokens: 1,
                context_window_tokens: 1,
            }),
            Outcome::ContextOverflow(Overflow::RequestSize { bytes: 1 }),
            Outcome::ContextOverflow(Overflow::EndpointRejected {
                status: 400,
                body_excerpt: text(),
            }),
            Outcome::InvalidResponse(Misbehaviour::EmptyTurn),
            Outcome::InvalidResponse(Misbehaviour::MalformedMessage),
            Outcome::InvalidResponse(Misbehaviour::MalformedToolCall { index: 0 }),
            Outcome::InvalidResponse(Misbehaviour::DuplicateCallId { id: text() }),
            Outcome::InvalidResponse(Misbehaviour::UnexpectedFinish {
                finish_reason: text(),
            }),
            Outcome::InvalidResponse(Misbehaviour::InvalidSummary),
            Outcome::Deadline {
                request_in_flight: true,
            },
            Outcome::Cancelled,
            Outcome::Infrastructure(Cause::Config { detail: text() }),
            Outcome::Infrastructure(Cause::EndpointUnavailable {
                attempts: 1,
                status: None,
                detail: text(),
            }),
            Outcome::Infrastructure(Cause::EndpointStatus {
                status: 500,
                body_excerpt: text(),
            }),
            Outcome::Infrastructure(Cause::ConnectionLost { detail: text() }),
            Outcome::Infrastructure(Cause::EndpointResponse { detail: text() }),
            Outcome::Infrastructure(Cause::ToolRuntime {
                tool: text(),
                detail: text(),
            }),
            Outcome::Infrastructure(Cause::Internal { detail: text() }),
        ];
        for outcome in &outcomes {
            match outcome {
                Outcome::Completed
                | Outcome::MaxTurns
                | Outcome::MaxTokens
                | Outcome::Deadline { .. }
                | Outcome::Cancelled => {}
                Outcome::ContextOverflow(
                    Overflow::ContextWindow { .. }
                    | Overflow::RequestSize { .. }
                    | Overflow::EndpointRejected { .. },
                ) => {}
                Outcome::InvalidResponse(
                    Misbehaviour::EmptyTurn
                    | Misbehaviour::MalformedMessage
                    | Misbehaviour::MalformedToolCall { .. }
                    | Misbehaviour::DuplicateCallId { .. }
                    | Misbehaviour::UnexpectedFinish { .. }
                    | Misbehaviour::InvalidSummary,
                ) => {}
                Outcome::Infrastructure(
                    Cause::Config { .. }
                    | Cause::EndpointUnavailable { .. }
                    | Cause::EndpointStatus { .. }
                    | Cause::ConnectionLost { .. }
                    | Cause::EndpointResponse { .. }
                    | Cause::ToolRuntime { .. }
                    | Cause::Internal { .. },
                ) => {}
            }
        }
        outcomes
    }

    #[test]
    fn outcomes_exit_codes_and_results_match_the_contract() {
        let contract: Value = serde_json::from_str(CONTRACT).unwrap();
        assert_eq!(contract["version"], CONTRACT_VERSION);
        let result_schema = jsonschema::validator_for(&contract["result"]).unwrap();
        let mut produced = BTreeSet::new();
        for outcome in every_outcome() {
            let gradable = outcome.gradable();
            let code = outcome.exit_code();
            let result = serde_json::to_value(RunResult::new(
                outcome,
                Tally::default(),
                false,
                Duration::ZERO,
            ))
            .unwrap();
            if let Err(error) = result_schema.validate(&result) {
                panic!("{result} violates the result schema: {error}");
            }
            let name = result["outcome"].as_str().unwrap();
            let entry = &contract["outcomes"][name];
            assert_eq!(entry["gradable"], gradable, "{name}");
            let class = if gradable {
                "gradable"
            } else {
                "infrastructure"
            };
            assert_eq!(contract["exit_codes"][class], code, "{name}");
            let kind = result["cause"]["kind"].as_str().map(str::to_owned);
            produced.insert((name.to_owned(), kind));
        }
        let listed: BTreeSet<_> = contract["outcomes"]
            .as_object()
            .unwrap()
            .iter()
            .flat_map(|(name, entry)| {
                let causes = entry["causes"].as_array().unwrap();
                let kinds: Vec<_> = if causes.is_empty() {
                    vec![None]
                } else {
                    causes
                        .iter()
                        .map(|kind| kind.as_str().map(str::to_owned))
                        .collect()
                };
                kinds.into_iter().map(|kind| (name.clone(), kind))
            })
            .collect();
        assert_eq!(produced, listed);
    }

    #[test]
    fn gradable_outcomes_exit_zero_and_keep_background_tasks() {
        for outcome in every_outcome() {
            let infrastructure = matches!(outcome, Outcome::Infrastructure(_));
            assert_eq!(outcome.exit_code(), if infrastructure { 70 } else { 0 });
            assert_eq!(outcome.keeps_background_tasks(), !infrastructure);
        }
    }

    #[test]
    fn result_is_written_atomically_with_outcome_and_cause() {
        let dir = tempfile::tempdir().unwrap();
        let result = RunResult::new(
            Outcome::Infrastructure(Cause::EndpointStatus {
                status: 502,
                body_excerpt: "bad gateway".into(),
            }),
            Tally::default(),
            false,
            Duration::from_millis(5),
        );
        result.write_atomically(dir.path()).unwrap();
        let written: Value =
            serde_json::from_slice(&std::fs::read(dir.path().join("result.json")).unwrap())
                .unwrap();
        assert_eq!(written["outcome"], "infrastructure");
        assert_eq!(
            written["cause"],
            json!({"kind": "endpoint_status", "status": 502, "body_excerpt": "bad gateway"})
        );
        assert_eq!(written["gradable"], false);
        assert_eq!(written["usage"]["complete"], true);
        assert!(!dir.path().join("result.json.tmp").exists());
    }
}
