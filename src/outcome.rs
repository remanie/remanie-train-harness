use std::path::Path;
use std::time::Duration;

use serde::Serialize;

pub const CONTRACT: &str = include_str!("../assets/contract.json");
pub const CONTRACT_VERSION: u32 = 2;
// `setsid --wait` reports a signal death as 1..=64 or 129..=192, so 70 always means the wrapper
// finished its own shutdown and reported a failure that must not be graded.
const EXIT_INFRASTRUCTURE: u8 = 70;

#[derive(Debug, Serialize)]
#[serde(tag = "outcome", content = "cause", rename_all = "snake_case")]
pub enum Outcome {
    Completed,
    MaxTurns,
    Cancelled(Stop),
    Error(GrokFailure),
    EndpointUnavailable(Unavailable),
    Infrastructure(Cause),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Stop {
    Deadline,
    Signal { signal: &'static str },
}

#[derive(Debug, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum GrokFailure {
    Grok { message: String },
}

#[derive(Debug, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Unavailable {
    Unreachable {
        detail: String,
    },
    Refused {
        status: Option<u16>,
        message: String,
    },
}

#[derive(Debug, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Cause {
    Config { detail: String },
    GrokBinary { detail: String },
    GrokStartup { message: String },
    ProxyBind { detail: String },
    Internal { detail: String },
}

impl Outcome {
    pub fn gradable(&self) -> bool {
        !matches!(
            self,
            Outcome::EndpointUnavailable(_) | Outcome::Infrastructure(_)
        )
    }

    pub fn exit_code(&self) -> u8 {
        if self.gradable() {
            0
        } else {
            EXIT_INFRASTRUCTURE
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum GrokExit {
    Code(i32),
    Signal(i32),
}

#[derive(Debug, Default, Serialize)]
pub struct Tally {
    pub final_answer: String,
    pub turns: u32,
    pub compactions: u32,
    pub model_calls: u32,
    pub usage: Usage,
}

#[derive(Debug, Serialize)]
pub struct RunResult {
    pub contract: u32,
    #[serde(flatten)]
    pub outcome: Outcome,
    pub gradable: bool,
    #[serde(flatten)]
    pub tally: Tally,
    pub wall_time_ms: u64,
    pub grok_exit: Option<GrokExit>,
    pub grok_session_dir: Option<String>,
}

impl RunResult {
    pub fn new(
        outcome: Outcome,
        tally: Tally,
        grok_exit: Option<GrokExit>,
        grok_session_dir: Option<String>,
        wall_time: Duration,
    ) -> Self {
        Self {
            contract: CONTRACT_VERSION,
            gradable: outcome.gradable(),
            outcome,
            tally,
            wall_time_ms: wall_time.as_millis() as u64,
            grok_exit,
            grok_session_dir,
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
            Outcome::Cancelled(Stop::Deadline),
            Outcome::Cancelled(Stop::Signal { signal: "SIGTERM" }),
            Outcome::Error(GrokFailure::Grok { message: text() }),
            Outcome::EndpointUnavailable(Unavailable::Unreachable { detail: text() }),
            Outcome::EndpointUnavailable(Unavailable::Refused {
                status: Some(401),
                message: text(),
            }),
            Outcome::Infrastructure(Cause::Config { detail: text() }),
            Outcome::Infrastructure(Cause::GrokBinary { detail: text() }),
            Outcome::Infrastructure(Cause::GrokStartup { message: text() }),
            Outcome::Infrastructure(Cause::ProxyBind { detail: text() }),
            Outcome::Infrastructure(Cause::Internal { detail: text() }),
        ];
        for outcome in &outcomes {
            match outcome {
                Outcome::Completed
                | Outcome::MaxTurns
                | Outcome::Cancelled(Stop::Deadline | Stop::Signal { .. })
                | Outcome::Error(GrokFailure::Grok { .. })
                | Outcome::EndpointUnavailable(
                    Unavailable::Unreachable { .. } | Unavailable::Refused { .. },
                )
                | Outcome::Infrastructure(
                    Cause::Config { .. }
                    | Cause::GrokBinary { .. }
                    | Cause::GrokStartup { .. }
                    | Cause::ProxyBind { .. }
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
                Some(GrokExit::Code(0)),
                None,
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
    fn result_is_written_atomically_with_outcome_and_cause() {
        let dir = tempfile::tempdir().unwrap();
        let result = RunResult::new(
            Outcome::EndpointUnavailable(Unavailable::Unreachable {
                detail: "connection refused".into(),
            }),
            Tally::default(),
            Some(GrokExit::Signal(15)),
            None,
            Duration::from_millis(5),
        );
        result.write_atomically(dir.path()).unwrap();
        let written: Value =
            serde_json::from_slice(&std::fs::read(dir.path().join("result.json")).unwrap())
                .unwrap();
        assert_eq!(written["outcome"], "endpoint_unavailable");
        assert_eq!(
            written["cause"],
            json!({"kind": "unreachable", "detail": "connection refused"})
        );
        assert_eq!(written["gradable"], false);
        assert_eq!(written["grok_exit"], json!({"signal": 15}));
        assert_eq!(written["usage"]["complete"], true);
        assert!(!dir.path().join("result.json.tmp").exists());
    }
}
