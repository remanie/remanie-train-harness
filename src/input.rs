use std::path::PathBuf;

use serde::Deserialize;
use serde_json::{Map, Value};

use crate::outcome::{CONTRACT, Cause};

pub const MAX_INPUT_BYTES: usize = 1024 * 1024;
const DEFAULT_THRESHOLD_PERCENT: u8 = 85;

#[derive(Debug, Deserialize)]
pub struct Input {
    pub base_url: String,
    pub model: String,
    #[serde(default)]
    pub api_key_env: Option<String>,
    pub user_query: String,
    pub workspace: PathBuf,
    pub session_dir: PathBuf,
    pub sampling_params: Map<String, Value>,
    pub max_turns: u32,
    pub deadline_secs: u64,
    pub context_window_tokens: u64,
    #[serde(default)]
    compaction: Option<Compaction>,
}

#[derive(Debug, Deserialize)]
struct Compaction {
    #[serde(default)]
    threshold_percent: Option<u8>,
}

impl Input {
    pub fn threshold_percent(&self) -> u8 {
        self.compaction
            .as_ref()
            .and_then(|compaction| compaction.threshold_percent)
            .unwrap_or(DEFAULT_THRESHOLD_PERCENT)
    }

    pub fn max_tokens(&self) -> u64 {
        self.sampling_params["max_tokens"].as_u64().unwrap_or(1)
    }

    pub fn temperature(&self) -> Option<f64> {
        self.sampling_params
            .get("temperature")
            .and_then(Value::as_f64)
    }
}

/// Creates `session_dir` as soon as the input names one, so even a rejected input leaves a result.json behind.
pub fn parse(stdin: &[u8], session_dir: &mut Option<PathBuf>) -> Result<(Value, Input), Cause> {
    if stdin.len() > MAX_INPUT_BYTES {
        return Err(config("input exceeds 1 MiB"));
    }
    let raw: Value = serde_json::from_slice(stdin)
        .map_err(|error| config(format!("input is not JSON: {error}")))?;
    if let Some(dir) = raw["session_dir"]
        .as_str()
        .map(PathBuf::from)
        .filter(|dir| dir.is_absolute())
    {
        std::fs::create_dir(&dir).map_err(|error| {
            config(format!(
                "session_dir must be new and its parent must exist: {error}"
            ))
        })?;
        *session_dir = Some(dir);
    }
    let contract: Value = serde_json::from_str(CONTRACT).map_err(|error| Cause::Internal {
        detail: format!("contract.json: {error}"),
    })?;
    let schema =
        jsonschema::validator_for(&contract["input"]).map_err(|error| Cause::Internal {
            detail: format!("contract.json input schema: {error}"),
        })?;
    if let Some(error) = schema.iter_errors(&raw).next() {
        return Err(config(format!("{error} at {}", error.instance_path)));
    }
    let input: Input =
        serde_json::from_value(raw.clone()).map_err(|error| config(error.to_string()))?;
    if !input.workspace.is_dir() {
        return Err(config("workspace must be an existing directory"));
    }
    Ok((raw, input))
}

pub fn config(detail: impl Into<String>) -> Cause {
    Cause::Config {
        detail: detail.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn job(dir: &std::path::Path) -> Value {
        json!({
            "base_url": "http://127.0.0.1:9/v1",
            "model": "m",
            "user_query": "q",
            "workspace": dir.to_str().unwrap(),
            "session_dir": dir.join("session").to_str().unwrap(),
            "sampling_params": {"max_tokens": 64, "provider": {"order": ["a"]}},
            "max_turns": 3,
            "deadline_secs": 60,
            "context_window_tokens": 32768,
        })
    }

    #[test]
    fn a_valid_input_parses_and_creates_the_session_dir() {
        let dir = tempfile::tempdir().unwrap();
        let mut session_dir = None;
        let (_, input) = parse(job(dir.path()).to_string().as_bytes(), &mut session_dir).unwrap();
        assert!(dir.path().join("session").is_dir());
        assert_eq!(input.threshold_percent(), 85);
        assert_eq!(input.max_tokens(), 64);
        assert_eq!(input.temperature(), None);
    }

    #[test]
    fn contract_violations_are_config_errors_with_a_session_dir() {
        let dir = tempfile::tempdir().unwrap();
        for (key, value) in [
            ("compaction", json!(null)),
            ("compaction", json!({"threshold_percent": 0})),
            ("compaction", json!({"max_tokens": 4096})),
            (
                "sampling_params",
                json!({"max_tokens": 64, "stream": false}),
            ),
            ("context_window_tokens", json!(null)),
            ("unknown", json!(1)),
        ] {
            let mut input = job(dir.path());
            input[key] = value;
            let session = dir.path().join("session");
            let _ = std::fs::remove_dir(&session);
            let mut session_dir = None;
            let error = parse(input.to_string().as_bytes(), &mut session_dir).unwrap_err();
            assert!(matches!(error, Cause::Config { .. }), "{key}: {error:?}");
            assert_eq!(session_dir.as_deref(), Some(session.as_path()), "{key}");
        }
    }

    #[test]
    fn an_existing_session_dir_is_never_reused() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir(dir.path().join("session")).unwrap();
        let mut session_dir = None;
        let error = parse(job(dir.path()).to_string().as_bytes(), &mut session_dir).unwrap_err();
        assert!(matches!(error, Cause::Config { .. }));
        assert_eq!(session_dir, None);
    }
}
