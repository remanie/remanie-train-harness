mod files;
mod list_dir;
mod terminal;

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::path::PathBuf;

use anyhow::{Result, anyhow, ensure};
use serde::Serialize;
use serde_json::{Value, json};

use files::FileTools;
use terminal::TerminalSession;
pub use terminal::{LeftRunning, TaskId};

const TOOL_NAMES: [&str; 8] = [
    "run_terminal_cmd",
    "get_terminal_command_output",
    "kill_terminal_command",
    "read_file",
    "search_replace",
    "list_dir",
    "grep",
    "todo_write",
];

pub struct ToolSession {
    terminal: TerminalSession,
    files: FileTools,
    catalog: BTreeMap<String, Value>,
    validators: BTreeMap<String, jsonschema::Validator>,
}

#[derive(Debug)]
pub struct ToolOutput {
    pub content: String,
    pub error: Option<ToolErrorKind>,
    pub task_id: Option<TaskId>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolErrorKind {
    UnknownTool,
    InvalidArguments,
    Failed,
}

/// The runner's tool machinery broke or a program it needs is missing; nothing the model did.
#[derive(Debug)]
pub struct InternalError(pub String);

impl fmt::Display for InternalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for InternalError {}

fn internal(error: impl fmt::Display) -> anyhow::Error {
    anyhow::Error::new(InternalError(error.to_string()))
}

impl ToolOutput {
    fn failed(kind: ToolErrorKind, message: impl fmt::Display) -> Self {
        Self {
            content: format!("Tool error: {message}"),
            error: Some(kind),
            task_id: None,
        }
    }
}

impl ToolSession {
    pub async fn new(cwd: PathBuf, log_dir: PathBuf) -> Result<Self> {
        let catalog: BTreeMap<String, Value> =
            serde_json::from_str(include_str!("../assets/tools.json"))?;
        ensure!(
            catalog.keys().map(String::as_str).collect::<BTreeSet<_>>()
                == TOOL_NAMES.into_iter().collect(),
            "tool catalog must contain exactly the eight permitted tools"
        );
        let validators = catalog
            .iter()
            .map(|(name, definition)| {
                let validator = jsonschema::validator_for(&definition["parameters"])
                    .map_err(|error| anyhow!("invalid schema for {name}: {error}"))?;
                Ok((name.clone(), validator))
            })
            .collect::<Result<_>>()?;
        Ok(Self {
            terminal: TerminalSession::new(cwd.clone(), log_dir).await?,
            files: FileTools::new(
                cwd,
                std::env::var_os("HOME").map(PathBuf::from),
                std::env::var_os("RG_BIN_PATH").map_or_else(|| "rg".into(), PathBuf::from),
            ),
            catalog,
            validators,
        })
    }

    pub fn definitions(&self) -> Vec<Value> {
        TOOL_NAMES
            .into_iter()
            .map(|name| {
                let definition = &self.catalog[name];
                json!({"type": "function", "function": {
                    "name": name,
                    "description": definition["description"],
                    "parameters": definition["parameters"]
                }})
            })
            .collect()
    }

    pub async fn call(&self, name: &str, arguments: &str) -> Result<ToolOutput, InternalError> {
        let Some(validator) = self.validators.get(name) else {
            return Ok(ToolOutput::failed(
                ToolErrorKind::UnknownTool,
                "tool is not permitted",
            ));
        };
        let mut args: Value = match serde_json::from_str(arguments) {
            Ok(args) => args,
            Err(error) => {
                return Ok(ToolOutput::failed(
                    ToolErrorKind::InvalidArguments,
                    format!("invalid tool argument JSON: {error}"),
                ));
            }
        };
        if let Err(error) = validator.validate(&args) {
            return Ok(ToolOutput::failed(
                ToolErrorKind::InvalidArguments,
                format!("invalid {name} arguments: {error}"),
            ));
        }
        integral_floats_as_integers(&mut args);
        let mut task_id = None;
        let result = match name {
            "run_terminal_cmd" => self.terminal.run(&args).await.map(|(content, id)| {
                task_id = Some(id);
                content
            }),
            "get_terminal_command_output" => self.terminal.output(&args).await,
            "kill_terminal_command" => self.terminal.kill(&args).await,
            _ => self.files.call(name, &args).await,
        };
        match result {
            Ok(content) => Ok(ToolOutput {
                content,
                error: None,
                task_id,
            }),
            Err(error) => match error.downcast::<InternalError>() {
                Ok(InternalError(detail)) => Err(InternalError(format!("{name}: {detail}"))),
                Err(error) => Ok(ToolOutput::failed(
                    ToolErrorKind::Failed,
                    format!("{error:#}"),
                )),
            },
        }
    }

    pub async fn state(&self) -> Value {
        json!({
            "todos": self.files.snapshot().await,
            "background_tasks": self.terminal.background_tasks()
        })
    }

    pub fn detach(self) -> Vec<LeftRunning> {
        self.terminal.detach()
    }

    pub async fn kill_all(self) {
        self.terminal.kill_all().await;
    }
}

// JSON Schema's "integer" accepts 1000.0, which as_u64 and serde's usize reject.
fn integral_floats_as_integers(args: &mut Value) {
    for value in args
        .as_object_mut()
        .into_iter()
        .flat_map(|args| args.values_mut())
    {
        if value.is_f64()
            && let Some(number) = value.as_f64()
            && number.fract() == 0.0
            && number >= 0.0
        {
            *value = json!(number as u64);
        }
    }
}

#[cfg(test)]
mod tests;
