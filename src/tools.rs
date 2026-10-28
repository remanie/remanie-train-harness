mod files;
mod list_dir;
mod terminal;

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fmt;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Result, anyhow, ensure};
use serde::Serialize;
use serde_json::{Value, json};
use tokio::sync::Mutex;

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

const READ_ONLY: [&str; 4] = [
    "get_terminal_command_output",
    "read_file",
    "list_dir",
    "grep",
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

#[derive(Debug)]
struct InvalidArgument(String);

impl fmt::Display for InvalidArgument {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for InvalidArgument {}

fn invalid(message: impl fmt::Display) -> anyhow::Error {
    anyhow::Error::new(InvalidArgument(message.to_string()))
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
    pub async fn new(cwd: PathBuf, log_root: PathBuf) -> Result<Self> {
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
            terminal: TerminalSession::new(cwd.clone(), log_root).await?,
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
        if let Err(error) = whole_numbers(&mut args) {
            return Ok(ToolOutput::failed(ToolErrorKind::InvalidArguments, error));
        }
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
                    if error.is::<InvalidArgument>() {
                        ToolErrorKind::InvalidArguments
                    } else {
                        ToolErrorKind::Failed
                    },
                    format!("{error:#}"),
                )),
            },
        }
    }

    /// Grok Build's per-path locks for one reply's calls: a path a writing call targets gets a
    /// lock, and every call on that path takes it.
    pub fn path_locks<'c>(
        &self,
        calls: impl IntoIterator<Item = (&'c str, &'c str)>,
    ) -> Vec<Option<Arc<Mutex<()>>>> {
        let keyed: Vec<_> = calls
            .into_iter()
            .map(|(name, arguments)| {
                let path = serde_json::from_str::<Value>(arguments)
                    .ok()
                    .and_then(|args| {
                        ["file_path", "path", "target_file"]
                            .into_iter()
                            .find_map(|key| args[key].as_str().map(str::to_owned))
                    })
                    .and_then(|path| self.files.path(&path).ok())
                    .map(|path| files::lock_key(&path));
                (READ_ONLY.contains(&name), path)
            })
            .collect();
        let mut locks = HashMap::new();
        for (_, path) in keyed.iter().filter(|(read_only, _)| !read_only) {
            if let Some(path) = path {
                locks.entry(path.clone()).or_insert_with(Arc::default);
            }
        }
        keyed
            .into_iter()
            .map(|(_, path)| locks.get(&path?).cloned())
            .collect()
    }

    pub async fn state(&self) -> Value {
        let todos = self.files.snapshot().await;
        let count = |status: &str| todos.iter().filter(|todo| todo["status"] == status).count();
        let tasks = self.terminal.background_tasks();
        json!({
            "tasks": tasks.as_array().into_iter().flatten().filter(|task| task["completed"] == false).collect::<Vec<_>>(),
            "todos": todos.iter().filter(|todo| todo["status"] == "pending" || todo["status"] == "in_progress").collect::<Vec<_>>(),
            "completed": count("completed"),
            "cancelled": count("cancelled"),
        })
    }

    pub fn detach(self) -> Vec<LeftRunning> {
        self.terminal.detach()
    }

    pub async fn kill_all(self) {
        self.terminal.kill_all().await;
    }
}

// JSON Schema's "integer" accepts 1000.0 and -1, which as_u64 and serde's usize reject.
fn whole_numbers(args: &mut Value) -> Result<(), String> {
    for (key, value) in args.as_object_mut().into_iter().flatten() {
        match value.as_f64() {
            Some(number) if number < 0.0 => return Err(format!("{key} must not be negative")),
            Some(number) if value.is_f64() && number.fract() == 0.0 => {
                *value = json!(number as u64);
            }
            _ => {}
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests;
