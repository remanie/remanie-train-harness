mod files;
mod list_dir;
mod terminal;

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{Context, Result, anyhow, ensure};
use serde_json::{Value, json};

use files::FileTools;
use terminal::TerminalSession;

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
    closed: AtomicBool,
    catalog: BTreeMap<String, Value>,
    validators: BTreeMap<String, jsonschema::Validator>,
}

impl ToolSession {
    pub async fn new(cwd: PathBuf, session_dir: PathBuf) -> Result<Self> {
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
            terminal: TerminalSession::new(cwd.clone(), session_dir).await?,
            files: FileTools::new(cwd),
            closed: AtomicBool::new(false),
            catalog,
            validators,
        })
    }

    pub async fn definitions(&self) -> Result<Vec<Value>> {
        Ok(TOOL_NAMES
            .into_iter()
            .map(|name| {
                let definition = &self.catalog[name];
                json!({"type": "function", "function": {
                    "name": name,
                    "description": definition["description"],
                    "parameters": definition["parameters"]
                }})
            })
            .collect())
    }

    pub async fn call(&self, name: &str, args: Value, call_id: &str) -> Result<String> {
        ensure!(
            !self.closed.load(Ordering::Acquire),
            "tool session has ended"
        );
        let validator = self.validators.get(name).context("tool is not permitted")?;
        validator
            .validate(&args)
            .map_err(|error| anyhow!("invalid {name} arguments: {error}"))?;
        match name {
            "run_terminal_cmd" | "get_terminal_command_output" | "kill_terminal_command" => {
                self.terminal.call(name, &args, call_id).await
            }
            _ => self.files.call(name, &args).await,
        }
    }

    pub async fn snapshot(&self) -> Result<Value> {
        Ok(json!({
            "todos": self.files.snapshot().await,
            "background_tasks": self.terminal.snapshot().await
        }))
    }

    pub async fn shutdown(&self) {
        self.closed.store(true, Ordering::Release);
        self.terminal.shutdown().await;
    }
}

#[cfg(test)]
mod tests;
