use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result, anyhow, bail, ensure};
use serde_json::{Value, json};
use xai_grok_tools::computer::local::{LocalFs, LocalTerminalBackend, SearchShadowConfig};
use xai_grok_tools::computer::types::TerminalBackend;
use xai_grok_tools::implementations::grok_build::{
    GetTerminalCommandOutputTool, GrepTool, KillTerminalCommandTool, ListDirTool, TodoWriteTool,
    todo::TodoState,
};
use xai_grok_tools::implementations::grok_build_concise::{
    BashConciseTool, ReadFileConciseTool, SearchReplaceConciseTool,
};
use xai_grok_tools::notification::ToolNotificationHandle;
use xai_grok_tools::registry::types::{
    FinalizedToolset, SessionContext, ToolConfig, ToolRegistryBuilder, ToolServerConfig,
};
use xai_grok_tools::types::context::{TruncationConfig, WholeReadPolicy};
use xai_grok_tools::types::output::{ReadFileOutput, ToolOutput};
use xai_grok_tools::types::resources::State;
use xai_grok_tools::util::{ProcessScope, ShellEnvironmentPolicy, ShellEnvironmentPolicyInherit};

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
    registry: Arc<FinalizedToolset>,
    terminal: Arc<LocalTerminalBackend>,
    scope: ProcessScope,
    catalog: BTreeMap<String, Value>,
    validators: BTreeMap<String, jsonschema::Validator>,
}

impl ToolSession {
    // Requires a LocalSet and a disposable OS environment; native file tools are not confined to cwd.
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

        let home = session_dir.join("home");
        let tmp = session_dir.join("tmp");
        tokio::fs::create_dir(&home)
            .await
            .context("create attempt home")?;
        tokio::fs::create_dir(&tmp)
            .await
            .context("create attempt tmp")?;
        let policy = ShellEnvironmentPolicy {
            inherit: ShellEnvironmentPolicyInherit::None,
            set: HashMap::from([
                (
                    "PATH".into(),
                    "/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin".into(),
                ),
                ("HOME".into(), home.to_string_lossy().into_owned()),
                ("TMPDIR".into(), tmp.to_string_lossy().into_owned()),
                ("LANG".into(), "C".into()),
            ]),
            ..Default::default()
        };
        let scope = ProcessScope::new();
        let terminal = Arc::new(LocalTerminalBackend::new_local_with_login_shell_capture(
            SearchShadowConfig::default(),
            false,
            Some(policy),
            Some(scope.clone()),
        ));
        let mut builder = ToolRegistryBuilder::new();
        builder.set_system_reminders_enabled(false);
        let mut configs = vec![
            ToolConfig::from(&BashConciseTool)
                .with_param("enabled_background", true)
                .with_param("auto_background_on_timeout", true)
                .with_param("foreground_block_budget_ms", 0)
                .with_param("default_block_until_ms", 30_000)
                .with_param("max_block_until_ms", 300_000)
                .with_param("max_timeout_secs", 300)
                .with_param("allow_background_operator", false)
                .with_param("surface_bg_completion_reminders", false),
            ToolConfig::from(&GetTerminalCommandOutputTool),
            ToolConfig::from(&KillTerminalCommandTool),
            ToolConfig::from(&ReadFileConciseTool).with_param("cursor_rules_on_read", false),
            ToolConfig::from(&SearchReplaceConciseTool)
                .with_param("empty_old_string_does_not_override", true)
                .with_param("unicode_normalized_fallback", false)
                .with_param("include_user_edit_hint", false),
            ToolConfig::from(&ListDirTool),
            ToolConfig::from(&GrepTool),
            ToolConfig::from(&TodoWriteTool),
        ];
        for (config, name) in configs.iter_mut().zip(TOOL_NAMES) {
            config.description_override = Some(
                catalog[name]["description"]
                    .as_str()
                    .context("tool description must be a string")?
                    .to_owned(),
            );
        }
        let context = SessionContext {
            backend: terminal.clone(),
            fs: Arc::new(LocalFs),
            cwd,
            session_folder: session_dir,
            session_env: Arc::new(HashMap::new()),
            notification_handle: ToolNotificationHandle::noop(),
            owner_session_id: None,
            subagent: None,
            parent_scheduler_handle: None,
            skills: vec![],
            state_path: PathBuf::new(),
            memory_backend: None,
            web_search_config: Default::default(),
            web_fetch_config: Default::default(),
            lsp: None,
            image_gen_config: Default::default(),
            video_gen_config: Default::default(),
            app_builder_deployer_config: Default::default(),
            api_key_provider: None,
            auth_provider: None,
            attribution_callback: None,
            system_reminder_tag: xai_grok_tools::reminders::DEFAULT_REMINDER_TAG,
        };
        let registry = builder
            .finalize_with_trunc_config(
                ToolServerConfig {
                    tools: configs,
                    behavior_preset: Some("current".into()),
                },
                context,
                TruncationConfig {
                    whole_read: WholeReadPolicy {
                        skill_markdown: false,
                        instruction_files: false,
                    },
                    ..Default::default()
                },
                None,
            )
            .map_err(|errors| anyhow!("native tool configuration failed: {errors:?}"))?;
        let session = Self {
            registry: Arc::new(registry),
            terminal,
            scope,
            catalog,
            validators,
        };
        session.definitions().await?;
        Ok(session)
    }

    pub async fn definitions(&self) -> Result<Vec<Value>> {
        let native = self.registry.tool_definitions();
        ensure!(
            native.len() == TOOL_NAMES.len()
                && native
                    .iter()
                    .map(|tool| tool.function.name.as_str())
                    .collect::<BTreeSet<_>>()
                    == TOOL_NAMES.into_iter().collect(),
            "native registry no longer matches the eight-tool allowlist"
        );
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
        ensure!(!self.scope.is_closed(), "tool session has ended");
        let validator = self.validators.get(name).context("tool is not permitted")?;
        validator
            .validate(&args)
            .map_err(|error| anyhow!("invalid {name} arguments: {error}"))?;
        let result = self.registry.call(name, args, call_id, None).await?;
        match &result.output {
            ToolOutput::GrepSearch(output) if !matches!(output.exit_code, 0 | 1) => {
                let error = if output.stderr.is_empty() {
                    &output.stdout
                } else {
                    &output.stderr
                };
                bail!(
                    "grep failed (exit {}): {}",
                    output.exit_code,
                    String::from_utf8_lossy(error)
                )
            }
            ToolOutput::ReadFile(
                ReadFileOutput::ImageContent(_) | ReadFileOutput::PdfPageImages(_),
            ) => {
                bail!("non-text tool output is unsupported; use format=text for PDFs")
            }
            ToolOutput::ReadFile(ReadFileOutput::FileContent(content))
                if !content.extracted_images.is_empty() =>
            {
                bail!("non-text tool output is unsupported")
            }
            _ => Ok(result.prompt_text),
        }
    }

    pub async fn snapshot(&self) -> Result<Value> {
        let todos = {
            let resources = self.registry.resources.lock().await;
            let empty = TodoState::default();
            serde_json::to_value(
                resources
                    .get::<State<TodoState>>()
                    .map(|state| &state.0)
                    .unwrap_or(&empty),
            )?
        };
        Ok(json!({"todos": todos, "background_tasks": self.terminal.list_tasks_metadata().await}))
    }

    pub async fn shutdown(&self) {
        self.scope.kill_all();
        self.terminal.kill_foreground_commands().await;
        self.terminal.kill_all_background_tasks().await;
        self.terminal.cancel();
    }
}

impl Drop for ToolSession {
    fn drop(&mut self) {
        self.scope.kill_all();
        self.terminal.cancel();
    }
}

#[cfg(test)]
mod tests;
