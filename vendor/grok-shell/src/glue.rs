//! Drivers for upstream `SessionActor` methods, which no dependency reaches; each cites the range it follows and calls upstream for every step.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use agent_client_protocol as acp;
use serde_json::{Value, json};
use xai_chat_state::compaction_utils::{
    self as cu, CompactedHistoryInput, CompactionInputs, CompactionStateContext,
    build_compacted_history, is_degenerate_summary,
};
use xai_chat_state::{ChatStateActor, ChatStateHandle, CompactionMode, NullChatPersistence};
use xai_grok_compaction::{FullReplaceConfig, FullReplaceError, sample_full_replace_summary};
use xai_grok_sampler::{SamplerConfig, SamplingClient};
use xai_grok_sampling_types::{
    ApiBackend, AssistantItem, ConversationItem, SamplingConfig, ToolCall, ToolDefinition,
    ToolSpec, conversation_to_chat_messages, truncate_bytes,
};
use xai_grok_telemetry::events::CompactionTrigger;
use xai_grok_tools::bridge::ToolBridge;
use xai_grok_tools::registry::types::{SessionContext, ToolConfig, ToolServerConfig};

use crate::session::helpers::compaction_context::to_system_reminder_sync;
use crate::session::helpers::full_replace_compaction::{
    ShellCompactionSampler, ShellFullReplaceObserver,
};
use crate::session::helpers::session_compact::{
    CompactOutput, build_compaction_prompt, generate_session_compact,
};
use crate::session::helpers::tool_input_parsing::{
    normalize_empty_arguments, try_extract_concatenated_json_objects,
};
use crate::session::two_pass::{
    TWO_PASS_DEFAULT_SPLIT_FRACTION, build_two_pass_pass1_history, build_two_pass_pass2_history,
    note_for_two_pass_pass2, split_conversation_for_two_pass,
};
use crate::util::config::CompactionToolChoice;

// crates/codegen/xai-grok-shell/src/session/compaction.rs:58, 62, 991-992; mvp_agent/mod.rs:1114; util/config/resolve/compaction.rs:2, 93.
const SUMMARY_BUDGET_RESERVE_TOKENS: u64 = 32_768;
const DEFAULT_PREFIRE_LEAD_PERCENT: u64 = 10;
const THRESHOLD_PERCENT: u8 = 85;
const MAX_ATTEMPTS: u32 = 3;
const RETRY_DELAY_SECS: u64 = 3;
const IDLE_TIMEOUT: Duration = Duration::from_secs(600);
const WALL_CLOCK_BUDGET_SECS: u64 = 300;
const MAX_ARGS_IN_ERROR: usize = 2_000;

pub enum ToolResult {
    Ok(String),
    ParseError(String),
    Failed(String),
}

pub struct Tools {
    bridge: ToolBridge,
}

impl Tools {
    pub async fn new(cwd: PathBuf, session_folder: PathBuf, ids: &[&str]) -> Result<Self, String> {
        let config = ToolServerConfig {
            tools: ids.iter().map(|id| ToolConfig::from_id(*id)).collect(),
            behavior_preset: None,
        };
        let ctx = SessionContext {
            backend: Arc::new(xai_grok_tools::computer::local::LocalTerminalBackend::new()),
            fs: Arc::new(xai_grok_tools::computer::local::LocalFs),
            cwd,
            session_folder,
            session_env: Arc::new(HashMap::new()),
            notification_handle: xai_grok_tools::notification::ToolNotificationHandle::noop(),
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
        let mut builder = ToolBridge::get_builder();
        builder.set_system_reminders_enabled(false);
        let bridge = ToolBridge::finalize_builder(builder, config, ctx)
            .await
            .map_err(|error| error.to_string())?;
        Ok(Self { bridge })
    }

    /// Follows tool_calls.rs:1517-1604 for the arguments and tool_calls.rs:3058-3110, 3266-3276 for the text the model sees.
    pub async fn call(&self, name: &str, arguments: &str, call_id: &str) -> ToolResult {
        let args_str = normalize_empty_arguments(arguments);
        let raw_input = match serde_json::from_str::<Value>(args_str) {
            Ok(value) => value,
            Err(_) => match try_extract_concatenated_json_objects(arguments) {
                Some(objects) => objects
                    .first()
                    .cloned()
                    .unwrap_or_else(|| json!({ "raw": arguments })),
                None => json!({ "raw": arguments }),
            },
        };
        if let Err(error) = self.bridge.try_parse(name, raw_input.clone()).await {
            return ToolResult::ParseError(build_tool_parse_error_message(name, &error, arguments));
        }
        match self.bridge.call(name, raw_input, call_id).await {
            Ok(result) => ToolResult::Ok(
                xai_grok_tools::util::base64_images::extract_base64_images(result.prompt_text).text,
            ),
            Err(error) => ToolResult::Failed(format!("Tool `{name}` failed: {error}")),
        }
    }
}

// Copied from crates/codegen/xai-grok-shell/src/session/acp_session_impl/tool_dispatch.rs:365-395 (pub(super) there).
fn build_tool_parse_error_message(
    function_name: &str,
    err: &xai_tool_runtime::ToolError,
    raw_arguments: &str,
) -> String {
    let mut msg = format!("Failed to parse arguments for tool `{function_name}`: {err}");
    if raw_arguments.is_empty() {
        return msg;
    }
    msg.push_str("\n\nYour original arguments:\n");
    let prefix = truncate_bytes(raw_arguments, MAX_ARGS_IN_ERROR);
    msg.push_str(prefix);
    if prefix.len() < raw_arguments.len() {
        msg.push_str("\n... (truncated)");
    }
    if let Err(json_err) = serde_json::from_str::<serde::de::IgnoredAny>(raw_arguments) {
        msg.push_str(&format!(
            "\n\nNote: the arguments above contain invalid JSON — {json_err}\n\
             Please fix the syntax and retry."
        ));
    }
    msg
}

pub struct Endpoint {
    pub base_url: String,
    pub api_key: Option<String>,
    pub model: String,
    pub max_completion_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub context_window: u64,
}

pub struct Sent {
    pub pass: &'static str,
    pub messages: Vec<Value>,
    pub reply: Result<String, String>,
}

struct Note1 {
    note1: String,
    prefix_len: usize,
    fingerprint: u64,
}

pub struct Compaction {
    chat: ChatStateHandle,
    client: SamplingClient,
    sampling: SamplerConfig,
    tool_defs: Vec<ToolDefinition>,
    context_window: u64,
    session_id: acp::SessionId,
    session_dir: PathBuf,
    note1: Option<Note1>,
    count: u64,
    pub sent: Vec<Sent>,
}

impl Compaction {
    pub fn new(
        endpoint: Endpoint,
        initial: &[Value],
        tool_definitions: &[Value],
        session_dir: PathBuf,
    ) -> Result<Self, String> {
        let window = std::num::NonZeroU64::new(endpoint.context_window)
            .ok_or("context_window_tokens must be positive")?;
        let sampling = SamplerConfig {
            api_key: endpoint.api_key,
            base_url: endpoint.base_url.trim_end_matches('/').to_owned(),
            model: endpoint.model,
            max_completion_tokens: endpoint.max_completion_tokens,
            temperature: endpoint.temperature,
            top_p: endpoint.top_p,
            api_backend: ApiBackend::ChatCompletions,
            context_window: endpoint.context_window,
            ..Default::default()
        };
        let client = SamplingClient::new(sampling.clone()).map_err(|error| error.to_string())?;
        let state_config = SamplingConfig {
            base_url: sampling.base_url.clone(),
            model: sampling.model.clone(),
            max_completion_tokens: sampling.max_completion_tokens,
            temperature: sampling.temperature,
            top_p: sampling.top_p,
            api_backend: ApiBackend::ChatCompletions,
            context_window: window,
            ..Default::default()
        };
        let (events, _) = tokio::sync::mpsc::unbounded_channel();
        let chat = ChatStateActor::spawn(
            initial
                .iter()
                .map(|message| item(message, None))
                .collect::<Result<_, _>>()?,
            state_config,
            Box::new(NullChatPersistence),
            events,
            tokio_util::sync::CancellationToken::new(),
        );
        let tool_defs = tool_definitions
            .iter()
            .map(|definition| serde_json::from_value(definition.clone()))
            .collect::<Result<_, _>>()
            .map_err(|error| format!("tool definition: {error}"))?;
        Ok(Self {
            chat,
            client,
            sampling,
            tool_defs,
            context_window: endpoint.context_window,
            session_id: acp::SessionId::new(uuid::Uuid::new_v4().to_string()),
            session_dir,
            note1: None,
            count: 0,
            sent: Vec::new(),
        })
    }

    pub fn push_reply(&self, message: &Value, total_tokens: Option<u64>) -> Result<(), String> {
        let reply = item(message, Some(&self.sampling.model))?;
        match total_tokens {
            Some(total) => {
                self.chat.push_model_output(reply);
                self.chat.record_token_usage(total);
            }
            None => self.chat.push_unreported_model_output(reply),
        }
        Ok(())
    }

    pub fn push_tool_result(&self, call_id: &str, content: &str) {
        self.chat
            .push_tool_result(ConversationItem::tool_result(call_id, content));
    }

    pub async fn estimated_total(&self) -> u64 {
        self.chat.get_estimated_total_tokens().await
    }

    /// The checks before each sample in turn.rs:2853-2879; returns the replacement history when compaction ran.
    pub async fn before_sample(
        &mut self,
        prefix: &str,
        inputs: CompactionInputs,
    ) -> Result<Option<Vec<Value>>, String> {
        let estimated = self.estimated_total().await;
        let lead = std::env::var("GROK_PREFIRE_LEAD_PERCENT")
            .ok()
            .and_then(|value| value.trim().parse::<u64>().ok())
            .unwrap_or(DEFAULT_PREFIRE_LEAD_PERCENT);
        let start = (THRESHOLD_PERCENT as u64).saturating_sub(lead) as u8;
        if self.note1.is_none()
            && xai_token_estimation::exceeds_threshold(estimated, self.context_window, start)
        {
            self.prefire_pass1().await;
        }
        if xai_token_estimation::exceeds_threshold(estimated, self.context_window, THRESHOLD_PERCENT)
        {
            return self.compact(prefix, inputs).await.map(Some);
        }
        Ok(None)
    }

    /// The preflight check after a turn's tools in turn.rs:3865-3875 and compaction.rs:2208-2237.
    pub async fn after_tools(
        &mut self,
        prefix: &str,
        inputs: CompactionInputs,
    ) -> Result<Option<Vec<Value>>, String> {
        if self.estimated_total().await <= self.context_window {
            return Ok(None);
        }
        self.compact(prefix, inputs).await.map(Some)
    }

    fn tools(&self) -> Vec<ToolSpec> {
        self.tool_defs.iter().cloned().map(ToolSpec::from).collect()
    }

    async fn record(&mut self, pass: &'static str, items: Vec<ConversationItem>, reply: Result<String, String>) {
        let messages = conversation_to_chat_messages(items)
            .into_iter()
            .map(|message| serde_json::to_value(message).unwrap_or_default())
            .collect();
        self.sent.push(Sent { pass, messages, reply });
    }

    /// compaction.rs:138-178.
    async fn two_pass_sample(&mut self, pass: &'static str, history: Vec<ConversationItem>) -> Option<CompactOutput> {
        let tools = self.tools();
        let tool_tokens = xai_chat_state::estimate_tool_specs_tokens(&tools);
        let cancel = tokio_util::sync::CancellationToken::new();
        let result = generate_session_compact(
            history.clone(),
            tool_tokens,
            tools,
            vec![],
            self.client.clone(),
            self.session_id.clone(),
            &self.sampling,
            IDLE_TIMEOUT,
            WALL_CLOCK_BUDGET_SECS,
            CompactionToolChoice::Auto,
            &cancel,
        )
        .await;
        let reply = match &result {
            Ok(out) => Ok(out.content.clone()),
            Err(failure) => Err(format!("{failure:?}")),
        };
        self.record(pass, history, reply).await;
        result.ok()
    }

    /// compaction.rs:232-312, run inline where upstream spawns it beside the turn.
    async fn prefire_pass1(&mut self) {
        let conversation = self.chat.get_conversation().await;
        if conversation.len() < 4 {
            return;
        }
        let split = split_conversation_for_two_pass(&conversation, TWO_PASS_DEFAULT_SPLIT_FRACTION);
        if split.prefix.is_empty() || split.tail.is_empty() {
            return;
        }
        let prefix_prepared = self
            .budget_and_prune(cu::prepare_conversation_for_verbatim_summarization(
                split.prefix.to_vec(),
                false,
            ))
            .await;
        let prompt = build_compaction_prompt(None, false);
        let pass1_history = build_two_pass_pass1_history(&prefix_prepared, &prompt);
        let split_idx = split.split_idx;
        let Some(out) = self.two_pass_sample("two_pass_1", pass1_history).await else {
            return;
        };
        let note1 = note_for_two_pass_pass2(&out.content);
        if note1.trim().is_empty() {
            return;
        }
        let Some(prefix) = conversation.get(..split_idx) else {
            return;
        };
        self.note1 = Some(Note1 {
            note1,
            prefix_len: split_idx,
            fingerprint: fingerprint_prefix(prefix),
        });
    }

    /// compaction.rs:313-397.
    async fn two_pass_pass2(&mut self) -> Option<CompactOutput> {
        let cache = self.note1.take()?;
        let live = self.chat.get_conversation().await;
        let (prefix, tail) = live.split_at_checked(cache.prefix_len)?;
        if cache.prefix_len == 0 || fingerprint_prefix(prefix) != cache.fingerprint {
            return None;
        }
        let prepared_tail = cu::prepare_conversation_for_verbatim_summarization(tail.to_vec(), false);
        let prompt = build_compaction_prompt(None, false);
        let pass2_history = build_two_pass_pass2_history(prefix, &prepared_tail, &cache.note1, &prompt);
        let out = self.two_pass_sample("two_pass_2", pass2_history).await?;
        if is_degenerate_summary(&out.content) {
            return None;
        }
        Some(out)
    }

    /// compaction.rs:424-434.
    async fn budget_and_prune(&self, items: Vec<ConversationItem>) -> Vec<ConversationItem> {
        let items =
            xai_chat_state::image_budget::apply_image_budget(items, self.sampling.max_request_bytes)
                .items;
        self.chat.apply_turn_request_pruning(items).await
    }

    /// The single-pass ladder and assembly of compaction.rs:922-2094 with the hooks, memory, goals, plan mode, MCP and forks left out.
    async fn compact(&mut self, prefix: &str, inputs: CompactionInputs) -> Result<Vec<Value>, String> {
        let system_message = self
            .chat
            .get_system_message()
            .await
            .ok_or("Compaction failed: no system message in conversation history")?;
        let full_conversation = self.chat.get_conversation().await;
        let mode = CompactionMode::default();
        let segment_messages = if mode.writes_segments() {
            cu::prepare_conversation_for_segment(full_conversation.clone())
        } else {
            Vec::new()
        };
        let simplified = self
            .budget_and_prune(cu::prepare_conversation_for_verbatim_summarization(full_conversation, false))
            .await;
        if !simplified.iter().any(|item| matches!(item, ConversationItem::System(_))) {
            return Err("Compaction failed: no system message in simplified conversation".into());
        }
        let tool_tokens = xai_chat_state::estimate_tool_definitions_tokens(&self.tool_defs);
        let tools = self.tools();
        let window = self.context_window;
        let (mut request_turns, mut stage) = start_verbatim_compact_turns(simplified, tool_tokens, window);
        let cancel = tokio_util::sync::CancellationToken::new();
        let sampler = ShellCompactionSampler::new(
            false,
            None,
            tools,
            vec![],
            tool_tokens,
            self.client.clone(),
            self.session_id.clone(),
            self.sampling.clone(),
            IDLE_TIMEOUT,
            WALL_CLOCK_BUDGET_SECS,
            CompactionToolChoice::Auto,
            cancel,
        );
        let observer = ShellFullReplaceObserver::new(
            CompactionTrigger::Auto,
            window,
            uuid::Uuid::new_v4().to_string(),
            self.session_id.to_string(),
            xai_chat_state::estimate_conversation_tokens(&request_turns),
            RETRY_DELAY_SECS,
        );
        let config = FullReplaceConfig {
            max_attempts: MAX_ATTEMPTS,
            retry_delay_secs: RETRY_DELAY_SECS,
            sampling_timeout_secs: 0,
        };
        let two_pass = self.two_pass_pass2().await;
        let summary = match two_pass {
            Some(out) => out.content,
            None => loop {
                let sampled = sample_full_replace_summary(&sampler, &request_turns, None, &config, &observer).await;
                let attempted = sampler.take_last_attempted_items().unwrap_or_default();
                match sampled {
                    Ok(summary) => {
                        self.record(stage.label(), attempted, Ok(summary.summary.clone())).await;
                        break summary.summary;
                    }
                    Err(FullReplaceError::Sampler { message, context_overflow: true, .. }) => {
                        self.record(stage.label(), attempted, Err(message.clone())).await;
                        let Some(next) = stage.next() else {
                            return Err(message);
                        };
                        let conversation = self.chat.get_conversation().await;
                        request_turns = match next {
                            Stage::VerbatimFitted => cu::fit_conversation_to_budget(
                                self.budget_and_prune(cu::prepare_conversation_for_verbatim_summarization(conversation, false)).await,
                                fitted_input_budget(window, tool_tokens),
                            ),
                            Stage::Lossy | Stage::Verbatim => cu::fit_conversation_to_budget(
                                cu::prepare_conversation_for_summarization(conversation),
                                lossy_input_budget(window, tool_tokens),
                            ),
                        };
                        stage = next;
                    }
                    Err(error) => {
                        self.record(stage.label(), attempted, Err(error.to_string())).await;
                        return Err(error.to_string());
                    }
                }
            },
        };
        let conversation = self.chat.get_conversation().await;
        let state_context = CompactionStateContext::build(&conversation, inputs).await;
        let system_reminder = to_system_reminder_sync(&state_context, &[], &[], None, None, None);
        let compaction_dir = self.session_dir.join(xai_compaction_transcript::COMPACTION_DIR);
        let transcript_hint = mode.transcript_hint(Some(&compaction_dir.to_string_lossy()));
        let compacted = cu::sanitize_compacted_history(build_compacted_history(CompactedHistoryInput {
            system_message,
            user_message_prefix: prefix.to_owned(),
            agents_md_reminder: None,
            state_context: &state_context.for_compaction(),
            compaction_summary: summary.clone(),
            system_reminder,
            summary_before_recent: false,
            transcript_hint,
            summary_count: self.count,
        }))
        .items;
        if let Some(detail) = mode.segment_detail() {
            write_compaction_segment(&compaction_dir, &segment_messages, &cu::format_compact_summary(&summary), detail)
                .await
                .map_err(|error| format!("write compaction segment: {error}"))?;
        }
        self.count += 1;
        self.chat.replace_conversation_for_compaction(compacted.clone());
        Ok(conversation_to_chat_messages(compacted)
            .into_iter()
            .map(|message| serde_json::to_value(message).unwrap_or_default())
            .collect())
    }
}

#[derive(Clone, Copy)]
enum Stage {
    Verbatim,
    VerbatimFitted,
    Lossy,
}

impl Stage {
    fn next(self) -> Option<Self> {
        match self {
            Self::Verbatim => Some(Self::VerbatimFitted),
            Self::VerbatimFitted => Some(Self::Lossy),
            Self::Lossy => None,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Verbatim => "verbatim",
            Self::VerbatimFitted => "verbatim_fitted",
            Self::Lossy => "lossy",
        }
    }
}

// compaction.rs:406-457.
fn lossy_input_budget(context_window: u64, tool_tokens: u64) -> u64 {
    (context_window.saturating_mul(7) / 10).saturating_sub(tool_tokens)
}

fn fitted_input_budget(context_window: u64, tool_tokens: u64) -> u64 {
    context_window
        .saturating_sub(SUMMARY_BUDGET_RESERVE_TOKENS)
        .saturating_sub(tool_tokens)
}

fn start_verbatim_compact_turns(
    turns: Vec<ConversationItem>,
    tool_tokens: u64,
    context_window: u64,
) -> (Vec<ConversationItem>, Stage) {
    let estimate = xai_chat_state::estimate_conversation_tokens(&turns);
    let budget = fitted_input_budget(context_window, tool_tokens);
    if estimate > budget {
        (cu::fit_conversation_to_budget(turns, budget), Stage::VerbatimFitted)
    } else {
        (turns, Stage::Verbatim)
    }
}

// compaction.rs:72-92.
fn fingerprint_prefix(items: &[ConversationItem]) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    items.len().hash(&mut h);
    for it in items {
        let tag: u8 = match it {
            ConversationItem::System(_) => 0,
            ConversationItem::User(_) => 1,
            ConversationItem::Assistant(_) => 2,
            ConversationItem::ToolResult(_) => 3,
            ConversationItem::BackendToolCall(_) => 4,
            ConversationItem::Reasoning(_) => 5,
        };
        tag.hash(&mut h);
        it.text_content().hash(&mut h);
    }
    h.finish()
}

// storage/jsonl/mod.rs:1167-1182, 1943-1979.
async fn write_compaction_segment(
    base: &std::path::Path,
    items: &[ConversationItem],
    summary: &str,
    detail: xai_compaction_transcript::CompactionDetail,
) -> std::io::Result<()> {
    use tokio::io::AsyncWriteExt;
    use xai_compaction_transcript::{
        INDEX_FILE, INDEX_HEADER, extract_keywords, render_index_row, render_segment_md,
        segment_filename,
    };
    tokio::fs::create_dir_all(base).await?;
    let mut index = 0u64;
    if let Ok(mut entries) = tokio::fs::read_dir(base).await {
        while let Ok(Some(entry)) = entries.next_entry().await {
            if let Some(n) = entry
                .file_name()
                .to_str()
                .and_then(xai_compaction_transcript::parse_segment_index)
            {
                index = index.max(n + 1);
            }
        }
    }
    let timestamp = chrono::Utc::now().format("%Y-%m-%dT%H:%M:%SZ").to_string();
    let md = render_segment_md(items, summary, index, detail, &timestamp);
    tokio::fs::write(base.join(segment_filename(index)), md.as_bytes()).await?;
    let index_path = base.join(INDEX_FILE);
    let needs_header = !tokio::fs::try_exists(&index_path).await.unwrap_or(false);
    let mut f = tokio::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&index_path)
        .await?;
    if needs_header {
        f.write_all(INDEX_HEADER.as_bytes()).await?;
    }
    let keywords = extract_keywords(summary);
    let row = render_index_row(index, items.len(), md.len(), &keywords);
    f.write_all(row.as_bytes()).await?;
    f.flush().await
}

/// The runner's chat-completions message as the item upstream would hold; the runner keeps JSON messages until the loop is rewritten.
fn item(message: &Value, model_id: Option<&str>) -> Result<ConversationItem, String> {
    let text = message["content"].as_str().unwrap_or_default();
    Ok(match message["role"].as_str() {
        Some("system") => ConversationItem::system(text),
        Some("user") => ConversationItem::user(text),
        Some("tool") => ConversationItem::tool_result(
            message["tool_call_id"].as_str().unwrap_or_default(),
            text,
        ),
        Some("assistant") => ConversationItem::Assistant(AssistantItem {
            content: text.into(),
            tool_calls: message["tool_calls"]
                .as_array()
                .into_iter()
                .flatten()
                .map(|call| ToolCall {
                    id: call["id"].as_str().unwrap_or_default().into(),
                    name: call["function"]["name"].as_str().unwrap_or_default().to_owned(),
                    arguments: call["function"]["arguments"].as_str().unwrap_or_default().into(),
                })
                .collect(),
            model_id: model_id.map(str::to_owned),
            model_fingerprint: None,
            reasoning_effort: None,
        }),
        role => return Err(format!("message role {role:?} has no conversation item")),
    })
}
