use std::collections::HashSet;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::PathBuf;
use std::sync::Mutex;
use std::time::Duration;

use crate::compaction::format_compact_summary;
use anyhow::{Context, Result, bail, ensure};
use reqwest::header::{AUTHORIZATION, HeaderValue};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::tools::ToolSession;

const MAX_RESPONSE_BYTES: usize = 16 * 1024 * 1024;
const MAX_TRACE_BYTES: u64 = 256 * 1024 * 1024;
const SAMPLING_FIELDS: &[&str] = &[
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "presence_penalty",
    "frequency_penalty",
    "repetition_penalty",
    "seed",
    "stop",
    "chat_template_kwargs",
    "stop_token_ids",
    "skip_special_tokens",
    "no_stop_trim",
    "reasoning",
    "reasoning_effort",
];

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub base_url: String,
    #[serde(default)]
    pub api_key_env: Option<String>,
    pub model: String,
    pub user_query: String,
    pub workspace: PathBuf,
    pub session_dir: PathBuf,
    #[serde(default = "empty_object")]
    pub sampling_params: Value,
    #[serde(default = "default_turns")]
    pub max_turns: usize,
    #[serde(default = "default_response_tokens")]
    pub max_response_tokens: u32,
    #[serde(default = "default_deadline")]
    pub deadline_secs: u64,
    #[serde(default)]
    pub compaction_threshold_tokens: u64,
    #[serde(default = "default_compaction_tokens")]
    pub compaction_max_tokens: u32,
}

fn empty_object() -> Value {
    json!({})
}
fn default_turns() -> usize {
    100
}
fn default_response_tokens() -> u32 {
    4096
}
fn default_deadline() -> u64 {
    600
}
fn default_compaction_tokens() -> u32 {
    2048
}

impl Config {
    pub fn validate(&self) -> Result<()> {
        let url = reqwest::Url::parse(&self.base_url).context("invalid base_url")?;
        ensure!(
            matches!(url.scheme(), "http" | "https") && url.host_str().is_some(),
            "base_url must be HTTP(S)"
        );
        ensure!(
            url.username().is_empty()
                && url.password().is_none()
                && url.query().is_none()
                && url.fragment().is_none(),
            "base_url cannot contain credentials, query, or fragment"
        );
        ensure!(
            url.path().trim_end_matches('/').ends_with("/v1"),
            "base_url must end with /v1"
        );
        if let Some(name) = &self.api_key_env {
            ensure!(
                !name.is_empty()
                    && name.bytes().enumerate().all(|(i, byte)| {
                        byte == b'_'
                            || byte.is_ascii_alphabetic()
                            || (i > 0 && byte.is_ascii_digit())
                    }),
                "api_key_env must be an environment variable name"
            );
        }
        ensure!(
            !self.model.trim().is_empty() && !self.user_query.trim().is_empty(),
            "model and user_query are required"
        );
        ensure!(
            self.workspace.is_absolute() && self.workspace.is_dir(),
            "workspace must be an existing absolute directory"
        );
        ensure!(
            self.session_dir.is_absolute(),
            "session_dir must be absolute"
        );
        ensure!(
            self.max_turns > 0 && self.max_turns <= 512,
            "max_turns must be 1..512"
        );
        ensure!(
            (1..=86400).contains(&self.deadline_secs),
            "deadline_secs must be 1..86400"
        );
        ensure!(
            self.max_response_tokens > 0 && self.compaction_max_tokens > 0,
            "token limits must be positive"
        );
        let params = self
            .sampling_params
            .as_object()
            .context("sampling_params must be an object")?;
        for key in params.keys() {
            ensure!(
                SAMPLING_FIELDS.contains(&key.as_str()),
                "unsupported sampling field: {key}"
            );
        }
        Ok(())
    }
}

#[derive(Debug, Serialize)]
pub struct RunResult {
    pub status: String,
    pub final_answer: String,
    pub turns: usize,
    pub compactions: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

impl RunResult {
    fn stopped(status: &str) -> Self {
        Self {
            status: status.into(),
            final_answer: String::new(),
            turns: 0,
            compactions: 0,
            error: None,
        }
    }

    pub fn error(error: anyhow::Error) -> Self {
        Self {
            error: Some(format!("{error:#}")),
            ..Self::stopped("error")
        }
    }
}

struct Trace(Mutex<File>);

impl Trace {
    fn record(&self, event: &str, body: &Value) -> Result<()> {
        let mut file = self
            .0
            .lock()
            .map_err(|_| anyhow::anyhow!("trace lock poisoned"))?;
        let line = serde_json::to_vec(&json!({"event": event, "body": body}))?;
        ensure!(
            file.metadata()?.len().saturating_add(line.len() as u64 + 1) <= MAX_TRACE_BYTES,
            "trace exceeds 256 MiB"
        );
        file.write_all(&line)?;
        writeln!(file)?;
        file.flush()?;
        Ok(())
    }
}

struct ModelClient<'a> {
    config: &'a Config,
    authorization: Option<&'a HeaderValue>,
    http: reqwest::Client,
    definitions: Vec<Value>,
    trace: &'a Trace,
}

impl ModelClient<'_> {
    async fn sample(&self, messages: &[Value], max_tokens: u32, compacting: bool) -> Result<Value> {
        let mut request = self.config.sampling_params.clone();
        let fields = request.as_object_mut().context("invalid sampling_params")?;
        fields.extend(
            json!({
                "model": self.config.model,
                "messages": messages,
                "tools": self.definitions,
                "tool_choice": if compacting { "none" } else { "auto" },
                "max_tokens": max_tokens,
                "stream": false
            })
            .as_object()
            .unwrap()
            .clone(),
        );
        let payload = serde_json::to_vec(&request)?;
        ensure!(
            payload.len() <= MAX_RESPONSE_BYTES,
            "model request exceeds 16 MiB"
        );
        self.trace.record(
            if compacting {
                "compaction_request"
            } else {
                "model_request"
            },
            &request,
        )?;
        let url = format!(
            "{}/chat/completions",
            self.config.base_url.trim_end_matches('/')
        );
        let mut request = self
            .http
            .post(url)
            .header("Content-Type", "application/json")
            .body(payload);
        if let Some(authorization) = self.authorization {
            request = request.header(AUTHORIZATION, authorization);
        }
        let mut response = request.send().await.context("model request failed")?;
        ensure!(
            response.status().is_success(),
            "model endpoint returned HTTP {}",
            response.status()
        );
        let mut bytes = Vec::new();
        while let Some(chunk) = response.chunk().await? {
            ensure!(
                bytes.len().saturating_add(chunk.len()) <= MAX_RESPONSE_BYTES,
                "model response exceeds 16 MiB"
            );
            bytes.extend_from_slice(&chunk);
        }
        let body: Value =
            serde_json::from_slice(&bytes).context("model endpoint returned invalid JSON")?;
        self.trace.record(
            if compacting {
                "compaction_response"
            } else {
                "model_response"
            },
            &body,
        )?;
        ensure!(
            body["choices"].as_array().is_some_and(|v| v.len() == 1),
            "expected exactly one model choice"
        );
        Ok(body)
    }
    async fn compact(&self, turns: &[Value]) -> Result<Option<String>> {
        let mut messages = turns.to_vec();
        messages.push(json!({"role": "user", "content": render(
            "compaction", include_str!("../assets/compaction.jinja2"), json!({})
        )?}));
        let response = self
            .sample(&messages, self.config.compaction_max_tokens, true)
            .await?;
        let choice = &response["choices"][0];
        let message = &choice["message"];
        if choice["finish_reason"] != "stop" || has_tools(message) || !valid_message_shape(message)
        {
            return Ok(None);
        }
        let summary = format_compact_summary(message["content"].as_str().unwrap_or_default());
        Ok((!summary.trim().is_empty() && summary.trim() != "Summary:").then_some(summary))
    }
}

fn render(name: &str, source: &str, context: Value) -> Result<String> {
    let mut env = minijinja::Environment::new();
    env.set_trim_blocks(true);
    env.set_lstrip_blocks(true);
    env.add_template(name, source)?;
    Ok(env
        .get_template(name)?
        .render(context)?
        .trim_end_matches('\n')
        .to_owned())
}

fn initial_messages(query: &str) -> Result<Vec<Value>> {
    Ok(vec![
        json!({"role": "system", "content": render("system", include_str!("../assets/system_prompt.jinja2"), json!({}))?}),
        json!({"role": "user", "content": render("user", include_str!("../assets/user_query.jinja2"), json!({"user_query": query}))?}),
    ])
}

fn has_tools(message: &Value) -> bool {
    message["tool_calls"]
        .as_array()
        .is_some_and(|v| !v.is_empty())
}

fn valid_message_shape(message: &Value) -> bool {
    message["role"] == "assistant"
        && message.get("tool_calls").is_none_or(|calls| {
            calls.is_null() || calls.as_array().is_some_and(|calls| calls.len() <= 128)
        })
        && message
            .get("content")
            .is_none_or(|content| content.is_null() || content.is_string())
}

pub async fn run(config: Config, authorization: Option<HeaderValue>) -> Result<RunResult> {
    config.validate()?;
    std::fs::create_dir(&config.session_dir)
        .context("session_dir must be new and its parent must exist")?;
    let file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(config.session_dir.join("trace.jsonl"))?;
    let trace = Trace(Mutex::new(file));
    let session = ToolSession::new(config.workspace.clone(), config.session_dir.clone()).await?;
    let mut result = RunResult::stopped("max_turns");
    let outcome = tokio::select! {
        result = execute(&config, authorization.as_ref(), &session, &trace, &mut result) => result,
        _ = tokio::time::sleep(Duration::from_secs(config.deadline_secs)) => { result.status = "deadline".into(); Ok(()) },
        signal = interrupted() => { result.status = "cancelled".into(); signal },
    };
    session.shutdown().await;
    if let Err(error) = outcome {
        result.status = "error".into();
        result.error = Some(format!("{error:#}"));
    }
    trace.record("result", &serde_json::to_value(&result)?)?;
    Ok(result)
}

async fn interrupted() -> Result<()> {
    #[cfg(unix)]
    {
        let mut terminate =
            tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())?;
        tokio::select! { signal = tokio::signal::ctrl_c() => { signal?; }, _ = terminate.recv() => {} }
    }
    #[cfg(not(unix))]
    tokio::signal::ctrl_c().await?;
    Ok(())
}

async fn execute(
    config: &Config,
    authorization: Option<&HeaderValue>,
    session: &ToolSession,
    trace: &Trace,
    result: &mut RunResult,
) -> Result<()> {
    let client = ModelClient {
        config,
        authorization,
        http: reqwest::Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .connect_timeout(Duration::from_secs(30))
            .timeout(Duration::from_secs(config.deadline_secs))
            .build()?,
        definitions: session.definitions().await?,
        trace,
    };
    let original = initial_messages(&config.user_query)?;
    let mut messages = original.clone();
    let mut seen_ids = HashSet::new();
    let mut compact_before_next_turn = false;
    for _ in 0..config.max_turns {
        if compact_before_next_turn {
            session.definitions().await?;
            let Some(summary) = client.compact(&messages).await? else {
                result.status = "invalid_response".into();
                return Ok(());
            };
            let content = render(
                "continuation",
                include_str!("../assets/continuation.jinja2"),
                json!({
                    "summary": summary,
                    "tool_state": serde_json::to_string(&session.snapshot().await?)?
                }),
            )?;
            messages = original.clone();
            messages.push(json!({"role":"user", "content":content}));
            result.compactions += 1;
            trace.record("compacted_context", &json!(messages))?;
        }
        session.definitions().await?;
        let response = client
            .sample(&messages, config.max_response_tokens, false)
            .await?;
        result.turns += 1;
        let choice = &response["choices"][0];
        let message = &choice["message"];
        let finish = choice["finish_reason"].as_str().unwrap_or_default();
        if finish == "length" {
            result.status = "max_tokens".into();
            result.final_answer = message["content"].as_str().unwrap_or_default().into();
            return Ok(());
        }
        if !valid_message_shape(message)
            || !matches!(finish, "stop" | "tool_calls")
            || (finish == "tool_calls" && !has_tools(message))
        {
            result.status = "invalid_response".into();
            return Ok(());
        }
        // Provider reasoning fields and argument spelling must survive conversation replay.
        messages.push(message.clone());
        if !has_tools(message) {
            result.final_answer = message["content"]
                .as_str()
                .or_else(|| message["refusal"].as_str())
                .unwrap_or_default()
                .into();
            result.status = if result.final_answer.trim().is_empty() {
                "invalid_response"
            } else {
                "completed"
            }
            .into();
            return Ok(());
        }
        let calls = message["tool_calls"]
            .as_array()
            .context("invalid tool_calls")?;
        let mut new_tool_bytes = 0_u64;
        for call in calls {
            let valid = call["id"]
                .as_str()
                .filter(|id| !id.is_empty())
                .is_some_and(|id| seen_ids.insert(id.to_owned()))
                && call["type"] == "function"
                && call["function"]["name"].as_str().is_some()
                && call["function"]["arguments"].as_str().is_some();
            if !valid {
                result.status = "invalid_response".into();
                return Ok(());
            }
        }
        for call in calls {
            let id = call["id"].as_str().context("missing call id")?;
            let name = call["function"]["name"]
                .as_str()
                .context("missing tool name")?;
            let arguments = call["function"]["arguments"]
                .as_str()
                .context("missing arguments")?;
            let output = match serde_json::from_str(arguments) {
                Ok(arguments) => session.call(name, arguments, id).await,
                Err(error) => Err(anyhow::anyhow!("invalid tool argument JSON: {error}")),
            };
            let content = output.unwrap_or_else(|error| format!("Tool error: {error:#}"));
            let tool_message = json!({"role":"tool", "tool_call_id":id, "content":content});
            new_tool_bytes =
                new_tool_bytes.saturating_add(serde_json::to_vec(&tool_message)?.len() as u64);
            trace.record("tool_result", &tool_message)?;
            messages.push(tool_message);
        }
        // shortcut: new tool bytes estimate tokens until the endpoint supports prompt counting.
        let used = response["usage"]["total_tokens"].as_u64();
        if config.compaction_threshold_tokens > 0 && used.is_none() {
            bail!("compaction requires usage.total_tokens from the model endpoint");
        }
        compact_before_next_turn = config.compaction_threshold_tokens > 0
            && used.is_some_and(|tokens| {
                tokens
                    .saturating_add(new_tool_bytes)
                    .saturating_add(u64::from(
                        config.max_response_tokens.max(config.compaction_max_tokens),
                    ))
                    >= config.compaction_threshold_tokens
            });
    }
    Ok(())
}

#[cfg(test)]
mod tests;
