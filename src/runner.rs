use std::collections::HashSet;
use std::ffi::OsString;
use std::path::PathBuf;
use std::time::Duration;

use reqwest::header::HeaderValue;
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::time::Instant;

use crate::compaction::format_compact_summary;
use crate::model::{Finish, ModelClient, Request, Sampled, SamplingParams, bearer_header};
use crate::outcome::{CONTRACT, Cause, Misbehaviour, Outcome, Overflow, RunResult, Tally};
use crate::tools::ToolSession;
use crate::trace::{CallKind, Event, MAX_TRACE_BYTES, Segment, Trace};

pub const MAX_INPUT_BYTES: usize = 1024 * 1024;
const MAX_TOOL_CALLS: usize = 128;

#[derive(Debug, Deserialize)]
struct Input {
    base_url: String,
    #[serde(default)]
    api_key_env: Option<String>,
    model: String,
    user_query: String,
    workspace: PathBuf,
    session_dir: PathBuf,
    sampling_params: SamplingParams,
    max_turns: u32,
    deadline_secs: u64,
    #[serde(default)]
    context_window_tokens: Option<u64>,
    #[serde(default)]
    compaction: Option<Compaction>,
}

#[derive(Debug, Deserialize)]
struct Compaction {
    threshold_tokens: u64,
    max_tokens: u64,
}

pub async fn run(stdin: &[u8], take_key: impl FnOnce(&str) -> Option<OsString>) -> RunResult {
    let started = std::time::Instant::now();
    let mut tally = Tally::default();
    let mut session_dir = None;
    let mut trace = None;
    let outcome = attempt(stdin, take_key, &mut session_dir, &mut trace, &mut tally)
        .await
        .unwrap_or_else(Outcome::Infrastructure);
    let truncated = trace.as_ref().is_some_and(Trace::truncated);
    let mut result = RunResult::new(outcome, tally, truncated, started.elapsed());
    if let Some(dir) = &session_dir
        && let Err(error) = result.write_atomically(dir)
    {
        result.outcome = Outcome::Infrastructure(Cause::Internal {
            detail: format!("write result.json: {error}"),
        });
        result.gradable = false;
    }
    if let Some(trace) = &mut trace {
        trace.record(&Event::Result(&result));
    }
    result
}

async fn attempt(
    stdin: &[u8],
    take_key: impl FnOnce(&str) -> Option<OsString>,
    session_dir: &mut Option<PathBuf>,
    trace: &mut Option<Trace>,
    tally: &mut Tally,
) -> Result<Outcome, Cause> {
    let (raw, input) = parse_input(stdin, session_dir)?;
    let authorization = match &input.api_key_env {
        Some(name) => Some(authorization(take_key(name), name)?),
        None => None,
    };
    let url = reqwest::Url::parse(&format!(
        "{}/chat/completions",
        input.base_url.trim_end_matches('/')
    ))
    .map_err(|error| config(format!("invalid base_url: {error}")))?;
    let http = crate::model::http_client()?;
    let initial = initial_messages(&input.user_query)?;
    let trace = trace.insert(
        Trace::create(&input.session_dir, MAX_TRACE_BYTES)
            .map_err(|error| internal(format!("create trace.jsonl: {error}")))?,
    );
    let tools = ToolSession::new(input.workspace.clone(), input.session_dir.join("terminal"))
        .await
        .map_err(|error| Cause::ToolRuntime {
            tool: "session".into(),
            detail: format!("{error:#}"),
        })?;
    let definitions = tools.definitions();
    trace.header(&raw, &definitions);
    let deadline = Instant::now() + Duration::from_secs(input.deadline_secs);
    let mut cx = Attempt {
        input: &input,
        model: ModelClient::new(
            http,
            url,
            authorization,
            &input.model,
            &input.sampling_params,
            &definitions,
            deadline,
        ),
        tools: &tools,
        segment: Segment::new(0, initial.clone(), trace),
        trace,
        budget: ContextBudget::default(),
        initial,
        tally,
        compact_next: false,
    };
    for message in cx.segment.messages() {
        cx.budget.append(message);
    }
    let stopped = tokio::select! {
        outcome = execute(&mut cx) => Some(outcome),
        () = tokio::time::sleep_until(deadline) => None,
        () = interrupted() => Some(Outcome::Cancelled),
    };
    let outcome = stopped.unwrap_or_else(|| Outcome::Deadline {
        request_in_flight: cx.model.in_flight(),
    });
    if outcome.keeps_background_tasks() {
        tally.background_tasks_left_running = tools.detach();
    } else {
        tools.kill_all().await;
    }
    Ok(outcome)
}

fn config(detail: impl Into<String>) -> Cause {
    Cause::Config {
        detail: detail.into(),
    }
}

fn internal(detail: impl Into<String>) -> Cause {
    Cause::Internal {
        detail: detail.into(),
    }
}

fn parse_input(stdin: &[u8], session_dir: &mut Option<PathBuf>) -> Result<(Value, Input), Cause> {
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
    let contract: Value = serde_json::from_str(CONTRACT)
        .map_err(|error| internal(format!("contract.json: {error}")))?;
    let schema = jsonschema::validator_for(&contract["input"])
        .map_err(|error| internal(format!("contract.json input schema: {error}")))?;
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

fn authorization(key: Option<OsString>, name: &str) -> Result<HeaderValue, Cause> {
    let key = key
        .ok_or_else(|| config(format!("environment variable {name} is not set")))?
        .into_string()
        .map_err(|_| config("API key must be UTF-8"))?;
    bearer_header(&key).map_err(config)
}

fn render(name: &str, source: &str, context: Value) -> Result<String, Cause> {
    let mut env = minijinja::Environment::new();
    env.set_trim_blocks(true);
    env.set_lstrip_blocks(true);
    let rendered = env
        .add_template(name, source)
        .and_then(|()| env.get_template(name)?.render(context))
        .map_err(|error| internal(format!("render {name}: {error}")))?;
    Ok(rendered.trim_end_matches('\n').to_owned())
}

fn initial_messages(query: &str) -> Result<Vec<Value>, Cause> {
    Ok(vec![
        json!({"role": "system", "content": render("system", include_str!("../assets/system_prompt.jinja2"), json!({}))?}),
        json!({"role": "user", "content": render("user", include_str!("../assets/user_query.jinja2"), json!({"user_query": query}))?}),
    ])
}

async fn interrupted() {
    let terminate = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate());
    let Ok(mut terminate) = terminate else {
        return std::future::pending().await;
    };
    tokio::select! {
        Ok(()) = tokio::signal::ctrl_c() => {}
        Some(()) = terminate.recv() => {}
        else => std::future::pending().await,
    }
}

#[derive(Default)]
struct ContextBudget {
    counted: u64,
    appended_bytes: u64,
}

impl ContextBudget {
    fn counted(&mut self, tokens: u64) {
        *self = Self {
            counted: tokens,
            appended_bytes: 0,
        };
    }

    fn append(&mut self, message: &Value) {
        self.appended_bytes += bytes(message);
    }

    fn at_most(&mut self, tokens: u64) {
        if tokens < self.estimate() {
            self.counted(tokens);
        }
    }

    fn estimate(&self) -> u64 {
        self.counted + self.appended_bytes
    }
}

fn bytes(message: &Value) -> u64 {
    serde_json::to_vec(message).map_or(0, |bytes| bytes.len() as u64)
}

struct Attempt<'a> {
    input: &'a Input,
    model: ModelClient<'a>,
    tools: &'a ToolSession,
    trace: &'a mut Trace,
    segment: Segment,
    initial: Vec<Value>,
    budget: ContextBudget,
    tally: &'a mut Tally,
    compact_next: bool,
}

async fn execute(cx: &mut Attempt<'_>) -> Outcome {
    loop {
        if let Err(outcome) = turn(cx).await {
            return outcome;
        }
    }
}

async fn turn(cx: &mut Attempt<'_>) -> Result<(), Outcome> {
    let input = cx.input;
    if cx.tally.turns == input.max_turns {
        return Err(Outcome::MaxTurns);
    }
    if cx.compact_next {
        compact(cx).await?;
        cx.compact_next = false;
    }
    let max_tokens = input.sampling_params.max_tokens();
    if let Some(window) = input.context_window_tokens {
        let estimated = cx.budget.estimate();
        if estimated + max_tokens > window {
            return Err(Outcome::ContextOverflow(Overflow::ContextWindow {
                estimated_prompt_tokens: estimated,
                max_tokens,
                context_window_tokens: window,
            }));
        }
    }
    let Sampled {
        message,
        finish,
        usage,
        context_tokens,
    } = cx
        .model
        .sample(
            Request {
                kind: CallKind::Turn,
                turn: cx.tally.turns + 1,
                prompt: cx.segment.range(),
                messages: cx.segment.messages(),
                max_tokens_override: None,
            },
            cx.trace,
        )
        .await?;
    cx.tally.turns += 1;
    cx.tally.count_response(usage);
    let reply = read_reply(&message);
    cx.segment.push(message, cx.trace);
    match context_tokens {
        Some(tokens) => cx.budget.counted(tokens),
        None if input.context_window_tokens.is_some() || input.compaction.is_some() => {
            return Err(Outcome::Infrastructure(Cause::EndpointResponse {
                detail: "usage.total_tokens is required with compaction or context_window_tokens"
                    .into(),
            }));
        }
        None => {}
    }
    let reply = match finish {
        Finish::Length => {
            cx.tally.final_answer = reply
                .ok()
                .and_then(|reply| reply.content)
                .unwrap_or_default();
            return Err(Outcome::MaxTokens);
        }
        Finish::Other(finish_reason) => {
            return Err(Outcome::InvalidResponse(Misbehaviour::UnexpectedFinish {
                finish_reason,
            }));
        }
        Finish::Stop | Finish::ToolCalls => reply.map_err(Outcome::InvalidResponse)?,
    };
    if reply.calls.is_empty() {
        if finish == Finish::ToolCalls {
            return Err(Outcome::InvalidResponse(Misbehaviour::UnexpectedFinish {
                finish_reason: "tool_calls".into(),
            }));
        }
        cx.tally.final_answer = [reply.content, reply.refusal]
            .into_iter()
            .flatten()
            .find(|text| !text.trim().is_empty())
            .ok_or(Outcome::InvalidResponse(Misbehaviour::EmptyTurn))?;
        return Err(Outcome::Completed);
    }
    for call in &reply.calls {
        let started = Instant::now();
        let output = cx
            .tools
            .call(&call.name, &call.arguments)
            .await
            .map_err(|error| {
                Outcome::Infrastructure(Cause::ToolRuntime {
                    tool: call.name.clone(),
                    detail: error.0,
                })
            })?;
        cx.tally.tool_calls.count(output.error);
        let message = json!({"role": "tool", "tool_call_id": call.id, "content": output.content});
        cx.budget.append(&message);
        cx.segment.push(message, cx.trace);
        cx.trace.record(&Event::ToolCall {
            call_id: &call.id,
            name: &call.name,
            class: output.error,
            task_id: output.task_id,
            latency_ms: started.elapsed().as_millis() as u64,
        });
    }
    cx.compact_next = input.compaction.as_ref().is_some_and(|compaction| {
        cx.budget.estimate() + max_tokens.max(compaction.max_tokens) >= compaction.threshold_tokens
    });
    Ok(())
}

async fn compact(cx: &mut Attempt<'_>) -> Result<(), Outcome> {
    let Some(compaction) = &cx.input.compaction else {
        return Ok(());
    };
    let messages = cx.segment.messages();
    let kept = messages[kept_turns_start(messages)..].to_vec();
    let prompt = render(
        "compaction",
        include_str!("../assets/compaction.jinja2"),
        json!({}),
    )
    .map_err(Outcome::Infrastructure)?;
    cx.segment
        .push(json!({"role": "user", "content": prompt}), cx.trace);
    let sampled = cx
        .model
        .sample(
            Request {
                kind: CallKind::Compaction,
                turn: cx.tally.turns,
                prompt: cx.segment.range(),
                messages: cx.segment.messages(),
                max_tokens_override: Some(compaction.max_tokens),
            },
            cx.trace,
        )
        .await?;
    cx.tally.count_response(sampled.usage);
    let summary = summary(&sampled);
    cx.segment.push(sampled.message, cx.trace);
    let summary = summary.ok_or(Outcome::InvalidResponse(Misbehaviour::InvalidSummary))?;
    let state = cx.tools.state().await;
    let continuation = render(
        "continuation",
        include_str!("../assets/continuation.jinja2"),
        json!({"summary": summary, "tool_state": state.to_string()}),
    )
    .map_err(Outcome::Infrastructure)?;
    let from = cx.segment.index();
    cx.trace.record(&Event::Compaction {
        from_segment: from,
        to_segment: from + 1,
        summary_call: cx.model.calls(),
        kept_messages: kept.len(),
    });
    let continuation = json!({"role": "user", "content": continuation});
    let counted = sampled
        .context_tokens
        .map(|tokens| tokens + bytes(&continuation));
    let mut messages = cx.initial.clone();
    messages.push(continuation);
    messages.extend(kept);
    cx.segment = Segment::new(from + 1, messages, cx.trace);
    cx.budget = ContextBudget::default();
    for message in cx.segment.messages() {
        cx.budget.append(message);
    }
    if let Some(tokens) = counted {
        cx.budget.at_most(tokens);
    }
    cx.tally.compactions += 1;
    Ok(())
}

fn kept_turns_start(messages: &[Value]) -> usize {
    let mut assistants = messages
        .iter()
        .enumerate()
        .rev()
        .filter(|(_, message)| message["role"] == "assistant")
        .map(|(index, _)| index);
    let last = assistants.next();
    assistants.next().or(last).unwrap_or(messages.len())
}

fn summary(sampled: &Sampled) -> Option<String> {
    let reply = read_reply(&sampled.message).ok()?;
    if sampled.finish != Finish::Stop || !reply.calls.is_empty() {
        return None;
    }
    let summary = format_compact_summary(&reply.content?);
    (!summary.is_empty() && summary != "Summary:").then_some(summary)
}

struct Reply {
    content: Option<String>,
    refusal: Option<String>,
    calls: Vec<ToolCall>,
}

pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: String,
}

fn read_reply(message: &Value) -> Result<Reply, Misbehaviour> {
    let content = match &message["content"] {
        Value::Null => None,
        Value::String(text) => Some(text.clone()),
        _ => return Err(Misbehaviour::MalformedMessage),
    };
    let calls = match &message["tool_calls"] {
        Value::Null => &[][..],
        Value::Array(calls) if calls.len() <= MAX_TOOL_CALLS => calls,
        _ => return Err(Misbehaviour::MalformedMessage),
    };
    let mut ids = HashSet::new();
    let calls = calls
        .iter()
        .enumerate()
        .map(|(index, call)| {
            let text = |value: &Value| value.as_str().map(str::to_owned);
            let (Some(id), true, Some(name), Some(arguments)) = (
                text(&call["id"]).filter(|id| !id.is_empty()),
                call["type"] == "function",
                text(&call["function"]["name"]),
                text(&call["function"]["arguments"]),
            ) else {
                return Err(Misbehaviour::MalformedToolCall { index });
            };
            if !ids.insert(id.clone()) {
                return Err(Misbehaviour::DuplicateCallId { id });
            }
            Ok(ToolCall {
                id,
                name,
                arguments,
            })
        })
        .collect::<Result<_, _>>()?;
    Ok(Reply {
        content,
        refusal: message["refusal"].as_str().map(str::to_owned),
        calls,
    })
}

#[cfg(test)]
mod tests;
