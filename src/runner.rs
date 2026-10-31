use std::collections::HashSet;
use std::ffi::OsString;
use std::ops::ControlFlow;
use std::path::{Path, PathBuf};
use std::time::Duration;

use futures_util::StreamExt;
use futures_util::stream::FuturesUnordered;
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::time::Instant;

use crate::model::{Finish, ModelClient, Request, Sampled, SamplingParams, bearer_header};
use crate::outcome::{CONTRACT, Cause, Misbehaviour, Outcome, Overflow, RunResult, Tally};
use crate::tools::ToolSession;
use crate::trace::{CallKind, CompactionInput, Event, MAX_TRACE_BYTES, Segment, Trace};

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
    #[serde(default)]
    threshold_tokens: Option<u64>,
    max_tokens: u64,
}

pub fn run(stdin: &[u8], take_key: impl FnOnce(&str) -> Option<OsString>) -> RunResult {
    let started = std::time::Instant::now();
    let mut tally = Tally::default();
    let mut session_dir = None;
    let mut trace = None;
    let outcome = parse_input(stdin, &mut session_dir)
        .and_then(|(raw, input)| {
            let api_key = input
                .api_key_env
                .as_deref()
                .map(|name| api_key(take_key(name), name))
                .transpose()?;
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .map_err(|error| internal(format!("start runtime: {error}")))?;
            let outcome = runtime.block_on(attempt(&raw, &input, api_key, &mut trace, &mut tally));
            // A tool thread stuck in a blocking call, such as opening a FIFO, must not delay the exit.
            runtime.shutdown_background();
            outcome
        })
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
    raw: &Value,
    input: &Input,
    api_key: Option<String>,
    trace: &mut Option<Trace>,
    tally: &mut Tally,
) -> Result<Outcome, Cause> {
    let authorization = api_key
        .as_deref()
        .map(bearer_header)
        .transpose()
        .map_err(config)?;
    let url = reqwest::Url::parse(&format!(
        "{}/chat/completions",
        input.base_url.trim_end_matches('/')
    ))
    .map_err(|error| config(format!("invalid base_url: {error}")))?;
    let http = crate::model::http_client()?;
    let initial = initial_messages(&input.user_query, &input.workspace)?;
    let trace = trace.insert(
        Trace::create(&input.session_dir, MAX_TRACE_BYTES)
            .map_err(|error| internal(format!("create trace.jsonl: {error}")))?,
    );
    let tools = ToolSession::new(input.workspace.clone(), log_root(&input.workspace))
        .await
        .map_err(|error| Cause::ToolRuntime {
            tool: "session".into(),
            detail: format!("{error:#}"),
        })?;
    let definitions = tools.definitions();
    trace.header(raw, &definitions);
    let upstream = upstream_compaction(input, api_key, &initial, &definitions)?;
    let prefix = initial[1]["content"]
        .as_str()
        .and_then(|text| text.split("\n\n<user_query>").next())
        .unwrap_or_default()
        .to_owned();
    let deadline = Instant::now() + Duration::from_secs(input.deadline_secs);
    let mut cx = Attempt {
        input,
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
        upstream,
        prefix,
        tally,
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
    if outcome.gradable() {
        tally.background_tasks_left_running = tools.detach();
    } else {
        tools.kill_all().await;
    }
    Ok(outcome)
}

fn log_root(workspace: &Path) -> PathBuf {
    let real = |path: &Path| path.canonicalize().unwrap_or_else(|_| path.to_owned());
    let temp = std::env::temp_dir();
    let root = if real(&temp).starts_with(real(workspace)) {
        Path::new("/var/tmp")
    } else {
        &temp
    };
    root.join(".remanie-terminal")
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

fn api_key(key: Option<OsString>, name: &str) -> Result<String, Cause> {
    let key = key
        .ok_or_else(|| config(format!("environment variable {name} is not set")))?
        .into_string()
        .map_err(|_| config("API key must be UTF-8"))?;
    bearer_header(&key).map_err(config)?;
    Ok(key)
}

fn upstream_compaction(
    input: &Input,
    api_key: Option<String>,
    initial: &[Value],
    definitions: &[Value],
) -> Result<Option<grok_shell::glue::Compaction>, Cause> {
    if input.compaction.is_none() {
        return Ok(None);
    }
    let window = input
        .context_window_tokens
        .ok_or_else(|| config("compaction needs context_window_tokens"))?;
    grok_shell::glue::Compaction::new(
        grok_shell::glue::Endpoint {
            base_url: input.base_url.clone(),
            api_key,
            model: input.model.clone(),
            max_completion_tokens: u32::try_from(input.sampling_params.max_tokens()).ok(),
            temperature: input.sampling_params.number("temperature"),
            top_p: input.sampling_params.number("top_p"),
            context_window: window,
        },
        initial,
        definitions,
        input.session_dir.clone(),
    )
    .map(Some)
    .map_err(internal)
}

fn render(name: &str, source: &str, context: Value) -> Result<String, Cause> {
    let mut env = minijinja::Environment::new();
    env.set_trim_blocks(true);
    env.set_lstrip_blocks(true);
    let rendered = env
        .add_template(name, source)
        .and_then(|()| env.get_template(name)?.render(context))
        .map_err(|error| internal(format!("render {name}: {error}")))?;
    Ok(rendered)
}

fn initial_messages(query: &str, workspace: &Path) -> Result<Vec<Value>, Cause> {
    Ok(vec![
        json!({"role": "system", "content": render("system", include_str!("../assets/system_prompt.jinja2"), json!({}))?}),
        json!({"role": "user", "content": render("user", include_str!("../assets/user_query.jinja2"), user_info(workspace, query))?}),
    ])
}

pub(crate) fn user_info(workspace: &Path, query: &str) -> Value {
    json!({
        "os": std::env::consts::OS,
        "shell": std::env::var("SHELL").unwrap_or_else(|_| "/bin/bash".to_owned()),
        "workspace": workspace.display().to_string(),
        "today": today(),
        "user_query": query,
    })
}

fn today() -> String {
    let seconds = std::env::var("SOURCE_DATE_EPOCH")
        .ok()
        .and_then(|value| value.parse::<i64>().ok())
        .unwrap_or_else(|| {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |elapsed| elapsed.as_secs() as i64)
        });
    let z = seconds.div_euclid(86_400) + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + i64::from(month <= 2);
    format!("{year:04}-{month:02}-{day:02}")
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

#[derive(Default, Clone, Copy)]
struct ContextBudget {
    counted: u64,
    appended_bytes: u64,
}

const BYTES_PER_TOKEN: u64 = 4;

impl ContextBudget {
    fn counted(&mut self, tokens: u64) {
        *self = Self {
            counted: tokens,
            appended_bytes: 0,
        };
    }

    fn append(&mut self, message: &Value) {
        self.appended_bytes += serde_json::to_vec(message).map_or(0, |bytes| bytes.len() as u64);
    }

    fn estimate(&self) -> u64 {
        self.counted + self.appended_bytes / BYTES_PER_TOKEN
    }
}

struct Attempt<'a> {
    input: &'a Input,
    model: ModelClient<'a>,
    tools: &'a ToolSession,
    trace: &'a mut Trace,
    segment: Segment,
    budget: ContextBudget,
    upstream: Option<grok_shell::glue::Compaction>,
    prefix: String,
    tally: &'a mut Tally,
}

async fn execute(cx: &mut Attempt<'_>) -> Outcome {
    let outcome = loop {
        if let ControlFlow::Break(outcome) = turn(cx).await {
            break outcome;
        }
    };
    match outcome {
        Outcome::ContextOverflow(overflow) if cx.tally.turns == 0 => {
            Outcome::Infrastructure(config(format!(
                "the first request does not fit: {}",
                serde_json::to_value(&overflow).unwrap_or_default()
            )))
        }
        outcome => outcome,
    }
}

fn proceed<T>(result: Result<T, Outcome>) -> ControlFlow<Outcome, T> {
    result.map_or_else(ControlFlow::Break, ControlFlow::Continue)
}

async fn turn(cx: &mut Attempt<'_>) -> ControlFlow<Outcome> {
    let input = cx.input;
    if cx.tally.turns == input.max_turns {
        return ControlFlow::Break(Outcome::MaxTurns);
    }
    let max_tokens = input.sampling_params.max_tokens();
    if cx.upstream.is_some() {
        let inputs = cx.tools.compaction_inputs().await;
        let compacted = match cx.upstream.as_mut() {
            Some(upstream) => upstream.before_sample(&cx.prefix, inputs).await,
            None => Ok(None),
        };
        replace_segment(cx, compacted);
    } else if let Some(window) = input.context_window_tokens {
        let estimated = cx.budget.estimate();
        if estimated + max_tokens > window {
            return ControlFlow::Break(Outcome::ContextOverflow(Overflow::ContextWindow {
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
    } = proceed(
        cx.model
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
            .await,
    )?;
    cx.tally.turns += 1;
    cx.tally.count_response(usage);
    let reply = read_reply(&message);
    if let Some(upstream) = &cx.upstream {
        proceed(
            upstream
                .push_reply(&message, context_tokens)
                .map_err(|detail| Outcome::Infrastructure(internal(detail))),
        )?;
    }
    cx.segment.push(message, cx.trace);
    match context_tokens {
        Some(tokens) => cx.budget.counted(tokens),
        None if input.context_window_tokens.is_some() || input.compaction.is_some() => {
            return ControlFlow::Break(Outcome::Infrastructure(Cause::EndpointResponse {
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
            return ControlFlow::Break(Outcome::MaxTokens);
        }
        Finish::Other(finish_reason) => {
            return ControlFlow::Break(Outcome::InvalidResponse(Misbehaviour::UnexpectedFinish {
                finish_reason,
            }));
        }
        Finish::Stop | Finish::ToolCalls => proceed(reply.map_err(Outcome::InvalidResponse))?,
    };
    if reply.calls.is_empty() {
        if finish == Finish::ToolCalls {
            return ControlFlow::Break(Outcome::InvalidResponse(Misbehaviour::UnexpectedFinish {
                finish_reason: "tool_calls".into(),
            }));
        }
        let Some(answer) = [reply.content, reply.refusal]
            .into_iter()
            .flatten()
            .find(|text| !text.trim().is_empty())
        else {
            return ControlFlow::Break(Outcome::InvalidResponse(Misbehaviour::EmptyTurn));
        };
        cx.tally.final_answer = answer;
        return ControlFlow::Break(Outcome::Completed);
    }
    let tools = cx.tools;
    let locks = tools.path_locks(
        reply
            .calls
            .iter()
            .map(|call| (call.name.as_str(), call.arguments.as_str())),
    );
    let mut running: FuturesUnordered<_> = reply
        .calls
        .iter()
        .zip(locks)
        .enumerate()
        .map(|(index, (call, lock))| async move {
            let started = Instant::now();
            let _guard = match &lock {
                Some(lock) => Some(lock.lock().await),
                None => None,
            };
            let output = tools.call(&call.name, &call.arguments, &call.id).await;
            (index, output, started.elapsed())
        })
        .collect();
    let mut outputs: Vec<_> = reply.calls.iter().map(|_| None).collect();
    while let Some((index, output, latency)) = running.next().await {
        let output = proceed(output.map_err(|error| {
            Outcome::Infrastructure(Cause::ToolRuntime {
                tool: reply.calls[index].name.clone(),
                detail: error.0,
            })
        }))?;
        outputs[index] = Some((output, latency));
    }
    for (call, (output, latency)) in reply.calls.iter().zip(outputs.into_iter().flatten()) {
        cx.tally.tool_calls.count(output.error);
        if let Some(upstream) = &cx.upstream {
            upstream.push_tool_result(&call.id, &output.content);
        }
        let message = json!({"role": "tool", "tool_call_id": call.id, "content": output.content});
        cx.budget.append(&message);
        cx.segment.push(message, cx.trace);
        cx.trace.record(&Event::ToolCall {
            call_id: &call.id,
            name: &call.name,
            class: output.error,
            task_id: output.task_id,
            latency_ms: latency.as_millis() as u64,
        });
    }
    if cx.upstream.is_some() {
        let inputs = cx.tools.compaction_inputs().await;
        let compacted = match cx.upstream.as_mut() {
            Some(upstream) => upstream.after_tools(&cx.prefix, inputs).await,
            None => Ok(None),
        };
        replace_segment(cx, compacted);
    }
    ControlFlow::Continue(())
}

fn replace_segment(cx: &mut Attempt<'_>, compacted: Result<Option<Vec<Value>>, String>) {
    let Some(upstream) = cx.upstream.as_mut() else {
        return;
    };
    let mut last = None;
    for request in std::mem::take(&mut upstream.sent) {
        cx.trace.record(&Event::CompactionRequest {
            pass: request.pass,
            messages: &request.messages,
            summary: request.reply.as_deref().ok(),
            error: request.reply.as_ref().err().map(String::as_str),
        });
        last = Some(request.pass);
    }
    // Upstream logs a failed compaction and samples on uncompacted (turn.rs:2872-2879).
    let Ok(Some(history)) = compacted else {
        return;
    };
    let from = cx.segment.index();
    cx.trace.record(&Event::Compaction {
        from_segment: from,
        to_segment: from + 1,
        summary_call: cx.model.calls(),
        input: match last {
            Some("two_pass_2") => CompactionInput::TwoPass,
            Some("verbatim_fitted") => CompactionInput::Fitted,
            Some("lossy") => CompactionInput::Lossy,
            _ => CompactionInput::Verbatim,
        },
    });
    cx.segment = Segment::new(from + 1, history, cx.trace);
    cx.budget = ContextBudget::default();
    for message in cx.segment.messages() {
        cx.budget.append(message);
    }
    cx.tally.compactions += 1;
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
