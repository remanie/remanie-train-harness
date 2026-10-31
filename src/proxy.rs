use std::convert::Infallible;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use bytes::Bytes;
use futures_util::StreamExt;
use http_body_util::combinators::BoxBody;
use http_body_util::{BodyExt, Full, StreamBody};
use hyper::body::{Frame, Incoming};
use hyper::header::{self, HeaderMap, HeaderName, HeaderValue};
use hyper::{Method, Request, Response, StatusCode};
use hyper_util::rt::TokioIo;
use serde_json::{Map, Value, json};
use tokio::net::TcpListener;
use tokio::sync::{mpsc, oneshot, watch};
use tokio::task::{JoinHandle, JoinSet};

use crate::outcome::Usage;
use crate::trace::{Event, Kind, Trace};

/// The opening of Grok Build's summary prompt, which ends every compaction request.
pub const SUMMARY_PROMPT: &str =
    "Your task is to produce a faithful, concise summary of the conversation so far";
const TITLE: &str = "Remanie trial";
const HOP_BY_HOP: [&str; 9] = [
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "proxy-connection",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
];

type Body = BoxBody<Bytes, std::io::Error>;

#[derive(Debug, Default, Clone)]
pub struct Record {
    pub turns: u32,
    pub compactions: u32,
    pub model_calls: u32,
    pub usage: Usage,
    pub answered: bool,
    pub last_status: Option<u16>,
}

pub struct Upstream {
    pub chat_completions: String,
    pub authorization: Option<HeaderValue>,
    pub sampling_params: Map<String, Value>,
    pub model: String,
    pub context_window: u64,
}

pub struct Proxy {
    listener: TcpListener,
    pub base_url: String,
}

pub struct Running {
    shared: Arc<Shared>,
    accept: JoinHandle<()>,
}

struct Shared {
    upstream: Upstream,
    client: reqwest::Client,
    trace: Arc<Trace>,
    calls: AtomicU32,
    record: Mutex<Record>,
    unreachable: watch::Sender<Option<String>>,
    tasks: Mutex<JoinSet<()>>,
}

impl Proxy {
    pub async fn bind() -> std::io::Result<Self> {
        let listener = TcpListener::bind("127.0.0.1:0").await?;
        let base_url = format!("http://{}/v1", listener.local_addr()?);
        Ok(Self { listener, base_url })
    }

    pub fn serve(self, upstream: Upstream, trace: Arc<Trace>) -> reqwest::Result<Running> {
        let client = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(30))
            .build()?;
        let shared = Arc::new(Shared {
            upstream,
            client,
            trace,
            calls: AtomicU32::new(0),
            record: Mutex::new(Record::default()),
            unreachable: watch::Sender::new(None),
            tasks: Mutex::new(JoinSet::new()),
        });
        let accept = tokio::spawn(accept(self.listener, shared.clone()));
        Ok(Running { shared, accept })
    }
}

impl Running {
    /// Resolves with the failure once the endpoint gave no HTTP response to the first model request.
    pub fn unreachable(&self) -> watch::Receiver<Option<String>> {
        self.shared.unreachable.subscribe()
    }

    pub async fn finish(self) -> Record {
        self.accept.abort();
        let mut tasks = std::mem::take(&mut *lock(&self.shared.tasks));
        let drained = tokio::time::timeout(Duration::from_secs(10), async {
            while tasks.join_next().await.is_some() {}
        });
        if drained.await.is_err() {
            tasks.abort_all();
        }
        lock(&self.shared.record).clone()
    }
}

fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

async fn accept(listener: TcpListener, shared: Arc<Shared>) {
    loop {
        let Ok((stream, _)) = listener.accept().await else {
            continue;
        };
        let shared = shared.clone();
        tokio::spawn(async move {
            let service =
                hyper::service::service_fn(move |request| handle(shared.clone(), request));
            let _ = hyper::server::conn::http1::Builder::new()
                .serve_connection(TokioIo::new(stream), service)
                .await;
        });
    }
}

async fn handle(
    shared: Arc<Shared>,
    request: Request<Incoming>,
) -> Result<Response<Body>, Infallible> {
    if request.method() == Method::GET && request.uri().path() == "/v1/models" {
        return Ok(shared.catalog());
    }
    if request.method() != Method::POST || request.uri().path() != "/v1/chat/completions" {
        return Ok(text(
            StatusCode::NOT_FOUND,
            "not served by the trial runner",
        ));
    }
    let (parts, incoming) = request.into_parts();
    let Ok(collected) = incoming.collect().await else {
        return Ok(text(StatusCode::BAD_REQUEST, "unreadable request body"));
    };
    let raw = collected.to_bytes();
    let Ok(Value::Object(body)) = serde_json::from_slice::<Value>(&raw) else {
        return Ok(text(
            StatusCode::BAD_REQUEST,
            "request body is not a JSON object",
        ));
    };
    let call = shared.calls.fetch_add(1, Ordering::Relaxed) + 1;
    let kind = classify(&body);
    if kind == Kind::Title {
        return Ok(shared.title(call, body));
    }
    let (forwarded, injected) = inject(&raw, &body, &shared.upstream.sampling_params);
    let sent = serde_json::from_slice(&forwarded).unwrap_or(Value::Object(body));
    shared.trace.record(&Event::Request {
        call,
        kind,
        body: &sent,
        injected: &injected,
    });
    let (head_tx, head_rx) = oneshot::channel();
    let (chunk_tx, chunk_rx) = mpsc::channel(64);
    lock(&shared.tasks).spawn(forward(
        shared.clone(),
        call,
        kind,
        upstream_headers(&parts.headers, shared.upstream.authorization.as_ref()),
        forwarded,
        head_tx,
        chunk_tx,
    ));
    let Ok((status, headers)) = head_rx.await else {
        return Ok(text(
            StatusCode::BAD_GATEWAY,
            "the trial runner's proxy failed",
        ));
    };
    let stream = futures_util::stream::unfold(chunk_rx, |mut chunks| async move {
        let chunk: Result<Bytes, std::io::Error> = chunks.recv().await?;
        Some((chunk.map(Frame::data), chunks))
    });
    let mut response = Response::new(BodyExt::boxed(StreamBody::new(stream)));
    *response.status_mut() = status;
    *response.headers_mut() = headers;
    Ok(response)
}

type Head = (StatusCode, HeaderMap);

async fn forward(
    shared: Arc<Shared>,
    call: u32,
    kind: Kind,
    headers: HeaderMap,
    body: Vec<u8>,
    mut head_tx: oneshot::Sender<Head>,
    chunk_tx: mpsc::Sender<Result<Bytes, std::io::Error>>,
) {
    let first = {
        let mut record = lock(&shared.record);
        record.model_calls += 1;
        record.model_calls == 1
    };
    let request = shared
        .client
        .post(&shared.upstream.chat_completions)
        .headers(headers)
        .body(body)
        .send();
    let response = tokio::select! {
        response = request => response,
        () = head_tx.closed() => {
            shared.respond(call, kind, Answer::failed("grok closed the connection before the endpoint answered"));
            return;
        }
    };
    let response = match response {
        Ok(response) => response,
        Err(error) => {
            let detail = transport_failure(&error);
            if first {
                shared.unreachable.send_replace(Some(detail.clone()));
            }
            shared.respond(call, kind, Answer::failed(&detail));
            let _ = head_tx.send((StatusCode::BAD_GATEWAY, HeaderMap::new()));
            let _ = chunk_tx.send(Ok(Bytes::from(detail))).await;
            return;
        }
    };
    let status = response.status();
    let sse = response
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .is_some_and(|value| value.starts_with("text/event-stream"));
    if head_tx
        .send((status, downstream_headers(response.headers())))
        .is_err()
    {
        shared.respond(
            call,
            kind,
            Answer::failed("grok closed the connection before the endpoint answered"),
        );
        return;
    }
    let mut raw = Vec::new();
    let mut error = None;
    let mut stream = response.bytes_stream();
    while let Some(chunk) = stream.next().await {
        match chunk {
            Ok(bytes) => {
                raw.extend_from_slice(&bytes);
                if chunk_tx.send(Ok(bytes)).await.is_err() {
                    error = Some("grok closed the connection during the response".to_owned());
                    break;
                }
            }
            Err(failure) => {
                let detail = transport_failure(&failure);
                let _ = chunk_tx
                    .send(Err(std::io::Error::other(detail.clone())))
                    .await;
                error = Some(detail);
                break;
            }
        }
    }
    shared.respond(call, kind, Answer::received(status, sse, &raw, error));
}

struct Answer {
    status: Option<u16>,
    body: Value,
    raw_sse: Option<String>,
    error: Option<String>,
}

impl Answer {
    fn failed(detail: &str) -> Self {
        Self {
            status: None,
            body: Value::Null,
            raw_sse: None,
            error: Some(detail.to_owned()),
        }
    }

    fn received(status: StatusCode, sse: bool, raw: &[u8], error: Option<String>) -> Self {
        let text = String::from_utf8_lossy(raw).into_owned();
        let body = if !status.is_success() {
            Value::String(text.clone())
        } else if sse {
            assemble(&text)
        } else {
            serde_json::from_slice(raw).unwrap_or_else(|_| Value::String(text.clone()))
        };
        Self {
            status: Some(status.as_u16()),
            body,
            raw_sse: (status.is_success() && sse).then_some(text),
            error,
        }
    }
}

impl Shared {
    fn respond(&self, call: u32, kind: Kind, answer: Answer) {
        self.trace.record(&Event::Response {
            call,
            status: answer.status,
            body: &answer.body,
            raw_sse: answer.raw_sse.as_deref(),
            error: answer.error.as_deref(),
        });
        let mut record = lock(&self.record);
        if answer.status.is_some() {
            record.last_status = answer.status;
        }
        if !answer
            .status
            .is_some_and(|status| (200..300).contains(&status))
        {
            return;
        }
        record.answered = true;
        if answer.error.is_some() {
            record.usage.add(None);
            return;
        }
        match kind {
            Kind::Turn => record.turns += 1,
            Kind::Compaction => record.compactions += 1,
            Kind::Title => {}
        }
        record.usage.add(usage(&answer.body));
    }

    /// The model list an OpenAI-compatible server gives for the one model it serves, so Grok's
    /// catalog holds the configured model whatever the endpoint itself lists.
    fn catalog(&self) -> Response<Body> {
        let list = json!({
            "object": "list",
            "data": [{
                "id": self.upstream.model,
                "object": "model",
                "created": 0,
                "owned_by": "remanie-train-harness",
                "context_length": self.upstream.context_window,
            }],
        });
        let mut response = Response::new(
            Full::new(Bytes::from(list.to_string()))
                .map_err(|never| match never {})
                .boxed(),
        );
        response.headers_mut().insert(
            header::CONTENT_TYPE,
            HeaderValue::from_static("application/json"),
        );
        response
    }

    fn title(&self, call: u32, body: Map<String, Value>) -> Response<Body> {
        let stream = body.get("stream") == Some(&Value::Bool(true));
        self.trace.record(&Event::Request {
            call,
            kind: Kind::Title,
            body: &Value::Object(body),
            injected: &[],
        });
        let completion = json!({
            "id": "remanie-title",
            "object": "chat.completion",
            "created": 0,
            "model": self.upstream.model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "remanie-title",
                        "type": "function",
                        "function": {"name": "session_title", "arguments": json!({"session_title": TITLE}).to_string()},
                    }],
                },
                "finish_reason": "tool_calls",
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        });
        let (raw, content_type) = if stream {
            (sse(&completion), "text/event-stream")
        } else {
            (completion.to_string(), "application/json")
        };
        self.trace.record(&Event::Response {
            call,
            status: Some(200),
            body: &if stream { assemble(&raw) } else { completion },
            raw_sse: stream.then_some(raw.as_str()),
            error: None,
        });
        let mut response = Response::new(
            Full::new(Bytes::from(raw))
                .map_err(|never| match never {})
                .boxed(),
        );
        response
            .headers_mut()
            .insert(header::CONTENT_TYPE, HeaderValue::from_static(content_type));
        response
    }
}

fn text(status: StatusCode, message: &'static str) -> Response<Body> {
    let mut response = Response::new(
        Full::new(Bytes::from_static(message.as_bytes()))
            .map_err(|never| match never {})
            .boxed(),
    );
    *response.status_mut() = status;
    response
}

fn transport_failure(error: &reqwest::Error) -> String {
    let mut detail = error.to_string();
    let mut source = std::error::Error::source(error);
    while let Some(cause) = source {
        detail.push_str(": ");
        detail.push_str(&cause.to_string());
        source = cause.source();
    }
    detail
}

fn upstream_headers(from: &HeaderMap, authorization: Option<&HeaderValue>) -> HeaderMap {
    let mut headers = HeaderMap::new();
    for (name, value) in from {
        let skip = HOP_BY_HOP.contains(&name.as_str())
            || [
                header::HOST,
                header::CONTENT_LENGTH,
                header::AUTHORIZATION,
                header::ACCEPT_ENCODING,
            ]
            .contains(name);
        if !skip {
            headers.append(name.clone(), value.clone());
        }
    }
    if let Some(value) = authorization {
        headers.insert(header::AUTHORIZATION, value.clone());
    }
    headers
}

fn downstream_headers(from: &HeaderMap) -> HeaderMap {
    let mut headers = HeaderMap::new();
    for (name, value) in from {
        if !HOP_BY_HOP.contains(&name.as_str()) && name != header::CONTENT_LENGTH {
            headers.append(HeaderName::clone(name), value.clone());
        }
    }
    headers
}

/// Tells turn, compaction and title requests apart by the rule `kind` in `checks/make_trace_fixture.py` also applies.
pub fn classify(body: &Map<String, Value>) -> Kind {
    let named = |tool: &Value| tool["function"]["name"] == "session_title";
    if body
        .get("tools")
        .and_then(Value::as_array)
        .is_some_and(|tools| tools.iter().any(named))
    {
        return Kind::Title;
    }
    let last = body
        .get("messages")
        .and_then(Value::as_array)
        .and_then(|messages| messages.last());
    match last {
        Some(message)
            if message["role"] == "user"
                && message["content"]
                    .as_str()
                    .is_some_and(|content| content.contains(SUMMARY_PROMPT)) =>
        {
            Kind::Compaction
        }
        _ => Kind::Turn,
    }
}

/// Appends each sampling parameter the body lacks to Grok's own bytes, so every byte Grok wrote reaches the endpoint unchanged.
pub fn inject(
    raw: &[u8],
    body: &Map<String, Value>,
    sampling_params: &Map<String, Value>,
) -> (Vec<u8>, Vec<String>) {
    let missing: Vec<_> = sampling_params
        .iter()
        .filter(|(key, _)| !body.contains_key(*key))
        .collect();
    let Some(close) = raw.iter().rposition(|byte| *byte == b'}') else {
        return (raw.to_vec(), Vec::new());
    };
    if missing.is_empty() {
        return (raw.to_vec(), Vec::new());
    }
    let mut out = raw[..close].to_vec();
    for (index, (key, value)) in missing.iter().enumerate() {
        if index > 0 || !body.is_empty() {
            out.push(b',');
        }
        out.extend(serde_json::to_vec(key).unwrap_or_default());
        out.push(b':');
        out.extend(serde_json::to_vec(value).unwrap_or_default());
    }
    out.extend_from_slice(&raw[close..]);
    (
        out,
        missing.into_iter().map(|(key, _)| key.clone()).collect(),
    )
}

/// Rebuilds a chat completion from a stream of `chat.completion.chunk` events: text fields join,
/// arrays extend, tool calls merge by index, and the last `usage` wins.
pub fn assemble(raw_sse: &str) -> Value {
    let mut top = Map::new();
    let mut choices: Vec<(u64, Map<String, Value>)> = Vec::new();
    for data in events(raw_sse) {
        let Ok(Value::Object(chunk)) = serde_json::from_str::<Value>(&data) else {
            continue;
        };
        for (key, value) in chunk {
            match key.as_str() {
                "object" => {}
                "usage" => {
                    if !value.is_null() {
                        top.insert(key, value);
                    }
                }
                "choices" => {
                    for choice in value.as_array().into_iter().flatten() {
                        merge_choice(&mut choices, choice);
                    }
                }
                _ => {
                    if !value.is_null() {
                        top.entry(key).or_insert(value);
                    }
                }
            }
        }
    }
    choices.sort_by_key(|(index, _)| *index);
    let mut out = Map::new();
    out.insert("object".into(), "chat.completion".into());
    for key in ["id", "created", "model"] {
        if let Some(value) = top.remove(key) {
            out.insert(key.into(), value);
        }
    }
    let usage = top.remove("usage");
    out.extend(top);
    out.insert(
        "choices".into(),
        choices
            .into_iter()
            .map(|(_, choice)| Value::Object(choice))
            .collect(),
    );
    if let Some(usage) = usage {
        out.insert("usage".into(), usage);
    }
    Value::Object(out)
}

fn events(raw: &str) -> Vec<String> {
    let mut events = Vec::new();
    let mut data: Vec<&str> = Vec::new();
    for line in raw.lines().chain([""]) {
        if line.is_empty() {
            if !data.is_empty() {
                let joined = data.join("\n");
                if joined != "[DONE]" {
                    events.push(joined);
                }
                data.clear();
            }
        } else if let Some(value) = line.strip_prefix("data:") {
            data.push(value.strip_prefix(' ').unwrap_or(value));
        }
    }
    events
}

fn merge_choice(choices: &mut Vec<(u64, Map<String, Value>)>, choice: &Value) {
    let index = choice["index"].as_u64().unwrap_or(0);
    let position = match choices.iter().position(|(existing, _)| *existing == index) {
        Some(position) => position,
        None => {
            let mut fresh = Map::new();
            fresh.insert("index".into(), index.into());
            fresh.insert("message".into(), Value::Object(Map::new()));
            choices.push((index, fresh));
            choices.len() - 1
        }
    };
    let target = &mut choices[position].1;
    for (key, value) in choice.as_object().into_iter().flatten() {
        match key.as_str() {
            "index" => {}
            "delta" => {
                let message = target
                    .get_mut("message")
                    .and_then(Value::as_object_mut)
                    .expect("every choice holds a message");
                for (field, part) in value.as_object().into_iter().flatten() {
                    if field == "tool_calls" {
                        merge_tool_calls(message, part);
                    } else {
                        merge_value(message, field, part);
                    }
                }
            }
            _ => merge_value(target, key, value),
        }
    }
}

fn merge_tool_calls(message: &mut Map<String, Value>, parts: &Value) {
    let calls = message
        .entry("tool_calls")
        .or_insert_with(|| Value::Array(Vec::new()));
    let Some(calls) = calls.as_array_mut() else {
        return;
    };
    for part in parts.as_array().into_iter().flatten() {
        let index = part["index"].as_u64().unwrap_or(calls.len() as u64) as usize;
        while calls.len() <= index {
            calls.push(Value::Object(Map::new()));
        }
        let Some(call) = calls[index].as_object_mut() else {
            continue;
        };
        for (field, value) in part.as_object().into_iter().flatten() {
            if field == "function" {
                let function = call
                    .entry("function")
                    .or_insert_with(|| Value::Object(Map::new()));
                if let Some(function) = function.as_object_mut() {
                    for (name, piece) in value.as_object().into_iter().flatten() {
                        merge_value(function, name, piece);
                    }
                }
            } else if field != "index" {
                merge_value(call, field, value);
            }
        }
    }
}

fn merge_value(target: &mut Map<String, Value>, key: &str, value: &Value) {
    if value.is_null() {
        target.entry(key).or_insert(Value::Null);
        return;
    }
    match (target.get_mut(key), value) {
        (Some(Value::String(existing)), Value::String(more))
            if key != "role" && key != "id" && key != "type" =>
        {
            existing.push_str(more)
        }
        (Some(Value::Array(existing)), Value::Array(more)) => existing.extend(more.iter().cloned()),
        _ => {
            target.insert(key.to_owned(), value.clone());
        }
    }
}

fn usage(body: &Value) -> Option<Usage> {
    let usage = body.get("usage")?;
    Some(Usage {
        prompt_tokens: usage.get("prompt_tokens")?.as_u64()?,
        completion_tokens: usage.get("completion_tokens")?.as_u64()?,
        cached_tokens: usage
            .pointer("/prompt_tokens_details/cached_tokens")
            .and_then(Value::as_u64)
            .unwrap_or(0),
        complete: true,
    })
}

/// Serializes a chat completion as the chunk stream an OpenAI-compatible server sends for it.
pub fn sse(completion: &Value) -> String {
    let head = json!({
        "id": completion["id"],
        "object": "chat.completion.chunk",
        "created": completion["created"],
        "model": completion["model"],
    });
    let chunk = |extra: Value| {
        let mut chunk = head.clone();
        if let (Some(chunk), Value::Object(extra)) = (chunk.as_object_mut(), extra) {
            chunk.extend(extra);
        }
        format!("data: {chunk}\n\n")
    };
    let mut out = String::new();
    for choice in completion["choices"].as_array().into_iter().flatten() {
        let mut delta = Map::new();
        for (key, value) in choice["message"].as_object().into_iter().flatten() {
            if key == "tool_calls" {
                let calls: Vec<Value> = value
                    .as_array()
                    .into_iter()
                    .flatten()
                    .enumerate()
                    .map(|(index, call)| {
                        let mut call = call.clone();
                        call["index"] = index.into();
                        call
                    })
                    .collect();
                delta.insert(key.clone(), calls.into());
            } else if !value.is_null() {
                delta.insert(key.clone(), value.clone());
            }
        }
        out += &chunk(
            json!({"choices": [{"index": choice["index"], "delta": delta, "finish_reason": null}]}),
        );
        out += &chunk(
            json!({"choices": [{"index": choice["index"], "delta": {}, "finish_reason": choice["finish_reason"]}]}),
        );
    }
    if !completion["usage"].is_null() {
        out += &chunk(json!({"choices": [], "usage": completion["usage"]}));
    }
    out + "data: [DONE]\n\n"
}

#[cfg(test)]
mod tests {
    use super::*;

    fn object(value: Value) -> Map<String, Value> {
        value.as_object().unwrap().clone()
    }

    #[test]
    fn requests_are_told_apart_like_the_oracle_does() {
        let turn = object(
            json!({"messages": [{"role": "user", "content": "hi"}], "tools": [{"type": "function", "function": {"name": "grep"}}]}),
        );
        let compaction = object(
            json!({"messages": [{"role": "user", "content": format!("{SUMMARY_PROMPT}, with care.")}]}),
        );
        let title = object(
            json!({"messages": [], "tools": [{"type": "function", "function": {"name": "session_title"}}]}),
        );
        let summary_from_the_model =
            object(json!({"messages": [{"role": "assistant", "content": SUMMARY_PROMPT}]}));
        assert_eq!(classify(&turn), Kind::Turn);
        assert_eq!(classify(&compaction), Kind::Compaction);
        assert_eq!(classify(&title), Kind::Title);
        assert_eq!(classify(&summary_from_the_model), Kind::Turn);
    }

    #[test]
    fn injection_appends_only_missing_keys_after_grok_bytes() {
        let raw = br#"{"model":"m","temperature":1.0,"max_tokens":512,"stream":true}"#;
        let body = object(serde_json::from_slice(raw).unwrap());
        let sampling = object(
            json!({"max_tokens": 64, "temperature": 0.2, "provider": {"order": ["a"]}, "reasoning": {"max_tokens": 9}}),
        );
        let (forwarded, injected) = inject(raw, &body, &sampling);
        assert_eq!(injected, ["provider", "reasoning"]);
        assert_eq!(
            std::str::from_utf8(&forwarded).unwrap(),
            r#"{"model":"m","temperature":1.0,"max_tokens":512,"stream":true,"provider":{"order":["a"]},"reasoning":{"max_tokens":9}}"#
        );
        let (unchanged, none) = inject(raw, &body, &object(json!({"max_tokens": 64})));
        assert_eq!(unchanged, raw);
        assert!(none.is_empty());
    }

    #[test]
    fn a_chunk_stream_reassembles_into_the_completion_it_carries() {
        let completion = json!({
            "id": "c1",
            "object": "chat.completion",
            "created": 7,
            "model": "m",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Done.",
                    "reasoning_content": "Think.",
                    "tool_calls": [
                        {"id": "a", "type": "function", "function": {"name": "grep", "arguments": "{\"pattern\":\"x\"}"}},
                        {"id": "b", "type": "function", "function": {"name": "read_file", "arguments": "{}"}},
                    ],
                },
                "finish_reason": "tool_calls",
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
        });
        assert_eq!(assemble(&sse(&completion)), completion);
    }

    #[test]
    fn split_deltas_join_and_the_last_usage_wins() {
        let raw = [
            r#"data: {"id":"c","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{"role":"assistant","content":"Hel","tool_calls":[{"index":0,"id":"t","type":"function","function":{"name":"grep","arguments":"{\"pa"}}]},"finish_reason":null}]}"#,
            r#"data: {"id":"c","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{"content":"lo","tool_calls":[{"index":0,"function":{"arguments":"ttern\":1}"}}]},"finish_reason":null}]}"#,
            r#"data: {"id":"c","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":null}"#,
            r#"data: {"id":"c","object":"chat.completion.chunk","created":1,"model":"m","choices":[],"usage":{"prompt_tokens":3,"completion_tokens":1,"prompt_tokens_details":{"cached_tokens":2}}}"#,
            "data: [DONE]",
        ]
        .join("\n\n");
        let assembled = assemble(&raw);
        let message = &assembled["choices"][0]["message"];
        assert_eq!(message["content"], "Hello");
        assert_eq!(
            message["tool_calls"][0]["function"]["arguments"],
            "{\"pattern\":1}"
        );
        assert_eq!(message["tool_calls"][0]["id"], "t");
        assert_eq!(assembled["choices"][0]["finish_reason"], "stop");
        assert_eq!(
            usage(&assembled),
            Some(Usage {
                prompt_tokens: 3,
                completion_tokens: 1,
                cached_tokens: 2,
                complete: true
            })
        );
    }
}
