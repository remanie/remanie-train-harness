use std::collections::hash_map::RandomState;
use std::hash::{BuildHasher, Hasher};
use std::time::Duration;

use reqwest::header::{AUTHORIZATION, CONTENT_TYPE, HeaderValue, RETRY_AFTER};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use tokio::time::Instant;

use crate::outcome::{Cause, Outcome, Overflow, Usage};
use crate::trace::{CallKind, Event, MessageRange, Trace};

pub const MAX_BODY_BYTES: usize = 16 * 1024 * 1024;
const MAX_ERROR_BODY_BYTES: usize = 64 * 1024;
const RETRY_BASE: Duration = Duration::from_secs(1);
const RETRY_CAP: Duration = Duration::from_secs(30);
const CONTEXT_LENGTH_PATTERNS: [&str; 5] = [
    "context length",
    "context_length",
    "maximum context",
    "max_total_tokens",
    "input length",
];

pub fn bearer_header(key: &str) -> Result<HeaderValue, &'static str> {
    const INVALID: &str =
        "API key must be nonempty and contain no whitespace or control characters";
    if key.is_empty() || !key.bytes().all(|byte| byte.is_ascii_graphic()) {
        return Err(INVALID);
    }
    let mut header = HeaderValue::from_str(&format!("Bearer {key}")).map_err(|_| INVALID)?;
    header.set_sensitive(true);
    Ok(header)
}

#[derive(Debug, Deserialize)]
#[serde(try_from = "Map<String, Value>")]
pub struct SamplingParams {
    params: Map<String, Value>,
    max_tokens: u64,
}

impl TryFrom<Map<String, Value>> for SamplingParams {
    type Error = &'static str;

    fn try_from(params: Map<String, Value>) -> Result<Self, Self::Error> {
        let max_tokens = params
            .get("max_tokens")
            .and_then(Value::as_u64)
            .filter(|tokens| *tokens > 0)
            .ok_or("sampling_params.max_tokens must be a positive integer")?;
        Ok(Self { params, max_tokens })
    }
}

impl SamplingParams {
    pub fn max_tokens(&self) -> u64 {
        self.max_tokens
    }
}

#[derive(Serialize)]
struct Body<'a> {
    #[serde(flatten)]
    sampling: &'a Map<String, Value>,
    model: &'a str,
    messages: &'a [Value],
    tools: &'a [Value],
    tool_choice: &'a str,
    stream: bool,
}

#[derive(Debug, Serialize)]
#[serde(tag = "failure", rename_all = "snake_case")]
pub enum Failure {
    Connect {
        detail: String,
    },
    Lost {
        detail: String,
    },
    Status {
        status: u16,
        body_excerpt: String,
        retry_after_ms: Option<u64>,
    },
    Response {
        detail: String,
        body_excerpt: String,
    },
}

impl Failure {
    fn retry_in(&self, attempt: u32, remaining: Duration) -> Option<Duration> {
        let at_least = match self {
            Failure::Connect { .. } => Duration::ZERO,
            Failure::Status {
                status: 429 | 503,
                retry_after_ms,
                ..
            } => Duration::from_millis(retry_after_ms.unwrap_or(0)),
            _ => return None,
        };
        Some(backoff(attempt, at_least)).filter(|delay| *delay < remaining)
    }

    fn into_outcome(self, attempts: u32) -> Outcome {
        Outcome::Infrastructure(match self {
            Failure::Connect { detail } => Cause::EndpointUnavailable {
                attempts,
                status: None,
                detail,
            },
            Failure::Status {
                status: status @ (429 | 503),
                body_excerpt,
                ..
            } => Cause::EndpointUnavailable {
                attempts,
                status: Some(status),
                detail: body_excerpt,
            },
            Failure::Status {
                status: 400,
                body_excerpt,
                ..
            } if names_context_length(&body_excerpt) => {
                return Outcome::ContextOverflow(Overflow::EndpointRejected {
                    status: 400,
                    body_excerpt,
                });
            }
            Failure::Status {
                status,
                body_excerpt,
                ..
            } => Cause::EndpointStatus {
                status,
                body_excerpt,
            },
            Failure::Lost { detail } => Cause::ConnectionLost { detail },
            Failure::Response { detail, .. } => Cause::EndpointResponse { detail },
        })
    }
}

fn names_context_length(body: &str) -> bool {
    let body = body.to_lowercase();
    CONTEXT_LENGTH_PATTERNS
        .iter()
        .any(|pattern| body.contains(pattern))
}

fn backoff(attempt: u32, at_least: Duration) -> Duration {
    let ceiling = RETRY_BASE
        .saturating_mul(1 << (attempt - 1).min(16))
        .min(RETRY_CAP);
    let unit = RandomState::new().build_hasher().finish() as f64 / u64::MAX as f64;
    ceiling.mul_f64(unit).max(at_least).min(RETRY_CAP)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Finish {
    Stop,
    ToolCalls,
    Length,
    Other(String),
}

pub struct Sampled {
    pub message: Value,
    pub finish: Finish,
    pub usage: Option<Usage>,
    pub context_tokens: Option<u64>,
}

pub struct Request<'a> {
    pub kind: CallKind,
    pub turn: u32,
    pub prompt: MessageRange,
    pub messages: &'a [Value],
    pub max_tokens_override: Option<u64>,
}

pub fn http_client() -> Result<reqwest::Client, Cause> {
    reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .connect_timeout(Duration::from_secs(30))
        // A pooled connection the server closed while tools ran would fail after send, which
        // is never retried; one connection per request keeps that from looking like a lost reply.
        .pool_max_idle_per_host(0)
        .build()
        .map_err(|error| Cause::Internal {
            detail: format!("build HTTP client: {error}"),
        })
}

pub struct ModelClient<'a> {
    http: reqwest::Client,
    url: reqwest::Url,
    authorization: Option<HeaderValue>,
    model: &'a str,
    sampling: &'a SamplingParams,
    tools: &'a [Value],
    deadline: Instant,
    calls: u32,
    in_flight: bool,
}

impl<'a> ModelClient<'a> {
    pub fn new(
        http: reqwest::Client,
        url: reqwest::Url,
        authorization: Option<HeaderValue>,
        model: &'a str,
        sampling: &'a SamplingParams,
        tools: &'a [Value],
        deadline: Instant,
    ) -> Self {
        Self {
            http,
            url,
            authorization,
            model,
            sampling,
            tools,
            deadline,
            calls: 0,
            in_flight: false,
        }
    }

    pub fn calls(&self) -> u32 {
        self.calls
    }

    pub fn in_flight(&self) -> bool {
        self.in_flight
    }

    pub async fn sample(
        &mut self,
        request: Request<'_>,
        trace: &mut Trace,
    ) -> Result<Sampled, Outcome> {
        self.calls += 1;
        let call = self.calls;
        let mut sampling = self.sampling.params.clone();
        if let Some(max_tokens) = request.max_tokens_override {
            sampling.insert("max_tokens".into(), max_tokens.into());
        }
        let body = serde_json::to_vec(&Body {
            sampling: &sampling,
            model: self.model,
            messages: request.messages,
            tools: self.tools,
            tool_choice: request.kind.tool_choice(),
            stream: false,
        })
        .map_err(|error| {
            Outcome::Infrastructure(Cause::Internal {
                detail: format!("serialize request: {error}"),
            })
        })?;
        if body.len() > MAX_BODY_BYTES {
            return Err(Outcome::ContextOverflow(Overflow::RequestSize {
                bytes: body.len(),
            }));
        }
        let started = Instant::now();
        let mut attempt = 0;
        loop {
            attempt += 1;
            let failure = match self.post(&body).await.and_then(parse) {
                Ok((mut response, finish)) => {
                    trace.record(&Event::ModelCall {
                        call,
                        turn: request.turn,
                        kind: request.kind,
                        prompt: request.prompt,
                        max_tokens_override: request.max_tokens_override,
                        attempts: attempt,
                        latency_ms: started.elapsed().as_millis() as u64,
                        response: &response,
                    });
                    let usage = usage(&response["usage"]);
                    return Ok(Sampled {
                        context_tokens: response["usage"]["total_tokens"]
                            .as_u64()
                            .or(usage.map(|usage| usage.prompt_tokens + usage.completion_tokens)),
                        usage,
                        message: response["choices"][0]["message"].take(),
                        finish,
                    });
                }
                Err(failure) => failure,
            };
            let retry_in = failure.retry_in(
                attempt,
                self.deadline.saturating_duration_since(Instant::now()),
            );
            trace.record(&Event::EndpointError {
                call,
                turn: request.turn,
                attempt,
                failure: &failure,
                retry_in_ms: retry_in.map(|delay| delay.as_millis() as u64),
            });
            match retry_in {
                Some(delay) => tokio::time::sleep(delay).await,
                None => return Err(failure.into_outcome(attempt)),
            }
        }
    }

    async fn post(&mut self, body: &[u8]) -> Result<Vec<u8>, Failure> {
        let mut request = self
            .http
            .post(self.url.clone())
            .header(CONTENT_TYPE, "application/json")
            .body(body.to_vec());
        if let Some(authorization) = &self.authorization {
            request = request.header(AUTHORIZATION, authorization.clone());
        }
        self.in_flight = true;
        let exchanged = exchange(request).await;
        self.in_flight = false;
        exchanged
    }
}

async fn exchange(request: reqwest::RequestBuilder) -> Result<Vec<u8>, Failure> {
    let mut response = request.send().await.map_err(|error| {
        let detail = error_chain(&error);
        if error.is_connect() {
            Failure::Connect { detail }
        } else {
            Failure::Lost { detail }
        }
    })?;
    let status = response.status();
    if !status.is_success() {
        let retry_after_ms = response
            .headers()
            .get(RETRY_AFTER)
            .and_then(|value| value.to_str().ok()?.trim().parse::<u64>().ok())
            .map(|seconds| seconds.saturating_mul(1000));
        let mut body = Vec::new();
        while body.len() < MAX_ERROR_BODY_BYTES {
            match response.chunk().await {
                Ok(Some(chunk)) => body.extend_from_slice(&chunk),
                _ => break,
            }
        }
        return Err(Failure::Status {
            status: status.as_u16(),
            body_excerpt: excerpt(&body),
            retry_after_ms,
        });
    }
    let mut body = Vec::new();
    while let Some(chunk) = response.chunk().await.map_err(|error| Failure::Lost {
        detail: error_chain(&error),
    })? {
        if body.len() + chunk.len() > MAX_BODY_BYTES {
            return Err(Failure::Response {
                detail: "response exceeds 16 MiB".into(),
                body_excerpt: excerpt(&body),
            });
        }
        body.extend_from_slice(&chunk);
    }
    Ok(body)
}

fn parse(body: Vec<u8>) -> Result<(Value, Finish), Failure> {
    let unusable = |detail: String| Failure::Response {
        detail,
        body_excerpt: excerpt(&body),
    };
    let response: Value = serde_json::from_slice(&body)
        .map_err(|error| unusable(format!("response is not JSON: {error}")))?;
    let choice = match response["choices"].as_array() {
        Some(choices) if choices.len() == 1 => &choices[0],
        _ => return Err(unusable("expected exactly one choice".into())),
    };
    if !choice["message"].is_object() || choice["message"]["role"] != "assistant" {
        return Err(unusable("choice has no assistant message".into()));
    }
    let finish = match choice["finish_reason"].as_str() {
        Some("stop") => Finish::Stop,
        Some("tool_calls") => Finish::ToolCalls,
        Some("length") => Finish::Length,
        Some(other) => Finish::Other(other.into()),
        None => return Err(unusable("finish_reason is missing".into())),
    };
    Ok((response, finish))
}

fn usage(usage: &Value) -> Option<Usage> {
    Some(Usage {
        prompt_tokens: usage["prompt_tokens"].as_u64()?,
        completion_tokens: usage["completion_tokens"].as_u64()?,
        cached_tokens: usage["prompt_tokens_details"]["cached_tokens"]
            .as_u64()
            .unwrap_or(0),
        complete: true,
    })
}

fn excerpt(body: &[u8]) -> String {
    String::from_utf8_lossy(&body[..body.len().min(MAX_ERROR_BODY_BYTES)]).into_owned()
}

fn error_chain(error: &dyn std::error::Error) -> String {
    let mut text = error.to_string();
    let mut source = error.source();
    while let Some(next) = source {
        text.push_str(": ");
        text.push_str(&next.to_string());
        source = next.source();
    }
    text
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn status(status: u16, body: &str) -> Failure {
        Failure::Status {
            status,
            body_excerpt: body.into(),
            retry_after_ms: None,
        }
    }

    #[test]
    fn only_requests_the_endpoint_did_not_process_are_retried() {
        let hour = Duration::from_secs(3600);
        let lost = Failure::Lost {
            detail: "reset".into(),
        };
        let response = Failure::Response {
            detail: "not JSON".into(),
            body_excerpt: String::new(),
        };
        let refused = Failure::Connect {
            detail: "refused".into(),
        };
        for retried in [&refused, &status(429, ""), &status(503, "")] {
            assert!(retried.retry_in(1, hour).is_some(), "{retried:?}");
        }
        for code in [400, 401, 403, 404, 408, 409, 413, 422, 500, 502, 504] {
            assert_eq!(status(code, "").retry_in(1, hour), None, "{code}");
        }
        assert_eq!(lost.retry_in(1, hour), None);
        assert_eq!(response.retry_in(1, hour), None);
        let after = Failure::Status {
            status: 429,
            body_excerpt: String::new(),
            retry_after_ms: Some(2000),
        };
        assert!(after.retry_in(1, hour).unwrap() >= Duration::from_secs(2));
    }

    #[test]
    fn retries_continue_until_the_next_wait_would_pass_the_deadline() {
        for attempt in 1..=40 {
            assert!(
                status(503, "")
                    .retry_in(attempt, Duration::from_secs(3600))
                    .is_some(),
                "attempt {attempt} was not retried with an hour left"
            );
        }
        assert_eq!(status(503, "").retry_in(1, Duration::ZERO), None);
        let after = Failure::Status {
            status: 429,
            body_excerpt: String::new(),
            retry_after_ms: Some(20_000),
        };
        assert_eq!(after.retry_in(1, Duration::from_secs(10)), None);
    }

    #[test]
    fn final_failures_become_outcomes_with_their_bodies() {
        let as_json = |failure: Failure| serde_json::to_value(failure.into_outcome(3)).unwrap();
        assert_eq!(
            as_json(status(
                400,
                "This model's maximum context length is 131072 tokens"
            )),
            json!({"outcome": "context_overflow", "cause": {"kind": "endpoint_rejected", "status": 400, "body_excerpt": "This model's maximum context length is 131072 tokens"}})
        );
        assert_eq!(
            as_json(status(400, "temperature 0.7 does not match")),
            json!({"outcome": "infrastructure", "cause": {"kind": "endpoint_status", "status": 400, "body_excerpt": "temperature 0.7 does not match"}})
        );
        assert_eq!(
            as_json(status(503, "UpstreamGenerationAbortedError")),
            json!({"outcome": "infrastructure", "cause": {"kind": "endpoint_unavailable", "attempts": 3, "status": 503, "detail": "UpstreamGenerationAbortedError"}})
        );
        assert_eq!(
            as_json(Failure::Lost {
                detail: "reset".into()
            })["cause"]["kind"],
            "connection_lost"
        );
        assert_eq!(
            as_json(Failure::Connect {
                detail: "refused".into()
            })["cause"],
            json!({"kind": "endpoint_unavailable", "attempts": 3, "status": null, "detail": "refused"})
        );
    }

    #[test]
    fn backoff_is_jittered_capped_and_honours_retry_after() {
        for attempt in 1..=10 {
            let ceiling = RETRY_BASE.saturating_mul(1 << (attempt - 1)).min(RETRY_CAP);
            assert!(backoff(attempt, Duration::ZERO) <= ceiling);
        }
        let delays: std::collections::BTreeSet<_> =
            (0..8).map(|_| backoff(3, Duration::ZERO)).collect();
        assert!(delays.len() > 1, "backoff is not jittered");
        assert!(backoff(1, Duration::from_secs(5)) >= Duration::from_secs(5));
        assert_eq!(backoff(1, Duration::from_secs(600)), RETRY_CAP);
    }

    #[test]
    fn bearer_headers_are_sensitive_and_invalid_keys_do_not_leak() {
        let header = bearer_header("test-private-key").unwrap();
        assert_eq!(header, "Bearer test-private-key");
        assert!(header.is_sensitive());
        assert!(!format!("{header:?}").contains("test-private-key"));
        for key in ["", "test-private-key\n", "test private key", "test-λ-key"] {
            assert!(!bearer_header(key).unwrap_err().contains("test"));
        }
    }

    #[test]
    fn sampling_params_require_max_tokens_and_keep_every_other_key() {
        let parse = |value: Value| serde_json::from_value::<SamplingParams>(value);
        assert!(parse(json!({"temperature": 1.0})).is_err());
        assert!(parse(json!({"max_tokens": 0})).is_err());
        assert!(parse(json!({"max_tokens": 1.5})).is_err());
        let params = parse(json!({"logprobs": true, "max_tokens": 64, "x": {"y": [1]}})).unwrap();
        assert_eq!(params.max_tokens(), 64);
        assert_eq!(
            Value::Object(params.params),
            json!({"logprobs": true, "max_tokens": 64, "x": {"y": [1]}})
        );
    }
}
