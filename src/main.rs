mod grok;
mod input;
mod outcome;
mod proxy;
mod trace;

use std::io::{Read, Write};
use std::path::PathBuf;
use std::process::{ExitCode, Stdio};
use std::sync::Arc;
use std::time::{Duration, Instant};

use hyper::header::HeaderValue;
use serde_json::Value;

use grok::{Home, Signals};
use input::Input;
use outcome::{Cause, GrokExit, Outcome, RunResult, Tally};
use proxy::{Proxy, Upstream};
use trace::{Event, Trace};

fn main() -> ExitCode {
    let started = Instant::now();
    grok::adopt_orphans();
    let mut stdin = Vec::new();
    let result = match std::io::stdin()
        .take(input::MAX_INPUT_BYTES as u64 + 1)
        .read_to_end(&mut stdin)
    {
        Ok(_) => run(&stdin, started),
        Err(error) => RunResult::new(
            Outcome::Infrastructure(Cause::Internal {
                detail: format!("read stdin: {error}"),
            }),
            Tally::default(),
            None,
            None,
            started.elapsed(),
        ),
    };
    let mut stdout = std::io::stdout().lock();
    if serde_json::to_writer(&mut stdout, &result).is_ok() {
        let _ = writeln!(stdout);
    }
    ExitCode::from(result.outcome.exit_code())
}

struct Attempt {
    outcome: Outcome,
    tally: Tally,
    grok_exit: Option<GrokExit>,
    grok_session_dir: Option<String>,
    trace: Option<Arc<Trace>>,
}

impl Attempt {
    fn failed(cause: Cause, trace: Option<Arc<Trace>>) -> Self {
        Self {
            outcome: Outcome::Infrastructure(cause),
            tally: Tally::default(),
            grok_exit: None,
            grok_session_dir: None,
            trace,
        }
    }
}

fn run(stdin: &[u8], started: Instant) -> RunResult {
    let mut session_dir: Option<PathBuf> = None;
    let attempt = match input::parse(stdin, &mut session_dir) {
        Err(cause) => Attempt::failed(cause, None),
        Ok((raw, input)) => match tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
        {
            Ok(runtime) => {
                let attempt = runtime.block_on(attempt(&raw, &input, started));
                runtime.shutdown_timeout(Duration::from_secs(1));
                attempt
            }
            Err(error) => Attempt::failed(internal(format!("start runtime: {error}")), None),
        },
    };
    let mut result = RunResult::new(
        attempt.outcome,
        attempt.tally,
        attempt.grok_exit,
        attempt.grok_session_dir,
        started.elapsed(),
    );
    if let Some(dir) = &session_dir
        && let Err(error) = result.write_atomically(dir)
    {
        result.outcome = Outcome::Infrastructure(internal(format!("write result.json: {error}")));
        result.gradable = false;
    }
    if let Some(trace) = &attempt.trace {
        trace.record(&Event::Result(&result));
    }
    result
}

fn internal(detail: impl Into<String>) -> Cause {
    Cause::Internal {
        detail: detail.into(),
    }
}

async fn attempt(raw: &Value, input: &Input, started: Instant) -> Attempt {
    let deadline =
        tokio::time::Instant::from_std(started + Duration::from_secs(input.deadline_secs));
    let mut signals = match Signals::install() {
        Ok(signals) => signals,
        Err(error) => {
            return Attempt::failed(internal(format!("install signal handlers: {error}")), None);
        }
    };
    let authorization = match authorization(input) {
        Ok(authorization) => authorization,
        Err(cause) => return Attempt::failed(cause, None),
    };
    let Some(program) = std::env::var_os("REMANIE_GROK").map(PathBuf::from) else {
        return Attempt::failed(
            grok::binary("REMANIE_GROK does not name the Grok Build binary"),
            None,
        );
    };
    let proxy = match Proxy::bind().await {
        Ok(proxy) => proxy,
        Err(error) => {
            return Attempt::failed(
                Cause::ProxyBind {
                    detail: error.to_string(),
                },
                None,
            );
        }
    };
    let home = match Home::prepare(input, &proxy.base_url) {
        Ok(home) => home,
        Err(error) => {
            return Attempt::failed(internal(format!("write the Grok home: {error}")), None);
        }
    };
    let env = grok::environment(input, &proxy.base_url, &home);
    let version = match grok::version(&program, &env) {
        Ok(version) => version,
        Err(cause) => return Attempt::failed(cause, None),
    };
    let trace = match Trace::create(&input.session_dir, started) {
        Ok(trace) => Arc::new(trace),
        Err(error) => {
            return Attempt::failed(internal(format!("create trace.jsonl: {error}")), None);
        }
    };
    trace.header(raw, grok::AGENT_FILE, &version);
    let upstream = Upstream {
        chat_completions: format!("{}/chat/completions", input.base_url.trim_end_matches('/')),
        authorization,
        sampling_params: input.sampling_params.clone(),
        model: input.model.clone(),
        context_window: input.context_window_tokens,
    };
    let running = match proxy.serve(upstream, trace.clone()) {
        Ok(running) => running,
        Err(error) => {
            return Attempt::failed(
                internal(format!("start the HTTP client: {error}")),
                Some(trace),
            );
        }
    };
    let child = tokio::process::Command::new(&program)
        .args(grok::arguments(input, &home))
        .env_clear()
        .envs(env)
        .current_dir(&input.workspace)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .process_group(0)
        .spawn();
    let child = match child {
        Ok(child) => child,
        Err(error) => {
            running.finish().await;
            return Attempt::failed(
                grok::binary(format!("start {}: {error}", program.display())),
                Some(trace),
            );
        }
    };
    let exited = grok::supervise(child, deadline, &mut signals, running.unreachable()).await;
    let record = running.finish().await;
    let grok_exit = exited.status.map(grok::exit_of);
    let (outcome, final_answer) = grok::outcome(
        exited.ended,
        grok_exit,
        grok::printed(&exited.stdout),
        &record,
    );
    Attempt {
        outcome,
        tally: Tally {
            final_answer,
            turns: record.turns,
            compactions: record.compactions,
            model_calls: record.model_calls,
            usage: record.usage,
        },
        grok_exit,
        grok_session_dir: home.session_dir(&input.session_dir),
        trace: Some(trace),
    }
}

fn authorization(input: &Input) -> Result<Option<HeaderValue>, Cause> {
    let Some(name) = input.api_key_env.as_deref() else {
        return Ok(None);
    };
    let key = std::env::var(name).map_err(|_| {
        input::config(format!(
            "environment variable {name} is not set or not UTF-8"
        ))
    })?;
    let mut value = HeaderValue::from_str(&format!("Bearer {key}")).map_err(|_| {
        input::config(format!(
            "{name} holds characters an HTTP header cannot carry"
        ))
    })?;
    value.set_sensitive(true);
    Ok(Some(value))
}
