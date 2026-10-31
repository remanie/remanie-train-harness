use std::ffi::{OsStr, OsString};
use std::path::{Path, PathBuf};
use std::process::{ExitStatus, Stdio};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde_json::Value;
use tokio::io::AsyncReadExt;
use tokio::process::Child;
use tokio::signal::unix::{Signal, SignalKind, signal};
use tokio::sync::watch;
use tokio::time::Instant;

use crate::input::Input;
use crate::outcome::{Cause, GrokExit, GrokFailure, Outcome, Stop, Unavailable};
use crate::proxy::Record;

pub const AGENT_FILE: &str = include_str!("../assets/agent.md");
const UPSTREAM: &str = include_str!("../upstream.json");
/// Grok reads its API key from this variable; the proxy replaces the key on every forwarded request.
const KEY_ENV: &str = "REMANIE_TRAIN_KEY";
const KEY_PLACEHOLDER: &str = "replaced-by-the-trial-runner-proxy";
const STOP_GRACE: Duration = Duration::from_secs(5);
const STDOUT_GRACE: Duration = Duration::from_secs(2);
const URL_VARIABLES: [&str; 8] = [
    "GROK_CLI_CHAT_PROXY_BASE_URL",
    "GROK_XAI_API_BASE_URL",
    "GROK_MODELS_BASE_URL",
    "GROK_FEEDBACK_BASE_URL",
    "GROK_TRACE_UPLOAD_URL",
    "GROK_MANAGED_CONFIG_URL",
    "GROK_CODE_WEB_URL",
    "GROK_CONVERSATIONS_BASE_URL",
];
const SETTINGS: [(&str, &str); 18] = [
    ("GROK_TELEMETRY_ENABLED", "false"),
    ("GROK_TELEMETRY_MIXPANEL_ENABLED", "false"),
    ("GROK_TELEMETRY_MIXPANEL_TOKEN", ""),
    ("GROK_TELEMETRY_EVENTS_URL", ""),
    ("GROK_TELEMETRY_EVENTS_API_KEY", ""),
    ("GROK_TELEMETRY_TRACE_UPLOAD", "false"),
    ("GROK_FEEDBACK_ENABLED", "false"),
    ("GROK_TRACE_UPLOAD", "false"),
    ("GROK_INSTRUMENTATION", "disabled"),
    ("OTEL_SDK_DISABLED", "true"),
    ("DISABLE_TELEMETRY", "1"),
    ("DISABLE_FEEDBACK_COMMAND", "1"),
    ("GROK_DISABLE_AUTOUPDATER", "1"),
    ("GROK_PROMPT_SUGGESTIONS", "false"),
    ("GROK_AGENT_ID", "remanie-train-harness"),
    ("GROK_TURN_SUMMARY", "0"),
    ("GROK_MEMORY", "0"),
    ("GROK_WORKFLOWS", "0"),
];

pub fn pinned_commit() -> String {
    serde_json::from_str::<Value>(UPSTREAM)
        .ok()
        .and_then(|pin| pin["commit"].as_str().map(str::to_owned))
        .unwrap_or_default()
}

pub struct Home {
    pub dir: PathBuf,
    pub agent_file: PathBuf,
    pub session_id: String,
}

impl Home {
    pub fn prepare(input: &Input, proxy_base_url: &str) -> std::io::Result<Self> {
        let dir = input.session_dir.join("grok-home");
        std::fs::create_dir(&dir)?;
        std::fs::write(dir.join("config.toml"), config_toml(input, proxy_base_url))?;
        let agent_file = dir.join("agent.md");
        std::fs::write(&agent_file, AGENT_FILE)?;
        Ok(Self {
            dir,
            agent_file,
            session_id: session_id(&input.session_dir),
        })
    }

    /// The session's own directory, relative to `session_dir`, once Grok has created it.
    pub fn session_dir(&self, session_dir: &Path) -> Option<String> {
        let sessions = std::fs::read_dir(self.dir.join("sessions")).ok()?;
        sessions
            .flatten()
            .map(|entry| entry.path().join(&self.session_id))
            .find(|path| path.is_dir())
            .and_then(|path| {
                path.strip_prefix(session_dir)
                    .ok()
                    .map(|relative| relative.to_string_lossy().into_owned())
            })
    }
}

pub fn session_id(session_dir: &Path) -> String {
    uuid::Uuid::new_v5(
        &uuid::Uuid::NAMESPACE_URL,
        session_dir.as_os_str().as_encoded_bytes(),
    )
    .to_string()
}

pub fn config_toml(input: &Input, proxy_base_url: &str) -> String {
    let quote = |text: &str| serde_json::to_string(text).unwrap_or_default();
    let model = quote(&input.model);
    let mut lines = vec![
        "[cli]".to_owned(),
        "auto_update = false".to_owned(),
        String::new(),
        "[features]".to_owned(),
        "telemetry = false".to_owned(),
        String::new(),
        "[harness]".to_owned(),
        "disable_codebase_upload = true".to_owned(),
        String::new(),
        "[session]".to_owned(),
        format!(
            "auto_compact_threshold_percent = {}",
            input.threshold_percent()
        ),
        String::new(),
        "[models]".to_owned(),
    ];
    for role in [
        "default",
        "session_summary",
        "image_description",
        "web_search",
    ] {
        lines.push(format!("{role} = {model}"));
    }
    lines.push(String::new());
    lines.push(format!("[model.{model}]"));
    lines.push(format!("name = {model}"));
    lines.push(format!("model = {model}"));
    lines.push(format!("base_url = {}", quote(proxy_base_url)));
    lines.push(format!("env_key = {}", quote(KEY_ENV)));
    lines.push("api_backend = \"chat_completions\"".to_owned());
    lines.push(format!("context_window = {}", input.context_window_tokens));
    lines.push(format!("max_completion_tokens = {}", input.max_tokens()));
    if let Some(temperature) = input.temperature() {
        lines.push(format!("temperature = {temperature:?}"));
    }
    lines.join("\n") + "\n"
}

/// Grok's environment: the wrapper's own, without the endpoint key or any inherited Grok setting,
/// with every Grok service URL pointed at the proxy and the oracle's switches applied.
pub fn environment(input: &Input, proxy_base_url: &str, home: &Home) -> Vec<(OsString, OsString)> {
    let inherited = |name: &OsStr| {
        let name = name.to_string_lossy();
        !(name.starts_with("GROK_")
            || name.starts_with("XAI_")
            || name.eq_ignore_ascii_case("no_proxy")
            || name == KEY_ENV
            || input.api_key_env.as_deref() == Some(&*name))
    };
    let mut no_proxy: Vec<String> = ["NO_PROXY", "no_proxy"]
        .iter()
        .filter_map(|name| std::env::var(name).ok())
        .filter(|value| !value.is_empty())
        .collect();
    no_proxy.dedup();
    no_proxy.push("127.0.0.1,localhost".to_owned());
    let no_proxy = no_proxy.join(",");
    let mut env: Vec<(OsString, OsString)> = std::env::vars_os()
        .filter(|(name, _)| inherited(name))
        .collect();
    let mut set = |name: &str, value: &OsStr| env.push((name.into(), value.to_owned()));
    set("GROK_HOME", home.dir.as_os_str());
    set(
        "GROK_LOG_FILE",
        input.session_dir.join("grok.log").as_os_str(),
    );
    for name in URL_VARIABLES {
        set(name, OsStr::new(proxy_base_url));
    }
    for (name, value) in SETTINGS {
        set(name, OsStr::new(value));
    }
    set(KEY_ENV, OsStr::new(KEY_PLACEHOLDER));
    set("XAI_API_KEY", OsStr::new(KEY_PLACEHOLDER));
    set("NO_PROXY", OsStr::new(&no_proxy));
    set("no_proxy", OsStr::new(&no_proxy));
    env
}

pub fn arguments(input: &Input, home: &Home) -> Vec<OsString> {
    // The attached form keeps a task that starts with a dash from parsing as a flag.
    let mut args: Vec<OsString> = vec![
        format!("-p={}", input.user_query).into(),
        "-m".into(),
        input.model.clone().into(),
    ];
    args.extend(["--cwd".into(), input.workspace.clone().into_os_string()]);
    args.extend(
        [
            "--output-format",
            "json",
            "--always-approve",
            "--no-auto-update",
        ]
        .map(OsString::from),
    );
    args.extend(["--agent".into(), home.agent_file.clone().into_os_string()]);
    args.extend(["--max-turns".into(), input.max_turns.to_string().into()]);
    args.extend(["--session-id".into(), home.session_id.clone().into()]);
    args
}

pub fn version(program: &Path, env: &[(OsString, OsString)]) -> Result<String, Cause> {
    let output = std::process::Command::new(program)
        .arg("--version")
        .env_clear()
        .envs(env.iter().cloned())
        .stdin(Stdio::null())
        .output()
        .map_err(|error| binary(format!("run {} --version: {error}", program.display())))?;
    let reported = String::from_utf8_lossy(&output.stdout).trim().to_owned();
    let commit = pinned_commit();
    if !output.status.success() || commit.len() < 12 || !reported.contains(&commit[..12]) {
        return Err(binary(format!(
            "{} reports {reported:?}, not Grok Build at {commit}",
            program.display()
        )));
    }
    Ok(reported)
}

pub fn binary(detail: impl Into<String>) -> Cause {
    Cause::GrokBinary {
        detail: detail.into(),
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Ended {
    Stopped(Stop),
    Unreachable(String),
}

pub struct Signals {
    terminate: Signal,
    interrupt: Signal,
}

impl Signals {
    pub fn install() -> std::io::Result<Self> {
        Ok(Self {
            terminate: signal(SignalKind::terminate())?,
            interrupt: signal(SignalKind::interrupt())?,
        })
    }
}

pub struct Exited {
    pub status: Option<ExitStatus>,
    pub ended: Option<Ended>,
    pub stdout: Vec<u8>,
}

/// Waits for Grok, stopping its process group at the deadline, on SIGTERM or SIGINT, or when the
/// endpoint was unreachable at the first request: SIGTERM, then SIGKILL after five seconds. A
/// stopped attempt then kills whatever Grok's commands left behind, so nothing runs past its end.
pub async fn supervise(
    mut child: Child,
    deadline: Instant,
    signals: &mut Signals,
    mut unreachable: watch::Receiver<Option<String>>,
) -> Exited {
    let printed = Arc::new(Mutex::new(Vec::new()));
    let reader = child.stdout.take().map(|mut stdout| {
        let printed = printed.clone();
        tokio::spawn(async move {
            let mut chunk = [0; 8192];
            while let Ok(read @ 1..) = stdout.read(&mut chunk).await {
                lock(&printed).extend_from_slice(&chunk[..read]);
            }
        })
    });
    let group = child.id().and_then(|pid| i32::try_from(pid).ok());
    let ended = tokio::select! {
        status = child.wait() => {
            return Exited { status: status.ok(), ended: None, stdout: collect(reader, &printed).await };
        }
        () = tokio::time::sleep_until(deadline) => Ended::Stopped(Stop::Deadline),
        Some(()) = signals.terminate.recv() => Ended::Stopped(Stop::Signal { signal: "SIGTERM" }),
        Some(()) = signals.interrupt.recv() => Ended::Stopped(Stop::Signal { signal: "SIGINT" }),
        Ok(detail) = unreachable.wait_for(Option::is_some) => {
            Ended::Unreachable(detail.clone().unwrap_or_default())
        }
    };
    let status = stop(&mut child, group).await;
    sweep().await;
    Exited {
        status,
        ended: Some(ended),
        stdout: collect(reader, &printed).await,
    }
}

/// A process Grok left behind can hold its stdout open, so the result is taken two seconds after Grok exits at the latest.
async fn collect(reader: Option<tokio::task::JoinHandle<()>>, printed: &Mutex<Vec<u8>>) -> Vec<u8> {
    if let Some(mut reader) = reader
        && tokio::time::timeout(STDOUT_GRACE, &mut reader)
            .await
            .is_err()
    {
        reader.abort();
    }
    std::mem::take(&mut *lock(printed))
}

fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

async fn stop(child: &mut Child, group: Option<i32>) -> Option<ExitStatus> {
    let kill = |signal| {
        if let Some(group) = group {
            // SAFETY: killpg only sends a signal to the process group Grok leads.
            unsafe { libc::killpg(group, signal) };
        }
    };
    kill(libc::SIGTERM);
    if let Ok(status) = tokio::time::timeout(STOP_GRACE, child.wait()).await {
        return status.ok();
    }
    kill(libc::SIGKILL);
    child.wait().await.ok()
}

/// Makes the wrapper inherit the processes Grok's commands orphan, which run in sessions of their own.
pub fn adopt_orphans() {
    #[cfg(target_os = "linux")]
    // SAFETY: PR_SET_CHILD_SUBREAPER only changes which process orphans are reparented to.
    unsafe {
        libc::prctl(libc::PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0);
    }
}

#[cfg(target_os = "linux")]
async fn sweep() {
    let wrapper = std::process::id() as i32;
    for _ in 0..40 {
        let left = descendants(wrapper);
        if left.is_empty() {
            return;
        }
        for pid in left {
            // SAFETY: every pid here descends from the wrapper, which adopted it as subreaper.
            unsafe { libc::kill(pid, libc::SIGKILL) };
        }
        // SAFETY: Grok is already reaped, so this only collects the adopted orphans.
        while unsafe { libc::waitpid(-1, std::ptr::null_mut(), libc::WNOHANG) } > 0 {}
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

#[cfg(not(target_os = "linux"))]
async fn sweep() {}

#[cfg(target_os = "linux")]
fn descendants(root: i32) -> Vec<i32> {
    let mut children: std::collections::HashMap<i32, Vec<i32>> = Default::default();
    for entry in std::fs::read_dir("/proc").into_iter().flatten().flatten() {
        let Some(pid) = entry
            .file_name()
            .to_str()
            .and_then(|name| name.parse::<i32>().ok())
        else {
            continue;
        };
        let Ok(stat) = std::fs::read_to_string(entry.path().join("stat")) else {
            continue;
        };
        let mut fields = stat
            .rsplit_once(')')
            .map(|(_, rest)| rest)
            .unwrap_or("")
            .split_whitespace();
        let (Some(state), Some(parent)) = (
            fields.next(),
            fields.next().and_then(|ppid| ppid.parse().ok()),
        ) else {
            continue;
        };
        if state != "Z" {
            children.entry(parent).or_default().push(pid);
        }
    }
    let mut found = Vec::new();
    let mut queue = vec![root];
    while let Some(pid) = queue.pop() {
        for child in children.remove(&pid).unwrap_or_default() {
            found.push(child);
            queue.push(child);
        }
    }
    found
}

pub fn exit_of(status: ExitStatus) -> GrokExit {
    use std::os::unix::process::ExitStatusExt;
    match status.code() {
        Some(code) => GrokExit::Code(code),
        None => GrokExit::Signal(status.signal().unwrap_or(0)),
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Printed {
    Result { text: String },
    Error { message: String },
    Nothing,
}

pub fn printed(stdout: &[u8]) -> Printed {
    let text = String::from_utf8_lossy(stdout);
    let last_line = text
        .lines()
        .rev()
        .find(|line| !line.trim().is_empty())
        .unwrap_or("");
    for candidate in [text.trim(), last_line.trim()] {
        let Ok(Value::Object(object)) = serde_json::from_str::<Value>(candidate) else {
            continue;
        };
        if object.get("type").and_then(Value::as_str) == Some("error") {
            let message = object
                .get("message")
                .and_then(Value::as_str)
                .unwrap_or_default();
            return Printed::Error {
                message: message.to_owned(),
            };
        }
        if object.contains_key("stopReason") {
            let text = object
                .get("text")
                .and_then(Value::as_str)
                .unwrap_or_default();
            return Printed::Result {
                text: text.to_owned(),
            };
        }
    }
    Printed::Nothing
}

pub fn outcome(
    ended: Option<Ended>,
    exit: Option<GrokExit>,
    printed: Printed,
    record: &Record,
) -> (Outcome, String) {
    match (ended, printed) {
        (Some(Ended::Stopped(stop)), _) => (Outcome::Cancelled(stop), String::new()),
        (Some(Ended::Unreachable(detail)), _) => (
            Outcome::EndpointUnavailable(Unavailable::Unreachable { detail }),
            String::new(),
        ),
        (None, Printed::Result { text }) if exit == Some(GrokExit::Code(0)) => {
            (Outcome::Completed, text)
        }
        (None, Printed::Result { text }) if exit == Some(GrokExit::Code(1)) => {
            (Outcome::MaxTurns, text)
        }
        (None, printed) => {
            let message = match printed {
                Printed::Error { message } => message,
                _ => format!("Grok exited with {} and printed no result", describe(exit)),
            };
            let outcome = if record.model_calls == 0 {
                Outcome::Infrastructure(Cause::GrokStartup { message })
            } else if !record.answered {
                Outcome::EndpointUnavailable(Unavailable::Refused {
                    status: record.last_status,
                    message,
                })
            } else {
                Outcome::Error(GrokFailure::Grok { message })
            };
            (outcome, String::new())
        }
    }
}

fn describe(exit: Option<GrokExit>) -> String {
    match exit {
        Some(GrokExit::Code(code)) => format!("code {code}"),
        Some(GrokExit::Signal(signal)) => format!("signal {signal}"),
        None => "an unknown status".to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn answered() -> Record {
        Record {
            model_calls: 2,
            answered: true,
            last_status: Some(200),
            ..Record::default()
        }
    }

    #[test]
    fn grok_output_maps_to_the_documented_outcomes() {
        let result = || Printed::Result {
            text: "done".into(),
        };
        let error = || Printed::Error {
            message: "boom".into(),
        };
        let code = |code| Some(GrokExit::Code(code));
        let cases = [
            (
                outcome(None, code(0), result(), &answered()),
                "completed",
                "done",
            ),
            (
                outcome(None, code(1), result(), &answered()),
                "max_turns",
                "done",
            ),
            (outcome(None, code(1), error(), &answered()), "error", ""),
            (
                outcome(None, code(1), error(), &Record::default()),
                "infrastructure",
                "",
            ),
            (
                outcome(
                    None,
                    code(1),
                    error(),
                    &Record {
                        model_calls: 3,
                        last_status: Some(401),
                        ..Record::default()
                    },
                ),
                "endpoint_unavailable",
                "",
            ),
            (
                outcome(
                    Some(Ended::Stopped(Stop::Deadline)),
                    code(143),
                    Printed::Nothing,
                    &answered(),
                ),
                "cancelled",
                "",
            ),
            (
                outcome(
                    Some(Ended::Unreachable("refused".into())),
                    code(143),
                    Printed::Nothing,
                    &Record::default(),
                ),
                "endpoint_unavailable",
                "",
            ),
            (
                outcome(
                    None,
                    Some(GrokExit::Signal(9)),
                    Printed::Nothing,
                    &answered(),
                ),
                "error",
                "",
            ),
        ];
        for ((outcome, answer), name, expected) in cases {
            let value = serde_json::to_value(&outcome).unwrap();
            assert_eq!(value["outcome"], name, "{value}");
            assert_eq!(answer, expected, "{value}");
        }
    }

    #[test]
    fn grok_failure_text_is_carried_in_the_cause() {
        let (outcome, _) = outcome(
            None,
            Some(GrokExit::Code(1)),
            Printed::Error {
                message: "Model returned an error".into(),
            },
            &answered(),
        );
        assert_eq!(
            serde_json::to_value(outcome).unwrap()["cause"],
            serde_json::json!({"kind": "grok", "message": "Model returned an error"})
        );
    }

    #[test]
    fn printed_output_is_read_from_the_pretty_result_or_the_error_line() {
        let pretty =
            b"{\n  \"text\": \"hi\",\n  \"stopReason\": \"end_turn\",\n  \"sessionId\": \"s\"\n}\n";
        assert_eq!(printed(pretty), Printed::Result { text: "hi".into() });
        let error = b"{\"type\":\"error\",\"message\":\"no model\"}\n";
        assert_eq!(
            printed(error),
            Printed::Error {
                message: "no model".into()
            }
        );
        assert_eq!(printed(b"panicked at main.rs\n"), Printed::Nothing);
    }

    #[test]
    fn the_session_id_is_a_function_of_the_session_dir() {
        assert_eq!(
            session_id(Path::new("/logs/remanie")),
            session_id(Path::new("/logs/remanie"))
        );
        assert_ne!(
            session_id(Path::new("/logs/a")),
            session_id(Path::new("/logs/b"))
        );
        assert_eq!(
            session_id(Path::new("/tmp/remanie-oracle/live/session")),
            "905d05b5-c66b-5973-b449-a26532b8791c"
        );
    }
}
