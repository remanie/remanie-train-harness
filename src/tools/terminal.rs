// Adapted from Grok Build 2bdd1d6a for this standalone runner; see NOTICE.
use std::collections::BTreeMap;
use std::io;
use std::os::unix::process::ExitStatusExt;
use std::path::{Path, PathBuf};
use std::process::{ExitStatus, Stdio};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant, SystemTime};

use anyhow::{Context, Result, bail, ensure};
use serde_json::{Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::process::{Child, Command};
use tokio::sync::watch;

const PREVIEW_BYTES: usize = 20_000;
const POLL_BYTES: usize = 40_000;
const MAX_LOG_BYTES: usize = 16 * 1024 * 1024;
const MAX_RUNTIME: Duration = Duration::from_secs(36_000);
const REAP_TIMEOUT: Duration = Duration::from_secs(5);
const PATH: &str = "/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin";
const PAGER_ENV: [(&str, &str); 10] = [
    ("PAGER", "cat"),
    ("GIT_PAGER", "cat"),
    ("GH_PAGER", "cat"),
    ("MANPAGER", "cat"),
    ("AWS_PAGER", ""),
    ("SYSTEMD_PAGER", "cat"),
    ("GIT_EDITOR", "true"),
    ("GIT_SEQUENCE_EDITOR", "true"),
    ("GIT_TERMINAL_PROMPT", "0"),
    ("GPG_TTY", ""),
];

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

struct State {
    closed: bool,
    next_id: u64,
    tasks: BTreeMap<String, Arc<Task>>,
}

pub struct TerminalSession {
    cwd: PathBuf,
    session_dir: PathBuf,
    state: Mutex<State>,
}

impl TerminalSession {
    pub async fn new(cwd: PathBuf, session_dir: PathBuf) -> Result<Self> {
        for name in ["home", "tmp", "terminal"] {
            tokio::fs::create_dir(session_dir.join(name))
                .await
                .with_context(|| format!("create attempt {name}"))?;
        }
        Ok(Self {
            cwd,
            session_dir,
            state: Mutex::new(State {
                closed: false,
                next_id: 0,
                tasks: BTreeMap::new(),
            }),
        })
    }

    pub async fn call(&self, name: &str, args: &Value, call_id: &str) -> Result<String> {
        ensure!(!lock(&self.state).closed, "tool session has ended");
        match name {
            "run_terminal_cmd" => self.run(args, call_id).await,
            "get_terminal_command_output" => self.output(args).await,
            "kill_terminal_command" => {
                let id = args["task_id"].as_str().context("missing task_id")?;
                let task = lock(&self.state).tasks.get(id).cloned();
                let Some(task) = task else {
                    return Ok(format!("Task {id} not found."));
                };
                let running = task.stop("cancelled", true);
                if running {
                    tokio::time::timeout(REAP_TIMEOUT, task.wait())
                        .await
                        .context("timed out reaping cancelled command")?;
                }
                Ok(if running {
                    format!("killed: Task {id} was terminated.")
                } else {
                    format!("already_exited: Task {id} has already exited.")
                })
            }
            _ => bail!("unknown terminal tool: {name}"),
        }
    }

    async fn run(&self, args: &Value, call_id: &str) -> Result<String> {
        let started = tokio::time::Instant::now();
        let command = args["command"].as_str().context("missing command")?;
        let description = args["description"]
            .as_str()
            .context("missing description")?;
        let block = args["block_until_ms"].as_u64().unwrap_or(30_000);
        ensure!(block <= 300_000, "block_until_ms exceeds 300000");
        ensure!(
            block == 0 || !unwaited_background(command),
            "Do not use an unwaited background '&'; use block_until_ms=0 instead."
        );
        let log_path = {
            let mut state = lock(&self.state);
            ensure!(!state.closed, "tool session has ended");
            state.next_id += 1;
            let local = format!("terminal-{}", state.next_id);
            self.session_dir
                .join("terminal")
                .join(format!("{local}.log"))
        };
        let log = tokio::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&log_path)
            .await
            .context("create command output log")?;
        let id = call_id.to_owned();
        let mut cmd = clean_command(Path::new("/bin/bash"), &self.cwd);
        cmd.args(["--noprofile", "--norc", "-c", command])
            .env("HOME", self.session_dir.join("home"))
            .env("TMPDIR", self.session_dir.join("tmp"))
            .env("TERM", "dumb")
            .env("NO_COLOR", "1")
            .env("FORCE_COLOR", "0")
            .envs(PAGER_ENV);
        if block == 0 {
            cmd.env("PYTHONUNBUFFERED", "1");
        }
        let task = {
            let mut state = lock(&self.state);
            ensure!(!state.closed, "tool session has ended");
            ensure!(!state.tasks.contains_key(&id), "duplicate terminal task id");
            let task = spawn(
                cmd,
                TaskState::new(
                    id.clone(),
                    command,
                    description,
                    &self.cwd,
                    Some(log_path),
                    POLL_BYTES,
                ),
                Some(log),
                MAX_LOG_BYTES,
                MAX_RUNTIME,
            )?;
            state.tasks.insert(id.clone(), task.clone());
            task
        };
        let mut abort = AbortOnDrop(Some(task.clone()));
        let finished = block > 0
            && tokio::time::timeout_at(started + Duration::from_millis(block), task.wait())
                .await
                .is_ok();
        abort.0 = None;
        if finished {
            lock(&self.state).tasks.remove(&id);
            let state = lock(&task.state);
            state.check_error()?;
            Ok(state.foreground())
        } else {
            let mut state = lock(&task.state);
            state.background = true;
            Ok(format!(
                "<task-id>{id}</task-id>\n<task-type>bash</task-type>\n<output-file>{}</output-file>\n<status>running</status>\n<summary>Background task {id} started</summary>\nUse get_terminal_command_output with task_ids=[\"{id}\"] when you need the output.",
                state.output_file.as_ref().unwrap().display()
            ))
        }
    }

    async fn output(&self, args: &Value) -> Result<String> {
        let ids = args["task_ids"].as_array().context("missing task_ids")?;
        ensure!(
            !ids.is_empty() && ids.len() <= 20,
            "task_ids must have 1–20 entries"
        );
        let timeout = args["timeout_ms"].as_u64().unwrap_or(0);
        ensure!(timeout <= 300_000, "timeout_ms exceeds 300000");
        let tasks = {
            let state = lock(&self.state);
            ids.iter()
                .map(|id| {
                    let id = id.as_str().context("task_id must be a string")?;
                    Ok((id.to_owned(), state.tasks.get(id).cloned()))
                })
                .collect::<Result<Vec<_>>>()?
        };
        let deadline = tokio::time::Instant::now() + Duration::from_millis(timeout);
        if timeout > 0 {
            for (_, task) in &tasks {
                if let Some(task) = task {
                    let _ = tokio::time::timeout_at(deadline, task.wait()).await;
                }
            }
        }
        let multiple = tasks.len() > 1;
        let mode = if timeout > 0 { "wait_all" } else { "poll" };
        let mut output = if multiple {
            format!("=== Multi-wait ({mode}) ===\n")
        } else {
            String::new()
        };
        let mut completed = 0;
        for (id, task) in &tasks {
            let Some(task) = task else {
                output.push_str(&format!("Task {id} not found.\n"));
                completed += 1;
                continue;
            };
            let state = lock(&task.state);
            state.check_error()?;
            completed += usize::from(state.completed);
            output.push_str(&state.poll(multiple));
            output.push('\n');
        }
        if multiple {
            output.push_str(&format!(
                "\n{completed}/{} tasks completed ({mode})",
                tasks.len()
            ));
        }
        Ok(output)
    }

    pub async fn snapshot(&self) -> Value {
        let state = lock(&self.state);
        Value::Array(state.tasks.values().filter_map(|task| {
            let state = lock(&task.state);
            state.background.then(|| json!({
                "task_id": state.id, "command": state.command, "cwd": state.cwd,
                "description": state.description, "start_time": state.started,
                "end_time": state.ended, "output": "", "output_file": state.output_file,
                "output_total_bytes": state.total, "truncated": state.total > state.output.len(),
                "exit_code": state.exit_code, "signal": state.signal, "completed": state.completed,
                "kind": "bash", "is_backgrounded": true, "explicitly_killed": state.explicitly_killed,
            }))
        }).collect())
    }

    pub async fn shutdown(&self) {
        let tasks = self.close();
        let deadline = tokio::time::Instant::now() + REAP_TIMEOUT;
        for task in tasks {
            let _ = tokio::time::timeout_at(deadline, task.wait()).await;
        }
    }

    fn close(&self) -> Vec<Arc<Task>> {
        let mut state = lock(&self.state);
        state.closed = true;
        let tasks = state.tasks.values().cloned().collect::<Vec<_>>();
        for task in &tasks {
            task.stop("cancelled", false);
        }
        tasks
    }
}

impl Drop for TerminalSession {
    fn drop(&mut self) {
        self.close();
    }
}

struct Task {
    state: Mutex<TaskState>,
    finished: watch::Sender<bool>,
}

struct TaskState {
    id: String,
    command: String,
    description: Option<String>,
    cwd: PathBuf,
    started: SystemTime,
    clock: Instant,
    ended: Option<SystemTime>,
    output_file: Option<PathBuf>,
    output: Vec<u8>,
    output_limit: usize,
    total: usize,
    pid: Option<libc::pid_t>,
    stop_deadline: Option<Instant>,
    completed: bool,
    background: bool,
    explicitly_killed: bool,
    exit_code: Option<i32>,
    signal: Option<String>,
    error: Option<String>,
}

impl TaskState {
    fn new(
        id: String,
        command: &str,
        description: &str,
        cwd: &Path,
        output_file: Option<PathBuf>,
        output_limit: usize,
    ) -> Self {
        Self {
            id,
            command: command.into(),
            description: (!description.trim().is_empty()).then(|| description.into()),
            cwd: cwd.into(),
            started: SystemTime::now(),
            clock: Instant::now(),
            ended: None,
            output_file,
            output: Vec::new(),
            output_limit,
            total: 0,
            pid: None,
            stop_deadline: None,
            completed: false,
            background: false,
            explicitly_killed: false,
            exit_code: None,
            signal: None,
            error: None,
        }
    }

    fn check_error(&self) -> Result<()> {
        if let Some(error) = &self.error {
            bail!("terminal command failed: {error}");
        }
        Ok(())
    }

    fn duration(&self) -> f64 {
        self.ended
            .and_then(|end| end.duration_since(self.started).ok())
            .unwrap_or_else(|| self.clock.elapsed())
            .as_secs_f64()
    }

    fn text(&self, limit: usize) -> String {
        let output = if self.output.len() > limit {
            let half = limit / 2;
            [
                self.output[..half].as_ref(),
                b"\n[... output truncated ...]\n".as_slice(),
                &self.output[self.output.len() - half..],
            ]
            .concat()
        } else {
            self.output.clone()
        };
        strip_ansi(&String::from_utf8_lossy(&output))
    }

    fn foreground(&self) -> String {
        let code = self
            .signal
            .as_ref()
            .map(|s| format!("killed ({s})"))
            .unwrap_or_else(|| self.exit_code.unwrap_or(-1).to_string());
        let note = if self.total > PREVIEW_BYTES {
            format!(
                " [truncated - full output at: {}]",
                self.output_file.as_ref().unwrap().display()
            )
        } else {
            String::new()
        };
        format!(
            "Exit code: {code}{note}\n\nCommand output:\n\n```\n{}\n```\n\nCommand completed.\n\nThe previous shell command ended, so on the next invocation of this tool, you will be using a new shell session.\n\nOn the next terminal tool call, the directory of the shell will be {}.",
            self.text(PREVIEW_BYTES),
            self.cwd.display()
        )
    }

    fn status(&self) -> &str {
        if !self.completed {
            "running"
        } else if self.explicitly_killed {
            "cancelled"
        } else if self.signal.as_deref() == Some("timeout") {
            "timed_out"
        } else if self.exit_code == Some(0) {
            "completed"
        } else {
            "failed"
        }
    }

    fn poll(&self, multiple: bool) -> String {
        let mut text = if multiple {
            format!(
                "--- Task {} [{}] ---\nCommand: {}\nDuration: {:.2}s",
                self.id,
                self.status(),
                self.command,
                self.duration()
            )
        } else {
            format!(
                "=== Task {} ===\nCommand: {}\nStatus: {}\nDuration: {:.2}s",
                self.id,
                self.command,
                self.status(),
                self.duration()
            )
        };
        if let Some(code) = self.exit_code {
            text.push_str(&format!("\nExit Code: {code}"));
        }
        if let Some(path) = &self.output_file {
            text.push_str(&format!("\nOutput File: {}", path.display()));
        }
        text.push_str("\n\n=== Output ===\n");
        text.push_str(if self.output.is_empty() {
            if self.completed {
                "(no output)"
            } else {
                "(no output yet)"
            }
        } else {
            ""
        });
        text.push_str(&self.text(POLL_BYTES));
        if self.total > self.output.len() {
            text.push_str("\n[truncated - use read_file on output_file for full content]");
        }
        text
    }

    fn append(&mut self, bytes: &[u8]) {
        self.total = self.total.saturating_add(bytes.len());
        if self.output_file.is_none() {
            let keep = bytes
                .len()
                .min(self.output_limit.saturating_sub(self.output.len()));
            self.output.extend_from_slice(&bytes[..keep]);
        } else {
            self.output.extend_from_slice(bytes);
            if self.output.len() > self.output_limit {
                let half = self.output_limit / 2;
                self.output.drain(half..self.output.len() - half);
            }
        }
    }
}

impl Task {
    fn stop(&self, reason: &str, graceful: bool) -> bool {
        let mut state = lock(&self.state);
        let Some(pid) = state.pid else {
            return false;
        };
        state.explicitly_killed |= reason == "cancelled";
        if state.signal.is_none() {
            state.signal = Some(reason.into());
        }
        signal_group(
            pid,
            if graceful {
                libc::SIGTERM
            } else {
                libc::SIGKILL
            },
        );
        state.stop_deadline = Some(
            Instant::now()
                + if graceful {
                    Duration::from_secs(1)
                } else {
                    Duration::ZERO
                },
        );
        true
    }

    async fn wait(&self) {
        let mut receiver = self.finished.subscribe();
        while !*receiver.borrow_and_update() {
            if receiver.changed().await.is_err() {
                break;
            }
        }
    }
}

struct AbortOnDrop(Option<Arc<Task>>);
impl Drop for AbortOnDrop {
    fn drop(&mut self) {
        if let Some(task) = &self.0 {
            task.stop("cancelled", false);
        }
    }
}

fn clean_command(program: &Path, cwd: &Path) -> Command {
    let mut command = Command::new(program);
    command
        .current_dir(cwd)
        .env_clear()
        .env("PATH", PATH)
        .env("LANG", "C")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    // SAFETY: setsid is async-signal-safe and only runs in the child before exec.
    unsafe {
        command.pre_exec(|| {
            if libc::setsid() < 0 {
                Err(io::Error::last_os_error())
            } else {
                Ok(())
            }
        });
    }
    command
}

fn signal_group(pid: libc::pid_t, signal: libc::c_int) {
    unsafe {
        libc::kill(-pid, signal);
    }
}

fn observe_exit(pid: libc::pid_t) -> io::Result<bool> {
    // WNOWAIT reserves the leader PID until residual descendants are killed and try_wait reaps it.
    unsafe {
        let mut info: libc::siginfo_t = std::mem::zeroed();
        if libc::waitid(
            libc::P_PID,
            pid as libc::id_t,
            &mut info,
            libc::WEXITED | libc::WNOHANG | libc::WNOWAIT,
        ) < 0
        {
            return Err(io::Error::last_os_error());
        }
        #[cfg(target_os = "linux")]
        let child_pid = info.si_pid();
        #[cfg(target_os = "macos")]
        let child_pid = info.si_pid;
        Ok(child_pid != 0)
    }
}

fn spawn(
    mut command: Command,
    mut state: TaskState,
    log: Option<tokio::fs::File>,
    log_limit: usize,
    timeout: Duration,
) -> Result<Arc<Task>> {
    let child = command.spawn().context("spawn command")?;
    state.pid = child.id().map(|pid| pid as libc::pid_t);
    let (finished, _) = watch::channel(false);
    let task = Arc::new(Task {
        state: Mutex::new(state),
        finished,
    });
    tokio::spawn(drive(child, task.clone(), log, log_limit, timeout));
    Ok(task)
}

async fn drive(
    mut child: Child,
    task: Arc<Task>,
    mut log: Option<tokio::fs::File>,
    log_limit: usize,
    timeout: Duration,
) {
    let mut guard = AbortOnDrop(Some(task.clone()));
    let mut stdout = child.stdout.take().unwrap();
    let mut stderr = child.stderr.take().unwrap();
    let mut out_buffer = [0u8; 8192];
    let mut err_buffer = [0u8; 8192];
    let (mut out_done, mut err_done) = (false, false);
    let mut written = 0;
    let mut status: Option<ExitStatus> = None;
    let mut drain_until = None;
    let mut tick = tokio::time::interval(Duration::from_millis(10));
    let deadline = Instant::now() + timeout;
    loop {
        let read = tokio::select! {
            read = stdout.read(&mut out_buffer), if !out_done => Some((true, read)),
            read = stderr.read(&mut err_buffer), if !err_done => Some((false, read)),
            _ = tick.tick() => None,
        };
        if let Some((is_out, read)) = read {
            match read {
                Ok(0) => {
                    if is_out {
                        out_done = true
                    } else {
                        err_done = true
                    }
                }
                Ok(count) => {
                    let bytes = if is_out {
                        &out_buffer[..count]
                    } else {
                        &err_buffer[..count]
                    };
                    lock(&task.state).append(bytes);
                    if let Some(file) = &mut log {
                        let keep = count.min(log_limit.saturating_sub(written));
                        if let Err(error) = file.write_all(&bytes[..keep]).await {
                            lock(&task.state).error = Some(format!("write output log: {error}"));
                            task.stop("output_error", false);
                        }
                        written += keep;
                        if keep < count {
                            task.stop("output_limit", false);
                        }
                    }
                }
                Err(error) => {
                    lock(&task.state).error = Some(format!("read command output: {error}"));
                    if is_out {
                        out_done = true
                    } else {
                        err_done = true
                    };
                    task.stop("output_error", false);
                }
            }
        }
        let now = Instant::now();
        if status.is_none() {
            if now >= deadline {
                task.stop("timeout", false);
            }
            let mut state = lock(&task.state);
            if let Some(pid) = state.pid {
                if state.stop_deadline.is_some_and(|end| now >= end) {
                    signal_group(pid, libc::SIGKILL);
                }
                match observe_exit(pid) {
                    Ok(true) => {
                        signal_group(pid, libc::SIGKILL);
                        match child.try_wait() {
                            Ok(Some(exit)) => {
                                status = Some(exit);
                                state.pid = None;
                                drain_until = Some(now + Duration::from_secs(2));
                            }
                            Ok(None) => {}
                            Err(error) => {
                                state.pid = None;
                                state.error = Some(format!("reap command: {error}"));
                                break;
                            }
                        }
                    }
                    Ok(false) => {}
                    Err(error) => {
                        state.error = Some(format!("observe command exit: {error}"));
                        signal_group(pid, libc::SIGKILL);
                        break;
                    }
                }
            }
        }
        if status.is_some() && ((out_done && err_done) || drain_until.is_some_and(|end| now >= end))
        {
            break;
        }
    }
    if status.is_none() {
        task.stop("cancelled", false);
        let mut state = lock(&task.state);
        if let Ok(Some(exit)) = child.try_wait() {
            status = Some(exit);
        }
        state.pid = None;
    }
    if let Some(file) = &mut log
        && let Err(error) = file.flush().await
    {
        lock(&task.state).error = Some(format!("flush output log: {error}"));
    }
    {
        let mut state = lock(&task.state);
        state.completed = true;
        state.ended = Some(SystemTime::now());
        state.exit_code = status.as_ref().map(|s| s.code().unwrap_or(-1));
        if state.signal.is_none() {
            state.signal = status
                .and_then(|s| s.signal())
                .map(|s| format!("signal {s}"));
        }
    }
    guard.0 = None;
    task.finished.send_replace(true);
}

pub async fn capture(
    program: &Path,
    args: &[String],
    cwd: &Path,
    timeout: Duration,
    max_bytes: usize,
) -> Result<(i32, String)> {
    let mut command = clean_command(program, cwd);
    command.args(args);
    let task = spawn(
        command,
        TaskState::new(String::new(), "", "", cwd, None, max_bytes),
        None,
        0,
        timeout,
    )?;
    let _abort = AbortOnDrop(Some(task.clone()));
    task.wait().await;
    let state = lock(&task.state);
    state.check_error()?;
    let mut output = String::from_utf8_lossy(&state.output).into_owned();
    if state.total > state.output.len() {
        output.push_str("\n[output truncated]");
    }
    if let Some(reason) = &state.signal {
        output.push_str(&format!("\n[command ended: {reason}]"));
    }
    Ok((state.exit_code.unwrap_or(-1), output))
}

fn strip_ansi(text: &str) -> String {
    let mut chars = text.chars().peekable();
    let mut output = String::with_capacity(text.len());
    while let Some(ch) = chars.next() {
        if ch != '\u{1b}' {
            output.push(ch);
            continue;
        }
        match chars.next() {
            Some('[') => {
                for next in chars.by_ref() {
                    if ('@'..='~').contains(&next) {
                        break;
                    }
                }
            }
            Some(']') => {
                while let Some(next) = chars.next() {
                    if next == '\u{7}' || (next == '\u{1b}' && chars.next() == Some('\\')) {
                        break;
                    }
                }
            }
            _ => {}
        }
    }
    output
}

fn ends_with_wait_builtin(command: &str) -> bool {
    let trimmed = command
        .trim_end()
        .trim_end_matches(';')
        .trim_end_matches('\n')
        .trim_end();
    let Some(prefix) = trimmed.strip_suffix("wait") else {
        return false;
    };
    let separated = prefix.ends_with([' ', '\t']);
    let prefix = prefix.trim_end_matches([' ', '\t']);
    let prefix = if separated {
        prefix
            .strip_suffix("builtin")
            .or_else(|| prefix.strip_suffix("command"))
            .unwrap_or(prefix)
            .trim_end_matches([' ', '\t'])
    } else {
        prefix
    };
    prefix.is_empty()
        || prefix.ends_with([';', '\n'])
        || (prefix.ends_with('&') && !prefix.ends_with("&&"))
}

fn parse_heredoc_start(chars: &[char], start: usize) -> Option<(String, bool, usize)> {
    let mut i = start + 2;

    let strip_tabs = chars.get(i).copied() == Some('-');
    if strip_tabs {
        i += 1;
    }

    while matches!(chars.get(i).copied(), Some(' ' | '\t')) {
        i += 1;
    }

    match chars.get(i).copied() {
        None | Some('\n') => None,
        Some('\'') => {
            i += 1;
            let d_start = i;
            while chars.get(i).is_some_and(|c| *c != '\'') {
                i += 1;
            }
            chars.get(i)?;
            let delimiter: String = chars.get(d_start..i)?.iter().collect();
            i += 1;
            if delimiter.is_empty() {
                return None;
            }
            Some((delimiter, strip_tabs, i))
        }
        Some('"') => {
            i += 1;
            let d_start = i;
            while chars.get(i).is_some_and(|c| *c != '"') {
                i += 1;
            }
            chars.get(i)?;
            let delimiter: String = chars.get(d_start..i)?.iter().collect();
            i += 1;
            if delimiter.is_empty() {
                return None;
            }
            Some((delimiter, strip_tabs, i))
        }
        Some(_) => {
            let d_start = i;
            while chars.get(i).is_some_and(|c| {
                !c.is_whitespace() && !matches!(*c, ';' | '&' | '|' | '(' | ')' | '<' | '>')
            }) {
                i += 1;
            }
            if i == d_start {
                return None;
            }
            let delimiter: String = chars
                .get(d_start..i)?
                .iter()
                .filter(|&&c| c != '\\')
                .collect();
            if delimiter.is_empty() {
                return None;
            }
            Some((delimiter, strip_tabs, i))
        }
    }
}

fn skip_heredoc_body(chars: &[char], start: usize, delimiter: &str, strip_tabs: bool) -> usize {
    let len = chars.len();
    let mut i = start;

    while i < len {
        let line_start = i;
        while chars.get(i).is_some_and(|c| *c != '\n') {
            i += 1;
        }
        let line: String = chars.get(line_start..i).unwrap_or(&[]).iter().collect();

        let check = if strip_tabs {
            line.trim_start_matches('\t')
        } else {
            &line
        };

        if check == delimiter {
            if i < len {
                i += 1;
            }
            return i;
        }

        if i < len {
            i += 1;
        }
    }

    len
}

fn unwaited_background(command: &str) -> bool {
    let mut chars: Vec<char> = command.chars().collect();
    let len = chars.len();
    let mut i = 0;
    let mut in_single_quote = false;
    let mut in_double_quote = false;
    let mut word_start = true;
    let mut background = false;
    let mut pending_heredocs: Vec<(String, bool)> = Vec::new();

    while i < len {
        let Some(&ch) = chars.get(i) else {
            break;
        };

        if ch == '\\' && !in_single_quote {
            if chars.get(i + 1) != Some(&'\n') {
                word_start = false;
            }
            i += 2;
            continue;
        }

        if ch == '\'' && !in_double_quote {
            in_single_quote = !in_single_quote;
            word_start = false;
            i += 1;
            continue;
        }
        if ch == '"' && !in_single_quote {
            in_double_quote = !in_double_quote;
            word_start = false;
            i += 1;
            continue;
        }

        if !in_single_quote && !in_double_quote {
            if ch == '#' && word_start {
                while i < len && chars[i] != '\n' {
                    chars[i] = ' ';
                    i += 1;
                }
                continue;
            }
            if ch == '(' && chars.get(i + 1) == Some(&'(') {
                word_start = i == 0 || chars[i - 1] != '$';
                i += 2;
                let mut depth = 2;
                while i < len && depth > 0 {
                    match chars[i] {
                        '(' => depth += 1,
                        ')' => depth -= 1,
                        _ => {}
                    }
                    i += 1;
                }
                continue;
            }
            word_start = matches!(
                ch,
                ' ' | '\t' | '\n' | ';' | '|' | '&' | '(' | ')' | '<' | '>'
            );
        }

        if ch == '\n' && !in_single_quote && !in_double_quote && !pending_heredocs.is_empty() {
            i += 1;
            for (delim, strip_tabs) in pending_heredocs.drain(..) {
                let start = i;
                i = skip_heredoc_body(&chars, i, &delim, strip_tabs);
                chars[start..i].fill(' ');
            }
            continue;
        }

        if ch == '<'
            && !in_single_quote
            && !in_double_quote
            && chars.get(i + 1).copied() == Some('<')
            && chars.get(i + 2).copied() != Some('<')
            && let Some((delim, strip_tabs, after)) = parse_heredoc_start(&chars, i)
        {
            pending_heredocs.push((delim, strip_tabs));
            i = after;
            continue;
        }

        if ch == '&' && !in_single_quote && !in_double_quote {
            if chars.get(i + 1).copied() == Some('&') {
                i += 2;
                continue;
            }

            if chars.get(i + 1).copied() == Some('>') {
                i += 2;
                if chars.get(i).copied() == Some('>') {
                    i += 1;
                }
                continue;
            }

            if i.checked_sub(1)
                .and_then(|j| chars.get(j).copied())
                .is_some_and(|prev| prev == '>' || prev == '<')
            {
                i += 1;
                continue;
            }

            background = true;
        }

        i += 1;
    }

    background && !ends_with_wait_builtin(&chars.into_iter().collect::<String>())
}

#[cfg(test)]
#[path = "terminal_tests.rs"]
mod tests;
