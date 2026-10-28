// Adapted from Grok Build 2bdd1d6a for this standalone runner; see NOTICE.
use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom};
use std::os::unix::process::ExitStatusExt;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant, SystemTime};

use anyhow::{Context, Result, ensure};
use serde::Serialize;
use serde_json::{Value, json};
use tokio::io::{AsyncRead, AsyncReadExt};
use tokio::process::{Child, Command};
use tokio::sync::watch;

use super::internal;

const PREVIEW_BYTES: usize = 20_000;
const POLL_BYTES: usize = 40_000;
const MAX_RUNTIME: Duration = Duration::from_secs(36_000);
const REAP_TIMEOUT: Duration = Duration::from_secs(5);
const TRUNCATED: &str = "\n[... output truncated ...]\n";
const NON_INTERACTIVE_ENV: [(&str, &str); 13] = [
    ("TERM", "dumb"),
    ("NO_COLOR", "1"),
    ("FORCE_COLOR", "0"),
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

static SPAWNED: Mutex<BTreeSet<libc::pid_t>> = Mutex::new(BTreeSet::new());

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct TaskId(u32);

impl std::fmt::Display for TaskId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "task-{}", self.0)
    }
}

impl Serialize for TaskId {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_str(self)
    }
}

impl TaskId {
    fn parse(id: &str) -> Option<Self> {
        id.strip_prefix("task-")?.parse().ok().map(Self)
    }
}

#[derive(Debug, Serialize)]
pub struct LeftRunning {
    pub task_id: TaskId,
    pub pid: i32,
    pub command: String,
    pub output_file: PathBuf,
}

struct State {
    next_id: u32,
    tasks: BTreeMap<TaskId, Arc<Task>>,
}

pub struct TerminalSession {
    cwd: PathBuf,
    log_dir: PathBuf,
    state: Mutex<State>,
}

impl TerminalSession {
    pub async fn new(cwd: PathBuf, log_root: PathBuf) -> Result<Self> {
        let log_dir = fresh_dir(&log_root).context("create command log directory")?;
        #[cfg(target_os = "linux")]
        adopt_orphans().context("become the subreaper of command descendants")?;
        Ok(Self {
            cwd,
            log_dir,
            state: Mutex::new(State {
                next_id: 0,
                tasks: BTreeMap::new(),
            }),
        })
    }

    fn task(&self, id: &str) -> Option<Arc<Task>> {
        TaskId::parse(id).and_then(|id| lock(&self.state).tasks.get(&id).cloned())
    }

    pub async fn kill(&self, args: &Value) -> Result<String> {
        let id = args["task_id"].as_str().context("missing task_id")?;
        let Some(task) = self.task(id) else {
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

    pub async fn run(&self, args: &Value) -> Result<(String, TaskId)> {
        let started = tokio::time::Instant::now();
        let command = args["command"].as_str().context("missing command")?;
        let description = args["description"]
            .as_str()
            .context("missing description")?;
        let block = args["block_until_ms"].as_u64().unwrap_or(30_000);
        ensure!(block <= 300_000, "block_until_ms exceeds 300000");
        ensure!(
            !unwaited_background(command),
            "To run a command in the background, pass block_until_ms=0 instead of using &."
        );
        let id = {
            let mut state = lock(&self.state);
            state.next_id += 1;
            TaskId(state.next_id)
        };
        let log_path = self.log_dir.join(format!("{id}.log"));
        let log = create_log(&self.log_dir, &log_path)
            .map_err(|error| internal(format!("create command output log: {error}")))?;
        let mut shell = Command::new("/bin/bash");
        shell
            .args(["--noprofile", "--norc", "-c", command])
            .current_dir(&self.cwd)
            .envs(NON_INTERACTIVE_ENV)
            .stdin(Stdio::null())
            .stdout(log.try_clone().map_err(internal)?)
            .stderr(log.try_clone().map_err(internal)?);
        // SAFETY: setsid is async-signal-safe and only runs in the child before exec.
        unsafe {
            shell.pre_exec(|| {
                if libc::setsid() < 0 {
                    Err(io::Error::last_os_error())
                } else {
                    Ok(())
                }
            });
        }
        let (child, pid) = spawn_tracked(&mut shell).context("spawn command")?;
        let task = Arc::new(Task {
            id,
            pid,
            command: command.into(),
            description: (!description.trim().is_empty()).then(|| description.into()),
            cwd: self.cwd.clone(),
            log: log_path,
            started: SystemTime::now(),
            clock: Instant::now(),
            state: Mutex::default(),
            finished: watch::channel(false).0,
        });
        lock(&self.state).tasks.insert(id, task.clone());
        tokio::spawn(drive(child, task.clone()));
        let mut abort = AbortOnDrop(Some(task.clone()));
        let finished = block > 0
            && tokio::time::timeout_at(started + Duration::from_millis(block), task.wait())
                .await
                .is_ok();
        abort.0 = None;
        if finished {
            lock(&self.state).tasks.remove(&id);
            task.check_error()?;
            Ok((task.foreground()?, id))
        } else {
            lock(&task.state).background = true;
            Ok((
                format!(
                    "<task-id>{id}</task-id>\n<task-type>bash</task-type>\n<output-file>{}</output-file>\n<status>running</status>\n<summary>Background task {id} started</summary>\nUse get_terminal_command_output with task_ids=[\"{id}\"] when you need the output.",
                    task.log.display()
                ),
                id,
            ))
        }
    }

    pub async fn output(&self, args: &Value) -> Result<String> {
        let ids = args["task_ids"].as_array().context("missing task_ids")?;
        ensure!(
            !ids.is_empty() && ids.len() <= 20,
            "task_ids must have 1–20 entries"
        );
        let timeout = args["timeout_ms"].as_u64().unwrap_or(0);
        ensure!(timeout <= 300_000, "timeout_ms exceeds 300000");
        let tasks = ids
            .iter()
            .map(|id| {
                let id = id.as_str().context("task_id must be a string")?;
                Ok((id.to_owned(), self.task(id)))
            })
            .collect::<Result<Vec<_>>>()?;
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
            task.check_error()?;
            completed += usize::from(lock(&task.state).completed);
            output.push_str(&task.poll(multiple)?);
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

    pub fn background_tasks(&self) -> Value {
        let state = lock(&self.state);
        Value::Array(state.tasks.values().filter_map(|task| {
            let state = lock(&task.state);
            state.background.then(|| json!({
                "task_id": task.id, "command": task.command, "cwd": task.cwd,
                "description": task.description, "start_time": task.started,
                "end_time": state.ended, "output_file": task.log,
                "output_total_bytes": std::fs::metadata(&task.log).map_or(0, |metadata| metadata.len()),
                "exit_code": state.exit_code, "signal": state.signal,
                "completed": state.completed, "kind": "bash", "is_backgrounded": true,
                "explicitly_killed": state.explicitly_killed,
            }))
        }).collect())
    }

    pub fn detach(self) -> Vec<LeftRunning> {
        lock(&self.state)
            .tasks
            .values()
            .filter_map(|task| {
                let state = lock(&task.state);
                (state.background && !state.completed && state.signal.is_none()).then(|| {
                    LeftRunning {
                        task_id: task.id,
                        pid: task.pid,
                        command: task.command.clone(),
                        output_file: task.log.clone(),
                    }
                })
            })
            .collect()
    }

    pub async fn kill_all(self) {
        let tasks = lock(&self.state)
            .tasks
            .values()
            .cloned()
            .collect::<Vec<_>>();
        for task in &tasks {
            task.stop("cancelled", false);
        }
        let deadline = tokio::time::Instant::now() + REAP_TIMEOUT;
        for task in &tasks {
            let _ = tokio::time::timeout_at(deadline, task.wait()).await;
        }
        #[cfg(target_os = "linux")]
        sweep_orphans().await;
    }
}

struct Task {
    id: TaskId,
    pid: libc::pid_t,
    command: String,
    description: Option<String>,
    cwd: PathBuf,
    log: PathBuf,
    started: SystemTime,
    clock: Instant,
    state: Mutex<TaskState>,
    finished: watch::Sender<bool>,
}

#[derive(Default)]
struct TaskState {
    background: bool,
    completed: bool,
    explicitly_killed: bool,
    ended: Option<SystemTime>,
    stop_deadline: Option<Instant>,
    exit_code: Option<i32>,
    signal: Option<String>,
    error: Option<String>,
}

impl TaskState {
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
}

impl Task {
    fn check_error(&self) -> Result<()> {
        if let Some(error) = &lock(&self.state).error {
            return Err(internal(format!("terminal command failed: {error}")));
        }
        Ok(())
    }

    fn duration(&self, state: &TaskState) -> f64 {
        state
            .ended
            .and_then(|end| end.duration_since(self.started).ok())
            .unwrap_or_else(|| self.clock.elapsed())
            .as_secs_f64()
    }

    fn foreground(&self) -> Result<String> {
        let (text, total) = read_log(&self.log, PREVIEW_BYTES).context("read command output")?;
        let state = lock(&self.state);
        let code = state
            .signal
            .as_ref()
            .map(|s| format!("killed ({s})"))
            .unwrap_or_else(|| state.exit_code.unwrap_or(-1).to_string());
        let note = if total > PREVIEW_BYTES as u64 {
            format!(" [truncated - full output at: {}]", self.log.display())
        } else {
            String::new()
        };
        Ok(format!(
            "Exit code: {code}{note}\n\nCommand output:\n\n```\n{text}\n```\n\nCommand completed.\n\nThe previous shell command ended, so on the next invocation of this tool, you will be using a new shell session.\n\nOn the next terminal tool call, the directory of the shell will be {}.",
            self.cwd.display()
        ))
    }

    fn poll(&self, multiple: bool) -> Result<String> {
        let (output, total) = read_log(&self.log, POLL_BYTES).context("read command output")?;
        let state = lock(&self.state);
        let mut text = if multiple {
            format!(
                "--- Task {} [{}] ---\nCommand: {}\nDuration: {:.2}s",
                self.id,
                state.status(),
                self.command,
                self.duration(&state)
            )
        } else {
            format!(
                "=== Task {} ===\nCommand: {}\nStatus: {}\nDuration: {:.2}s",
                self.id,
                self.command,
                state.status(),
                self.duration(&state)
            )
        };
        if let Some(code) = state.exit_code {
            text.push_str(&format!("\nExit Code: {code}"));
        }
        text.push_str(&format!("\nOutput File: {}", self.log.display()));
        text.push_str("\n\n=== Output ===\n");
        if output.is_empty() {
            text.push_str(if state.completed {
                "(no output)"
            } else {
                "(no output yet)"
            });
        }
        text.push_str(&output);
        if total > POLL_BYTES as u64 {
            text.push_str("\n[truncated - use read_file on output_file for full content]");
        }
        Ok(text)
    }

    fn stop(&self, reason: &str, graceful: bool) -> bool {
        let mut state = lock(&self.state);
        if state.completed {
            return false;
        }
        state.explicitly_killed |= reason == "cancelled";
        if state.signal.is_none() {
            state.signal = Some(reason.into());
        }
        signal_session(
            self.pid,
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

fn fresh_dir(root: &Path) -> io::Result<PathBuf> {
    std::fs::create_dir_all(root)?;
    let mut n = 0;
    loop {
        let dir = root.join(format!("{}-{n}", std::process::id()));
        match std::fs::create_dir(&dir) {
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => n += 1,
            created => return created.map(|()| dir),
        }
    }
}

fn create_log(dir: &Path, path: &Path) -> io::Result<File> {
    std::fs::create_dir_all(dir)?;
    File::options().append(true).create_new(true).open(path)
}

fn read_log(path: &Path, limit: usize) -> io::Result<(String, u64)> {
    let mut file = File::open(path)?;
    let total = file.metadata()?.len();
    let mut bytes = Vec::new();
    if total <= limit as u64 {
        file.take(limit as u64).read_to_end(&mut bytes)?;
    } else {
        let half = limit / 2;
        (&mut file).take(half as u64).read_to_end(&mut bytes)?;
        bytes.extend_from_slice(TRUNCATED.as_bytes());
        file.seek(SeekFrom::End(-(half as i64)))?;
        file.take(half as u64).read_to_end(&mut bytes)?;
    }
    Ok((strip_ansi(&String::from_utf8_lossy(&bytes)), total))
}

fn spawn_tracked(command: &mut Command) -> io::Result<(Child, libc::pid_t)> {
    let mut spawned = lock(&SPAWNED);
    let child = command.spawn()?;
    let pid = child.id().expect("an unwaited child has a pid") as libc::pid_t;
    spawned.insert(pid);
    Ok((child, pid))
}

fn signal_session(session: libc::pid_t, signal: libc::c_int) {
    for pid in processes() {
        if unsafe { libc::getsid(pid) } == session {
            unsafe { libc::kill(pid, signal) };
        }
    }
}

#[cfg(target_os = "linux")]
fn processes() -> Vec<libc::pid_t> {
    std::fs::read_dir("/proc")
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|entry| entry.file_name().to_str()?.parse().ok())
        .collect()
}

#[cfg(target_os = "macos")]
fn processes() -> Vec<libc::pid_t> {
    let mut pids: Vec<libc::pid_t> = vec![0; 4096];
    loop {
        let size = (pids.len() * size_of::<libc::pid_t>()) as libc::c_int;
        let count = unsafe { libc::proc_listallpids(pids.as_mut_ptr().cast(), size) };
        let Ok(count) = usize::try_from(count) else {
            return Vec::new();
        };
        if count < pids.len() {
            pids.truncate(count);
            return pids;
        }
        pids.resize(pids.len() * 2, 0);
    }
}

#[cfg(target_os = "linux")]
fn adopt_orphans() -> io::Result<()> {
    if unsafe { libc::prctl(libc::PR_SET_CHILD_SUBREAPER, 1) } != 0 {
        return Err(io::Error::last_os_error());
    }
    let mut children = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::child())?;
    tokio::spawn(async move {
        while children.recv().await.is_some() {
            reap_orphans();
        }
    });
    Ok(())
}

#[cfg(target_os = "linux")]
struct Process {
    pid: libc::pid_t,
    parent: libc::pid_t,
    zombie: bool,
}

#[cfg(target_os = "linux")]
fn adopted_processes() -> Vec<Process> {
    let own_session = unsafe { libc::getsid(0) };
    processes()
        .into_iter()
        .filter_map(|pid| {
            let stat = std::fs::read_to_string(format!("/proc/{pid}/stat")).ok()?;
            let fields = stat.rsplit_once(") ")?.1.split(' ').collect::<Vec<_>>();
            let session: libc::pid_t = fields.get(3)?.parse().ok()?;
            (session != own_session).then_some(Process {
                pid,
                parent: fields.get(1)?.parse().ok()?,
                zombie: fields.first()? == &"Z",
            })
        })
        .collect()
}

#[cfg(target_os = "linux")]
fn reap_orphans() {
    let spawned = lock(&SPAWNED);
    let runner = std::process::id() as libc::pid_t;
    for process in adopted_processes() {
        if process.zombie && process.parent == runner && !spawned.contains(&process.pid) {
            unsafe { libc::waitpid(process.pid, std::ptr::null_mut(), libc::WNOHANG) };
        }
    }
}

#[cfg(target_os = "linux")]
async fn sweep_orphans() {
    let runner = std::process::id() as libc::pid_t;
    for _ in 0..100 {
        let table = adopted_processes();
        let mut doomed = {
            let spawned = lock(&SPAWNED);
            table
                .iter()
                .filter(|process| process.parent == runner && !spawned.contains(&process.pid))
                .map(|process| process.pid)
                .collect::<Vec<_>>()
        };
        let mut next = 0;
        while let Some(&parent) = doomed.get(next) {
            doomed.extend(
                table
                    .iter()
                    .filter(|process| process.parent == parent)
                    .map(|process| process.pid),
            );
            next += 1;
        }
        doomed.retain(|pid| {
            table
                .iter()
                .any(|process| process.pid == *pid && !process.zombie)
        });
        if doomed.is_empty() {
            break;
        }
        for pid in doomed {
            unsafe { libc::kill(pid, libc::SIGKILL) };
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
        reap_orphans();
    }
    reap_orphans();
}

async fn drive(mut child: Child, task: Arc<Task>) {
    let deadline = Instant::now() + MAX_RUNTIME;
    let mut tick = tokio::time::interval(Duration::from_millis(10));
    loop {
        tick.tick().await;
        if Instant::now() >= deadline {
            task.stop("timeout", false);
        }
        let mut state = lock(&task.state);
        if state.stop_deadline.is_some_and(|end| Instant::now() >= end) {
            signal_session(task.pid, libc::SIGKILL);
        }
        let exit = match observe_exit(task.pid) {
            Ok(false) => None,
            Ok(true) => {
                signal_session(task.pid, libc::SIGKILL);
                child
                    .try_wait()
                    .map_err(|error| format!("reap command: {error}"))
                    .transpose()
            }
            Err(error) => {
                signal_session(task.pid, libc::SIGKILL);
                Some(Err(format!("observe command exit: {error}")))
            }
        };
        let Some(exit) = exit else {
            continue;
        };
        match exit {
            Ok(status) => {
                state.exit_code = Some(status.code().unwrap_or(-1));
                if state.signal.is_none() {
                    state.signal = status.signal().map(|s| format!("signal {s}"));
                }
            }
            Err(error) => state.error = Some(error),
        }
        state.completed = true;
        state.ended = Some(SystemTime::now());
        lock(&SPAWNED).remove(&task.pid);
        drop(state);
        task.finished.send_replace(true);
        return;
    }
}

pub async fn capture(
    program: &Path,
    args: &[String],
    cwd: &Path,
    timeout: Duration,
    max_bytes: usize,
) -> io::Result<(i32, String)> {
    let mut command = Command::new(program);
    command
        .args(args)
        .current_dir(cwd)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    let (mut child, pid) = spawn_tracked(&mut command)?;
    let (stdout, stderr) = (child.stdout.take(), child.stderr.take());
    let (mut output, mut errors) = (Vec::new(), Vec::new());
    let deadline = tokio::time::Instant::now() + timeout;
    let drained = tokio::time::timeout_at(deadline, async {
        tokio::try_join!(
            drain(stdout, max_bytes, &mut output),
            drain(stderr, max_bytes, &mut errors)
        )
    })
    .await;
    let mut timed_out = drained.is_err();
    let mut truncated = matches!(drained, Ok(Ok((true, _) | (_, true))));
    if let Ok(Err(error)) = drained {
        return Err(error);
    }
    if !timed_out
        && tokio::time::timeout_at(deadline, child.wait())
            .await
            .is_err()
    {
        timed_out = true;
    }
    if timed_out {
        let _ = child.start_kill();
    }
    let status = child.wait().await?;
    lock(&SPAWNED).remove(&pid);
    output.extend_from_slice(&errors);
    truncated |= output.len() > max_bytes;
    output.truncate(max_bytes);
    let mut text = String::from_utf8_lossy(&output).into_owned();
    if truncated {
        text.push_str("\n[output truncated]");
    }
    if timed_out {
        text.push_str("\n[command ended: timeout]");
    }
    Ok((status.code().unwrap_or(-1), text))
}

async fn drain(
    pipe: Option<impl AsyncRead + Unpin>,
    keep: usize,
    buffer: &mut Vec<u8>,
) -> io::Result<bool> {
    let Some(mut pipe) = pipe else {
        return Ok(false);
    };
    let mut chunk = [0; 8192];
    let mut truncated = false;
    loop {
        let count = pipe.read(&mut chunk).await?;
        if count == 0 {
            return Ok(truncated);
        }
        let room = keep.saturating_sub(buffer.len()).min(count);
        buffer.extend_from_slice(&chunk[..room]);
        truncated |= room < count;
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
    let trimmed = command.trim_end().trim_end_matches(';');
    let (prefix, last) =
        trimmed.split_at(trimmed.rfind([';', '\n', '&', '|']).map_or(0, |i| i + 1));
    let mut words = last.split_whitespace();
    let first = match words.next() {
        Some("builtin" | "command") => words.next(),
        first => first,
    };
    let prefix = prefix.trim_end_matches([' ', '\t']);
    first == Some("wait")
        && (prefix.is_empty()
            || prefix.ends_with([';', '\n'])
            || (prefix.ends_with('&') && !prefix.ends_with("&&")))
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
            if ch == '$' && chars.get(i + 1) == Some(&'{') {
                word_start = false;
                i += 2;
                let mut depth = 1;
                while i < len && depth > 0 {
                    match chars[i] {
                        '{' => depth += 1,
                        '}' => depth -= 1,
                        _ => {}
                    }
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
                .is_some_and(|prev| matches!(prev, '>' | '<' | '|'))
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
