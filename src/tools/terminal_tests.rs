use super::*;
use tempfile::TempDir;

async fn fixture() -> (TempDir, TerminalSession) {
    let dir = tempfile::tempdir().unwrap();
    let terminal = TerminalSession::new(dir.path().into(), dir.path().join("logs"))
        .await
        .unwrap();
    (dir, terminal)
}

fn args(command: &str, block: u64) -> Value {
    json!({"command":command,"description":"exercise terminal","block_until_ms":block})
}

async fn run(terminal: &TerminalSession, command: &str, block: u64) -> (String, TaskId) {
    terminal.run(&args(command, block)).await.unwrap()
}

async fn output(terminal: &TerminalSession, ids: &[TaskId], timeout_ms: u64) -> String {
    terminal
        .output(&json!({"task_ids":ids,"timeout_ms":timeout_ms}))
        .await
        .unwrap()
}

async fn kill(terminal: &TerminalSession, id: impl std::fmt::Display) -> String {
    terminal
        .kill(&json!({"task_id":id.to_string()}))
        .await
        .unwrap()
}

async fn read_pid(path: &Path) -> i32 {
    tokio::time::timeout(Duration::from_secs(3), async {
        loop {
            if let Ok(contents) = tokio::fs::read_to_string(path).await
                && let Ok(pid) = contents.parse()
            {
                break pid;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .unwrap()
}

async fn assert_reaped(pid: i32) {
    tokio::time::timeout(Duration::from_secs(3), async {
        loop {
            if unsafe { libc::kill(pid, 0) } < 0 {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("child survived command cleanup");
}

#[tokio::test]
async fn foreground_background_wait_all_and_kill() {
    let (_dir, terminal) = fixture().await;
    let (foreground, _) = run(
        &terminal,
        "printf '\\033[31mhello\\033[0m'; printf error >&2; exit 7",
        1000,
    )
    .await;
    assert!(
        foreground.contains("Exit code: 7")
            && foreground.contains("hello")
            && foreground.contains("error")
    );
    assert!(!foreground.contains('\u{1b}'));
    assert_eq!(terminal.background_tasks(), json!([]));

    let (_, first) = run(&terminal, "sleep 0.05; printf first-done", 1).await;
    let (started, second) = run(&terminal, "sleep 0.1; printf second-done", 0).await;
    assert!(started.contains(&format!("<task-id>{second}</task-id>")));
    let waited = output(&terminal, &[first, second], 2000).await;
    assert!(
        waited.contains("2/2 tasks completed (wait_all)"),
        "{waited}"
    );
    assert!(waited.contains("first-done") && waited.contains("second-done"));
    assert!(waited.find(&first.to_string()).unwrap() < waited.find(&second.to_string()).unwrap());

    let (_, sleeper) = run(&terminal, "sleep 60", 1).await;
    assert!(
        output(&terminal, &[sleeper], 0)
            .await
            .contains("Status: running")
    );
    assert!(kill(&terminal, sleeper).await.starts_with("killed:"));
    assert!(
        kill(&terminal, sleeper)
            .await
            .starts_with("already_exited:")
    );
    assert!(
        output(&terminal, &[sleeper], 0)
            .await
            .contains("Status: cancelled")
    );
    assert!(kill(&terminal, "unknown").await.contains("not found"));
    assert!(kill(&terminal, "task-99").await.contains("not found"));
    terminal.kill_all().await;
}

#[tokio::test]
async fn task_ids_count_up_across_foreground_and_background_commands() {
    let (_dir, terminal) = fixture().await;
    let (started, background) = run(&terminal, "sleep 60", 0).await;
    let (finished, foreground) = run(&terminal, "printf done", 1000).await;
    assert_eq!(
        (background.to_string(), foreground.to_string()),
        ("task-1".to_owned(), "task-2".to_owned())
    );
    assert!(started.contains("<task-id>task-1</task-id>"));
    assert!(finished.contains("Exit code: 0") && finished.contains("done"));
    terminal.kill_all().await;
}

#[tokio::test]
async fn foreground_comments_and_arithmetic_run_without_background_errors() {
    let (_dir, terminal) = fixture().await;
    for command in [
        "printf comment-ok # R&D",
        "((mask = 3 & (5 & 1))); printf 'arithmetic-%s' \"$mask\"",
        "printf 'expansion-%s' $((3 & (5 & 1)))",
        "sleep 0.01 & wait # R&D",
        "sleep 0.01 & builtin wait",
        "sleep 0.01 & command wait",
    ] {
        let (output, _) = run(&terminal, command, 1000).await;
        assert!(output.contains("Exit code: 0"), "{command}: {output}");
    }
    terminal.kill_all().await;
}

#[tokio::test]
async fn commands_inherit_the_runner_environment() {
    let (_dir, terminal) = fixture().await;
    let (output, _) = run(
        &terminal,
        "printf '%s|%s|%s|%s|%s' \"$HOME\" \"$PATH\" \"$CARGO_PKG_NAME\" \"$TERM\" \"$GIT_PAGER\"",
        1000,
    )
    .await;
    let expected = format!(
        "{}|{}|remanie-train-harness|dumb|cat",
        std::env::var("HOME").unwrap(),
        std::env::var("PATH").unwrap()
    );
    assert!(output.contains(&expected), "{output}");
    terminal.kill_all().await;
}

#[tokio::test]
async fn bounded_logs() {
    let (dir, terminal) = fixture().await;
    let (output, _) = run(
        &terminal,
        "head -c 200000 /dev/zero | tr '\\0' x; printf tail-marker",
        1000,
    )
    .await;
    assert!(output.len() < PREVIEW_BYTES + 1500);
    assert!(output.contains("tail-marker") && output.contains("truncated"));

    let log_path = dir.path().join("small.log");
    let log = tokio::fs::File::create(&log_path).await.unwrap();
    let mut command = clean_command(Path::new("/bin/bash"), dir.path());
    command.args([
        "--noprofile",
        "--norc",
        "-c",
        "head -c 200000 /dev/zero; sleep 60",
    ]);
    let task = spawn(
        command,
        TaskState::new(
            TaskId(1),
            "",
            "",
            dir.path(),
            Some(log_path.clone()),
            POLL_BYTES,
        ),
        Some(log),
        1000,
        Duration::from_secs(60),
    )
    .unwrap();
    tokio::time::timeout(Duration::from_secs(3), task.wait())
        .await
        .unwrap();
    assert_eq!(tokio::fs::metadata(log_path).await.unwrap().len(), 1000);
    assert_eq!(lock(&task.state).signal.as_deref(), Some("output_limit"));
    terminal.kill_all().await;
}

#[tokio::test]
async fn capture_bounds_timeout_and_cancellation_reap() {
    let dir = tempfile::tempdir().unwrap();
    let (code, output) = capture(
        Path::new("/bin/bash"),
        &[
            "-c".into(),
            "printf beginning; head -c 100000 /dev/zero; printf error >&2".into(),
        ],
        dir.path(),
        Duration::from_secs(1),
        100,
    )
    .await
    .unwrap();
    assert_eq!(code, 0);
    assert!(output.starts_with("beginning") && output.contains("truncated") && output.len() < 150);
    let (code, output) = capture(
        Path::new("/bin/bash"),
        &["-c".into(), "printf partial; exec sleep 60".into()],
        dir.path(),
        Duration::from_millis(30),
        100,
    )
    .await
    .unwrap();
    assert_eq!(code, -1);
    assert!(output.contains("partial") && output.contains("timeout"));

    let pid_path = dir.path().join("pid");
    let workspace = dir.path().to_path_buf();
    let running = tokio::spawn(async move {
        capture(
            Path::new("/bin/bash"),
            &["-c".into(), "printf '%s' $$ > pid; exec sleep 60".into()],
            &workspace,
            Duration::from_secs(60),
            100,
        )
        .await
    });
    let pid = read_pid(&pid_path).await;
    running.abort();
    let _ = running.await;
    assert_reaped(pid).await;
}

#[tokio::test]
async fn dropped_session_and_aborted_foreground_leave_no_leader() {
    let (dir, terminal) = fixture().await;
    run(&terminal, "printf '%s' $$ > bg.pid; exec sleep 60", 0).await;
    let pid = read_pid(&dir.path().join("bg.pid")).await;
    drop(terminal);
    assert_reaped(pid).await;

    let (dir, terminal) = fixture().await;
    let arguments = args("printf '%s' $$ > fg.pid; exec sleep 60", 300_000);
    let call = terminal.run(&arguments);
    assert!(
        tokio::time::timeout(Duration::from_millis(100), call)
            .await
            .is_err()
    );
    let pid = read_pid(&dir.path().join("fg.pid")).await;
    assert_reaped(pid).await;
    terminal.kill_all().await;
}

#[tokio::test]
async fn kill_shutdown_and_drop_stop_the_entire_process_group() {
    async fn assert_descendant_stopped(pid: i32) {
        tokio::time::timeout(Duration::from_secs(3), async {
            loop {
                if unsafe { libc::kill(pid, 0) } < 0 {
                    break;
                }
                #[cfg(target_os = "linux")]
                if let Ok(stat) = tokio::fs::read_to_string(format!("/proc/{pid}/stat")).await
                    && stat
                        .rsplit_once(") ")
                        .is_some_and(|(_, rest)| rest.starts_with("Z "))
                {
                    // Docker's PID 1 can delay reaping an orphan that has already exited.
                    break;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("live descendant survived process-group cleanup");
    }

    for action in ["kill", "shutdown", "drop"] {
        let (dir, terminal) = fixture().await;
        let (_, id) = run(
            &terminal,
            "printf '%s' $$ > leader.pid; sleep 60 & printf '%s' $! > child.pid; wait",
            0,
        )
        .await;
        let leader = read_pid(&dir.path().join("leader.pid")).await;
        let child = read_pid(&dir.path().join("child.pid")).await;
        assert_eq!(unsafe { libc::kill(leader, 0) }, 0);
        assert_eq!(unsafe { libc::kill(child, 0) }, 0);
        match action {
            "kill" => {
                kill(&terminal, id).await;
            }
            "shutdown" => terminal.kill_all().await,
            "drop" => drop(terminal),
            _ => unreachable!(),
        }
        assert_reaped(leader).await;
        assert_descendant_stopped(child).await;
    }
}

#[test]
fn background_guard_handles_shell_syntax() {
    for command in [
        "sleep 1 &",
        "sleep 1 & echo started",
        "cat <<EOF &\nhello\nEOF",
        "sleep 1 & # started",
        "sleep 1 &\n# wait",
        "printf x#literal &",
        "printf \\ #literal &",
        "printf ''#literal &",
        "((mask = 3 & 1)); sleep 1 &",
        "printf '%s' $((3 & 1)) &",
        "printf x$((3 & 1))#literal &",
        "sleep 1 & echo wait",
        "sleep 1 & false && wait",
        "sleep 1 & cat <<wait\nbody\nwait",
        "sleep 1 & builtinwait",
        "sleep 1 & commandwait",
        "sleep 1 & echo ${x}",
        "sleep 1 |& tee log &",
        "sleep 1 & wait_for_it",
    ] {
        assert!(unwaited_background(command), "accepted {command}");
    }
    for command in [
        "true && true",
        "echo 'a & b'",
        "echo \"a & b\"",
        "echo a\\&b",
        "printf x 2>&1",
        "true & wait",
        "true & true & wait;",
        "cat <<'EOF'\nfoo & bar\nEOF",
        "cat <<- EOF\n\tfoo & bar\n\tEOF",
        "cat <<< 'foo & bar'",
        "printf comment-ok # R&D",
        "# &\nprintf comment-ok",
        "((mask = 3 & (5 & 1))); printf '%s' \"$mask\"",
        "printf '%s' $((3 & (5 & 1)))",
        "sleep 0.01 & wait # R&D",
        "sleep 0.01 & # started\nwait # done &",
        "sleep 0.01 & true; wait",
        "sleep 0.01 & builtin wait",
        "sleep 0.01 & command wait",
        "sleep 0.01 & builtin\t wait # completed",
        "sleep 0.01 & command   wait",
        "sleep 0.01 & cat <<wait\nbody & text\nwait\nwait",
        "make |& tee build.log",
        "sleep 0.2 & wait $!",
        "sleep 0.2 & pid=$!; wait $pid",
        "sleep 0.2 & sleep 0.1 & wait %1 %2",
        "sleep 0.2 &\nwait -n",
        "x='a&b'; echo ${x//&/+}",
        "echo ${x:-a&b} ${#y}",
    ] {
        assert!(!unwaited_background(command), "rejected {command}");
    }
}
