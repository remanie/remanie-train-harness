use super::*;
use tempfile::TempDir;

async fn fixture() -> (TempDir, TerminalSession) {
    let dir = tempfile::tempdir().unwrap();
    let session = dir.path().join("session");
    tokio::fs::create_dir(&session).await.unwrap();
    let terminal = TerminalSession::new(dir.path().into(), session)
        .await
        .unwrap();
    (dir, terminal)
}

fn args(command: &str, block: u64) -> Value {
    json!({"command":command,"description":"exercise terminal","block_until_ms":block})
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
    let output = terminal
        .call(
            "run_terminal_cmd",
            &args(
                "printf '\\033[31mhello\\033[0m'; printf error >&2; exit 7",
                1000,
            ),
            "first",
        )
        .await
        .unwrap();
    assert!(
        output.contains("Exit code: 7") && output.contains("hello") && output.contains("error")
    );
    assert!(!output.contains('\u{1b}'));
    assert_eq!(terminal.snapshot().await, json!([]));

    terminal
        .call(
            "run_terminal_cmd",
            &args("sleep 0.05; printf first-done", 1),
            "first-bg",
        )
        .await
        .unwrap();
    let second = terminal
        .call(
            "run_terminal_cmd",
            &args("sleep 0.1; printf second-done", 0),
            "second-bg",
        )
        .await
        .unwrap();
    let second_id = second
        .split_once("<task-id>")
        .unwrap()
        .1
        .split_once("</task-id>")
        .unwrap()
        .0;
    let output = terminal
        .call(
            "get_terminal_command_output",
            &json!({"task_ids":["first-bg", second_id],"timeout_ms":2000}),
            "wait",
        )
        .await
        .unwrap();
    assert!(
        output.contains("2/2 tasks completed (wait_all)"),
        "{output}"
    );
    assert!(output.contains("first-done") && output.contains("second-done"));
    assert!(output.find("first-bg").unwrap() < output.find(second_id).unwrap());

    terminal
        .call("run_terminal_cmd", &args("sleep 60", 1), "kill-me")
        .await
        .unwrap();
    let snapshot = terminal
        .call(
            "get_terminal_command_output",
            &json!({"task_ids":["kill-me"]}),
            "poll",
        )
        .await
        .unwrap();
    assert!(snapshot.contains("Status: running"));
    assert!(
        terminal
            .call(
                "kill_terminal_command",
                &json!({"task_id":"kill-me"}),
                "kill"
            )
            .await
            .unwrap()
            .starts_with("killed:")
    );
    assert!(
        terminal
            .call(
                "kill_terminal_command",
                &json!({"task_id":"kill-me"}),
                "again"
            )
            .await
            .unwrap()
            .starts_with("already_exited:")
    );
    assert!(
        terminal
            .call(
                "get_terminal_command_output",
                &json!({"task_ids":["kill-me"]}),
                "poll"
            )
            .await
            .unwrap()
            .contains("Status: cancelled")
    );
    assert!(
        terminal
            .call(
                "kill_terminal_command",
                &json!({"task_id":"unknown"}),
                "missing"
            )
            .await
            .unwrap()
            .contains("not found")
    );
    assert!(
        terminal
            .snapshot()
            .await
            .as_array()
            .unwrap()
            .iter()
            .all(|task| task["output"] == "")
    );
    terminal.shutdown().await;
    assert!(
        terminal
            .call("run_terminal_cmd", &args("true", 0), "closed")
            .await
            .is_err()
    );
}

#[tokio::test]
async fn task_ids_stay_unique_across_foreground_and_background_calls() {
    let (_dir, terminal) = fixture().await;
    let background = terminal
        .call("run_terminal_cmd", &args("sleep 60", 0), "background")
        .await
        .unwrap();
    assert!(background.contains("<task-id>background</task-id>"));
    let foreground = terminal
        .call(
            "run_terminal_cmd",
            &args("printf collision-ok", 1000),
            "terminal-1",
        )
        .await
        .unwrap();
    assert!(foreground.contains("Exit code: 0") && foreground.contains("collision-ok"));
    terminal.shutdown().await;
}

#[tokio::test]
async fn foreground_comments_and_arithmetic_run_without_background_errors() {
    let (_dir, terminal) = fixture().await;
    for (id, command) in [
        ("comment", "printf comment-ok # R&D"),
        (
            "arithmetic",
            "((mask = 3 & (5 & 1))); printf 'arithmetic-%s' \"$mask\"",
        ),
        ("expansion", "printf 'expansion-%s' $((3 & (5 & 1)))"),
        ("wait", "sleep 0.01 & wait # R&D"),
        ("builtin-wait", "sleep 0.01 & builtin wait"),
        ("command-wait", "sleep 0.01 & command wait"),
    ] {
        let output = terminal
            .call("run_terminal_cmd", &args(command, 1000), id)
            .await
            .unwrap();
        assert!(output.contains("Exit code: 0"), "{command}: {output}");
    }
    terminal.shutdown().await;
}

#[tokio::test]
async fn commands_inherit_the_runner_environment() {
    let (_dir, terminal) = fixture().await;
    let output = terminal
        .call(
            "run_terminal_cmd",
            &args(
                "printf '%s|%s|%s|%s|%s' \"$HOME\" \"$PATH\" \"$CARGO_PKG_NAME\" \"$TERM\" \"$GIT_PAGER\"",
                1000,
            ),
            "env",
        )
        .await
        .unwrap();
    let expected = format!(
        "{}|{}|remanie-train-harness|dumb|cat",
        std::env::var("HOME").unwrap(),
        std::env::var("PATH").unwrap()
    );
    assert!(output.contains(&expected), "{output}");
    terminal.shutdown().await;
}

#[tokio::test]
async fn bounded_logs_and_safe_names() {
    let (dir, terminal) = fixture().await;
    terminal
        .call("run_terminal_cmd", &args("true", 1000), "../../escape")
        .await
        .unwrap();
    assert!(!dir.path().join("escape.log").exists());
    let output = terminal
        .call(
            "run_terminal_cmd",
            &args(
                "head -c 200000 /dev/zero | tr '\\0' x; printf tail-marker",
                1000,
            ),
            "large",
        )
        .await
        .unwrap();
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
            "cap".into(),
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
    terminal.shutdown().await;
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
    terminal
        .call(
            "run_terminal_cmd",
            &args("printf '%s' $$ > bg.pid; exec sleep 60", 0),
            "drop",
        )
        .await
        .unwrap();
    let pid = read_pid(&dir.path().join("bg.pid")).await;
    drop(terminal);
    assert_reaped(pid).await;

    let (dir, terminal) = fixture().await;
    let arguments = args("printf '%s' $$ > fg.pid; exec sleep 60", 300_000);
    let call = terminal.call("run_terminal_cmd", &arguments, "abort");
    assert!(
        tokio::time::timeout(Duration::from_millis(100), call)
            .await
            .is_err()
    );
    let pid = read_pid(&dir.path().join("fg.pid")).await;
    assert_reaped(pid).await;
    terminal.shutdown().await;
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
        terminal
            .call(
                "run_terminal_cmd",
                &args(
                    "printf '%s' $$ > leader.pid; sleep 60 & printf '%s' $! > child.pid; wait",
                    0,
                ),
                action,
            )
            .await
            .unwrap();
        let leader = read_pid(&dir.path().join("leader.pid")).await;
        let child = read_pid(&dir.path().join("child.pid")).await;
        assert_eq!(unsafe { libc::kill(leader, 0) }, 0);
        assert_eq!(unsafe { libc::kill(child, 0) }, 0);
        match action {
            "kill" => {
                let id = terminal.snapshot().await[0]["task_id"]
                    .as_str()
                    .unwrap()
                    .to_owned();
                terminal
                    .call("kill_terminal_command", &json!({"task_id":id}), "kill")
                    .await
                    .unwrap();
            }
            "shutdown" => terminal.shutdown().await,
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
