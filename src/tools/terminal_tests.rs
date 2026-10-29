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
async fn background_commands_use_block_until_ms_zero_and_never_an_ampersand() {
    let (dir, terminal) = fixture().await;
    for block in [0, 1000] {
        let error = terminal.run(&args("sleep 30 &", block)).await.unwrap_err();
        assert!(
            error.to_string().contains(
                "To run a command in the background, pass block_until_ms=0 instead of using &."
            ),
            "{error}"
        );
    }
    let (started, id) = run(&terminal, "printf '%s' $$ > server.pid; exec sleep 30", 0).await;
    assert!(started.contains("<status>running</status>"));
    let pid = read_pid(&dir.path().join("server.pid")).await;
    tokio::time::sleep(Duration::from_millis(500)).await;
    assert_eq!(
        unsafe { libc::kill(pid, 0) },
        0,
        "the background command died"
    );
    assert!(
        output(&terminal, &[id], 0)
            .await
            .contains("Status: running")
    );
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
async fn previews_and_polls_mark_where_they_cut_the_log() {
    let (_dir, terminal) = fixture().await;
    let (preview, _) = run(
        &terminal,
        "head -c 200000 /dev/zero | tr '\\0' x; printf tail-marker",
        1000,
    )
    .await;
    assert!(preview.len() < PREVIEW_BYTES + 1500);
    assert!(preview.contains("tail-marker") && preview.contains("truncated"));
    let (_, id) = run(&terminal, "seq 1 100000", 0).await;
    let polled = output(&terminal, &[id], 10_000).await;
    let shown = polled.split_once("=== Output ===\n").unwrap().1;
    let (head, tail) = shown.split_once(TRUNCATED).expect("no marker at the cut");
    assert!(head.starts_with("1\n2\n3\n"), "{head}");
    assert!(tail.contains("\n99999\n100000"), "{tail}");
    terminal.kill_all().await;
}

#[tokio::test]
async fn the_log_keeps_every_byte_and_the_reply_shows_its_ends() {
    let (_dir, terminal) = fixture().await;
    let (output, id) = run(
        &terminal,
        "head -c 17000000 /dev/zero | tr '\\0' x; printf finished",
        60_000,
    )
    .await;
    assert!(output.starts_with("Exit code: 0"), "{}", &output[..100]);
    assert!(output.len() < PREVIEW_BYTES + 1500);
    assert!(output.contains(TRUNCATED) && output.contains("finished\n```"));
    let log = std::fs::read(terminal.log_dir.join(format!("{id}.log"))).unwrap();
    assert_eq!(log.len(), 17_000_008);
    assert!(log.ends_with(b"xfinished"));
    terminal.kill_all().await;
}

#[tokio::test]
async fn capture_bounds_timeout_and_cancellation_reap() {
    let dir = tempfile::tempdir().unwrap();
    let (code, output, errors) = capture(
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
    assert_eq!((code, errors.as_str()), (0, "error"));
    assert!(output.starts_with("beginning") && output.contains("truncated") && output.len() < 150);
    let (code, output, _) = capture(
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

#[test]
fn background_tasks_outlive_detach_and_the_runtime() {
    let dir = tempfile::tempdir().unwrap();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let (left, server) = runtime.block_on(async {
        let terminal = TerminalSession::new(dir.path().into(), dir.path().join("logs"))
            .await
            .unwrap();
        let (_, id) = run(&terminal, "printf '%s' $$ > server.pid; exec sleep 60", 0).await;
        run(&terminal, "sleep 0.01", 1000).await;
        let (_, killed) = run(&terminal, "sleep 60", 0).await;
        kill(&terminal, killed).await;
        let pid = read_pid(&dir.path().join("server.pid")).await;
        (terminal.detach(), (id, pid))
    });
    drop(runtime);
    std::thread::sleep(Duration::from_millis(100));
    let alive = unsafe { libc::kill(server.1, 0) } == 0;
    unsafe { libc::kill(server.1, libc::SIGKILL) };
    assert!(alive, "background task died when the runtime ended");
    assert_eq!(left.len(), 1, "{left:?}");
    assert_eq!((left[0].task_id, left[0].pid), server);
    assert!(left[0].output_file.ends_with("task-1.log"));
}

#[tokio::test]
async fn aborted_foreground_leaves_no_leader() {
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

async fn assert_stopped(pid: i32) {
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
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("live descendant survived cleanup");
}

#[tokio::test]
async fn kill_and_kill_all_stop_every_process_in_the_session() {
    for action in ["kill", "kill_all"] {
        let (dir, terminal) = fixture().await;
        let (_, id) = run(
            &terminal,
            "printf '%s' $$ > leader.pid; sleep 60 & printf '%s' $! > child.pid; set -m; sleep 60 & printf '%s' $! > job.pid; wait",
            0,
        )
        .await;
        let leader = read_pid(&dir.path().join("leader.pid")).await;
        let child = read_pid(&dir.path().join("child.pid")).await;
        let job = read_pid(&dir.path().join("job.pid")).await;
        assert_ne!(unsafe { libc::getpgid(job) }, leader);
        match action {
            "kill" => assert!(kill(&terminal, id).await.starts_with("killed:")),
            _ => terminal.kill_all().await,
        }
        assert_reaped(leader).await;
        assert_stopped(child).await;
        assert_stopped(job).await;
    }
}

#[cfg(target_os = "linux")]
#[test]
fn kill_all_reaches_descendants_that_left_the_session() {
    const NAME: &str = "tools::terminal::tests::kill_all_reaches_descendants_that_left_the_session";
    if std::env::var_os("RTH_ISOLATED_TEST").is_none() {
        let child = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", NAME, "--nocapture"])
            .env("RTH_ISOLATED_TEST", "1")
            .output()
            .unwrap();
        let output = String::from_utf8_lossy(&child.stdout);
        assert!(
            child.status.success() && output.contains("1 passed"),
            "{output}"
        );
        return;
    }
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(async {
        let (dir, terminal) = fixture().await;
        run(
            &terminal,
            "timeout 600 sh -c 'printf %s $$ > timeout.pid; exec sleep 4001'; echo end",
            0,
        )
        .await;
        run(
            &terminal,
            "setsid -f sh -c 'printf %s $$ > setsid.pid; exec sleep 912'",
            5000,
        )
        .await;
        run(
            &terminal,
            "sh -c \"(setsid sh -c 'printf %s \\$\\$ > forked.pid; exec sleep 913' &)\"; sleep 0.2",
            0,
        )
        .await;
        let pids = [
            read_pid(&dir.path().join("timeout.pid")).await,
            read_pid(&dir.path().join("setsid.pid")).await,
            read_pid(&dir.path().join("forked.pid")).await,
        ];
        for pid in pids {
            assert_eq!(
                unsafe { libc::kill(pid, 0) },
                0,
                "{pid} died before kill_all"
            );
        }
        terminal.kill_all().await;
        for pid in pids {
            assert_stopped(pid).await;
        }
    });
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
