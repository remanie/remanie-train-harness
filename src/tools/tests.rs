use super::*;
use std::time::Duration;
use tempfile::TempDir;

async fn fixture() -> (TempDir, ToolSession) {
    let dir = tempfile::tempdir().unwrap();
    let session = dir.path().join("session");
    tokio::fs::create_dir(&session).await.unwrap();
    let tools = ToolSession::new(dir.path().to_path_buf(), session)
        .await
        .unwrap();
    (dir, tools)
}

async fn running_pid(tools: &ToolSession, dir: &TempDir) -> String {
    tools.call("run_terminal_cmd", json!({"command":"printf '%s' $$ > tracked.pid; exec sleep 60", "description":"test process cleanup", "block_until_ms":0}), "cleanup").await.unwrap();
    tokio::time::timeout(Duration::from_secs(3), async {
        loop {
            if let Ok(pid) = tokio::fs::read_to_string(dir.path().join("tracked.pid")).await
                && !pid.is_empty()
            {
                break pid;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .unwrap()
}

async fn assert_process_exited(pid: &str) {
    tokio::time::timeout(Duration::from_secs(3), async {
        loop {
            if unsafe { libc::kill(pid.parse().unwrap(), 0) } != 0 {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("attempt process survived cleanup");
}

#[tokio::test]
async fn files_catalog_and_validation() {
    tokio::task::LocalSet::new().run_until(async {
        let (dir, tools) = fixture().await;
        let definitions = tools.definitions().await.unwrap();
        assert_eq!(definitions.len(), 8);
        for definition in definitions {
            let function = &definition["function"];
            let name = function["name"].as_str().unwrap();
            assert_eq!(function["parameters"], tools.catalog[name]["parameters"]);
            assert_eq!(function["description"], tools.catalog[name]["description"]);
        }
        for (name, args) in [
            ("task", json!({})),
            ("run_terminal_cmd", json!({"command":"true", "description":"test", "timeout":1})),
            ("run_terminal_cmd", json!({"command":"true", "description":"test", "block_until_ms":300001})),
            ("run_terminal_cmd", json!({"command":"true", "description":"test", "block_until_ms":"0"})),
            ("get_terminal_command_output", json!({"task_ids":[], "timeout_ms":0})),
            ("grep", json!({"pattern":"test", "output_mode":"content"})),
            ("todo_write", json!({"todos":[{"id":"x", "status":"wrong"}]})),
        ] {
            assert!(tools.call(name, args, "invalid").await.is_err(), "accepted {name}");
        }
        tools.call("search_replace", json!({"file_path":"sample.txt", "old_string":"", "new_string":"before\nkeep\n"}), "create").await.unwrap();
        tools.call("search_replace", json!({"file_path":"sample.txt", "old_string":"", "new_string":"erase"}), "guard").await.ok();
        assert_eq!(tokio::fs::read_to_string(dir.path().join("sample.txt")).await.unwrap(), "before\nkeep\n");
        tools.call("search_replace", json!({"file_path":"sample.txt", "old_string":"before", "new_string":"after"}), "edit").await.unwrap();
        let read = tools.call("read_file", json!({"target_file":"sample.txt"}), "read").await.unwrap();
        assert!(read.contains("after"));
        assert!(!read.contains("system-reminder"));
        let listing = tools.call("list_dir", json!({"target_directory":"."}), "list").await.unwrap();
        assert!(listing.contains("sample.txt"));
        let grep = tools.call("grep", json!({"pattern":"after", "path":"sample.txt"}), "grep").await.unwrap();
        assert!(grep.contains("after"), "native grep result: {grep}");
        let invalid_regex = tools.call("grep", json!({"pattern":"[", "path":"sample.txt"}), "invalid-regex").await.unwrap_err();
        assert!(invalid_regex.to_string().contains("grep failed"));
        for filename in ["AGENTS.md", "SKILL.md"] {
            tokio::fs::write(dir.path().join(filename), "excluded-first\nincluded-middle\nexcluded-last\n").await.unwrap();
            let read = tools.call("read_file", json!({"target_file":filename, "offset":2, "limit":1}), "window").await.unwrap();
            assert!(read.contains("included-middle"));
            assert!(!read.contains("excluded-first"));
            assert!(!read.contains("excluded-last"));
        }
        let png = [137,80,78,71,13,10,26,10,0,0,0,13,73,72,68,82,0,0,0,1,0,0,0,1,8,6,0,0,0,31,21,196,137,0,0,0,13,73,68,65,84,120,156,99,96,96,96,248,15,0,1,4,1,0,95,229,195,75,0,0,0,0,73,69,78,68,174,66,96,130];
        tokio::fs::write(dir.path().join("pixel.png"), png).await.unwrap();
        let error = tools.call("read_file", json!({"target_file":"pixel.png"}), "image").await.unwrap_err();
        assert!(error.to_string().contains("non-text"));
        tools.shutdown().await;
        assert!(tools.call("list_dir", json!({"target_directory":"."}), "closed").await.is_err());
    }).await;
}

#[tokio::test]
async fn todos_merge_and_reset_per_attempt() {
    tokio::task::LocalSet::new().run_until(async {
        let (_dir, tools) = fixture().await;
        tools.call("todo_write", json!({"todos":[{"id":"a", "content":"first"}, {"id":"b", "content":"second"}], "merge":false}), "todos").await.unwrap();
        tools.call("todo_write", json!({"todos":[{"id":"a", "status":"completed"}], "merge":false}), "partial").await.unwrap();
        let state = tools.snapshot().await.unwrap();
        assert_eq!(state["todos"]["todos"]["a"]["content"], "first");
        assert_eq!(state["todos"]["todos"]["a"]["status"], "completed");
        assert_eq!(state["todos"]["todos"]["b"]["content"], "second");
        let (_other_dir, other) = fixture().await;
        assert_eq!(other.snapshot().await.unwrap()["todos"]["todos"], json!({}));
        tools.call("todo_write", json!({"todos":[], "merge":false}), "clear").await.unwrap();
        assert_eq!(tools.snapshot().await.unwrap()["todos"]["todos"], json!({}));
        tools.shutdown().await;
        other.shutdown().await;
    }).await;
}

#[tokio::test]
async fn background_wait_kill_and_shutdown() {
    tokio::task::LocalSet::new().run_until(async {
        let (dir, tools) = fixture().await;
        let output = tools.call("run_terminal_cmd", json!({"command":"printf '%s\\n' \"$HOME\" \"$TMPDIR\" \"$PATH\"", "description":"check attempt environment"}), "env").await.unwrap();
        assert!(output.contains(&dir.path().join("session/home").display().to_string()));
        assert!(output.contains(&dir.path().join("session/tmp").display().to_string()));
        assert!(output.contains("/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"));
        tools.call("run_terminal_cmd", json!({"command":"env | cut -d= -f1 > child-env.txt", "description":"check environment inheritance"}), "env-names").await.unwrap();
        let mut allowed: BTreeSet<String> = ["PATH", "HOME", "TMPDIR", "LANG", "TERM", "NO_COLOR", "FORCE_COLOR", "GROK_AGENT", "PWD", "OLDPWD", "SHLVL", "_"].into_iter().map(str::to_owned).collect();
        allowed.extend(["PAGER", "GIT_PAGER", "SYSTEMD_PAGER", "MANPAGER", "AWS_PAGER", "GIT_EDITOR", "GIT_SEQUENCE_EDITOR", "GIT_TERMINAL_PROMPT", "GPG_TTY", "LESS", "GH_PAGER", "PYTHONUNBUFFERED"].into_iter().map(str::to_owned));
        allowed.insert("LOGNAME".into()); // zsh derives this from the OS account even under env -i -f.
        let names = tokio::fs::read_to_string(dir.path().join("child-env.txt")).await.unwrap();
        for name in names.lines() {
            assert!(allowed.contains(name), "inherited unexpected environment variable {name}");
        }

        tools.call("run_terminal_cmd", json!({"command":"sleep 0.1; printf finished", "description":"test background waiting", "block_until_ms":1}), "waited").await.unwrap();
        let tasks = tools.snapshot().await.unwrap();
        let id = tasks["background_tasks"][0]["task_id"].as_str().unwrap().to_owned();
        let output = tools.call("get_terminal_command_output", json!({"task_ids":[id], "timeout_ms":2000}), "wait").await.unwrap();
        assert!(output.contains("finished"));
        assert!(output.contains("completed"));
        assert!(tools.snapshot().await.unwrap()["background_tasks"].as_array().unwrap().iter().any(|task| task["task_id"] == id && task["completed"] == true));

        tools.call("run_terminal_cmd", json!({"command":"sleep 60", "description":"test explicit cancellation", "block_until_ms":0}), "kill-me").await.unwrap();
        let tasks = tools.snapshot().await.unwrap();
        let id = tasks["background_tasks"].as_array().unwrap().iter().find(|task| task["completed"] == false).unwrap()["task_id"].as_str().unwrap().to_owned();
        let output = tools.call("get_terminal_command_output", json!({"task_ids":[id], "timeout_ms":0}), "snapshot").await.unwrap();
        assert!(output.contains("running"));
        tools.call("kill_terminal_command", json!({"task_id":id}), "kill").await.unwrap();
        assert!(tools.snapshot().await.unwrap()["background_tasks"].as_array().unwrap().iter().any(|task| task["task_id"] == id && task["completed"] == true));

        let pid = running_pid(&tools, &dir).await;
        tokio::time::timeout(Duration::from_secs(8), tools.shutdown()).await.unwrap();
        assert_process_exited(&pid).await;
        let (dir, tools) = fixture().await;
        let pid = running_pid(&tools, &dir).await;
        drop(tools);
        assert_process_exited(&pid).await;
    }).await;
}
