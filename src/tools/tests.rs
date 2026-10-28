use super::*;
use std::time::Duration;
use tempfile::TempDir;

async fn fixture() -> (TempDir, ToolSession) {
    let dir = tempfile::tempdir().unwrap();
    let tools = ToolSession::new(dir.path().to_path_buf(), dir.path().join("logs"))
        .await
        .unwrap();
    (dir, tools)
}

async fn call(tools: &ToolSession, name: &str, args: Value) -> ToolOutput {
    tools.call(name, &args.to_string()).await.unwrap()
}

async fn running_pid(tools: &ToolSession, dir: &TempDir) -> String {
    call(tools, "run_terminal_cmd", json!({"command":"printf '%s' $$ > tracked.pid; exec sleep 60", "description":"test process cleanup", "block_until_ms":0})).await;
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
    let (dir, tools) = fixture().await;
    let definitions = tools.definitions();
    assert_eq!(definitions.len(), 8);
    for definition in definitions {
        let function = &definition["function"];
        let name = function["name"].as_str().unwrap();
        assert_eq!(function["parameters"], tools.catalog[name]["parameters"]);
        assert_eq!(function["description"], tools.catalog[name]["description"]);
    }
    for (name, args, kind) in [
        ("task", "{}", ToolErrorKind::UnknownTool),
        (
            "read_file",
            "{\"target_file\":",
            ToolErrorKind::InvalidArguments,
        ),
        (
            "run_terminal_cmd",
            r#"{"command":"true", "description":"test", "block_until_ms":-1}"#,
            ToolErrorKind::InvalidArguments,
        ),
        (
            "run_terminal_cmd",
            r#"{"command":"true", "description":"test", "block_until_ms":"0"}"#,
            ToolErrorKind::InvalidArguments,
        ),
        (
            "run_terminal_cmd",
            r#"{"command":"true", "description":"test", "block_until_ms":0.5}"#,
            ToolErrorKind::InvalidArguments,
        ),
        (
            "get_terminal_command_output",
            r#"{"task_ids":[], "timeout_ms":0}"#,
            ToolErrorKind::InvalidArguments,
        ),
        (
            "get_terminal_command_output",
            "{}",
            ToolErrorKind::InvalidArguments,
        ),
        (
            "get_terminal_command_output",
            r#"{"task_ids":["task-1"], "timeout_ms":3600001}"#,
            ToolErrorKind::InvalidArguments,
        ),
        (
            "grep",
            r#"{"pattern":"test", "head_limit":0}"#,
            ToolErrorKind::InvalidArguments,
        ),
        (
            "read_file",
            r#"{"target_file":"missing.txt", "offset":0}"#,
            ToolErrorKind::InvalidArguments,
        ),
        (
            "todo_write",
            r#"{"todos":[{"id":"x", "status":"wrong"}]}"#,
            ToolErrorKind::InvalidArguments,
        ),
        (
            "read_file",
            r#"{"target_file":"missing.txt"}"#,
            ToolErrorKind::Failed,
        ),
    ] {
        let output = tools.call(name, args).await.unwrap();
        assert_eq!(
            output.error,
            Some(kind),
            "{name} {args}: {}",
            output.content
        );
        assert!(
            output.content.starts_with("Tool error: "),
            "{}",
            output.content
        );
    }
    call(
        &tools,
        "search_replace",
        json!({"file_path":"sample.txt", "old_string":"", "new_string":"before\nkeep\n"}),
    )
    .await;
    let guard = call(
        &tools,
        "search_replace",
        json!({"file_path":"sample.txt", "old_string":"", "new_string":"erase"}),
    )
    .await;
    assert_eq!(guard.error, Some(ToolErrorKind::Failed));
    assert_eq!(
        tokio::fs::read_to_string(dir.path().join("sample.txt"))
            .await
            .unwrap(),
        "before\nkeep\n"
    );
    call(
        &tools,
        "search_replace",
        json!({"file_path":"sample.txt", "old_string":"before", "new_string":"after"}),
    )
    .await;
    let read = call(&tools, "read_file", json!({"target_file":"sample.txt"})).await;
    assert!(read.error.is_none() && read.content.contains("after"));
    assert!(!read.content.contains("system-reminder"));
    let listing = call(&tools, "list_dir", json!({"target_directory":"."})).await;
    assert!(listing.content.contains("sample.txt"));
    let grep = call(
        &tools,
        "grep",
        json!({"pattern":"after", "path":"sample.txt"}),
    )
    .await;
    assert!(
        grep.content.contains("after"),
        "native grep result: {}",
        grep.content
    );
    let invalid_regex = call(&tools, "grep", json!({"pattern":"[", "path":"sample.txt"})).await;
    assert_eq!(invalid_regex.error, Some(ToolErrorKind::Failed));
    assert!(invalid_regex.content.contains("grep failed"));
    for filename in ["AGENTS.md", "SKILL.md"] {
        tokio::fs::write(
            dir.path().join(filename),
            "excluded-first\nincluded-middle\nexcluded-last\n",
        )
        .await
        .unwrap();
        let read = call(
            &tools,
            "read_file",
            json!({"target_file":filename, "offset":2, "limit":1}),
        )
        .await;
        assert!(read.content.contains("included-middle"));
        assert!(!read.content.contains("excluded-first"));
        assert!(!read.content.contains("excluded-last"));
    }
    let png = [
        137, 80, 78, 71, 13, 10, 26, 10, 0, 0, 0, 13, 73, 72, 68, 82, 0, 0, 0, 1, 0, 0, 0, 1, 8, 6,
        0, 0, 0, 31, 21, 196, 137, 0, 0, 0, 13, 73, 68, 65, 84, 120, 156, 99, 96, 96, 96, 248, 15,
        0, 1, 4, 1, 0, 95, 229, 195, 75, 0, 0, 0, 0, 73, 69, 78, 68, 174, 66, 96, 130,
    ];
    tokio::fs::write(dir.path().join("pixel.png"), png)
        .await
        .unwrap();
    let image = call(&tools, "read_file", json!({"target_file":"pixel.png"})).await;
    assert_eq!(image.error, Some(ToolErrorKind::Failed));
    assert!(image.content.contains("non-text"));
    tools.kill_all().await;
}

#[tokio::test]
async fn integral_floats_are_honoured_as_integers() {
    let (_dir, tools) = fixture().await;
    let started = std::time::Instant::now();
    let background = call(
        &tools,
        "run_terminal_cmd",
        json!({"command":"sleep 2", "description":"float block", "block_until_ms":0.0}),
    )
    .await;
    assert!(
        background.content.contains("<task-id>"),
        "{}",
        background.content
    );
    assert!(started.elapsed() < Duration::from_secs(1));
    let started = std::time::Instant::now();
    let blocked = call(
        &tools,
        "run_terminal_cmd",
        json!({"command":"sleep 3", "description":"float block", "block_until_ms":1000.0}),
    )
    .await;
    assert!(blocked.content.contains("<task-id>"), "{}", blocked.content);
    assert!(started.elapsed() < Duration::from_millis(2500));
    let id = blocked.task_id.unwrap();
    let waited = call(
        &tools,
        "get_terminal_command_output",
        json!({"task_ids":[id], "timeout_ms":10000.0}),
    )
    .await;
    assert!(
        waited.content.contains("Status: completed"),
        "{}",
        waited.content
    );
    tools.kill_all().await;
}

#[tokio::test]
async fn a_broken_tool_runtime_is_an_internal_error() {
    let (dir, mut tools) = fixture().await;
    tools.files = FileTools::new(
        dir.path().to_path_buf(),
        None,
        dir.path().join("no-such-rg"),
    );
    let Err(error) = tools.call("grep", r#"{"pattern":"x"}"#).await else {
        panic!("a missing ripgrep reached the model");
    };
    assert!(error.0.starts_with("grep: ripgrep at "), "{error}");
    tools.kill_all().await;
}

#[tokio::test]
async fn todos_merge_and_reset_per_attempt() {
    let (_dir, tools) = fixture().await;
    call(&tools, "todo_write", json!({"todos":[{"id":"a", "content":"first"}, {"id":"b", "content":"second"}], "merge":false})).await;
    call(
        &tools,
        "todo_write",
        json!({"todos":[{"id":"a", "status":"completed"}], "merge":true}),
    )
    .await;
    assert_eq!(
        tools.state().await,
        json!({"tasks": [], "todos": [{"id": "b", "content": "second", "status": "pending"}], "completed": 1, "cancelled": 0})
    );
    let (_other_dir, other) = fixture().await;
    assert_eq!(other.state().await["todos"], json!([]));
    call(&tools, "todo_write", json!({"todos":[], "merge":false})).await;
    assert_eq!(tools.state().await["completed"], 0);
    tools.kill_all().await;
    other.kill_all().await;
}

#[tokio::test]
async fn background_wait_kill_detach_and_kill_all() {
    let (dir, tools) = fixture().await;
    let waited = call(&tools, "run_terminal_cmd", json!({"command":"sleep 0.1; printf finished", "description":"test background waiting", "block_until_ms":1})).await;
    let id = waited.task_id.unwrap();
    let output = call(
        &tools,
        "get_terminal_command_output",
        json!({"task_ids":[id], "timeout_ms":2000}),
    )
    .await;
    assert!(output.content.contains("finished"));
    assert!(output.content.contains("completed"));
    assert!(
        tools
            .terminal
            .background_tasks()
            .as_array()
            .unwrap()
            .iter()
            .any(|task| task["task_id"] == id.to_string() && task["completed"] == true)
    );

    let running = call(&tools, "run_terminal_cmd", json!({"command":"sleep 60", "description":"test explicit cancellation", "block_until_ms":0})).await;
    let id = running.task_id.unwrap();
    let output = call(
        &tools,
        "get_terminal_command_output",
        json!({"task_ids":[id], "timeout_ms":0}),
    )
    .await;
    assert!(output.content.contains("running"));
    call(&tools, "kill_terminal_command", json!({"task_id":id})).await;
    assert!(
        tools
            .terminal
            .background_tasks()
            .as_array()
            .unwrap()
            .iter()
            .any(|task| task["task_id"] == id.to_string() && task["completed"] == true)
    );

    let pid = running_pid(&tools, &dir).await;
    tokio::time::timeout(Duration::from_secs(8), tools.kill_all())
        .await
        .unwrap();
    assert_process_exited(&pid).await;
    let (dir, tools) = fixture().await;
    let pid = running_pid(&tools, &dir).await;
    let left = tools.detach();
    assert_eq!(left.len(), 1);
    assert_eq!(left[0].pid.to_string(), pid);
    assert_eq!(unsafe { libc::kill(left[0].pid, 0) }, 0);
    unsafe { libc::kill(left[0].pid, libc::SIGKILL) };
}

#[tokio::test]
async fn the_trained_schema_keeps_its_tool_and_parameter_names() {
    let (dir, tools) = fixture().await;
    let names = tools
        .definitions()
        .iter()
        .map(|definition| {
            let function = &definition["function"];
            let parameters = function["parameters"]["properties"]
                .as_object()
                .unwrap()
                .keys()
                .cloned()
                .collect::<Vec<_>>();
            (function["name"].as_str().unwrap().to_owned(), parameters)
        })
        .collect::<Vec<_>>();
    let expected = [
        (
            "run_terminal_cmd",
            &["command", "description", "block_until_ms"][..],
        ),
        ("get_terminal_command_output", &["task_ids", "timeout_ms"]),
        ("kill_terminal_command", &["task_id"]),
        (
            "read_file",
            &["target_file", "offset", "limit", "pages", "format"],
        ),
        (
            "search_replace",
            &["file_path", "old_string", "new_string", "replace_all"],
        ),
        ("list_dir", &["target_directory"]),
        (
            "grep",
            &[
                "pattern",
                "path",
                "glob",
                "-B",
                "-A",
                "-C",
                "-i",
                "type",
                "head_limit",
                "multiline",
            ],
        ),
        ("todo_write", &["merge", "todos"]),
    ]
    .map(|(name, parameters)| {
        let parameters = parameters.iter().map(|p| p.to_string()).collect::<Vec<_>>();
        (name.to_owned(), parameters)
    });
    assert_eq!(names, expected);
    tokio::fs::write(dir.path().join("notes.txt"), "text\n")
        .await
        .unwrap();
    for format in ["image", "text"] {
        let text = call(
            &tools,
            "read_file",
            json!({"target_file":"notes.txt", "format":format}),
        )
        .await;
        assert_eq!(text.content, "1→text\n");
    }
    tokio::fs::write(dir.path().join("paper.pdf"), "%PDF-1.4\n")
        .await
        .unwrap();
    for args in [
        json!({"target_file":"paper.pdf"}),
        json!({"target_file":"paper.pdf", "format":"image"}),
    ] {
        let pdf = call(&tools, "read_file", args).await;
        assert_eq!(
            (pdf.error, pdf.content.as_str()),
            (None, "Vision is not available in this environment.")
        );
    }
    tools.kill_all().await;
}
