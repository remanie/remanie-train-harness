use super::*;
use std::collections::VecDeque;
use std::sync::Arc;
use tempfile::TempDir;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, Request, ResponseTemplate};

fn reply(message: Value, finish: &str, tokens: u64) -> ResponseTemplate {
    ResponseTemplate::new(200).set_body_json(json!({
        "choices": [{"message": message, "finish_reason": finish}],
        "usage": {"total_tokens": tokens}
    }))
}

fn call(id: &str, name: &str, arguments: Value) -> Value {
    json!({"id": id, "type": "function", "function": {
        "name": name, "arguments": serde_json::to_string(&arguments).unwrap()
    }})
}

fn tool_reply(calls: Vec<Value>, tokens: u64) -> ResponseTemplate {
    reply(
        json!({"role": "assistant", "content": null, "tool_calls": calls}),
        "tool_calls",
        tokens,
    )
}

fn final_reply() -> ResponseTemplate {
    reply(json!({"role": "assistant", "content": "done"}), "stop", 10)
}

async fn fixture(responses: Vec<ResponseTemplate>) -> (TempDir, MockServer, Config) {
    let dir = tempfile::tempdir().unwrap();
    let workspace = dir.path().join("workspace");
    std::fs::create_dir(&workspace).unwrap();
    let server = MockServer::start().await;
    let responses = Arc::new(Mutex::new(VecDeque::from(responses)));
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(move |_: &Request| {
            responses
                .lock()
                .unwrap()
                .pop_front()
                .unwrap_or_else(|| ResponseTemplate::new(500))
        })
        .mount(&server)
        .await;
    let config = Config {
        base_url: format!("{}/v1", server.uri()),
        api_key_env: None,
        model: "local-test-model".into(),
        user_query: "Create result.txt and verify the result.".into(),
        workspace,
        session_dir: dir.path().join("attempt"),
        sampling_params: json!({"temperature":0.2, "top_p":0.9, "seed":42}),
        max_turns: 5,
        max_response_tokens: 2048,
        deadline_secs: 10,
        compaction_threshold_tokens: 0,
        compaction_max_tokens: 1024,
    };
    (dir, server, config)
}

async fn requests(server: &MockServer) -> Vec<Value> {
    server
        .received_requests()
        .await
        .unwrap()
        .iter()
        .map(|request| {
            assert_eq!(request.url.path(), "/v1/chat/completions");
            serde_json::from_slice(&request.body).unwrap()
        })
        .collect()
}

fn trace(dir: &TempDir) -> Vec<Value> {
    std::fs::read_to_string(dir.path().join("attempt/trace.jsonl"))
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect()
}

#[tokio::test]
async fn bearer_auth_and_reasoning_reach_all_requests_without_entering_tools_or_traces() {
    let (dir, server, mut config) = fixture(vec![
            tool_reply(vec![call("environment", "run_terminal_cmd", json!({
                "command":"env > inherited-env.txt", "description":"inspect inherited environment", "block_until_ms":1000
            }))], 30_000),
            reply(json!({"role":"assistant", "content":"<summary>Inspected the environment.</summary>"}), "stop", 10),
            final_reply(),
        ]).await;
    config.api_key_env = Some("OPENROUTER_API_KEY".into());
    config.sampling_params = json!({"reasoning":{"effort":"high"}});
    config.compaction_threshold_tokens = 20_000;
    let secret = "test-auth-private-key";
    let result = run(config, Some(crate::bearer_header(secret).unwrap()))
        .await
        .unwrap();
    assert_eq!(result.status, "completed");
    assert_eq!(result.compactions, 1);
    let requests = server.received_requests().await.unwrap();
    assert_eq!(requests.len(), 3);
    for request in requests {
        assert_eq!(request.headers[AUTHORIZATION], format!("Bearer {secret}"));
        let body: Value = serde_json::from_slice(&request.body).unwrap();
        assert_eq!(body["reasoning"], json!({"effort":"high"}));
        assert!(body.get("api_key_env").is_none());
        assert!(!String::from_utf8_lossy(&request.body).contains(secret));
    }
    let environment =
        std::fs::read_to_string(dir.path().join("workspace/inherited-env.txt")).unwrap();
    assert!(!environment.contains("OPENROUTER_API_KEY"));
    assert!(!environment.contains(secret));
    assert!(
        !std::fs::read_to_string(dir.path().join("attempt/trace.jsonl"))
            .unwrap()
            .contains(secret)
    );
}

#[tokio::test]
async fn exact_prompt_catalog_and_assistant_replay_edit_a_real_file() {
    let arguments = "{\n  \"file_path\": \"result.txt\", \"old_string\": \"\",\n  \"new_string\": \"changed\\n\"\n}";
    let assistant = json!({
        "role":"assistant", "content":null, "reasoning_content":"Keep this field unchanged.",
        "reasoning":"Keep this other provider field too.", "refusal":null,
        "reasoning_details":[{"type":"reasoning.text", "text":"Preserve OpenRouter reasoning details.", "index":0}],
        "tool_calls":[{"id":"edit", "type":"function", "function":{"name":"search_replace", "arguments":arguments}}]
    });
    let (dir, server, config) = fixture(vec![
        reply(assistant.clone(), "tool_calls", 100),
        final_reply(),
    ])
    .await;
    let result = run(config, None).await.unwrap();
    assert!(
        server
            .received_requests()
            .await
            .unwrap()
            .iter()
            .all(|request| !request.headers.contains_key(AUTHORIZATION))
    );
    assert_eq!(result.status, "completed");
    assert_eq!(result.final_answer, "done");
    assert_eq!(
        std::fs::read_to_string(dir.path().join("workspace/result.txt")).unwrap(),
        "changed\n"
    );
    let requests = requests(&server).await;
    assert_eq!(requests.len(), 2);
    assert_eq!(
        requests[0]["messages"],
        json!([
            {"role":"system", "content":"You are an autonomous coding agent. Your main goal is to complete the user's request denoted with the <user_query> tag."},
            {"role":"user", "content":"<user_query>\nCreate result.txt and verify the result.\n</user_query>"}
        ])
    );
    let catalog: serde_json::Map<String, Value> =
        serde_json::from_str(include_str!("../../assets/tools.json")).unwrap();
    let advertised = requests[0]["tools"].as_array().unwrap();
    assert_eq!(advertised.len(), 8);
    for tool in advertised {
        let function = &tool["function"];
        let name = function["name"].as_str().unwrap();
        assert_eq!(tool["type"], "function");
        assert_eq!(function["description"], catalog[name]["description"]);
        assert_eq!(function["parameters"], catalog[name]["parameters"]);
    }
    assert_eq!(requests[1]["messages"][2], assistant);
    assert_eq!(
        requests[1]["messages"][2]["tool_calls"][0]["function"]["arguments"],
        arguments
    );
    assert_eq!(requests[1]["messages"][3]["tool_call_id"], "edit");
    assert_eq!(requests[1]["temperature"], 0.2);
    assert_eq!(requests[1]["top_p"], 0.9);
    assert_eq!(requests[1]["seed"], 42);
    assert_eq!(requests[1]["tools"], requests[0]["tools"]);
    assert_eq!(
        trace(&dir)
            .iter()
            .filter(|event| event["event"] == "tool_result")
            .count(),
        1
    );
}

#[tokio::test]
async fn forced_compaction_keeps_original_query_todos_and_background_tasks() {
    let summary = format!("<summary>1. The original task remains to create and verify result.txt. {} </summary>", "The todo named work is still in progress. A terminal command remains active and its task ID must remain available. ".repeat(6));
    assert!(format_compact_summary(&summary).chars().count() >= 500);
    let first_calls = vec![
        call(
            "todo",
            "todo_write",
            json!({"todos":[{"id":"work", "content":"verify result", "status":"in_progress"}]}),
        ),
        call(
            "background",
            "run_terminal_cmd",
            json!({"command":"printf retained-log; sleep 60", "description":"wait for verification", "block_until_ms":0}),
        ),
    ];
    let (dir, server, mut config) = fixture(vec![
        tool_reply(first_calls, 30_000),
        reply(json!({"role":"assistant", "content":summary}), "stop", 100),
        tool_reply(
            vec![call(
                "finish-todo",
                "todo_write",
                json!({"todos":[{"id":"work", "status":"completed"}]}),
            )],
            100,
        ),
        final_reply(),
    ])
    .await;
    config.compaction_threshold_tokens = 20_000;
    let result = run(config, None).await.unwrap();
    assert_eq!(result.status, "completed");
    assert_eq!(result.compactions, 1);
    let requests = requests(&server).await;
    assert_eq!(requests.len(), 4);
    assert_eq!(
        requests
            .iter()
            .filter(|request| request["tool_choice"] == "none")
            .count(),
        1
    );
    assert_eq!(requests[1]["max_tokens"], 1024);
    assert_eq!(requests[2]["max_tokens"], 2048);
    assert_eq!(requests[2]["messages"][0], requests[0]["messages"][0]);
    assert_eq!(requests[2]["messages"][1], requests[0]["messages"][1]);
    assert_eq!(requests[2]["messages"].as_array().unwrap().len(), 3);
    let continuation = requests[2]["messages"][2]["content"].as_str().unwrap();
    assert!(continuation.contains(&format_compact_summary(&summary)));
    let state: Value =
        serde_json::from_str(continuation.split_once("Current tool state:\n").unwrap().1).unwrap();
    assert_eq!(state["todos"]["todos"]["work"]["content"], "verify result");
    assert_eq!(state["todos"]["todos"]["work"]["status"], "in_progress");
    let tasks = state["background_tasks"].as_array().unwrap();
    assert_eq!(tasks.len(), 1);
    assert_eq!(tasks[0]["completed"], false);
    assert_eq!(tasks[0]["output"], "");
    assert!(!tasks[0]["task_id"].as_str().unwrap().is_empty());
    let messages = requests[3]["messages"].as_array().unwrap();
    let todo_result = messages.last().unwrap()["content"].as_str().unwrap();
    assert!(todo_result.contains("verify result"));
    assert!(todo_result.contains("completed"));
    let trace = trace(&dir);
    assert_eq!(
        trace
            .iter()
            .filter(|event| event["event"] == "compaction_response")
            .count(),
        1
    );
    assert_eq!(
        trace
            .iter()
            .filter(|event| event["event"] == "compacted_context")
            .count(),
        1
    );
    assert_eq!(
        trace
            .iter()
            .find(|event| event["event"] == "compaction_response")
            .unwrap()["body"]["choices"][0]["message"]["content"],
        summary
    );
}

#[tokio::test]
async fn malformed_tool_call_shapes_stop_without_dispatch() {
    for (message, finish) in [
        (json!({"role":"assistant", "content":null}), "tool_calls"),
        (json!({"role":"assistant", "tool_calls":{}}), "tool_calls"),
        (json!({"role":"assistant", "tool_calls":{}}), "stop"),
        (json!({"role":"assistant", "tool_calls":null}), "tool_calls"),
        (json!({"role":"assistant", "tool_calls":[]}), "tool_calls"),
        (json!({"role":"assistant", "content":null}), "stop"),
        (json!({"role":"assistant", "content":"   "}), "stop"),
        (json!({"role":"assistant", "content":[]}), "stop"),
    ] {
        let (dir, server, config) = fixture(vec![reply(message, finish, 10)]).await;
        assert_eq!(run(config, None).await.unwrap().status, "invalid_response");
        assert_eq!(requests(&server).await.len(), 1);
        assert!(
            !trace(&dir)
                .iter()
                .any(|event| event["event"] == "tool_result")
        );
    }
}

#[tokio::test]
async fn duplicate_ids_reject_the_entire_batch_before_writing() {
    let edits = ["first.txt", "second.txt"]
        .into_iter()
        .map(|file| {
            call(
                "same-id",
                "search_replace",
                json!({"file_path":file, "old_string":"", "new_string":"must not exist"}),
            )
        })
        .collect();
    let (dir, _server, config) = fixture(vec![tool_reply(edits, 10)]).await;
    assert_eq!(run(config, None).await.unwrap().status, "invalid_response");
    assert!(!dir.path().join("workspace/first.txt").exists());
    assert!(!dir.path().join("workspace/second.txt").exists());
    assert!(
        !trace(&dir)
            .iter()
            .any(|event| event["event"] == "tool_result")
    );
}

#[tokio::test]
async fn tool_contract_errors_return_to_the_model_without_side_effects() {
    let errors = vec![
        call("unknown", "task", json!({})),
        call(
            "bad-args",
            "search_replace",
            json!({"file_path":"result.txt", "old_string":"", "new_string":"must not exist", "extra":true}),
        ),
        json!({"id":"bad-json", "type":"function", "function":{"name":"list_dir", "arguments":"{"}}),
    ];
    let (dir, server, config) = fixture(vec![tool_reply(errors, 10), final_reply()]).await;
    assert_eq!(run(config, None).await.unwrap().status, "completed");
    let requests = requests(&server).await;
    let messages = requests[1]["messages"].as_array().unwrap();
    let errors: Vec<_> = messages
        .iter()
        .filter(|message| message["role"] == "tool")
        .collect();
    assert_eq!(errors.len(), 3);
    assert!(errors.iter().all(|error| {
        error["content"]
            .as_str()
            .unwrap()
            .starts_with("Tool error:")
    }));
    assert!(!dir.path().join("workspace/result.txt").exists());
}

#[tokio::test]
async fn invalid_compaction_preserves_original_transcript_and_runs_once() {
    for (summary, finish) in [
        (
            json!({"role":"assistant", "content":"<summary></summary>"}),
            "stop",
        ),
        (
            json!({"role":"assistant", "content":"<analysis>reasoning only</analysis>"}),
            "stop",
        ),
        (
            json!({"role":"assistant", "content":"unfinished summary"}),
            "length",
        ),
        (
            json!({"role":"assistant", "content":"summary", "tool_calls":{}}),
            "stop",
        ),
        (
            json!({"role":"assistant", "content":null, "tool_calls":[call("unexpected", "list_dir", json!({"target_directory":"."}))]}),
            "tool_calls",
        ),
    ] {
        let assistant = json!({"role":"assistant", "content":null, "tool_calls":[call("edit", "search_replace", json!({"file_path":"result.txt", "old_string":"", "new_string":"kept"}))]});
        let (dir, server, mut config) = fixture(vec![
            reply(assistant.clone(), "tool_calls", 30_000),
            reply(summary, finish, 10),
        ])
        .await;
        config.compaction_threshold_tokens = 20_000;
        let result = run(config, None).await.unwrap();
        assert_eq!(result.status, "invalid_response");
        assert_eq!(result.compactions, 0);
        assert!(result.error.is_none());
        let requests = requests(&server).await;
        assert_eq!(requests.len(), 2);
        assert_eq!(requests[1]["messages"][0], requests[0]["messages"][0]);
        assert_eq!(requests[1]["messages"][1], requests[0]["messages"][1]);
        assert_eq!(requests[1]["messages"][2], assistant);
        assert_eq!(requests[1]["messages"][3]["role"], "tool");
        let trace = trace(&dir);
        assert_eq!(
            trace
                .iter()
                .find(|event| event["event"] == "model_response")
                .unwrap()["body"]["choices"][0]["message"],
            assistant
        );
        assert!(
            !trace
                .iter()
                .any(|event| event["event"] == "compacted_context")
        );
        assert_eq!(
            std::fs::read_to_string(dir.path().join("workspace/result.txt")).unwrap(),
            "kept"
        );
    }
}

#[tokio::test]
async fn a_short_valid_summary_is_accepted_and_server_errors_remain_errors() {
    for fail in [false, true] {
        let summary = "<summary>Created result.txt; verify it.</summary>";
        let (dir, server, mut config) = fixture(vec![
            tool_reply(
                vec![call(
                    "edit",
                    "search_replace",
                    json!({"file_path":"result.txt", "old_string":"", "new_string":"kept"}),
                )],
                30_000,
            ),
            if fail {
                ResponseTemplate::new(503)
            } else {
                reply(json!({"role":"assistant", "content":summary}), "stop", 10)
            },
            final_reply(),
        ])
        .await;
        config.compaction_threshold_tokens = 20_000;
        let result = run(config, None).await.unwrap();
        if fail {
            assert_eq!(result.status, "error");
            assert!(result.error.unwrap().contains("HTTP 503"));
            assert_eq!(result.compactions, 0);
            assert!(
                !trace(&dir)
                    .iter()
                    .any(|event| event["event"] == "compacted_context")
            );
        } else {
            assert_eq!(result.status, "completed");
            assert_eq!(result.compactions, 1);
            let requests = requests(&server).await;
            assert!(
                requests[2]["messages"][2]["content"]
                    .as_str()
                    .unwrap()
                    .contains("Created result.txt; verify it.")
            );
        }
    }
}

#[tokio::test]
async fn configuration_and_attempt_reuse_fail_before_model_calls() {
    let (_dir, server, mut config) = fixture(vec![final_reply()]).await;
    config.max_turns = 513;
    assert!(config.validate().is_err());
    config.max_turns = 100;
    config.sampling_params = json!({"input_ids":[1,2]});
    assert!(config.validate().is_err());
    config.sampling_params = json!({});
    config.api_key_env = Some("invalid=name".into());
    assert!(config.validate().is_err());
    config.api_key_env = None;
    config.deadline_secs = 0;
    assert!(config.validate().is_err());
    config.deadline_secs = 10;
    std::fs::create_dir(&config.session_dir).unwrap();
    std::fs::write(config.session_dir.join("trace.jsonl"), "existing evidence").unwrap();
    let existing = config.session_dir.join("trace.jsonl");
    assert!(run(config, None).await.is_err());
    assert_eq!(
        std::fs::read_to_string(existing).unwrap(),
        "existing evidence"
    );
    assert!(requests(&server).await.is_empty());
}

#[tokio::test]
async fn deadline_kills_background_processes() {
    let (dir, _server, mut config) = fixture(vec![
            tool_reply(vec![call("background", "run_terminal_cmd", json!({"command":"printf '%s' $$ > tracked.pid; exec sleep 60", "description":"test deadline cleanup", "block_until_ms":0}))], 10),
            final_reply().set_delay(Duration::from_secs(5)),
        ]).await;
    config.deadline_secs = 1;
    let result = run(config, None).await.unwrap();
    assert_eq!(result.status, "deadline");
    let pid = std::fs::read_to_string(dir.path().join("workspace/tracked.pid")).unwrap();
    assert!(
        unsafe { libc::kill(pid.parse().unwrap(), 0) } != 0,
        "background command survived attempt deadline"
    );
    assert_eq!(trace(&dir).last().unwrap()["body"]["status"], "deadline");
}

fn minify(json: &str) -> String {
    let (mut out, mut in_string, mut escaped) = (String::new(), false, false);
    for c in json.chars() {
        if in_string {
            out.push(c);
            (escaped, in_string) = match (escaped, c) {
                (true, _) => (false, true),
                (false, '\\') => (true, true),
                (false, '"') => (false, false),
                _ => (false, true),
            };
        } else if !c.is_whitespace() {
            in_string = c == '"';
            out.push(c);
        }
    }
    out
}

#[tokio::test]
async fn tool_schemas_reach_the_wire_byte_for_byte_in_file_order() {
    let (_dir, server, config) = fixture(vec![final_reply()]).await;
    run(config, None).await.unwrap();
    let request = &server.received_requests().await.unwrap()[0];
    let raw = String::from_utf8(request.body.clone()).unwrap();
    let wire: Value = serde_json::from_str(&raw).unwrap();
    let file = minify(include_str!("../../assets/tools.json"));
    let mut positions = Vec::new();
    for tool in wire["tools"].as_array().unwrap() {
        let function = &tool["function"];
        let (name, description, parameters) = (
            function["name"].as_str().unwrap(),
            function["description"].to_string(),
            function["parameters"].to_string(),
        );
        assert!(raw.contains(&format!(
            r#"{{"type":"function","function":{{"name":"{name}","description":{description},"parameters":{parameters}}}}}"#
        )));
        let entry =
            format!(r#""{name}":{{"description":{description},"parameters":{parameters}}}"#);
        positions.push(
            file.find(&entry)
                .expect("tool differs from assets/tools.json"),
        );
    }
    assert_eq!(positions.len(), 8);
    assert!(positions.is_sorted());
}

#[tokio::test]
async fn call_ids_may_repeat_across_turns() {
    let todo = |status| {
        call(
            "functions.todo_write:0",
            "todo_write",
            json!({"todos":[{"id":"work", "content":"verify", "status":status}]}),
        )
    };
    let (_dir, server, config) = fixture(vec![
        tool_reply(vec![todo("in_progress")], 10),
        tool_reply(vec![todo("completed")], 10),
        final_reply(),
    ])
    .await;
    assert_eq!(run(config, None).await.unwrap().status, "completed");
    assert_eq!(requests(&server).await.len(), 3);
}
