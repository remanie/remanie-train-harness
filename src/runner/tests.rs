use super::*;
use reqwest::header::AUTHORIZATION;
use std::collections::{BTreeMap, VecDeque};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use tempfile::TempDir;
use tokio::io::AsyncReadExt;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, Request as Received, ResponseTemplate};

fn reply(message: Value, finish: &str, tokens: u64) -> ResponseTemplate {
    ResponseTemplate::new(200).set_body_json(json!({
        "id": "chatcmpl-test",
        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
        "usage": {"prompt_tokens": tokens, "completion_tokens": 10, "total_tokens": tokens + 10}
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

fn list_dir(id: &str) -> Value {
    call(id, "list_dir", json!({"target_directory": "."}))
}

struct Fixture {
    dir: TempDir,
    server: MockServer,
    config: Value,
}

async fn fixture(responses: Vec<ResponseTemplate>) -> Fixture {
    let dir = tempfile::tempdir().unwrap();
    let workspace = dir.path().join("workspace");
    std::fs::create_dir(&workspace).unwrap();
    let server = MockServer::start().await;
    let responses = Arc::new(Mutex::new(VecDeque::from(responses)));
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(move |_: &Received| {
            responses
                .lock()
                .unwrap()
                .pop_front()
                .unwrap_or_else(|| ResponseTemplate::new(599))
        })
        .mount(&server)
        .await;
    let config = json!({
        "base_url": format!("{}/v1", server.uri()),
        "model": "local-test-model",
        "user_query": "Create result.txt and verify the result.",
        "workspace": workspace,
        "session_dir": dir.path().join("attempt"),
        "sampling_params": {"temperature": 0.2, "top_p": 0.9, "seed": 42, "max_tokens": 2048},
        "max_turns": 5,
        "deadline_secs": 10
    });
    Fixture {
        dir,
        server,
        config,
    }
}

impl Fixture {
    fn session(&self) -> PathBuf {
        self.dir.path().join("attempt")
    }

    fn workspace(&self) -> PathBuf {
        self.dir.path().join("workspace")
    }

    async fn launch(&self, key: Option<&str>) -> (Value, u8) {
        let input = serde_json::to_vec(&self.config).unwrap();
        let key = key.map(OsString::from);
        let result = tokio::task::spawn_blocking(move || run(&input, |_| key))
            .await
            .unwrap();
        let exit = result.outcome.exit_code();
        (serde_json::to_value(&result).unwrap(), exit)
    }

    async fn run_with_key(&self, key: Option<&str>) -> Value {
        let (result, exit) = self.launch(key).await;
        assert_eq!(exit, if result["gradable"] == true { 0 } else { 70 });
        let written = std::fs::read(self.session().join("result.json")).unwrap();
        assert_eq!(serde_json::from_slice::<Value>(&written).unwrap(), result);
        if self.session().join("trace.jsonl").exists() {
            let mut last = self.trace().pop().unwrap();
            let last_fields = last.as_object_mut().unwrap();
            assert_eq!(last_fields.remove("event").unwrap(), "result");
            last_fields.remove("t_ms");
            assert_eq!(last, result);
        }
        result
    }

    async fn run(&self) -> Value {
        self.run_with_key(None).await
    }

    async fn raw_requests(&self) -> Vec<Vec<u8>> {
        self.server
            .received_requests()
            .await
            .unwrap()
            .into_iter()
            .map(|request| request.body)
            .collect()
    }

    async fn requests(&self) -> Vec<Value> {
        self.raw_requests()
            .await
            .iter()
            .map(|body| serde_json::from_slice(body).unwrap())
            .collect()
    }

    fn trace(&self) -> Vec<Value> {
        std::fs::read_to_string(self.session().join("trace.jsonl"))
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect()
    }

    fn events(&self, kind: &str) -> Vec<Value> {
        self.trace()
            .into_iter()
            .filter(|event| event["event"] == kind)
            .collect()
    }
}

fn alive(pid_file: &Path) -> bool {
    let pid: i32 = std::fs::read_to_string(pid_file).unwrap().parse().unwrap();
    unsafe { libc::kill(pid, 0) == 0 }
}

#[tokio::test]
async fn bearer_auth_and_reasoning_reach_all_requests_without_entering_tools_or_traces() {
    let mut f = fixture(vec![
        tool_reply(vec![call("environment", "run_terminal_cmd", json!({
            "command": "env > inherited-env.txt", "description": "inspect inherited environment", "block_until_ms": 1000
        }))], 30_000),
        reply(json!({"role": "assistant", "content": "<summary>Inspected the environment.</summary>"}), "stop", 10),
        final_reply(),
    ]).await;
    f.config["api_key_env"] = json!("OPENROUTER_API_KEY");
    f.config["sampling_params"] = json!({"reasoning": {"effort": "high"}, "max_tokens": 2048});
    f.config["compaction"] = json!({"threshold_tokens": 20_000, "max_tokens": 1024});
    let secret = "test-auth-private-key";
    let result = f.run_with_key(Some(secret)).await;
    assert_eq!(result["outcome"], "completed");
    assert_eq!(result["compactions"], 1);
    let requests = f.server.received_requests().await.unwrap();
    assert_eq!(requests.len(), 3);
    for request in requests {
        assert_eq!(request.headers[AUTHORIZATION], format!("Bearer {secret}"));
        let body: Value = serde_json::from_slice(&request.body).unwrap();
        assert_eq!(body["reasoning"], json!({"effort": "high"}));
        assert!(body.get("api_key_env").is_none());
        assert!(!String::from_utf8_lossy(&request.body).contains(secret));
    }
    let environment = std::fs::read_to_string(f.workspace().join("inherited-env.txt")).unwrap();
    assert!(!environment.contains(secret));
    assert!(
        !std::fs::read_to_string(f.session().join("trace.jsonl"))
            .unwrap()
            .contains(secret)
    );
}

#[tokio::test]
async fn a_missing_key_is_a_config_outcome() {
    let mut f = fixture(vec![final_reply()]).await;
    f.config["api_key_env"] = json!("REMANIE_TEST_UNSET_KEY");
    let result = f.run().await;
    assert_eq!(result["cause"]["kind"], "config");
    assert!(f.requests().await.is_empty());
}

#[tokio::test]
async fn exact_prompt_catalog_and_assistant_replay_edit_a_real_file() {
    let arguments = "{\n  \"file_path\": \"result.txt\", \"old_string\": \"\",\n  \"new_string\": \"changed\\n\"\n}";
    let assistant = json!({
        "role": "assistant", "content": null, "reasoning_content": "Keep this field unchanged.",
        "reasoning": "Keep this other provider field too.", "refusal": null,
        "reasoning_details": [{"type": "reasoning.text", "text": "Preserve OpenRouter reasoning details.", "index": 0}],
        "tool_calls": [{"id": "edit", "type": "function", "function": {"name": "search_replace", "arguments": arguments}}]
    });
    let f = fixture(vec![
        reply(assistant.clone(), "tool_calls", 100),
        final_reply(),
    ])
    .await;
    let result = f.run().await;
    assert!(
        f.server
            .received_requests()
            .await
            .unwrap()
            .iter()
            .all(|request| !request.headers.contains_key(AUTHORIZATION))
    );
    assert_eq!(result["outcome"], "completed");
    assert_eq!(result["final_answer"], "done");
    assert_eq!(result["turns"], 2);
    assert_eq!(
        result["usage"],
        json!({"prompt_tokens": 110, "completion_tokens": 20, "cached_tokens": 0, "complete": true})
    );
    assert_eq!(
        std::fs::read_to_string(f.workspace().join("result.txt")).unwrap(),
        "changed\n"
    );
    let requests = f.requests().await;
    assert_eq!(requests.len(), 2);
    assert_eq!(
        requests[0]["messages"],
        json!([
            {"role": "system", "content": "You are an autonomous coding agent. Your main goal is to complete the user's request denoted with the <user_query> tag."},
            {"role": "user", "content": render("user", include_str!("../../assets/user_query.jinja2"), user_info(&f.workspace(), "Create result.txt and verify the result.")).unwrap()}
        ])
    );
    let first = requests[0]["messages"][1]["content"].as_str().unwrap();
    assert!(first.starts_with("<user_info>\nOS Version: "), "{first}");
    assert!(
        first.contains(&format!("Workspace Path: {}\n", f.workspace().display())),
        "{first}"
    );
    assert!(
        first.ends_with(
            "</user_info>\n\n<user_query>\nCreate result.txt and verify the result.\n</user_query>"
        ),
        "{first}"
    );
    let raw = String::from_utf8(f.raw_requests().await[1].clone()).unwrap();
    assert!(raw.contains(&serde_json::to_string(&assistant).unwrap()));
    assert_eq!(requests[1]["messages"][3]["tool_call_id"], "edit");
    assert_eq!(requests[1]["temperature"], 0.2);
    assert_eq!(requests[1]["tools"], requests[0]["tools"]);
    assert_eq!(f.events("model_call").len(), 2);
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
    let f = fixture(vec![final_reply()]).await;
    f.run().await;
    let raw = String::from_utf8(f.raw_requests().await[0].clone()).unwrap();
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
async fn sampling_params_reach_the_wire_verbatim_before_the_runner_keys() {
    let mut f = fixture(vec![final_reply()]).await;
    f.config["sampling_params"] = json!({
        "temperature": 1.0, "logprobs": true, "top_logprobs": 2, "max_tokens": 64,
        "chat_template_kwargs": {"enable_thinking": false}, "custom_backend_knob": [1, 2]
    });
    assert_eq!(f.run().await["outcome"], "completed");
    let raw = String::from_utf8(f.raw_requests().await[0].clone()).unwrap();
    assert!(raw.starts_with(
        r#"{"temperature":1.0,"logprobs":true,"top_logprobs":2,"max_tokens":64,"chat_template_kwargs":{"enable_thinking":false},"custom_backend_knob":[1,2],"model":"local-test-model","messages":["#
    ));
    assert!(raw.ends_with(r#""tool_choice":"auto","stream":false}"#));
}

#[tokio::test]
async fn invalid_input_is_a_config_outcome_with_a_result_file() {
    for (field, value) in [
        ("max_turns", json!(513)),
        ("deadline_secs", json!(0)),
        ("api_key_env", json!("invalid=name")),
        (
            "sampling_params",
            json!({"max_tokens": 64, "model": "other"}),
        ),
        ("sampling_params", json!({"max_tokens": 64, "stream": true})),
        ("sampling_params", json!({"temperature": 1.0})),
        ("base_url", json!("http://user:secret@example.com/v1")),
        (
            "compaction",
            json!({"threshold_tokens": 0, "max_tokens": 1}),
        ),
        ("compaction", json!({"max_tokens": 1})),
        ("max_response_tokens", json!(4096)),
    ] {
        let mut f = fixture(vec![final_reply()]).await;
        f.config[field] = value;
        let result = f.run().await;
        assert_eq!(result["outcome"], "infrastructure", "{field}");
        assert_eq!(result["cause"]["kind"], "config", "{field}");
        assert!(f.requests().await.is_empty());
        assert!(!f.session().join("trace.jsonl").exists());
    }
}

#[tokio::test]
async fn a_reused_session_dir_is_refused_and_left_untouched() {
    let f = fixture(vec![final_reply()]).await;
    std::fs::create_dir(f.session()).unwrap();
    std::fs::write(f.session().join("trace.jsonl"), "existing evidence").unwrap();
    let (result, exit) = f.launch(None).await;
    assert_eq!(result["cause"]["kind"], "config");
    assert_eq!(exit, 70);
    assert_eq!(
        std::fs::read_to_string(f.session().join("trace.jsonl")).unwrap(),
        "existing evidence"
    );
    assert!(!f.session().join("result.json").exists());
    assert!(f.requests().await.is_empty());
}

#[tokio::test]
async fn malformed_replies_end_the_attempt_without_dispatch() {
    let missing_type = json!({"id": "c1", "function": {"name": "list_dir", "arguments": "{\"target_directory\": \".\"}"}});
    let object_arguments = json!({"id": "c1", "type": "function", "function": {"name": "list_dir", "arguments": {"target_directory": "."}}});
    for (message, finish, kind) in [
        (
            json!({"role": "assistant", "content": null}),
            "tool_calls",
            "unexpected_finish",
        ),
        (
            json!({"role": "assistant", "tool_calls": {}}),
            "tool_calls",
            "malformed_message",
        ),
        (
            json!({"role": "assistant", "tool_calls": {}}),
            "stop",
            "malformed_message",
        ),
        (
            json!({"role": "assistant", "tool_calls": null}),
            "tool_calls",
            "unexpected_finish",
        ),
        (
            json!({"role": "assistant", "tool_calls": []}),
            "tool_calls",
            "unexpected_finish",
        ),
        (
            json!({"role": "assistant", "content": null}),
            "stop",
            "empty_turn",
        ),
        (
            json!({"role": "assistant", "content": "   "}),
            "stop",
            "empty_turn",
        ),
        (
            json!({"role": "assistant", "content": []}),
            "stop",
            "malformed_message",
        ),
        (
            json!({"role": "assistant", "content": "x"}),
            "content_filter",
            "unexpected_finish",
        ),
        (
            json!({"role": "assistant", "tool_calls": [missing_type]}),
            "tool_calls",
            "malformed_tool_call",
        ),
        (
            json!({"role": "assistant", "tool_calls": [object_arguments]}),
            "tool_calls",
            "malformed_tool_call",
        ),
    ] {
        let f = fixture(vec![reply(message.clone(), finish, 10)]).await;
        let result = f.run().await;
        assert_eq!(result["outcome"], "invalid_response", "{message}");
        assert_eq!(result["cause"]["kind"], kind, "{message}");
        assert_eq!(result["gradable"], true);
        assert_eq!(f.requests().await.len(), 1);
        let messages = f.events("message");
        assert_eq!(messages.last().unwrap()["message"], message);
        assert!(
            messages
                .iter()
                .all(|event| event["message"]["role"] != "tool")
        );
    }
}

#[tokio::test]
async fn an_aborted_generation_is_infrastructure_and_not_retried() {
    let f = fixture(vec![
        tool_reply(vec![list_dir("list")], 10),
        reply(
            json!({"role": "assistant", "content": "partial answ"}),
            "abort",
            10,
        ),
        final_reply(),
    ])
    .await;
    let (result, exit) = f.launch(None).await;
    assert_eq!(
        (result["outcome"].as_str(), exit),
        (Some("infrastructure"), 70)
    );
    assert_eq!(result["cause"]["kind"], "endpoint_response");
    assert_eq!(f.requests().await.len(), 2);
}

#[tokio::test]
async fn a_generation_that_errored_mid_reply_is_infrastructure_and_not_retried() {
    let failed = ResponseTemplate::new(200).set_body_json(json!({
        "id": "chatcmpl-test",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "partial answ"},
            "finish_reason": "error",
            "error": {"code": 502, "message": "Upstream error from host"}
        }],
        "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20}
    }));
    let f = fixture(vec![
        tool_reply(vec![list_dir("list")], 10),
        failed,
        final_reply(),
    ])
    .await;
    let (result, exit) = f.launch(None).await;
    assert_eq!(
        (result["outcome"].as_str(), exit),
        (Some("infrastructure"), 70)
    );
    assert_eq!(result["cause"]["kind"], "endpoint_response");
    assert_eq!(result["gradable"], false);
    assert_eq!(f.requests().await.len(), 2);
}

#[tokio::test]
async fn duplicate_ids_in_one_reply_reject_the_batch_before_writing() {
    let edits = ["first.txt", "second.txt"]
        .into_iter()
        .map(|file| {
            call(
                "same-id",
                "search_replace",
                json!({"file_path": file, "old_string": "", "new_string": "must not exist"}),
            )
        })
        .collect();
    let f = fixture(vec![tool_reply(edits, 10)]).await;
    let result = f.run().await;
    assert_eq!(
        result["cause"],
        json!({"kind": "duplicate_call_id", "id": "same-id"})
    );
    assert!(!f.workspace().join("first.txt").exists());
    assert!(!f.workspace().join("second.txt").exists());
}

#[tokio::test]
async fn call_ids_may_repeat_across_turns() {
    let todo = |status| {
        call(
            "functions.todo_write:0",
            "todo_write",
            json!({"todos": [{"id": "work", "content": "verify", "status": status}]}),
        )
    };
    let f = fixture(vec![
        tool_reply(vec![todo("in_progress")], 10),
        tool_reply(vec![todo("completed")], 10),
        final_reply(),
    ])
    .await;
    assert_eq!(f.run().await["outcome"], "completed");
    assert_eq!(f.requests().await.len(), 3);
}

#[tokio::test]
async fn tool_contract_errors_return_to_the_model_without_side_effects() {
    let errors = vec![
        call("unknown", "task", json!({})),
        call(
            "bad-args",
            "search_replace",
            json!({"file_path": "result.txt", "old_string": "", "new_string": "must not exist", "replace_all": "yes"}),
        ),
        json!({"id": "bad-json", "type": "function", "function": {"name": "list_dir", "arguments": "{"}}),
    ];
    let f = fixture(vec![tool_reply(errors, 10), final_reply()]).await;
    let result = f.run().await;
    assert_eq!(result["outcome"], "completed");
    assert_eq!(
        result["tool_calls"],
        json!({"total": 3, "ok": 0, "failed": 0, "invalid_arguments": 2, "unknown_tool": 1})
    );
    let requests = f.requests().await;
    let tool_messages: Vec<_> = requests[1]["messages"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|message| message["role"] == "tool")
        .collect();
    assert_eq!(tool_messages.len(), 3);
    assert!(tool_messages.iter().all(|message| {
        message["content"]
            .as_str()
            .unwrap()
            .starts_with("Tool error:")
    }));
    let kinds: Vec<_> = f
        .events("tool_call")
        .iter()
        .map(|event| event["class"].clone())
        .collect();
    assert_eq!(
        kinds,
        [
            json!("unknown_tool"),
            json!("invalid_arguments"),
            json!("invalid_arguments")
        ]
    );
    assert!(!f.workspace().join("result.txt").exists());
}

#[tokio::test]
async fn tool_results_keep_call_order_while_the_calls_run_at_once() {
    let calls = vec![
        call(
            "wait",
            "run_terminal_cmd",
            json!({"command": "until [ -e order.txt ]; do sleep 0.05; done; sleep 0.3; printf waited", "description": "wait for the file", "block_until_ms": 5000}),
        ),
        call(
            "create",
            "search_replace",
            json!({"file_path": "order.txt", "old_string": "", "new_string": "first\n"}),
        ),
        call(
            "edit",
            "search_replace",
            json!({"file_path": "./sub/../order.txt", "old_string": "first", "new_string": "second"}),
        ),
        call("read", "read_file", json!({"target_file": "order.txt"})),
    ];
    let f = fixture(vec![tool_reply(calls, 10), final_reply()]).await;
    std::fs::create_dir(f.workspace().join("sub")).unwrap();
    assert_eq!(f.run().await["outcome"], "completed");
    let requests = f.requests().await;
    let tools: Vec<_> = requests[1]["messages"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|message| message["role"] == "tool")
        .collect();
    let ids: Vec<_> = tools
        .iter()
        .map(|message| message["tool_call_id"].as_str().unwrap())
        .collect();
    assert_eq!(ids, ["wait", "create", "edit", "read"]);
    assert!(tools.iter().all(|message| {
        !message["content"]
            .as_str()
            .unwrap()
            .starts_with("Tool error")
    }));
    assert!(
        tools[0]["content"].as_str().unwrap().contains("waited"),
        "{}",
        tools[0]["content"]
    );
    assert_eq!(tools[3]["content"], "1→second\n");
    let traced: Vec<_> = f
        .events("tool_call")
        .iter()
        .map(|event| event["call_id"].as_str().unwrap().to_owned())
        .collect();
    assert_eq!(traced, ids);
}

#[tokio::test]
async fn compaction_is_off_unless_configured() {
    let f = fixture(vec![
        tool_reply(vec![list_dir("list")], 10_000_000),
        final_reply(),
    ])
    .await;
    let result = f.run().await;
    assert_eq!(result["outcome"], "completed");
    assert_eq!(result["compactions"], 0);
    assert!(
        f.requests()
            .await
            .iter()
            .all(|request| request["tool_choice"] == "auto")
    );
}

#[tokio::test]
async fn compaction_at_85_percent_replaces_the_history_with_the_summary_and_reminder() {
    let todos = call(
        "todo",
        "todo_write",
        json!({"todos": [
            {"id": "work", "content": "verify result", "status": "in_progress"},
            {"id": "setup", "content": "create result", "status": "completed"}
        ]}),
    );
    let server = call(
        "server",
        "run_terminal_cmd",
        json!({"command": "exec sleep 60", "description": "start a server", "block_until_ms": 0}),
    );
    let summary = "<summary>\n1. Primary Request and Intent: create result.txt\n9. Optional Next Step: verify it\n</summary>";
    let mut f = fixture(vec![
        tool_reply(vec![todos, server], 32_000),
        tool_reply(vec![list_dir("second")], 33_990),
        reply(
            json!({"role": "assistant", "content": summary}),
            "stop",
            100,
        ),
        final_reply(),
    ])
    .await;
    f.config["context_window_tokens"] = json!(40_000);
    f.config["sampling_params"]["max_tokens"] = json!(1000);
    f.config["compaction"] = json!({"max_tokens": 1024});
    let result = f.run().await;
    let server = result["background_tasks_left_running"][0]["pid"]
        .as_i64()
        .unwrap();
    unsafe { libc::kill(server as i32, libc::SIGKILL) };
    assert_eq!(result["outcome"], "completed", "{result}");
    assert_eq!(result["compactions"], 1);
    assert_eq!(result["turns"], 3);
    let requests = f.requests().await;
    assert_eq!(requests.len(), 4);
    let summary_request = &requests[2];
    assert_eq!(summary_request["tool_choice"], "none");
    assert_eq!(summary_request["max_tokens"], 1024);
    let before = summary_request["messages"].as_array().unwrap();
    assert_eq!(before.len(), 8);
    let prompt = before[7]["content"].as_str().unwrap();
    assert!(prompt.starts_with("Your task is to produce a faithful, concise summary"));
    assert!(prompt.contains("a few thousand words.\n\nCRITICAL:"));
    assert!(prompt.ends_with("from the conversation text only.\n"));
    let after = requests[3]["messages"].as_array().unwrap();
    assert_eq!(after.len(), 3);
    assert_eq!(after[..2], before[..2]);
    assert_eq!(
        after[2],
        json!({"role": "user", "content": "This session is being continued from a previous conversation that ran out of context. The summary below covers the earlier portion of the conversation.\n\nSummary:\n1. Primary Request and Intent: create result.txt\n9. Optional Next Step: verify it\n\n<system-reminder>\n## Running Background Tasks\nThese tasks are still running:\n- \"task-1\": `exec sleep 60` (running, run_terminal_cmd)\n\n## TODO List\nThis is your task list from before the conversation was compacted — it is still active. Keep working through the items below and update their status as you make progress:\n- [in_progress] work: verify result\n(1 completed)\n</system-reminder>"})
    );
    let compaction = &f.events("compaction")[0];
    assert_eq!(
        (
            &compaction["from_segment"],
            &compaction["to_segment"],
            &compaction["summary_call"]
        ),
        (&json!(0), &json!(1), &json!(3))
    );
    let segment: Vec<_> = f
        .events("message")
        .into_iter()
        .filter(|event| event["segment"] == 1)
        .map(|event| event["message"].clone())
        .collect();
    assert_eq!(segment[..3], after[..]);
}

#[tokio::test]
async fn without_a_reminder_the_continuation_ends_with_the_summary() {
    let mut f = fixture(vec![
        tool_reply(vec![list_dir("first")], 30_000),
        reply(
            json!({"role": "assistant", "content": "<summary>Listed the workspace.</summary>"}),
            "stop",
            100,
        ),
        final_reply(),
    ])
    .await;
    f.config["compaction"] = json!({"threshold_tokens": 20_000, "max_tokens": 1024});
    assert_eq!(f.run().await["compactions"], 1);
    assert_eq!(
        f.requests().await[2]["messages"][2]["content"],
        "This session is being continued from a previous conversation that ran out of context. The summary below covers the earlier portion of the conversation.\n\nSummary:\nListed the workspace."
    );
}

#[tokio::test]
async fn compaction_runs_before_a_turn_that_would_not_fit() {
    let mut f = fixture(vec![
        tool_reply(vec![list_dir("first")], 7_000),
        reply(
            json!({"role": "assistant", "content": "<summary>Listed the workspace.</summary>"}),
            "stop",
            100,
        ),
        final_reply(),
    ])
    .await;
    f.config["context_window_tokens"] = json!(10_000);
    f.config["sampling_params"]["max_tokens"] = json!(4_000);
    f.config["compaction"] = json!({"max_tokens": 1024});
    let result = f.run().await;
    assert_eq!(result["outcome"], "completed", "{result}");
    assert_eq!(result["compactions"], 1);
}

#[tokio::test]
async fn after_compaction_the_budget_never_exceeds_the_new_segment_bytes() {
    let mut f = fixture(vec![
        tool_reply(vec![list_dir("first")], 100),
        tool_reply(vec![list_dir("second")], 50_000),
        reply(
            json!({"role": "assistant", "content": "<summary>Listed the workspace twice.</summary>"}),
            "stop",
            59_000,
        ),
        final_reply(),
    ])
    .await;
    f.config["context_window_tokens"] = json!(60_000);
    f.config["sampling_params"]["max_tokens"] = json!(1000);
    f.config["compaction"] = json!({"threshold_tokens": 50_000, "max_tokens": 2048});
    let result = f.run().await;
    assert_eq!(result["outcome"], "completed", "{result}");
    assert_eq!(result["compactions"], 1);
}

#[tokio::test]
async fn an_unusable_summary_ends_the_attempt_and_keeps_the_transcript() {
    for (summary, finish) in [
        (
            json!({"role": "assistant", "content": "<summary></summary>"}),
            "stop",
        ),
        (
            json!({"role": "assistant", "content": "<analysis>reasoning only</analysis>"}),
            "stop",
        ),
        (
            json!({"role": "assistant", "content": "summary", "tool_calls": {}}),
            "stop",
        ),
        (
            json!({"role": "assistant", "content": null, "tool_calls": [list_dir("unexpected")]}),
            "tool_calls",
        ),
    ] {
        let assistant = json!({"role": "assistant", "content": null, "tool_calls": [call("edit", "search_replace", json!({"file_path": "result.txt", "old_string": "", "new_string": "kept"}))]});
        let mut f = fixture(vec![
            reply(assistant.clone(), "tool_calls", 30_000),
            reply(summary.clone(), finish, 10),
        ])
        .await;
        f.config["compaction"] = json!({"threshold_tokens": 20_000, "max_tokens": 1024});
        let result = f.run().await;
        assert_eq!(result["cause"]["kind"], "invalid_summary", "{summary}");
        assert_eq!(result["compactions"], 0);
        let requests = f.requests().await;
        assert_eq!(requests.len(), 2);
        assert_eq!(requests[1]["messages"][2], assistant);
        assert_eq!(requests[1]["messages"][3]["role"], "tool");
        let messages = f.events("message");
        assert_eq!(messages.last().unwrap()["message"], summary);
        assert!(f.events("compaction").is_empty());
        assert_eq!(
            std::fs::read_to_string(f.workspace().join("result.txt")).unwrap(),
            "kept"
        );
    }
}

#[tokio::test]
async fn a_summary_cut_off_at_max_tokens_is_kept() {
    let cut = "<summary>\n1. Primary Request and Intent: create result.txt\n8. Current Work: wri";
    let mut f = fixture(vec![
        tool_reply(vec![list_dir("first")], 30_000),
        reply(json!({"role": "assistant", "content": cut}), "length", 100),
        final_reply(),
    ])
    .await;
    f.config["compaction"] = json!({"threshold_tokens": 20_000, "max_tokens": 1024});
    let result = f.run().await;
    assert_eq!(result["outcome"], "completed", "{result}");
    assert_eq!(result["compactions"], 1);
    let continuation = &f.requests().await[2]["messages"][2]["content"];
    assert!(
        continuation
            .as_str()
            .unwrap()
            .ends_with("\n1. Primary Request and Intent: create result.txt\n8. Current Work: wri"),
        "{continuation}"
    );
}

async fn compaction_at_32k(tokens: u64, compaction_max_tokens: u64) -> Fixture {
    let mut f = fixture(vec![
        tool_reply(vec![list_dir("first")], tokens),
        reply(
            json!({"role": "assistant", "content": "<summary>Listed the workspace.</summary>"}),
            "stop",
            100,
        ),
        final_reply(),
    ])
    .await;
    f.config["context_window_tokens"] = json!(32_768);
    f.config["compaction"] = json!({"max_tokens": compaction_max_tokens});
    f
}

#[tokio::test]
async fn the_summary_request_is_sized_to_fit_the_window() {
    let f = compaction_at_32k(28_000, 4096).await;
    let result = f.run().await;
    assert_eq!(result["outcome"], "completed", "{result}");
    let summary_request = &f.requests().await[1];
    let appended: usize = summary_request["messages"].as_array().unwrap()[3..]
        .iter()
        .map(|message| serde_json::to_vec(message).unwrap().len())
        .sum();
    let estimate = 28_010 + appended as u64 / 4;
    assert_eq!(
        summary_request["max_tokens"],
        32_768 - 32_768 / 20 - estimate
    );
}

#[tokio::test]
async fn compaction_leaves_room_for_the_summary_request() {
    let f = compaction_at_32k(24_000, 8192).await;
    let result = f.run().await;
    assert_eq!(result["compactions"], 1, "{result}");
    assert_eq!(f.requests().await[1]["tool_choice"], "none");
}

#[tokio::test]
async fn a_summary_request_that_cannot_fit_is_never_sent() {
    let f = compaction_at_32k(31_000, 4096).await;
    let result = f.run().await;
    assert_eq!(result["outcome"], "context_overflow", "{result}");
    assert_eq!(result["cause"]["kind"], "context_window");
    assert_eq!(result["cause"]["max_tokens"], 4096);
    assert_eq!(f.requests().await.len(), 1);

    let mut f = fixture(vec![
        tool_reply(vec![list_dir("first")], 100),
        tool_reply(vec![list_dir("second")], 31_000),
        final_reply(),
    ])
    .await;
    f.config["context_window_tokens"] = json!(32_768);
    f.config["compaction"] = json!({"max_tokens": 4096});
    let result = f.run().await;
    assert_eq!(result["cause"]["kind"], "context_window", "{result}");
    assert_eq!(result["compactions"], 0);
    assert_eq!(f.requests().await.len(), 2);
    assert!(f.events("compaction").is_empty());
    assert_eq!(
        f.events("message").last().unwrap()["message"]["role"],
        "tool"
    );
}

fn write_big_file(f: &Fixture) {
    let line = "a line of tool output that the model reads again and again\n";
    std::fs::write(f.workspace().join("big.txt"), line.repeat(1000)).unwrap();
}

fn read_big_file(id: &str) -> Value {
    call(id, "read_file", json!({"target_file": "big.txt"}))
}

fn summary_reply() -> ResponseTemplate {
    reply(
        json!({"role": "assistant", "content": "<summary>Listed the workspace.</summary>"}),
        "stop",
        100,
    )
}

fn refusal() -> ResponseTemplate {
    ResponseTemplate::new(400).set_body_string(
        "This model's maximum context length is 32768 tokens. However, you requested 40000 tokens.",
    )
}

#[tokio::test]
async fn a_summary_request_that_does_not_fit_drops_the_oldest_turns_first() {
    let mut f = fixture(vec![
        tool_reply(vec![read_big_file("read")], 100),
        tool_reply(vec![list_dir("second")], 29_000),
        summary_reply(),
        final_reply(),
    ])
    .await;
    write_big_file(&f);
    f.config["context_window_tokens"] = json!(32_768);
    f.config["compaction"] = json!({"max_tokens": 4096});
    let result = f.run().await;
    assert_eq!(result["outcome"], "completed", "{result}");
    assert_eq!(result["compactions"], 1);
    let requests = f.requests().await;
    assert_eq!(requests.len(), 4);
    let summary_request = &requests[2];
    assert_eq!(summary_request["tool_choice"], "none");
    assert_eq!(summary_request["max_tokens"], 4096);
    let history: Vec<Value> = f
        .events("message")
        .into_iter()
        .filter(|event| event["segment"] == 0)
        .map(|event| event["message"].clone())
        .collect();
    assert_eq!(history.len(), 8);
    assert_eq!(history[2]["tool_calls"][0]["id"], "read");
    let sent = summary_request["messages"].as_array().unwrap();
    assert_eq!(sent.len(), 5);
    assert_eq!(sent[..2], history[..2]);
    assert_eq!(sent[2..], history[4..7]);
    assert_eq!(sent[2]["tool_calls"][0]["id"], "second");
    let summary_call = &f.events("model_call")[2];
    assert_eq!(summary_call["kind"], "compaction");
    assert_eq!(
        summary_call["prompt"],
        json!({"segment": 0, "end": 7, "omitted": [[2, 4]]})
    );
    assert_eq!(f.events("compaction")[0]["input"], "fitted");
    assert_eq!(rebuilt_requests(&f.trace()), f.raw_requests().await);

    let mut f = fixture(vec![
        tool_reply(vec![list_dir("first")], 21_000),
        summary_reply(),
        tool_reply(vec![read_big_file("read")], 100),
        tool_reply(vec![list_dir("second")], 29_000),
        summary_reply(),
        final_reply(),
    ])
    .await;
    write_big_file(&f);
    f.config["context_window_tokens"] = json!(32_768);
    f.config["compaction"] = json!({"threshold_tokens": 20_000, "max_tokens": 4096});
    assert_eq!(f.run().await["compactions"], 2);
    assert_eq!(
        f.events("model_call")[4]["prompt"],
        json!({"segment": 1, "end": 8, "omitted": [[3, 5]]})
    );
    assert_eq!(rebuilt_requests(&f.trace()), f.raw_requests().await);
}

#[tokio::test]
async fn a_summary_request_refused_for_its_length_is_trimmed_to_70_percent_and_sent_again() {
    let compacting = |f: &mut Fixture| {
        f.config["context_window_tokens"] = json!(32_768);
        f.config["compaction"] = json!({"threshold_tokens": 20_000, "max_tokens": 4096});
    };

    let mut f = fixture(vec![
        tool_reply(vec![read_big_file("read")], 100),
        tool_reply(vec![list_dir("second")], 24_000),
        refusal(),
        summary_reply(),
        final_reply(),
    ])
    .await;
    write_big_file(&f);
    compacting(&mut f);
    let result = f.run().await;
    assert_eq!(result["outcome"], "completed", "{result}");
    assert_eq!(result["compactions"], 1);
    let requests = f.requests().await;
    assert_eq!(requests.len(), 5);
    assert_eq!(requests[2]["messages"].as_array().unwrap().len(), 7);
    assert_eq!(requests[3]["messages"].as_array().unwrap().len(), 5);
    assert_eq!(requests[3]["max_tokens"], 4096);
    let calls = f.events("model_call");
    assert_eq!(calls.len(), 4);
    assert_eq!(
        calls[2]["prompt"],
        json!({"segment": 0, "end": 7, "omitted": [[2, 4]]})
    );
    assert_eq!(f.events("endpoint_error")[0]["call"], 3);
    assert_eq!(calls[2]["call"], 4);
    let compaction = &f.events("compaction")[0];
    assert_eq!(compaction["input"], "lossy");
    assert_eq!(compaction["summary_call"], 4);
    let mut accepted = f.raw_requests().await;
    accepted.remove(2);
    assert_eq!(rebuilt_requests(&f.trace()), accepted);

    let mut f = fixture(vec![
        tool_reply(vec![read_big_file("read")], 100),
        tool_reply(vec![list_dir("second")], 24_000),
        refusal(),
        refusal(),
        final_reply(),
    ])
    .await;
    write_big_file(&f);
    compacting(&mut f);
    let result = f.run().await;
    assert_eq!(result["cause"]["kind"], "endpoint_rejected", "{result}");
    assert_eq!(result["compactions"], 0);
    assert_eq!(f.requests().await.len(), 4);

    let mut f = fixture(vec![
        tool_reply(vec![list_dir("only")], 24_000),
        refusal(),
        final_reply(),
    ])
    .await;
    compacting(&mut f);
    let result = f.run().await;
    assert_eq!(result["cause"]["kind"], "endpoint_rejected", "{result}");
    assert_eq!(f.requests().await.len(), 2);

    let mut f = fixture(vec![
        tool_reply(vec![read_big_file("read")], 100),
        tool_reply(vec![list_dir("second")], 21_000),
        refusal(),
        summary_reply(),
        final_reply(),
    ])
    .await;
    write_big_file(&f);
    compacting(&mut f);
    f.config["compaction"]["max_tokens"] = json!(10_000);
    let result = f.run().await;
    assert_eq!(result["cause"]["kind"], "endpoint_rejected", "{result}");
    let requests = f.requests().await;
    assert_eq!(requests.len(), 3);
    assert_eq!(requests[2]["messages"].as_array().unwrap().len(), 5);
}

#[tokio::test]
async fn unprocessed_requests_are_retried_with_identical_bytes() {
    let f = fixture(vec![
        ResponseTemplate::new(503).set_body_string("UpstreamGenerationAbortedError"),
        ResponseTemplate::new(429)
            .insert_header("Retry-After", "1")
            .set_body_string("slow down"),
        final_reply(),
    ])
    .await;
    let started = Instant::now();
    assert_eq!(f.run().await["outcome"], "completed");
    assert!(started.elapsed() >= Duration::from_secs(1));
    let raw = f.raw_requests().await;
    assert_eq!(raw.len(), 3);
    assert!(raw.iter().all(|body| *body == raw[0]));
    let errors = f.events("endpoint_error");
    assert_eq!(errors.len(), 2);
    assert_eq!(errors[0]["status"], 503);
    assert_eq!(errors[0]["body_excerpt"], "UpstreamGenerationAbortedError");
    assert_eq!(errors[1]["status"], 429);
    assert!(errors[1]["retry_in_ms"].as_u64().unwrap() >= 1000);
    assert_eq!(f.events("model_call")[0]["attempts"], 3);
}

#[tokio::test]
async fn endpoint_errors_keep_their_body_and_are_never_retried_once_processed() {
    for status in [500, 502, 504, 401, 404] {
        let body = format!("{{\"error\":\"upstream {status}\"}}");
        let f = fixture(vec![
            tool_reply(vec![list_dir("list")], 10),
            ResponseTemplate::new(status).set_body_string(body.clone()),
        ])
        .await;
        let result = f.run().await;
        assert_eq!(
            result["cause"],
            json!({"kind": "endpoint_status", "status": status, "body_excerpt": body})
        );
        assert_eq!(result["turns"], 1);
        assert_eq!(f.requests().await.len(), 2);
        let error = &f.events("endpoint_error")[0];
        assert_eq!(
            (
                &error["status"],
                &error["body_excerpt"],
                &error["retry_in_ms"]
            ),
            (&json!(status), &json!(body), &Value::Null)
        );
    }
}

#[tokio::test]
async fn error_bodies_are_cut_at_64_kib() {
    let f = fixture(vec![
        ResponseTemplate::new(500).set_body_string("x".repeat(200_000)),
    ])
    .await;
    let result = f.run().await;
    assert_eq!(
        result["cause"]["body_excerpt"].as_str().unwrap().len(),
        64 * 1024
    );
}

#[tokio::test]
async fn an_endpoint_context_length_rejection_is_a_graded_overflow() {
    let body = "This model's maximum context length is 131072 tokens. However, you requested 140000 tokens.";
    let f = fixture(vec![
        tool_reply(vec![list_dir("list")], 10),
        ResponseTemplate::new(400).set_body_string(body),
    ])
    .await;
    let result = f.run().await;
    assert_eq!(
        result["cause"],
        json!({"kind": "endpoint_rejected", "status": 400, "body_excerpt": body})
    );
    assert_eq!(result["outcome"], "context_overflow");
}

#[tokio::test]
async fn a_turn_refused_for_its_length_is_compacted_once_and_sent_again() {
    let summary = summary_reply;
    let compacting = |responses| async {
        let mut f = fixture(responses).await;
        f.config["compaction"] = json!({"threshold_tokens": 20_000, "max_tokens": 1024});
        f
    };

    let f = compacting(vec![
        tool_reply(vec![list_dir("first")], 100),
        refusal(),
        summary(),
        final_reply(),
    ])
    .await;
    let result = f.run().await;
    assert_eq!(result["outcome"], "completed", "{result}");
    assert_eq!(result["compactions"], 1);
    let requests = f.requests().await;
    assert_eq!(requests.len(), 4);
    assert_eq!(requests[2]["tool_choice"], "none");
    assert_eq!(requests[3]["messages"].as_array().unwrap().len(), 3);

    let f = compacting(vec![
        tool_reply(vec![list_dir("first")], 100),
        refusal(),
        summary(),
        refusal(),
        final_reply(),
    ])
    .await;
    let result = f.run().await;
    assert_eq!(result["cause"]["kind"], "endpoint_rejected", "{result}");
    assert_eq!(result["compactions"], 1);
    assert_eq!(f.requests().await.len(), 4);

    let f = compacting(vec![refusal(), summary(), final_reply()]).await;
    let result = f.run().await;
    assert_eq!(result["cause"]["kind"], "config", "{result}");
    assert_eq!(f.requests().await.len(), 1);
}

#[tokio::test]
async fn a_full_context_window_ends_the_attempt_before_sending() {
    let mut f = fixture(vec![tool_reply(vec![list_dir("list")], 990), final_reply()]).await;
    f.config["context_window_tokens"] = json!(1200);
    f.config["sampling_params"]["max_tokens"] = json!(256);
    let result = f.run().await;
    assert_eq!(result["outcome"], "context_overflow");
    assert_eq!(result["cause"]["kind"], "context_window");
    assert!(result["cause"]["estimated_prompt_tokens"].as_u64().unwrap() > 1000);
    assert_eq!(f.requests().await.len(), 1);
}

#[tokio::test]
async fn an_overflow_before_the_first_reply_is_a_config_error() {
    let mut f = fixture(vec![final_reply()]).await;
    f.config["context_window_tokens"] = json!(8192);
    f.config["sampling_params"]["max_tokens"] = json!(16_384);
    let result = f.run().await;
    assert_eq!(result["outcome"], "infrastructure", "{result}");
    assert_eq!(result["cause"]["kind"], "config");
    assert!(
        result["cause"]["detail"]
            .as_str()
            .unwrap()
            .contains("context_window")
    );
    assert!(f.requests().await.is_empty());

    let body =
        "This model's maximum context length is 8192 tokens. However, you requested 16384 tokens.";
    let f = fixture(vec![ResponseTemplate::new(400).set_body_string(body)]).await;
    let result = f.run().await;
    assert_eq!(result["outcome"], "infrastructure", "{result}");
    assert_eq!(result["cause"]["kind"], "config");
    assert!(result["cause"]["detail"].as_str().unwrap().contains(body));
}

#[tokio::test]
async fn budgeting_without_usage_is_an_endpoint_fault() {
    let mut f = fixture(vec![
        ResponseTemplate::new(200).set_body_json(json!({
            "choices": [{"message": {"role": "assistant", "content": null, "tool_calls": [list_dir("list")]}, "finish_reason": "tool_calls"}]
        })),
        final_reply(),
    ])
    .await;
    f.config["context_window_tokens"] = json!(100_000);
    let result = f.run().await;
    assert_eq!(result["cause"]["kind"], "endpoint_response");
    assert_eq!(result["usage"]["complete"], false);
}

#[tokio::test]
async fn a_connection_lost_after_send_is_not_retried() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let accepted = Arc::new(AtomicUsize::new(0));
    let count = accepted.clone();
    tokio::spawn(async move {
        loop {
            let (mut socket, _) = listener.accept().await.unwrap();
            count.fetch_add(1, Ordering::SeqCst);
            let mut request = [0; 1024];
            let _ = socket.read(&mut request).await;
        }
    });
    let mut f = fixture(vec![]).await;
    f.config["base_url"] = json!(format!("http://{address}/v1"));
    let result = f.run().await;
    assert_eq!(result["cause"]["kind"], "connection_lost");
    assert_eq!(f.events("endpoint_error")[0]["failure"], "lost");
    assert_eq!(accepted.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn an_unreachable_endpoint_is_retried_until_the_next_wait_would_pass_the_deadline() {
    let port = std::net::TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port();
    let mut f = fixture(vec![]).await;
    f.config["base_url"] = json!(format!("http://127.0.0.1:{port}/v1"));
    f.config["deadline_secs"] = json!(3);
    let result = f.run().await;
    assert_eq!(result["cause"]["kind"], "endpoint_unavailable");
    assert_eq!(result["cause"]["status"], Value::Null);
    let errors = f.events("endpoint_error");
    assert!(errors.len() >= 2);
    assert_eq!(result["cause"]["attempts"], errors.len());
    assert!(errors.iter().all(|error| error["failure"] == "connect"));
    assert_eq!(errors.last().unwrap()["retry_in_ms"], Value::Null);
}

#[tokio::test]
async fn a_deadline_during_a_request_is_always_a_graded_deadline() {
    async fn late_first_reply() -> (Value, u8) {
        let mut f = fixture(vec![final_reply().set_delay(Duration::from_secs(4))]).await;
        f.config["deadline_secs"] = json!(1);
        let ran = f.launch(None).await;
        assert!(f.session().join("result.json").exists());
        ran
    }
    let runs = tokio::join!(
        late_first_reply(),
        late_first_reply(),
        late_first_reply(),
        late_first_reply(),
        late_first_reply(),
        late_first_reply(),
        late_first_reply(),
        late_first_reply(),
    );
    for (result, exit) in [
        runs.0, runs.1, runs.2, runs.3, runs.4, runs.5, runs.6, runs.7,
    ] {
        assert_eq!(result["outcome"], "deadline", "{result}");
        assert_eq!(result["cause"], json!({"request_in_flight": true}));
        assert_eq!(exit, 0);
    }
}

#[tokio::test]
async fn a_wait_past_the_deadline_ends_the_attempt_at_the_deadline() {
    let mut f = fixture(vec![tool_reply(
        vec![call(
            "wait",
            "run_terminal_cmd",
            json!({"command": "sleep 60", "description": "wait", "block_until_ms": 36_000_000}),
        )],
        10,
    )])
    .await;
    f.config["deadline_secs"] = json!(1);
    let result = f.run().await;
    assert_eq!(result["outcome"], "deadline");
    assert_eq!(result["cause"], json!({"request_in_flight": false}));
    assert!(result["wall_time_ms"].as_u64().unwrap() < 3_000, "{result}");
}

#[tokio::test]
async fn infrastructure_outcomes_kill_background_processes() {
    let f = fixture(vec![
        tool_reply(vec![call("background", "run_terminal_cmd", json!({
            "command": "printf '%s' $$ > tracked.pid; exec sleep 60", "description": "start a server", "block_until_ms": 0
        }))], 10),
        ResponseTemplate::new(500)
            .set_body_string("upstream broke")
            .set_delay(Duration::from_secs(1)),
    ]).await;
    let result = f.run().await;
    assert_eq!(result["outcome"], "infrastructure");
    assert_eq!(result["background_tasks_left_running"], json!([]));
    assert!(!alive(&f.workspace().join("tracked.pid")));
}

#[tokio::test]
async fn gradable_outcomes_leave_background_processes_running() {
    let mut f = fixture(vec![
        tool_reply(vec![call("server", "run_terminal_cmd", json!({
            "command": "printf '%s' $$ > tracked.pid; exec sleep 60", "description": "start a server", "block_until_ms": 0
        }))], 10),
        final_reply().set_delay(Duration::from_secs(5)),
    ]).await;
    f.config["deadline_secs"] = json!(1);
    let result = f.run().await;
    assert_eq!(result["outcome"], "deadline");
    let pid_file = f.workspace().join("tracked.pid");
    let survived = alive(&pid_file);
    let pid: i32 = std::fs::read_to_string(&pid_file).unwrap().parse().unwrap();
    unsafe { libc::kill(pid, libc::SIGKILL) };
    assert!(survived);
    assert_eq!(
        result["background_tasks_left_running"][0]["task_id"],
        "task-1"
    );
}

#[tokio::test]
async fn command_logs_stay_in_the_sandbox_temp_dir_outside_the_session_dir() {
    let f = fixture(vec![
        tool_reply(
            vec![call(
                "server",
                "run_terminal_cmd",
                json!({
                    "command": "exec sleep 60", "description": "start a server", "block_until_ms": 0
                }),
            )],
            10,
        ),
        final_reply(),
    ])
    .await;
    let result = f.run().await;
    let left = &result["background_tasks_left_running"][0];
    unsafe { libc::kill(left["pid"].as_i64().unwrap() as i32, libc::SIGKILL) };
    let log = PathBuf::from(left["output_file"].as_str().unwrap());
    assert!(
        log.starts_with(std::env::temp_dir().join(".remanie-terminal")),
        "{log:?}"
    );
    assert!(log.exists());
    assert!(!f.session().join("terminal").exists());
}

#[tokio::test]
async fn command_logs_leave_a_workspace_that_is_the_temp_dir() {
    let mut f = fixture(vec![
        tool_reply(
            vec![call(
                "server",
                "run_terminal_cmd",
                json!({
                    "command": "exec sleep 60", "description": "start a server", "block_until_ms": 0
                }),
            )],
            10,
        ),
        final_reply(),
    ])
    .await;
    f.config["workspace"] = json!(std::env::temp_dir());
    let result = f.run().await;
    let left = &result["background_tasks_left_running"][0];
    unsafe { libc::kill(left["pid"].as_i64().unwrap() as i32, libc::SIGKILL) };
    let log = PathBuf::from(left["output_file"].as_str().unwrap());
    assert!(log.starts_with("/var/tmp/.remanie-terminal"), "{log:?}");
    assert!(log.exists());
    std::fs::remove_dir_all(log.parent().unwrap()).unwrap();
}

#[tokio::test]
async fn every_request_is_rebuilt_byte_for_byte_from_the_trace() {
    let todo = call(
        "todo",
        "todo_write",
        json!({"todos": [{"id": "work", "content": "verify", "status": "in_progress"}]}),
    );
    let mut f = fixture(vec![
        tool_reply(vec![todo], 100),
        tool_reply(vec![list_dir("list")], 30_000),
        ResponseTemplate::new(503).set_body_string("aborted for a weight update"),
        reply(
            json!({"role": "assistant", "content": "<summary>Listed the workspace.</summary>"}),
            "stop",
            100,
        ),
        tool_reply(vec![list_dir("again")], 100),
        final_reply(),
    ])
    .await;
    f.config["compaction"] = json!({"threshold_tokens": 20_000, "max_tokens": 1024});
    f.config["max_turns"] = json!(8);
    assert_eq!(f.run().await["outcome"], "completed");
    let trace = f.trace();
    assert_eq!(
        trace
            .iter()
            .filter(|event| event["event"] == "message" && event["index"] == 0)
            .count(),
        2
    );
    assert_eq!(rebuilt_requests(&trace), f.raw_requests().await);
}

fn rebuilt_requests(trace: &[Value]) -> Vec<Vec<u8>> {
    let header = &trace[0];
    assert_eq!(header["event"], "header");
    let mut segments: BTreeMap<u64, Vec<Value>> = BTreeMap::new();
    let mut rebuilt = Vec::new();
    for event in trace {
        match event["event"].as_str().unwrap() {
            "message" => {
                let segment = segments
                    .entry(event["segment"].as_u64().unwrap())
                    .or_default();
                assert_eq!(event["index"], segment.len());
                segment.push(event["message"].clone());
            }
            "model_call" => {
                let prompt = &event["prompt"];
                let segment = &segments[&prompt["segment"].as_u64().unwrap()];
                let omitted: Vec<[usize; 2]> = prompt
                    .get("omitted")
                    .map(|ranges| serde_json::from_value(ranges.clone()).unwrap())
                    .unwrap_or_default();
                let messages: Vec<Value> = segment[..prompt["end"].as_u64().unwrap() as usize]
                    .iter()
                    .enumerate()
                    .filter(|(index, _)| {
                        !omitted.iter().any(|[from, to]| (from..to).contains(&index))
                    })
                    .map(|(_, message)| message.clone())
                    .collect();
                let mut body = header["config"]["sampling_params"]
                    .as_object()
                    .unwrap()
                    .clone();
                if let Some(max_tokens) = event.get("max_tokens_override") {
                    body.insert("max_tokens".into(), max_tokens.clone());
                }
                body.insert("model".into(), header["config"]["model"].clone());
                body.insert("messages".into(), Value::Array(messages));
                let tool_choice = if event["kind"] == "compaction" {
                    "none"
                } else {
                    "auto"
                };
                body.insert("tools".into(), header["tools"].clone());
                body.insert("tool_choice".into(), json!(tool_choice));
                body.insert("stream".into(), json!(false));
                let bytes = serde_json::to_vec(&body).unwrap();
                for _ in 0..event["attempts"].as_u64().unwrap() {
                    rebuilt.push(bytes.clone());
                }
            }
            _ => {}
        }
    }
    rebuilt
}

#[tokio::test]
async fn the_trace_grows_linearly_with_the_conversation() {
    let turns = 30;
    let mut responses: Vec<_> = (0..turns)
        .map(|turn| {
            tool_reply(
                vec![call(
                    &format!("read-{turn}"),
                    "read_file",
                    json!({"target_file": "big.txt"}),
                )],
                100,
            )
        })
        .collect();
    responses.push(final_reply());
    let mut f = fixture(responses).await;
    f.config["max_turns"] = json!(turns + 1);
    let line = "a line of tool output that the model reads again and again\n";
    std::fs::write(f.workspace().join("big.txt"), line.repeat(250)).unwrap();
    assert_eq!(f.run().await["outcome"], "completed");
    let last_request = f.raw_requests().await.pop().unwrap().len() as u64;
    let trace = std::fs::metadata(f.session().join("trace.jsonl"))
        .unwrap()
        .len();
    assert!(
        trace < 2 * last_request,
        "trace {trace} bytes for a {last_request}-byte conversation"
    );
}
