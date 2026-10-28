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
        let result = run(&input, |_| key.map(OsString::from)).await;
        let exit = result.outcome.exit_code();
        (serde_json::to_value(&result).unwrap(), exit)
    }

    /// Also checks that result.json, the trace's last line and the returned result agree.
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
            {"role": "user", "content": "<user_query>\nCreate result.txt and verify the result.\n</user_query>"}
        ])
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
            json!({"file_path": "result.txt", "old_string": "", "new_string": "must not exist", "extra": true}),
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
        .events("message")
        .iter()
        .filter_map(|event| event.get("tool").map(|tool| tool["error"].clone()))
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
async fn tool_calls_run_one_at_a_time_in_the_model_order() {
    let calls = vec![
        call(
            "write",
            "search_replace",
            json!({"file_path": "order.txt", "old_string": "", "new_string": "first\n"}),
        ),
        call("read-1", "read_file", json!({"target_file": "order.txt"})),
        call(
            "append",
            "run_terminal_cmd",
            json!({"command": "sleep 0.3; echo second >> order.txt", "description": "append", "block_until_ms": 10000}),
        ),
        call("read-2", "read_file", json!({"target_file": "order.txt"})),
    ];
    let f = fixture(vec![tool_reply(calls, 10), final_reply()]).await;
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
    assert_eq!(ids, ["write", "read-1", "append", "read-2"]);
    let content = |index: usize| tools[index]["content"].as_str().unwrap();
    assert!(content(1).contains("first") && !content(1).contains("second"));
    assert!(content(3).contains("second"));
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
async fn compaction_keeps_the_last_two_turns_and_the_todo_list_after_the_summary() {
    let todo = call(
        "todo",
        "todo_write",
        json!({"todos": [{"id": "work", "content": "verify result", "status": "in_progress"}]}),
    );
    let mut f = fixture(vec![
        tool_reply(vec![todo], 100),
        tool_reply(vec![list_dir("second")], 100),
        tool_reply(vec![list_dir("third")], 30_000),
        reply(
            json!({"role": "assistant", "content": "<summary>The work is half done.</summary>"}),
            "stop",
            100,
        ),
        final_reply(),
    ])
    .await;
    f.config["compaction"] = json!({"threshold_tokens": 20_000, "max_tokens": 1024});
    let result = f.run().await;
    assert_eq!(result["outcome"], "completed");
    assert_eq!(result["compactions"], 1);
    assert_eq!(result["turns"], 4);
    assert_eq!(result["model_calls"], 5);
    let requests = f.requests().await;
    assert_eq!(requests.len(), 5);
    let summary_request = &requests[3];
    assert_eq!(summary_request["tool_choice"], "none");
    assert_eq!(summary_request["max_tokens"], 1024);
    assert_eq!(summary_request["tools"], requests[0]["tools"]);
    let before = summary_request["messages"].as_array().unwrap();
    assert_eq!(before.len(), 9);
    assert_eq!(before[8]["role"], "user");
    let after = requests[4]["messages"].as_array().unwrap();
    assert_eq!(requests[4]["max_tokens"], 2048);
    assert_eq!(after[..2], before[..2]);
    let continuation = after[2]["content"].as_str().unwrap();
    assert!(continuation.contains("Summary:\nThe work is half done."));
    let state: Value =
        serde_json::from_str(continuation.split_once("Current tool state:\n").unwrap().1).unwrap();
    assert_eq!(state["todos"]["todos"]["work"]["status"], "in_progress");
    assert_eq!(after[3..], before[4..8]);
    assert_eq!(after.len(), 7);
    let compaction = &f.events("compaction")[0];
    assert_eq!(
        (
            &compaction["from_segment"],
            &compaction["to_segment"],
            &compaction["summary_call"],
            &compaction["kept_messages"]
        ),
        (&json!(0), &json!(1), &json!(4), &json!(4))
    );
    let segment: Vec<_> = f
        .events("message")
        .into_iter()
        .filter(|event| event["segment"] == 1)
        .map(|event| event["message"].clone())
        .collect();
    assert_eq!(segment[..7], after[..]);
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
            json!({"role": "assistant", "content": "unfinished summary"}),
            "length",
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
    let started = Instant::now();
    let result = f.run().await;
    assert!(started.elapsed() < Duration::from_secs(3));
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
#[ignore = "needs ToolSession::detach from v2/rust-tools"]
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
    let header = &trace[0];
    assert_eq!(header["event"], "header");
    let mut segments: BTreeMap<u64, Vec<Value>> = BTreeMap::new();
    let mut rebuilt = Vec::new();
    for event in &trace {
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
                let messages = &segments[&prompt["segment"].as_u64().unwrap()]
                    [..prompt["end"].as_u64().unwrap() as usize];
                let mut body = header["input"]["sampling_params"]
                    .as_object()
                    .unwrap()
                    .clone();
                body.insert("max_tokens".into(), event["max_tokens"].clone());
                body.insert("model".into(), header["input"]["model"].clone());
                body.insert("messages".into(), Value::Array(messages.to_vec()));
                body.insert("tools".into(), header["tools"].clone());
                body.insert("tool_choice".into(), event["tool_choice"].clone());
                body.insert("stream".into(), json!(false));
                let bytes = serde_json::to_vec(&body).unwrap();
                for _ in 0..event["attempts"].as_u64().unwrap() {
                    rebuilt.push(bytes.clone());
                }
            }
            _ => {}
        }
    }
    assert_eq!(segments.len(), 2);
    assert_eq!(rebuilt, f.raw_requests().await);
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
