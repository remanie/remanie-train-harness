//! The shell's `sampling` module re-exports the sampler crates; these are the items the vendored files import from it.

pub use self::error::SamplingError;
pub use xai_grok_sampler::ApiBackend;
pub use xai_grok_sampler::SamplingClient as Client;
pub use xai_grok_sampling_types::conversation::*;
pub use xai_grok_sampling_types::types::*;

pub mod error {
    // Copied from crates/codegen/xai-grok-shell/src/sampling/error.rs:274-337; the rest of that file needs shell-only types.
    use agent_client_protocol as acp;
    pub use xai_grok_sampling_types::error::*;

    pub const SERVICE_NAME_REWRITES: &[(&str, &str)] = &[
        ("cli-chat-proxy", "build backend"),
        ("cli_chat_proxy", "build backend"),
        ("inference-api", "inference backend"),
        ("inference_api", "inference backend"),
        ("research-api", "research backend"),
        ("research_api", "research backend"),
        ("grok-code-backend", "code backend"),
        ("grok_code_backend", "code backend"),
    ];

    pub fn rewrite_service_names(text: &str) -> String {
        let mut result = text.to_owned();
        for (pattern, replacement) in SERVICE_NAME_REWRITES {
            result = replace_ascii_case_insensitive(&result, pattern, replacement);
        }
        result
    }

    fn replace_ascii_case_insensitive(text: &str, pattern: &str, replacement: &str) -> String {
        if pattern.is_empty() {
            return text.to_owned();
        }
        let lower_text = text.to_ascii_lowercase();
        let lower_pattern = pattern.to_ascii_lowercase();
        let mut out = String::with_capacity(text.len());
        let mut idx = 0;
        while let Some(pos) = lower_text.get(idx..).and_then(|s| s.find(&lower_pattern)) {
            let start = idx + pos;
            if let Some(chunk) = text.get(idx..start) {
                out.push_str(chunk);
            }
            out.push_str(replacement);
            idx = start + pattern.len();
        }
        if let Some(tail) = text.get(idx..) {
            out.push_str(tail);
        }
        out
    }

    pub fn error_detail_from_data(data: &serde_json::Value) -> Option<String> {
        if let Some(m) = data.get("message").and_then(|v| v.as_str()) {
            return Some(m.to_owned());
        }
        if let Some(s) = data.as_str() {
            return Some(s.to_owned());
        }
        data.get("detail")
            .and_then(|v| v.as_str())
            .map(str::to_owned)
    }

    pub(crate) fn acp_error_message(err: &acp::Error) -> String {
        err.data
            .as_ref()
            .and_then(error_detail_from_data)
            .unwrap_or_else(|| err.message.clone())
    }
}
