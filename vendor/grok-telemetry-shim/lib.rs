//! Stands in for `xai-grok-telemetry` under the vendored shell files. Telemetry is a sanctioned removal, and `agent_id` feeds only a header.

pub mod id {
    pub fn agent_id() -> String {
        std::env::var("GROK_AGENT_ID").unwrap_or_else(|_| "remanie-train-harness".to_owned())
    }
}

pub mod events {
    #[derive(Clone, Copy, Debug)]
    pub enum CompactionTrigger {
        Manual,
        Auto,
    }

    pub struct CompactionRetryDegraded {
        pub trigger: CompactionTrigger,
        pub reason: &'static str,
        pub from_stage: Option<&'static str>,
        pub to_stage: Option<&'static str>,
        pub summary_chars: Option<u64>,
        pub attempt: u32,
        pub context_window: u64,
        pub compaction_id: String,
    }
}

pub mod session_ctx {
    pub fn log_event<E>(_event: E) {}
}

pub mod memory_log {
    pub const TARGET: &str = "memory";
}
