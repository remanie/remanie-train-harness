//! Grok Build shell files copied verbatim (see vendor/UPSTREAM.txt) behind the module paths they expect, plus the runner's glue in `glue`.
#![allow(dead_code)] // The verbatim files keep upstream items the runner does not call.

pub mod glue;
pub mod sampling;
pub mod util;

pub mod session {
    pub mod two_pass;

    pub mod helpers {
        pub mod compaction_context;
        pub mod full_replace_compaction;
        pub mod prepared_compaction_history;
        pub mod session_compact;
        pub mod tool_input_parsing;

        pub mod memory_context {
            use xai_grok_tools::types::memory_backend::MemorySearchResult;

            // Memory is a sanctioned removal: the runner passes no memory backend, so upstream's formatter only ever sees no results.
            pub fn format_memory_reminder(results: &[MemorySearchResult]) -> Option<String> {
                debug_assert!(results.is_empty());
                None
            }
        }
    }
}
