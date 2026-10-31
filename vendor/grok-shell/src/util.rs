pub mod config {
    // Copied from crates/codegen/xai-grok-shell/src/util/config/resolve/compaction.rs:4-9, whose remaining items need the shell's config.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub enum CompactionToolChoice {
        #[default]
        Auto,
        None,
    }
}
