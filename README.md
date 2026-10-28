# remanie-train-harness

An eight-tool coding agent built from Grok Build's Rust tools. It runs one attempt, executes the model's tool calls, optionally compacts the conversation, and saves the execution trace.

The tools are `run_terminal_cmd`, `get_terminal_command_output`, `kill_terminal_command`, `read_file`, `search_replace`, `list_dir`, `grep`, and `todo_write`.

Start with the [harness guide](crates/remanie-train-harness/README.md). The [Harbor adapter](crates/remanie-train-harness/harbor/) runs the same executable inside benchmark task environments. Evaluation does not require Miles.

```sh
cargo build -p remanie-train-harness
cargo test -p remanie-train-harness
```

The runtime uses native filesystem and shell access. Run it inside a disposable task environment. The launcher owns isolation, resource limits, grading, and cleanup.

This fork retains Grok Build's workspace and build dependencies. See the [upstream README](README.upstream.md) for build prerequisites and the original application. Upstream source starts at `2bdd1d6a6369de0e8c68132ea4539e9abd9e14a8`; upstream licenses and notices remain in place.
