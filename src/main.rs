mod compaction;
mod model;
mod outcome;
mod runner;
mod tools;
mod trace;

use std::ffi::OsString;
use std::io::{Read, Write};
use std::process::ExitCode;

use outcome::{Cause, Outcome, RunResult, Tally};

fn main() -> ExitCode {
    let started = std::time::Instant::now();
    let mut stdin = Vec::new();
    let result = match std::io::stdin()
        .take(runner::MAX_INPUT_BYTES as u64 + 1)
        .read_to_end(&mut stdin)
    {
        Ok(_) => runner::run(&stdin, take_env),
        Err(error) => RunResult::new(
            Outcome::Infrastructure(Cause::Internal {
                detail: format!("read stdin: {error}"),
            }),
            Tally::default(),
            false,
            started.elapsed(),
        ),
    };
    let mut stdout = std::io::stdout().lock();
    if serde_json::to_writer(&mut stdout, &result).is_ok() {
        let _ = writeln!(stdout);
    }
    ExitCode::from(result.outcome.exit_code())
}

fn take_env(name: &str) -> Option<OsString> {
    let value = std::env::var_os(name);
    // SAFETY: `runner::run` calls this before it starts the runtime, while this is the only thread.
    unsafe { std::env::remove_var(name) };
    value
}
