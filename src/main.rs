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
    let read = std::io::stdin()
        .take(runner::MAX_INPUT_BYTES as u64 + 1)
        .read_to_end(&mut stdin);
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build();
    let result = match (read, runtime) {
        (Ok(_), Ok(runtime)) => {
            let result = runtime.block_on(runner::run(&stdin, take_env));
            // A tool thread stuck in a blocking call (a FIFO open) must not hold the exit.
            runtime.shutdown_background();
            result
        }
        (Err(error), _) => before_start(format!("read stdin: {error}"), started),
        (_, Err(error)) => before_start(format!("start runtime: {error}"), started),
    };
    let mut stdout = std::io::stdout().lock();
    if serde_json::to_writer(&mut stdout, &result).is_ok() {
        let _ = writeln!(stdout);
    }
    ExitCode::from(result.outcome.exit_code())
}

fn before_start(detail: String, started: std::time::Instant) -> RunResult {
    RunResult::new(
        Outcome::Infrastructure(Cause::Internal { detail }),
        Tally::default(),
        false,
        started.elapsed(),
    )
}

fn take_env(name: &str) -> Option<OsString> {
    let value = std::env::var_os(name);
    // SAFETY: `runner::run` calls this before its first await, while this is the only thread.
    unsafe { std::env::remove_var(name) };
    value
}
