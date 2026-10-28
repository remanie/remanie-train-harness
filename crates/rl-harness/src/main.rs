mod runner;
mod tools;

use std::io::{Read, Write};
use std::process::ExitCode;

fn main() -> ExitCode {
    // The terminal reads these process-wide limits before any worker threads start.
    unsafe {
        std::env::set_var("GROK_MAX_WAIT_BLOCK_MS", "300000");
        std::env::set_var("GROK_MAX_OUTPUT_FILE_BYTES", "16777216");
    }
    let result = (|| -> anyhow::Result<runner::RunResult> {
        let mut input = String::new();
        std::io::stdin()
            .take(1_048_577)
            .read_to_string(&mut input)?;
        anyhow::ensure!(input.len() <= 1_048_576, "input exceeds 1 MiB");
        let config: runner::Config = serde_json::from_str(&input)?;
        config.validate()?;
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()?;
        runtime.block_on(tokio::task::LocalSet::new().run_until(runner::run(config)))
    })();
    let result = result.unwrap_or_else(runner::RunResult::error);
    let exit = if result.status == "error" {
        ExitCode::FAILURE
    } else {
        ExitCode::SUCCESS
    };
    if serde_json::to_writer(std::io::stdout().lock(), &result).is_err()
        || writeln!(std::io::stdout()).is_err()
    {
        return ExitCode::FAILURE;
    }
    exit
}
