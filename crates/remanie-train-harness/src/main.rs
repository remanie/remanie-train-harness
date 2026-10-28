mod runner;
mod tools;

use std::io::{Read, Write};
use std::process::ExitCode;

use anyhow::{Context, Result, ensure};
use reqwest::header::HeaderValue;

fn bearer_header(key: &str) -> Result<HeaderValue> {
    ensure!(
        !key.is_empty() && key.bytes().all(|byte| byte.is_ascii_graphic()),
        "API key must be nonempty and contain no whitespace or control characters"
    );
    let mut header = HeaderValue::from_str(&format!("Bearer {key}"))?;
    header.set_sensitive(true);
    Ok(header)
}

fn main() -> ExitCode {
    // Set process-wide limits before starting threads.
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
        let authorization = if let Some(name) = &config.api_key_env {
            let key = std::env::var_os(name);
            // Remove credentials before the runtime or native tools can spawn threads or processes.
            unsafe { std::env::remove_var(name) };
            let key = key
                .context("configured API key environment variable is missing")?
                .into_string()
                .map_err(|_| anyhow::anyhow!("API key must be UTF-8"))?;
            Some(bearer_header(&key)?)
        } else {
            None
        };
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()?;
        runtime.block_on(tokio::task::LocalSet::new().run_until(runner::run(config, authorization)))
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bearer_headers_are_sensitive_and_invalid_keys_do_not_leak() {
        let header = bearer_header("test-private-key").unwrap();
        assert_eq!(header, "Bearer test-private-key");
        assert!(header.is_sensitive());
        assert!(!format!("{header:?}").contains("test-private-key"));
        for key in ["", "test-private-key\n", "test private key", "test-λ-key"] {
            let error = bearer_header(key).unwrap_err().to_string();
            assert!(!error.contains("test"));
        }
    }
}
