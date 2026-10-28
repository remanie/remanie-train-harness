// Adapted from Grok Build 2bdd1d6a for this standalone runner; see NOTICE.
use std::borrow::Cow;
use std::fmt::Write as _;
use std::path::{Component, Path, PathBuf};
use std::time::Duration;

use anyhow::{Context, Result, bail, ensure};
use indexmap::IndexMap;
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, AsyncReadExt, BufReader};
use tokio::sync::Mutex;

use super::{list_dir, terminal::capture};

const READ_MAX_BYTES: usize = 100_003;
const MAX_DOCUMENT_BYTES: u64 = 50 * 1024 * 1024;
const MAX_LINES: usize = 1_000;

pub(super) struct FileTools {
    cwd: PathBuf,
    home: Option<PathBuf>,
    todos: Mutex<IndexMap<String, Value>>,
}

impl FileTools {
    pub(super) fn new(cwd: PathBuf, home: Option<PathBuf>) -> Self {
        Self {
            cwd,
            home,
            todos: Mutex::new(IndexMap::new()),
        }
    }

    pub(super) async fn call(&self, name: &str, args: &Value) -> Result<String> {
        match name {
            "read_file" => self.read(serde_json::from_value(args.clone())?).await,
            "search_replace" => self.edit(serde_json::from_value(args.clone())?).await,
            "list_dir" => {
                list_dir::list(
                    self.path(
                        args["target_directory"]
                            .as_str()
                            .context("missing target_directory")?,
                    )?,
                )
                .await
            }
            "grep" => self.grep(serde_json::from_value(args.clone())?).await,
            "todo_write" => self.todo(serde_json::from_value(args.clone())?).await,
            _ => bail!("tool is not permitted"),
        }
    }

    pub(super) async fn snapshot(&self) -> Value {
        json!({"todos": *self.todos.lock().await})
    }

    fn path(&self, input: &str) -> Result<PathBuf> {
        if input != "~" && !input.starts_with("~/") {
            return Ok(self.cwd.join(input));
        }
        let home = self
            .home
            .clone()
            .context("HOME is not set, so ~ cannot be expanded")?;
        Ok(match input.strip_prefix("~/") {
            Some(relative) => home.join(relative.trim_start_matches('/')),
            None => home,
        })
    }

    async fn read(&self, input: ReadInput) -> Result<String> {
        let path = resolve_file(self.path(&input.target_file)?).await;
        let file = tokio::fs::OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_NONBLOCK)
            .open(&path)
            .await
            .context("open file")?;
        let metadata = file.metadata().await?;
        ensure!(metadata.is_file(), "source is not a regular file");
        let mut reader = BufReader::new(file);
        let extension = path
            .extension()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();
        let sample = reader.fill_buf().await?;
        if extension == "pdf" || sample.starts_with(b"%PDF-") {
            ensure!(
                input.format.as_deref() == Some("text"),
                "non-text tool output is unsupported; use format=text for PDFs"
            );
            ensure!(
                metadata.len() <= MAX_DOCUMENT_BYTES,
                "PDF exceeds the 50 MB limit"
            );
            return tokio::time::timeout(
                Duration::from_secs(60),
                self.pdf(&path, input.pages.as_deref()),
            )
            .await
            .context("PDF processing timed out after 60 seconds")?;
        }
        ensure!(
            extension != "pptx" && !is_binary(&extension, sample),
            "non-text tool output is unsupported"
        );
        let start = input.offset.unwrap_or(1);
        ensure!(start > 0, "offset must be at least 1");
        let limit = input.limit.unwrap_or(MAX_LINES).min(MAX_LINES);
        ensure!(limit > 0, "limit must be at least 1");
        let mut raw = Vec::new();
        let mut line_number = 1usize;
        let end = start.saturating_add(limit);
        let mut saw_bytes = false;
        let mut last_newline = false;
        let mut trailing_empty = false;
        let mut more = false;
        tokio::time::timeout(Duration::from_secs(30), async {
            loop {
                let chunk = reader.fill_buf().await?;
                if chunk.is_empty() {
                    trailing_empty = saw_bytes && last_newline && line_number >= start && line_number < end;
                    break;
                }
                let mut consumed = 0;
                for segment in chunk.split_inclusive(|byte| *byte == b'\n') {
                    saw_bytes = true;
                    last_newline = segment.ends_with(b"\n");
                    if line_number >= start {
                        ensure!(raw.len().saturating_add(segment.len()) <= READ_MAX_BYTES + MAX_LINES + 1, "file content exceeds 25000 tokens; use offset and limit to read a shorter range");
                        raw.extend_from_slice(segment);
                    }
                    consumed += segment.len();
                    if last_newline {
                        line_number += 1;
                        if line_number >= end {
                            break;
                        }
                    }
                }
                reader.consume(consumed);
                if line_number >= end {
                    more = !reader.fill_buf().await?.is_empty();
                    break;
                }
            }
            Ok::<_, anyhow::Error>(())
        }).await.context("file read timed out after 30 seconds")??;
        let text = String::from_utf8_lossy(&raw);
        let mut lines: Vec<&str> = text
            .split_inclusive('\n')
            .map(|line| {
                line.strip_suffix('\n')
                    .map(|s| s.strip_suffix('\r').unwrap_or(s))
                    .unwrap_or(line)
            })
            .collect();
        if trailing_empty {
            lines.push("");
        }
        let mut output = String::new();
        for (i, line) in lines.into_iter().enumerate() {
            if i > 0 {
                output.push('\n');
            }
            let number = start.saturating_add(i);
            if i == 0 || number.is_multiple_of(10) {
                write!(output, "{number}→")?;
            }
            output.push_str(line);
        }
        ensure!(
            output.len() <= READ_MAX_BYTES,
            "file content exceeds 25000 tokens; use offset and limit to read a shorter range"
        );
        if more && input.limit.is_none_or(|limit| limit >= MAX_LINES) {
            write!(
                output,
                "\n[lines {start}-{} shown; file continues, use offset={end}]",
                end - 1
            )?;
        }
        Ok(output)
    }

    async fn edit(&self, input: EditInput) -> Result<String> {
        ensure!(
            input.old_string != input.new_string,
            "Old string and new string are the same"
        );
        ensure!(
            input.new_string.len() as u64 <= MAX_DOCUMENT_BYTES,
            "replacement exceeds the 50 MB edit limit"
        );
        for component in Path::new(&input.file_path).components() {
            if let Component::Normal(name) = component {
                ensure!(
                    name.as_encoded_bytes().len() <= 255,
                    "file name exceeds the 255-byte limit"
                );
            }
        }
        let path = resolve_file(self.path(&input.file_path)?).await;
        let mut existing = None;
        let old = match tokio::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .custom_flags(libc::O_NONBLOCK)
            .open(&path)
            .await
        {
            Ok(file) => {
                let metadata = file.metadata().await?;
                ensure!(metadata.is_file(), "File path is not a regular file");
                let mut bytes = Vec::new();
                let mut file = file;
                (&mut file)
                    .take(MAX_DOCUMENT_BYTES + 1)
                    .read_to_end(&mut bytes)
                    .await?;
                ensure!(
                    bytes.len() as u64 <= MAX_DOCUMENT_BYTES,
                    "file exceeds the 50 MB edit limit"
                );
                existing = Some((file.into_std().await, metadata));
                Some(String::from_utf8(bytes).context("cannot edit a non-UTF-8 file")?)
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
            Err(error) => return Err(error).context("open file for editing"),
        };
        let (text, created, count) = if input.old_string.is_empty() {
            ensure!(
                old.as_deref().is_none_or(str::is_empty),
                "old_string is empty, which is only allowed when creating a new file or when the file is empty"
            );
            (input.new_string, old.is_none(), 1)
        } else {
            let old = old.context("file not found")?;
            let crlf = old.contains("\r\n");
            let normalize = crlf && !old.contains(&input.old_string);
            let matching = if normalize {
                Cow::Owned(old.replace("\r\n", "\n"))
            } else {
                Cow::Borrowed(old.as_str())
            };
            let needle = if normalize {
                Cow::Owned(input.old_string.replace("\r\n", "\n"))
            } else {
                Cow::Borrowed(input.old_string.as_str())
            };
            let count = matching.match_indices(needle.as_ref()).count();
            ensure!(
                count > 0,
                "The string to replace was not found in the file, use the read_file tool to see the correct string."
            );
            let overlapping = matching.find(needle.as_ref()).is_some_and(|start| {
                let next = start + needle.chars().next().map_or(1, char::len_utf8);
                matching[next..].contains(needle.as_ref())
            });
            ensure!(
                (count == 1 && !overlapping) || input.replace_all,
                "The string to replace was found multiple times in the file. Use replace_all to replace all occurrences, or include more context to only edit one occurrence."
            );
            let replacement = if normalize {
                let size = input.new_string.len() + input.new_string.matches('\n').count()
                    - input.new_string.matches("\r\n").count();
                ensure!(
                    size as u64 <= MAX_DOCUMENT_BYTES,
                    "replacement exceeds the 50 MB edit limit"
                );
                input.new_string.replace("\r\n", "\n").replace('\n', "\r\n")
            } else {
                input.new_string
            };
            let expanded = count.checked_mul(replacement.len()).and_then(|inserted| {
                matching
                    .len()
                    .checked_sub(count.checked_mul(needle.len())?)?
                    .checked_add(inserted)
            });
            ensure!(
                expanded.is_some_and(|size| size as u64 <= MAX_DOCUMENT_BYTES),
                "replacement exceeds the 50 MB edit limit"
            );
            let mut crlf_offsets = old
                .match_indices("\r\n")
                .filter(|_| normalize)
                .map(|(offset, _)| offset)
                .peekable();
            let mut skipped_carriage_returns = 0;
            let mut original_index = |normalized_index| {
                while crlf_offsets
                    .peek()
                    .is_some_and(|offset| *offset < normalized_index + skipped_carriage_returns)
                {
                    crlf_offsets.next();
                    skipped_carriage_returns += 1;
                }
                normalized_index + skipped_carriage_returns
            };
            let mut text = String::with_capacity(expanded.unwrap());
            let mut copied = 0;
            for (normalized_start, _) in matching.match_indices(needle.as_ref()) {
                let start = original_index(normalized_start);
                let end = original_index(normalized_start + needle.len());
                for piece in [&old[copied..start], &replacement] {
                    ensure!(
                        text.len().saturating_add(piece.len()) as u64 <= MAX_DOCUMENT_BYTES,
                        "replacement exceeds the 50 MB edit limit"
                    );
                    text.push_str(piece);
                }
                copied = end;
            }
            ensure!(
                text.len().saturating_add(old.len() - copied) as u64 <= MAX_DOCUMENT_BYTES,
                "replacement exceeds the 50 MB edit limit"
            );
            text.push_str(&old[copied..]);
            (text, false, count)
        };
        write_file(&path, existing.as_ref(), &text)?;
        Ok(format!(
            "The file {} has been {}.{}",
            input.file_path,
            if created { "created" } else { "updated" },
            if count > 1 {
                " All occurrences were replaced."
            } else {
                ""
            }
        ))
    }

    async fn grep(&self, input: GrepInput) -> Result<String> {
        let root = self.path(input.path.as_deref().unwrap_or(""))?;
        let mut args: Vec<String> = [
            "--heading",
            "--with-filename",
            "--line-number",
            "--color=never",
            "--max-columns=1000",
            "--max-columns-preview",
            "--max-filesize=5M",
        ]
        .into_iter()
        .map(str::to_owned)
        .collect();
        if input.case_insensitive {
            args.push("--ignore-case".into());
        }
        if input.multiline {
            args.extend(["-U".into(), "--multiline-dotall".into()]);
        }
        for (flag, value) in [("--glob", input.glob), ("--type", input.file_type)] {
            if let Some(value) = value.filter(|s| !s.is_empty()) {
                args.extend([flag.into(), value]);
            }
        }
        for (flag, value) in [
            ("-C", input.context),
            ("-B", input.before),
            ("-A", input.after),
        ] {
            if let Some(value) = value {
                args.extend([flag.into(), value.to_string()]);
            }
        }
        args.extend([
            "-e".into(),
            input.pattern,
            "--".into(),
            root.to_string_lossy().into_owned(),
        ]);
        let program = std::env::var_os("RG_BIN_PATH")
            .map(PathBuf::from)
            .unwrap_or_else(|| "rg".into());
        let (code, output) = capture(&program, &args, &self.cwd, Duration::from_secs(20), 40_000)
            .await
            .context("grep failed; install ripgrep or set RG_BIN_PATH")?;
        ensure!(matches!(code, 0 | 1), "grep failed (exit {code}): {output}");
        if code == 1 || output.is_empty() {
            return Ok("No matches found".into());
        }
        let limit = input.head_limit.unwrap_or(200).min(2_000);
        let mut lines = output.lines();
        let mut text = lines.by_ref().take(limit).collect::<Vec<_>>().join("\n");
        if lines.next().is_some() {
            text.push_str("\n... [output truncated; narrow the path or pattern] ...");
        }
        Ok(text)
    }

    async fn pdf(&self, path: &Path, spec: Option<&str>) -> Result<String> {
        let path = path.to_str().context("PDF path is not UTF-8")?.to_owned();
        let (code, info) = capture(
            Path::new("pdfinfo"),
            std::slice::from_ref(&path),
            &self.cwd,
            Duration::from_secs(60),
            40_000,
        )
        .await
        .context("PDF text requires the optional Poppler utilities pdfinfo and pdftotext")?;
        ensure!(code == 0, "pdfinfo failed (exit {code}): {info}");
        ensure!(
            !info.ends_with("\n[output truncated]"),
            "pdfinfo output exceeded the limit"
        );
        let count: usize = info
            .lines()
            .find_map(|line| line.strip_prefix("Pages:"))
            .context("pdfinfo did not report a page count")?
            .trim()
            .parse()?;
        let pages = match spec {
            Some(spec) => page_range(spec, count)?,
            None => {
                ensure!(
                    count > 0 && count <= 10,
                    "PDF has {count} pages; use pages to select up to 20 pages per call"
                );
                (1..=count).collect()
            }
        };
        let mut output = String::new();
        for page in pages {
            let args = [
                "-f".into(),
                page.to_string(),
                "-l".into(),
                page.to_string(),
                "-enc".into(),
                "UTF-8".into(),
                "-nopgbrk".into(),
                path.clone(),
                "-".into(),
            ];
            let (code, text) = capture(
                Path::new("pdftotext"),
                &args,
                &self.cwd,
                Duration::from_secs(60),
                READ_MAX_BYTES,
            )
            .await
            .context("PDF text requires the optional Poppler utilities pdfinfo and pdftotext")?;
            ensure!(code == 0, "pdftotext failed (exit {code}): {text}");
            ensure!(
                !text.ends_with("\n[output truncated]"),
                "PDF text exceeds 25000 tokens; select fewer pages"
            );
            if !output.is_empty() {
                output.push('\n');
            }
            writeln!(output, "--- Page {page} ---")?;
            output.push_str(&text);
            ensure!(
                output.len() <= READ_MAX_BYTES,
                "PDF text exceeds 25000 tokens; select fewer pages"
            );
        }
        let output = output
            .lines()
            .enumerate()
            .map(|(i, line)| format!("{}→{line}", i + 1))
            .collect::<Vec<_>>()
            .join("\n");
        ensure!(
            output.len() <= READ_MAX_BYTES,
            "PDF text exceeds 25000 tokens; select fewer pages"
        );
        Ok(output)
    }

    async fn todo(&self, input: TodoInput) -> Result<String> {
        let mut seen = std::collections::HashSet::new();
        for update in &input.todos {
            ensure!(
                seen.insert(&update.id),
                "Duplicate todo ID in request: {}",
                update.id
            );
        }
        let mut todos = self.todos.lock().await;
        if !input.merge {
            todos.clear();
        }
        for update in input.todos {
            let item = todos.entry(update.id.clone()).or_insert_with(
                || json!({"content":update.id,"priority":"medium","status":"pending"}),
            );
            if let Some(content) = update.content.filter(|s| !s.is_empty()) {
                item["content"] = content.into();
            }
            if let Some(status) = update.status {
                item["status"] = status.into();
            }
        }
        if todos.is_empty() {
            return Ok("No tasks currently tracked.".into());
        }
        Ok(todos
            .iter()
            .map(|(id, item)| {
                format!(
                    "- [{}] {id}: {}\n",
                    item["status"].as_str().unwrap_or("pending"),
                    item["content"].as_str().unwrap_or(id)
                )
            })
            .collect())
    }
}

// In place rather than rename, so the inode, owner, mode and hardlinks survive; a crash
// mid-write can leave the file partly written.
fn write_file(
    path: &Path,
    existing: Option<&(std::fs::File, std::fs::Metadata)>,
    text: &str,
) -> Result<()> {
    use std::io::Write as _;
    use std::os::unix::fs::{FileExt, MetadataExt};

    let Some((file, expected)) = existing else {
        std::fs::create_dir_all(path.parent().context("file path has no parent")?)?;
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)?;
        return Ok(file.write_all(text.as_bytes())?);
    };
    let current =
        std::fs::symlink_metadata(path).context("file changed during edit; read_file again")?;
    let fingerprint = |metadata: &std::fs::Metadata| {
        (
            metadata.dev(),
            metadata.ino(),
            metadata.len(),
            metadata.mtime(),
            metadata.mtime_nsec(),
            metadata.ctime(),
            metadata.ctime_nsec(),
        )
    };
    ensure!(
        current.is_file() && fingerprint(&current) == fingerprint(expected),
        "file changed during edit; read_file again"
    );
    file.write_all_at(text.as_bytes(), 0)?;
    file.set_len(text.len() as u64)?;
    Ok(())
}

#[derive(Deserialize)]
struct ReadInput {
    target_file: String,
    offset: Option<usize>,
    limit: Option<usize>,
    pages: Option<String>,
    format: Option<String>,
}

#[derive(Deserialize)]
struct EditInput {
    file_path: String,
    old_string: String,
    new_string: String,
    #[serde(default)]
    replace_all: bool,
}

#[derive(Deserialize)]
struct GrepInput {
    pattern: String,
    path: Option<String>,
    glob: Option<String>,
    #[serde(rename = "type")]
    file_type: Option<String>,
    #[serde(rename = "-B")]
    before: Option<u64>,
    #[serde(rename = "-A")]
    after: Option<u64>,
    #[serde(rename = "-C")]
    context: Option<u64>,
    #[serde(default, rename = "-i")]
    case_insensitive: bool,
    head_limit: Option<usize>,
    #[serde(default)]
    multiline: bool,
}

#[derive(Deserialize)]
struct TodoInput {
    #[serde(default = "default_merge")]
    merge: bool,
    todos: Vec<TodoUpdate>,
}

fn default_merge() -> bool {
    true
}

#[derive(Deserialize)]
struct TodoUpdate {
    id: String,
    content: Option<String>,
    status: Option<String>,
}

async fn resolve_file(path: PathBuf) -> PathBuf {
    match tokio::fs::canonicalize(&path).await {
        Ok(path) => return path,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(_) => return path,
    }
    let Some(target) = path.file_name().and_then(|s| s.to_str()) else {
        return path;
    };
    let Some(parent) = path.parent() else {
        return path;
    };
    let normalize = |s: &str| s.replace(['\u{202f}', '\u{a0}'], " ");
    let mut entries = match tokio::fs::read_dir(parent).await {
        Ok(entries) => entries,
        Err(_) => return path,
    };
    let mut found = None;
    while let Ok(Some(entry)) = entries.next_entry().await {
        let name = entry.file_name();
        let Some(name) = name.to_str() else {
            continue;
        };
        if normalize(name) == normalize(target) {
            if found.is_some() {
                return path;
            }
            found = Some(entry.path());
        }
    }
    found.unwrap_or(path)
}

fn is_binary(extension: &str, bytes: &[u8]) -> bool {
    const EXTENSIONS: &[&str] = &[
        "7z", "a", "avi", "avif", "bin", "bmp", "class", "dat", "dll", "doc", "docx", "dylib",
        "exe", "gif", "gz", "ico", "jar", "jpeg", "jpg", "lib", "mov", "mp3", "mp4", "o", "obj",
        "odp", "ods", "odt", "png", "ppt", "pyc", "pyd", "pyo", "qoi", "rar", "so", "tar", "tif",
        "tiff", "war", "wasm", "webp", "xls", "xlsx", "zip",
    ];
    if EXTENSIONS.binary_search(&extension).is_ok() {
        return true;
    }
    let sample = &bytes[..bytes.len().min(8192)];
    sample.contains(&0)
        || (!sample.is_empty()
            && sample
                .iter()
                .filter(|&&byte| byte < 9 || (14..=31).contains(&byte))
                .count() as f64
                / sample.len() as f64
                > 0.3)
}

fn page_range(spec: &str, page_count: usize) -> Result<Vec<usize>> {
    let mut pages = std::collections::BTreeSet::new();
    for part in spec.split(',').map(str::trim).filter(|s| !s.is_empty()) {
        let (start, end) = match part.split_once('-') {
            Some((start, end)) => (
                start.trim().parse::<usize>()?,
                if end.trim().is_empty() {
                    page_count
                } else {
                    end.trim().parse::<usize>()?
                },
            ),
            None => {
                let page = part.parse::<usize>()?;
                (page, page)
            }
        };
        ensure!(
            start > 0 && start <= page_count,
            "page {start} out of range (document has {page_count} pages)"
        );
        ensure!(start <= end, "invalid page range: {start}-{end}");
        let end = end.min(page_count);
        ensure!(end - start < 20, "maximum is 20 pages per call");
        pages.extend(start..=end);
        ensure!(pages.len() <= 20, "maximum is 20 pages per call");
    }
    ensure!(!pages.is_empty(), "no pages specified");
    Ok(pages.into_iter().collect())
}

#[cfg(test)]
#[path = "files_tests.rs"]
mod tests;
