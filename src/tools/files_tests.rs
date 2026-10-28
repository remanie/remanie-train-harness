use super::*;
use crate::tools::InternalError;

fn file_tools(dir: &Path) -> FileTools {
    let ripgrep = std::env::var_os("RG_BIN_PATH").map_or_else(|| "rg".into(), PathBuf::from);
    FileTools::new(dir.to_path_buf(), Some(dir.join("home")), ripgrep)
}

#[tokio::test]
async fn edits_preserve_literal_whitespace_and_quote_paths() {
    let dir = tempfile::tempdir().unwrap();
    let tools = file_tools(dir.path());
    for name in [" a.txt", "a.txt ", "'a.txt'", "a.txt"] {
        tokio::fs::write(dir.path().join(name), "before")
            .await
            .unwrap();
    }
    for name in [" a.txt", "a.txt ", "'a.txt'"] {
        let path = if name.ends_with(' ') {
            dir.path().join(name)
        } else {
            PathBuf::from(name)
        };
        tools
            .call(
                "search_replace",
                &json!({"file_path":path,"old_string":"before","new_string":"after"}),
            )
            .await
            .unwrap();
        assert_eq!(
            tokio::fs::read_to_string(dir.path().join(name))
                .await
                .unwrap(),
            "after"
        );
        assert_eq!(
            tokio::fs::read_to_string(dir.path().join("a.txt"))
                .await
                .unwrap(),
            "before"
        );
    }
}

#[tokio::test]
async fn tilde_paths_use_home() {
    let dir = tempfile::tempdir().unwrap();
    let home = dir.path().join("private-home");
    tokio::fs::create_dir(&home).await.unwrap();
    tokio::fs::write(home.join("file.txt"), "before")
        .await
        .unwrap();
    let tools = FileTools::new(dir.path().to_path_buf(), Some(home.clone()), "rg".into());
    assert_eq!(
        tools
            .call("read_file", &json!({"target_file":"~/file.txt"}))
            .await
            .unwrap(),
        "1→before"
    );
    tools
        .call(
            "search_replace",
            &json!({"file_path":"~//file.txt","old_string":"before","new_string":"after"}),
        )
        .await
        .unwrap();
    assert_eq!(
        tokio::fs::read_to_string(home.join("file.txt"))
            .await
            .unwrap(),
        "after"
    );
    assert!(
        tools
            .call("list_dir", &json!({"target_directory":"~"}))
            .await
            .unwrap()
            .contains("file.txt")
    );
    let homeless = FileTools::new(dir.path().to_path_buf(), None, "rg".into());
    let error = homeless
        .call("read_file", &json!({"target_file":"~/file.txt"}))
        .await
        .unwrap_err();
    assert!(error.to_string().contains("HOME is not set"), "{error}");
}

#[tokio::test]
async fn read_windows_keep_unicode_crlf_anchors_and_trailing_lines() {
    let dir = tempfile::tempdir().unwrap();
    let tools = file_tools(dir.path());
    let path = dir.path().join("AGENTS.md");
    tokio::fs::write(&path, "α\r\nβ\r\nγ\r\n").await.unwrap();
    assert_eq!(
        tools
            .call(
                "read_file",
                &json!({"target_file":"AGENTS.md","offset":2,"limit":1})
            )
            .await
            .unwrap(),
        "2→β"
    );
    assert_eq!(
        tools
            .call("read_file", &json!({"target_file":path}))
            .await
            .unwrap(),
        "1→α\nβ\nγ\n"
    );
    assert_eq!(
        tools
            .call(
                "read_file",
                &json!({"target_file":"AGENTS.md","offset":4,"limit":1})
            )
            .await
            .unwrap(),
        "4→"
    );
    assert_eq!(
        tools
            .call("read_file", &json!({"target_file":"AGENTS.md","offset":5}))
            .await
            .unwrap(),
        ""
    );
    tokio::fs::write(&path, "").await.unwrap();
    assert_eq!(
        tools
            .call("read_file", &json!({"target_file":"AGENTS.md"}))
            .await
            .unwrap(),
        ""
    );
    tokio::fs::write(
        &path,
        (1..=15).map(|n| format!("line{n}\n")).collect::<String>(),
    )
    .await
    .unwrap();
    let text = tools
        .call(
            "read_file",
            &json!({"target_file":"AGENTS.md","offset":8,"limit":4}),
        )
        .await
        .unwrap();
    assert_eq!(text, "8→line8\nline9\n10→line10\nline11");
}

#[tokio::test]
async fn reads_bound_the_returned_window_and_reject_non_text() {
    let dir = tempfile::tempdir().unwrap();
    let tools = file_tools(dir.path());
    tokio::fs::write(
        dir.path().join("large.txt"),
        format!("{}\nsmall\n", "x".repeat(READ_MAX_BYTES + 1)),
    )
    .await
    .unwrap();
    assert!(
        tools
            .call("read_file", &json!({"target_file":"large.txt"}))
            .await
            .unwrap_err()
            .to_string()
            .contains("100000 bytes")
    );
    assert_eq!(
        tools
            .call(
                "read_file",
                &json!({"target_file":"large.txt","offset":2,"limit":1})
            )
            .await
            .unwrap(),
        "2→small"
    );
    tokio::fs::write(dir.path().join("many.txt"), "line\n".repeat(1_100))
        .await
        .unwrap();
    let text = tools
        .call("read_file", &json!({"target_file":"many.txt","limit":2000}))
        .await
        .unwrap();
    let (window, marker) = text.rsplit_once('\n').unwrap();
    assert_eq!(window.lines().count(), MAX_LINES);
    assert_eq!(
        marker,
        "[lines 1-1000 shown. The file continues at offset=1001.]"
    );
    let crlf = format!(
        "{}\r\n{}",
        "x".repeat(108),
        format!("{}\r\n", "x".repeat(98)).repeat(999)
    );
    assert!(crlf.len() > READ_MAX_BYTES);
    tokio::fs::write(dir.path().join("crlf.txt"), crlf)
        .await
        .unwrap();
    let text = tools
        .call("read_file", &json!({"target_file":"crlf.txt"}))
        .await
        .unwrap();
    assert!(text.len() <= READ_MAX_BYTES && text.lines().count() == MAX_LINES);
    for (name, bytes) in [
        ("binary.txt", b"data\0more".as_slice()),
        ("pixel.png", b"not an actual image".as_slice()),
        ("slides.pptx", b"PK".as_slice()),
    ] {
        tokio::fs::write(dir.path().join(name), bytes)
            .await
            .unwrap();
        assert!(
            tools
                .call("read_file", &json!({"target_file":name}))
                .await
                .unwrap_err()
                .to_string()
                .contains("non-text")
        );
    }
    assert!(
        tools
            .call("read_file", &json!({"target_file":"."}))
            .await
            .is_err()
    );
}

#[tokio::test]
async fn edits_enforce_creation_unique_matches_and_preserve_exact_unicode_and_crlf() {
    let dir = tempfile::tempdir().unwrap();
    let tools = file_tools(dir.path());
    let path = dir.path().join("nested/file.txt");
    tools.call("search_replace", &json!({"file_path":"nested/file.txt","old_string":"","new_string":"café\r\n猫\r\ncafé\r\n"})).await.unwrap();
    assert!(
        tools
            .call(
                "search_replace",
                &json!({"file_path":"nested/file.txt","old_string":"","new_string":"erase"})
            )
            .await
            .is_err()
    );
    assert!(
        tools
            .call(
                "search_replace",
                &json!({"file_path":"nested/file.txt","old_string":"café","new_string":"thé"})
            )
            .await
            .is_err()
    );
    assert!(
        tools
            .call(
                "search_replace",
                &json!({"file_path":"nested/file.txt","old_string":"猫","new_string":"猫"})
            )
            .await
            .is_err()
    );
    tools
        .call(
            "search_replace",
            &json!({"file_path":path,"old_string":"café","new_string":"thé","replace_all":true}),
        )
        .await
        .unwrap();
    tools
        .call(
            "search_replace",
            &json!({"file_path":path,"old_string":"猫\nthé","new_string":"犬\n抹茶"}),
        )
        .await
        .unwrap();
    assert_eq!(
        tokio::fs::read_to_string(&path).await.unwrap(),
        "thé\r\n犬\r\n抹茶\r\n"
    );
    assert!(
        tools
            .call(
                "search_replace",
                &json!({"file_path":path,"old_string":"the","new_string":"bad"})
            )
            .await
            .is_err()
    );
    tokio::fs::write(&path, [0xff, b'a']).await.unwrap();
    assert!(
        tools
            .call(
                "search_replace",
                &json!({"file_path":path,"old_string":"a","new_string":"b"})
            )
            .await
            .is_err()
    );
    assert_eq!(tokio::fs::read(&path).await.unwrap(), vec![0xff, b'a']);
    let too_long = "x".repeat(256);
    assert!(
        tools
            .call(
                "search_replace",
                &json!({"file_path":too_long,"old_string":"","new_string":"x"})
            )
            .await
            .is_err()
    );
}

#[tokio::test]
async fn edits_accept_crlf_or_read_text_and_preserve_untouched_line_endings() {
    let dir = tempfile::tempdir().unwrap();
    let tools = file_tools(dir.path());
    let path = dir.path().join("mixed.txt");
    tokio::fs::write(&path, "α\r\none\ntwo\r\n猫\r\nend\n")
        .await
        .unwrap();
    tools.call("search_replace", &json!({"file_path":"mixed.txt","old_string":"one\ntwo\n猫","new_string":"ONE\nTWO\n犬"})).await.unwrap();
    assert_eq!(
        tokio::fs::read_to_string(&path).await.unwrap(),
        "α\r\nONE\r\nTWO\r\n犬\r\nend\n"
    );
    tools
        .call(
            "search_replace",
            &json!({"file_path":"mixed.txt","old_string":"α\r\nONE","new_string":"β\r\nUNO"}),
        )
        .await
        .unwrap();
    assert_eq!(
        tokio::fs::read_to_string(&path).await.unwrap(),
        "β\r\nUNO\r\nTWO\r\n犬\r\nend\n"
    );
    tools
        .call(
            "search_replace",
            &json!({"file_path":"mixed.txt","old_string":"\n","new_string":"|","replace_all":true}),
        )
        .await
        .unwrap();
    assert_eq!(
        tokio::fs::read_to_string(&path).await.unwrap(),
        "β\r|UNO\r|TWO\r|犬\r|end|"
    );
}

#[tokio::test]
async fn edits_prefer_exact_newlines_before_normalizing_read_text() {
    let dir = tempfile::tempdir().unwrap();
    let tools = file_tools(dir.path());
    let path = dir.path().join("mixed.txt");
    tokio::fs::write(&path, "one\r\ntwo\r\none\ntwo\n")
        .await
        .unwrap();
    tools
        .call(
            "search_replace",
            &json!({"file_path":"mixed.txt","old_string":"one\r\ntwo","new_string":"ONE\r\nTWO"}),
        )
        .await
        .unwrap();
    assert_eq!(
        tokio::fs::read_to_string(&path).await.unwrap(),
        "ONE\r\nTWO\r\none\ntwo\n"
    );
    tools
        .call(
            "search_replace",
            &json!({"file_path":"mixed.txt","old_string":"one\ntwo","new_string":"ONE TWO"}),
        )
        .await
        .unwrap();
    assert_eq!(
        tokio::fs::read_to_string(&path).await.unwrap(),
        "ONE\r\nTWO\r\nONE TWO\n"
    );
    tools
        .call(
            "search_replace",
            &json!({"file_path":"mixed.txt","old_string":"\r","new_string":"","replace_all":true}),
        )
        .await
        .unwrap();
    assert_eq!(
        tokio::fs::read_to_string(&path).await.unwrap(),
        "ONE\nTWO\nONE TWO\n"
    );
}

#[tokio::test]
async fn literal_edits_preserve_replacement_bytes_at_crlf_boundaries() {
    let dir = tempfile::tempdir().unwrap();
    let tools = file_tools(dir.path());
    let path = dir.path().join("file.txt");
    for (needle, expected) in [("\n", "a\r\nX\nb"), ("\r\n", "a\nX\nb")] {
        tokio::fs::write(&path, "a\r\nb").await.unwrap();
        tools
            .call(
                "search_replace",
                &json!({"file_path":"file.txt","old_string":needle,"new_string":"\nX\n"}),
            )
            .await
            .unwrap();
        assert_eq!(tokio::fs::read_to_string(&path).await.unwrap(), expected);
    }
}

#[tokio::test]
async fn oversized_replacements_fail_before_expanding_or_changing_the_file() {
    let dir = tempfile::tempdir().unwrap();
    let tools = file_tools(dir.path());
    let path = dir.path().join("expanded.txt");
    let original = "x".repeat(1_000);
    tokio::fs::write(&path, &original).await.unwrap();
    let error = tools.call("search_replace", &json!({"file_path":"expanded.txt","old_string":"x","new_string":"y".repeat(65_536),"replace_all":true})).await.unwrap_err();
    assert!(error.to_string().contains("50 MB edit limit"));
    assert_eq!(tokio::fs::read_to_string(&path).await.unwrap(), original);
}

#[test]
fn in_place_writes_refuse_concurrent_changes_and_never_clobber() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("file");
    std::fs::write(&path, "original").unwrap();
    let open = || {
        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(&path)
            .unwrap();
        let metadata = file.metadata().unwrap();
        (file, metadata)
    };
    let stale = open();
    std::fs::write(&path, "concurrent change").unwrap();
    assert!(
        write_file(&path, Some(&stale), "discarded")
            .unwrap_err()
            .to_string()
            .contains("file changed")
    );
    assert_eq!(std::fs::read_to_string(&path).unwrap(), "concurrent change");
    write_file(&path, Some(&open()), "short").unwrap();
    assert_eq!(std::fs::read_to_string(&path).unwrap(), "short");
    assert!(write_file(&path, None, "must not overwrite").is_err());
    assert_eq!(std::fs::read_to_string(&path).unwrap(), "short");
    assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 1);
}

#[tokio::test]
async fn edits_keep_inode_owner_and_mode_and_new_files_follow_the_umask() {
    use std::os::unix::fs::{MetadataExt, PermissionsExt};
    let dir = tempfile::tempdir().unwrap();
    let tools = file_tools(dir.path());
    let path = dir.path().join("linked.txt");
    std::fs::write(&path, "v1").unwrap();
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o640)).unwrap();
    std::fs::hard_link(&path, dir.path().join("alias.txt")).unwrap();
    if unsafe { libc::geteuid() } == 0 {
        std::os::unix::fs::chown(&path, Some(65534), Some(65534)).unwrap();
    }
    let before = std::fs::metadata(&path).unwrap();
    tools
        .call(
            "search_replace",
            &json!({"file_path":"linked.txt","old_string":"v1","new_string":"v2"}),
        )
        .await
        .unwrap();
    let after = std::fs::metadata(&path).unwrap();
    assert_eq!(
        std::fs::read_to_string(dir.path().join("alias.txt")).unwrap(),
        "v2"
    );
    assert_eq!(
        (after.ino(), after.uid(), after.gid(), after.mode()),
        (before.ino(), before.uid(), before.gid(), before.mode())
    );
    std::fs::write(dir.path().join("shell.html"), "x").unwrap();
    tools
        .call(
            "search_replace",
            &json!({"file_path":"tool.html","old_string":"","new_string":"x"}),
        )
        .await
        .unwrap();
    assert_eq!(
        std::fs::metadata(dir.path().join("tool.html"))
            .unwrap()
            .mode(),
        std::fs::metadata(dir.path().join("shell.html"))
            .unwrap()
            .mode()
    );
}

#[tokio::test]
async fn cancelling_edits_leaves_an_original_or_complete_file() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("file");
    let original = "before\n".repeat(100_000);
    let replacement = "after\n".repeat(100_000);
    let tools = std::sync::Arc::new(file_tools(dir.path()));
    for delay in [0, 1, 5] {
        tokio::fs::write(&path, &original).await.unwrap();
        let tools = tools.clone();
        let edit = tokio::spawn(async move {
            tools.call("search_replace", &json!({"file_path":"file","old_string":"before","new_string":"after","replace_all":true})).await
        });
        tokio::time::sleep(Duration::from_millis(delay)).await;
        edit.abort();
        let _ = edit.await;
        let text = tokio::fs::read_to_string(&path).await.unwrap();
        assert!(text == original || text == replacement);
        assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 1);
    }
}

#[tokio::test]
async fn unicode_filename_fallback_is_unique_and_symlink_edits_keep_the_link() {
    use std::os::unix::fs::symlink;
    let dir = tempfile::tempdir().unwrap();
    let tools = file_tools(dir.path());
    let actual = dir.path().join("screen\u{202f}shot.txt");
    tokio::fs::write(&actual, "before").await.unwrap();
    assert!(
        tools
            .call("read_file", &json!({"target_file":"screen shot.txt"}))
            .await
            .unwrap()
            .contains("before")
    );
    tools
        .call(
            "search_replace",
            &json!({"file_path":"screen shot.txt","old_string":"before","new_string":"after"}),
        )
        .await
        .unwrap();
    symlink(&actual, dir.path().join("link.txt")).unwrap();
    tools
        .call(
            "search_replace",
            &json!({"file_path":"link.txt","old_string":"after","new_string":"linked"}),
        )
        .await
        .unwrap();
    assert!(
        tokio::fs::symlink_metadata(dir.path().join("link.txt"))
            .await
            .unwrap()
            .file_type()
            .is_symlink()
    );
    assert_eq!(tokio::fs::read_to_string(&actual).await.unwrap(), "linked");
    tokio::fs::write(dir.path().join("screen\u{a0}shot.txt"), "other")
        .await
        .unwrap();
    assert!(
        tools
            .call("read_file", &json!({"target_file":"screen shot.txt"}))
            .await
            .is_err()
    );
}

#[tokio::test]
async fn special_files_fail_promptly_instead_of_waiting_for_a_writer() {
    use std::ffi::CString;
    use std::os::unix::ffi::OsStrExt;
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("pipe");
    let c_path = CString::new(path.as_os_str().as_bytes()).unwrap();
    assert_eq!(unsafe { libc::mkfifo(c_path.as_ptr(), 0o600) }, 0);
    let tools = file_tools(dir.path());
    for (name, args) in [
        ("read_file", json!({"target_file":"pipe"})),
        (
            "search_replace",
            json!({"file_path":"pipe","old_string":"","new_string":"x"}),
        ),
    ] {
        assert!(
            tokio::time::timeout(Duration::from_secs(1), tools.call(name, &args))
                .await
                .unwrap()
                .is_err()
        );
    }
}

#[tokio::test]
async fn todos_preserve_insertion_order_partial_updates_and_duplicate_atomicity() {
    let dir = tempfile::tempdir().unwrap();
    let tools = file_tools(dir.path());
    tools.call("todo_write", &json!({"todos":[{"id":"b","content":"second"},{"id":"a","content":"first"}],"merge":false})).await.unwrap();
    let output = tools
        .call(
            "todo_write",
            &json!({"todos":[{"id":"b","status":"completed"}],"merge":true}),
        )
        .await
        .unwrap();
    assert!(output.starts_with("- [completed] b: second\n- [pending] a: first\n"));
    tools.call("todo_write", &json!({"todos":[{"id":"b","content":""},{"id":"c","content":"","status":"cancelled"}]})).await.unwrap();
    let before = tools.snapshot().await;
    assert_eq!(
        before[0],
        json!({"id":"b","content":"second","status":"completed"})
    );
    assert_eq!(
        before[2],
        json!({"id":"c","content":"c","status":"cancelled"})
    );
    assert!(tools.call("todo_write", &json!({"todos":[{"id":"a","content":"changed"},{"id":"a","status":"completed"}],"merge":false})).await.is_err());
    assert_eq!(tools.snapshot().await, before);
    tools
        .call(
            "todo_write",
            &json!({"todos":[{"id":"b","status":"pending"}],"merge":false}),
        )
        .await
        .unwrap();
    assert_eq!(
        tools.snapshot().await,
        [json!({"id":"b","content":"b","status":"pending"})]
    );
    tools
        .call("todo_write", &json!({"todos":[],"merge":false}))
        .await
        .unwrap();
    assert!(tools.snapshot().await.is_empty());
}

#[tokio::test]
async fn grep_honors_paths_filters_regex_flags_and_output_limit() {
    let dir = tempfile::tempdir().unwrap();
    let tools = file_tools(dir.path());
    tokio::fs::create_dir(dir.path().join(".git"))
        .await
        .unwrap();
    tokio::fs::write(dir.path().join(".gitignore"), "ignored.rs\n")
        .await
        .unwrap();
    for (name, text) in [
        ("visible.rs", "UPPER α\nsecond\n"),
        ("visible.txt", "UPPER wrong type\n"),
        (".hidden.rs", "UPPER hidden\n"),
        ("ignored.rs", "UPPER ignored\n"),
    ] {
        tokio::fs::write(dir.path().join(name), text).await.unwrap();
    }
    let output = tools
        .call("grep", &json!({"pattern":"UPPER"}))
        .await
        .unwrap();
    assert!(
        output.contains("UPPER α") && output.contains("wrong type"),
        "{output}"
    );
    assert!(
        !output.contains("hidden") && !output.contains("ignored"),
        "{output}"
    );
    let output = tools
        .call(
            "grep",
            &json!({"pattern":"upper","-i":true,"type":"rust","-A":1}),
        )
        .await
        .unwrap();
    assert!(
        output.contains("UPPER α") && output.contains("second"),
        "{output}"
    );
    assert!(
        !output.contains("wrong type") && output.contains("hidden") && !output.contains("ignored"),
        "{output}"
    );
    let output = tools
        .call("grep", &json!({"pattern":"UPPER","glob":"*.txt"}))
        .await
        .unwrap();
    assert!(
        output.contains("wrong type") && !output.contains("UPPER α"),
        "{output}"
    );
    assert!(
        tools
            .call("grep", &json!({"pattern":"["}))
            .await
            .unwrap_err()
            .to_string()
            .contains("grep failed")
    );
    assert_eq!(
        tools
            .call("grep", &json!({"pattern":"no-such-match"}))
            .await
            .unwrap(),
        "No matches found"
    );
    tokio::fs::write(dir.path().join("many.txt"), "hit\n".repeat(50))
        .await
        .unwrap();
    let output = tools
        .call(
            "grep",
            &json!({"pattern":"hit","path":"many.txt","head_limit":3}),
        )
        .await
        .unwrap();
    assert!(
        output.contains("truncated") && output.lines().count() < 6,
        "{output}"
    );
    tokio::fs::write(dir.path().join("multi.txt"), "alpha\nbeta\n")
        .await
        .unwrap();
    assert!(
        tools
            .call(
                "grep",
                &json!({"pattern":"alpha.*beta","path":"multi.txt","multiline":true})
            )
            .await
            .unwrap()
            .contains("beta")
    );
}

#[tokio::test]
async fn pdfs_are_read_as_text_and_page_ranges_are_bounded() {
    let dir = tempfile::tempdir().unwrap();
    let tools = file_tools(dir.path());
    tokio::fs::write(dir.path().join("document.pdf"), b"%PDF-1.4\n")
        .await
        .unwrap();
    let error = tools
        .call(
            "read_file",
            &json!({"target_file":"document.pdf","format":"text"}),
        )
        .await
        .unwrap_err()
        .to_string();
    assert!(error.contains("pdfinfo"), "{error}");
    assert_eq!(page_range("3, 1-2, 2", 3).unwrap(), vec![1, 2, 3]);
    assert_eq!(page_range("2-", 3).unwrap(), vec![2, 3]);
    assert_eq!(page_range("1-99", 3).unwrap(), vec![1, 2, 3]);
    assert_eq!(page_range("1-20", 20).unwrap().len(), 20);
    for bad in ["0", "4", "2-1", "", ",", "word", "1-21"] {
        assert!(
            page_range(bad, if bad == "1-21" { 21 } else { 3 }).is_err(),
            "accepted {bad}"
        );
    }
}

#[tokio::test]
async fn pdf_text_and_selected_pages_when_poppler_is_installed() {
    let dir = tempfile::tempdir().unwrap();
    for program in ["pdfinfo", "pdftotext"] {
        if capture(
            Path::new(program),
            &["-v".into()],
            dir.path(),
            Duration::from_secs(2),
            4_000,
        )
        .await
        .is_err()
        {
            eprintln!(
                "skipping PDF text integration: optional Poppler utility {program} is absent"
            );
            return;
        }
    }
    let tools = file_tools(dir.path());
    tokio::fs::write(
        dir.path().join("pages.pdf"),
        test_pdf(&["First page text", "Second page text"]),
    )
    .await
    .unwrap();
    let text = tools
        .call(
            "read_file",
            &json!({"target_file":"pages.pdf","format":"text","pages":"2"}),
        )
        .await
        .unwrap();
    assert!(
        text.contains("--- Page 2 ---") && text.contains("Second page text"),
        "{text}"
    );
    assert!(!text.contains("First page text"), "{text}");
    let text = tools
        .call(
            "read_file",
            &json!({"target_file":"pages.pdf","format":"text"}),
        )
        .await
        .unwrap();
    assert!(
        text.contains("First page text") && text.contains("Second page text"),
        "{text}"
    );
    tokio::fs::write(dir.path().join("long.pdf"), test_pdf(&["Page text"; 11]))
        .await
        .unwrap();
    assert!(
        tools
            .call(
                "read_file",
                &json!({"target_file":"long.pdf","format":"text"})
            )
            .await
            .unwrap_err()
            .to_string()
            .contains("use pages")
    );
}

fn test_pdf(texts: &[&str]) -> Vec<u8> {
    let kids = (0..texts.len())
        .map(|i| format!("{} 0 R", 4 + i * 2))
        .collect::<Vec<_>>()
        .join(" ");
    let mut objects = vec![
        "<< /Type /Catalog /Pages 2 0 R >>".to_owned(),
        format!("<< /Type /Pages /Kids [{kids}] /Count {} >>", texts.len()),
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>".to_owned(),
    ];
    for (i, text) in texts.iter().enumerate() {
        objects.push(format!("<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 3 0 R >> >> /Contents {} 0 R >>", 5 + i * 2));
        let stream = format!("BT /F1 12 Tf 72 720 Td ({text}) Tj ET\n");
        objects.push(format!(
            "<< /Length {} >>\nstream\n{stream}endstream",
            stream.len()
        ));
    }
    let mut pdf = "%PDF-1.4\n".to_owned();
    let mut offsets = vec![0];
    for (i, object) in objects.iter().enumerate() {
        offsets.push(pdf.len());
        write!(pdf, "{} 0 obj\n{object}\nendobj\n", i + 1).unwrap();
    }
    let xref = pdf.len();
    write!(pdf, "xref\n0 {}\n0000000000 65535 f \n", offsets.len()).unwrap();
    for offset in offsets.iter().skip(1) {
        writeln!(pdf, "{offset:010} 00000 n ").unwrap();
    }
    write!(
        pdf,
        "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n",
        offsets.len()
    )
    .unwrap();
    pdf.into_bytes()
}

#[tokio::test]
async fn overlapping_matches_are_ambiguous_unless_replace_all() {
    let dir = tempfile::tempdir().unwrap();
    let tools = file_tools(dir.path());
    let path = dir.path().join("closing.rb");
    tokio::fs::write(&path, "end\nend\nend\n").await.unwrap();
    let edit = json!({"file_path":"closing.rb","old_string":"end\nend","new_string":"done"});
    let error = tools.call("search_replace", &edit).await.unwrap_err();
    assert!(
        error.to_string().contains("found multiple times"),
        "{error}"
    );
    assert_eq!(
        tokio::fs::read_to_string(&path).await.unwrap(),
        "end\nend\nend\n"
    );
    let mut edit = edit;
    edit["replace_all"] = json!(true);
    tools.call("search_replace", &edit).await.unwrap();
    assert_eq!(
        tokio::fs::read_to_string(&path).await.unwrap(),
        "done\nend\n"
    );
}

#[tokio::test]
async fn a_full_window_says_where_the_file_continues() {
    let dir = tempfile::tempdir().unwrap();
    let tools = file_tools(dir.path());
    let text = (1..=1500)
        .map(|n| format!("line {n}\n"))
        .collect::<String>();
    tokio::fs::write(dir.path().join("long.txt"), text)
        .await
        .unwrap();
    let first = tools
        .call("read_file", &json!({"target_file":"long.txt"}))
        .await
        .unwrap();
    assert!(
        first.contains("1000→line 1000\n[lines 1-1000 shown. The file continues at offset=1001.]")
    );
    let rest = tools
        .call(
            "read_file",
            &json!({"target_file":"long.txt","offset":1001}),
        )
        .await
        .unwrap();
    assert!(rest.starts_with("1001→line 1001") && rest.ends_with("1500→line 1500\n"));
    let chosen = tools
        .call(
            "read_file",
            &json!({"target_file":"long.txt","offset":10,"limit":2}),
        )
        .await
        .unwrap();
    assert_eq!(chosen, "10→line 10\nline 11");
}

#[tokio::test]
async fn a_missing_ripgrep_is_internal_and_a_bad_pattern_is_not() {
    let dir = tempfile::tempdir().unwrap();
    let missing = FileTools::new(
        dir.path().to_path_buf(),
        None,
        dir.path().join("no-such-rg"),
    );
    let error = missing
        .call("grep", &json!({"pattern":"x"}))
        .await
        .unwrap_err();
    assert!(error.downcast_ref::<InternalError>().is_some(), "{error:#}");
    let error = file_tools(dir.path())
        .call("grep", &json!({"pattern":"["}))
        .await
        .unwrap_err();
    assert!(error.downcast_ref::<InternalError>().is_none(), "{error:#}");
}

#[tokio::test]
async fn grep_returns_its_matches_and_three_errors_when_files_are_unreadable() {
    use std::os::unix::fs::PermissionsExt;
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("a.txt"), "needle here\n").unwrap();
    for n in 0..5 {
        let locked = dir.path().join(format!("locked-{n}.txt"));
        std::fs::write(&locked, "needle too\n").unwrap();
        std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o000)).unwrap();
    }
    if std::fs::read(dir.path().join("locked-0.txt")).is_ok() {
        eprintln!("skipping: this user can read a mode-000 file");
        return;
    }
    let output = file_tools(dir.path())
        .call("grep", &json!({"pattern":"needle"}))
        .await
        .unwrap();
    let (matches, note) = output.split_once("\n\n").unwrap();
    assert!(
        matches.contains("a.txt") && matches.contains("needle here"),
        "{output}"
    );
    let note: Vec<_> = note.lines().collect();
    assert_eq!(note.len(), 4, "{output}");
    assert!(
        note[..3]
            .iter()
            .all(|line| line.contains("locked-") && line.contains("Permission denied")),
        "{output}"
    );
    assert_eq!(note[3], "(2 more rg errors)");
    for n in 5..600 {
        let locked = dir
            .path()
            .join(format!("locked-{n}-{}.txt", "x".repeat(60)));
        std::fs::write(&locked, "needle too\n").unwrap();
        std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o000)).unwrap();
    }
    let output = file_tools(dir.path())
        .call("grep", &json!({"pattern":"needle"}))
        .await
        .unwrap();
    let last = output.lines().last().unwrap();
    assert!(
        last.starts_with("(at least ") && last.ends_with(" more rg errors)"),
        "{last}"
    );
}
