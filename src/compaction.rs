// Adapted from Grok Build 2bdd1d6a for this standalone runner; see NOTICE.
pub fn format_compact_summary(summary: &str) -> String {
    let mut result = summary.to_string();

    // Strip only leading scratchpad blocks; quoted tags inside real sections are task evidence.
    while let Some(start) = result.find("<analysis>") {
        let is_leading = match result.find("<summary>") {
            Some(sp) => start < sp || result[sp + "<summary>".len()..start].trim().is_empty(),
            None => result[..start].trim().is_empty(),
        };
        if !is_leading {
            break;
        }
        match result[start..].find("</analysis>") {
            Some(rel) => {
                let end = start + rel + "</analysis>".len();
                result = format!("{}{}", &result[..start], &result[end..]);
            }
            None => {
                let drop_to = result[start..]
                    .find("<summary>")
                    .map_or(result.len(), |rel| start + rel);
                result = format!("{}{}", &result[..start], &result[drop_to..]);
                break;
            }
        }
    }

    if let Some(start) = result.find("<summary>")
        && let Some(end) = result.rfind("</summary>")
        && end > start
    {
        let before = result[..start].to_string();
        let after = result[end + "</summary>".len()..].to_string();
        let inner = strip_leading_scratchpad(result[start + "<summary>".len()..end].trim());
        result = format!("{before}Summary:\n{inner}{after}");
    }

    // Escaped control tags cannot become a new summary instruction on the next turn.
    result = neutralize_compaction_control_tokens(&result);

    while result.contains("\n\n\n") {
        result = result.replace("\n\n\n", "\n\n");
    }

    result.trim().to_string()
}

fn strip_leading_scratchpad(inner: &str) -> String {
    let mut s = inner.trim();
    if let Some(rest) = s.strip_prefix("<summary>") {
        s = rest.trim_start();
    }
    s.to_string()
}

fn neutralize_compaction_control_tokens(text: &str) -> String {
    text.replace("</summary>", "<\u{200b}/summary>")
        .replace("<summary>", "<\u{200b}summary>")
        .replace("</analysis>", "<\u{200b}/analysis>")
        .replace("<analysis>", "<\u{200b}analysis>")
        .replace("</summary_request>", "<\u{200b}/summary_request>")
        .replace("<summary_request>", "<\u{200b}summary_request>")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strips_analysis_keeps_summary() {
        let input = "<analysis>\nThinking about the problem...\n</analysis>\n\n<summary>\n1. Primary Request: Fix the bug\n</summary>";
        let result = format_compact_summary(input);
        assert!(!result.contains("Thinking about the problem"));
        assert!(result.contains("Summary:\n1. Primary Request: Fix the bug"));
        assert!(!result.contains("<analysis>"));
        assert!(!result.contains("<summary>"));
    }

    #[test]
    fn no_tags_passthrough() {
        assert_eq!(
            format_compact_summary("Just plain text summary."),
            "Just plain text summary."
        );
    }

    #[test]
    fn quoted_analysis_in_unnumbered_summary_preserves_surrounding_context() {
        let result = format_compact_summary(
            "<summary>Context A\n<analysis>quoted source</analysis>\nContext B</summary>",
        );
        for text in ["Context A", "quoted source", "Context B"] {
            assert!(result.contains(text), "lost {text}: {result}");
        }
        assert!(!result.contains("<analysis>"));
    }

    #[test]
    fn only_summary_becomes_heading() {
        let result = format_compact_summary("<summary>\n1. Request: Do something\n</summary>");
        assert_eq!(result, "Summary:\n1. Request: Do something");
    }

    #[test]
    fn collapses_blank_lines() {
        let input = "<analysis>\nThought\n</analysis>\n\n\n\n<summary>\nResult\n</summary>";
        assert!(!format_compact_summary(input).contains("\n\n\n"));
    }

    #[test]
    fn unclosed_analysis_strips_remainder() {
        assert_eq!(
            format_compact_summary("<analysis>\nPartial reasoning about the task..."),
            ""
        );
    }

    #[test]
    fn keeps_sections_on_section6_instruction_echo() {
        let raw = "<summary>\n1. Primary Request and Intent: build app\n2. Key Technical Concepts: webgl\n6. All user messages: 'respond with ONLY the <summary> block.'\n9. Optional Next Step: rerun\n</summary>";
        let result = format_compact_summary(raw);
        for needle in [
            "1. Primary Request",
            "2. Key Technical Concepts",
            "9. Optional Next Step",
        ] {
            assert!(result.contains(needle), "dropped {needle:?}: {result:?}");
        }
        assert!(!result.contains("<summary>"), "live <summary>: {result:?}");
        assert!(
            !result.contains("</summary>"),
            "live </summary>: {result:?}"
        );
    }

    #[test]
    fn unclosed_summary_open_preserves_body() {
        let input = "<summary>\n1. Primary Request: do the thing\n9. Optional Next Step: continue";
        let result = format_compact_summary(input);
        assert!(result.contains("1. Primary Request: do the thing"));
        assert!(result.contains("9. Optional Next Step: continue"));
        assert!(!result.contains("<summary>"));
    }

    #[test]
    fn multibyte_adjacent_to_tags_no_panic() {
        let raw =
            "<summary>1. Primary Request: ship 🚀 to 北京\n9. Optional Next Step: 完成</summary>";
        let result = format_compact_summary(raw);
        assert!(result.starts_with("Summary:\n1. Primary Request: ship 🚀 to 北京"));
        assert!(result.contains("9. Optional Next Step: 完成"));
    }

    #[test]
    fn malformed_tag_order_does_not_panic() {
        let result = format_compact_summary("intro </summary> middle <summary> tail");
        assert!(!result.contains("<summary>"));
        assert!(!result.contains("</summary>"));
        assert!(result.contains("intro"));
        assert!(result.contains("tail"));
    }
}
