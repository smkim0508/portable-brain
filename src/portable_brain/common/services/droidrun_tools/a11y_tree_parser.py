import re

# NOTE: helper to parse the raw a11y tree into a more human-readable format without noise
# used for LLM inference on actions without fragile guessing

# patterns that indicate a quoted string is a resource ID / class name, not human-readable text
_NOISE_PATTERNS = re.compile(
    r'^(com\.|android\.|androidx\.|org\.)'
)
# patterns that look like internal UI identifiers (camelCase, snake_case, PascalCase dev labels)
# e.g., "ConversationScreenUi", "message_list", "top_app_bar", "ComposeRowIcon:Shortcuts"
_INTERNAL_ID_PATTERNS = re.compile(
    r'^[a-z]+[A-Z]'          # camelCase: "monogramTest", "messageList"
    r'|^[A-Z][a-z]+[A-Z]'   # PascalCase compound: "ConversationScreenUi", "GlideMonogram"
    r'|^[a-z]+_[a-z]'       # snake_case: "message_list", "top_app_bar", "text_separator"
    r'|^[A-Z]\w+:[A-Z]'     # PascalCase colon-separated: "ComposeRowIcon:Shortcuts", "Compose:Draft:Send"
)
# generic action buttons that don't carry semantic value, so we can filter out this noise
_GENERIC_ACTIONS = {
    "more options", "more actions", "action menu",
}

def denoise_formatted_text(formatted_text: str, max_lines: int = 50) -> str:
    """
    Denoises a formatted_text string from DroidRun's get_state().
    Filters to elements with human-readable text, strips resource IDs,
    deduplicates content, and removes generic action buttons.

    Args:
        formatted_text: The formatted_text string (raw_state[0] from get_state())
        max_lines: Maximum number of lines to keep

    Returns:
        Compressed formatted text with header + text-bearing elements only
    """
    if not formatted_text:
        return ""

    lines = formatted_text.strip().split("\n")
    compressed = []
    seen_text = set()  # track seen readable strings to deduplicate

    for line in lines:
        # keep phone state header lines (app name, keyboard, focused element)
        if line.startswith("**") or line.startswith("•"):
            compressed.append(line)
            continue

        # skip the schema description line
        if line.startswith("Current Clickable UI elements"):
            continue

        # extract all quoted strings from the line
        quoted = re.findall(r'"([^"]*)"', line)
        if not quoted:
            continue

        # separate readable text from resource IDs and internal UI identifiers
        readable = [
            q for q in quoted
            if q
            and not _NOISE_PATTERNS.match(q)
            and not _INTERNAL_ID_PATTERNS.search(q)
        ]
        if not readable:
            continue

        # skip generic action buttons
        if len(readable) == 1 and readable[0].lower() in _GENERIC_ACTIONS:
            continue

        # deduplicate by readable text content
        text_key = tuple(readable)
        if text_key in seen_text:
            continue
        seen_text.add(text_key)

        # rebuild line: strip resource IDs from quoted strings, keep only readable text
        # extract the element prefix (e.g., "24. Button: ")
        prefix_match = re.match(r'^(\d+\.\s*\w+:\s*)', line)
        if prefix_match:
            prefix = prefix_match.group(1)
            readable_str = ", ".join(f'"{r}"' for r in readable)
            cleaned = f"{prefix}{readable_str}"
        else:
            cleaned = line

        # strip bounds info (e.g., "- (389,1990,1017,2053)")
        cleaned = re.sub(r'\s*-\s*\(\d+,\d+,\d+,\d+\)\s*$', '', cleaned).strip()
        if cleaned:
            compressed.append(cleaned)

    return "\n".join(compressed[:max_lines])
