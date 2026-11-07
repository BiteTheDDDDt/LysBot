"""Reuse summary reading to select original messages for durable memory."""
import json


def parse_summary_selection(raw, messages):
    text = str(raw).strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        data = json.loads(text)
    except (ValueError, TypeError):
        if text.startswith(("{", "[")):
            raise ValueError("invalid_summary_json")
        return text, messages  # Legacy plain summaries preserve the extraction job.
    if not isinstance(data, dict) or not isinstance(data.get("summary"), str) or not data["summary"].strip():
        raise ValueError("invalid_summary")
    selected = data.get("memory_candidates")
    if not isinstance(selected, list) or any(type(i) is not int or not 1 <= i <= len(messages) for i in selected):
        raise ValueError("invalid_memory_candidates")
    indices = set()
    for i in selected:
        if messages[i - 1].role != "user" or messages[i - 1].is_thinking:
            continue
        indices.update(range(max(0, i - 2), min(len(messages), i + 1)))
    return data["summary"].strip(), [m for i, m in enumerate(messages) if i in indices and not m.is_thinking]
