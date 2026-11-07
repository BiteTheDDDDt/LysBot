"""Normalize malformed UTF-16 received through otherwise valid JSON."""

import re

_SURROGATE = re.compile(r"[\ud800-\udfff]")


def unicode_safe(value):
    """Keep valid text, combine surrogate pairs, and replace lone surrogates."""
    if isinstance(value, str):
        if _SURROGATE.search(value):
            return value.encode("utf-16-le", "surrogatepass").decode("utf-16-le", "replace")
        return value
    if isinstance(value, list):
        return [unicode_safe(item) for item in value]
    if isinstance(value, tuple):
        return tuple(unicode_safe(item) for item in value)
    if isinstance(value, dict):
        return {unicode_safe(key): unicode_safe(item) for key, item in value.items()}
    return value
