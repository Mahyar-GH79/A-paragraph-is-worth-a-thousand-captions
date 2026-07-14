"""Text normalisation helpers shared by the training and evaluation code."""

from typing import Any


def clean_text(value: Any) -> str | None:
    """Return a stripped string, or ``None`` if ``value`` is not usable text."""
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def clean_str_list(values: Any) -> list[str]:
    """Return the non-empty strings of ``values``, stripped and in order."""
    if not isinstance(values, (list, tuple)):
        return []
    cleaned = []
    for value in values:
        text = clean_text(value)
        if text is not None:
            cleaned.append(text)
    return cleaned
