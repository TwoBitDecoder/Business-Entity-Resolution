import re
import unicodedata

_SPACE_RE = re.compile(r"\s+")
_NON_WORD_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
_DIGIT_RE = re.compile(r"\d+")


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value or "").casefold()
    value = _NON_WORD_RE.sub(" ", value)
    return _SPACE_RE.sub(" ", value).strip()


def compact_text(value: str) -> str:
    return normalize_text(value).replace(" ", "")


def numeric_tokens(value: str) -> tuple[str, ...]:
    return tuple(_DIGIT_RE.findall(normalize_text(value)))
