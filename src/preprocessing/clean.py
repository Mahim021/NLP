"""Text cleaning for MailSense.

Policy (PROJECT_SPEC.md Sec. 5): repair the text, never strip meaning.

We normalise whitespace, repair mojibake/encoding damage, unescape HTML
entities, drop stray HTML tags and replace URLs / e-mail addresses / long
digit strings with stable placeholder tokens.

We deliberately do NOT remove stopwords, modal or auxiliary verbs, action
verbs, temporal expressions, or politeness markers ("please", "ASAP",
"deadline", ...), because those are exactly the cues that signal whether an
e-mail asks the recipient to do something.  Case is also preserved here; each
model applies its own casing/tokenisation downstream.
"""
import html
import re
import unicodedata

# The dataset already uses "$LINK" for some redacted URLs, so we reuse that
# convention instead of inventing a second one.
URL_TOKEN = "$LINK"
EMAIL_TOKEN = "$EMAIL"
NUMBER_TOKEN = "$NUM"

_URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_HTML_TAG_RE = re.compile(r"<[^<>]{1,200}>")
_LONG_NUMBER_RE = re.compile(r"\b\d[\d,.\-]{7,}\b")
_WS_RE = re.compile(r"\s+")

# Classic UTF-8-read-as-cp1252 damage that survives in this dataset.
_MOJIBAKE = {
    "\ufffd": "",      # replacement character: the byte is already lost
    "\u00e2\u0080\u0099": "'",
    "\u00e2\u0080\u009c": '"',
    "\u00e2\u0080\u009d": '"',
    "\u00e2\u0080\u0093": "-",
    "\u00e2\u0080\u0094": "-",
    "\u00c2\u00a0": " ",
    "\u2018": "'",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u2013": "-",
    "\u2014": "-",
    "\u00a0": " ",
}


def clean_text(text: str) -> str:
    """Return the cleaned version of one raw dataset line."""
    if text is None:
        return ""
    out = unicodedata.normalize("NFKC", str(text))
    for bad, good in _MOJIBAKE.items():
        out = out.replace(bad, good)
    out = html.unescape(out)
    out = _HTML_TAG_RE.sub(" ", out)
    out = _URL_RE.sub(URL_TOKEN, out)
    out = _EMAIL_RE.sub(EMAIL_TOKEN, out)
    out = _LONG_NUMBER_RE.sub(NUMBER_TOKEN, out)
    # Drop control characters that carry no information.
    out = "".join(ch for ch in out if ch == "\t" or not unicodedata.category(ch).startswith("C"))
    out = _WS_RE.sub(" ", out).strip()
    return out


def is_meaningless(text: str, min_letters: int = 2) -> bool:
    """True for fragments with essentially no linguistic content."""
    letters = sum(ch.isalpha() for ch in text)
    return len(text) == 0 or letters < min_letters
