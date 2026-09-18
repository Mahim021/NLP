import html
import re
import unicodedata

URL_TOKEN = "$LINK"
EMAIL_TOKEN = "$EMAIL"
NUMBER_TOKEN = "$NUM"

URL_RE = re.compile(
    r"<(?:https?://|www\.)[^>\s]+>"
    r"|(?:https?://|www\.)[^\s<>]+",
    re.IGNORECASE
)

EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")

HTML_RE = re.compile(r"<[^>]+>")

NUMBER_RE = re.compile(r"\b\d[\d,.\-]*\b")

WHITESPACE_RE = re.compile(r"\s+")


def clean_text(text):
    if text is None:
        return ""

    text = str(text)

    text = unicodedata.normalize("NFKC", text)
    text = html.unescape(text)

    # Replace URLs before removing HTML-like brackets.
    text = URL_RE.sub(URL_TOKEN, text)

    # Replace email addresses.
    text = EMAIL_RE.sub(EMAIL_TOKEN, text)

    # Remove remaining HTML tags.
    text = HTML_RE.sub(" ", text)

    # Replace numbers.
    text = NUMBER_RE.sub(NUMBER_TOKEN, text)

    # Normalize whitespace.
    text = WHITESPACE_RE.sub(" ", text).strip()

    return text


def is_meaningless(text, min_letters=2):
    if not text:
        return True

    letter_count = sum(
        character.isalpha()
        for character in text
    )

    return letter_count < min_letters