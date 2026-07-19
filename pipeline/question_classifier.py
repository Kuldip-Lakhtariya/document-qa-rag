import re

_BROAD_QUESTION_PATTERN = re.compile(
    r"\bsummar\w*\b"                              # summary, summarize, summarization
    r"|\boverview\b"
    r"|\bgist\b"
    r"|\bmain point\w*\b"
    r"|\bkey point\w*\b"
    r"|\b(whole|entire|full) document\b"
    r"|\bhow many (chapters|sections|pages)\b"
    r"|\boutline\b"
    r"|\bwhat.{0,15}(doc|document).{0,10}about\b"  # "what is this doc about", "what's this document all about"
    r"|\btl;?dr\b"
    r"|\bwalk me through\b",
    re.IGNORECASE
)


def is_broad_question(question: str) -> bool:
    """
    True if the question needs the entire document as context (summary,
    overview, count of sections, etc.), False if it's answerable from a
    handful of relevant chunks via similarity search.
    """
    return bool(_BROAD_QUESTION_PATTERN.search(question))
