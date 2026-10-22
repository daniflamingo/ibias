"""Turn article text into words and phrases.

Gentzkow & Shapiro work with two- and three-word phrases ("bigrams" and
"trigrams") after removing very common words and reducing words to a stem.
We start simpler: lowercase words, drop stopwords, then build n-grams.
Stemming can come later; it is easy to add here without touching other files.
"""

import re
from collections import Counter

# Letters, optionally joined by an apostrophe or hyphen: "don't", "co-operate".
WORD_RE = re.compile(r"[a-z]+(?:['\-][a-z]+)*")

# A short stopword list. Words this common say little about politics,
# and dropping them makes the phrase counts far more informative.
STOPWORDS = frozenset("""
a about above after again against all also am an and any are as at be because been before
being below between both but by can could did do does doing down during each few for from
further had has have having he her here hers herself him himself his how i if in into is it
its itself just me more most my myself no nor not now of off on once only or other our ours
ourselves out over own same she should so some such than that the their theirs them
themselves then there these they this those through to too under until up very was we were
what when where which while who whom why will with would you your yours yourself yourselves
said says say one two new year years like get got also mr mrs ms
""".split())


def tokenize(text: str) -> list[str]:
    text = text.lower().replace("’", "'")  # curly apostrophes to straight
    return [w for w in WORD_RE.findall(text) if w not in STOPWORDS and len(w) > 1]


def ngrams(tokens: list[str], n: int) -> list[str]:
    return [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def count_terms(text: str, sizes: tuple[int, ...] = (1, 2, 3)) -> Counter:
    """Count words (n=1) and phrases (n=2, 3) in one text."""
    tokens = tokenize(text)
    counts: Counter = Counter()
    for n in sizes:
        counts.update(ngrams(tokens, n))
    return counts
