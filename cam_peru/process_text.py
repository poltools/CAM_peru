"""
Spanish text preprocessing utilities.

The ``preprocess_spanish_text`` function lemmatises a Spanish string with
spaCy, strips stop-words and selected function-word POS tags, and returns a
list of normalised lemmas. It is used upstream of both the word-cloud and
embedding pipelines.

Before first use::

    python -m spacy download es_core_news_md
    python -c "import nltk; nltk.download('stopwords')"

(The spaCy model is also pinned as a wheel in ``requirements.txt``.)
"""

from __future__ import annotations

import re
from typing import Iterable, Optional

import nltk
import spacy
from nltk.corpus import stopwords
from unidecode import unidecode

# Download resources lazily-idempotently. ``quiet=True`` keeps stdout clean
# when imported inside notebooks.
try:
    stopwords.words("spanish")
except LookupError:  # pragma: no cover — first run only
    nltk.download("stopwords", quiet=True)

# Loading the spaCy model at import time keeps inference cheap inside loops.
_NLP = spacy.load("es_core_news_md")


def process_annotations(annotations: str) -> list[str]:
    """Parse a pseudo-list string (e.g. ``"[foo, bar]"``) into ``[foo, bar]``.

    Useful when extracted-reason columns are serialised as text in CSVs.
    """
    anns_ = (
        annotations.replace("' ", "")
        .replace("'", "")
        .replace("[", "")
        .replace("]", "")
        .replace('"', "")
        .split(",")
    )
    return [a.strip() for a in anns_]


def remove_accents(text: str) -> str:
    """Deterministic accent stripping via :mod:`unidecode`."""
    return unidecode(text)


def preprocess_spanish_text(
    text: str,
    nlp=_NLP,
    preserve_words: Optional[Iterable[str]] = None,
    stop_words: Optional[Iterable[str]] = None,
    custom_excluded: Optional[Iterable[str]] = None,
    excluded_lemmas: Optional[Iterable[str]] = None,
    excluded_pos: Optional[Iterable[str]] = None,
    return_debug: bool = False,
):
    """Preprocess a Spanish string and return a list of cleaned lemmas.

    Parameters
    ----------
    text
        Raw Spanish text.
    nlp
        A loaded spaCy pipeline. Defaults to ``es_core_news_md``.
    preserve_words
        Words that should be kept verbatim (after lower-casing and accent
        stripping) rather than lemmatised. Useful for culturally specific
        terms like ``"cuy"`` or ``"susto"``.
    stop_words
        Iterable of stop-words. Defaults to NLTK Spanish stop-words.
    custom_excluded
        Additional words to remove after accent normalisation (useful for
        over-frequent but non-informative tokens).
    excluded_lemmas
        Verb/common-word lemmas to drop (e.g. ``{"ser", "hacer"}``).
    excluded_pos
        spaCy POS tags to drop (default removes pronouns, determiners, and
        most function-word tags).
    return_debug
        If True, also returns the discarded ``(word, lemma, pos)`` triples.

    Returns
    -------
    list[str] | tuple[list[str], list[tuple]]
    """
    preserve_words = set(preserve_words or [])
    stop_words = set(
        remove_accents(w.lower())
        for w in (stop_words if stop_words is not None else stopwords.words("spanish"))
    )
    custom_excluded = set(
        remove_accents(w.lower())
        for w in (custom_excluded or {
            "yo", "tu", "vos", "usted", "nosotros", "vosotros", "ellos", "ellas",
            "el", "ella", "uno", "una", "vida", "emocion", "sentimiento", "alma",
            "algo", "alguien", "nadie",
        })
    )
    excluded_lemmas = set(
        remove_accents(w.lower())
        for w in (excluded_lemmas or {
            "ser", "estar", "haber", "tener", "hacer", "poder", "decir", "ir", "ver", "dar",
            "saber", "querer", "llegar", "pasar", "deber", "poner", "parecer", "quedar",
            "creer", "llevar", "dejar", "seguir", "encontrar", "llamar", "venir", "volver",
            "ademas",
        })
    )
    excluded_pos = set(excluded_pos or {"PRON", "DET", "ADP", "CCONJ", "SCONJ", "PART", "INTJ"})

    discarded: list[tuple[str, str, str]] = []

    # Lower-case, strip non-letters (keep Spanish accents + ñ), collapse whitespace.
    text = text.lower()
    text = re.sub(r"[^a-záéíóúüñ\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    doc = nlp(text)
    clean_tokens: list[str] = []

    for token in doc:
        word = token.text
        lemma = token.lemma_

        if word in preserve_words:
            clean_tokens.append(remove_accents(word.lower()))
            continue

        norm_lemma = remove_accents(lemma.lower())

        if (
            token.is_alpha
            and norm_lemma not in stop_words
            and norm_lemma not in custom_excluded
            and norm_lemma not in excluded_lemmas
            and token.pos_ not in excluded_pos
            and " " not in norm_lemma
            and len(norm_lemma.split()) == 1
        ):
            clean_tokens.append(norm_lemma)
        elif return_debug:
            discarded.append((word, lemma, token.pos_))

    return (clean_tokens, discarded) if return_debug else clean_tokens
