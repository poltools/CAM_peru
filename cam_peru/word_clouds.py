"""
Bilingual word-cloud helpers.

Two pipelines are exposed:

- :func:`generate_wordcloud_from_token_column` — renders a Spanish word
  cloud from a pandas Series of token lists.
- :func:`plot_wordcloud_EN` — translates the top N tokens to English
  (using :data:`CUSTOM_OVERRIDES` plus a CSV-backed per-project dictionary
  plus a Google Translate fallback) and renders the translated cloud.

The translation dictionary lives in ``data/CAM_translations.csv`` and is
loaded lazily to avoid hard-coding a path.
"""

from __future__ import annotations

import os
from collections import Counter
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from deep_translator import GoogleTranslator
from wordcloud import WordCloud

from .config import DATA_DIR

# --------------------------------------------------------------------------- #
# Translation overrides                                                       #
# --------------------------------------------------------------------------- #
# Built-in, small dictionary of culturally specific Peruvian terms that
# GoogleTranslator tends to get wrong.
CUSTOM_OVERRIDES: dict[str, str] = {
    "usar":       "use",
    "pie":        "foot",
    "tecnico":    "technical",
    "mayor":      "larger",
    "mano":       "hand",
    "escuchar":   "listen",
    "milenario":  "ancient",
    "cuy":        "guinea pig",
    "sobada":     "cleansing",
    "pasada":     "cleansing",
    "alumbre":    "alum",
    "pago":       "offering",
    "coca":       "coca",
    "persona":    "person",
    "musico":     "musician",
    "musica":     "music",
    "estr":       "stress",
    "medico":     "doctor",
    "fisico":     "physical",
    "efectivo":   "effective",
    "sentir":     "feel",
    "tratar":     "treat",
    "agua":       "water",
    "solo":       "only",
    "bueno":      "good",
    "asi":        "like that",
    "dia":        "day",
    "columna":    "spine",
    "susto":      "susto",
    "ojo":        "eye",
    "mal":        "evil",
    "ano":        "year",
    "animo":      "mood",
    "servir":     "useful",
    "sano":       "healthy",
    "lesion":     "injury",
    "nino":       "child",
    "verdad":     "truth",
    "propio":     "own",
    "vibra":      "vibration",
    "buena":      "good",
    "organo":     "organ",
    "amigo":      "friend",
    "traves":     "through",
    "bano":       "bath",
    "dano":       "damage",
    "organismo":  "organism",
    "piel":       "skin",
    "gran":       "big",
    "tema":       "topic",
    "chino":      "chinese",
    "trabajo":    "work",
    "soler":      "often",
    "curandero":  "healer",
}


def load_translation_overrides(
    csv_path: os.PathLike | str | None = None,
) -> dict[str, str]:
    """Return the combined ``{spanish → english}`` override dictionary.

    Reads ``data/CAM_translations.csv`` by default, with columns
    ``word, translation``. Missing file returns just :data:`CUSTOM_OVERRIDES`.
    """
    csv_path = Path(csv_path) if csv_path else DATA_DIR / "CAM_translations.csv"
    if not csv_path.exists():
        return dict(CUSTOM_OVERRIDES)
    project_overrides = pd.read_csv(csv_path)
    project_overrides.columns = ["word", "translation"]
    project_map = dict(zip(project_overrides["word"], project_overrides["translation"]))
    return {**CUSTOM_OVERRIDES, **project_map}


def flatten_tokens(tokens):
    """Recursively flatten nested iterables of tokens into a generator of str."""
    for token in tokens:
        if isinstance(token, str):
            yield token
        elif isinstance(token, Iterable) and not isinstance(token, (str, bytes)):
            yield from flatten_tokens(token)
        else:
            yield str(token)


def _save(fig_path: Path, title: str) -> None:
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.title(title, fontsize=18)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")


def generate_wordcloud_from_token_column(
    token_column: pd.Series,
    title: str,
    max_words: int = 200,
    output_dir: os.PathLike | str = "wordclouds",
) -> Path:
    """Render a Spanish word cloud from a Series of token lists."""
    flat_tokens = list(flatten_tokens(token_column.sum()))
    wordcloud = WordCloud(
        width=1600,
        height=800,
        background_color="white",
        colormap="viridis",
        max_words=max_words,
    ).generate(" ".join(flat_tokens))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_title = title.lower().replace(" ", "_")
    fig_path = Path(output_dir) / f"{safe_title}_{timestamp}.png"

    plt.figure(figsize=(12, 6))
    plt.imshow(wordcloud, interpolation="bilinear")
    _save(fig_path, title)
    plt.close()
    print(f"✅ Word cloud saved to: {fig_path}")
    return fig_path


def get_word_frequencies(token_column: pd.Series) -> Counter:
    return Counter(flatten_tokens(token_column.sum()))


def translate_words_with_overrides(
    words: Iterable[str],
    source_lang: str = "auto",
    target_lang: str = "en",
    overrides: dict[str, str] | None = None,
) -> dict[str, str]:
    """Translate tokens with a manual override map first, then fall back to GT."""
    overrides = overrides or {}
    translator = GoogleTranslator(source=source_lang, target=target_lang)
    translation_map: dict[str, str] = {}
    for word in words:
        if word in overrides:
            translation_map[word] = overrides[word]
            continue
        try:
            translation_map[word] = translator.translate(word)
        except Exception:
            translation_map[word] = word
    return translation_map


def plot_translated_wordcloud(
    word_freq: Counter,
    translation_map: dict[str, str],
    title: str,
    max_words: int = 200,
    output_dir: os.PathLike | str = "wordclouds",
) -> Path:
    translated_freq = {
        translation_map[word]: freq
        for word, freq in word_freq.items()
        if word in translation_map
    }

    wordcloud = WordCloud(
        width=1600,
        height=800,
        background_color="white",
        colormap="viridis",
        max_words=max_words,
    ).generate_from_frequencies(translated_freq)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_title = title.lower().replace(" ", "_")
    fig_path = Path(output_dir) / f"{safe_title}_{timestamp}.png"

    plt.figure(figsize=(12, 6))
    plt.imshow(wordcloud, interpolation="bilinear")
    _save(fig_path, title)
    plt.close()
    print(f"✅ Word cloud saved to: {fig_path}")
    return fig_path


def plot_wordcloud_EN(
    tokenized_column: pd.Series,
    custom_overrides: dict[str, str] | None = None,
    title: str = "",
    n_words: int = 100,
    translation_map: dict[str, str] | None = None,
) -> dict[str, str]:
    """Render an English word cloud; returns the effective translation map."""
    overrides = custom_overrides or load_translation_overrides()
    frequencies = get_word_frequencies(tokenized_column)
    top_words = [w for w, _ in frequencies.most_common(n_words)]
    if not translation_map:
        translation_map = translate_words_with_overrides(top_words, overrides=overrides)
    plot_translated_wordcloud(frequencies, translation_map, title, max_words=n_words)
    return translation_map


__all__ = [
    "CUSTOM_OVERRIDES",
    "load_translation_overrides",
    "flatten_tokens",
    "generate_wordcloud_from_token_column",
    "get_word_frequencies",
    "translate_words_with_overrides",
    "plot_translated_wordcloud",
    "plot_wordcloud_EN",
]
