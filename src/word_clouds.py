import os
from datetime import datetime
from collections import Counter
from collections.abc import Iterable
import pandas as pd

import spacy
import nltk
from wordcloud import WordCloud
import matplotlib.pyplot as plt

from collections import Counter
from deep_translator import GoogleTranslator
from wordcloud import WordCloud
import matplotlib.pyplot as plt


# loading custom dictionary for the project

custom_overrides = {
    "usar": "use",
    "pie": "foot",
    "tecnico": "technical",
    "mayor": "larger",
    "mano": "hand",
    "escuchar": "listen",
    "milenario": "ancient",
    "cuy": "guinnea pig",
    "sobada": "cleansing",
    "pasada": "cleansing",
    "alumbre": "alum",
    "pago": "offering",
    "coca": "coca",
    "persona": "person",
    #"mal": "bad",
    "musico": "musician",
    "musica": "music",
    "estr": "stress",
    "medico": "doctor",
    "fisico": "physical",
    "efectivo": "effective",
    "sentir": "feel",
    "tratar": "treat",
    "agua": "water",
    "solo": "only",
    "bueno": "good",
    "asi": "like that",
    "dia": "day",
    "columna": "spine",
    "susto": "susto",
    "ojo": "eye",
    "mal": "evil",
    "ano": "year",
    "animo": "mood",
    "servir": "useful",
    "sano": "healthy",
    "lesion": "injury",
    "nino": "child",
    "verdad": "truth",
    "propio": "own",
    "vibra": "vibration",
    "buena": "good",
    "organo": "organ",
    "amigo": "friend",
    "traves": "through",
    "bano": "bath",
    "dano": "damage",
    "organismo": "organism",
    "piel": "skin",
    "gran": "big",
    "tema": "topic",
    "chino": "chinese",
    "trabajo": "work",
    "soler": "often",
    "curandero": "healer"
}

overrides_dict = pd.read_csv('CAM_translations.csv')
overrides_dict.columns = ['word', 'translation']
overrides_dict = dict(zip(overrides_dict['word'], overrides_dict['translation']))

combined_dict = custom_overrides | overrides_dict


def flatten_tokens(tokens):
    """Recursively flatten nested token structures (e.g., lists, tuples)."""
    for token in tokens:
        if isinstance(token, str):
            yield token
        elif isinstance(token, Iterable) and not isinstance(token, (str, bytes)):
            yield from flatten_tokens(token)
        else:
            yield str(token)

def generate_wordcloud_from_token_column(token_column, title, max_words=200, output_dir='wordclouds'):
    # Flatten the token lists
    flat_tokens = list(flatten_tokens(token_column.sum()))

    # Generate word cloud
    wordcloud = WordCloud(
        width=1600,  # double resolution
        height=800,
        background_color='white',
        colormap='viridis',
        max_words=max_words
    ).generate(' '.join(flat_tokens))

    # Prepare filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_title = title.lower().replace(' ', '_')
    filename = f"{safe_title}_{timestamp}.png"
    filepath = os.path.join(output_dir, filename)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save figure with high quality
    plt.figure(figsize=(12, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title, fontsize=18)
    plt.tight_layout()
    # plt.show()
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    print(f"✅ Word cloud saved to: {filepath}")

####
# Translated to English
####

# Step 1: Get flattened word frequencies
def get_word_frequencies(token_column):
    raw_tokens = token_column.sum()
    flat_tokens = list(flatten_tokens(raw_tokens))
    return Counter(flat_tokens)

# Step 2: Translate words with optional overrides
def translate_words_with_overrides(words, source_lang='auto', target_lang='en', overrides=None):
    overrides = overrides or {}

    translation_map = {}
    for word in words:
        if word in overrides:
            translation_map[word] = overrides[word]
        else:
            try:
                translated = GoogleTranslator(source=source_lang, target=target_lang).translate(word)
            except Exception:
                translated = word  # fallback if translation fails
            translation_map[word] = translated
    return translation_map

# Step 3: Plot translated word cloud
def plot_translated_wordcloud(word_freq, translation_map, title, max_words=200, output_dir='wordclouds'):
    # Translate word frequencies
    translated_freq = {
        translation_map[word]: freq
        for word, freq in word_freq.items()
        if word in translation_map
    }

    # Create WordCloud object
    wordcloud = WordCloud(
        width=1600,  # double resolution
        height=800,
        background_color='white',
        colormap='viridis',
        max_words=max_words
    ).generate_from_frequencies(translated_freq)

    # Prepare filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_title = title.lower().replace(' ', '_')
    filename = f"{safe_title}_{timestamp}.png"
    filepath = os.path.join(output_dir, filename)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Plot and save with high quality
    plt.figure(figsize=(12, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title, fontsize=18)
    plt.tight_layout()
    # plt.show()
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    print(f"✅ Word cloud saved to: {filepath}")
    return filepath


custom_overrides = {
    "usar": "use",
    "pie": "foot",
    "tecnico": "technical",
    "mayor": "larger",
    "mano": "hand",
    "escuchar": "listen",
    "milenario": "ancient",
    "cuy": "guinnea pig",
    "sobada": "cleansing",
    "pasada": "cleansing",
    "alumbre": "alum",
    "pago": "offering",
    "coca": "coca",
    "persona": "person",
    #"mal": "bad",
    "musico": "musician",
    "musica": "music",
    "estr": "stress",
    "medico": "doctor",
    "fisico": "physical",
    "efectivo": "effective",
    "sentir": "feel",
    "tratar": "treat",
    "agua": "water",
    "solo": "only",
    "bueno": "good",
    "asi": "like that",
    "dia": "day",
    "columna": "spine",
    "susto": "susto",
    "ojo": "eye",
    "mal": "evil",
    "ano": "year",
    "animo": "mood",
    "servir": "useful",
    "sano": "healthy",
    "lesion": "injury",
    "nino": "child",
    "verdad": "truth",
    "propio": "own",
    "vibra": "vibration",
    "buena": "good",
    "organo": "organ",
    "amigo": "friend",
    "traves": "through",
    "bano": "bath",
    "dano": "damage",
    "organismo": "organism",
    "piel": "skin",
    "gran": "big",
    "tema": "topic",
    "chino": "chinese",
    "trabajo": "work",
    "soler": "often",
    "curandero": "healer"
}


def plot_wordcloud_EN(tokenized_column: pd.Series, custom_overrides: dict, title: str, n_words:int = 100, translation_map=None):
    frequencies = get_word_frequencies(tokenized_column)
    top_words = [word for word, _ in frequencies.most_common(n_words)]
    if not translation_map:
        translation_map = translate_words_with_overrides(top_words, overrides=custom_overrides)
    plot_translated_wordcloud(frequencies, translation_map, title, max_words=n_words)
    return translation_map