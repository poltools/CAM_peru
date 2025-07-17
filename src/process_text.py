from unidecode import unidecode
import nltk, spacy

nltk.download('stopwords')
nlp = spacy.load("es_core_news_md")


def process_annotations(annotations):
    anns_ = annotations.replace("' ", "").replace("'", "").replace("[", "").replace("]", "").replace('"', "").split(",")
    anns_ = [a.strip() for a in anns_]
    return anns_


# Helper function
def remove_accents(text):
    return unidecode(text)

def preprocess_spanish_text(
    text,
    nlp=nlp,
    preserve_words=None,
    stop_words=None,
    custom_excluded=None,
    excluded_lemmas=None,
    excluded_pos=None,
    return_debug=False
):
    """
    Preprocess Spanish text: normalize, lemmatize, remove stopwords, and filter by POS.
    
    Parameters:
    - text (str): Raw text input.
    - nlp: spaCy language model.
    - preserve_words (set): Words to preserve without lemmatizing.
    - stop_words (set): Optional preloaded stopwords.
    - custom_excluded (set): Extra words to exclude.
    - excluded_lemmas (set): Lemmas to exclude (e.g., common verbs).
    - excluded_pos (set): Parts of speech to exclude.
    - return_debug (bool): If True, also return discarded tokens.

    Returns:
    - List of cleaned tokens.
    """
    # --- Defaults ---
    preserve_words = set(preserve_words or [])
    stop_words = set(remove_accents(w.lower()) for w in (stop_words or stopwords.words('spanish')))
    custom_excluded = set(remove_accents(w.lower()) for w in (custom_excluded or {
        'yo', 'tu', 'vos', 'usted', 'nosotros', 'vosotros', 'ellos', 'ellas',
        'el', 'ella', 'uno', 'una', 'vida', 'emocion', 'sentimiento', 'alma',
        'algo', 'alguien', 'nadie'
    }))
    excluded_lemmas = set(remove_accents(w.lower()) for w in (excluded_lemmas or {
        'ser', 'estar', 'haber', 'tener', 'hacer', 'poder', 'decir', 'ir', 'ver', 'dar',
        'saber', 'querer', 'llegar', 'pasar', 'deber', 'poner', 'parecer', 'quedar',
        'creer', 'llevar', 'dejar', 'seguir', 'encontrar', 'llamar', 'venir', 'volver',
        'ademas'
    }))
    excluded_pos = set(excluded_pos or {'PRON', 'DET', 'ADP', 'CCONJ', 'SCONJ', 'PART', 'INTJ'})

    discarded = []  # Optional for debug

    # Normalize case (accents preserved for now)
    text = text.lower()
    text = re.sub(r'[^a-záéíóúüñ\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    # Tokenize
    doc = nlp(text)

    clean_tokens = []

    for token in doc:
        word = token.text
        lemma = token.lemma_

        # Preserve word as-is?
        if word in preserve_words:
            norm_word = remove_accents(word.lower())
            clean_tokens.append(norm_word)
            continue

        norm_lemma = remove_accents(lemma.lower())
        norm_word = remove_accents(word.lower())

        if (
            token.is_alpha and
            norm_lemma not in stop_words and
            norm_lemma not in custom_excluded and
            norm_lemma not in excluded_lemmas and
            token.pos_ not in excluded_pos and
            ' ' not in norm_lemma and
            len(norm_lemma.split()) == 1
        ):
            clean_tokens.append(norm_lemma)
        else:
            if return_debug:
                discarded.append((word, lemma, token.pos_))

    return (clean_tokens, discarded) if return_debug else clean_tokens
