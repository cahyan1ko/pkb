import spacy
from collections import Counter
import re

nlp = spacy.load('en_core_web_sm')

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text.lower()

def analyze_text(text):
    text = clean_text(text)

    doc = nlp(text)

    entities = [ent.text for ent in doc.ents]
    words = [token.text for token in doc if not token.is_stop and not token.is_punct]

    word_freq = Counter(words)

    return {
        'entities': entities,
        'keywords': word_freq.most_common(5)
    }
