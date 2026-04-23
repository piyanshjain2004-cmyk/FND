import re

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


lemmatizer = WordNetLemmatizer()
DEFAULT_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has",
    "he", "in", "is", "it", "its", "of", "on", "that", "the", "to", "was",
    "were", "will", "with",
}

try:
    stop_words = set(stopwords.words("english"))
except LookupError:
    stop_words = DEFAULT_STOPWORDS


def safe_tokenize(text):
    try:
        return nltk.word_tokenize(text)
    except LookupError:
        return text.split()


def safe_lemmatize(word):
    try:
        return lemmatizer.lemmatize(word)
    except LookupError:
        return word


def preprocess(text):
    text = re.sub(r"[^a-zA-Z\s]", "", str(text))
    text = text.lower()
    words = safe_tokenize(text)
    words = [safe_lemmatize(word) for word in words if word and word not in stop_words]
    return " ".join(words)
