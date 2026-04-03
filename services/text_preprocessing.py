import re

import nltk
from nltk import data
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


def ensure_nltk_resource(resource_path, download_name):
    try:
        data.find(resource_path)
    except LookupError:
        try:
            nltk.download(download_name, quiet=True)
        except Exception:
            pass


ensure_nltk_resource("tokenizers/punkt", "punkt")
ensure_nltk_resource("corpora/stopwords", "stopwords")
ensure_nltk_resource("corpora/wordnet", "wordnet")
ensure_nltk_resource("corpora/omw-1.4", "omw-1.4")

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))


def preprocess(text):
    text = re.sub(r"[^a-zA-Z\s]", "", str(text))
    text = text.lower()
    words = nltk.word_tokenize(text)
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words)
