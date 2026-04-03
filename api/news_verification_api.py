import html
import os
import re
import xml.etree.ElementTree as ET
from urllib.parse import quote_plus

import requests

from services.text_preprocessing import preprocess

GOOGLE_NEWS_RSS_URL = os.getenv(
    "GOOGLE_NEWS_RSS_URL",
    "https://news.google.com/rss/search?q={query}",
)
POLICY_KEYWORDS = {
    "government", "ministry", "cabinet", "approved", "approval", "policy", "mandate",
    "launched", "launch", "nationwide", "official", "notification", "circular", "gazette",
    "scheme", "mission", "centre", "union", "press release"
}
HIGH_SIGNAL_TERMS = {
    "government", "ministry", "policy", "mandate", "nationwide", "official", "scheme",
    "mission", "cabinet", "notification", "gazette", "launch", "announced", "approves",
    "private", "sector", "india", "delhi"
}
TRUSTED_PUBLISHERS = {
    "associated press", "ap", "reuters", "bbc", "bbc news", "ndtv", "the hindu",
    "indian express", "times of india", "hindustan times", "firstpost", "news18",
    "the print", "deccan herald", "business standard", "livemint", "mint", "ani",
    "al jazeera", "the guardian", "cnn", "npr", "abc news", "cbs news", "nbc news",
    "washington post", "the new york times", "bloomberg", "financial times",
    "wall street journal", "republic world"
}
RAW_STOPWORDS = {
    "this", "that", "with", "from", "have", "been", "into", "their", "they", "them",
    "about", "after", "amid", "during", "under", "since", "began", "calls", "called",
    "noting", "which", "only", "also", "highlighted", "importance", "ensuring", "safe",
    "free", "through", "waterway", "waterways", "situation", "meeting", "summit"
}


def build_news_query(text, max_terms=10):
    terms = []
    raw_words = re.findall(r"[A-Za-z][A-Za-z'-]{2,}", str(text))
    prioritized_words = []
    for word in raw_words:
        cleaned = word.strip("'").lower()
        if len(cleaned) < 4 or cleaned in RAW_STOPWORDS:
            continue
        if word[:1].isupper() or cleaned in HIGH_SIGNAL_TERMS:
            prioritized_words.append(cleaned)

    for word in prioritized_words:
        if word not in terms:
            terms.append(word)
        if len(terms) >= max_terms:
            return " ".join(terms)

    processed = preprocess(text)
    for word in processed.split():
        if len(word) < 4:
            continue
        if word not in terms:
            terms.append(word)
        if len(terms) >= max_terms:
            break
    if not terms:
        return None
    return " ".join(terms)


def score_article_match(claim_tokens, article_text):
    article_tokens = set(preprocess(article_text).split())
    if not claim_tokens or not article_tokens:
        return 0.0
    overlap = claim_tokens.intersection(article_tokens)
    return len(overlap) / max(len(claim_tokens), 1)


def looks_like_official_claim(text):
    lowered = text.lower()
    return any(keyword in lowered for keyword in POLICY_KEYWORDS)


def normalize_source_name(source_name):
    return " ".join(source_name.lower().strip().split())


def infer_source_name(title):
    if " - " not in title:
        return ""
    return title.rsplit(" - ", 1)[-1].strip()


def is_trusted_source(source_name):
    normalized = normalize_source_name(source_name)
    if not normalized:
        return False
    return any(
        normalized == publisher or normalized.endswith(publisher)
        for publisher in TRUSTED_PUBLISHERS
    )


def fetch_related_news(text, timeout=8):
    query = build_news_query(text)
    if not query:
        return {"status": "unavailable", "reason": "No usable query terms.", "articles": []}

    url = GOOGLE_NEWS_RSS_URL.format(query=quote_plus(query))
    try:
        response = requests.get(url, timeout=timeout, headers={"User-Agent": "TruthLens/1.0"})
        response.raise_for_status()
    except requests.RequestException as exc:
        return {
            "status": "unavailable",
            "reason": f"Live verification request failed: {exc}",
            "articles": [],
            "query": query,
        }

    try:
        root = ET.fromstring(response.text)
    except ET.ParseError:
        return {
            "status": "unavailable",
            "reason": "Live verification response could not be parsed.",
            "articles": [],
            "query": query,
        }

    items = []
    claim_tokens = set(preprocess(text).split())
    for item in root.findall("./channel/item")[:8]:
        title = html.unescape(item.findtext("title", default="")).strip()
        description = html.unescape(item.findtext("description", default="")).strip()
        link = item.findtext("link", default="").strip()
        source_name = html.unescape(item.findtext("source", default="")).strip() or infer_source_name(title)
        trusted_source = is_trusted_source(source_name)
        combined_text = f"{title} {description}"
        overlap_score = score_article_match(claim_tokens, combined_text)
        score = overlap_score + (0.06 if trusted_source else 0.0)
        items.append({
            "title": title,
            "link": link,
            "score": round(score, 4),
            "overlap_score": round(overlap_score, 4),
            "source": source_name,
            "trusted_source": trusted_source,
        })

    strong_matches = [article for article in items if article["score"] >= 0.18]
    moderate_matches = [article for article in items if article["score"] >= 0.12]
    trusted_matches = [article for article in items if article["trusted_source"] and article["score"] >= 0.12]

    if len(trusted_matches) >= 2 or len(strong_matches) >= 2 or (strong_matches and strong_matches[0]["score"] >= 0.30):
        status = "supported"
    elif trusted_matches or moderate_matches:
        status = "mixed"
    else:
        status = "unsupported"

    return {
        "status": status,
        "reason": None,
        "query": query,
        "articles": items,
        "strong_match_count": len(strong_matches),
        "moderate_match_count": len(moderate_matches),
        "trusted_match_count": len(trusted_matches),
        "policy_like": looks_like_official_claim(text),
    }
