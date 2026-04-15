from __future__ import annotations

import re
from functools import lru_cache

import pymorphy3

from .config import AnalysisConfig
from .vocab import LATIN_KEEP, LATIN_MAP, AI_SINGLE, AI_SUBSTRINGS, STOPWORDS, TOKEN_RE

TOKEN_PATTERN = re.compile(TOKEN_RE, re.I)
MORPH = pymorphy3.MorphAnalyzer()


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\u200b", " ").replace("\xa0", " ")
    text = re.sub(r"```.*?```", " ", text, flags=re.S)
    text = re.sub(r"`[^`]+`", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"file:///\S+", " ", text)
    text = re.sub(r"[█▉▊▋▍▎▏■□▪▫◆◇▲△▼▽◼◻]+", " ", text)
    text = re.sub(r"\b\d{1,2}(?:am|pm)\b", " ", text, flags=re.I)
    text = re.sub(r"[#@]\w+", " ", text)
    text = re.sub(r"\+\d+\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


@lru_cache(maxsize=500_000)
def normalize_token(token: str) -> str | None:
    token = token.lower().replace("ё", "е")
    if token.isdigit() or len(token) < 3:
        return None

    if re.fullmatch(r"[a-z][a-z0-9_\-]*", token):
        token = LATIN_MAP.get(token, token)
        return token if token in LATIN_KEEP else None

    lemma = MORPH.parse(token)[0].normal_form
    if len(lemma) < 3 or lemma in STOPWORDS:
        return None
    return lemma


def tokenize_and_normalize(text: str) -> list[str]:
    tokens = TOKEN_PATTERN.findall(text.lower())
    norm = [normalize_token(t) for t in tokens]
    return [t for t in norm if t]


def is_ai_relevant(text: str) -> bool:
    low = text.lower()
    tokens = TOKEN_PATTERN.findall(low)

    score = 0
    strong = 0
    for t in tokens:
        if t in AI_SINGLE:
            score += 1
            if t in LATIN_KEEP or t in {"ии", "нейросеть", "нейронка", "промпт"}:
                strong += 1

    for s in AI_SUBSTRINGS:
        if s in low:
            score += 1

    return strong >= 1 or score >= 2


def build_weighted_text(post_text: str | None, comments_text: str | None, post_weight: int) -> str:
    post = post_text or ""
    comments = comments_text or ""
    weighted_text = ((post + " ") * post_weight) + comments
    return clean_text(weighted_text.strip())
