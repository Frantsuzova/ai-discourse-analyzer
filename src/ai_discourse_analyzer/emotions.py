from __future__ import annotations

import re
import textwrap
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
from gensim.models import Word2Vec

AFFECTIVE_LABELS = [
    "Удивление, интерес",
    "Радость, энтузиазм",
    "Страх",
    "Злость",
    "Грусть",
    "Отвращение",
]

NON_AFFECTIVE_LABELS = [
    "Нейтральные эмоции",
    "Неопределённый эмоциональный регистр",
]

EMOTION_LABELS = AFFECTIVE_LABELS + NON_AFFECTIVE_LABELS

EMOTION_LEXICON = {
    "Удивление, интерес": {
        "lemma": {
            "удивить", "удивление", "неожиданно", "внезапно", "вдруг", "вау", "ого",
            "интересно", "любопытно", "любопытство", "забавно", "поразить", "шокировать",
            "невероятно", "впечатлить", "впечатление", "внимание", "заметить", "поражать",
            "любопытный", "удивительный"
        },
        "bigram": {
            "ничто_себе", "вот_это", "очень_интересно", "крайне_интересно",
            "неожиданно_оказаться", "вызвать_интерес"
        }
    },
    "Радость, энтузиазм": {
        "lemma": {
            "радость", "рад", "круто", "классно", "супер", "обожать", "любить",
            "восторг", "вдохновение", "вдохновлять", "энтузиазм", "отлично",
            "прекрасно", "кайф", "счастье", "ура", "потрясающе", "шикарно",
            "нравиться", "восхищать", "здорово", "мощно", "топ", "огонь"
        },
        "bigram": {
            "очень_круто", "просто_супер", "очень_нравиться", "вызвать_восторг",
            "полный_восторг", "безумно_круто"
        }
    },
    "Страх": {
        "lemma": {
            "страх", "бояться", "опасаться", "тревога", "тревожный", "паника",
            "угроза", "риск", "страшно", "ужасать", "кошмар", "пугать",
            "небезопасно", "катастрофа", "потерять", "заменить", "уволить",
            "запретить", "заблокировать", "конец", "тревожно", "опасный"
        },
        "bigram": {
            "очень_страшно", "становиться_страшно", "реальный_угроза",
            "вызывать_тревога", "потерять_работа", "остаться_без"
        }
    },
    "Злость": {
        "lemma": {
            "злость", "злой", "бесить", "раздражать", "беситься", "ненавидеть",
            "ярость", "агрессия", "агрессивный", "возмущать", "достать",
            "бред", "идиотизм", "тупой", "мерзкий", "взбесить", "мудак",
            "выбесить", "дерьмо", "хрень"
        },
        "bigram": {
            "полный_бред", "это_бесить", "дико_раздражать", "ужасно_бесить",
            "что_за", "просто_бесить"
        }
    },
    "Грусть": {
        "lemma": {
            "грусть", "грустно", "печаль", "печально", "тоскливо", "жаль", "жалко",
            "обидно", "разочарование", "разочаровать", "уныние", "усталость",
            "выгорание", "потеря", "тяжело", "больно", "сложно", "сломаться",
            "грустный", "обидеть"
        },
        "bigram": {
            "очень_жаль", "очень_грустно", "становиться_жалко",
            "просто_печально", "довести_до_слеза"
        }
    },
    "Нейтральные эмоции": {
        "lemma": {
            "обзор", "исследование", "анализ", "модель", "данные", "система",
            "метод", "подход", "результат", "оценка", "разработка", "компания",
            "технология", "инструмент", "платформа", "обучение", "архитектура",
            "release", "update", "benchmark", "dataset"
        },
        "bigram": {
            "искусственный_интеллект", "языковой_модель", "машинный_обучение",
            "нейронный_сеть", "генерация_изображение", "рабочий_процесс",
            "открытый_исходный", "набор_данные"
        }
    },
    "Отвращение": {
        "lemma": {
            "отвращение", "омерзение", "мерзость", "мерзкий", "противно",
            "тошнить", "тошно", "гадость", "отстой", "фу", "мерзко",
            "омерзительно", "омерзительный", "склизкий", "гнилой"
        },
        "bigram": {
            "это_противно", "просто_мерзость", "реально_тошно", "какая_гадость"
        }
    }
}

INTENSIFIERS = {"очень", "крайне", "безумно", "дико", "сильно", "невероятно", "максимально", "совсем"}
INTERJECTIONS = {"вау", "ого", "ух", "ох", "эх", "увы", "блин", "жесть", "капец", "пипец", "черт", "фу"}
NEUTRALIZERS = {"обзор", "исследование", "анализ", "сравнение", "подборка", "дайджест", "рассмотреть", "обсудить", "метод", "результат", "оценка", "система", "подход"}

EMOTION_PROTOTYPES = {
    "Удивление, интерес": [
        "Текст выражает интерес, удивление, внимание к новизне и любопытство.",
        "Высказывание оформлено как удивление и заинтересованность."
    ],
    "Радость, энтузиазм": [
        "Текст выражает радость, воодушевление, восторг и энтузиазм.",
        "Высказывание эмоционально положительное и вдохновленное."
    ],
    "Страх": [
        "Текст выражает страх, тревогу, угрозу и опасение.",
        "Высказывание оформлено как беспокойство и тревожное ожидание."
    ],
    "Злость": [
        "Текст выражает раздражение, злость, возмущение и агрессию.",
        "Высказывание оформлено как негативная реакция и гнев."
    ],
    "Грусть": [
        "Текст выражает грусть, печаль, разочарование и эмоциональную усталость.",
        "Высказывание оформлено как печальное и подавленное."
    ],
    "Нейтральные эмоции": [
        "Текст нейтральный, описательный, аналитический и информационный.",
        "Высказывание не содержит выраженного эмоционального сигнала."
    ],
    "Отвращение": [
        "Текст выражает отвращение, омерзение, неприятие и сильное отталкивание.",
        "Высказывание оформлено как эмоциональное неприятие."
    ],
}

EMOTION_SEED_WORDS = {
    "Удивление, интерес": ["удивление", "интересно", "любопытство", "неожиданно", "вау"],
    "Радость, энтузиазм": ["радость", "восторг", "круто", "обожать", "супер"],
    "Страх": ["страх", "тревога", "опасность", "угроза", "страшно"],
    "Злость": ["злость", "раздражение", "бесить", "ярость", "идиотизм"],
    "Грусть": ["грусть", "печаль", "жаль", "разочарование", "тоскливо"],
    "Нейтральные эмоции": ["обзор", "анализ", "метод", "система", "результат"],
    "Отвращение": ["отвращение", "мерзость", "противно", "гадость", "тошно"],
}


@dataclass(slots=True)
class EmotionResources:
    sentence_model: Any | None
    label_prototypes: dict[str, np.ndarray]
    w2v_expanded: dict[str, dict[str, float]]


def build_local_bigrams(tokens: list[str]) -> list[str]:
    return [f"{tokens[i]}_{tokens[i+1]}" for i in range(len(tokens) - 1)]


def count_caps_ratio(text: str) -> float:
    letters = re.findall(r"[A-Za-zА-ЯЁа-яё]", text)
    if not letters:
        return 0.0
    upper = sum(1 for ch in letters if ch.isupper())
    return upper / len(letters)


def short_hover_text(text: str, width: int = 52, max_chars: int = 120) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = text[:max_chars]
    return "<br>".join(textwrap.wrap(text, width=width))


def normalize_score_dict(scores: dict[str, float], labels: list[str]) -> dict[str, float]:
    vals = np.array([max(0.0, scores.get(lbl, 0.0)) for lbl in labels], dtype=float)
    if vals.sum() == 0:
        return {lbl: 0.0 for lbl in labels}
    vals = vals / vals.sum()
    return {lbl: float(v) for lbl, v in zip(labels, vals)}


def build_emotion_w2v_expansion(w2v_model: Word2Vec, topn: int = 12) -> dict[str, dict[str, float]]:
    expanded = {}
    for label, seeds in EMOTION_SEED_WORDS.items():
        weights: dict[str, float] = {}
        for seed in seeds:
            if seed in w2v_model.wv:
                weights[seed] = max(weights.get(seed, 0.0), 1.0)
                for neighbor, sim in w2v_model.wv.most_similar(seed, topn=topn):
                    if sim >= 0.45:
                        weights[neighbor] = max(weights.get(neighbor, 0.0), float(sim))
        expanded[label] = weights
    return expanded


def build_rubert_resources(model_name: str):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    proto_texts = []
    proto_map = []
    for label, texts in EMOTION_PROTOTYPES.items():
        for t in texts:
            proto_texts.append(t)
            proto_map.append(label)

    proto_emb = model.encode(proto_texts, normalize_embeddings=True, show_progress_bar=False)
    grouped: dict[str, list[np.ndarray]] = defaultdict(list)
    for label, emb in zip(proto_map, proto_emb):
        grouped[label].append(emb)

    label_proto = {label: np.mean(np.vstack(embs), axis=0) for label, embs in grouped.items()}
    return model, label_proto


def rubert_emotion_scores(text: str, resources: EmotionResources) -> dict[str, float]:
    if resources.sentence_model is None:
        return {label: 1 / len(EMOTION_PROTOTYPES) for label in EMOTION_PROTOTYPES.keys()}

    emb = resources.sentence_model.encode([text], normalize_embeddings=True, show_progress_bar=False)[0]
    scores = {}
    for label in EMOTION_PROTOTYPES.keys():
        sim = float(np.dot(emb, resources.label_prototypes[label]))
        sim = (sim + 1.0) / 2.0
        scores[label] = sim
    return scores


def compute_emotion_profiles(counter: Counter, total: int):
    raw = {label: (0.0 if total == 0 else counter.get(label, 0) * 100.0 / total) for label in EMOTION_LABELS}
    affective_total = sum(counter.get(label, 0) for label in AFFECTIVE_LABELS)

    focused = {}
    if affective_total == 0:
        for label in AFFECTIVE_LABELS:
            focused[label] = 0.0
    else:
        for label in AFFECTIVE_LABELS:
            focused[label] = counter.get(label, 0) * 100.0 / affective_total
    return raw, focused


def format_emotion_structure(counter: Counter, total: int) -> str:
    raw, _ = compute_emotion_profiles(counter, total)
    affective_share = sum(raw[label] for label in AFFECTIVE_LABELS)
    return "<br>".join([
        f"Нейтральные эмоции: {raw['Нейтральные эмоции']:.1f}%",
        f"Неопределённый эмоциональный регистр: {raw['Неопределённый эмоциональный регистр']:.1f}%",
        f"Эмоционально маркированные публикации: {affective_share:.1f}%",
    ])


def classify_emotion(text_raw: str, tokens: list[str], cluster_name: str, resources: EmotionResources, config) -> tuple[str, dict]:
    content_labels = [x for x in EMOTION_LABELS if x != "Неопределённый эмоциональный регистр"]

    bigrams = build_local_bigrams(tokens)
    token_set = set(tokens)
    bigram_set = set(bigrams)

    lexicon_scores = {label: 0.0 for label in content_labels}
    discourse_scores = {label: 0.0 for label in content_labels}
    w2v_scores = {label: 0.0 for label in content_labels}

    for label in content_labels:
        lex = EMOTION_LEXICON[label]
        lemma_hits = len(token_set.intersection(lex["lemma"]))
        bigram_hits = len(bigram_set.intersection(lex["bigram"]))
        lexicon_scores[label] += lemma_hits * 1.0 + bigram_hits * 1.5

    intens_count = sum(1 for t in tokens if t in INTENSIFIERS)
    interj_count = sum(1 for t in tokens if t in INTERJECTIONS)
    exclam_count = text_raw.count("!")
    question_count = text_raw.count("?")
    caps_ratio = count_caps_ratio(text_raw)
    neutralizer_hits = sum(1 for t in tokens if t in NEUTRALIZERS)

    if exclam_count >= 2:
        discourse_scores["Радость, энтузиазм"] += 1.0
        discourse_scores["Злость"] += 0.8
        discourse_scores["Удивление, интерес"] += 0.8
    if question_count >= 2:
        discourse_scores["Удивление, интерес"] += 0.8
    if interj_count > 0:
        discourse_scores["Удивление, интерес"] += 0.5 * interj_count
        discourse_scores["Радость, энтузиазм"] += 0.3 * interj_count
        discourse_scores["Злость"] += 0.3 * interj_count
        discourse_scores["Отвращение"] += 0.25 * interj_count
    if intens_count > 0:
        discourse_scores["Радость, энтузиазм"] += 0.35 * intens_count
        discourse_scores["Злость"] += 0.35 * intens_count
        discourse_scores["Страх"] += 0.25 * intens_count
        discourse_scores["Грусть"] += 0.2 * intens_count
        discourse_scores["Отвращение"] += 0.25 * intens_count
    if caps_ratio > 0.18:
        discourse_scores["Злость"] += 0.7
        discourse_scores["Радость, энтузиазм"] += 0.5
        discourse_scores["Удивление, интерес"] += 0.4
    if neutralizer_hits >= 2:
        discourse_scores["Нейтральные эмоции"] += 1.4

    cl = cluster_name.lower()
    if "midjourney" in cl or "изображение" in cl or "промпт" in cl:
        discourse_scores["Удивление, интерес"] += 0.2
        discourse_scores["Радость, энтузиазм"] += 0.2
    if "обучение" in cl or "llm" in cl or "модель" in cl:
        discourse_scores["Нейтральные эмоции"] += 0.3
    if "компания" in cl or "рынок" in cl or "openai" in cl:
        discourse_scores["Нейтральные эмоции"] += 0.2
        discourse_scores["Страх"] += 0.1

    rubert_scores = rubert_emotion_scores(text_raw, resources)

    freq = Counter(tokens)
    for label in content_labels:
        expanded = resources.w2v_expanded.get(label, {})
        s = 0.0
        for tok, cnt in freq.items():
            if tok in expanded:
                s += expanded[tok] * cnt
        w2v_scores[label] = s

    lexicon_scores_n = normalize_score_dict(lexicon_scores, content_labels)
    discourse_scores_n = normalize_score_dict(discourse_scores, content_labels)
    rubert_scores_n = normalize_score_dict(rubert_scores, content_labels)
    w2v_scores_n = normalize_score_dict(w2v_scores, content_labels)

    final_scores = {}
    for label in content_labels:
        final_scores[label] = (
            config.weight_lexicon * lexicon_scores_n[label]
            + config.weight_rubert * rubert_scores_n[label]
            + config.weight_w2v * w2v_scores_n[label]
            + config.weight_discourse * discourse_scores_n[label]
        )

    ordered = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
    best_label, best_score = ordered[0]
    second_label, second_score = ordered[1]
    neutral_score = final_scores["Нейтральные эмоции"]
    affective_max = max(final_scores[label] for label in AFFECTIVE_LABELS)

    if best_label == "Нейтральные эмоции" and neutral_score >= second_score + config.neutral_margin and affective_max < (neutral_score - 0.01):
        label = best_label
    elif best_score < config.emotion_min_confidence:
        label = "Неопределённый эмоциональный регистр"
    elif abs(best_score - second_score) < config.emotion_margin:
        label = "Неопределённый эмоциональный регистр"
    else:
        label = best_label

    return label, {
        "final": final_scores,
        "lexicon": lexicon_scores_n,
        "rubert": rubert_scores_n,
        "w2v": w2v_scores_n,
        "discourse": discourse_scores_n,
    }
