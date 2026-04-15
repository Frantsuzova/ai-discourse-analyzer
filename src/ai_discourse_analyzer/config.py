from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class AnalysisConfig:
    random_state: int = 42
    n_clusters: int = 7
    post_weight: int = 3

    tfidf_max_features: int = 7000
    tfidf_min_df: int = 12
    tfidf_max_df: float = 0.18
    svd_components: int = 60

    lda_topics_per_cluster: int = 3
    lda_words_per_topic: int = 7
    lda_min_cluster_size: int = 50

    plot_max_points: int = 5000
    plot_outlier_quantile: float = 0.03

    use_rubert: bool = True
    use_pacmap: bool = True
    rubert_model_name: str = "cointegrated/rubert-tiny2"

    w2v_vector_size: int = 100
    w2v_window: int = 5
    w2v_min_count: int = 5
    w2v_epochs: int = 12

    weight_lexicon: float = 0.35
    weight_rubert: float = 0.35
    weight_w2v: float = 0.20
    weight_discourse: float = 0.10

    emotion_min_confidence: float = 0.18
    emotion_margin: float = 0.025
    neutral_margin: float = 0.04

    input_format: str = "jsonl"
    report_title: str = "Кластеризация постов и комментариев об AI в Telegram"

    cluster_name_stopwords: set[str] = field(default_factory=lambda: {
        "что", "какой", "какая", "какие", "пост", "еще", "ещё", "это", "весь", "свой",
        "тот", "мочь", "дать", "читать", "почему", "понимать", "результат", "всё", "то",
        "код", "бот"
    })
