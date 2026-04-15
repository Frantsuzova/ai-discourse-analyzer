from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections import Counter

import pandas as pd
from gensim.models import Word2Vec
from gensim.models.phrases import Phrases, Phraser

from .config import AnalysisConfig
from .io import load_dataframe, ensure_output_dir
from .preprocessing import build_weighted_text, is_ai_relevant, tokenize_and_normalize
from .clustering import (
    build_tfidf_matrix,
    reduce_and_cluster,
    top_terms_for_cluster,
    cluster_display_name,
    lda_topics_for_cluster,
    sample_for_plot,
    reduce_for_plot,
)
from .emotions import (
    EmotionResources,
    build_emotion_w2v_expansion,
    build_rubert_resources,
    classify_emotion,
    short_hover_text,
    format_emotion_structure,
)
from .visualization import build_cluster_map, build_affective_radar
from .report import render_html_report


@dataclass(slots=True)
class AnalysisResult:
    report_path: Path
    summary_path: Path
    topics_path: Path
    emotions_path: Path
    points_path: Path


class DiscourseAnalyzer:
    def __init__(self, config: AnalysisConfig | None = None):
        self.config = config or AnalysisConfig()

    def _prepare_records(self, df: pd.DataFrame) -> tuple[list[dict], int]:
        records: list[dict] = []
        total_posts = len(df)
        for _, row in df.iterrows():
            post_text = row.get("text", "")
            comments_text = row.get("comments_text", "")
            combined = build_weighted_text(post_text, comments_text, self.config.post_weight)
            if not combined or not is_ai_relevant(combined):
                continue
            tokens = tokenize_and_normalize(combined)
            if len(tokens) < 6:
                continue
            records.append({
                "channel": row.get("channel_username") or str(row.get("channel_id") or ""),
                "date": str(row.get("date_utc") or "")[:10],
                "link": row.get("link") or "",
                "text_raw": combined,
                "tokens": tokens,
            })
        return records, total_posts

    def _add_bigrams(self, records: list[dict]) -> None:
        phrases = Phrases([r["tokens"] for r in records], min_count=18, threshold=10.0, delimiter="_")
        phraser = Phraser(phrases)
        for r in records:
            r["all_tokens"] = list(phraser[r["tokens"]])

    def _train_w2v(self, records: list[dict]) -> Word2Vec:
        return Word2Vec(
            sentences=[r["all_tokens"] for r in records],
            vector_size=self.config.w2v_vector_size,
            window=self.config.w2v_window,
            min_count=self.config.w2v_min_count,
            workers=4,
            sg=1,
            epochs=self.config.w2v_epochs,
            seed=self.config.random_state,
        )

    def _build_emotion_resources(self, w2v_model: Word2Vec) -> EmotionResources:
        sentence_model = None
        label_proto = {}
        if self.config.use_rubert:
            try:
                sentence_model, label_proto = build_rubert_resources(self.config.rubert_model_name)
            except Exception:
                sentence_model, label_proto = None, {}
        w2v_expanded = build_emotion_w2v_expansion(w2v_model)
        return EmotionResources(sentence_model=sentence_model, label_prototypes=label_proto, w2v_expanded=w2v_expanded)

    def run(self, input_path: Path, output_dir: Path) -> AnalysisResult:
        output_dir = ensure_output_dir(output_dir)
        df = load_dataframe(input_path, self.config.input_format)
        records, total_posts = self._prepare_records(df)
        if not records:
            raise RuntimeError("После очистки и фильтрации не осталось релевантных публикаций.")

        self._add_bigrams(records)
        total_norm_tokens = sum(len(r["all_tokens"]) for r in records)

        w2v_model = self._train_w2v(records)
        emotion_resources = self._build_emotion_resources(w2v_model)

        texts = [" ".join(r["all_tokens"]) for r in records]
        X, terms = build_tfidf_matrix(texts, self.config)
        X_dense, labels, centers = reduce_and_cluster(X, self.config)

        cluster_names = {}
        for c in range(self.config.n_clusters):
            top_tokens, top_bigrams = top_terms_for_cluster(X, terms, labels, c, topn=10)
            cluster_names[c] = cluster_display_name(top_tokens, top_bigrams, self.config.cluster_name_stopwords)

        emotion_rows = []
        doc_emotions = []
        for i, rec in enumerate(records):
            cluster_id = int(labels[i])
            emotion, _ = classify_emotion(rec["text_raw"], rec["all_tokens"], cluster_names[cluster_id], emotion_resources, self.config)
            doc_emotions.append(emotion)
            emotion_rows.append({
                "doc_id": i,
                "cluster": cluster_id,
                "cluster_name": cluster_names[cluster_id],
                "date": rec["date"],
                "channel": rec["channel"],
                "emotion": emotion,
                "link": rec["link"],
            })
        doc_emotions_df = pd.DataFrame(emotion_rows)

        summary_rows = []
        topic_rows = []
        for c in range(self.config.n_clusters):
            idx = [i for i, lbl in enumerate(labels) if lbl == c]
            top_tokens, top_bigrams = top_terms_for_cluster(X, terms, labels, c, topn=10)
            cluster_texts = [texts[i] for i in idx]
            topics = lda_topics_for_cluster(cluster_texts, self.config)
            emotion_counter = Counter(doc_emotions_df.loc[doc_emotions_df["cluster"] == c, "emotion"])
            summary_rows.append({
                "cluster": c,
                "cluster_name": cluster_names[c],
                "size": len(idx),
                "top_tokens": ", ".join(top_tokens),
                "top_bigrams": ", ".join(top_bigrams),
                "lda_topics": "<br>".join([f"{i+1}) " + ", ".join(t) for i, t in enumerate(topics)]),
                "emotion_structure": format_emotion_structure(emotion_counter, len(idx)),
            })
            for i, t in enumerate(topics, start=1):
                topic_rows.append({"cluster": c, "cluster_name": cluster_names[c], "topic_id": i, "topic_terms": ", ".join(t)})

        summary_df = pd.DataFrame(summary_rows).sort_values("size", ascending=False)
        topics_df = pd.DataFrame(topic_rows).sort_values(["cluster", "topic_id"])

        sampled = sample_for_plot(X_dense, labels, centers, self.config)
        coords, _ = reduce_for_plot(X_dense[sampled], self.config)
        plot_rows = []
        for j, i in enumerate(sampled):
            plot_rows.append({
                "x": float(coords[j, 0]),
                "y": float(coords[j, 1]),
                "cluster": int(labels[i]),
                "cluster_name": cluster_names[int(labels[i])],
                "channel": records[i]["channel"],
                "date": records[i]["date"],
                "text_short": short_hover_text(records[i]["text_raw"]),
                "emotion": doc_emotions[i],
            })
        plot_df = pd.DataFrame(plot_rows)
        plot_df["legend"] = plot_df.apply(lambda r: f"Кластер {int(r['cluster'])}: {r['cluster_name']}", axis=1)

        summary_path = output_dir / "cluster_summary.csv"
        topics_path = output_dir / "cluster_lda_topics.csv"
        emotions_path = output_dir / "document_emotions.csv"
        points_path = output_dir / "cluster_points.csv"
        report_path = output_dir / "report.html"

        summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
        topics_df.to_csv(topics_path, index=False, encoding="utf-8-sig")
        doc_emotions_df.to_csv(emotions_path, index=False, encoding="utf-8-sig")
        plot_df.to_csv(points_path, index=False, encoding="utf-8-sig")

        cluster_map_html = build_cluster_map(plot_df)
        radar_html = build_affective_radar(summary_df, doc_emotions_df)
        render_html_report(
            report_path=report_path,
            title=self.config.report_title,
            total_posts=total_posts,
            total_norm_tokens=total_norm_tokens,
            summary_df=summary_df,
            cluster_map_html=cluster_map_html,
            radar_html=radar_html,
        )

        return AnalysisResult(
            report_path=report_path,
            summary_path=summary_path,
            topics_path=topics_path,
            emotions_path=emotions_path,
            points_path=points_path,
        )
