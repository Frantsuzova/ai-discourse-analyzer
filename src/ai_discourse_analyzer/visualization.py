from __future__ import annotations

from collections import Counter

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

from .emotions import AFFECTIVE_LABELS, EMOTION_LABELS, compute_emotion_profiles


def build_cluster_map(plot_df: pd.DataFrame) -> str:
    fig = px.scatter(
        plot_df,
        x="x",
        y="y",
        color="legend",
        custom_data=["legend", "date", "channel", "emotion", "text_short"],
        opacity=0.78,
        title="Карта кластеров",
    )
    fig.update_traces(
        marker={"size": 5},
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Дата: %{customdata[1]}<br>"
            "Канал: %{customdata[2]}<br>"
            "Эмоция: %{customdata[3]}<br><br>"
            "%{customdata[4]}<extra></extra>"
        )
    )

    centers_plot = plot_df.groupby(["cluster", "cluster_name"], as_index=False)[["x", "y"]].median()
    annotations = []
    for _, row in centers_plot.iterrows():
        short_name = row["cluster_name"]
        if len(short_name) > 28:
            short_name = short_name[:28] + "…"
        annotations.append(
            dict(
                x=row["x"], y=row["y"],
                text=f"кл.{int(row['cluster'])}, {short_name}",
                showarrow=False,
                font=dict(size=11),
                bgcolor="rgba(255,255,255,0.7)",
            )
        )

    fig.update_layout(
        template="plotly_white",
        height=760,
        margin={"l": 20, "r": 20, "t": 60, "b": 20},
        legend_title_text="",
        annotations=annotations,
    )
    return pio.to_html(fig, include_plotlyjs="cdn", full_html=False, div_id="cluster_plot")


def build_affective_radar(summary_df: pd.DataFrame, doc_emotions_df: pd.DataFrame) -> str:
    cluster_label_map = {}
    for _, row in summary_df.iterrows():
        short_name = row["cluster_name"]
        if len(short_name) > 24:
            short_name = short_name[:24] + "…"
        cluster_label_map[row["cluster"]] = f"кл.{int(row['cluster'])}, {short_name}"

    focused_rows = []
    for cluster_id, g in doc_emotions_df.groupby("cluster"):
        counter = Counter(g["emotion"])
        _, focused = compute_emotion_profiles(counter, len(g))
        for label in AFFECTIVE_LABELS:
            focused_rows.append({
                "cluster_label": cluster_label_map[cluster_id],
                "emotion": label,
                "value": focused[label],
            })

    focused_df = pd.DataFrame(focused_rows)
    cluster_order = [cluster_label_map[c] for c in summary_df["cluster"].tolist()]

    fig = go.Figure()
    for emotion in AFFECTIVE_LABELS:
        sub = focused_df[focused_df["emotion"] == emotion].copy()
        sub = sub.set_index("cluster_label").loc[cluster_order].reset_index()
        theta = sub["cluster_label"].tolist()
        r = sub["value"].tolist()
        theta.append(theta[0])
        r.append(r[0])
        fig.add_trace(go.Scatterpolar(r=r, theta=theta, fill="toself", name=emotion, opacity=0.22))

    fig.update_layout(
        title="Распределение эмоционально маркированных публикаций по кластерам",
        template="plotly_white",
        height=640,
        margin={"l": 20, "r": 20, "t": 60, "b": 20},
        polar=dict(radialaxis=dict(visible=True, ticksuffix="%")),
        legend_title_text="",
    )
    return pio.to_html(fig, include_plotlyjs=False, full_html=False, div_id="emotion_radar")
