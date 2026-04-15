from __future__ import annotations

import html
from pathlib import Path

import pandas as pd


def render_html_report(
    report_path: Path,
    title: str,
    total_posts: int,
    total_norm_tokens: int,
    summary_df: pd.DataFrame,
    cluster_map_html: str,
    radar_html: str,
):
    rows = []
    for _, row in summary_df.iterrows():
        rows.append(
            "<tr>"
            f"<td><b>Кл.{int(row['cluster'])}</b><br>{html.escape(str(row['cluster_name']))}</td>"
            f"<td>{int(row['size'])}</td>"
            f"<td>{html.escape(str(row['top_tokens']))}</td>"
            f"<td>{html.escape(str(row['top_bigrams']))}</td>"
            f"<td>{row['lda_topics']}</td>"
            f"<td>{row['emotion_structure']}</td>"
            "</tr>"
        )

    html_doc = f"""<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8"/>
<title>AIDA Report — {html.escape(title)}</title>
<style>
body {{
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  margin: 24px;
  line-height: 1.45;
  color: #222;
}}
h1, h2 {{
  line-height: 1.2;
}}
.meta {{
  background: #f6f8fa;
  border: 1px solid #e5e7eb;
  border-radius: 10px;
  padding: 12px 14px;
  margin: 12px 0 18px;
}}
table {{
  border-collapse: collapse;
  width: 100%;
  margin: 12px 0 24px;
}}
th {{
  background: #f4f6f8;
}}
td, th {{
  border: 1px solid #d8dee4;
  padding: 8px;
  text-align: left;
  vertical-align: top;
}}
.small {{
  color: #556;
  font-size: 14px;
}}
</style>
</head>
<body>

<h1>{html.escape(title)}</h1>

<div class="meta">
  <div><b>Корпус:</b> {total_posts:,} публикаций с агрегированными комментариями.</div>
  <div><b>Единица анализа:</b> текст поста с повышенным весом и агрегированный текст комментариев.</div>
  <div><b>Объём после очистки и нормализации:</b> {total_norm_tokens:,} токенов.</div>
  <div class="small">Report generated with <b>AIDA</b> (AI Discourse Analyzer).</div>
</div>

<h2>Сводка по кластерам</h2>
<table>
<tr>
  <th>Кластер</th>
  <th>Размер</th>
  <th>Топ-10 токенов</th>
  <th>Топ-10 биграмм</th>
  <th>Подтемы (LDA)</th>
  <th>Эмоциональная структура</th>
</tr>
{''.join(rows)}
</table>

<h2>Интерактивная карта</h2>
<p class="small">Наведите курсор, чтобы увидеть краткий фрагмент текста, дату, канал и эмоциональный класс публикации.</p>
{cluster_map_html}

<h2>Распределение эмоционально маркированных публикаций</h2>
{radar_html}

</body>
</html>
"""
    report_path.write_text(html_doc, encoding="utf-8")
