# ai-discourse-analyzer

`ai-discourse-analyzer` is a standalone Python package for corpus-level analysis of AI-related discourse in Telegram and other social-media-like datasets.

It supports:
- weighted combination of post text and aggregated comments
- Russian lemmatization with `pymorphy3`
- phrase extraction with `gensim.Phrases`
- TF-IDF + SVD + KMeans clustering
- LDA subtopics within clusters
- hybrid emotion detection using lexicons, discourse features, optional RuBERT, and Word2Vec
- interactive HTML reporting

The package can be used:
1. **standalone**, on raw JSONL/CSV-like corpora;
2. **as a second-stage analysis layer** after `corpus_cluster_explorer`.

## Installation

Base install:

```bash
pip install ai-discourse-analyzer
```

Full install with optional RuBERT and PaCMAP:

```bash
pip install "ai-discourse-analyzer[all]"
```

## Expected JSONL schema

The default loader expects records with the following fields:
- `text`
- `comments_text`
- `date_utc`
- `channel_username` or `channel_id`
- `link`

Only `text` is strictly required. If `comments_text` is absent, the package still works.

## Minimal example

```python
from pathlib import Path
from ai_discourse_analyzer import AnalysisConfig, DiscourseAnalyzer

config = AnalysisConfig()
analyzer = DiscourseAnalyzer(config)
result = analyzer.run(
    input_path=Path("social_data_ai_raw_2026.jsonl"),
    output_dir=Path("report_output")
)
print(result.report_path)
```

## CLI

```bash
aida-report social_data_ai_raw_2026.jsonl --output-dir report_output
```

## Integration with corpus_cluster_explorer

The package can also be used as an advanced layer after `corpus_cluster_explorer`.

Suggested workflow:
1. run baseline preprocessing and clustering with `corpus_cluster_explorer`;
2. export tokenized or clustered corpus;
3. pass the exported file to `ai-discourse-analyzer` for:
   - refined cluster naming,
   - LDA subtopics,
   - hybrid emotional profiling,
   - final HTML report.

## Report design

The default report intentionally stays compact. It includes:
- corpus metadata;
- cluster summary table;
- cluster map;
- one emotion visualization: **distribution of affectively marked texts by clusters**.

The table uses a clear emotion structure that sums to 100%:
- neutral share;
- undefined emotional register share;
- affectively marked share.

Affective class composition is then shown separately in the radar chart.
