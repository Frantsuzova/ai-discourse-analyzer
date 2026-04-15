from __future__ import annotations

import argparse
from pathlib import Path

from .config import AnalysisConfig
from .pipeline import DiscourseAnalyzer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build an AI discourse analysis report from JSONL/CSV/TSV data.")
    parser.add_argument("input_path", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("report_output"))
    parser.add_argument("--format", choices=["jsonl", "csv", "tsv"], default="jsonl")
    parser.add_argument("--clusters", type=int, default=7)
    parser.add_argument("--no-rubert", action="store_true")
    parser.add_argument("--no-pacmap", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    config = AnalysisConfig(
        input_format=args.format,
        n_clusters=args.clusters,
        use_rubert=not args.no_rubert,
        use_pacmap=not args.no_pacmap,
    )
    analyzer = DiscourseAnalyzer(config)
    result = analyzer.run(args.input_path, args.output_dir)
    print(result.report_path)


if __name__ == "__main__":
    main()
