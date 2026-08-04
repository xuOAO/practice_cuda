from __future__ import annotations

import argparse
import json
from pathlib import Path

from fp8_bench.reporting import default_columns, format_frame, records_frame


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Display benchmark JSONL results as a pandas table."
    )
    parser.add_argument("path", type=Path, help="Benchmark JSONL file.")
    parser.add_argument("--kind", help="Keep only one record kind.")
    parser.add_argument(
        "--column",
        action="append",
        help="Column to display; repeat to select and order several columns.",
    )
    parser.add_argument("--sort", help="Sort by one displayed DataFrame column.")
    parser.add_argument("--descending", action="store_true")
    parser.add_argument("--csv", type=Path, help="Also write the selected table to CSV.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    with args.path.open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    if args.kind is not None:
        records = [record for record in records if record.get("kind") == args.kind]

    frame = records_frame(records)
    if frame.empty:
        raise ValueError("no matching records")
    if args.sort is not None:
        if args.sort not in frame.columns:
            raise ValueError(f"unknown sort column: {args.sort}")
        frame = frame.sort_values(args.sort, ascending=not args.descending)

    columns = args.column or default_columns(frame)
    print(format_frame(frame, columns=columns))
    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        frame.loc[:, columns].to_csv(args.csv, index=False)


if __name__ == "__main__":
    main()
