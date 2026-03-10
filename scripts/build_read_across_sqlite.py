from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from core.read_across_sqlite import build_read_across_sqlite


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build SQLite storage for ChemReport read-across datasets.")
    parser.add_argument("--db", required=True, help="Output SQLite database path.")
    parser.add_argument("--category-csv", required=True, help="CSV with category labels.")
    parser.add_argument("--logp-csv", help="CSV for LogP analogues.")
    parser.add_argument("--pesticide-csv", help="CSV for pesticide class analogues.")
    parser.add_argument("--toxicity-csv", help="CSV for toxicity analogues.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = build_read_across_sqlite(
        Path(args.db),
        category_csv=Path(args.category_csv),
        logp_csv=Path(args.logp_csv) if args.logp_csv else None,
        pesticide_csv=Path(args.pesticide_csv) if args.pesticide_csv else None,
        toxicity_csv=Path(args.toxicity_csv) if args.toxicity_csv else None,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
