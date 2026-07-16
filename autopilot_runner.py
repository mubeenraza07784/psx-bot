from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from cwt_bot.advanced_scanners import parse_symbols
from cwt_bot.autopilot_manager import run_autopilot_cycle, persist_autopilot_outputs


def _load_optional_csv(path: str) -> pd.DataFrame | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    return pd.read_csv(p)


def main() -> None:
    parser = argparse.ArgumentParser(description="PSX autonomous research cycle runner")
    parser.add_argument("--symbols", default="NBP,OGDC,MARI,SYS,UBL,SAZEW,NATF", help="Comma-separated symbols")
    parser.add_argument("--portfolio", default="", help="Optional portfolio CSV")
    parser.add_argument("--fundamentals", default="", help="Optional fundamentals CSV")
    parser.add_argument("--max-symbols", type=int, default=20)
    parser.add_argument("--output-dir", default="data", help="Output folder")
    args = parser.parse_args()

    symbols = parse_symbols(args.symbols)
    portfolio = _load_optional_csv(args.portfolio)
    fundamentals = _load_optional_csv(args.fundamentals)
    bundle = run_autopilot_cycle(
        symbols,
        portfolio_df=portfolio,
        fundamentals_df=fundamentals,
        max_symbols=args.max_symbols,
    )
    paths = persist_autopilot_outputs(bundle, Path(args.output_dir))
    print("PSX autopilot cycle complete.")
    print(bundle["market_brief"].to_string(index=False))
    print("Saved outputs:")
    for key, path in paths.items():
        print(f"- {key}: {path}")


if __name__ == "__main__":
    main()
