from __future__ import annotations

"""
Optional local scheduled watchtower runner.

Usage examples:
    python watchtower_runner.py --symbols NBP,OGDC,MARI,SYS,UBL
    python watchtower_runner.py --symbols NBP,OGDC --max-symbols 2

This script writes:
- data/watchtower_news_events.csv
- data/watchtower_hazards.csv
- data/watchtower_hazard_details.csv

Run it manually or schedule it in Windows Task Scheduler for repeated local checks.
"""

import argparse
from pathlib import Path
import pandas as pd

from cwt_bot.advanced_scanners import parse_symbols
from cwt_bot.psx_data import load_psx_yahoo_ohlcv
from cwt_bot.signals import analyze_symbol
from cwt_bot.market_hazard import detect_price_hazards
from cwt_bot.event_risk_monitor import build_news_event_risk_snapshot


DATA_DIR = Path(__file__).resolve().parent / "data"


def run(symbols: list[str], max_symbols: int = 20) -> None:
    DATA_DIR.mkdir(exist_ok=True)
    selected = symbols[:max_symbols]

    news = build_news_event_risk_snapshot(symbols=selected)
    events = news.get("events")
    if isinstance(events, pd.DataFrame):
        events.to_csv(DATA_DIR / "watchtower_news_events.csv", index=False)
    else:
        pd.DataFrame().to_csv(DATA_DIR / "watchtower_news_events.csv", index=False)

    hazard_rows = []
    hazard_details = []
    failures = []
    for symbol in selected:
        try:
            higher = load_psx_yahoo_ohlcv(symbol, interval="1d", period="2y")
            lower = load_psx_yahoo_ohlcv(symbol, interval="1d", period="2y")
            result = analyze_symbol(
                symbol=symbol,
                higher_df=higher,
                lower_df=lower,
                asset_class="Stock",
                analysis_tf="1d",
                execution_tf="1d",
                risk_context={},
            )
            hazard = detect_price_hazards(
                result["execution_frame"],
                symbol=symbol,
                support_resistance=result.get("support_resistance", {}),
            )
            hazard_rows.append({
                "Symbol": symbol,
                "Hazard Level": hazard.get("hazard_level"),
                "Hazard Points": hazard.get("hazard_points"),
                "Gap %": hazard.get("latest_gap_pct"),
                "Range / ATR": hazard.get("latest_range_atr_ratio"),
                "Volume Ratio": hazard.get("latest_volume_ratio"),
                "Drawdown %": hazard.get("latest_drawdown_pct"),
                "Candle Return %": hazard.get("latest_candle_return_pct"),
                "Quick Action": hazard.get("summary"),
            })
            for alert in hazard.get("alerts", []):
                hazard_details.append({"Symbol": symbol, **alert})
        except Exception as exc:
            failures.append({"Symbol": symbol, "Error": str(exc)})

    pd.DataFrame(hazard_rows).to_csv(DATA_DIR / "watchtower_hazards.csv", index=False)
    pd.DataFrame(hazard_details).to_csv(DATA_DIR / "watchtower_hazard_details.csv", index=False)
    pd.DataFrame(failures).to_csv(DATA_DIR / "watchtower_failures.csv", index=False)

    print(f"Saved watchtower outputs to: {DATA_DIR}")
    print(f"News risk level: {news.get('risk_level')}")
    print(f"Hazard rows: {len(hazard_rows)}, detailed alerts: {len(hazard_details)}, failures: {len(failures)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="PSX CWT Watchtower Runner")
    parser.add_argument("--symbols", default="NBP,OGDC,MARI,SYS,UBL,SAZEW,NATF", help="Comma-separated PSX symbols")
    parser.add_argument("--max-symbols", type=int, default=20, help="Maximum symbols to scan")
    args = parser.parse_args()
    symbols = parse_symbols(args.symbols)
    run(symbols, max_symbols=args.max_symbols)


if __name__ == "__main__":
    main()
