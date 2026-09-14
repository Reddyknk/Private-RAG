#!/usr/bin/env python3
"""
stock_tools.py - Stock Market Performance & Screener Tool for Autonomous Agents.

Identifies:
1. Stocks with the highest percentage increase (top gainers).
2. Stocks with the lowest percentage decrease / biggest drop (top losers).

Leverages public market screener endpoints (no API key required) with
automatic failover to the local flat-file database (data/registry.csv).
"""

import os
import sys
import csv
import json
import argparse
import urllib.request
from pathlib import Path
from typing import List, Dict, Any

REGISTRY_PATH = Path(__file__).resolve().parent.parent / "data" / "registry.csv"


def load_registry_stocks() -> List[Dict[str, Any]]:
    """Load stock database from local CSV flat-file."""
    stocks = []
    if not REGISTRY_PATH.exists():
        return stocks

    with open(REGISTRY_PATH, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                stocks.append({
                    "symbol": row["symbol"].strip(),
                    "name": row["name"].strip(),
                    "sector": row.get("sector", "").strip(),
                    "previous_close": float(row.get("previous_close", 0)),
                    "current_price": float(row.get("current_price", 0)),
                    "change_percent": float(row.get("change_percent", 0)),
                    "volume": int(row.get("volume", 0)),
                    "source": "local_registry"
                })
            except Exception:
                continue
    return stocks


_executed_external_apis: List[Dict[str, Any]] = []


def fetch_yahoo_screener(scr_id: str = "day_gainers", count: int = 10) -> List[Dict[str, Any]]:
    """
    Fetch public screener from Yahoo Finance without an API key.
    scr_id can be 'day_gainers' or 'day_losers'.
    """
    url = f"https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved?formatted=false&scrIds={scr_id}&count={count}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    }
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=6) as resp:
            status_code = resp.status if hasattr(resp, "status") else 200
            data = json.loads(resp.read().decode("utf-8"))
            results = data.get("finance", {}).get("result", [])
            quotes = results[0].get("quotes", []) if results else []
            parsed = []
            for q in quotes:
                chg_pct = q.get("regularMarketChangePercent")
                if chg_pct is None:
                    continue
                parsed.append({
                    "symbol": q.get("symbol"),
                    "name": q.get("shortName") or q.get("longName") or q.get("symbol"),
                    "sector": "",
                    "current_price": round(float(q.get("regularMarketPrice", 0)), 2),
                    "change_amount": round(float(q.get("regularMarketChange", 0)), 2),
                    "change_percent": round(float(chg_pct), 2),
                    "volume": q.get("regularMarketVolume", 0),
                    "source": "live_screener"
                })

            _executed_external_apis.append({
                "name": f"Yahoo Finance Screener API ({scr_id})",
                "url": url,
                "method": "GET",
                "status_code": status_code,
                "request": {
                    "method": "GET",
                    "url": url,
                    "scr_id": scr_id,
                    "count": count
                },
                "response": {
                    "status_code": status_code,
                    "quotes_retrieved": len(parsed),
                    "top_ticker": parsed[0]["symbol"] if parsed else None,
                    "sample": parsed[:3]
                }
            })
            return parsed
    except Exception as e:
        _executed_external_apis.append({
            "name": f"Yahoo Finance Screener API ({scr_id})",
            "url": url,
            "method": "GET",
            "status_code": 500,
            "request": {"method": "GET", "url": url, "scr_id": scr_id},
            "response": {"status_code": 500, "error": str(e), "fallback": "data/registry.csv"}
        })
        return []


def get_top_gainers(limit: int = 5) -> List[Dict[str, Any]]:
    """
    Retrieve stocks with the highest percentage increase.
    Attempts live fetch first, falls back to local data/registry.csv.
    """
    live_gainers = fetch_yahoo_screener("day_gainers", count=limit * 2)
    if live_gainers:
        sorted_gainers = sorted(live_gainers, key=lambda x: x["change_percent"], reverse=True)
        return sorted_gainers[:limit]

    # Fallback to local registry
    stocks = load_registry_stocks()
    sorted_gainers = sorted(stocks, key=lambda x: x["change_percent"], reverse=True)
    return sorted_gainers[:limit]


def get_top_losers(limit: int = 5) -> List[Dict[str, Any]]:
    """
    Retrieve stocks with the lowest percentage decrease (greatest decline / negative percentage).
    Attempts live fetch first, falls back to local data/registry.csv.
    """
    live_losers = fetch_yahoo_screener("day_losers", count=limit * 2)
    if live_losers:
        sorted_losers = sorted(live_losers, key=lambda x: x["change_percent"])
        return sorted_losers[:limit]

    # Fallback to local registry
    stocks = load_registry_stocks()
    sorted_losers = sorted(stocks, key=lambda x: x["change_percent"])
    return sorted_losers[:limit]


def main():
    parser = argparse.ArgumentParser(description="Analyze stocks with highest increase or lowest decrease.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--gainers", action="store_true", help="Find stocks with highest percentage increase")
    group.add_argument("--losers", action="store_true", help="Find stocks with lowest percentage decrease (biggest drops)")
    parser.add_argument("--limit", type=int, default=5, help="Number of stocks to display (default: 5)")
    parser.add_argument("--json", action="store_true", help="Output raw JSON format")
    args = parser.parse_args()

    if args.gainers:
        title = "🚀 STOCKS WITH HIGHEST PERCENTAGE INCREASE (TOP GAINERS)"
        results = get_top_gainers(limit=args.limit)
    else:
        title = "📉 STOCKS WITH LOWEST PERCENTAGE DECREASE (BIGGEST DROPS)"
        results = get_top_losers(limit=args.limit)

    if args.json:
        print(json.dumps({
            "metric": "gainers" if args.gainers else "losers",
            "count": len(results),
            "data": results,
            "external_apis": _executed_external_apis
        }, indent=2))
        return

    print("=" * 70)
    print(title)
    print("=" * 70)
    print(f"{'Symbol':<8} {'Company Name':<28} {'Price ($)':<12} {'Change %':<10} {'Data Source'}")
    print("-" * 70)
    for s in results:
        sign = "+" if s["change_percent"] > 0 else ""
        pct_str = f"{sign}{s['change_percent']:.2f}%"
        print(f"{s['symbol']:<8} {s['name'][:26]:<28} ${s['current_price']:<11.2f} {pct_str:<10} {s['source']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
