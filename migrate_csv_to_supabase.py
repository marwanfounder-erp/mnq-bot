"""
migrate_csv_to_supabase.py
──────────────────────────
One-time script to import historical trades from trades.csv into Supabase.
Skips any trade_id that already exists in the database (safe to re-run).

Usage:
    python3 migrate_csv_to_supabase.py
"""

import csv
import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass

import os
from supabase import create_client

SUPABASE_URL = os.environ.get("SUPABASE_URL", "").strip()
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "").strip()
CSV_FILE     = Path(__file__).parent / "trades.csv"
TIMEZONE     = "US/Eastern"

if not SUPABASE_URL or not SUPABASE_KEY:
    raise SystemExit("ERROR: SUPABASE_URL and SUPABASE_KEY must be set in .env")

client = create_client(SUPABASE_URL, SUPABASE_KEY)
tz     = ZoneInfo(TIMEZONE)


def _to_ts(date_str: str, time_str: str):
    """Combine 'YYYY-MM-DD' + 'HH:MM:SS' → ISO 8601 timestamptz string."""
    if not date_str or not time_str:
        return None
    try:
        dt = datetime.datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M:%S")
        dt = dt.replace(tzinfo=tz)
        return dt.isoformat()
    except ValueError:
        return None


def _float_or_none(val: str):
    try:
        return float(val) if val.strip() else None
    except (ValueError, AttributeError):
        return None


def main():
    if not CSV_FILE.exists():
        raise SystemExit(f"ERROR: {CSV_FILE} not found")

    # Fetch trade_ids already in Supabase so we don't double-insert
    existing = set()
    response = client.table("trades").select("trade_id").execute()
    for row in response.data:
        existing.add(int(row["trade_id"]))
    print(f"[MIGRATE] {len(existing)} trade(s) already in Supabase: {sorted(existing) or 'none'}")

    inserted = 0
    skipped  = 0

    with open(CSV_FILE, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            trade_id = int(row["trade_id"])

            if trade_id in existing:
                print(f"[MIGRATE] Skipping trade #{trade_id} — already in DB")
                skipped += 1
                continue

            # Only migrate fully closed trades (exit_price must be present)
            if not row.get("exit_price", "").strip():
                print(f"[MIGRATE] Skipping trade #{trade_id} — not yet closed (no exit_price)")
                skipped += 1
                continue

            record = {
                "trade_id":     trade_id,
                "symbol":       row.get("symbol", ""),
                "side":         row.get("side", ""),
                "entry_price":  _float_or_none(row.get("entry_price", "")),
                "exit_price":   _float_or_none(row.get("exit_price", "")),
                "stop_price":   None,   # not stored in old CSV
                "target_price": None,   # not stored in old CSV
                "entry_time":   _to_ts(row.get("date", ""), row.get("entry_time", "")),
                "exit_time":    _to_ts(row.get("date", ""), row.get("exit_time", "")),
                "exit_reason":  row.get("exit_reason", ""),
                "pnl_dollars":  _float_or_none(row.get("pnl_dollars", "")),
                "daily_pnl":    _float_or_none(row.get("daily_pnl", "")),
                "paper_trade":  row.get("paper", "True").strip().lower() == "true",
                "indicators": {
                    "ema_fast": _float_or_none(row.get("ema_fast", "")),
                    "ema_slow": _float_or_none(row.get("ema_slow", "")),
                    "rsi":      _float_or_none(row.get("rsi", "")),
                },
            }

            client.table("trades").insert(record).execute()
            print(f"[MIGRATE] Inserted trade #{trade_id}  {record['side']}  "
                  f"entry={record['entry_price']}  exit={record['exit_price']}  "
                  f"P&L=${record['pnl_dollars']:+.2f}")
            inserted += 1

    print(f"\n[MIGRATE] Done — {inserted} inserted, {skipped} skipped.")


if __name__ == "__main__":
    main()
