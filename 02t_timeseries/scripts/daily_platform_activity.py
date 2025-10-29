#!/usr/bin/env python3
"""
Build daily platform activity from a HUGE hashtags_by_post_full.csv

Input columns (per row = one (post, hashtag)):
  timestamp_utc, epoch_seconds, username, post_id, shortcode, hashtag

Output (per day, UTC):
  date, total_mentions, total_posts, unique_authors, unique_hashtags
"""

import argparse, os, sys, duckdb

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True, help="Path to hashtags_by_post_full.csv (csv or csv.gz)")
    ap.add_argument("--out_csv", required=True, help="Where to write daily_platform_activity.csv")
    ap.add_argument("--threads", type=int, default=4, help="DuckDB threads (default: 4)")
    ap.add_argument("--memory_limit", default="", help="Optional DuckDB memory cap, e.g. '8GB'")
    ap.add_argument("--format", choices=["csv","parquet"], default="csv", help="Output format")
    ap.add_argument("--date_from", default="", help="Optional inclusive start date YYYY-MM-DD (UTC)")
    ap.add_argument("--date_to",   default="", help="Optional inclusive end date YYYY-MM-DD (UTC)")
    args = ap.parse_args()

    in_csv  = os.path.abspath(args.in_csv)
    out_csv = os.path.abspath(args.out_csv)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    con = duckdb.connect(database=":memory:")
    con.execute(f"PRAGMA threads={args.threads};")
    if args.memory_limit:
        con.execute(f"PRAGMA memory_limit='{args.memory_limit}';")

    # Build an efficient view over the CSV (DuckDB streams it)
    # NOTE: using epoch_seconds → DATE is faster and safer than parsing ISO strings.
    where = "WHERE epoch_seconds IS NOT NULL"
    if args.date_from:
        where += f" AND CAST(to_timestamp(epoch_seconds) AS DATE) >= DATE '{args.date_from}'"
    if args.date_to:
        where += f" AND CAST(to_timestamp(epoch_seconds) AS DATE) <= DATE '{args.date_to}'"

    con.execute(f"""
        CREATE OR REPLACE TEMP VIEW input AS
        SELECT
            CAST(to_timestamp(epoch_seconds) AS DATE) AS date,
            post_id,
            username,
            hashtag
        FROM read_csv_auto('{in_csv}', header=True)
        {where};
    """)

    # Aggregate per day. DISTINCT ignores NULLs automatically.
    con.execute("""
        CREATE OR REPLACE TABLE daily AS
        SELECT
            date,
            COUNT(*)                              AS total_mentions,
            COUNT(DISTINCT post_id)               AS total_posts,
            COUNT(DISTINCT username)              AS unique_authors,
            COUNT(DISTINCT hashtag)               AS unique_hashtags
        FROM input
        GROUP BY 1
        ORDER BY 1;
    """)

    # Write output
    if args.format == "csv":
        con.execute(f"""
            COPY daily TO '{out_csv}'
            WITH (HEADER, DELIMITER ',');
        """)
    else:
        # Parquet is smaller/faster if you’ll read many times later
        con.execute(f"""
            COPY daily TO '{out_csv}'
            (FORMAT PARQUET);
        """)

    # Quick summary to console
    n_days, min_d, max_d = con.execute("""
        SELECT COUNT(*), MIN(date), MAX(date) FROM daily;
    """).fetchone()
    print(f"✓ Wrote {n_days} daily rows → {out_csv}")
    print(f"   Range: {min_d} .. {max_d}")

    # (Optional) peek at a few rows
    sample = con.execute("SELECT * FROM daily ORDER BY date LIMIT 5;").fetchdf()
    print(sample.to_string(index=False))

if __name__ == "__main__":
    sys.exit(main())
