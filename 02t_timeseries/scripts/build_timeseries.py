#!/usr/bin/env python3
"""
Build daily hashtag time series from hashtags_by_post.csv

Run format:
python /Users/xiaoqiao/Documents/GitHub/yu-4yp/Ins_dataset/build_timeseries.py \
 --in_csv /Users/xiaoqiao/Documents/GitHub/yu-4yp/Ins_dataset/hashtags_by_post_full.csv \
 --out_csv /Users/xiaoqiao/Documents/GitHub/yu-4yp/Ins_dataset/hashtag_timeseries_norm_smooth.csv \
 --platform_activity_csv /Users/xiaoqiao/Documents/GitHub/yu-4yp/Ins_dataset/daily_platform_activity.csv
"""

import argparse
from pathlib import Path
import sys
import pandas as pd
import numpy as np



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True, help="Path to hashtags_by_post.csv")
    ap.add_argument("--out_csv", required=True, help="Output CSV with hashtag time series")
    ap.add_argument("--platform_activity_csv", default="",
                help="Optional: path to daily_platform_activity.csv with daily total posts")
    ap.add_argument("--per_k", type=int, default=1000,
                help="Scale normalised rate per K posts (default: per 1000 posts)")

    args = ap.parse_args()

    in_path = Path(args.in_csv)
    out_path = Path(args.out_csv)

    print(f"[info] Python: {sys.executable}", flush=True)
    print(f"[info] Input : {in_path.resolve()}", flush=True)
    print(f"[info] Output: {out_path.resolve()}", flush=True)

    # 0) Validate input exists
    if not in_path.exists():
        print(f"[error] Input CSV not found: {in_path}", file=sys.stderr, flush=True)
        sys.exit(2)

    try:
        df = pd.read_csv(in_path)
        print(f"[ok] Loaded {len(df):,} rows with columns: {list(df.columns)}", flush=True)

        # 2) Parse timestamp_utc explicitly
        print("[step] Converting 'timestamp_utc' to datetime...", flush=True)
        if "timestamp_utc" not in df.columns:
            print("[error] Column 'timestamp_utc' missing in input CSV.", file=sys.stderr, flush=True)
            sys.exit(3)

        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], errors="coerce", utc=True)
        df["date"] = df["timestamp_utc"].dt.date

        print("[step] Deriving total posts per day for normalisation...", flush=True)

        daily_posts = None

        if args.platform_activity_csv:
            # Prefer external totals if provided
            try:
                plat = pd.read_csv(args.platform_activity_csv)
                # Try to guess the date column and total-posts column robustly
                date_col_candidates = [c for c in plat.columns if c.lower() in ("date", "day")]
                posts_col_candidates = [c for c in plat.columns if c.lower() in ("posts", "total_posts", "n_posts")]

                if not date_col_candidates or not posts_col_candidates:
                    raise ValueError(f"Could not find date/posts columns in {args.platform_activity_csv}. "
                                    f"Have columns: {list(plat.columns)}")

                dcol = date_col_candidates[0]
                pcol = posts_col_candidates[0]

                # Normalise date -> date (no time) to match df["date"]
                plat[dcol] = pd.to_datetime(plat[dcol], errors="coerce", utc=True).dt.date
                daily_posts = (plat[[dcol, pcol]]
                            .rename(columns={dcol: "date", pcol: "total_posts_day"})
                            .dropna())
                # If multiple rows per date, sum (or max) — choose sum by default
                daily_posts = (daily_posts.groupby("date", as_index=False)["total_posts_day"].sum())
                print(f"[ok] Loaded platform totals from {args.platform_activity_csv} with "
                    f"{len(daily_posts):,} days.", flush=True)
            except Exception as e:
                print(f"[warn] Failed to use platform_activity_csv ({e}); "
                    f"falling back to computing from hashtags file.", flush=True)

        if daily_posts is None:
            # Fallback: compute daily total posts from this hashtags file
            # (distinct posts per day across ALL hashtags)
            required = {"post_id", "date"}
            if not required.issubset(df.columns):
                raise ValueError(f"Missing columns to compute daily totals: {required}")
            daily_posts = (df.groupby("date")["post_id"]
                            .nunique()
                            .reset_index(name="total_posts_day"))
            print(f"[ok] Computed daily total posts from input file: {len(daily_posts):,} days.", flush=True)



        # 3) Group by (hashtag, date) → mentions
        print("[step] Aggregating per (hashtag, date)...", flush=True)
        ts = (
            df.groupby(["hashtag", "date"])
              .size()
              .reset_index(name="mentions")
        )

        # Merge daily totals
        ts = ts.merge(daily_posts, on="date", how="left")

        # keep as numeric with NaN (not pd.NA), so numpy division is smooth
        ts["total_posts_day"] = pd.to_numeric(ts["total_posts_day"], errors="coerce")


        

        # ensure numerics
        num = pd.to_numeric(ts["mentions"], errors="coerce").to_numpy(dtype="float64")
        den = pd.to_numeric(ts["total_posts_day"], errors="coerce").to_numpy(dtype="float64")

        # mentions_per_post = mentions / total_posts_day, but only where den>0; otherwise NaN
        ts["mentions_per_post"] = np.divide(
            num, den,
            out=np.full_like(num, np.nan, dtype="float64"),
            where=den > 0
        )

        ts[f"mentions_per_{args.per_k}posts"] = ts["mentions_per_post"] * float(args.per_k)

        # Optional: 7-day moving average for the per-K signal (smoother)
        ts = ts.sort_values(["hashtag", "date"])
        ts[f"mentions_per_{args.per_k}posts_ma7"] = (
            ts.groupby("hashtag")[f"mentions_per_{args.per_k}posts"]
            .transform(lambda s: s.rolling(15, min_periods=1).mean())
        )

        print(f"[ok] Aggregated to {len(ts):,} rows.", flush=True)

        # 🔥 NEW: total frequency per hashtag across all days
        print("[step] Counting total mentions per hashtag...", flush=True)
        freq = (
            df.groupby("hashtag")
            .size()
            .reset_index(name="total_mentions")
            .sort_values("total_mentions", ascending=False)
        )
        print("[ok] Top 200 hashtags:")
        print(freq.head(200).to_string(index=False))

        # 4) Ensure output folder exists
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # 5) Save
        ts.to_csv(out_path, index=False)
        print(f"[done] Wrote daily hashtag counts to {out_path} 🎉", flush=True)

        # 6) Tiny preview
        print(ts.head(5).to_string(index=False), flush=True)

    except Exception as e:
        print(f"[exception] {type(e).__name__}: {e}", file=sys.stderr, flush=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
