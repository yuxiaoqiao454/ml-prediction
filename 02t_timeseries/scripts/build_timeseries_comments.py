# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# Build per-hashtag daily engagement time series (normalized & smoothed).

# Inputs:
#     01-intermediate/hashtags/hashtags_with_engagement.csv
#     01-intermediate/platform/daily_platform_engagement.csv

# Outputs:
#     02-timeseries/hashtag_timeseries_engagement_norm_smooth.csv

# Each row:
#     hashtag, date, total_posts, total_engagement,
#     norm_self, norm_platform, smoothed_norm_self, smoothed_norm_platform
# """

# import pandas as pd
# from pathlib import Path

# # --- parameters ---
# ROLLING_WINDOW = 15  # days for smoothing
# MIN_DAYS = 5         # skip hashtags with fewer total days
# EPS = 1e-9

# # --- paths ---
# BASE = Path(__file__).resolve().parents[1]
# HASHTAG_PATH = BASE / "01-intermediate" / "hashtags" / "hashtags_with_engagement.csv"
# PLATFORM_PATH = BASE / "01-intermediate" / "platform" / "daily_platform_engagement.csv"
# OUT_PATH = BASE / "02-timeseries" / "hashtag_timeseries_engagement_norm_smooth.csv"

# # --- load data ---
# print("Loading hashtags_with_engagement.csv ...")
# hashtags = pd.read_csv(HASHTAG_PATH, low_memory=False)

# print("Loading daily_platform_engagement.csv ...")
# platform = pd.read_csv(PLATFORM_PATH, parse_dates=["date"])

# # --- convert epoch to date ---
# hashtags["date"] = pd.to_datetime(hashtags["epoch_seconds"], unit="s", utc=True, errors="coerce").dt.date


# # --- check required columns ---
# for col in ["hashtag", "total_engagement", "post_id"]:
#     if col not in hashtags.columns:
#         raise ValueError(f"Missing required column: {col}")

# # --- aggregate daily totals per hashtag ---
# print("Aggregating per hashtag per day ...")
# daily = (
#     hashtags.groupby(["hashtag", "date"], as_index=False)
#     .agg(
#         total_posts=("post_id", "count"),
#         total_engagement=("total_engagement", "sum"),
#     )
# )

# # --- join with platform-level engagement for normalization ---
# platform["date"] = pd.to_datetime(platform["date"]).dt.date
# merged = pd.merge(daily, platform, on="date", how="left", suffixes=("", "_platform"))

# # --- compute normalized metrics ---
# merged["norm_self"] = merged["total_engagement"] / (merged["total_posts"] + EPS)
# merged["norm_platform"] = (
#     merged["total_engagement"] / (merged["total_engagement_platform"] + EPS) * 1000
# )

# # --- smoothing (rolling mean per hashtag) ---
# print(f"Smoothing with rolling window = {ROLLING_WINDOW} days ...")
# merged = merged.sort_values(["hashtag", "date"])
# merged["smoothed_norm_self"] = (
#     merged.groupby("hashtag")["norm_self"]
#     .transform(lambda x: x.rolling(ROLLING_WINDOW, min_periods=1).mean())
# )
# merged["smoothed_norm_platform"] = (
#     merged.groupby("hashtag")["norm_platform"]
#     .transform(lambda x: x.rolling(ROLLING_WINDOW, min_periods=1).mean())
# )

# # --- optional: filter out extremely sparse hashtags ---
# keep_tags = (
#     merged.groupby("hashtag")["date"].count().reset_index(name="n_days")
# )
# keep_tags = keep_tags[keep_tags["n_days"] >= MIN_DAYS]["hashtag"]
# merged = merged[merged["hashtag"].isin(keep_tags)]

# # --- save ---
# OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
# merged.to_csv(OUT_PATH, index=False)

# print(f"✅ Saved normalized, smoothed engagement time series → {OUT_PATH}")
# print("Sample:")
# print(merged.sample(5)[["hashtag", "date", "total_engagement", "smoothed_norm_self", "smoothed_norm_platform"]])


# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# Build per-hashtag daily engagement time series (DuckDB version).

# Features:
#     - Handles 7+ GB CSVs efficiently (streamed aggregation on disk)
#     - Aggregates total_engagement per (hashtag, date)
#     - Joins with daily platform engagement for normalization
#     - Computes per-post and per-platform normalized metrics
#     - Applies 15-day rolling smoothing in pandas (lightweight)

# Inputs:
#     01-intermediate/hashtags/hashtags_with_engagement.csv
#         columns must include: epoch_seconds, hashtag, post_id, total_engagement
#     01-intermediate/platform/daily_platform_engagement.csv

# Outputs:
#     02-timeseries/hashtag_timeseries_engagement_norm_smooth.csv
# """

# import duckdb
# import pandas as pd
# from pathlib import Path

# # --- parameters ---
# ROLLING_WINDOW = 15
# EPS = 1e-9

# # --- paths ---
# BASE = Path(__file__).resolve().parents[1]
# HASHTAG_PATH = BASE / "01-intermediate" / "hashtags" / "hashtags_with_engagement.csv"
# PLATFORM_PATH = BASE / "01-intermediate" / "platform" / "daily_platform_engagement.csv"
# OUT_PATH = BASE / "02-timeseries" / "hashtag_timeseries_engagement_norm_smooth.csv"

# # --- connect DuckDB ---
# con = duckdb.connect()

# print("Aggregating daily engagement per hashtag directly from CSV (DuckDB streaming)...")

# # DuckDB query: aggregate on disk
# query = f"""
# SELECT
#     hashtag,
#     DATE_TRUNC('day', TO_TIMESTAMP(epoch_seconds))::DATE AS date,
#     COUNT(post_id) AS total_posts,
#     SUM(total_engagement) AS total_engagement
# FROM read_csv_auto('{HASHTAG_PATH}', SAMPLE_SIZE=-1)
# WHERE hashtag IS NOT NULL AND total_engagement IS NOT NULL
# GROUP BY 1, 2
# ORDER BY 2
# """
# daily = con.execute(query).df()
# print(f"Aggregated {len(daily):,} hashtag–date rows")

# # --- join with platform totals ---
# print("Joining with daily platform engagement baseline...")
# # Ensure both sides have the same dtype for 'date'
# platform = pd.read_csv(PLATFORM_PATH, parse_dates=["date"])
# platform["date"] = pd.to_datetime(platform["date"], errors="coerce").dt.date

# # convert DuckDB date to plain date
# daily["date"] = pd.to_datetime(daily["date"], errors="coerce").dt.date

# merged = pd.merge(daily, platform, on="date", how="left", suffixes=("", "_platform"))


# # --- normalized metrics ---
# merged["norm_self"] = merged["total_engagement"] / (merged["total_posts"] + EPS)
# merged["norm_platform"] = (
#     merged["total_engagement"] / (merged["total_engagement_platform"] + EPS) * 1000
# )

# # --- smoothing ---
# print(f"Smoothing with rolling window = {ROLLING_WINDOW} days ...")
# merged = merged.sort_values(["hashtag", "date"])
# merged["smoothed_norm_self"] = (
#     merged.groupby("hashtag")["norm_self"]
#     .transform(lambda x: x.rolling(ROLLING_WINDOW, min_periods=1).mean())
# )
# merged["smoothed_norm_platform"] = (
#     merged.groupby("hashtag")["norm_platform"]
#     .transform(lambda x: x.rolling(ROLLING_WINDOW, min_periods=1).mean())
# )

# # --- save ---
# OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
# merged.to_csv(OUT_PATH, index=False)
# print(f"✅ Saved → {OUT_PATH}")
# print(merged.sample(5))


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build per-hashtag daily comment time series (normalized & smoothed).

Powered by DuckDB — fast and memory-safe.

Inputs
------
01-intermediate/hashtags/hashtags_by_post_full_users_comments.csv
01-intermediate/hashtags/hashtags_by_post_full.csv
01-intermediate/platform/daily_platform_comments.csv

Outputs
-------
02-timeseries/hashtag_timeseries_comments_norm_smooth.csv
"""

import duckdb
import pandas as pd
from pathlib import Path

# --- parameters ---
ROLLING_WINDOW = 15
EPS = 1e-9

# --- paths ---
BASE = Path(__file__).resolve().parents[1]
COMMENTS_PATH = BASE / "01-intermediate" / "hashtags" / "hashtags_by_post_full_users_comments.csv"
HASHTAGS_PATH = BASE / "01-intermediate" / "hashtags" / "hashtags_by_post_full.csv"
PLATFORM_PATH = BASE / "01-intermediate" / "platform" / "daily_platform_comments.csv"
OUT_PATH = BASE / "02-timeseries" / "hashtag_timeseries_comments_norm_smooth.csv"

# --- connect DuckDB ---
con = duckdb.connect()

print("Joining comments ↔ hashtags and aggregating daily counts (DuckDB streaming)...")

# Aggregate comment counts per (hashtag, date)
query = f"""
SELECT
    h.hashtag,
    DATE_TRUNC('day', TO_TIMESTAMP(c.comment_epoch))::DATE AS date,
    COUNT(*) AS total_comments
FROM read_csv_auto('{COMMENTS_PATH}', SAMPLE_SIZE=-1) AS c
JOIN read_csv_auto('{HASHTAGS_PATH}', SAMPLE_SIZE=-1) AS h
  ON c.post_id = h.post_id
WHERE c.comment_epoch IS NOT NULL
GROUP BY 1, 2
ORDER BY 2
"""
daily = con.execute(query).df()
print(f"Aggregated {len(daily):,} hashtag–date rows.")

# --- join with platform-level totals ---
print("Joining with daily platform totals...")
platform = pd.read_csv(PLATFORM_PATH, parse_dates=["date"])
platform["date"] = platform["date"].dt.date
daily["date"] = pd.to_datetime(daily["date"]).dt.date

merged = pd.merge(daily, platform, on="date", how="left", suffixes=("", "_platform"))

# --- normalize ---
merged["norm_platform"] = (
    merged["total_comments"] / (merged["total_comments_platform"] + EPS) * 1000
)

# --- smoothing ---
print(f"Smoothing with rolling window = {ROLLING_WINDOW} days ...")
merged = merged.sort_values(["hashtag", "date"])
merged["smoothed_norm_platform"] = (
    merged.groupby("hashtag")["norm_platform"]
    .transform(lambda x: x.rolling(ROLLING_WINDOW, min_periods=1).mean())
)

# --- save ---
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
merged.to_csv(OUT_PATH, index=False)
print(f"✅ Saved normalized, smoothed comment time series → {OUT_PATH}")
print(merged.sample(5))
