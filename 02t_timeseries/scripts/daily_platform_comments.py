# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# Compute daily total engagement (likes + comments) across the entire platform.

# Purpose:
#     To provide a platform-level normalization baseline for engagement-based
#     hashtag time series (e.g. engagement_per_k_posts or engagement_ratio).

# Inputs:
#     01-intermediate/hashtags/engagement_by_post.csv

# Outputs:
#     01-intermediate/platform/daily_platform_engagement.csv

# Columns:
#     date, total_likes, total_comments, total_engagement, total_posts
# """

# import pandas as pd
# from pathlib import Path
# from datetime import datetime

# # --- paths ---
# BASE = Path(__file__).resolve().parents[1]
# ENG_PATH = BASE / "01-intermediate" / "hashtags" / "engagement_by_post.csv"
# OUT_PATH = BASE / "01-intermediate" / "platform" / "daily_platform_engagement.csv"

# # --- load ---
# print("Loading engagement_by_post.csv ...")
# df = pd.read_csv(ENG_PATH, low_memory=False)
# print(f"Loaded {len(df):,} rows")

# # --- sanity check for required columns ---
# required = {"post_epoch", "likes_count", "comments_count"}
# missing = required - set(df.columns)
# if missing:
#     raise ValueError(f"Missing required columns: {missing}")

# # --- compute engagement per post ---
# df["total_engagement"] = df[["likes_count", "comments_count"]].sum(axis=1, skipna=True)

# # --- convert timestamp to date ---
# df["date"] = pd.to_datetime(df["post_epoch"], unit="s", utc=True).dt.date

# # --- aggregate by date ---
# daily = (
#     df.groupby("date", as_index=False)
#       .agg(
#           total_posts=("post_id", "count"),
#           total_likes=("likes_count", "sum"),
#           total_comments=("comments_count", "sum"),
#           total_engagement=("total_engagement", "sum")
#       )
#       .sort_values("date")
# )

# # --- save ---
# OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
# daily.to_csv(OUT_PATH, index=False)
# print(f"✅ Saved daily platform engagement → {OUT_PATH}")
# print(daily.tail())


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute daily total comment activity across the platform.

Fast DuckDB version (streaming, no full in-memory load).

Inputs
------
01-intermediate/hashtags/hashtags_by_post_full_users_comments.csv
    columns: comment_epoch, post_id, ...

Outputs
-------
01-intermediate/platform/daily_platform_comments.csv
    columns: date, total_comments
"""

import duckdb
from pathlib import Path

# --- paths ---
BASE = Path(__file__).resolve().parents[1]
COMMENTS_PATH = BASE / "01-intermediate" / "hashtags" / "hashtags_by_post_full_users_comments.csv"
OUT_PATH = BASE / "01-intermediate" / "platform" / "daily_platform_comments.csv"

# --- DuckDB query ---
con = duckdb.connect()

print("Aggregating daily total comments (DuckDB streaming)...")

query = f"""
SELECT
    DATE_TRUNC('day', TO_TIMESTAMP(comment_epoch))::DATE AS date,
    COUNT(*) AS total_comments
FROM read_csv_auto('{COMMENTS_PATH}', SAMPLE_SIZE=-1)
WHERE comment_epoch IS NOT NULL
GROUP BY 1
ORDER BY 1
"""

daily = con.execute(query).df()

# --- save ---
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
daily.to_csv(OUT_PATH, index=False)

print(f"✅ Saved → {OUT_PATH}")
print(daily.tail())
