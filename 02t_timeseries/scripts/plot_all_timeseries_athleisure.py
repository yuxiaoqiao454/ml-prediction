# # #!/usr/bin/env python3
# # # -*- coding: utf-8 -*-
# # """
# # Plot all variants of the time-series signal for a single hashtag (#athleisure)
# # with aligned time axes.

# # Inputs
# # ------
# # 02-timeseries/hashtag_timeseries_norm_smooth.csv
# # 02-timeseries/hashtag_timeseries_engagement_norm_smooth.csv

# # Outputs
# # -------
# # 04-figures/timeseries_compare/athleisure_all_signals.png
# # """

# # import pandas as pd
# # import matplotlib.pyplot as plt
# # from pathlib import Path

# # # --- paths ---
# # BASE = Path(__file__).resolve().parents[1]
# # MENTION_PATH = BASE / "02-timeseries" / "hashtag_timeseries_norm_smooth.csv"
# # ENG_PATH = BASE / "02-timeseries" / "hashtag_timeseries_engagement_norm_smooth.csv"
# # OUT_PATH = BASE / "04-figures" / "timeseries_compare" / "athleisure_all_signals.png"

# # # --- load ---
# # mention = pd.read_csv(MENTION_PATH, parse_dates=["date"])
# # eng = pd.read_csv(ENG_PATH, parse_dates=["date"])

# # # --- filter to hashtag ---
# # tag = "athleisure"
# # m = mention[mention["hashtag"].str.lower() == tag]
# # e = eng[eng["hashtag"].str.lower() == tag]

# # # --- prepare figure ---
# # plt.style.use("seaborn-v0_8-whitegrid")
# # fig, axes = plt.subplots(7, 1, sharex=True, figsize=(11, 10))
# # fig.suptitle(f"Time-Series Signals for #{tag}", fontsize=14, weight="bold")

# # # --- 1. Raw mention counts ---
# # axes[0].plot(m["date"], m["total_posts"], color="gray")
# # axes[0].set_ylabel("Raw\nMentions")

# # # --- 2. Normalized (per-K-posts) ---
# # if "mentions_per_k_posts" in m.columns:
# #     axes[1].plot(m["date"], m["mentions_per_k_posts"], color="steelblue")
# #     axes[1].set_ylabel("Norm\n(per-K-posts)")
# # else:
# #     axes[1].plot(m["date"], m["total_posts"], color="steelblue")
# #     axes[1].set_ylabel("Norm")

# # # --- 3. Smoothed mention signal ---
# # if "smoothed_mentions" in m.columns:
# #     axes[2].plot(m["date"], m["smoothed_mentions"], color="navy")
# # else:
# #     # fall back to normalized smoothed
# #     axes[2].plot(m["date"], m.get("smoothed_mentions_per_k_posts", m["total_posts"]), color="navy")
# # axes[2].set_ylabel("Smoothed\nMentions")

# # # --- 4. Raw engagement (sum of likes + comments) ---
# # axes[3].plot(e["date"], e["total_engagement"], color="orange")
# # axes[3].set_ylabel("Raw\nEngagement")

# # # --- 5. Self-normalized (per-post) ---
# # axes[4].plot(e["date"], e["norm_self"], color="darkorange")
# # axes[4].set_ylabel("Per-Post\nEngagement")

# # # --- 6. Platform-normalized ---
# # axes[5].plot(e["date"], e["norm_platform"], color="tomato")
# # axes[5].set_ylabel("Per-Platform\nEngagement")

# # # --- 7. Smoothed (15-day rolling) engagement signal ---
# # axes[6].plot(e["date"], e["smoothed_norm_platform"], color="firebrick")
# # axes[6].set_ylabel("Smoothed\nEngagement")
# # axes[6].set_xlabel("Date")

# # # --- tidy layout ---
# # plt.tight_layout(rect=[0, 0, 1, 0.96])
# # OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
# # plt.savefig(OUT_PATH, dpi=300)
# # plt.show()

# # print(f"✅ Saved figure → {OUT_PATH}")


# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# Plot all variants of the time-series signal for a single hashtag (#athleisure)
# with aligned time axes.

# Inputs
# ------
# 02-timeseries/hashtag_timeseries_norm_smooth.csv
# 02-timeseries/hashtag_timeseries_engagement_norm_smooth.csv

# Outputs
# -------
# 04-figures/timeseries_compare/athleisure_all_signals.png
# """

# import pandas as pd
# import matplotlib.pyplot as plt
# from pathlib import Path

# # --- paths ---
# BASE = Path(__file__).resolve().parents[1]
# MENTION_PATH = BASE / "02-timeseries" / "hashtag_timeseries_norm_smooth.csv"
# ENG_PATH = BASE / "02-timeseries" / "hashtag_timeseries_comments_norm_smooth.csv"
# OUT_PATH = BASE / "04-figures" / "timeseries_compare" / "athleisure_all_signals.png"

# # --- load ---
# mention = pd.read_csv(MENTION_PATH, parse_dates=["date"])
# eng = pd.read_csv(ENG_PATH, parse_dates=["date"])

# # --- filter to hashtag ---
# tag = "athleisure"
# m = mention[mention["hashtag"].str.lower() == tag]
# e = eng[eng["hashtag"].str.lower() == tag]

# # --- verify columns exist ---
# print("Mention columns:", m.columns.tolist())
# print("Engagement columns:", e.columns.tolist())

# # --- prepare figure ---
# plt.style.use("seaborn-v0_8-whitegrid")
# fig, axes = plt.subplots(7, 1, sharex=True, figsize=(11, 10))
# fig.suptitle(f"Time-Series Signals for #{tag}", fontsize=14, weight="bold")

# # 1️⃣ Raw mention counts
# if "mentions" in m.columns:
#     axes[0].plot(m["date"], m["mentions"], color="gray")
# else:
#     axes[0].plot(m["date"], m[m.columns[1]], color="gray")  # fallback
# axes[0].set_ylabel("Raw\nMentions")

# # 2️⃣ Normalized (per-K-posts)
# col_norm = None
# for c in ["mentions_per_1000posts", "mentions_per_post", "mentions_per_k_posts"]:
#     if c in m.columns:
#         col_norm = c
#         break
# axes[1].plot(m["date"], m[col_norm], color="steelblue")
# axes[1].set_ylabel("Norm.\nMentions")

# # 3️⃣ Smoothed (rolling mean)
# col_smooth = None
# for c in ["mentions_per_1000posts_ma7", "mentions_per_1000posts_ma15", "smoothed_mentions"]:
#     if c in m.columns:
#         col_smooth = c
#         break
# axes[2].plot(m["date"], m[col_smooth], color="navy")
# axes[2].set_ylabel("Smoothed\nMentions")

# # 4️⃣ Raw engagement (sum of likes + comments)
# axes[3].plot(e["date"], e["total_engagement"], color="orange")
# axes[3].set_ylabel("Raw\nEngagement")

# # 5️⃣ Self-normalized (per-post)
# axes[4].plot(e["date"], e["norm_self"], color="darkorange")
# axes[4].set_ylabel("Per-Post\nEngagement")

# # 6️⃣ Platform-normalized
# axes[5].plot(e["date"], e["norm_platform"], color="tomato")
# axes[5].set_ylabel("Per-Platform\nEngagement")

# # 7️⃣ Smoothed engagement
# axes[6].plot(e["date"], e["smoothed_norm_platform"], color="firebrick")
# axes[6].set_ylabel("Smoothed\nEngagement")
# axes[6].set_xlabel("Date")

# # --- tidy layout ---
# plt.tight_layout(rect=[0, 0, 1, 0.96])
# OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
# plt.savefig(OUT_PATH, dpi=300)
# plt.show()

# print(f"✅ Saved figure → {OUT_PATH}")



#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot all variants of the time-series signal for a single hashtag (#athleisure)
with aligned time axes.

Inputs
------
02-timeseries/hashtag_timeseries_norm_smooth.csv
02-timeseries/hashtag_timeseries_comments_norm_smooth.csv

Outputs
-------
04-figures/timeseries_compare/athleisure_all_signals.png
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# --- paths ---
BASE = Path(__file__).resolve().parents[1]
MENTION_PATH = BASE / "02-timeseries" / "hashtag_timeseries_norm_smooth.csv"
COMM_PATH = BASE / "02-timeseries" / "hashtag_timeseries_comments_norm_smooth.csv"
OUT_PATH = BASE / "04-figures" / "timeseries_compare" / "athleisure_all_signals.png"

# --- load ---
mention = pd.read_csv(MENTION_PATH, parse_dates=["date"])
comm = pd.read_csv(COMM_PATH, parse_dates=["date"])

# --- filter to hashtag ---
tag = "athleisure"
m = mention[mention["hashtag"].str.lower() == tag]
c = comm[comm["hashtag"].str.lower() == tag]


print("Mention columns:", m.columns.tolist())
print("Comment columns:", c.columns.tolist())

# --- count total unique influencers posting under this hashtag ---
user_col = None
for candidate in ["username", "author", "user", "poster"]:
    if candidate in m.columns:
        user_col = candidate
        break

if user_col:
    n_influencers = m[user_col].nunique()
    print(f"👤 Total unique influencers who ever posted under #{tag}: {n_influencers}")
else:
    print("⚠️ Could not find a user column to count unique influencers.")


# --- prepare figure ---
plt.style.use("seaborn-v0_8-whitegrid")
fig, axes = plt.subplots(6, 1, sharex=True, figsize=(11, 9))
fig.suptitle(f"Time-Series Signals for #{tag}", fontsize=14, weight="bold")

# 1️⃣ Raw mention counts
axes[0].plot(m["date"], m["mentions"], color="gray")
axes[0].set_ylabel("Raw\nMentions")

# 2️⃣ Normalized (per-K-posts)
col_norm = None
for ccol in ["mentions_per_1000posts", "mentions_per_post", "mentions_per_k_posts"]:
    if ccol in m.columns:
        col_norm = ccol
        break
axes[1].plot(m["date"], m[col_norm], color="steelblue")
axes[1].set_ylabel("Norm.\nMentions")

# 3️⃣ Smoothed (rolling mean)
col_smooth = None
for ccol in ["mentions_per_1000posts_ma7", "mentions_per_1000posts_ma15", "smoothed_mentions"]:
    if ccol in m.columns:
        col_smooth = ccol
        break
axes[2].plot(m["date"], m[col_smooth], color="navy")
axes[2].set_ylabel("Smoothed\nMentions")

# 4️⃣ Raw comments (sum of comment events per day)
axes[3].plot(c["date"], c["total_comments"], color="orange")
axes[3].set_ylabel("Raw\nComments")

# 5️⃣ Platform-normalized comment signal
axes[4].plot(c["date"], c["norm_platform"], color="tomato")
axes[4].set_ylabel("Per-Platform\nComments")

# 6️⃣ Smoothed (rolling mean) normalized comment signal
axes[5].plot(c["date"], c["smoothed_norm_platform"], color="firebrick")
axes[5].set_ylabel("Smoothed\nComments")
axes[5].set_xlabel("Date")

# --- tidy layout ---
plt.tight_layout(rect=[0, 0, 1, 0.96])
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT_PATH, dpi=300)
plt.show()

print(f"✅ Saved figure → {OUT_PATH}")
