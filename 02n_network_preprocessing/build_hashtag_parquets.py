#!/usr/bin/env python3
"""
Build propagation edges & daily snapshots from three CSVs using DuckDB.

Inputs:
  --hashtags_csv   path to hashtags_by_post.csv
  --comments_csv   path to hashtags_by_post_comments.csv  (optional if you did not extract comments)
  --tags_csv       path to hashtags_by_post_tags.csv      (optional if you did not extract tags)

Options:
  --out_dir        where to write parquet outputs (default: current directory)
  --hashtags       comma-separated list to filter (e.g. "strawberryblonde,balayage"); default = ALL
  --exposure_clamp_hours   clamp comment exposure time to at most N hours after the post (default: 72)
  --threads        DuckDB threads to use (default: 4)

Outputs (Parquet):
  user_hashtag_first_post.parquet
  exposures.parquet
  edges.parquet
  snapshots.parquet

Use format:
python /Users/xiaoqiao/Documents/GitHub/yu-4yp/Ins_dataset/build_hashtags_network.py \
  --hashtags_csv   /Users/xiaoqiao/Documents/GitHub/yu-4yp/Ins_dataset/hashtags_by_post_full_users.csv \
  --comments_csv   /Users/xiaoqiao/Documents/GitHub/yu-4yp/Ins_dataset/hashtags_by_post_full_users_comments.csv \
  --tags_csv       /Users/xiaoqiao/Documents/GitHub/yu-4yp/Ins_dataset/hashtags_by_post_full_users_tags.csv \
  --out_dir        /Users/xiaoqiao/Documents/GitHub/yu-4yp/Ins_dataset/networks_inmyfeelings \
  --hashtags       inmyfeelings \
  --exposure_clamp_hours   360
  

"""

import argparse, os, sys, duckdb

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="YAML config path")
    ap.add_argument("--hashtag", required=True, help="Single hashtag (no #)")
    ap.add_argument("--out_dir", help="Override output directory (optional)") 
    args = ap.parse_args()

    import yaml
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    hashtags_csv = os.path.abspath(cfg["hashtags_csv"])
    comments_csv = os.path.abspath(cfg.get("comments_csv", ""))
    tags_csv     = os.path.abspath(cfg.get("tags_csv", ""))
    clamp_sec    = cfg.get("exposure_clamp_hours", 72) * 3600
    threads      = cfg.get("threads", 4)
    if args.out_dir:
        out_dir = args.out_dir
    else:
        out_dir = os.path.join(cfg["base_dir"], "outputs", args.hashtag)
    os.makedirs(out_dir, exist_ok=True)
    # hashtag_filter = [args.hashtag.lower()]


    # Connect to an in-memory DuckDB; set threads
    con = duckdb.connect(database=":memory:")
    # con.execute(f"PRAGMA threads={args.threads};")
    con.execute(f"PRAGMA threads={threads};")
    # Helpful if you ever need to cap memory (uncomment & tune):
    # con.execute("PRAGMA memory_limit='8GB';")

    # Normalize optional filters
    # hashtag_filter = [h.strip().lower() for h in args.hashtags.split(",") if h.strip()]  # may be empty
    hashtag_filter = [args.hashtag.lower()]

    # Build WHERE clause for hashtag filtering
    where_hashtag = ""
    if hashtag_filter:
        in_list = ",".join("'" + h.replace("'", "''") + "'" for h in hashtag_filter)
        where_hashtag = f"WHERE lower(hashtag) IN ({in_list})"

    # Register CSVs on the fly via read_csv_auto (DuckDB scans them efficiently)
    # hashtags_csv = os.path.abspath(args.hashtags_csv)
    # comments_csv = os.path.abspath(args.comments_csv) if args.comments_csv else ""
    # tags_csv     = os.path.abspath(args.tags_csv) if args.tags_csv else ""

    # clamp_sec = max(0, args.exposure_clamp_hours) * 3600
    clamp_sec = clamp_sec

    # 1) First adoption time per (user, hashtag)
    con.execute(f"""
        CREATE OR REPLACE TABLE user_hashtag_first_post AS
        SELECT
            username AS user,
            lower(hashtag) AS hashtag,
            MIN(CAST(epoch_seconds AS BIGINT)) AS first_post_epoch
        FROM read_csv_auto('{hashtags_csv}')
        {where_hashtag}
        GROUP BY 1,2;
    """)
    con.execute(f"COPY user_hashtag_first_post TO '{os.path.join(out_dir, 'user_hashtag_first_post.parquet')}' (FORMAT PARQUET);")

    # 2) Build exposures (comments, tags), then union
    # Base posts table for joins
    con.execute(f"""
        CREATE OR REPLACE TEMP VIEW h AS
        SELECT
            post_id, shortcode, username AS author,
            lower(hashtag) AS hashtag,
            CAST(epoch_seconds AS BIGINT) AS post_epoch
        FROM read_csv_auto('{hashtags_csv}')
        {where_hashtag};
    """)

    # Start with an empty exposures table to UNION into safely
    con.execute("""
        CREATE OR REPLACE TABLE exposures (
            hashtag TEXT,
            A TEXT,
            B TEXT,
            exposure_epoch BIGINT,
            post_epoch BIGINT,
            post_id TEXT,
            shortcode TEXT,
            exposure_type TEXT
        );
    """)

    # 2a) Comment exposures (if provided): A(author of a hashtagged post) exposes B(commenter)
    if comments_csv:
        con.execute(f"""
            CREATE OR REPLACE TEMP VIEW c AS
            SELECT
                post_id, shortcode, author, commenter,
                CAST(comment_epoch AS BIGINT) AS comment_epoch
            FROM read_csv_auto('{comments_csv}');
        """)
        # Optional clamp: exposure cannot be later than post_epoch + clamp_sec
        # If clamp_sec == 0, LEAST(comment_epoch, post_epoch) == post_epoch (i.e., clamp to post time).
        # If you want "no clamp", pass a huge number (e.g., --exposure_clamp_hours 100000).
        con.execute(f"""
            INSERT INTO exposures
            SELECT
                h.hashtag,
                h.author AS A,
                c.commenter AS B,
                CASE WHEN {clamp_sec} = 0
                     THEN h.post_epoch
                     ELSE LEAST(c.comment_epoch, h.post_epoch + {clamp_sec})
                END AS exposure_epoch,
                h.post_epoch,
                h.post_id, h.shortcode,
                'comment' AS exposure_type
            FROM h
            JOIN c USING (post_id, shortcode, author)
            WHERE c.commenter IS NOT NULL AND c.commenter <> h.author;
        """)

    # 2b) Tag exposures (if provided): exposure time = post time
    if tags_csv:
        con.execute(f"""
            CREATE OR REPLACE TEMP VIEW t AS
            SELECT
                CAST(post_epoch AS BIGINT) AS post_epoch,
                post_id, shortcode, author, tagged_user
            FROM read_csv_auto('{tags_csv}');
        """)
        con.execute("""
            INSERT INTO exposures
            SELECT
                h.hashtag,
                h.author AS A,
                t.tagged_user AS B,
                h.post_epoch AS exposure_epoch,
                h.post_epoch,
                h.post_id, h.shortcode,
                'tag' AS exposure_type
            FROM h
            JOIN t USING (post_id, shortcode, post_epoch, author)
            WHERE t.tagged_user IS NOT NULL AND t.tagged_user <> h.author;
        """)

    # Write exposures
    con.execute(f"COPY exposures TO '{os.path.join(out_dir, 'exposures.parquet')}' (FORMAT PARQUET);")

    # 3) Propagation edges A→B: B adopted after exposure for the same hashtag
    con.execute("""
        CREATE OR REPLACE TEMP VIEW ad AS
        SELECT * FROM user_hashtag_first_post;
    """)
    con.execute("""
        CREATE OR REPLACE TABLE edges AS
        WITH e AS (
            SELECT
                hashtag, A, B,
                exposure_epoch,
                post_epoch,
                post_id, shortcode,
                exposure_type
            FROM exposures
        )
        SELECT
            e.hashtag,
            e.A, e.B,
            MIN(e.exposure_epoch)            AS first_exposure_epoch,
            adA.first_post_epoch             AS A_adopt_epoch,
            adB.first_post_epoch             AS B_adopt_epoch,
            ANY_VALUE(e.exposure_type)       AS exposure_type
        FROM e
        JOIN ad AS adA ON adA.user = e.A AND adA.hashtag = e.hashtag
        JOIN ad AS adB ON adB.user = e.B AND adB.hashtag = e.hashtag
        WHERE adB.first_post_epoch > e.exposure_epoch
        GROUP BY 1,2,3,5,6;
    """)
    con.execute(f"COPY edges TO '{os.path.join(out_dir, 'edges.parquet')}' (FORMAT PARQUET);")

    # 4) Daily snapshots: new edges and adopters per day (cumulative edges too)
    con.execute("""
        CREATE OR REPLACE TABLE snapshots AS
        WITH edges_ts AS (
            SELECT
                hashtag, A, B,
                B_adopt_epoch,
                CAST(to_timestamp(B_adopt_epoch) AS DATE) AS adopt_day
            FROM edges
        )
        SELECT
            hashtag,
            adopt_day,
            COUNT(*)                      AS new_edges,
            COUNT(DISTINCT B)             AS new_adopters,
            SUM(COUNT(*)) OVER (
                PARTITION BY hashtag ORDER BY adopt_day
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            )                             AS cum_edges
        FROM edges_ts
        GROUP BY 1,2
        ORDER BY hashtag, adopt_day;
    """)
    con.execute(f"COPY snapshots TO '{os.path.join(out_dir, 'snapshots.parquet')}' (FORMAT PARQUET);")

    # Also nice to have: quick counts printed to console
    n_adopt = con.execute("SELECT COUNT(*) FROM user_hashtag_first_post;").fetchone()[0]
    n_exp   = con.execute("SELECT COUNT(*) FROM exposures;").fetchone()[0]
    n_edges = con.execute("SELECT COUNT(*) FROM edges;").fetchone()[0]
    n_snap  = con.execute("SELECT COUNT(*) FROM snapshots;").fetchone()[0]

    print(f"✓ user_hashtag_first_post.parquet rows: {n_adopt}")
    print(f"✓ exposures.parquet rows: {n_exp}")
    print(f"✓ edges.parquet rows: {n_edges}")
    print(f"✓ snapshots.parquet rows: {n_snap}")
    print(f"Done. Files written to: {out_dir}")

if __name__ == "__main__":
    main()
