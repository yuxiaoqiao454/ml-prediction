#!/usr/bin/env python3
"""
03_networks/scripts/run_all_hashtags.py

Top-level orchestrator that loops over a master hashtag list and runs all network pipeline stages in order.

Stages:
  A. Build parquets (02n_network_preprocessing/build_hashtag_parquets.py)
  B. Build bipartite networks (03_networks/scripts/build_bipartite_all_windows.py)
  C. Run projection (03_networks/scripts/run_projection_all_windows.py)
  D. Run clustering (03_networks/scripts/run_clustering_all_windows.py)

Usage:
  python 03_networks/scripts/run_all_hashtags.py --config 03_networks/configs/default.yaml --limit 3
"""

import argparse
import subprocess
import sys
import os
import time
from pathlib import Path
from datetime import datetime
import pandas as pd
import shutil

# Add 03_networks directory to path so we can import pipeline modules
# pipeline/ is located at 03_networks/pipeline/
networks_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(networks_dir))

from pipeline.utils.io import load_config, ensure_dir
from pipeline.utils.logging_utils import log_message, log_summary_csv


EXPECTED_PARQUET_FILES = [
    "user_hashtag_first_post.parquet",
    "exposures.parquet",
    "edges.parquet",
    "snapshots.parquet"
]

SKIP_CHECK_FILES = [
    "exposures.parquet",
    "user_hashtag_first_post.parquet"
]

SUMMARY_HEADER = [
    "hashtag",
    "stage_parquets",
    "stage_bipartite",
    "stage_projection",
    "stage_clustering",
    "total_runtime_sec",
    "timestamp"
]


def parse_args():
    parser = argparse.ArgumentParser(description="Run all network pipeline stages for all hashtags")
    parser.add_argument("--config", default="03_networks/configs/default.yaml", help="Path to config YAML")
    parser.add_argument("--limit", type=int, help="Process only first N hashtags")
    parser.add_argument("--skip_existing", action="store_true", help="Skip parquet build if key files exist")
    parser.add_argument("--dry_run", action="store_true", help="Print planned commands without executing")
    return parser.parse_args()


def run_command(cmd, log_file, dry_run=False):
    """
    Execute a subprocess command and log output.
    Returns (success: bool, returncode: int)
    """
    if dry_run:
        print(f"  [DRY-RUN] would run: {' '.join(cmd)}")
        return True, 0
    
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    
    # Log stdout and stderr
    with open(log_file, 'a') as f:
        f.write(f"\n{'='*80}\n")
        f.write(f"Command: {' '.join(cmd)}\n")
        f.write(f"Return code: {result.returncode}\n")
        f.write(f"{'='*80}\n")
        if result.stdout:
            f.write("STDOUT:\n")
            f.write(result.stdout)
            f.write("\n")
        if result.stderr:
            f.write("STDERR:\n")
            f.write(result.stderr)
            f.write("\n")
    
    return result.returncode == 0, result.returncode


def sync_parquet_files(source_dir, dest_dir, log_file, dry_run=False):
    """
    Copy parquet files from source to destination.
    Returns list of files that were copied/would be copied.
    """
    source_path = Path(source_dir)
    dest_path = Path(dest_dir)
    
    copied_files = []
    missing_files = []
    
    for filename in EXPECTED_PARQUET_FILES:
        src_file = source_path / filename
        dst_file = dest_path / filename
        
        if src_file.exists():
            if dry_run:
                print(f"  [DRY-RUN] would copy: {src_file} → {dst_file}")
                copied_files.append(filename)
            else:
                ensure_dir(dest_path)
                shutil.copy2(src_file, dst_file)
                log_message(log_file, f"Copied {filename} from {source_path} to {dest_path}")
                copied_files.append(filename)
        else:
            missing_files.append(filename)
    
    if missing_files and not dry_run:
        for filename in missing_files:
            log_message(log_file, f"⚠️ missing {filename} after build_hashtag_parquets.py")
    
    return copied_files, missing_files


def should_skip_parquet_stage(net_out_dir):
    """
    Check if both key files exist to determine if parquet stage can be skipped.
    """
    net_path = Path(net_out_dir)
    if not net_path.exists():
        return False
    
    for filename in SKIP_CHECK_FILES:
        if not (net_path / filename).exists():
            return False
    
    return True


def process_hashtag(tag, cfg, args, log_file):
    """
    Run all pipeline stages for a single hashtag.
    Returns dict with stage results and runtime.
    """
    start_time = time.time()
    
    # Compute paths based on actual config structure
    base_dir = Path(cfg.get("base_dir", "03_networks/data"))
    preprocess_outputs = base_dir.parent.parent / "02n_network_preprocessing" / "outputs"
    
    pre_out = preprocess_outputs / f"networks_{tag}"
    net_out = base_dir / "parquets"/ f"networks_{tag}"
    
    results = {
        "hashtag": tag,
        "stage_parquets": "pending",
        "stage_bipartite": "pending",
        "stage_projection": "pending",
        "stage_clustering": "pending",
        "total_runtime_sec": 0.0,
        "timestamp": datetime.now().isoformat()
    }
    
    config_path = args.config
    
    # ============================================================================
    # STAGE A: Build parquets
    # ============================================================================
    skip_parquet = args.skip_existing and should_skip_parquet_stage(net_out)
    
    if skip_parquet:
        print(f"  parquet... skipped", end=" | ", flush=True)
        log_message(log_file, f"[{tag}] Stage A (parquet) skipped - files exist")
        results["stage_parquets"] = "skipped"
    else:
        print(f"  parquet...", end=" ", flush=True)
        
        # cmd_parquet = [
        #     sys.executable,
        #     "02n_network_preprocessing/build_hashtag_parquets.py",
        #     "--config", config_path,
        #     "--hashtag", tag,
        #     "--out_dir", str(pre_out)
        # ]
        # Use the preprocess-specific config
        preprocess_config = "02n_network_preprocessing/config/preprocess.yaml"
        cmd_parquet = [
            sys.executable,
            "02n_network_preprocessing/build_hashtag_parquets.py",
            "--config", preprocess_config,  # ← Use preprocess config
            "--hashtag", tag,
            "--out_dir", str(pre_out)
        ]
        
        success, retcode = run_command(cmd_parquet, log_file, args.dry_run)
        
        if not success:
            print(f"failed", end=" | ", flush=True)
            log_message(log_file, f"[{tag}] Stage A (parquet) FAILED with code {retcode}")
            results["stage_parquets"] = "failed"
            results["stage_bipartite"] = "skipped"
            results["stage_projection"] = "skipped"
            results["stage_clustering"] = "skipped"
            results["total_runtime_sec"] = time.time() - start_time
            return results
        
        # Sync files
        if args.dry_run:
            print(f"  [DRY-RUN] would copy parquet files from {pre_out} → {net_out}")
        
        copied, missing = sync_parquet_files(pre_out, net_out, log_file, args.dry_run)
        
        print(f"ok", end=" | ", flush=True)
        log_message(log_file, f"[{tag}] Stage A (parquet) completed - copied {len(copied)} files")
        results["stage_parquets"] = "ok"
    
    # ============================================================================
    # STAGE B: Build bipartite networks
    # ============================================================================
    print(f"bipartite...", end=" ", flush=True)
    
    cmd_bipartite = [
        sys.executable,
        "03_networks/scripts/build_bipartite_all_windows.py",
        "--config", config_path,
        "--hashtag", tag
    ]
    
    success, retcode = run_command(cmd_bipartite, log_file, args.dry_run)
    
    if not success:
        print(f"failed", end=" | ", flush=True)
        log_message(log_file, f"[{tag}] Stage B (bipartite) FAILED with code {retcode}")
        results["stage_bipartite"] = "failed"
        results["stage_projection"] = "skipped"
        results["stage_clustering"] = "skipped"
        results["total_runtime_sec"] = time.time() - start_time
        return results
    
    print(f"ok", end=" | ", flush=True)
    log_message(log_file, f"[{tag}] Stage B (bipartite) completed")
    results["stage_bipartite"] = "ok"
    
    # ============================================================================
    # STAGE C: Run projection
    # ============================================================================
    print(f"projection...", end=" ", flush=True)
    
    cmd_projection = [
        sys.executable,
        "03_networks/scripts/run_projection_all_windows.py",
        "--config", config_path,
        "--hashtag", tag
    ]
    
    success, retcode = run_command(cmd_projection, log_file, args.dry_run)
    
    if not success:
        print(f"failed", end=" | ", flush=True)
        log_message(log_file, f"[{tag}] Stage C (projection) FAILED with code {retcode}")
        results["stage_projection"] = "failed"
        results["stage_clustering"] = "skipped"
        results["total_runtime_sec"] = time.time() - start_time
        return results
    
    print(f"ok", end=" | ", flush=True)
    log_message(log_file, f"[{tag}] Stage C (projection) completed")
    results["stage_projection"] = "ok"
    
    # ============================================================================
    # STAGE D: Run clustering
    # ============================================================================
    print(f"clustering...", end=" ", flush=True)
    
    cmd_clustering = [
        sys.executable,
        "03_networks/scripts/run_clustering_all_windows.py",
        "--config", config_path,
        "--hashtag", tag
    ]
    
    success, retcode = run_command(cmd_clustering, log_file, args.dry_run)
    
    if not success:
        print(f"failed", end="  ", flush=True)
        log_message(log_file, f"[{tag}] Stage D (clustering) FAILED with code {retcode}")
        results["stage_clustering"] = "failed"
        results["total_runtime_sec"] = time.time() - start_time
        return results
    
    print(f"ok", end="  ", flush=True)
    log_message(log_file, f"[{tag}] Stage D (clustering) completed")
    results["stage_clustering"] = "ok"
    
    # Record final runtime
    results["total_runtime_sec"] = time.time() - start_time
    
    return results


def main():
    args = parse_args()
    
    # Load config
    cfg = load_config(args.config)
    
    # Setup paths based on actual config structure
    base_dir = Path(cfg.get("base_dir", "03_networks/data"))
    log_dir = base_dir.parent / "logs"
    ensure_dir(log_dir)
    
    log_file = log_dir / "run_all_hashtags.log"
    summary_csv = log_dir / "run_summary.csv"
    
    # Master list path
    masterlist_path = base_dir / "_meta" / "hashtags_masterlist.csv"
    if not masterlist_path.exists():
        print(f"❌ Master list not found: {masterlist_path}")
        sys.exit(1)
    
    df_master = pd.read_csv(masterlist_path)
    
    if "hashtag" not in df_master.columns:
        print(f"❌ Master list must have 'hashtag' column")
        sys.exit(1)
    
    # Clean hashtag list
    tags = df_master["hashtag"].dropna().str.lower().str.strip().tolist()
    
    if args.limit:
        tags = tags[:args.limit]
    
    # Header
    print(f"{'='*80}")
    print(f"run_all_hashtags: {len(tags)} tags")
    if args.dry_run:
        print(f"[DRY-RUN MODE - no actual execution]")
    print(f"{'='*80}")
    
    log_message(log_file, f"{'='*80}")
    log_message(log_file, f"run_all_hashtags started: {len(tags)} tags")
    log_message(log_file, f"Config: {args.config}")
    log_message(log_file, f"Dry run: {args.dry_run}")
    log_message(log_file, f"Skip existing: {args.skip_existing}")
    log_message(log_file, f"{'='*80}")
    
    # Process each hashtag
    all_results = []
    failed_count = 0
    
    for i, tag in enumerate(tags, 1):
        print(f"[{i}/{len(tags)}] [{tag}]", end=" ", flush=True)
        log_message(log_file, f"\n{'='*80}")
        log_message(log_file, f"Processing hashtag {i}/{len(tags)}: {tag}")
        log_message(log_file, f"{'='*80}")
        
        results = process_hashtag(tag, cfg, args, log_file)
        all_results.append(results)
        
        # Check if any stage failed
        has_failure = any(
            results[f"stage_{stage}"] == "failed"
            for stage in ["parquets", "bipartite", "projection", "clustering"]
        )
        
        if has_failure:
            failed_count += 1
            print(f"❌ ({results['total_runtime_sec']:.1f}s)")
        else:
            print(f"✅ ({results['total_runtime_sec']:.1f}s)")
        
        # Write to summary CSV (append mode)
        if not args.dry_run:
            log_summary_csv(summary_csv, results, SUMMARY_HEADER)
    
    # Final summary
    print(f"\n{'='*80}")
    print(f"→ Summary: {len(tags)} processed, {failed_count} with errors")
    if not args.dry_run:
        print(f"✓ {summary_csv} updated")
    print(f"{'='*80}")
    
    log_message(log_file, f"\n{'='*80}")
    log_message(log_file, f"run_all_hashtags completed")
    log_message(log_file, f"Total processed: {len(tags)}")
    log_message(log_file, f"Failed: {failed_count}")
    log_message(log_file, f"{'='*80}")


if __name__ == "__main__":
    main()