#!/usr/bin/env python3
"""
Script to count QA pairs in all 'qa_pairs' folders under a root directory.
Reports totals per category, year, and month, based on folder structure.
"""

import os
import json
from collections import defaultdict
from datasets import load_from_disk

SUBCAT_EXPANSIONS = {
    "physics": [
        "gr-qc", "math-ph", "nlin.SI", "physics.comp-ph", "physics.flu-dyn"
    ]
}

SUBCAT_PARENT = {
    subcat: category
    for category, subcats in SUBCAT_EXPANSIONS.items()
    for subcat in subcats
}

def get_category(topic: str) -> str:
    """
    Return the parent category for a topic (subcategory or top-level).
    Fallback: if unknown but contains a dot, use the part before the dot.
    Otherwise return the topic itself (or raise, your choice).
    """
    topic = topic.strip()
    if topic in SUBCAT_PARENT:
        return SUBCAT_PARENT[topic]
    if "." in topic:
        return topic.split(".", 1)[0]
    # Choose behavior: return as-is OR raise
    # raise KeyError(f"Unknown topic: {topic}")
    return topic

def find_qa_pairs_dirs(root_dir):
    """Recursively find all directories named 'qa_pairs'."""
    qa_dirs = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if os.path.basename(dirpath) == "qa_pairs":
            qa_dirs.append(dirpath)
    return qa_dirs

def parse_metadata_from_path(path, root_dir):
    """
    Extract category, year, and month from the path.
    Assumes structure: root/category/year/month/qa_pairs
    """
    rel = os.path.relpath(path, root_dir)
    parts = rel.split(os.sep)
    # Example: ['cs.AI', '2024', '06', 'qa_pairs']
    category, year, month = None, None, None
    if len(parts) >= 4:
        category, year, month = parts[-4], parts[-3], parts[-2]
    return category, year, month

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Count QA pairs in all qa_pairs folders under a root directory.")
    parser.add_argument("--input", type=str, default="output", help="Root directory to search for qa_pairs folders")
    parser.add_argument("--json_out", type=str, default="qa_summary.json", help="Optional: Path to save JSON summary")
    args = parser.parse_args()

    root_dir = args.input
    qa_dirs = find_qa_pairs_dirs(root_dir)
    if not qa_dirs:
        print(f"No qa_pairs folders found in {root_dir}")
        return

    summary = {
        "total": 0,
        "by_category": defaultdict(int),
        "by_year": defaultdict(int),
        "by_month": defaultdict(int),
        "details": []
    }

    for qa_dir in qa_dirs:
        try:
            ds = load_from_disk(qa_dir)
            count = len(ds)
        except Exception as e:
            print(f"Could not load dataset at {qa_dir}: {e}")
            continue

        category, year, month = parse_metadata_from_path(qa_dir, root_dir)
        summary["total"] += count
        if category:
            summary["by_category"][category] += count
        if year:
            summary["by_year"][year] += count
        if year and month:
            summary["by_month"][f"{year}-{month}"] += count
        summary["details"].append({
            "path": qa_dir,
            "category": category,
            "year": year,
            "month": month,
            "count": count
        })

    # Convert defaultdicts to dicts for JSON serialization
    summary["by_category"] = dict(summary["by_category"])
    summary["by_year"] = dict(summary["by_year"])
    summary["by_month"] = dict(summary["by_month"])

    # Print summary
    print("\n========== QA PAIRS SUMMARY ==========")
    print(f"Total QA pairs: {summary['total']}\n")

    print("Breakdown by category:")
    for cat, cnt in sorted(summary["by_category"].items(), key=lambda x: x[1], reverse=True):
        print(f"  {cat}: {cnt}")

    print("\nBreakdown by year:")
    for yr, cnt in sorted(summary["by_year"].items()):
        print(f"  {yr}: {cnt}")

    print("\nBreakdown by month:")
    for mo, cnt in sorted(summary["by_month"].items()):
        print(f"  {mo}: {cnt}")

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSaved summary to {args.json_out}")

if __name__ == "__main__":
    main()