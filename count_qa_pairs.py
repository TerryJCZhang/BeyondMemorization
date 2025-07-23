#!/usr/bin/env python3
"""
Script to count QA pairs in all 'qa_pairs' folders under a root directory.
Warning : Only works with a structure like root/category/year/month/qa_pairs.
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
    parser = argparse.ArgumentParser(description="Count QA pairs in all qa_pairs folders under a root directory. " \
                                                    "Only works with a structure like root/category/year/month/qa_pairs.")
    parser.add_argument("--input", type=str, default="output", help="Root directory to search for qa_pairs folders")
    parser.add_argument("--json_out", type=str, default=None, help="Optional: Path to save JSON summary")
    parser.add_argument("--show_subcats", type=bool, default=False, help="Display subcategories in the summary")
    args = parser.parse_args()

    root_dir = args.input
    qa_dirs = find_qa_pairs_dirs(root_dir)
    if not qa_dirs:
        print(f"No qa_pairs folders found in {root_dir}")
        return

    # Structure: {month: {category: {subcategory: count, ...}, ...}, ...}
    summary = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    month_totals = defaultdict(int)
    cat_totals = defaultdict(lambda: defaultdict(int))  # month -> cat -> total

    for qa_dir in qa_dirs:
        try:
            ds = load_from_disk(qa_dir)
            count = len(ds)
        except Exception as e:
            print(f"Could not load dataset at {qa_dir}: {e}")
            continue

        subcat, year, month = parse_metadata_from_path(qa_dir, root_dir)
        if not (year and month and subcat):
            continue
        month_key = f"{year}-{month}"
        cat = get_category(subcat)

        summary[month_key][cat][subcat] += count
        month_totals[month_key] += count
        cat_totals[month_key][cat] += count

    # Print summary
    print("\n========== QA PAIRS SUMMARY ==========")
    for month in sorted(summary.keys()):
        print(f"\n[ {month} ]  Total QA pairs: {month_totals[month]}")
        for cat in sorted(summary[month].keys()):
            print(f"  - {cat}: {cat_totals[month][cat]}")
            if args.show_subcats:
                for subcat in sorted(summary[month][cat].keys()):
                    print(f"      * {subcat}: {summary[month][cat][subcat]}")

    # Save as JSON if requested
    if args.json_out:

        # Convert defaultdicts to dicts for JSON serialization
        def dictify(d):
            if isinstance(d, defaultdict):
                d = {k: dictify(v) for k, v in d.items()}
            return d
        
        out_json = {
            "by_month": dictify(summary),
            "month_totals": dict(month_totals),
            "cat_totals": {m: dict(c) for m, c in cat_totals.items()}
        }

        for month in summary:
            out_json["by_month"][month] = {}
            for cat in summary[month]:
                if args.show_subcats:
                    out_json["by_month"][month][cat] = dict(summary[month][cat])
                else:
                    # Only save category totals, not subcat breakdown
                    out_json["by_month"][month][cat] = cat_totals[month][cat]

        with open(args.json_out, "w") as f:
            json.dump(out_json, f, indent=2)
        print(f"\nSaved summary to {args.json_out}")

if __name__ == "__main__":
    main()

'''
python count_qa_pairs.py --input output --json_out qa_count.json --show_subcats False
'''