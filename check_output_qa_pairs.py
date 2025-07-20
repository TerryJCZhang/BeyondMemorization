#!/usr/bin/env python3
"""
Script to calculate total QA pairs from log files and provide breakdowns by category and month.
Includes functionality to search for QA pair logs recursively in a directory structure.
"""

import re
import os
import subprocess
from collections import defaultdict
from pathlib import Path


def search_qa_logs(output_dir="./output"):
    """
    Search for QA pair logs recursively in directory structure.
    grep -ir "Total QA pairs available" --include="monthly_qa_pipeline.log" ./output | sort
    """

    matching_lines = []

    try:
        # Walk through directory structure
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                if file == "monthly_qa_pipeline.log":
                    file_path = os.path.join(root, file)

                    try:
                        with open(
                            file_path, "r", encoding="utf-8", errors="ignore"
                        ) as f:
                            for line_num, line in enumerate(f, 1):
                                if "Total QA pairs available" in line:
                                    # Format similar to grep output: filepath:line_content
                                    formatted_line = f"{file_path}:{line.strip()}"
                                    matching_lines.append(formatted_line)

                    except Exception as e:
                        print(f"Warning: Could not read {file_path}: {e}")

    except Exception as e:
        print(f"Error searching directory {output_dir}: {e}")
        return []

    # Sort the results (equivalent to piping to sort)
    matching_lines.sort()

    print(f"Found {len(matching_lines)} matching lines in {output_dir}")

    return matching_lines


def parse_qa_logs_from_lines(log_lines):

    total_qa_pairs = 0
    category_totals = defaultdict(int)
    monthly_totals = defaultdict(int)

    print(f"Processing {len(log_lines)} log entries")

    for line in log_lines:
        line = line.strip()

        qa_match = re.search(r"Total QA pairs available: (\d+)", line)
        if qa_match:
            count = int(qa_match.group(1))
            total_qa_pairs += count

            category_match = re.search(r"cs\.([A-Z]+)", line)
            if category_match:
                category = category_match.group(1)
                category_totals[category] += count

            date_match = re.search(r"(\d{4})/(\d{2})", line)
            if date_match:
                year, month = date_match.groups()
                month_key = f"{year}-{month}"
                monthly_totals[month_key] += count

    return {
        "total": total_qa_pairs,
        "by_category": dict(category_totals),
        "by_month": dict(monthly_totals),
    }


def display_results(results):
    """Display the calculation results in a formatted way."""

    if not results:
        return

    print(f"\n{'='*50}")
    print(f"QA PAIRS ANALYSIS RESULTS")
    print(f"{'='*50}")

    print(f"\nTotal QA pairs across all logs: {results['total']}")

    print(f"\nBreakdown by category:")
    print(f"{'-'*30}")
    sorted_categories = sorted(
        results["by_category"].items(), key=lambda x: x[1], reverse=True
    )
    for category, total in sorted_categories:
        print(f"cs.{category}: {total:>3} pairs")

    print(f"\nBreakdown by month:")
    print(f"{'-'*20}")
    sorted_months = sorted(results["by_month"].items())
    for month, total in sorted_months:
        print(f"{month}: {total:>3} pairs")


def main():
    print("QA Pairs Calculator")
    print("==================")
    output_dir = "./output"
    print(f"Searching for QA logs in {output_dir}...")
    log_lines = search_qa_logs(output_dir)

    if not log_lines:
        print("No matching log files found in ./output directory.")
        print(
            "Make sure the ./output directory exists and contains monthly_qa_pipeline.log files."
        )
        return

    # print(f"Found {len(log_lines)} matching log entries")
    # print("\nPreview of found entries:")
    # for i, line in enumerate(log_lines[:5]):
    #     print(f"  {line}")
    # if len(log_lines) > 5:
    #     print(f"  ... and {len(log_lines) - 5} more entries")
    results = parse_qa_logs_from_lines(log_lines)
    display_results(results)


if __name__ == "__main__":
    main()
