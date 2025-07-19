#!/usr/bin/env python3
# run:  python monthly_qa_pipeline.py --year 2024 --start 1 --end 6 -c cs,math

import argparse, os, subprocess, shlex, sys
import os
from pathlib import Path
from datasets import load_from_disk

TARGET_QA = 20 # Target number of QA pairs to generate per month
PYTHON = os.environ.get("PYTHON") if os.environ.get("PYTHON") else "python"

def count_rows(path: Path) -> int:
    try:
        return len(load_from_disk(str(path)))
    except Exception:
        return 0

def run(cmd: str, log_fh):
    proc = subprocess.run(shlex.split(cmd),
                          stdout=log_fh,
                          stderr=subprocess.STDOUT)
    if proc.returncode:
        log_fh.write(f"\nCommand failed: {cmd}\n")
        sys.exit(proc.returncode)

def process_month(topic: str, yr: int, mo: int, papers_step: int, output_root: Path, main_log_fh, is_subcategory: bool):
    iter_count = 0
    base = output_root / topic / str(yr) / f"{mo:02d}"
    papers_dir = base / "papers"
    qa_dir     = base / "qa_pairs"

    qa_prev = count_rows(qa_dir)
    paper_prev = count_rows(papers_dir)

    while qa_prev < TARGET_QA:
        main_log_fh.write(f"\n Starting Iteration #{iter_count} for {topic.upper()} - {yr}-{mo:02d} \n")
        main_log_fh.write("────────────────────────────────────────────────────────────\n")
        iter_count += 1

        topic_arg = "--subcategories" if is_subcategory else "--categories"

        # Define step-specific log files
        arxiv_log = open(base / "arxiv_fetch.log", "a", encoding="utf-8")
        latex_log = open(base / "latex_extract.log", "a", encoding="utf-8")
        thm_log   = open(base / "theorem_extract.log", "a", encoding="utf-8")
        qa_log    = open(base / "qa_generate.log", "a", encoding="utf-8")

        # Step 1: Fetch papers
        run(
            f"{PYTHON} helpers/arxiv_retriever.py "
            f"--year {yr} --month {mo} {topic_arg} {topic} "
            f"--output {output_root} --max-results {papers_step} --append",
            arxiv_log
        )

        paper_now = count_rows(papers_dir)
        new_papers = paper_now - paper_prev
        paper_prev = paper_now
        main_log_fh.write(f"New papers fetched: {new_papers}\n")
        main_log_fh.write(f"Total papers available: {paper_now}\n")

        # Step 2: Extract LaTeX
        run(f"{PYTHON} helpers/extract_latex_text.py --input {base} --output {base} --append", latex_log)

        # Step 3: Extract Theorems
        run(f"{PYTHON} helpers/extract_theorems.py --input {base} --output {base} --append", thm_log)

        # Step 4: Generate QA
        run(f"{PYTHON} helpers/generate_qa.py --input {base} --output {base} --append", qa_log)

        qa_now = count_rows(qa_dir)
        new_qas = qa_now - qa_prev
        qa_prev = qa_now
        main_log_fh.write(f"New QA pairs generated: {new_qas}\n")
        main_log_fh.write(f"Total QA pairs available: {qa_now}\n")

        if new_papers == 0 and new_qas == 0:
            main_log_fh.write(f"\nNo new papers or QAs, assuming month exhausted.\n")
            break

        # Close logs for this iteration
        arxiv_log.close()
        latex_log.close()
        thm_log.close()
        qa_log.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--start", type=int, default=1, help="start month (1-12)")
    ap.add_argument("--end",   type=int, default=12, help="end month (1-12)")
    ap.add_argument('-c', '--categories', type=str, default='cs, math, physics, q-fin, q-bio', 
                        help='List of arXiv categories to search (default: None). Can only be a main category with subcategories (e.g., cs, math)')
    ap.add_argument('-sc', "--subcategories", type=str, default=None,
                        help="List of specific subcategories to search (e.g., cs.IT, math.AG).")
    ap.add_argument("--output_root", type=str, default='output', help='Output directory for the dataset')
    ap.add_argument("--papers-step", type=int, default=100,
                    help="how many new papers to request each loop")
    args = ap.parse_args()

    if args.start < 1 or args.start > 12 or args.end < 1 or args.end > 12:
        print("Error: start and end months must be between 1 and 12.")
        sys.exit(1)

    if args.categories is None and args.subcategories is None:
        print("Error: You must specify at least one category or subcategory.")
        sys.exit(1)

    cats = [c.strip() for c in args.categories.split(",") if c.strip()] if args.categories else []
    subcats = [c.strip() for c in args.subcategories.split(",") if c.strip()] if args.subcategories else []

    for topic in cats + subcats:
        for mo in range(args.start, args.end + 1):

            log_path = (Path(args.output_root) / topic / str(args.year) / f"{mo:02d}" / "monthly_qa_pipeline.log")
            log_path.parent.mkdir(parents=True, exist_ok=True)

            with open(log_path, "a", encoding="utf-8") as log_fh:
                log_fh.write(f"\n=== {topic} {args.year}-{mo:02d} ===\n")
                process_month(
                    topic, args.year, mo, args.papers_step, Path(args.output_root), log_fh, topic in subcats
                )
                log_fh.write(f"Finished {topic} {args.year}-{mo:02d}\n")

"""
Example:
python monthly_qa_pipeline.py \
  --year 2024 \
  --start 1 \
  --end 3 \
  --categories cs,math \
  --subcategories cs.CL,math.AG \
  --output_root output \
  --papers-step 100
"""

if __name__ == "__main__":
    main()

'''
python monthly_qa_pipeline.py --year 2024 --start 5 --end 12
'''