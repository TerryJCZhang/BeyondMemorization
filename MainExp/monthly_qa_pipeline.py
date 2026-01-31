#!/usr/bin/env python3
# run:  python monthly_qa_pipeline.py --year 2024 --start 1 --end 6 -c cs,math

import argparse, os, subprocess, shlex, sys
from typing import List, Tuple
from pathlib import Path
from datasets import load_from_disk

SUBCAT_TARGET_QA = 30  # Target number of QA pairs to generate per month per subcategory
TARGET_QA = 30  # Target number of QA pairs to generate per month per main category
PYTHON = os.environ.get("PYTHON") if os.environ.get("PYTHON") else "python"


SUBCATEGORY_EXPANSIONS = {
    "physics": [
        "gr-qc", "math-ph", "nlin.SI", "physics.comp-ph", "physics.flu-dyn"
    ]
}

PARENT_CATEGORY: dict[str, str] = {}
for parent, subs in SUBCATEGORY_EXPANSIONS.items():
    PARENT_CATEGORY[parent] = parent  # parent maps to itself
    for sub in subs:
        PARENT_CATEGORY[sub] = parent



def build_base_path(topic: str, yr: int, mo: int, output_root: Path) -> Path:
    """
    Return the directory path  output_root / <main_cat>/<topic>/YYYY/MM
    If `topic` itself is a main (top‑level) category, we skip the extra level.
    """
    main_cat = PARENT_CATEGORY.get(
        topic,
        topic.split(".")[0] if "." in topic else topic
    )
    if topic == main_cat:
        return output_root / topic / str(yr) / f"{mo:02d}"
    return output_root / main_cat / topic / str(yr) / f"{mo:02d}"

def build_jobs(cats: List[str], subcats: List[str]) -> List[Tuple[str, bool]]:
    """
    Expand categories→sub‑categories and de‑duplicate while preserving order.
    Each job is a tuple (topic, is_subcategory_bool).
    """
    jobs: List[Tuple[str, bool]] = []
    seen: set[str] = set()

    def _add(t: str, is_sub: bool):
        if t not in seen:
            seen.add(t)
            jobs.append((t, is_sub))

    # First pass: top‑level categories (treat_as_sub = False)
    for cat in cats:
        if cat in SUBCATEGORY_EXPANSIONS:
            for subcat in SUBCATEGORY_EXPANSIONS[cat]:
                _add(subcat, True)
        else:
            _add(cat, False)

    # Second pass: explicit subcategories
    for entry in subcats:
        if entry in SUBCATEGORY_EXPANSIONS:
            for subcat in SUBCATEGORY_EXPANSIONS[entry]:
                _add(subcat, True)
        else:
            _add(entry, True)

    return jobs

def count_rows(path: Path) -> int:
    try:
        return len(load_from_disk(str(path)))
    except Exception:
        return 0

def run(cmd: str, log_fh):
    parts = shlex.split(cmd, posix=(os.name != 'nt'))
    proc = subprocess.run(parts,
                          stdout=log_fh,
                          stderr=subprocess.STDOUT)
    if proc.returncode:
        log_fh.write(f"\nCommand failed: {cmd}\n")
        sys.exit(proc.returncode)

def process_month(topic: str, yr: int, mo: int, papers_step: int, output_root: Path, main_log_fh, is_subcategory: bool):
    iter_count = 0
    base = build_base_path(topic, yr, mo, output_root)
    papers_dir = base / "papers"
    qa_dir     = base / "qa_pairs"
    # Convert to POSIX strings to avoid back‑slash escaping issues on Windows
    base_str = base.as_posix()
    papers_dir_str = papers_dir.as_posix()

    qa_prev = count_rows(qa_dir)
    paper_prev = count_rows(papers_dir)

    target_num = SUBCAT_TARGET_QA if is_subcategory else TARGET_QA

    while qa_prev < target_num:
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
            f"--full-output-path {papers_dir_str} --max-results {papers_step} --append",
            arxiv_log
        )

        paper_now = count_rows(papers_dir)
        new_papers = paper_now - paper_prev
        paper_prev = paper_now
        main_log_fh.write(f"New papers fetched: {new_papers}\n")
        main_log_fh.write(f"Total papers available: {paper_now}\n")

        if new_papers == 0:
            main_log_fh.write(f"\nNo new papers, assuming month exhausted.\n")
            break

        # Step 2: Extract LaTeX
        run(f"{PYTHON} helpers/extract_latex_text.py --input {base_str} --output {base_str} --append", latex_log)

        # Step 3: Extract Theorems
        run(f"{PYTHON} helpers/extract_theorems.py --input {base_str} --output {base_str} --append", thm_log)

        # Step 4: Generate QA
        run(f"{PYTHON} helpers/generate_qa.py --input {base_str} --output {base_str} --append", qa_log)

        qa_now = count_rows(qa_dir)
        new_qas = qa_now - qa_prev
        qa_prev = qa_now
        main_log_fh.write(f"New QA pairs generated: {new_qas}\n")
        main_log_fh.write(f"Total QA pairs available: {qa_now}\n")

        

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
    ap.add_argument('-c', '--categories', type=str, default='math', 
                        help='List of arXiv categories to search (default: None). Can only be a main category with subcategories (e.g., cs, math)')
    ap.add_argument('-sc', "--subcategories", type=str, default='physics',
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

    jobs = build_jobs(cats, subcats)

    for topic, is_sub in jobs:
        for mo in range(args.start, args.end + 1):
            log_path = build_base_path(topic, args.year, mo, Path(args.output_root)) / "monthly_qa_pipeline.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)

            with open(log_path, "a", encoding="utf-8") as log_fh:
                log_fh.write(f"\n=== {topic} {args.year}-{mo:02d} ===\n")
                process_month(
                    topic, args.year, mo, args.papers_step, Path(args.output_root), log_fh, is_sub
                )
                log_fh.write(f"Finished {topic} {args.year}-{mo:02d}\n")

"""
Example:
python monthly_qa_pipeline.py \
  --year 2024 \
  --start 1 \
  --end 3 \
  --categories cs,math \
  --subcategories math-ph,math.AG \
  --output_root output \
  --papers-step 100
"""

if __name__ == "__main__":
    main()

'''
python monthly_qa_pipeline.py --year 2024 --start 5 --end 12
'''