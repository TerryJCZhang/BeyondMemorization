import os
import tempfile
import shutil
from datasets import load_from_disk, concatenate_datasets

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
    parser = argparse.ArgumentParser(description="Fetch QA pairs in all qa_pairs folders under a root directory. " \
                                                    "Only works with a structure like root/category/year/month/qa_pairs.")
    parser.add_argument("--input", type=str, default="output", help="Root directory to get qa_pairs data")
    parser.add_argument("--output", type=str, default="data", help="Root directory to save the collated data")
    args = parser.parse_args()

    root_dir = args.input
    out_root = args.output
    qa_dirs = find_qa_pairs_dirs(root_dir)
    if not qa_dirs:
        print(f"No qa_pairs folders found in {root_dir}")
        return

    for qa_dir in qa_dirs:
        print(f"\nProcessing qa_pairs directory: {qa_dir}")
        try:
            ds = load_from_disk(qa_dir)
            print(f"  Loaded {len(ds)} new QA pairs from {qa_dir}")
        except Exception as e:
            print(f"  Could not load dataset at {qa_dir}: {e}")
            continue

        cat, year, month = parse_metadata_from_path(qa_dir, root_dir)
        cat = get_category(cat)
        if not (cat and year and month):
            print(f"  Could not parse metadata from {qa_dir}")
            continue

        save_dir = os.path.join(out_root, cat, year, month, "qa_pairs")
        os.makedirs(save_dir, exist_ok=True)

        # If data already exists, load and append
        existing_count = 0
        if os.path.exists(save_dir):
            try:
                existing_ds = load_from_disk(save_dir)
                existing_count = len(existing_ds)
                print(f"  Found {existing_count} existing QA pairs in {save_dir}")
                ds = concatenate_datasets([existing_ds, ds])
            except Exception as e:
                print(f"  Could not load existing dataset at {save_dir}: {e}")

        print(f"  Saving total {len(ds)} QA pairs to {save_dir} (added {len(ds) - existing_count} new)")
        # Save to a temporary directory first to avoid overwrite error
        with tempfile.TemporaryDirectory() as tmp_save_path:
            ds.save_to_disk(tmp_save_path)
            shutil.rmtree(save_dir)
            shutil.move(tmp_save_path, save_dir)

if __name__ == "__main__":
    main()