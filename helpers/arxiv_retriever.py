#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import arxiv
import datetime
import os
import logging
from datasets import Dataset
from typing import List, Dict, Any, Optional
import argparse
import feedparser, urllib.parse, datetime as dt
import requests, time

_OA = requests.Session()
_S2 = requests.Session()

class ArxivPaperRetriever:
    """
    A class to retrieve papers from arXiv starting from a specified time
    until reaching a maximum number of papers.
    """
    
    def __init__(self, start_time: datetime.datetime, category: str, log_path: str = None, is_subcategory: bool = False):
        """
        Initialize the retriever with a start time and category.
        
        Args:
            start_time: The start time for paper retrieval
            category: The arXiv category to search (default: 'None')
                      Can be a main category (e.g., 'math', 'cs') or specific subcategory (e.g., 'cs.IT', 'math.AG')
            log_path: Path to save logs (default: None)
            is_subcategory: Whether the category is a specific subcategory (default: False)
        """
        self.start_time = start_time
        self.is_subcategory = is_subcategory
        self.category = category
        self.search_query = self.build_cat_query(category)
        self.current_end_time = None
        
        # Setup logging
        self.logger = logging.getLogger('arxiv_retriever')
        self.logger.setLevel(logging.INFO)
        
        # Create a formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # Create file handler if log_path is provided
        if log_path:
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            file_handler = logging.FileHandler(log_path)
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

    def arxiv_count(self, category_q: str, start: dt.datetime, end: dt.datetime) -> int:
        """
        Count the number of arXiv papers in a given category within a specific time range.
        """
        q = f"{category_q} AND submittedDate:[{start:%Y%m%d%H%M%S} TO {end:%Y%m%d%H%M%S}]"
        url = "https://export.arxiv.org/api/query?" + urllib.parse.urlencode({
            "search_query": q,
            "start": 0,
            "max_results": 0   # we only care about the metadata
        })
        feed = feedparser.parse(url)
        return int(feed.feed.opensearch_totalresults)

    def build_cat_query(self, category: str) -> str:
        """
        Build a search query for the specified arXiv category.
        """
        if self.is_subcategory:
            search_query = f'cat:{category}'     # exact sub-category
        else:
            search_query = f'cat:{category}.*'   # wildcard for all sub-cats
        return f'({search_query})'

    def is_primary_category(self, result: arxiv.Result, target: str) -> bool:
        """
        Check if the result's primary category matches the target category.
        """
        primary = result.primary_category
        if "." in target:
            return primary == target
        else:
            return primary.startswith(f"{target}.")
    
    def _cite_openalex(self, arxiv_id: str) -> int | None:
        """ Retrieve the citation count for an arXiv paper using OpenAlex."""
        base = arxiv_id.split("v")[0]
        url  = f"https://api.openalex.org/works/https://doi.org/10.48550/arXiv.{base}"
        r = _OA.get(url, timeout=10)
        if r.status_code == 200:
            return int(r.json().get("cited_by_count", 0))
        if r.status_code == 404:
            return None
        raise RuntimeError(f"OpenAlex {r.status_code}")

    def _cite_semanticscholar(self, arxiv_id: str) -> int | None:
        """ Retrieve the citation count for an arXiv paper using Semantic Scholar."""
        base = arxiv_id.split("v")[0]
        url  = f"https://api.semanticscholar.org/graph/v1/paper/arXiv:{base}?fields=citationCount"
        r = _S2.get(url, timeout=10)
        if r.status_code == 200:
            return int(r.json().get("citationCount", 0))
        if r.status_code == 404:
            return None
        raise RuntimeError(f"S2 {r.status_code}")
    
    def citation_count(self, arxiv_id: str, snooze=0.2) -> int:
        """Helper: get an integer citation count w/ robust fallbacks.."""
        try:
            n = self._cite_openalex(arxiv_id)
            if n is not None:
                time.sleep(snooze)
                return n
        except Exception as e:
            self.logger.debug(f"OpenAlex failed for {arxiv_id}: {e}")

        try:
            n = self._cite_semanticscholar(arxiv_id)
            if n is not None:
                time.sleep(snooze)
                return n
        except Exception as e:
            self.logger.debug(f"S2 failed for {arxiv_id}: {e}")

        self.logger.info(f"No citation data for {arxiv_id}; default 0")
        return 0

    def retrieve_papers(self, max_results: int = 100, time_window_days: int = 30, seen_ids: Optional[set] = None) -> List[Dict[str, Any]]:
        """
        Retrieve papers from arXiv within the specified category starting from start_time,
        incrementally increasing the time window until reaching max_results.
        
        Args:
            max_results: Maximum number of results to retrieve
            time_window_days: Number of days to extend the search window each iteration
            seen_ids: Set of paper IDs already retrieved
            
        Returns:
            A list of dictionaries containing paper information
        """
        papers = []
        seen_paper_ids = set() if seen_ids is None else set(seen_ids)
        current_end_time = self.start_time + datetime.timedelta(days=time_window_days)

        total_paper_count = self.arxiv_count(self.search_query, self.start_time, current_end_time)
        max_fetch = min(total_paper_count, 30000)  # arXiv API max is 30000 per query
        
        self.logger.info(f"Starting search for {self.category} papers from {self.start_time}...")
        self.logger.info(f"Total papers found in this window: {total_paper_count}")

        search_query = self.search_query + ' AND submittedDate:[{} TO {}]'.format(
            self.start_time.strftime('%Y%m%d%H%M%S'),
            current_end_time.strftime('%Y%m%d%H%M%S')
        )
        
        # Set up the arXiv client
        client = arxiv.Client(
            page_size=100,  # Number of results per query
            delay_seconds=3,  # Be nice to the API
            num_retries=5    # Retry on failure
        )
        
        # Create the search
        search = arxiv.Search(
            query=search_query,
            max_results=max_fetch,  # Only retrieve what we still need
            sort_by=arxiv.SortCriterion.SubmittedDate
        )
        
        # Execute the search and process results
        try:
            for result in client.results(search, offset=len(seen_paper_ids)):
                paper_id = result.get_short_id()
                
                # Skip this paper if we've already seen it
                if paper_id in seen_paper_ids:
                    continue

                if not self.is_primary_category(result, self.category):
                    continue
                    
                # Add to seen set
                seen_paper_ids.add(paper_id)
                
                title = result.title
                paper_link = result.entry_id
                
                # Construct the LaTeX source link
                latex_link = f"https://arxiv.org/e-print/{paper_id}"
                
                papers.append({
                    'id': paper_id,
                    'category': result.primary_category,
                    'citations': self.citation_count(paper_id),
                    'paper_link': paper_link,
                    'latex_link': latex_link,
                    'title': title
                })
                
                # Print progress and check if we have enough papers
                if len(papers) % 100 == 0:
                    self.logger.info(f"Retrieved {len(papers)} papers so far...")
                
                if len(papers) >= max_results:
                    break
            
        except arxiv.UnexpectedEmptyPageError as e:
            self.logger.warning(f"Encountered empty page error: {e}. Continuing further...")
            # papers_after remains the same as initialized
        except Exception as e:
            self.logger.error(f"Error during paper retrieval: {str(e)}. Continuing further...")
            # papers_after remains the same as initialized

        self.logger.info(f"Retrieved a total of {len(papers)} papers from {self.start_time} to {current_end_time}.")
        return papers
    
    def build_dataset(self, papers: Optional[List[Dict[str, Any]]] = None, max_results: int = 100, time_window_days: int = 30, seen_ids: Optional[set] = None) -> Dataset:
        """
        Build a Hugging Face dataset from the retrieved papers.
        
        Args:
            papers: List of paper dictionaries. If None, papers will be retrieved.
            max_results: Maximum number of results to retrieve if papers is None
            seen_ids: Set of paper IDs already retrieved
        Returns:
            A Hugging Face Dataset object
        """
        if papers is None:
            papers = self.retrieve_papers(max_results=max_results, time_window_days=time_window_days, seen_ids=seen_ids)
        
        # Create a Hugging Face dataset
        dataset = Dataset.from_list(papers)
        
        self.logger.info(f"Created dataset with {len(dataset)} papers and columns: {dataset.column_names}")
        return dataset

    def save_dataset(self, output_path: str = None, max_results: int = 100, time_window_days: int = 30, append: bool = False) -> None:
        """
        Retrieve papers, build a dataset, and save it to disk.
        
        Args:
            output_path: Path where the dataset will be saved
            max_results: Maximum number of results to retrieve
            time_window_days: Number of days for each iteration
            append: Whether to append to an existing dataset (default: False)
        """
        if output_path is None:
            output_path = f"arxiv_{self.category}_papers"

        seen_ids = set()
        existing_papers = []
        if append and os.path.exists(output_path):
            try:
                existing_dataset = Dataset.load_from_disk(output_path)
                existing_papers = existing_dataset.to_list()
                seen_ids = {paper['id'] for paper in existing_papers}
                self.logger.info(f"Loaded {len(seen_ids)} existing papers for deduplication.")
            except Exception as e:
                self.logger.warning(f"Could not load existing dataset for append: {e}")

        new_papers = self.retrieve_papers(max_results=max_results, time_window_days=time_window_days, seen_ids=seen_ids)
        all_papers = existing_papers + new_papers

        # Deduplicate just in case
        deduped = []
        seen = set()
        for paper in all_papers:
            if paper['id'] not in seen:
                seen.add(paper['id'])
                deduped.append(paper)
        final_dataset = Dataset.from_list(deduped)

        final_dataset.save_to_disk(output_path)
        self.logger.info(f"Dataset saved to {output_path} with {len(final_dataset)} unique papers.")
        self.logger.info(f"Final time window: {self.start_time} to {self.current_end_time}")


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Retrieve arXiv papers starting from a specific month until reaching max results.')
    parser.add_argument('--year', type=int, default=2024, help='Starting year to retrieve papers (default: 2023)')
    parser.add_argument('--month', type=int, default=4, help='Starting month to retrieve papers (default: 3 for March)')
    parser.add_argument('-c', '--categories', type=str, default=None, 
                        help='List of arXiv categories to search (default: None). Can only be a main category with subcategories (e.g., cs, math)')
    parser.add_argument('-sc', "--subcategories", type=str, default=None,
                        help="List of specific subcategories to search (e.g., cs.IT, math.AG).")
    parser.add_argument('--output', type=str, default=None, help='Output directory for the dataset')
    parser.add_argument('--max-results', type=int, default=100, help='Maximum number of results to retrieve')
    parser.add_argument('--time-window-days', type=int, default=30, help='Days to extend search window in each iteration')
    parser.add_argument('--append', action='store_true', help='Append new papers to existing dataset instead of overwriting')
    
    args = parser.parse_args()
    
    # Validate month
    if args.month < 1 or args.month > 12:
        raise ValueError("Month must be between 1 and 12")
    
    if not args.categories and not args.subcategories:
        raise ValueError("You must specify at least one category or subcategory to retrieve papers.")
    
    if args.categories and args.subcategories:
        raise ValueError("You cannot specify both categories and subcategories. Use either --categories or --subcategories.")
    
    is_subcategory=bool(args.subcategories)
    
    if is_subcategory:
        cat_list = [c.strip() for c in args.subcategories.split(',') if c.strip()]
    else:
        cat_list = [c.strip() for c in args.categories.split(',') if c.strip()]

    # Calculate the start time
    start_time = datetime.datetime(args.year, args.month, 1)
    
    for cat in cat_list:
        # Determine output path
        output_path = os.path.join(args.output if args.output else "output", cat, str(args.year), f"{args.month:02d}", "papers")

        # Ensure the output directory exists
        os.makedirs(output_path, exist_ok=True)

        # Create log file path
        log_file = os.path.join(output_path, 'retrieval_log.txt')

        print(f"Retrieving {cat} papers starting from {datetime.datetime.strftime(start_time, '%B %Y')}")
        print(f"Logs will be saved to {log_file}")

        # Create the retriever with log path
        retriever = ArxivPaperRetriever(start_time, category=cat, log_path=log_file, is_subcategory=is_subcategory)

        # Retrieve papers and build dataset
        retriever.save_dataset(output_path=output_path, max_results=args.max_results, time_window_days=args.time_window_days, append=args.append)

        # Show a sample of the dataset
        dataset = Dataset.load_from_disk(output_path)
        print("\nSample of the dataset:")
        print(dataset[:1])

"""
Examples:
python arxiv_retriever.py --year 2022 --month 6 --categories cs, math --output output --max-results 200
python arxiv_retriever.py --year 2022 --month 6 --subcategories cs.IT, math-ph --output output --max-results 200
"""

if __name__ == "__main__":
    main() 