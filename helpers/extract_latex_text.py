#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import requests
import tempfile
import shutil
import tarfile
import zipfile
import gzip
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging
from datasets import Dataset, load_from_disk
import time
import glob
import subprocess
import multiprocessing as mp
from tqdm import tqdm
import math
import argparse

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ArxivLatexExtractor:
    """
    A class to extract LaTeX source from arXiv papers and build a dataset with full text.
    """
    
    def __init__(self, dataset_path: str = "arxiv_papers"):
        """
        Initialize the extractor with the path to the dataset.
        
        Args:
            dataset_path: Path to the Hugging Face dataset
        """
        self.dataset_path = dataset_path
        self.dataset = load_from_disk(dataset_path)
        # Add session for connection pooling and consistent headers
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (compatible; ArxivLatexExtractor/1.0; +mailto:zj.jayzhang@gmail.com)'
        })
        logger.info(f"Loaded dataset with {len(self.dataset)} papers")
    
    def _process_chunk(self, chunk_data: List[Dict], output_path: str, chunk_id: int) -> Dict:
        """
        Process a chunk of papers in parallel.
        
        Args:
            chunk_data: List of papers to process
            output_path: Path to save the results
            chunk_id: ID of the chunk for logging
            
        Returns:
            Dictionary containing processed data
        """
        chunk_output = {
            'id': [],
            'paper_link': [],
            'title': [],
            'full_text': []
        }
        
        for paper in chunk_data:
            paper_id = paper['id']
            latex_link = paper['latex_link']
            
            logger.info(f"Processing paper {paper_id} in chunk {chunk_id}")
            
            success, full_text = self.process_paper(paper_id, latex_link)
            if success and full_text.strip():
                chunk_output['id'].append(paper_id)
                chunk_output['paper_link'].append(paper['paper_link'])
                chunk_output['title'].append(paper['title'])
                chunk_output['full_text'].append(full_text)
            
            # Be nice to the arXiv API
            time.sleep(1)
        
        return chunk_output
    
    def build_full_text_dataset(self, output_path: str = "latex_text", 
                              max_papers: Optional[int] = None,
                              num_processes: Optional[int] = None,
                              overwrite: bool = True) -> Dataset:
        """
        Build a dataset with full text for each paper.
        
        Args:
            output_path: Path to save the new dataset
            max_papers: Maximum number of papers to process (None for all)
            num_processes: Number of parallel processes to use (None for all available cores)
            overwrite: Whether to overwrite an existing dataset (True) or append to it (False)
            
        Returns:
            The new dataset
        """
        # Filter out already processed papers
        papers_to_process = []
        for paper in self.dataset:
            papers_to_process.append(paper)
        
        # Apply max_papers limit if specified
        if max_papers is not None:
            papers_to_process = papers_to_process[:max_papers]
        
        if not papers_to_process:
            logger.info("No papers to process")
            return Dataset.load_from_disk(output_path) if os.path.exists(output_path) else None
        
        # Determine number of processes
        if num_processes is None:
            num_processes = mp.cpu_count()
        
        # Split papers into chunks for parallel processing
        chunk_size = math.ceil(len(papers_to_process) / num_processes)
        chunks = [papers_to_process[i:i + chunk_size] for i in range(0, len(papers_to_process), chunk_size)]
        
        # Process chunks in parallel
        with mp.Pool(num_processes) as pool:
            chunk_results = []
            for i, chunk in enumerate(chunks):
                result = pool.apply_async(self._process_chunk, (chunk, output_path, i))
                chunk_results.append(result)
            
            # Collect results
            processed_chunk_outputs = []
            for result in tqdm(chunk_results, desc="Processing chunks"):
                chunk_output = result.get()
                processed_chunk_outputs.append(chunk_output)
        
        # Combine results from all chunks
        batch_data = {
            'id': [],
            'paper_link': [],
            'title': [],
            'full_text': []
        }
        
        for chunk_output in processed_chunk_outputs:
            for key in batch_data:
                batch_data[key].extend(chunk_output[key])
        
        # Load existing dataset if we're appending
        if os.path.exists(output_path) and not overwrite:
            try:
                existing_dataset = Dataset.load_from_disk(output_path)
                # Create a new dictionary combining existing and new data
                final_data = {
                    'id': existing_dataset['id'] + batch_data['id'],
                    'paper_link': existing_dataset['paper_link'] + batch_data['paper_link'],
                    'title': existing_dataset['title'] + batch_data['title'],
                    'full_text': existing_dataset['full_text'] + batch_data['full_text']
                }
                logger.info(f"Appended {len(batch_data['id'])} new papers to existing dataset with {len(existing_dataset)} papers")
            except Exception as e:
                logger.warning(f"Error loading existing dataset: {e}. Creating a new one.")
                final_data = batch_data
        else:
            # We're creating a new dataset or overwriting the old one
            if overwrite and os.path.exists(output_path):
                logger.info(f"Overwriting existing dataset at {output_path}")
            final_data = batch_data
        
        # Create and save the dataset
        dataset = Dataset.from_dict(final_data)
        dataset.save_to_disk(output_path)
        logger.info(f"Saved dataset with {len(dataset)} papers to {output_path}")
        
        return dataset
    
    def download_latex_source(self, latex_link: str, output_dir: str) -> str:
        """
        Download the LaTeX source from arXiv with proper rate limiting and retries.
        
        Args:
            latex_link: URL to the LaTeX source
            output_dir: Directory to save the downloaded file
            
        Returns:
            Path to the downloaded file
        """
        try:
            # Convert e-print URL to the proper format if needed
            if 'arxiv.org/e-print' in latex_link:
                paper_id = latex_link.split('/')[-1]
                latex_link = f"https://export.arxiv.org/e-print/{paper_id}"
            
            # Implement exponential backoff for retries
            max_retries = 5
            retry_delay = 3
            
            for attempt in range(max_retries):
                try:
                    response = self.session.get(latex_link, stream=True)
                    response.raise_for_status()
                    break
                except requests.exceptions.RequestException as e:
                    if attempt == max_retries - 1:  # Last attempt
                        raise
                    wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                    logger.warning(f"Download attempt {attempt + 1} failed: {e}. Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
            
            # Save the downloaded file with a generic name
            output_file = os.path.join(output_dir, "source")
            with open(output_file, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Add additional delay between downloads
            time.sleep(5)  # Be extra nice to arXiv's servers
            
            logger.info(f"Downloaded source to {output_file}")
            return output_file
        
        except Exception as e:
            logger.error(f"Error downloading LaTeX source: {e}")
            return ""
    
    def extract_archive(self, archive_path: str, extract_dir: str) -> bool:
        """
        Extract the downloaded archive using multiple methods.
        
        Args:
            archive_path: Path to the archive file
            extract_dir: Directory to extract the files to
            
        Returns:
            True if extraction was successful, False otherwise
        """
        # First, check if the file exists and has content
        if not os.path.exists(archive_path):
            logger.error(f"Archive file does not exist: {archive_path}")
            return False
            
        if os.path.getsize(archive_path) == 0:
            logger.error(f"Archive file is empty: {archive_path}")
            return False

        # Try to identify the file type
        try:
            with open(archive_path, 'rb') as f:
                magic_bytes = f.read(4)
            file_type = "unknown"
            if magic_bytes.startswith(b'\x1f\x8b'):  # gzip
                file_type = "gzip"
            elif magic_bytes.startswith(b'PK'):  # zip
                file_type = "zip"
            elif magic_bytes.startswith(b'\\documentclass') or magic_bytes.startswith(b'\\begin'):  # raw tex
                file_type = "tex"
            logger.info(f"Detected file type: {file_type}")
        except Exception as e:
            logger.error(f"Error reading file header: {e}")
            file_type = "unknown"
        
        # Try different extraction methods based on file type
        methods = []
        if file_type == "gzip":
            methods = [self._extract_with_gzip, self._extract_with_tarfile, self._extract_with_system_tar]
        elif file_type == "zip":
            methods = [self._extract_with_zipfile]
        elif file_type == "tex":
            # If it's already a tex file, just copy it
            try:
                output_file = os.path.join(extract_dir, "main.tex")
                shutil.copy2(archive_path, output_file)
                logger.info(f"Copied tex file directly to {output_file}")
                return True
            except Exception as e:
                logger.error(f"Error copying tex file: {e}")
                return False
        else:
            # Try all methods if type is unknown
            methods = [
                self._extract_with_tarfile,
                self._extract_with_zipfile,
                self._extract_with_gzip,
                self._extract_with_system_tar
            ]
        
        for method in methods:
            try:
                logger.info(f"Trying extraction method: {method.__name__}")
                if method(archive_path, extract_dir):
                    return True
            except Exception as e:
                logger.error(f"Extraction method {method.__name__} failed with error: {str(e)}")
        
        # If all methods failed, try to read the file as text
        try:
            with open(archive_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read(1000)  # Read first 1000 chars to check
                if '\\documentclass' in content or '\\begin{document}' in content:
                    output_file = os.path.join(extract_dir, "main.tex")
                    shutil.copy2(archive_path, output_file)
                    logger.info(f"File appears to be raw LaTeX, copied to {output_file}")
                    return True
        except Exception as e:
            logger.error(f"Error checking for raw LaTeX content: {e}")
        
        logger.error(f"All extraction methods failed for {archive_path}")
        return False
    
    def _extract_with_tarfile(self, archive_path: str, extract_dir: str) -> bool:
        """Extract using Python's tarfile module."""
        try:
            with tarfile.open(archive_path, 'r:*') as tar:
                # Use a safe extraction filter (Python 3.12+)
                if hasattr(tarfile, 'data_filter'):
                    tar.extractall(path=extract_dir, filter='data')
                else:
                    tar.extractall(path=extract_dir)
            logger.info(f"Extracted archive with tarfile to {extract_dir}")
            return True
        except Exception as e:
            logger.debug(f"tarfile extraction failed: {e}")
            return False
    
    def _extract_with_zipfile(self, archive_path: str, extract_dir: str) -> bool:
        """Extract using Python's zipfile module."""
        try:
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            logger.info(f"Extracted archive with zipfile to {extract_dir}")
            return True
        except Exception as e:
            logger.debug(f"zipfile extraction failed: {e}")
            return False
    
    def _extract_with_gzip(self, archive_path: str, extract_dir: str) -> bool:
        """Extract using Python's gzip module for single file archives."""
        try:
            with gzip.open(archive_path, 'rb') as f_in:
                # Try to determine the filename
                output_file = os.path.join(extract_dir, "extracted.tex")
                with open(output_file, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            logger.info(f"Extracted archive with gzip to {extract_dir}")
            return True
        except Exception as e:
            logger.debug(f"gzip extraction failed: {e}")
            return False
    
    def _extract_with_system_tar(self, archive_path: str, extract_dir: str) -> bool:
        """Extract using system tar command."""
        try:
            result = subprocess.run(
                ["tar", "-xf", archive_path, "-C", extract_dir],
                capture_output=True,
                text=True,
                check=False
            )
            if result.returncode == 0:
                logger.info(f"Extracted archive with system tar to {extract_dir}")
                return True
            else:
                logger.debug(f"system tar extraction failed: {result.stderr}")
                return False
        except Exception as e:
            logger.debug(f"system tar execution failed: {e}")
            return False
    
    def find_tex_files(self, extract_dir: str) -> List[str]:
        """
        Find all .tex files in the extracted directory.
        
        Args:
            extract_dir: Directory containing the extracted files
            
        Returns:
            List of paths to .tex files
        """
        # First, try to find .tex files
        tex_files = glob.glob(os.path.join(extract_dir, "**", "*.tex"), recursive=True)
        
        # If no .tex files found, look for any text files that might contain LaTeX
        if not tex_files:
            # Look for common text file extensions
            for ext in [".txt", ".text", ".latex", ""]:
                text_files = glob.glob(os.path.join(extract_dir, "**", f"*{ext}"), recursive=True)
                for file_path in text_files:
                    # Check if the file contains LaTeX commands
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read(1000)  # Read first 1000 chars
                            if r'\documentclass' in content or r'\begin{document}' in content:
                                tex_files.append(file_path)
                    except Exception:
                        pass
        
        return tex_files
    
    def find_main_tex_file(self, extract_dir: str) -> str:
        """
        Find the main .tex file in the extracted directory.
        
        Args:
            extract_dir: Directory containing the extracted files
            
        Returns:
            Path to the main .tex file
        """
        # Get all .tex files
        tex_files = self.find_tex_files(extract_dir)
        
        if not tex_files:
            logger.warning(f"No .tex files found in {extract_dir}")
            return ""
        
        # If there's only one .tex file, that's our main file
        if len(tex_files) == 1:
            return tex_files[0]
        
        # Look for common main file names
        common_main_names = ['main.tex', 'paper.tex', 'article.tex', 'manuscript.tex']
        for name in common_main_names:
            for tex_file in tex_files:
                if os.path.basename(tex_file).lower() == name:
                    return tex_file
        
        # Look for .tex files that contain \documentclass or \begin{document}
        for tex_file in tex_files:
            try:
                with open(tex_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    if r'\documentclass' in content or r'\begin{document}' in content:
                        return tex_file
            except Exception:
                pass
        
        # If we can't determine the main file, return the largest .tex file
        try:
            return max(tex_files, key=os.path.getsize)
        except Exception:
            # If there's an error getting file size, just return the first file
            return tex_files[0] if tex_files else ""
    
    def determine_tex_file_order(self, extract_dir: str) -> List[str]:
        """
        Determine the correct order of .tex files.
        
        Args:
            extract_dir: Directory containing the extracted files
            
        Returns:
            List of .tex files in the correct order
        """
        main_tex_file = self.find_main_tex_file(extract_dir)
        if not main_tex_file:
            return []
        
        # Start with the main file
        ordered_files = [main_tex_file]
        processed_files = set([main_tex_file])
        
        # Process the main file and any included files recursively
        self._process_tex_file_includes(main_tex_file, extract_dir, ordered_files, processed_files)
        
        # Add any remaining .tex files that weren't referenced
        all_tex_files = self.find_tex_files(extract_dir)
        for tex_file in all_tex_files:
            if tex_file not in processed_files:
                ordered_files.append(tex_file)
                processed_files.add(tex_file)
        
        return ordered_files
    
    def _process_tex_file_includes(self, tex_file: str, extract_dir: str, 
                                  ordered_files: List[str], processed_files: set) -> None:
        """
        Recursively process a .tex file to find included files.
        
        Args:
            tex_file: Path to the .tex file to process
            extract_dir: Directory containing the extracted files
            ordered_files: List to append found files to
            processed_files: Set of already processed files
        """
        try:
            with open(tex_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Find all \input and \include commands
            input_pattern = r'\\input\{([^}]+)\}'
            include_pattern = r'\\include\{([^}]+)\}'
            
            input_files = re.findall(input_pattern, content)
            include_files = re.findall(include_pattern, content)
            
            # Process found files
            for file_ref in input_files + include_files:
                # Add .tex extension if not present
                if not file_ref.endswith('.tex'):
                    file_ref_with_ext = file_ref + '.tex'
                else:
                    file_ref_with_ext = file_ref
                
                # Find the actual file path
                file_path = None
                
                # Try with the exact name
                potential_path = os.path.join(os.path.dirname(tex_file), file_ref_with_ext)
                if os.path.exists(potential_path):
                    file_path = potential_path
                
                # Try searching in the entire extract directory
                if not file_path:
                    for root, _, files in os.walk(extract_dir):
                        for file in files:
                            if file == os.path.basename(file_ref_with_ext):
                                file_path = os.path.join(root, file)
                                break
                        if file_path:
                            break
                
                # If found and not already processed, add it and process its includes
                if file_path and file_path not in processed_files:
                    ordered_files.append(file_path)
                    processed_files.add(file_path)
                    self._process_tex_file_includes(file_path, extract_dir, ordered_files, processed_files)
        
        except Exception as e:
            logger.error(f"Error processing includes in {tex_file}: {e}")
    
    def extract_text_from_tex(self, tex_file: str) -> str:
        """
        Extract text from a .tex file, ignoring commented lines.
        
        Args:
            tex_file: Path to the .tex file
            
        Returns:
            Extracted text
        """
        try:
            with open(tex_file, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            
            # Remove commented lines and parts of lines
            processed_lines = []
            in_comment_environment = False
            
            for line in lines:
                # Check for comment environment start/end
                if r'\begin{comment}' in line or r'\iffalse' in line:
                    in_comment_environment = True
                    # Keep the part before the comment environment starts
                    start_pos = min(
                        line.find(r'\begin{comment}') if r'\begin{comment}' in line else float('inf'),
                        line.find(r'\iffalse') if r'\iffalse' in line else float('inf')
                    )
                    if start_pos > 0:
                        processed_lines.append(line[:start_pos])
                    continue
                
                if r'\end{comment}' in line or r'\fi' in line:
                    in_comment_environment = False
                    # Keep the part after the comment environment ends
                    end_pos = max(
                        line.find(r'\end{comment}') + len(r'\end{comment}') if r'\end{comment}' in line else -1,
                        line.find(r'\fi') + len(r'\fi') if r'\fi' in line else -1
                    )
                    if end_pos < len(line) and end_pos > 0:
                        processed_lines.append(line[end_pos:])
                    continue
                
                # Skip lines in comment environments
                if in_comment_environment:
                    continue
                
                # Skip lines that start with %
                if line.strip().startswith('%'):
                    continue
                
                # Remove inline comments from the line
                comment_pos = line.find('%')
                if comment_pos >= 0:
                    # Check if the % is escaped or part of a command
                    i = comment_pos - 1
                    is_comment = True
                    
                    # Check for escaped % or % in a command like \%
                    while i >= 0 and line[i] == '\\':
                        is_comment = not is_comment
                        i -= 1
                    
                    # Also check if % is inside a math environment (between $ or $$)
                    in_math = False
                    dollar_positions = [pos for pos, char in enumerate(line[:comment_pos]) if char == '$']
                    if len(dollar_positions) % 2 == 1:  # Odd number of $ before %, so we're in math mode
                        is_comment = False
                    
                    # Check for % inside braces like \command{...%...}
                    open_braces = line[:comment_pos].count('{')
                    close_braces = line[:comment_pos].count('}')
                    if open_braces > close_braces:
                        # We're inside braces, check if it's a command parameter
                        last_open_brace = line[:comment_pos].rfind('{')
                        if last_open_brace > 0 and line[last_open_brace-1] == '\\':
                            is_comment = False
                    
                    if is_comment:
                        line = line[:comment_pos]
                
                processed_lines.append(line)
            
            return ''.join(processed_lines)
        
        except Exception as e:
            logger.error(f"Error extracting text from {tex_file}: {e}")
            return ""
    
    def process_paper(self, paper_id: str, latex_link: str) -> Tuple[bool, str]:
        """
        Process a single paper: download, extract, and get the full text.
        
        Args:
            paper_id: ID of the paper
            latex_link: URL to the LaTeX source
            
        Returns:
            Tuple of (success, full_text)
        """
        try:
            # Create temporary directories
            # create a temp directory in the current directory
            temp_dir = os.path.join(os.getcwd(), "tmp")
            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir)
            with tempfile.TemporaryDirectory(dir=temp_dir) as temp_dir:
                download_dir = os.path.join(temp_dir, "download")
                extract_dir = os.path.join(temp_dir, "extract")
                os.makedirs(download_dir, exist_ok=True)
                os.makedirs(extract_dir, exist_ok=True)
                
                # Download the LaTeX source
                archive_path = self.download_latex_source(latex_link, download_dir)
                if not archive_path:
                    return False, ""
                
                # Extract the archive
                if not self.extract_archive(archive_path, extract_dir):
                    logger.warning(f"Could not extract archive for paper {paper_id}, trying direct text extraction")
                    # Try to read the file directly as text
                    try:
                        with open(archive_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                        if r'\documentclass' in content or r'\begin{document}' in content:
                            # This looks like a LaTeX file
                            # Apply comment removal to the direct content
                            content = self._remove_comments_from_text(content)
                            return True, content
                    except Exception:
                        pass
                    return False, ""
                
                # Determine the order of .tex files
                ordered_tex_files = self.determine_tex_file_order(extract_dir)
                if not ordered_tex_files:
                    logger.warning(f"No .tex files found for paper {paper_id}")
                    return False, ""
                
                # Extract text from each .tex file
                full_text = ""
                for tex_file in ordered_tex_files:
                    text = self.extract_text_from_tex(tex_file)
                    full_text += text + "\n\n"
                
                # Final pass to remove any remaining comments
                full_text = self._remove_comments_from_text(full_text)
                
                return True, full_text
        
        except Exception as e:
            logger.error(f"Error processing paper {paper_id}: {e}")
            return False, ""
    
    def _remove_comments_from_text(self, text: str) -> str:
        """
        Remove comments from a text string.
        
        Args:
            text: The text to process
            
        Returns:
            Text with comments removed
        """
        # Split into lines for processing
        lines = text.split('\n')
        processed_lines = []
        in_comment_environment = False
        
        for line in lines:
            # Check for comment environment start/end
            if r'\begin{comment}' in line or r'\iffalse' in line:
                in_comment_environment = True
                # Keep the part before the comment environment starts
                start_pos = min(
                    line.find(r'\begin{comment}') if r'\begin{comment}' in line else float('inf'),
                    line.find(r'\iffalse') if r'\iffalse' in line else float('inf')
                )
                if start_pos > 0:
                    processed_lines.append(line[:start_pos])
                continue
            
            if r'\end{comment}' in line or r'\fi' in line:
                in_comment_environment = False
                # Keep the part after the comment environment ends
                end_pos = max(
                    line.find(r'\end{comment}') + len(r'\end{comment}') if r'\end{comment}' in line else -1,
                    line.find(r'\fi') + len(r'\fi') if r'\fi' in line else -1
                )
                if end_pos < len(line) and end_pos > 0:
                    processed_lines.append(line[end_pos:])
                continue
            
            # Skip lines in comment environments
            if in_comment_environment:
                continue
            
            # Skip lines that start with %
            if line.strip().startswith('%'):
                continue
            
            # Remove inline comments from the line
            comment_pos = line.find('%')
            if comment_pos >= 0:
                # Check if the % is escaped or part of a command
                i = comment_pos - 1
                is_comment = True
                
                # Check for escaped % or % in a command like \%
                while i >= 0 and line[i] == '\\':
                    is_comment = not is_comment
                    i -= 1
                
                # Also check if % is inside a math environment (between $ or $$)
                dollar_positions = [pos for pos, char in enumerate(line[:comment_pos]) if char == '$']
                if len(dollar_positions) % 2 == 1:  # Odd number of $ before %, so we're in math mode
                    is_comment = False
                
                # Check for % inside braces like \command{...%...}
                open_braces = line[:comment_pos].count('{')
                close_braces = line[:comment_pos].count('}')
                if open_braces > close_braces:
                    # We're inside braces, check if it's a command parameter
                    last_open_brace = line[:comment_pos].rfind('{')
                    if last_open_brace > 0 and line[last_open_brace-1] == '\\':
                        is_comment = False
                
                if is_comment:
                    line = line[:comment_pos]
            
            processed_lines.append(line)
        
        return '\n'.join(processed_lines)


"""
python arxiv_latex_extractor.py --batch-size 10 --processes 2
"""
def main():
    parser = argparse.ArgumentParser(description='Extract LaTeX source from arXiv papers.')
    parser.add_argument('--input', default="arxiv_papers",
                      help='Input dataset path (default: arxiv_papers)')
    parser.add_argument('--output', default="latex_text",
                      help='Output dataset path (default: latex_text)')
    parser.add_argument('--max-papers', type=int, default=None,
                      help='Maximum number of papers to process (default: all)')
    parser.add_argument('--processes', type=int, default=3,
                      help='Number of parallel processes (default: number of CPU cores)')
    parser.add_argument('--batch-size', type=int, default=10,
                      help='Number of papers to process in each batch (default: 10)')
    parser.add_argument('--append', action='store_true',
                      help='Append to existing dataset instead of overwriting it (default: overwrite)')
    args = parser.parse_args()


    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    # Set up file logging in addition to console logging
    log_file = os.path.join(args.output, "extraction_log.txt")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    logger.info(f"Logs will be saved to {log_file}")
    if args.append:
        logger.info("Append mode enabled - results will be added to any existing dataset")
    else:
        logger.info("Overwrite mode enabled (default) - existing dataset will be replaced")

    # Create the extractor
    extractor = ArxivLatexExtractor(dataset_path=args.input)
    
    # Get all papers to process
    papers_to_process = []
    for paper in extractor.dataset:
        papers_to_process.append(paper)
    total_papers = len(papers_to_process)
    logger.info(f"Found {total_papers} papers to process")
    
    if total_papers == 0:
        logger.info("No papers to process!")
        return
    
    # Apply max_papers limit if specified
    if args.max_papers:
        papers_to_process = papers_to_process[:args.max_papers]
        total_papers = len(papers_to_process)
    
    # Remove duplicates by paper_link while preserving order
    unique_papers = []
    seen_links = set()
    for paper in papers_to_process:
        paper_link = paper['paper_link']
        if paper_link not in seen_links:
            seen_links.add(paper_link)
            unique_papers.append(paper)
    
    # Update papers_to_process with deduplicated list
    papers_to_process = unique_papers
    total_papers = len(papers_to_process)
    logger.info(f"After removing duplicates: {total_papers} unique papers to process")
    
    # Calculate number of batches
    num_batches = (total_papers + args.batch_size - 1) // args.batch_size
    
    # Track successfully processed papers
    total_processed = 0

    
    # Process papers in batches
    for batch in range(num_batches):
        start_idx = batch * args.batch_size
        end_idx = min((batch + 1) * args.batch_size, total_papers)
        batch_papers = papers_to_process[start_idx:end_idx]
        
        logger.info(f"\nProcessing batch {batch + 1}/{num_batches} (papers {start_idx} to {end_idx-1} of {total_papers-1})")
        
        # Create a temporary dataset for this batch
        batch_dataset = Dataset.from_list(batch_papers)
        extractor.dataset = batch_dataset
        
        # Build the full text dataset for this batch
        # First batch overwrites if not in append mode, rest always append
        should_overwrite = not (args.append or batch > 0)
        
        dataset = extractor.build_full_text_dataset(
            output_path=args.output,
            max_papers=len(batch_papers),
            num_processes=args.processes,
            overwrite=should_overwrite
        )
        
        # Track progress
        if dataset is not None:
            batch_processed = len(dataset) - total_processed if batch > 0 else len(dataset)
            total_processed = len(dataset)
            logger.info(f"Batch {batch + 1} completed. Processed {batch_processed} papers in this batch.")
            logger.info(f"Total papers processed so far: {total_processed}")
            
            # Show a sample from the first batch
            if batch == 0:
                print("\nSample from first batch:")
                sample = dataset[:1]
                print(f"ID: {sample['id'][0]}")
                print(f"Title: {sample['title'][0]}")
                print(f"Paper Link: {sample['paper_link'][0]}")
                print(f"Full Text (first 500 chars): {sample['full_text'][0][:500]}...")
        else:
            logger.warning(f"No papers were successfully processed in batch {batch + 1}")
    
    logger.info("\nProcessing completed!")
    
    # Load and show final statistics
    if os.path.exists(args.output):
        final_dataset = Dataset.load_from_disk(args.output)
        logger.info(f"Final dataset contains {len(final_dataset)} papers")
    else:
        logger.warning("No output dataset found. No papers may have been successfully processed.")


if __name__ == "__main__":
    main() 