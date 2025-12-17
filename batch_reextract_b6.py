#!/usr/bin/env python3
"""
Batch script to run reextract_final_answers_b6.py on all B6 word-limited files.
Finds all files matching "*B6*words.jsonl" in the results directory and creates
reextracted versions with "_reextract.jsonl" appended to the filename.
"""

import os
import glob
import subprocess
import logging
from pathlib import Path
from typing import List

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def find_b6_word_files(results_dir: str) -> List[str]:
    """
    Find all files matching the pattern "*B6*words.jsonl" in the results directory.
    
    Args:
        results_dir: Path to the results directory
        
    Returns:
        List of matching file paths
    """
    pattern = os.path.join(results_dir, "hf_s1_results*B6*words.jsonl")
    files = glob.glob(pattern)
    # Filter out any files that already have "_reextract" in the name
    files = [f for f in files if "_reextract" not in f]
    return sorted(files)


def generate_output_filename(input_file: str) -> str:
    """
    Generate output filename by replacing .jsonl with _reextract.jsonl
    
    Args:
        input_file: Path to input file
        
    Returns:
        Path to output file
    """
    if input_file.endswith('.jsonl'):
        return input_file[:-6] + '_reextract.jsonl'
    else:
        return input_file + '_reextract.jsonl'


def run_reextraction(input_file: str, output_file: str, reextract_script: str) -> bool:
    """
    Run the reextraction script on a single file.
    
    Args:
        input_file: Path to input file
        output_file: Path to output file
        reextract_script: Path to the reextraction script
        
    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info(f"Processing: {os.path.basename(input_file)}")
        logger.info(f"  Output: {os.path.basename(output_file)}")
        
        # Run the reextraction script
        cmd = [
            'python3',
            reextract_script,
            input_file,
            output_file,
            '--baseline_style', 'B6'
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        
        logger.info(f"  ✓ Successfully processed {os.path.basename(input_file)}")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"  ✗ Error processing {os.path.basename(input_file)}")
        logger.error(f"  Error: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"  ✗ Unexpected error processing {os.path.basename(input_file)}: {e}")
        return False


def main():
    # Get the script directory and results directory
    script_dir = Path(__file__).parent
    results_dir = script_dir / "results"
    reextract_script = script_dir / "reextract_final_answers_b6.py"
    
    # Verify paths exist
    if not results_dir.exists():
        logger.error(f"Results directory not found: {results_dir}")
        return
    
    if not reextract_script.exists():
        logger.error(f"Reextraction script not found: {reextract_script}")
        return
    
    # Find all matching files
    logger.info(f"Searching for files matching pattern '*B6*words.jsonl' in {results_dir}")
    input_files = find_b6_word_files(str(results_dir))
    
    if not input_files:
        logger.warning("No matching files found!")
        return
    
    logger.info(f"Found {len(input_files)} files to process")
    
    # Process each file
    success_count = 0
    failure_count = 0
    
    for input_file in input_files:
        output_file = generate_output_filename(input_file)
        
        # Check if output file already exists
        if os.path.exists(output_file):
            logger.warning(f"  ⚠ Output file already exists, skipping: {os.path.basename(output_file)}")
            continue
        
        # Run reextraction
        if run_reextraction(input_file, output_file, str(reextract_script)):
            success_count += 1
        else:
            failure_count += 1
    
    # Summary
    logger.info("=" * 80)
    logger.info(f"Batch processing complete!")
    logger.info(f"  Total files: {len(input_files)}")
    logger.info(f"  Successful: {success_count}")
    logger.info(f"  Failed: {failure_count}")
    logger.info(f"  Skipped: {len(input_files) - success_count - failure_count}")


if __name__ == "__main__":
    main()
