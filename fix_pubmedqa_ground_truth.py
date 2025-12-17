#!/usr/bin/env python3
"""
Script to fix ground_truth values in PubMedQA result files.
Converts 'yes' -> 'A. yes', 'maybe' -> 'B. maybe', 'no' -> 'C. no'
"""

import os
import json
import glob
import logging
import argparse
from typing import Dict, Any

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def convert_ground_truth(value: str) -> str:
    """Convert ground truth value to the correct format.
    
    Args:
        value: Original ground truth value
        
    Returns:
        Converted ground truth value
    """
    if not isinstance(value, str):
        logger.warning(f"Non-string ground_truth value: {value} (type: {type(value)})")
        value = str(value)
    
    value_lower = value.lower().strip()
    
    if 'yes' in value_lower:
        return 'A. yes'
    elif 'maybe' in value_lower:
        return 'B. maybe'
    elif 'no' in value_lower:
        return 'C. no'
    else:
        logger.warning(f"Unexpected ground_truth value: {value}")
        return value


def process_file(input_path: str, output_path: str, dry_run: bool = False) -> Dict[str, int]:
    """Process a single JSONL file and update ground_truth values.
    
    Args:
        input_path: Path to input JSONL file
        output_path: Path to output JSONL file
        dry_run: If True, don't write changes, just report what would be done
        
    Returns:
        Dictionary with statistics (total, updated, unchanged)
    """
    stats = {
        'total': 0,
        'updated': 0,
        'unchanged': 0,
        'errors': 0
    }
    
    logger.info(f"Processing: {input_path}")
    
    try:
        with open(input_path, 'r') as infile:
            lines = infile.readlines()
        
        updated_lines = []
        
        for line_num, line in enumerate(lines, 1):
            stats['total'] += 1
            
            try:
                data = json.loads(line)
                
                if 'ground_truth' in data:
                    original_value = data['ground_truth']
                    new_value = convert_ground_truth(original_value)
                    
                    if original_value != new_value:
                        logger.debug(f"Line {line_num}: '{original_value}' -> '{new_value}'")
                        data['ground_truth'] = new_value
                        stats['updated'] += 1
                    else:
                        stats['unchanged'] += 1
                else:
                    logger.warning(f"Line {line_num}: No 'ground_truth' field found")
                
                updated_lines.append(json.dumps(data) + '\n')
                
            except json.JSONDecodeError as e:
                logger.error(f"Line {line_num}: JSON decode error: {e}")
                stats['errors'] += 1
                updated_lines.append(line)  # Keep original line if parsing fails
        
        # Write output
        if not dry_run:
            with open(output_path, 'w') as outfile:
                outfile.writelines(updated_lines)
            logger.info(f"Written to: {output_path}")
        else:
            logger.info(f"[DRY RUN] Would write to: {output_path}")
        
    except Exception as e:
        logger.error(f"Error processing file {input_path}: {e}")
        stats['errors'] += 1
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Fix ground_truth values in PubMedQA result files"
    )
    parser.add_argument(
        '--results_dir',
        type=str,
        default=None,
        help="Directory containing result files (default: s1_test_time/results)"
    )
    parser.add_argument(
        '--pattern',
        type=str,
        default='*pubmedqa*.jsonl',
        help="File pattern to match (default: *pubmedqa*.jsonl)"
    )
    parser.add_argument(
        '--in_place',
        action='store_true',
        help="Modify files in place (default: create .fixed.jsonl files)"
    )
    parser.add_argument(
        '--dry_run',
        action='store_true',
        help="Don't write changes, just report what would be done"
    )
    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Determine results directory
    if args.results_dir:
        results_dir = args.results_dir
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        results_dir = os.path.join(script_dir, 'results')
    
    if not os.path.exists(results_dir):
        logger.error(f"Results directory not found: {results_dir}")
        return
    
    logger.info(f"Searching for files in: {results_dir}")
    logger.info(f"Pattern: {args.pattern}")
    
    # Find matching files
    pattern_path = os.path.join(results_dir, args.pattern)
    files = glob.glob(pattern_path)
    
    if not files:
        logger.warning(f"No files found matching pattern: {pattern_path}")
        return
    
    logger.info(f"Found {len(files)} files to process")
    
    if args.dry_run:
        logger.info("DRY RUN MODE - No files will be modified")
    
    # Process each file
    total_stats = {
        'total': 0,
        'updated': 0,
        'unchanged': 0,
        'errors': 0
    }
    
    for file_path in sorted(files):
        # Determine output path
        if args.in_place:
            output_path = file_path
        else:
            base, ext = os.path.splitext(file_path)
            output_path = f"{base}.fixed{ext}"
        
        # Process file
        file_stats = process_file(file_path, output_path, args.dry_run)
        
        # Update totals
        for key in total_stats:
            total_stats[key] += file_stats[key]
        
        logger.info(
            f"  Stats: {file_stats['total']} total, "
            f"{file_stats['updated']} updated, "
            f"{file_stats['unchanged']} unchanged, "
            f"{file_stats['errors']} errors"
        )
    
    # Print summary
    logger.info("=" * 80)
    logger.info("SUMMARY:")
    logger.info(f"  Files processed: {len(files)}")
    logger.info(f"  Total entries: {total_stats['total']}")
    logger.info(f"  Updated: {total_stats['updated']}")
    logger.info(f"  Unchanged: {total_stats['unchanged']}")
    logger.info(f"  Errors: {total_stats['errors']}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
