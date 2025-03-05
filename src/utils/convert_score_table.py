import json
import argparse
from pathlib import Path
import sys
from loguru import logger
from tqdm import tqdm

# Add src to path
src_path = str(Path(__file__).parents[1].absolute())
sys.path.append(src_path)

from modules.provean_db import ProveanScoreDB

def parse_args() -> argparse.Namespace:
    '''Parse command line arguments.'''
    parser = argparse.ArgumentParser(
        description='Convert PROVEAN JSON score tables to SQLite database'
    )
    parser.add_argument(
        '-i', '--input',
        required=True,
        help='Path to input JSON score table'
    )
    parser.add_argument(
        '-o', '--output',
        required=True,
        help='Path for output SQLite database'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1000,
        help='Number of scores to insert in each batch (default: 1000)'
    )
    return parser.parse_args()

def convert_score_table(
    json_path: str,
    db_path: str,
    batch_size: int = 1000
) -> None:
    '''Convert JSON score table to SQLite database.
    
    Args:
        json_path: Path to input JSON file
        db_path: Path for output SQLite database
        batch_size: Number of scores to insert in each batch
    '''
    logger.info(f"Converting {json_path} to SQLite database")
    
    # Initialize database
    db = ProveanScoreDB(db_path)
    
    # Load JSON data
    logger.info("Loading JSON score table")
    with open(json_path) as f:
        score_table = json.load(f)
    
    # Count total entries for progress bar
    total_entries = sum(
        len(mutations) 
        for mutations in score_table.values()
    )
    
    logger.info(f"Converting {total_entries} scores")
    
    # Process in batches
    batch = {}
    batch_count = 0
    
    with tqdm(total=total_entries) as pbar:
        for gene, mutations in score_table.items():
            batch[gene] = mutations
            batch_count += len(mutations)
            pbar.update(len(mutations))
            
            # Insert batch when size threshold reached
            if batch_count >= batch_size:
                try:
                    db.add_scores(batch)
                except Exception as e:
                    logger.error(f"Error inserting batch: {e}")
                    raise
                batch = {}
                batch_count = 0
        
        # Insert any remaining entries
        if batch:
            try:
                db.add_scores(batch)
            except Exception as e:
                logger.error(f"Error inserting final batch: {e}")
                raise
    
    logger.success(f"Successfully converted {total_entries} scores to {db_path}")

def main() -> None:
    '''Main function to run the conversion.'''
    args = parse_args()
    
    # Setup logging
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    log_path = Path(args.output).parent / "conversion.log"
    logger.add(log_path, level="DEBUG")
    
    try:
        convert_score_table(
            args.input,
            args.output,
            args.batch_size
        )
    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 