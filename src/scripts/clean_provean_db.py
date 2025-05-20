import sys
import os
import json
import sqlite3
from typing import Dict, Tuple
import argparse
from loguru import logger

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Remove None values from PROVEAN score database')
    parser.add_argument('db_path', help='Path to PROVEAN score database (.json or .sqlite)')
    parser.add_argument('--force', action='store_true', help='Skip confirmation prompt')
    return parser.parse_args()

def count_none_entries(db_path: str) -> Tuple[int, int]:
    """Count total entries and None entries in database."""
    total_entries = 0
    none_entries = 0
    
    if db_path.endswith('.db'):
        # SQLite database
        with sqlite3.connect(db_path) as conn:
            c = conn.cursor()
            c.execute("SELECT COUNT(*) FROM provean_scores")
            total_entries = c.fetchone()[0]
            c.execute("SELECT COUNT(*) FROM provean_scores WHERE score IS NULL")
            none_entries = c.fetchone()[0]
    else:
        # JSON database
        with open(db_path) as f:
            db = json.load(f)
            for gene_id, mutations in db.items():
                total_entries += len(mutations)
                none_entries += sum(1 for score in mutations.values() if score is 0)
                for mut, score in mutations.items():
                    if score is 0:
                        print(f"Gene {gene_id} has mutation {mut} with score {score}")
                        
    return total_entries, none_entries

def remove_none_entries(db_path: str) -> None:
    """Remove all entries with None values."""
    if db_path.endswith('.db'):
        # SQLite database
        with sqlite3.connect(db_path) as conn:
            c = conn.cursor()
            c.execute("DELETE FROM provean_scores WHERE score IS NULL")
            conn.commit()
    else:
        # JSON database
        with open(db_path) as f:
            db = json.load(f)
        
        # Remove None values
        cleaned_db = {}
        for gene_id, mutations in db.items():
            valid_mutations = {
                mut: score 
                for mut, score in mutations.items() 
                if score is not None
            }
            if valid_mutations:  # Only keep genes with remaining mutations
                cleaned_db[gene_id] = valid_mutations
        
        # Write to temporary file first
        temp_path = f"{db_path}.temp"
        with open(temp_path, 'w') as f:
            json.dump(cleaned_db, f)
        
        # Atomic replace
        os.replace(temp_path, db_path)

def main():
    args = parse_args()
    
    # Setup logging
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    
    try:
        # Count entries
        total, none_count = count_none_entries(args.db_path)
        
        logger.info(f"Database statistics:")
        logger.info(f"Total entries: {total}")
        logger.info(f"None entries: {none_count}")
        logger.info(f"Percentage None: {(none_count/total)*100:.2f}%")
        
        if none_count == 0:
            logger.info("No None entries found. Nothing to do.")
            return
        
        # Get confirmation
        if not args.force:
            response = input(f"\nRemove {none_count} None entries? [y/N] ")
            if response.lower() != 'y':
                logger.info("Operation cancelled.")
                return
        
        # Remove None entries
        remove_none_entries(args.db_path)
        logger.success(f"Successfully removed {none_count} None entries")
        
        # Verify
        new_total, new_none = count_none_entries(args.db_path)
        logger.info(f"New database statistics:")
        logger.info(f"Total entries: {new_total}")
        logger.info(f"None entries: {new_none}")
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main() 