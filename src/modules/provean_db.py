from typing import Dict, Optional
from loguru import logger
import time
from threading import Lock
from collections import defaultdict
import sqlite3
import json
import os

class ProveanScoreDB:
    """Class to handle PROVEAN score database operations."""
    
    def __init__(self, db_path: str):
        """Initialize the database by loading it entirely into memory.
        
        Args:
            db_path: Path to the database file (either .sqlite or .json)
        """
        self.db_path = db_path
        self.db = defaultdict(dict)  # Thread-safe for reads
        self.lock = Lock()  # For write operations
        
        # Determine database type and load
        if db_path.endswith('.db'):
            self.db_type = 'sqlite'
            self._setup_sqlite_db()
            self._load_from_sqlite()
        else:
            self.db_type = 'json'
            self._load_from_json()
            
    def _setup_sqlite_db(self):
        """Setup SQLite database if it doesn't exist."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                c = conn.cursor()
                # Get table info
                c.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = c.fetchall()
                
                if not tables:
                    # Create new table if database is empty
                    c.execute('''CREATE TABLE IF NOT EXISTS provean_scores
                               (gene TEXT, mutation TEXT, score REAL,
                                PRIMARY KEY (gene, mutation))''')
                    conn.commit()
                else:
                    # Get column info for existing table
                    table_name = tables[0][0]  # Use first table
                    c.execute(f"PRAGMA table_info({table_name})")
                    columns = c.fetchall()
                    
                    # Store column names for later use
                    self.columns = [col[1] for col in columns]
                    logger.info(f"Found existing table with columns: {self.columns}")
                    
        except Exception as e:
            logger.error(f"Error setting up SQLite database: {e}")
            
    def _load_from_sqlite(self):
        """Load scores from SQLite into memory."""
        logger.info("Loading scores from SQLite database...")
        start_time = time.time()
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                c = conn.cursor()
                
                # Get table name
                c.execute("SELECT name FROM sqlite_master WHERE type='table'")
                table_name = c.fetchone()[0]
                
                # Adapt query based on existing columns
                if hasattr(self, 'columns'):
                    # Map column names based on what's available
                    gene_col = 'gene' if 'gene' in self.columns else 'gene_id'
                    mut_col = 'mutation' if 'mutation' in self.columns else 'variant'
                    score_col = 'score' if 'score' in self.columns else 'provean_score'
                    
                    query = f"SELECT {gene_col}, {mut_col}, {score_col} FROM {table_name}"
                else:
                    # Use default column names
                    query = "SELECT gene, mutation, score FROM provean_scores"
                
                c.execute(query)
                for gene_id, mutation, score in c.fetchall():
                    self.db[gene_id][mutation] = score
                    
            elapsed = time.time() - start_time
            logger.info(f"Loaded {len(self.db)} genes from SQLite in {elapsed:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Error loading from SQLite: {e}")
            # Fallback to empty database
            logger.warning("Starting with empty database")
            
    def _load_from_json(self):
        """Load scores from JSON file into memory."""
        logger.info("Loading scores from JSON file...")
        start_time = time.time()
        
        try:
            if os.path.exists(self.db_path):
                with open(self.db_path, 'r') as f:
                    data = json.load(f)
                    for gene_id, mutations in data.items():
                        self.db[gene_id].update(mutations)
                        
            elapsed = time.time() - start_time
            logger.info(f"Loaded {len(self.db)} genes from JSON in {elapsed:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Error loading from JSON: {e}")
            
    def get_score(self, gene_id: str, mutation: str) -> Optional[float]:
        """Get a score from the in-memory database.
        Thread-safe for reads."""
        try:
            return self.db[gene_id][mutation]
        except KeyError:
            return None
            
    def add_scores_batch(self, gene_id: str, new_scores: Dict[str, float]) -> None:
        """Add multiple scores for a gene in a batch and persist to storage.
        Thread-safe for writes."""
        with self.lock:
            # Update in-memory database
            self.db[gene_id].update(new_scores)
            
            # Persist to storage
            if self.db_type == 'sqlite':
                self._save_to_sqlite(gene_id, new_scores)
            else:
                self._save_to_json()
                
    def _save_to_sqlite(self, gene_id: str, new_scores: Dict[str, float]):
        """Save new scores to SQLite database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                c = conn.cursor()
                c.executemany(
                    "INSERT OR REPLACE INTO provean_scores (gene, mutation, score) VALUES (?, ?, ?)",
                    [(gene_id, mutation, score) for mutation, score in new_scores.items()]
                )
                conn.commit()
        except Exception as e:
            logger.error(f"Error saving to SQLite: {e}")
            
    def _save_to_json(self):
        """Save entire database to JSON file."""
        try:
            # Write to temporary file first
            temp_path = f"{self.db_path}.temp"
            with open(temp_path, 'w') as f:
                json.dump(self.db, f)
            # Atomic replace
            os.replace(temp_path, self.db_path)
        except Exception as e:
            logger.error(f"Error saving to JSON: {e}")
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
    def convert_sqlite_to_json(self, output_path: str):
        """Convert SQLite database to JSON format."""
        if self.db_type != 'sqlite':
            logger.error("Database is not in SQLite format")
            return
            
        try:
            with open(output_path, 'w') as f:
                json.dump(self.db, f)
            logger.info(f"Successfully converted database to JSON: {output_path}")
        except Exception as e:
            logger.error(f"Error converting database to JSON: {e}") 