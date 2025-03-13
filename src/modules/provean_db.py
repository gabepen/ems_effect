import sqlite3
from pathlib import Path
from typing import Dict, Optional, List, Tuple
from contextlib import contextmanager
from loguru import logger

class ProveanScoreDB:
    def __init__(self, db_path: str, timeout: int = 30):
        '''Initialize PROVEAN score database.
        
        Args:
            db_path: Path to SQLite database file
            timeout: Seconds to wait when database is locked (default: 30)
        '''
        self.db_path = db_path
        self.timeout = timeout  # Initialize timeout before _setup_db
        self._setup_db()
    
    def _setup_db(self):
        '''Create database table if it doesn't exist.'''
        conn = self._connect()  # Use _connect directly
        try:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS provean_scores (
                    gene TEXT,
                    mutation TEXT,
                    score REAL,
                    PRIMARY KEY (gene, mutation)
                )
            ''')
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_gene 
                ON provean_scores(gene)
            ''')
            conn.commit()
        finally:
            conn.close()
    
    def _connect(self):
        '''Create connection with busy timeout.'''
        conn = sqlite3.connect(self.db_path, timeout=self.timeout)
        conn.execute("PRAGMA journal_mode=WAL")  # Write-Ahead Logging for better concurrency
        return conn
    
    def get_score(self, gene: str, mutation: str) -> Optional[float]:
        '''Get PROVEAN score for a specific mutation.
        
        Args:
            gene: Gene identifier
            mutation: HGVS mutation string
            
        Returns:
            Score if found, None if not in database
        '''
        with self._connect() as conn:
            result = conn.execute(
                'SELECT score FROM provean_scores WHERE gene=? AND mutation=?',
                (gene, mutation)
            ).fetchone()
            return result[0] if result else None
    
    def get_gene_scores(self, gene: str) -> Dict[str, float]:
        '''Get all stored scores for a gene.
        
        Args:
            gene: Gene identifier
            
        Returns:
            Dictionary mapping mutations to scores
        '''
        with self._connect() as conn:
            results = conn.execute(
                'SELECT mutation, score FROM provean_scores WHERE gene=?',
                (gene,)
            ).fetchall()
            return {mut: score for mut, score in results}
    
    def add_scores(self, scores: Dict[str, Dict[str, float]]):
        '''Add multiple scores in single transaction.'''
        with self._connect() as conn:
            try:
                conn.execute("BEGIN IMMEDIATE")
                conn.executemany(
                    'INSERT OR REPLACE INTO provean_scores VALUES (?, ?, ?)',
                    [
                        (gene, mutation, score)
                        for gene, mutations in scores.items()
                        for mutation, score in mutations.items()
                    ]
                )
                conn.commit()
            except sqlite3.OperationalError as e:
                logger.warning(f"Database locked, retrying: {e}")
                conn.rollback()
                raise
    
    def add_score(self, gene: str, mutation: str, score: float):
        '''Add single score with transaction.'''
        with self._connect() as conn:
            try:
                conn.execute("BEGIN IMMEDIATE")  # Get write lock immediately
                conn.execute(
                    'INSERT OR REPLACE INTO provean_scores VALUES (?, ?, ?)',
                    (gene, mutation, score)
                )
                conn.commit()
            except sqlite3.OperationalError as e:
                logger.warning(f"Database locked, retrying: {e}")
                conn.rollback()
                raise 