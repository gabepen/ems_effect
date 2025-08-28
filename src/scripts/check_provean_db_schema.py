import sys
import sqlite3
from loguru import logger

def check_schema(db_path):
    try:
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        # Print CREATE TABLE statement
        c.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='provean_scores'")
        create_stmt = c.fetchone()
        if create_stmt:
            print("\nCREATE TABLE statement:")
            print(create_stmt[0])
        else:
            print("Table 'provean_scores' does not exist.")
            return
        # Print columns and types
        c.execute("PRAGMA table_info(provean_scores)")
        columns = c.fetchall()
        print("\nColumns:")
        for col in columns:
            print(f"  {col[1]} ({col[2]})")
        conn.close()
    except Exception as e:
        logger.error(f"Error checking schema: {e}")

def main():
    if len(sys.argv) != 2:
        print("Usage: python check_provean_db_schema.py <path_to_db>")
        sys.exit(1)
    db_path = sys.argv[1]
    check_schema(db_path)

if __name__ == "__main__":
    main() 