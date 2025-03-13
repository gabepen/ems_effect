import json
import sys
from pathlib import Path
from typing import Optional

def validate_json(filepath: str, chunk_size: int = 1000000) -> Optional[tuple[int, str]]:
    '''Validate JSON file and find corruption location.
    
    Args:
        filepath: Path to JSON file to validate
        chunk_size: Size of chunks to read for large files
        
    Returns:
        Tuple of (line number, context) if error found, None if valid
    '''
    try:
        # First try loading entire file
        with open(filepath) as f:
            json.load(f)
        print(f"JSON file {filepath} is valid")
        return None
            
    except json.JSONDecodeError as e:
        print(f"\nError in JSON file: {filepath}")
        print(f"Error message: {str(e)}")
        
        # Get context around error
        with open(filepath) as f:
            content = f.read()
        
        # Find line number and position
        line_no = content.count('\n', 0, e.pos) + 1
        print(f"\nError on line {line_no}")
        
        # Get context around error (100 chars before and after)
        start = max(0, e.pos - 100)
        end = min(len(content), e.pos + 100)
        context = content[start:end]
        
        print("\nContext around error:")
        print("..."+ context + "...")
        print(" " * (103) + "^")  # Point to error location
        
        return line_no, context

def main():
    if len(sys.argv) != 2:
        print("Usage: python validate_json.py <path_to_json>")
        sys.exit(1)
        
    json_path = sys.argv[1]
    if not Path(json_path).exists():
        print(f"File not found: {json_path}")
        sys.exit(1)
        
    validate_json(json_path)

if __name__ == "__main__":
    main() 