import json
import argparse
import gzip
from collections import defaultdict
from Bio import SeqIO
from typing import Union, TextIO
from mimetypes import guess_type

def open_file(filename: str) -> Union[TextIO, gzip.GzipFile]:
    """Open file handling gzip if needed.
    
    Args:
        filename (str): Path to file
        
    Returns:
        File handle: Regular or gzip file handle
    """
    encoding = guess_type(filename)[1]
    if encoding == 'gzip':
        return gzip.open(filename, 'rt')
    return open(filename)

def count_kmers(sequence: str, k: int = 3) -> dict:
    """Count occurrences of all kmers in a sequence.
    
    Args:
        sequence (str): DNA sequence to analyze
        k (int): Size of kmer window (default=3)
        
    Returns:
        dict: Dictionary mapping kmers to their counts
    """
    kmer_counts = defaultdict(int)
    
    # Convert sequence to uppercase
    sequence = sequence.upper()
    
    # Count kmers
    for i in range(len(sequence) - k + 1):
        kmer = sequence[i:i+k]
        # Only count if kmer contains valid bases
        if all(base in 'ACGT' for base in kmer):
            kmer_counts[kmer] += 1
            
    return dict(kmer_counts)

def main():
    parser = argparse.ArgumentParser(description='Count kmers in reference genome')
    parser.add_argument('-r', '--reference', required=True, 
                       help='Path to reference genome FASTA (can be gzipped)')
    parser.add_argument('-k', '--kmer-size', type=int, default=3,
                       help='Size of kmer window (default: 3)')
    parser.add_argument('-o', '--output', required=True, 
                       help='Path to output JSON file')
    args = parser.parse_args()
    
    # Load reference genome
    with open_file(args.reference) as handle:
        genome = str(next(SeqIO.parse(handle, 'fasta')).seq)
    
    # Count kmers
    kmer_counts = count_kmers(genome, args.kmer_size)
    
    # Save to JSON
    with open(args.output, 'w') as f:
        json.dump(kmer_counts, f, indent=2)

if __name__ == '__main__':
    main()
