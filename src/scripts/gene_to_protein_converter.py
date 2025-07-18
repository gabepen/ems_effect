#!/usr/bin/env python3

"""
NCBI Gene ID to Protein ID Converter
====================================

This script converts NCBI gene IDs to protein IDs using the NCBI E-utilities API.
It handles rate limiting and provides detailed output.

Usage:
    python gene_to_protein_converter.py <input_file> [output_file]

Example:
    python gene_to_protein_converter.py genes.txt proteins.txt
"""

import sys
import time
import requests
import xml.etree.ElementTree as ET
from pathlib import Path
import argparse

def convert_gene_to_protein(gene_id, session, max_retries=3):
    """
    Convert a single gene ID to protein IDs using NCBI E-utilities.
    
    Args:
        gene_id (str): NCBI gene ID
        session: requests.Session object for connection reuse
        max_retries (int): Maximum number of retry attempts
        
    Returns:
        list: List of protein IDs
    """
    url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi"
    params = {
        'dbfrom': 'gene',
        'db': 'protein',
        'id': gene_id,
        'retmode': 'xml'
    }
    
    for attempt in range(max_retries):
        try:
            # Increase timeout and add retry delay
            timeout = 30 + (attempt * 10)  # Progressive timeout: 30s, 40s, 50s
            response = session.get(url, params=params, timeout=timeout)
            response.raise_for_status()
            
            # Parse XML response
            root = ET.fromstring(response.text)
            protein_ids = []
            
            # Find all Id elements in the response
            for id_elem in root.findall('.//Id'):
                protein_ids.append(id_elem.text)
            
            return protein_ids
            
        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                print(f"  Timeout (attempt {attempt + 1}/{max_retries}), retrying in {2 ** attempt}s...")
                time.sleep(2 ** attempt)  # Exponential backoff: 1s, 2s, 4s
            else:
                print(f"  Error: Timeout after {max_retries} attempts")
                return []
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                print(f"  Request error (attempt {attempt + 1}/{max_retries}): {e}, retrying...")
                time.sleep(2 ** attempt)
            else:
                print(f"  Error: {e}")
                return []
        except ET.ParseError as e:
            print(f"  XML parsing error: {e}")
            return []
    
    return []

def get_protein_accession(protein_id, session, max_retries=2):
    """
    Get protein accession number for a protein ID.
    
    Args:
        protein_id (str): NCBI protein ID
        session: requests.Session object for connection reuse
        max_retries (int): Maximum number of retry attempts
        
    Returns:
        str: Protein accession number or empty string if not found
    """
    url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {
        'db': 'protein',
        'id': protein_id,
        'rettype': 'acc',
        'retmode': 'text'
    }
    
    for attempt in range(max_retries):
        try:
            timeout = 20 + (attempt * 5)  # Progressive timeout: 20s, 25s
            response = session.get(url, params=params, timeout=timeout)
            response.raise_for_status()
            return response.text.strip()
            
        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                time.sleep(1)  # Short delay before retry
            else:
                return ""
        except requests.exceptions.RequestException:
            if attempt < max_retries - 1:
                time.sleep(1)
            else:
                return ""
    
    return ""

def main():
    parser = argparse.ArgumentParser(
        description="Convert NCBI gene IDs to protein IDs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python gene_to_protein_converter.py genes.txt
  python gene_to_protein_converter.py genes.txt output_proteins.txt
        """
    )
    
    parser.add_argument('input_file', help='Input file containing gene IDs (one per line)')
    parser.add_argument('output_file', nargs='?', 
                       help='Output file (default: input_file_proteins.txt)')
    parser.add_argument('--delay', type=float, default=1.0,
                       help='Delay between API calls in seconds (default: 1.0)')
    parser.add_argument('--max-genes', type=int,
                       help='Maximum number of genes to process (for testing)')
    
    args = parser.parse_args()
    
    # Validate input file
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: Input file '{args.input_file}' not found.")
        sys.exit(1)
    
    # Set output file
    if args.output_file:
        output_path = Path(args.output_file)
    else:
        output_path = input_path.with_suffix('').with_suffix('_proteins.txt')
    
    print("NCBI Gene ID to Protein ID Converter")
    print("====================================")
    print(f"Input file: {input_path}")
    print(f"Output file: {output_path}")
    print(f"Delay between requests: {args.delay}s")
    print()
    
    # Read input genes
    try:
        with open(input_path, 'r') as f:
            genes = [line.strip() for line in f if line.strip()]
    except Exception as e:
        print(f"Error reading input file: {e}")
        sys.exit(1)
    
    print(f"Found {len(genes)} genes to convert")
    print(f"First 5 genes: {', '.join(genes[:5])}")
    print()
    
    # Limit genes if specified
    if args.max_genes:
        genes = genes[:args.max_genes]
        print(f"Limited to first {len(genes)} genes for testing")
        print()
    
    # Convert genes to proteins
    results = []
    converted_count = 0
    
    # Use session for connection reuse
    with requests.Session() as session:
        for i, gene_id in enumerate(genes, 1):
            print(f"Converting gene {i} of {len(genes)}: {gene_id}")
            
            protein_ids = convert_gene_to_protein(gene_id, session)
            
            if protein_ids:
                print(f"  Found {len(protein_ids)} protein ID(s)")
                
                for protein_id in protein_ids:
                    # Get protein accession
                    accession = get_protein_accession(protein_id, session)
                    
                    results.append({
                        'gene_id': gene_id,
                        'protein_id': protein_id,
                        'accession': accession
                    })
                    
                    converted_count += 1
            else:
                print("  No protein IDs found")
                results.append({
                    'gene_id': gene_id,
                    'protein_id': None,
                    'accession': None
                })
            
            # Add delay to avoid rate limiting
            if i < len(genes):  # Don't delay after the last gene
                time.sleep(args.delay)
    
    # Save results
    try:
        with open(output_path, 'w') as f:
            # Write header
            f.write("gene_id\tprotein_id\taccession\n")
            
            # Write data
            for result in results:
                gene_id = result['gene_id']
                protein_id = result['protein_id'] or ''
                accession = result['accession'] or ''
                f.write(f"{gene_id}\t{protein_id}\t{accession}\n")
        
        print(f"\nResults saved to: {output_path}")
        
    except Exception as e:
        print(f"Error saving results: {e}")
        sys.exit(1)
    
    # Summary
    print("\nConversion Summary:")
    print(f"Total genes processed: {len(genes)}")
    print(f"Genes with protein IDs: {sum(1 for r in results if r['protein_id'] is not None)}")
    print(f"Total protein IDs found: {converted_count}")
    print(f"Genes without protein IDs: {sum(1 for r in results if r['protein_id'] is None)}")
    
    if converted_count > 0:
        print("\nFirst 10 conversions:")
        for i, result in enumerate(results[:10]):
            if result['protein_id']:
                print(f"  {result['gene_id']} -> {result['protein_id']} ({result['accession']})")
    
    print("\nConversion completed!")

if __name__ == "__main__":
    main() 