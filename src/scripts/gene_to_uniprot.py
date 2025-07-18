#!/usr/bin/env python3

import requests
import time
import sys
import argparse
from pathlib import Path

def submit_id_mapping(ids, from_db="GeneID", to_db="UniProtKB"):
    url = "https://rest.uniprot.org/idmapping/run"
    data = {
        "from": from_db,
        "to": to_db,
        "ids": ",".join(ids)
    }
    print(f"Submitting request to: {url}")
    print(f"Mapping from: {from_db} to: {to_db}")
    print(f"Data: {data}")
    response = requests.post(url, data=data)
    response.raise_for_status()
    result = response.json()
    print(f"Submit response: {result}")
    return result["jobId"]

def check_id_mapping_status(job_id):
    url = f"https://rest.uniprot.org/idmapping/status/{job_id}"
    response = requests.get(url)
    response.raise_for_status()
    return response.json()

def get_id_mapping_results(job_id):
    url = f"https://rest.uniprot.org/idmapping/uniprotkb/results/{job_id}"
    print(f"Downloading results from: {url}")
    response = requests.get(url, params={"format": "tsv"})
    response.raise_for_status()
    return response.text

def main():
    parser = argparse.ArgumentParser(description='Map gene IDs to UniProt using UniProt ID mapping service')
    parser.add_argument('gene_file', help='File containing gene IDs (one per line)')
    parser.add_argument('-o', '--output', help='Output file (default: input_file.uniprot.tsv)')
    parser.add_argument('--from-db', default='GeneID', 
                       help='Source database type (default: GeneID)')
    parser.add_argument('--to-db', default='UniProtKB',
                       help='Target database type (default: UniProtKB)')
    parser.add_argument('--list-dbs', action='store_true',
                       help='List available database types and exit')
    
    args = parser.parse_args()
    
    if args.list_dbs:
        print("Available database types for UniProt ID mapping:")
        print("Common 'from' databases:")
        print("  GeneID - NCBI Gene IDs (e.g., 1, 2, 3)")
        print("  Ensembl - Ensembl Gene IDs (e.g., ENSG00000139618)")
        print("  Ensembl_Transcript - Ensembl Transcript IDs")
        print("  Ensembl_Protein - Ensembl Protein IDs")
        print("  RefSeq_NT - RefSeq Nucleotide IDs (e.g., NM_123456.1)")
        print("  RefSeq_Protein - RefSeq Protein IDs (e.g., NP_123456.1)")
        print("  EMBL - EMBL IDs")
        print("  EMBL-CDS - EMBL CDS IDs")
        print("\nCommon 'to' databases:")
        print("  UniProtKB - UniProt Knowledgebase")
        print("  UniProtKB-Swiss-Prot - Swiss-Prot entries only")
        print("  UniProtKB-TrEMBL - TrEMBL entries only")
        print("  Ensembl - Ensembl database")
        print("  RefSeq_NT - RefSeq Nucleotide")
        print("  RefSeq_Protein - RefSeq Protein")
        return
    
    gene_file = Path(args.gene_file)
    output_file = Path(args.output) if args.output else gene_file.with_suffix('.uniprot.tsv')

    if not gene_file.exists():
        print(f"Error: File {gene_file} does not exist")
        sys.exit(1)

    with open(gene_file) as f:
        gene_ids = [line.strip() for line in f if line.strip()]

    print(f"Read {len(gene_ids)} gene IDs from {gene_file}")
    print(f"First 5 gene IDs: {gene_ids[:5]}")
    
    if not gene_ids:
        print("Error: No gene IDs found in file")
        sys.exit(1)

    print(f"Submitting {len(gene_ids)} gene IDs to UniProt for mapping...")
    job_id = submit_id_mapping(gene_ids, from_db=args.from_db, to_db=args.to_db)
    print(f"Job ID: {job_id}")

    # Poll for completion
    max_attempts = 60  # 5 minutes max wait time
    attempt = 0
    
    while attempt < max_attempts:
        status = check_id_mapping_status(job_id)
        print(f"Status response: {status}")  # Debug output
        
        # Check for different possible status keys
        job_status = None
        if "jobStatus" in status:
            job_status = status["jobStatus"]
        elif "status" in status:
            job_status = status["status"]
        elif "results" in status:
            # If results are already available, job is finished
            job_status = "FINISHED"
        
        if job_status == "RUNNING" or job_status == "PENDING":
            print(f"Mapping in progress... (status: {job_status}) waiting 5s")
            time.sleep(5)
        elif job_status == "FINISHED":
            print("Mapping complete!")
            break
        elif job_status == "ERROR" or job_status == "FAILED":
            print(f"Job failed with status: {job_status}")
            print(f"Error details: {status}")
            sys.exit(1)
        else:
            print(f"Unknown status: {job_status}")
            print(f"Full response: {status}")
            # Continue waiting if status is unclear
            time.sleep(5)
        
        attempt += 1
    
    if attempt >= max_attempts:
        print("Timeout waiting for job completion")
            sys.exit(1)

    # Download results
    print("Downloading results...")
    try:
    results = get_id_mapping_results(job_id)
        print(f"Raw results length: {len(results)}")
        print(f"First 500 characters of results: {results[:500]}")
        
        if not results.strip():
            print("Warning: Results are empty!")
            print("This could mean:")
            print("1. The gene IDs are not in the expected format")
            print("2. No mappings were found for these gene IDs")
            print("3. The API returned an error")
            print(f"4. The from_db='{args.from_db}' is not correct for your ID type")
            
            # Try to get more info about the job
            final_status = check_id_mapping_status(job_id)
            print(f"Final job status: {final_status}")
            
            # Create empty output file with header
            with open(output_file, "w") as f:
                f.write("From\tTo\n")
            print(f"Created empty output file with header: {output_file}")
        else:
    with open(output_file, "w") as f:
        f.write(results)
    print(f"Results saved to {output_file}")
            
    except requests.exceptions.HTTPError as e:
        print(f"Error downloading results: {e}")
        print("Trying to get status again...")
        final_status = check_id_mapping_status(job_id)
        print(f"Final status: {final_status}")
        sys.exit(1)

if __name__ == "__main__":
    main() 