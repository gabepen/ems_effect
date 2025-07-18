#!/usr/bin/env python3

import json
import requests
import time
import sys
import re
from pathlib import Path
import argparse

def load_gene_info_cache(json_file):
    """Load the gene information cache from JSON file"""
    with open(json_file, 'r') as f:
        return json.load(f)

def extract_wd_ids_from_aliases(aliases_str):
    """Extract WD_Ids from the otheraliases field"""
    if not aliases_str:
        return []
    
    aliases = aliases_str.split(', ')
    if not aliases:
        return []
    
    # Get the last alias
    last_alias = aliases[-1].strip()
    
    # Check if it starts with WD_ but not WD_RS
    if last_alias.startswith('WD_') and not last_alias.startswith('WD_RS'):
        return [last_alias]
    
    return []

def submit_wdid_mapping(wd_ids, from_db="Gene_Name", to_db="UniProtKB"):
    """Submit WD_Ids to UniProt for mapping"""
    url = "https://rest.uniprot.org/idmapping/run"
    data = {
        "from": from_db,
        "to": to_db,
        "ids": ",".join(wd_ids)
    }
    print(f"Submitting {len(wd_ids)} WD_Ids to UniProt for mapping...")
    print(f"Mapping from: {from_db} to: {to_db}")
    print(f"First 5 WD_Ids: {wd_ids[:5]}")
    response = requests.post(url, data=data)
    response.raise_for_status()
    result = response.json()
    print(f"Submit response: {result}")
    return result["jobId"]

def search_gene_name_uniprot(gene_name):
    """Search for a gene name in UniProt and find Wolbachia wMel entries"""
    url = "https://rest.uniprot.org/uniprotkb/search"
    query = f"gene:{gene_name} AND organism_id:163164"  # 163164 is Wolbachia pipientis wMel
    params = {
        "query": query,
        "fields": [
            "accession",
            "gene_names",
            "organism_name",
            "protein_name",
            "length"
        ],
        "sort": "accession desc",
        "includeIsoform": "false",
        "size": "50"
        }
    headers = {
        "accept": "application/json"
    }
    
    try:
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()
        results = response.json()
        if results and results.get("results") and len(results["results"]) > 0:
            # Return the first result (most relevant)
            result = results["results"][0]
            
            # Extract data in the same format as WD_Id mapping results
            from_id = f"GENE_NAME:{gene_name}"
            entry = result.get("primaryAccession", "")
            entry_name = result.get("uniProtkbId", entry)  # Use accession if uniProtkbId not available
            reviewed = "reviewed" if result.get("entryType", "").startswith("UniProtKB reviewed") else "unreviewed"
            
            # Extract protein name - try recommendedName first, then alternativeNames
            protein_names = ""
            protein_desc = result.get("proteinDescription", {})
            if "recommendedName" in protein_desc and protein_desc["recommendedName"]:
                protein_names = protein_desc["recommendedName"].get("fullName", {}).get("value", "")
            elif "alternativeNames" in protein_desc and protein_desc["alternativeNames"]:
                protein_names = protein_desc["alternativeNames"][0].get("fullName", {}).get("value", "")
            
            # Extract gene names
            gene_names = ""
            if "genes" in result and result["genes"]:
                gene_data = result["genes"][0]
                if "geneName" in gene_data and gene_data["geneName"]:
                    gene_names = gene_data["geneName"].get("value", "")
                elif "orderedLocusNames" in gene_data and gene_data["orderedLocusNames"]:
                    gene_names = gene_data["orderedLocusNames"][0].get("value", "")
            
            organism = result.get("organism", {}).get("scientificName", "")
            length = result.get("sequence", {}).get("length", "")
            
            # Return as TSV line in the same format as WD_Id results
            tsv_line = f"{from_id}\t{entry}\t{entry_name}\t{reviewed}\t{protein_names}\t{gene_names}\t{organism}\t{length}"
            return tsv_line
            
    except Exception as e:
        print(f"Error searching for gene name {gene_name}: {e}")
        return None

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
    parser = argparse.ArgumentParser(description='Convert gene IDs to UniProt accessions using JSON gene info cache')
    parser.add_argument('gene_id_file', help='File containing gene IDs (one per line)')
    parser.add_argument('json_cache', help='JSON file containing gene information cache')
    parser.add_argument('-o', '--output', help='Output file (default: input_file.uniprot.tsv)')
    parser.add_argument('--from-db', default='Gene_Name', 
                       help='Source database type for UniProt mapping (default: Gene_Name)')
    parser.add_argument('--to-db', default='UniProtKB',
                       help='Target database type for UniProt mapping (default: UniProtKB)')
    parser.add_argument('--wdid-only', action='store_true',
                       help='Only extract WD_Ids and save to file, skip UniProt mapping')
    
    args = parser.parse_args()
    
    gene_file = Path(args.gene_id_file)
    json_file = Path(args.json_cache)
    output_file = Path(args.output) if args.output else gene_file.with_suffix('.uniprot.tsv')
    
    if not gene_file.exists():
        print(f"Error: Gene ID file {gene_file} does not exist")
        sys.exit(1)
    
    if not json_file.exists():
        print(f"Error: JSON cache file {json_file} does not exist")
        sys.exit(1)
    
    # Load gene IDs
    with open(gene_file) as f:
        gene_ids = [line.strip() for line in f if line.strip()]
    
    print(f"Read {len(gene_ids)} gene IDs from {gene_file}")
    print(f"First 5 gene IDs: {gene_ids[:5]}")
    
    # Load gene info cache
    print(f"Loading gene information cache from {json_file}")
    gene_cache = load_gene_info_cache(json_file)
    print(f"Loaded {len(gene_cache)} gene records from cache")
    
    # Convert gene IDs to WD_Ids
    conversions = []
    wd_ids = []
    gene_name_mappings = []  # Store gene name mappings for later processing
    
    for gene_id in gene_ids:
        if gene_id in gene_cache:
            gene_info = gene_cache[gene_id]
            aliases = gene_info.get('otheraliases', '')
            extracted_wd_ids = extract_wd_ids_from_aliases(aliases)
            
            if extracted_wd_ids:
                # Use the first WD_Id found
                wd_id = extracted_wd_ids[0]
                conversions.append({
                    'gene_id': gene_id,
                    'wd_id': wd_id,
                    'gene_name': gene_info.get('name', 'Unknown'),
                    'organism': gene_info.get('organism', {}).get('scientificname', 'Unknown'),
                    'mapping_type': 'WD_Id'
                })
                wd_ids.append(wd_id)
                print(f"Gene {gene_id} -> {wd_id} ({gene_info.get('name', 'Unknown')})")
            else:
                # No WD_Id found, try gene name
                gene_name = gene_info.get('name', '')
                if gene_name and gene_name != 'Unknown':
                    print(f"Gene {gene_id} -> No WD_Id found, trying gene name: {gene_name}")
                    gene_name_mappings.append({
                        'gene_id': gene_id,
                        'gene_name': gene_name,
                        'organism': gene_info.get('organism', {}).get('scientificname', 'Unknown')
                    })
                else:
                    conversions.append({
                        'gene_id': gene_id,
                        'wd_id': 'NOT_FOUND',
                        'gene_name': gene_info.get('name', 'Unknown'),
                        'organism': gene_info.get('organism', {}).get('scientificname', 'Unknown'),
                        'mapping_type': 'NOT_FOUND'
                    })
                    print(f"Gene {gene_id} -> No WD_Id found, no gene name available")
        else:
            conversions.append({
                'gene_id': gene_id,
                'wd_id': 'NOT_IN_CACHE',
                'gene_name': 'Unknown',
                'organism': 'Unknown',
                'mapping_type': 'NOT_IN_CACHE'
            })
            print(f"Gene {gene_id} -> Not found in cache")
    
    # Process gene name mappings
    print(f"\nProcessing {len(gene_name_mappings)} gene names for UniProt search...")
    for mapping in gene_name_mappings:
        gene_id = mapping['gene_id']
        gene_name = mapping['gene_name']
        
        print(f"Searching for gene name: {gene_name}")
        uniprot_result = search_gene_name_uniprot(gene_name)
        
        if uniprot_result:
            # Parse the TSV result (format: from_id, entry, entry_name, reviewed, protein_names, gene_names, organism, length)
            parts = uniprot_result.split('\t')
            if len(parts) >= 8:
                from_id = parts[0]
                entry = parts[1]
                entry_name = parts[2]
                reviewed = parts[3]
                protein_names = parts[4]
                gene_names = parts[5]
                organism = parts[6]
                length = parts[7]
                
                conversions.append({
                    'gene_id': gene_id,
                    'wd_id': from_id,
                    'gene_name': gene_name,
                    'organism': organism,
                    'mapping_type': 'Gene_Name',
                    'uniprot_accession': entry,
                    'protein_name': protein_names,
                    'length': length
                })
                print(f"Gene {gene_id} -> Found via gene name: {entry} ({protein_names})")
            else:
                conversions.append({
                    'gene_id': gene_id,
                    'wd_id': 'GENE_NAME_SEARCH_FAILED',
                    'gene_name': gene_name,
                    'organism': mapping['organism'],
                    'mapping_type': 'Gene_Name_Failed'
                })
                print(f"Gene {gene_id} -> Gene name search failed")
        else:
            conversions.append({
                'gene_id': gene_id,
                'wd_id': 'GENE_NAME_NOT_FOUND',
                'gene_name': gene_name,
                'organism': mapping['organism'],
                'mapping_type': 'Gene_Name_NotFound'
            })
            print(f"Gene {gene_id} -> Gene name not found in UniProt")
    
    # Summary of conversion
    wd_id_count = sum(1 for c in conversions if c['mapping_type'] == 'WD_Id')
    gene_name_count = sum(1 for c in conversions if c['mapping_type'] == 'Gene_Name')
    not_found_count = sum(1 for c in conversions if c['mapping_type'] in ['NOT_FOUND', 'NOT_IN_CACHE', 'Gene_Name_NotFound', 'Gene_Name_Failed'])
    
    print(f"\nConversion Summary:")
    print(f"Total gene IDs: {len(gene_ids)}")
    print(f"Gene IDs with WD_Ids: {wd_id_count}")
    print(f"Gene IDs mapped via gene name: {gene_name_count}")
    print(f"Gene IDs not found: {not_found_count}")
    
    # Save conversion results
    conversion_file = gene_file.with_suffix('.gid_to_wdid.tsv')
    with open(conversion_file, "w") as f:
        f.write("Gene_ID\tWD_Id\tGene_Name\tOrganism\tMapping_Type\tUniProt_Accession\tProtein_Name\tLength\n")
        for conv in conversions:
            uniprot_acc = conv.get('uniprot_accession', '')
            protein_name = conv.get('protein_name', '')
            length = conv.get('length', '')
            f.write(f"{conv['gene_id']}\t{conv['wd_id']}\t{conv['gene_name']}\t{conv['organism']}\t{conv['mapping_type']}\t{uniprot_acc}\t{protein_name}\t{length}\n")
    print(f"Conversion results saved to: {conversion_file}")
    
    if args.wdid_only:
        print("WD_Id extraction complete. Skipping UniProt mapping.")
        return
    
    # Collect all mappings that need UniProt processing
    wd_id_mappings = [c for c in conversions if c['mapping_type'] == 'WD_Id']
    gene_name_mappings = [c for c in conversions if c['mapping_type'] == 'Gene_Name']
    
    if not wd_id_mappings and not gene_name_mappings:
        print("No mappings found that need UniProt processing.")
        return
    
    # Process WD_Ids through UniProt mapping in batches
    if wd_id_mappings:
        wd_ids = [c['wd_id'] for c in wd_id_mappings]
        print(f"\nMapping {len(wd_ids)} WD_Ids to UniProt in batches...")
        
        # Process in batches of 50 (UniProt might have limits)
        batch_size = 50
        all_results = []
        
        for i in range(0, len(wd_ids), batch_size):
            batch = wd_ids[i:i+batch_size]
            print(f"Processing batch {i//batch_size + 1}/{(len(wd_ids) + batch_size - 1)//batch_size} ({len(batch)} WD_Ids)")
            
            job_id = submit_wdid_mapping(batch, from_db=args.from_db, to_db=args.to_db)
            print(f"Job ID: {job_id}")
            
            # Poll for completion of this batch
            max_attempts = 60
            attempt = 0
            
            while attempt < max_attempts:
                status = check_id_mapping_status(job_id)
                
                # Check if results are already in the status response
                if "results" in status and status["results"]:
                    print(f"Results found in status response for batch {i//batch_size + 1}!")
                    results_data = status["results"]
                    print(f"Found {len(results_data)} results in this batch")
                    
                    # Convert to TSV format
                    for result in results_data:
                        from_id = result.get("from", "")
                        to_data = result.get("to", {})
                        entry = to_data.get("primaryAccession", "")
                        entry_name = to_data.get("uniProtkbId", "")
                        reviewed = "reviewed" if to_data.get("entryType", "").startswith("UniProtKB reviewed") else "unreviewed"
                        
                        # Extract protein name - try recommendedName first, then submissionNames
                        protein_names = ""
                        protein_desc = to_data.get("proteinDescription", {})
                        if "recommendedName" in protein_desc and protein_desc["recommendedName"]:
                            protein_names = protein_desc["recommendedName"].get("fullName", {}).get("value", "")
                        elif "submissionNames" in protein_desc and protein_desc["submissionNames"]:
                            protein_names = protein_desc["submissionNames"][0].get("fullName", {}).get("value", "")
                        
                        gene_names = ""
                        if "genes" in to_data and to_data["genes"]:
                            gene_data = to_data["genes"][0]
                            if "orderedLocusNames" in gene_data and gene_data["orderedLocusNames"]:
                                gene_names = gene_data["orderedLocusNames"][0].get("value", "")
                        organism = to_data.get("organism", {}).get("scientificName", "")
                        length = to_data.get("sequence", {}).get("length", "")
                        
                        tsv_line = f"{from_id}\t{entry}\t{entry_name}\t{reviewed}\t{protein_names}\t{gene_names}\t{organism}\t{length}"
                        all_results.append(tsv_line)
                    break
                    
                job_status = None
                if "jobStatus" in status:
                    job_status = status["jobStatus"]
                elif "status" in status:
                    job_status = status["status"]
                
                if job_status == "RUNNING" or job_status == "PENDING":
                    print(f"Batch {i//batch_size + 1} in progress... (status: {job_status}) waiting 5s")
                    time.sleep(5)
                elif job_status == "FINISHED":
                    print(f"Batch {i//batch_size + 1} complete!")
                    break
                elif job_status == "ERROR" or job_status == "FAILED":
                    print(f"Batch {i//batch_size + 1} failed with status: {job_status}")
                    print(f"Error details: {status}")
                    break
                else:
                    print(f"Unknown status for batch {i//batch_size + 1}: {job_status}")
                    time.sleep(5)
                
                attempt += 1
            
            if attempt >= max_attempts:
                print(f"Timeout waiting for batch {i//batch_size + 1} completion")
                continue
            
            # If we didn't get results from status, try downloading them
            if not any(r.startswith(batch[0]) for r in all_results):
                print(f"Downloading results for batch {i//batch_size + 1}...")
                try:
                    downloaded_results = get_id_mapping_results(job_id)
                    if downloaded_results.strip():
                        lines = downloaded_results.strip().split('\n')
                        if len(lines) > 1:  # Has header + data
                            all_results.extend(lines[1:])  # Skip header, add data lines
                except requests.exceptions.HTTPError as e:
                    print(f"Error downloading results for batch {i//batch_size + 1}: {e}")
            
            print(f"Total results so far: {len(all_results)}")
        print(f"Results in this batch: {[r.split('\t')[0] for r in all_results[-len(results_data):]] if 'results_data' in locals() else 'N/A'}")
        
        # Store WD_Id mapping results
        wd_id_results = []
        if all_results:
            wd_id_results = all_results
    

    
    # Combine WD_Id mapping results with gene name mappings
    all_results = []
    
    # Add WD_Id mapping results
    if wd_id_results:
        all_results.extend(wd_id_results)
    
    # Add gene name mappings (already have UniProt accessions)
    for mapping in conversions:
        if mapping['mapping_type'] == 'Gene_Name' and 'uniprot_accession' in mapping:
            # Create a result line in the same format as WD_Id results
            result_line = f"{mapping['wd_id']}\t{mapping['uniprot_accession']}\t{mapping['uniprot_accession']}\tunreviewed\t{mapping['protein_name']}\t{mapping['gene_name']}\t{mapping['organism']}\t{mapping['length']}"
            all_results.append(result_line)
    
    # Save combined results
    if all_results:
        header = "From\tEntry\tEntry Name\tReviewed\tProtein names\tGene Names\tOrganism\tLength"
        final_results = header + "\n" + "\n".join(all_results)
        
        with open(output_file, "w") as f:
            f.write(final_results)
        print(f"Combined results saved to {output_file}")
        print(f"Total mappings: {len(all_results)}")
        
        # Debug summary
        wd_id_mappings_count = len([r for r in all_results if r.split('\t')[0].startswith('WD_')])
        gene_name_mappings_count = len([r for r in all_results if r.split('\t')[0].startswith('GENE_NAME:')])
        print(f"WD_Id mappings: {wd_id_mappings_count}")
        print(f"Gene name mappings: {gene_name_mappings_count}")
        print(f"Expected WD_Ids: {len(wd_ids)}")
        print(f"Expected gene names: {len([c for c in conversions if c['mapping_type'] == 'Gene_Name'])}")
    else:
        print("Warning: No results found!")
        print("This could mean:")
        print("1. The WD_Ids are not recognized by UniProt")
        print("2. No mappings were found for these WD_Ids")
        print("3. The API returned an error")
        print("4. Try using a different from_db parameter")
        
        with open(output_file, "w") as f:
            f.write("From\tEntry\tEntry Name\tReviewed\tProtein names\tGene Names\tOrganism\tLength\n")
        print(f"Created empty output file with header: {output_file}")

if __name__ == "__main__":
    main() 