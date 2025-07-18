#!/usr/bin/env python3

import sys
from pathlib import Path

def main():
    # Read the gid_to_wdid.tsv file
    gid_file = Path("/storage1/gabe/ems_effect_code/kegg_analysis/ptest_niave_moreeffect.gid_to_wdid.tsv")
    
    if not gid_file.exists():
        print(f"Error: File {gid_file} does not exist")
        sys.exit(1)
    
    # Extract WD_Ids that were successfully found
    wd_ids = []
    with open(gid_file) as f:
        next(f)  # Skip header
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                wd_id = parts[1]
                if wd_id not in ['NOT_FOUND', 'NOT_IN_CACHE', 'GENE_NAME_NOT_FOUND', 'GENE_NAME_SEARCH_FAILED']:
                    wd_ids.append(wd_id)
    
    print(f"Found {len(wd_ids)} WD_Ids")
    print("\nWD_Ids:")
    for wd_id in wd_ids:
        print(wd_id)
    
    # Save to file
    output_file = Path("/storage1/gabe/ems_effect_code/kegg_analysis/wd_ids_list.txt")
    with open(output_file, "w") as f:
        for wd_id in wd_ids:
            f.write(f"{wd_id}\n")
    print(f"\nWD_Ids saved to: {output_file}")

if __name__ == "__main__":
    main() 