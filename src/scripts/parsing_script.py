import os
import json
import sys
import argparse
from typing import Dict, List, Any, Tuple
from pathlib import Path
from glob import glob
import gzip
from functools import partial
from mimetypes import guess_type 
from datetime import datetime
from Bio import SeqIO
from BCBio import GFF
from tqdm import tqdm

def parse_args() -> argparse.Namespace:
    '''Parse command line arguments.
    
    Returns:
        Namespace: Parsed command line arguments containing:
            - expdir (str): Directory containing mpileup files
            - outdir (str): Output directory name (defaults to timestamp)
    '''
    parser = argparse.ArgumentParser(description='Parse mpileup files and generate mutation JSONs')
    parser.add_argument('-e', '--expdir', required=True,
                      help='Experiment directory with mpileup files')
    parser.add_argument('-o', '--outdir', 
                      default=datetime.now().strftime("%d-%m-%Y_%H:%M:%S"),
                      help='Output directory name')
    return parser.parse_args()

def setup_directories(outdir: str) -> Dict[str, str]:
    '''Create necessary output directories.
    
    Args:
        outdir (str): Base output directory name
        
    Returns:
        Dict[str, str]: Dictionary containing output paths
    '''
    base_dir = Path.cwd() / 'nuc_jsons' / outdir
    paths = {
        'base': base_dir,
        'counts': base_dir / 'counts'
    }
    
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
        
    return paths

def check_mutations(
    entry: List[str], 
    genestart: int, 
    sample: str,
    counts: Dict[str, Dict[str, int]]
) -> Tuple[Dict[str, int], int]:
    '''Process a mpileup entry to identify mutations.
    
    Args:
        entry (List[str]): Split line from mpileup file
        genestart (int): Start position of current gene
        sample (str): Sample identifier
        counts (Dict[str, Dict[str, int]]): Running count of base occurrences
        
    Returns:
        Tuple containing:
            - Dict[str, int]: Dictionary of mutations and their counts
            - int: Read depth at this position
    '''
    muts = {}    
    bp = str(int(entry[1]) - int(genestart)) + '_'
    ref = entry[2]
    depth = int(entry[3])

    counts[sample][ref] += depth

    reads = enumerate(entry[4])
    r_iter = iter(reads)
    
    for i, r in reads:
        if r not in ['.', 'N', '$', '*']:
            mut = None
            if r == '+':
                insert = next(r_iter)[1]
                mut = f"{bp}{ref}>"
                for _ in range(int(insert)):
                    mut += next(r_iter)[1]
            elif r == '-':
                delete = next(r_iter)[1]
                mut = f"{bp}del"
                for _ in range(int(delete)):
                    mut += next(r_iter)[1]
            else:
                mut = f"{bp}{ref}>{r}"
                
            if mut:
                muts[mut] = muts.get(mut, 0) + 1

    return muts, depth

def process_mpileup(
    mpileup: str,
    gff_record: Any,
    counts: Dict[str, Dict[str, int]]
) -> Dict[str, Dict[str, Any]]:
    '''Process a single mpileup file to identify mutations in genes.
    
    Args:
        mpileup (str): Path to mpileup file
        gff_record: GFF record containing gene annotations
        counts (Dict[str, Dict[str, int]]): Running count of base occurrences
        
    Returns:
        Dict[str, Dict[str, Any]]: Dictionary containing mutations and metadata per gene
    '''
    mutations = {}
    encoding = guess_type(mpileup)[1]
    _open = partial(gzip.open, mode='rt') if encoding == 'gzip' else open
    
    sample = Path(mpileup).stem
    counts[sample] = {'A': 0, 'T': 0, 'G': 0, 'C': 0}
    
    with _open(mpileup) as mf:
        for feat in tqdm(gff_record.features):
            if feat.type == 'gene':
                refid = feat.qualifiers['Dbxref'][0].split(':')[-1]
                mutations[refid] = {
                    'mutations': {},
                    'avg_cov': 0,
                    'gene_len': 0
                }
                
                loc = feat.location
                depths = []
                
                for line in mf:
                    entry = line.split()
                    pos = int(entry[1]) - 1
                    
                    if pos in loc:
                        muts, depth = check_mutations(entry, loc.start, sample, counts)
                        depths.append(depth)
                        mutations[refid]['mutations'].update(muts)
                    elif pos > loc.end:
                        break
                        
                mutations[refid]['avg_cov'] = sum(depths) / len(depths) if depths else 0
                mutations[refid]['gene_len'] = loc.end - loc.start
                
    return mutations

def main() -> None:
    '''Main function to run the mutation parsing pipeline.
    
    Workflow:
    1. Parse command line arguments
    2. Set up output directories
    3. Process each mpileup file
    4. Save mutation data and base counts
    '''
    args = parse_args()
    paths = setup_directories(args.outdir)
    
    # Reference paths
    genomic_fna = "/more_storage/russ/EMS/circleseq/references/GCF_000008025.1_ASM802v1_genomic.fna.gz"
    annotation = "/more_storage/russ/EMS/circleseq/references/GCF_000008025.1_ASM802v1_genomic.gff"
    
    # Process mpileup files
    mpileups = glob(f"{args.expdir}/*.mpileup*")
    counts: Dict[str, Dict[str, int]] = {}
    
    with open(annotation) as gff:
        for rec in GFF.parse(gff):
            print(f"Processing {rec.id}, length {len(rec.seq)}, with {len(rec.features)} features")
            
            for mpileup in mpileups:
                print(f"Processing {Path(mpileup).name}")
                mutations = process_mpileup(mpileup, rec, counts)
                
                # Save mutations
                output_path = paths['base'] / f"{Path(mpileup).stem}.json"
                with open(output_path, 'w') as f:
                    json.dump(mutations, f)
    
    # Save base counts
    with open(paths['counts'] / 'totalcounts.json', 'w') as f:
        json.dump(counts, f)

if __name__ == '__main__':
    main() 