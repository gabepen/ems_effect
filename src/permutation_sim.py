import json
import argparse
from pathlib import Path
from typing import Dict, List, Any
from tqdm import tqdm
import random
import sys
import os
import time
import psutil
import subprocess
from glob import glob
import yaml

from modules import parse
from modules import translate
from provean_effectscore import process_mutations, calculate_provean_scores

def parse_args() -> argparse.Namespace:
    '''Parse command line arguments.'''
    parser = argparse.ArgumentParser(description='Run EMS mutation permutation simulation')
    parser.add_argument('-m', '--mpileups', required=True,
                      help='Path to mpileup files')
    parser.add_argument('-p', '--permutations', required=True, type=int,
                      help='Number of permutations to run')
    parser.add_argument('-o', '--output', required=True,
                      help='Working directory')
    parser.add_argument('-c', '--config', required=True,
                      help='Path to config file with reference paths')
    parser.add_argument('-e', '--email', 
                      help='Email for monitoring')
    parser.add_argument('-r', '--resume', action='store_true',
                      help='Resume from partial results')
    return parser.parse_args()

def setup_directories(outdir: str) -> Dict[str, str]:
    '''Create necessary output directories.'''
    paths = {
        'nuc_muts': f"{outdir}/nuc_muts",
        'shuffled': f"{outdir}/shuffled", 
        'aa_muts': f"{outdir}/aa_muts_shuffled",
        'provean': f"{outdir}/provean_files",
        'results': f"{outdir}/results"
    }
    
    for path in paths.values():
        Path(path).mkdir(parents=True, exist_ok=True)
        
    return paths

def precompute_mutation_sites(gene_seqs: Dict[str, str]) -> Dict[str, Dict[str, List[int]]]:
    '''Precompute C/G sites for each gene to avoid repeated scanning.
    
    Args:
        gene_seqs: Dictionary mapping gene IDs to sequences
        
    Returns:
        Dictionary mapping gene IDs to their C and G site positions
    '''
    site_cache = {}
    for gene, seq in gene_seqs.items():
        site_cache[gene] = {
            'C': [i for i, base in enumerate(seq) if base == 'C'],
            'G': [i for i, base in enumerate(seq) if base == 'G']
        }
    return site_cache

def shuffle_mutations(mut_dict: Dict[str, Any], site_cache: Dict[str, Dict[str, List[int]]]) -> Dict[str, Any]:
    '''Shuffle EMS mutations to random C/G sites within each gene.'''
    shuffled = {}
    
    for gene in tqdm(mut_dict):
        shuffled[gene] = {
            'mutations': {},
            'avg_cov': mut_dict[gene]['avg_cov'],
            'gene_len': mut_dict[gene]['gene_len']
        }
        
        # Use precomputed sites
        C_sites = site_cache[gene]['C']
        G_sites = site_cache[gene]['G']
                
        # Shuffle mutations
        for mut in mut_dict[gene]['mutations']:
            transition = mut.split('_')[1]
            if transition in ['C>T', 'G>A']:
                mutation_count = mut_dict[gene]['mutations'][mut]
                
                if transition == 'C>T' and C_sites:
                    n = random.randint(0, len(C_sites)-1)
                    pos = C_sites[n]
                    ems_mut = f"{pos}_C>T"
                elif transition == 'G>A' and G_sites:
                    n = random.randint(0, len(G_sites)-1)
                    pos = G_sites[n]
                    ems_mut = f"{pos}_G>A"
                else:
                    continue
                    
                if ems_mut in shuffled[gene]['mutations']:
                    shuffled[gene]['mutations'][ems_mut] += mutation_count
                else:
                    shuffled[gene]['mutations'][ems_mut] = mutation_count
                    
    return shuffled

def load_partial_results(outdir: str) -> tuple[Dict, int]:
    '''Load partial results if resuming.'''
    try:
        partial_files = glob(f"{outdir}/results/*.partial.json")
        if not partial_files:
            raise FileNotFoundError
        
        with open(partial_files[0]) as f:
            p = int(f.name.split('.')[-3])
            results = json.load(f)
        return results, p
    except FileNotFoundError:
        print('No partial results found to resume, remove -r flag.')
        sys.exit(1)

def load_config(config_path: str) -> Dict[str, str]:
    '''Load reference paths from config file.'''
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config['references']

def main() -> None:
    '''Main function to run permutation simulation.'''
    args = parse_args()
    paths = setup_directories(args.output.rstrip('/'))
    
    # Load reference paths from config
    refs = load_config(args.config)
    
    # Initialize genome context
    wolgenome = parse.SeqContext(refs['genomic_fna'], refs['annotation'])
    gene_seqs = wolgenome.gene_seqs()
    features = wolgenome.genome_features()
    
    # Precompute mutation sites
    site_cache = precompute_mutation_sites(gene_seqs)
    
    # Resume or start new
    if args.resume:
        results, p = load_partial_results(args.output)
    else:
        results = {}
        p = 0
    
    # Run permutations
    while p < args.permutations:
        print(f"\nPermutation {p+1}/{args.permutations}")
        tic = time.perf_counter()
        
        # Process each sample
        for mut_file in Path(paths['nuc_muts']).glob('*.json'):
            sample = mut_file.stem
            
            # Load and shuffle mutations
            with open(mut_file) as f:
                mut_dict = json.load(f)
            shuffled = shuffle_mutations(mut_dict, site_cache)
            
            # Save shuffled mutations
            with open(f"{paths['shuffled']}/{sample}.json", 'w') as f:
                json.dump(shuffled, f)
        
        # Convert mutations and calculate PROVEAN scores
        nuc_muts_shuffled = glob(f"{paths['shuffled']}/*.json")
        process_mutations(nuc_muts_shuffled, wolgenome, refs['codon_table'], features, paths)
        
        # Calculate PROVEAN scores
        provean_jsons = glob(f"{paths['provean']}/*.json")
        scores = calculate_provean_scores(provean_jsons, refs['prov_score_table'], paths)
        
        # Store results
        for sample in scores:
            if sample not in results:
                results[sample] = {}
            for gene in scores[sample]:
                if gene not in results[sample]:
                    results[sample][gene] = []
                results[sample][gene].append(scores[sample][gene]['effect'])
        
        toc = time.perf_counter()
        print(f"Permutation {p} took {(toc-tic)/60:.4f} minutes")
        
        p += 1
        
        # Save intermediate results every 100 permutations
        if p % 100 == 0:
            # Remove previous partial save
            for f in glob(f"{paths['results']}/*.partial.json"):
                os.remove(f)
                
            # Save new partial results    
            with open(f"{paths['results']}/scored.{p}.partial.json", 'w') as f:
                json.dump(results, f)
        
        # Monitor memory usage
        if p % 1000 == 0:
            process = psutil.Process(os.getpid())
            usage_kb = process.memory_info().rss / 1000
            print(f"Memory usage: {usage_kb}kb")
    
    # Save final results
    with open(f"{paths['results']}/scored.complete.json", 'w') as f:
        json.dump(results, f)

if __name__ == '__main__':
    main() 