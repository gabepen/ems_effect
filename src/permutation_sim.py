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
from loguru import logger

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

def setup_logging(outdir: str) -> None:
    '''Setup loguru logger with file and console outputs.'''
    log_path = Path(outdir) / 'logs'
    log_path.mkdir(exist_ok=True)
    
    # Remove any existing handlers
    logger.remove()
    
    # Add console handler with INFO level
    logger.add(sys.stderr, level="INFO")
    
    # Add file handler with DEBUG level
    logger.add(
        log_path / "permutation_sim_{time}.log",
        level="DEBUG",
        rotation="1 day"
    )

def shuffle_mutations(mut_dict: Dict[str, Any], site_cache: Dict[str, Dict[str, List[int]]]) -> Dict[str, Any]:
    '''Shuffle EMS mutations to random C/G sites within each gene.'''
    shuffled = {}
    
    logger.debug(f"Shuffling mutations across {len(mut_dict)} genes")
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
        logger.error('No partial results found to resume, remove -r flag.')
        sys.exit(1)

def load_config(config_path: str) -> tuple[Dict[str, str], Dict[str, str]]:
    '''Load reference paths from config file.'''
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config['references'], config['provean']

def main() -> None:
    '''Main function to run permutation simulation.'''
    args = parse_args()
    
    # Setup logging first
    setup_logging(args.output.rstrip('/'))
    logger.info("Starting permutation simulation")
    
    paths = setup_directories(args.output.rstrip('/'))
    
    # Load reference paths from config
    refs, provean_config = load_config(args.config)
    logger.info("Loaded configuration files")
    
    # Log configuration settings
    logger.info("Configuration settings:")
    logger.info(f"Reference files:")
    logger.info(f"  Genome: {refs['genomic_fna']}")
    logger.info(f"  Annotation: {refs['annotation']}")
    logger.info(f"  Codon table: {refs['codon_table']}")
    logger.info(f"  PROVEAN score table: {refs['prov_score_db']}")
    logger.info(f"PROVEAN settings:")
    logger.info(f"  Data directory: {provean_config['data_dir']}")
    logger.info(f"  Executable: {provean_config['executable']}")
    logger.info(f"  Threads: {provean_config['num_threads']}")
    logger.info(f"Run parameters:")
    logger.info(f"  Number of permutations: {args.permutations}")
    logger.info(f"  Input mpileups: {args.mpileups}")
    logger.info(f"  Output directory: {args.output}")
    logger.info(f"  Resume from partial: {args.resume}")
    
    # Initialize genome context
    logger.info("Initializing genome context")
    wolgenome = parse.SeqContext(refs['genomic_fna'], refs['annotation'])
    gene_seqs = wolgenome.gene_seqs()
    features = wolgenome.genome_features()
    
    # First process mpileups to generate nucleotide mutation JSONs
    logger.info("Processing mpileup files to generate mutation JSONs")
    base_counts = {'A': 0, 'T': 0, 'G': 0, 'C': 0}
    context_counts = {}
    
    # Process mpileups if directory
    if os.path.isdir(args.mpileups):
        mpileup_files = glob(os.path.join(args.mpileups, "*.mpileup*"))
    else:
        mpileup_files = [args.mpileups]
        
    for mpileup in mpileup_files:
        sample = Path(mpileup).stem
        logger.info(f"Processing {sample}")
        
        # Process mutations and count bases/contexts
        nuc_muts, contexts, intergenic_counts = parse.parse_mpile(
            mpileup, 
            wolgenome, 
            True,  # ems_only=True since this is EMS analysis
            base_counts,
            context_counts
        )
        
        # Save mutation data
        with open(f"{paths['nuc_muts']}/{sample}.json", 'w') as of:
            json.dump(nuc_muts, of)
    
    # Save summary files
    with open(f"{paths['results']}/basecounts.json", 'w') as of:
        json.dump(base_counts, of)
        
    with open(f"{paths['results']}/contextcounts.json", 'w') as of:
        json.dump(context_counts, of)
        
    with open(f"{paths['results']}/intergeniccounts.json", 'w') as of:
        json.dump(intergenic_counts, of)
    
    # Precompute mutation sites
    logger.info("Precomputing mutation sites")
    site_cache = precompute_mutation_sites(gene_seqs)
    
    # Resume or start new
    if args.resume:
        results, p = load_partial_results(args.output)
    else:
        results = {}
        p = 0
    
    # Run permutations
    while p < args.permutations:
        logger.info(f"\nStarting permutation {p+1}/{args.permutations}")
        tic = time.perf_counter()
        
        # Process each sample
        for mut_file in Path(paths['nuc_muts']).glob('*.json'):
            sample = mut_file.stem
            
            # Load and shuffle mutations
            logger.info(f"Shuffling mutations for {sample}")   
            with open(mut_file) as f:
                mut_dict = json.load(f)
            shuffled = shuffle_mutations(mut_dict, site_cache)
            
            # Save shuffled mutations
            logger.info(f"Saving shuffled mutations for {sample}")
            with open(f"{paths['shuffled']}/{sample}.json", 'w') as f:
                json.dump(shuffled, f)
        
        # Convert mutations and calculate PROVEAN scores
        logger.info("Converting mutations and calculating PROVEAN scores")
        nuc_muts_shuffled = glob(f"{paths['shuffled']}/*.json")
        process_mutations(nuc_muts_shuffled, wolgenome, refs['codon_table'], features, paths)
        
        # Calculate PROVEAN scores with updated config handling
        provean_jsons = glob(f"{paths['provean']}/*.json")
        scores = calculate_provean_scores(
            provean_jsons, 
            refs['prov_score_db'], 
            paths,
            provean_config
        )
        
        # Store results
        for sample in scores:
            if sample not in results:
                results[sample] = {}
            for gene in scores[sample]:
                if gene not in results[sample]:
                    results[sample][gene] = []
                results[sample][gene].append(scores[sample][gene]['effect'])
        
        toc = time.perf_counter()
        logger.info(f"Permutation {p} took {(toc-tic)/60:.4f} minutes")
        
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
            logger.debug(f"Memory usage: {usage_kb}kb")
    
    # Save final results
    with open(f"{paths['results']}/scored.complete.json", 'w') as f:
        json.dump(results, f)
    
    logger.success("Permutation simulation completed successfully")

if __name__ == '__main__':
    main() 