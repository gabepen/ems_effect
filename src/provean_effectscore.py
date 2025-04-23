import sys
import os
import argparse
from typing import Dict, List, Any, Tuple, Set, Optional
from argparse import Namespace
from pathlib import Path
import yaml
from loguru import logger
import getopt
import json
import random
import subprocess
import numpy as np
from scipy import stats
from tqdm import tqdm
from Bio.Seq import Seq

# Add src to path 
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from modules import parse
from modules.parse import SeqContext
from modules import translate
from modules.provean_db import ProveanScoreDB
from mimetypes import guess_type 
from glob import glob

def parse_args() -> Namespace:
    '''Parse command line arguments.
    
    Returns:
        Namespace: Parsed command line arguments containing:
            - mpileups (str): Path to mpileup files
            - output (str): Working directory path
            - config (str): Path to config file with reference paths
            - exclude (bool): Flag for EMS mutations only
            - skip_parse (bool): Flag to skip parsing step
    '''
    parser = argparse.ArgumentParser(description='Process mpileup files and calculate PROVEAN effect scores')
    parser.add_argument('-m', '--mpileups', required=True,
                        help='Path to mpileup files')
    parser.add_argument('-o', '--output', required=True,
                        help='Working directory')
    parser.add_argument('-c', '--config', required=True,
                        help='Path to config file with reference paths')
    parser.add_argument('-e', '--exclude', action='store_true',
                        help='EMS mutations only')
    parser.add_argument('-s', '--skip_parse', action='store_true',
                        help='Skip parsing step')
    return parser.parse_args()

def load_config(config_path: str) -> Dict[str, str]:
    '''Load reference paths from config file.'''
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config['references'], config['provean']

def setup_directories(outdir: str) -> Dict[str, str]:
    '''Create necessary output directories if they don't exist.
    
    Args:
        outdir (str): Base output directory path
        
    Returns:
        Dict[str, str]: Dictionary containing paths for different outputs:
            - mutpath: Path for nucleotide mutation files
            - aapath: Path for amino acid mutation files
            - provpath: Path for PROVEAN input files
    '''
    directories = [
        outdir,
        f"{outdir}/nuc_muts",
        f"{outdir}/aa_muts",
        f"{outdir}/provean_files",
        f"{outdir}/results",
        f"{outdir}/logs"
    ]
    for directory in directories:
        if not os.path.isdir(directory):
            os.mkdir(directory)
    
    return {
        'mutpath': f"{outdir}/nuc_muts",
        'aapath': f"{outdir}/aa_muts",
        'provpath': f"{outdir}/provean_files",
        'results': f"{outdir}/results",
        'logs': f"{outdir}/logs"
    }

def setup_logging(outdir: str) -> None:
    '''Setup loguru logger with file and console outputs.'''
    log_path = Path(outdir) / 'logs'
    log_path.mkdir(exist_ok=True)
    
    # Remove any existing handlers
    logger.remove()
    
    # Add console handler with DEBUG level
    logger.add(sys.stderr, level="INFO")
    
    # Add file handler with DEBUG level
    logger.add(
        log_path / "provean_effect_{time}.log",
        level="DEBUG",
        rotation="1 day"
    )

def process_mpileups(pile: str, mutpath: str, outdir: str, wolgenome: Any, ems_only: bool) -> None:
    '''Process mpileup files and generate nucleotide mutation JSONs.
    
    Args:
        pile (str): Path to directory containing mpileup files
        mutpath (str): Output directory path for mutation JSON files
        outdir (str): Main output directory for summary files
        wolgenome (SeqContext): Genome context object containing sequence and annotation info
        ems_only (bool): Flag to process only EMS mutations
    '''
    piles = glob(pile + '/*_filtered.txt')
    base_counts = {}
    context_counts = {}
    all_intergenic = {}  # Store intergenic counts for all samples
    
    for mpileup in piles: 
        sample = Path(mpileup).stem
        base_counts[sample] = {'A':0, 'T':0, 'G':0, 'C':0}
        context_counts[sample] = {}
        
        logger.info(f"Processing {sample} mpileup")

        # Process mutations and count bases/contexts
        nuc_muts, contexts, intergenic_counts = parse.parse_mpile(
            mpileup, 
            wolgenome, 
            ems_only, 
            base_counts[sample],
            context_counts[sample]
        )
        
        # Store this sample's intergenic counts
        all_intergenic[sample] = intergenic_counts
        
        # Save mutation data
        with open(f"{mutpath}/{sample}.json", 'w') as of:
            json.dump(nuc_muts, of)
    
    # Save summary files to main results directory
    with open(f"{outdir}/results/basecounts.json", 'w') as of:
        json.dump(base_counts, of)
        
    with open(f"{outdir}/results/contextcounts.json", 'w') as of:
        json.dump(context_counts, of)
        
    with open(f"{outdir}/results/intergeniccounts.json", 'w') as of:
        json.dump(all_intergenic, of)  # Save all samples' intergenic counts

def process_mutations(
    nuc_mut_files: List[str],
    wolgenome: SeqContext,
    codon_table: str,
    features: Dict[str, Any],
    paths: Dict[str, str]
) -> None:
    '''Calculate genome-wide mutation statistics.'''
    # Dictionary to collect genome-wide stats for all samples
    all_genome_stats = {}
    
    for js in nuc_mut_files:
        sample = js.split('/')[-1].split('.')[0]
        with open(js) as jf:
            mut_dict = json.load(jf)

            # Only calculate genome stats
            _, genome_stats = translate.convert_mutations(
                mut_dict,
                wolgenome,
                codon_table,
                features
            )
            
            # Store genome stats for this sample
            all_genome_stats[sample] = genome_stats

    # Save genome-wide dN/dS stats
    with open(f"{paths['results']}/genome_dnds.json", 'w') as of:
        json.dump(all_genome_stats, of, indent=2)

def load_filtered_positions(sample: str, paths: Dict[str, str], mpileup_dir: str) -> Dict[str, List[int]]:
    """Load filtered positions for a sample from JSON file.
    
    Args:
        sample (str): Sample name
        paths (Dict[str, str]): Dictionary containing output paths
        mpileup_dir (str): Directory containing mpileup files and filtered positions
        
    Returns:
        Dict[str, List[int]]: Dictionary mapping gene IDs to filtered positions
    """
    filtered_file = os.path.join(mpileup_dir, f"{sample}_positions.json")
    if os.path.exists(filtered_file):
        with open(filtered_file) as f:
            return json.load(f)
    return {}

def nuc_to_provean_score(
    nuc_mutations: Dict[str, Any],
    gene_id: str,
    score_db: ProveanScoreDB,
    provean_config: Dict[str, Any],
    paths: Dict[str, str],
    features: Dict[str, Any],
    wolgenome: SeqContext,
    codon_table: str,
    aa_mutations_dict: Optional[Dict] = None
) -> float:
    """Convert nucleotide mutations to amino acid mutations and calculate PROVEAN score."""
    
    for mut in nuc_mutations['mutations'].keys():
        pos, change = mut.split('_')

    # Convert nucleotide mutations to amino acid mutations
    aa_muts, _ = translate.convert_mutations(
        {gene_id: nuc_mutations},
        wolgenome,
        codon_table,
        features
    )
    
    for mut in aa_muts[gene_id]['mutations'].keys():
        pos, change = mut.split('_')
        logger.debug(f"  Codon {pos}: {change}")
        
    # Convert to HGVS format
    hgvs_dict = translate.prep_provean(
        aa_muts,
        wolgenome,
        gene_id,
        paths['provpath'],
        features
    )
    
    # Store in aa_mutations_dict if provided
    if aa_mutations_dict is not None:
        aa_mutations_dict[gene_id] = hgvs_dict[gene_id]
    
    # Calculate PROVEAN scores
    total_score = 0
    new_mutations = []
    
    for hgvs_mut in hgvs_dict[gene_id]['mutations']:
        score = score_db.get_score(gene_id, hgvs_mut)
        if score is not None:
            total_score += score
        else:
            new_mutations.append(hgvs_mut)
    
    # Calculate new scores if needed
    if new_mutations:
        logger.info(f"Calculating {len(new_mutations)} new scores for {gene_id}")
        logger.debug("New mutations to score:")
        for mut in new_mutations[:5]:  # Show first 5
            logger.debug(f"  {mut}")
        if len(new_mutations) > 5:
            logger.debug("  ...")
            
        # Write new variants to file
        var_file = os.path.join(paths['provpath'], f"{gene_id}.new.var")
        with open(var_file, 'w') as f:
            for mutation in new_mutations:
                f.write(f"{mutation}\n")
                
        # Get script path for running PROVEAN
        script_dir = os.path.dirname(os.path.abspath(__file__))
        subp_provean_path = os.path.join(script_dir, "subp_provean.sh")
        
        try:
            # Run PROVEAN
            cmd = [
                subp_provean_path,
                gene_id,
                paths['provpath'],
                str(provean_config['num_threads']),
                provean_config['data_dir'],
                provean_config['executable']
            ]
            subprocess.run(cmd, check=True)
            
            # Read and process new scores
            csv_file = os.path.join(paths['provpath'], f"{gene_id}.csv")
            new_scores = {}
            if os.path.exists(csv_file):
                with open(csv_file, 'r') as scores:
                    for line in scores:
                        if line.strip() and ',' in line:
                            mutation, score = line.strip().split(',')
                            if score.strip():
                                score = float(score)
                                score_db.add_score(gene_id, mutation, score)
                                new_scores[mutation] = score
                                total_score += score
                
                logger.info(f"Added {len(new_scores)} new scores to database")
                logger.debug("New scores:")
                for mut, score in list(new_scores.items())[:5]:  # Show first 5
                    logger.debug(f"  {mut}: {score}")
                if len(new_scores) > 5:
                    logger.debug("  ...")
                                
        except subprocess.CalledProcessError as e:
            logger.error(f"Error running PROVEAN for gene {gene_id}: {e}")
            return 0
        finally:
            # Cleanup temporary files
            if os.path.exists(var_file):
                os.remove(var_file)
            if os.path.exists(csv_file):
                os.remove(csv_file)
            
    return total_score

def permute_mutations(
    nuc_mutations: Dict[str, Any],
    gene_id: str, 
    filtered_positions: List[int],
    score_db: ProveanScoreDB,
    provean_config: Dict[str, Any],
    paths: Dict[str, str],
    features: Dict[str, Any],
    wolgenome: SeqContext,
    codon_table: str,
    n_permutations: int = 1000,
    pbar: Optional[tqdm] = None
) -> List[float]:
    """Randomly place nucleotide mutations and calculate PROVEAN scores."""
    
    gene_length = nuc_mutations['gene_len']
    nuc_muts = list(nuc_mutations['mutations'].keys())
    n_mutations = len(nuc_muts)
    
    # Get gene sequence
    gene_seq = str(wolgenome.genome[features[gene_id].start:features[gene_id].end])
    if features[gene_id].strand == -1:
        gene_seq = str(Seq(gene_seq).reverse_complement())
    
    # Store original valid positions for each base type
    original_valid_positions = {'C': [], 'G': []}
    for pos in range(gene_length):
        if pos not in filtered_positions:  # Skip filtered positions
            base = gene_seq[pos]
            if base in original_valid_positions:
                original_valid_positions[base].append(pos)
    
    permuted_scores = []
    for _ in range(n_permutations):
        # Reset valid positions from original for each permutation
        valid_positions = {
            'C': original_valid_positions['C'][:],  # Make copy
            'G': original_valid_positions['G'][:]
        }
        
        permuted_muts = {
            'mutations': {},
            'gene_len': nuc_mutations['gene_len'],
            'avg_cov': nuc_mutations['avg_cov']
        }
        
        # Place each mutation at a valid position for its type
        for orig_mut in nuc_muts:
            ref_base = orig_mut.split('_')[1].split('>')[0]
            if valid_positions[ref_base]:  # If we have valid positions for this base
                pos_idx = random.randrange(len(valid_positions[ref_base]))
                new_pos = valid_positions[ref_base].pop(pos_idx)
                mut_type = orig_mut.split('_')[1]
                permuted_muts['mutations'][f"{new_pos + 1}_{mut_type}"] = 1
        
        if permuted_muts['mutations']:
            score = nuc_to_provean_score(
                permuted_muts,
                gene_id,
                score_db,
                provean_config,
                paths,
                features,
                wolgenome,
                codon_table,
                None
            )
            permuted_scores.append(score)
        
        if pbar is not None:
            pbar.update(1)
    
    return permuted_scores

def calculate_provean_scores(
    mutation_jsons: List[str],
    score_db: ProveanScoreDB,
    paths: Dict[str, str],
    provean_config: Dict[str, Any],
    features: Dict[str, Any],
    wolgenome: SeqContext,
    codon_table: str,
    mpileup_dir: str
) -> Dict[str, Dict[str, Any]]:
    '''Calculate PROVEAN scores and perform permutation tests.'''
    results = {}
    permutation_results = {}
    
    # Process each sample with progress bar
    for js in tqdm(mutation_jsons, desc="Processing samples"):
        sample = js.split('/')[-1].split('.')[0]
        logger.info(f"Processing sample: {sample}")
        results[sample] = {}
        permutation_results[sample] = {}
        
        # Dictionary to store amino acid mutations for this sample
        aa_mutations = {}
        
        # Load filtered positions from correct directory
        filtered_positions = load_filtered_positions(sample, paths, mpileup_dir)
        logger.info(f"Loaded {len(filtered_positions)} filtered positions for {sample}")
        
        # Load original nucleotide mutations
        nuc_mut_file = os.path.join(paths['mutpath'], f"{sample}.json")
        with open(nuc_mut_file) as f:
            nuc_mutations = json.load(f)
        
        # Process each gene with progress bar
        for gene in tqdm(nuc_mutations, desc=f"Processing genes for {sample}", leave=False):
            # Initialize results
            results[sample][gene] = {
                'effect': 0,
                'gene_len': nuc_mutations[gene]['gene_len'],
                'avg_cov': nuc_mutations[gene]['avg_cov'],
                'filtered_positions': len(filtered_positions.get(gene, []))
            }
            
            # Calculate PROVEAN score for original mutations
            original_score = nuc_to_provean_score(
                nuc_mutations[gene],
                gene,
                score_db,
                provean_config,
                paths,
                features,
                wolgenome,
                codon_table,
                aa_mutations  # Store amino acid mutations for this gene
            )
            results[sample][gene]['effect'] = original_score
            
            # Perform permutation test using original nucleotide mutations
            permuted_scores = []
            with tqdm(total=1000, desc=f"Permutations for {gene}", leave=False) as pbar:
                permuted_scores = permute_mutations(
                    nuc_mutations[gene],  # Pass original nucleotide mutations
                    gene,
                    filtered_positions.get(gene, []),
                    score_db,
                    provean_config,
                    paths,
                    features,
                    wolgenome,
                    codon_table,
                    n_permutations=1000,
                    pbar=pbar
                )
            
            # Store permutation results
            if permuted_scores:
                permutation_results[sample][gene] = {
                    'scores': permuted_scores,
                    'mean': np.mean(permuted_scores),
                    'std': np.std(permuted_scores),
                    'percentile': stats.percentileofscore(permuted_scores, original_score)
                }
        
        # Save amino acid mutations for this sample
        with open(f"{paths['provpath']}/{sample}_aa_mutations.json", 'w') as f:
            json.dump(aa_mutations, f, indent=2)
    
    # Save permutation results
    permutation_file = os.path.join(paths['results'], 'permutation_results.json')
    with open(permutation_file, 'w') as f:
        json.dump(permutation_results, f, indent=2)
        
    return results

def normalize_scores(sample: str, results: Dict[str, Dict[str, Any]]) -> None:
    '''Normalize effect scores for each gene.
    
    Args:
        sample (str): Sample identifier
        results (Dict[str, Dict[str, Any]]): Results dictionary to normalize
            Contains gene-level dictionaries with 'effect', 'gene_len', and 'avg_cov'
            
    Returns:
        None: Results dictionary is modified in place
    '''
    for gene in results[sample]:
        effect_score = results[sample][gene]['effect'] * 1000
        _score = effect_score / results[sample][gene]['gene_len']
        if results[sample][gene]['avg_cov'] != 0:
            results[sample][gene]['effect'] = _score / results[sample][gene]['avg_cov']
        else:
            results[sample][gene]['effect'] = 0

def main() -> None:
    '''Main function to run the PROVEAN effect score calculation pipeline.
    
    Workflow:
    1. Parse command line arguments
    2. Set up output directories
    3. Initialize genome context and load reference data
    4. Process mpileup files to get nucleotide mutations (if not skipped)
    5. Convert nucleotide mutations to amino acid mutations
    6. Calculate and normalize PROVEAN scores
    7. Save results to JSON file
    
    Returns:
        None: Results are written to output files
    '''
    args = parse_args()
    
    # Setup paths and directories
    paths = setup_directories(args.output.rstrip('/'))
    
    # Setup logging first
    setup_logging(args.output.rstrip('/'))
    logger.info("Starting PROVEAN effect score calculation pipeline")
    
    # Load reference paths from config
    refs, provean_config = load_config(args.config)
    
    # Initialize PROVEAN score database once
    logger.info("Initializing PROVEAN score database")
    score_db = ProveanScoreDB(refs['prov_score_db'])
    
    # Log configuration settings
    logger.info("Configuration settings:")
    logger.info(f"Reference files:")
    logger.info(f"  Genome: {refs['genomic_fna']}")
    logger.info(f"  Annotation: {refs['annotation']}")
    logger.info(f"  Codon table: {refs['codon_table']}")
    logger.info(f"  PROVEAN score database: {refs['prov_score_db']}")
    logger.info(f"PROVEAN settings:")
    logger.info(f"  Data directory: {provean_config['data_dir']}")
    logger.info(f"  Executable: {provean_config['executable']}")
    logger.info(f"  Threads: {provean_config['num_threads']}")
    logger.info(f"Run parameters:")
    logger.info(f"  Input mpileups: {args.mpileups}")
    logger.info(f"  Output directory: {args.output}")
    logger.info(f"  EMS mutations only: {args.exclude}")
    logger.info(f"  Skip parsing: {args.skip_parse}")
    
    # Initialize genome context
    wolgenome = parse.SeqContext(refs['genomic_fna'], refs['annotation'])
    gene_seqs = wolgenome.gene_seqs()
    features = wolgenome.genome_features()
    
    # Add masking stats to output
    stats = {
        'total_genome_length': len(wolgenome.genome),
        'masked_bases': len(wolgenome.genome) - sum(wolgenome.overlap_mask),
        'percent_masked': (1 - sum(wolgenome.overlap_mask)/len(wolgenome.genome)) * 100
    }
    with open(f"{args.output}/masking_stats.json", 'w') as f:
        json.dump(stats, f)
        
    # Process mpileups if not skipped
    if not args.skip_parse:
        process_mpileups(args.mpileups, paths['mutpath'], args.output.rstrip('/'), wolgenome, args.exclude)
    
    # Process mutations
    #nuc_mut_files = glob(paths['mutpath'] + '/*.json')
    #process_mutations(nuc_mut_files, wolgenome, refs['codon_table'], features, paths)
    
    # Calculate PROVEAN scores with permutation testing
    mutation_jsons = glob(paths['mutpath'] + '/*.json')
    results = calculate_provean_scores(
        mutation_jsons,
        score_db,
        paths,
        provean_config,
        features,
        wolgenome,
        refs['codon_table'],
        args.mpileups
    )
    
    # Save final results
    with open(f"{args.output.rstrip('/')}/results/normalized_scores.json", 'w') as of:
        json.dump(results, of)

    

    logger.success("Pipeline completed successfully")

if __name__ == '__main__':
    main()