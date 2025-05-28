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
from concurrent.futures import ProcessPoolExecutor, as_completed
import shutil
from collections import defaultdict
import time

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
    context_counts = {}
    all_intergenic = {}
    
    # Calculate genome-wide base counts once
    genome_base_counts = calculate_genome_base_counts(wolgenome)
    
    # New: Statistics tracking
    all_sample_stats = {}
    all_gene_stats = {}
    
    for mpileup in piles: 
        sample = Path(mpileup).stem
        # Initialize empty dict for this sample's context counts
        context_counts[sample] = {}
        
        logger.info(f"Processing {sample} mpileup")

        # Process mutations and count contexts (no longer tracking base counts per sample)
        nuc_muts, contexts, intergenic_counts = parse.parse_mpile(
            mpileup, 
            wolgenome, 
            ems_only, 
            {},  # Empty dict since we don't need per-sample base counts
            context_counts[sample]
        )
        
        # Store this sample's intergenic counts
        all_intergenic[sample] = intergenic_counts
        
        # NEW: Calculate sample and gene statistics
        sample_stats, gene_stats = calculate_sample_statistics(nuc_muts, sample)
        all_sample_stats[sample] = sample_stats
        all_gene_stats[sample] = gene_stats
        
        # Save mutation data
        with open(f"{mutpath}/{sample}.json", 'w') as of:
            json.dump(nuc_muts, of)
    
    # After processing all samples, normalize context counts
    normalized_contexts = normalize_context_counts(context_counts, wolgenome)
    
    # Add normalized analysis to context_counts
    for sample in normalized_contexts:
        if sample == 'genome_kmer_counts':
            # Add genome-wide counts directly to context_counts
            context_counts[sample] = normalized_contexts[sample]
        else:
            context_counts[sample]['normalized_analysis'] = normalized_contexts[sample]
    
    # Save summary files to main results directory
    # Use genome-wide base counts for all samples
    with open(f"{outdir}/results/basecounts.json", 'w') as of:
        json.dump(genome_base_counts, of)
        
    with open(f"{outdir}/results/contextcounts.json", 'w') as of:
        json.dump(context_counts, of)
        
    with open(f"{outdir}/results/intergeniccounts.json", 'w') as of:
        json.dump(all_intergenic, of)
    
    # NEW: Save statistics tables
    with open(f"{outdir}/results/sample_statistics.json", 'w') as of:
        json.dump(all_sample_stats, of, indent=2)
        
    with open(f"{outdir}/results/gene_statistics.json", 'w') as of:
        json.dump(all_gene_stats, of, indent=2)

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
        Dict[str, List[int]]: Dictionary mapping gene IDs to 0-based relative filtered positions
    """
    filtered_file = os.path.join(mpileup_dir, f"{sample}_filtered_positions.json")
    if os.path.exists(filtered_file):
        with open(filtered_file) as f:
            raw_positions = json.load(f)
            # Convert from 1-based relative to 0-based relative positions
            corrected_positions = {}
            for gene_id, positions in raw_positions.items():
                corrected_positions[gene_id] = [pos - 1 for pos in positions]
            return corrected_positions
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
) -> Tuple[float, float, Dict[str, Any]]:
    """Convert nucleotide mutations to amino acid mutations and calculate PROVEAN score."""
    DELETERIOUS_THRESHOLD = -2.5
    
   
    
    # Verify gene ID
    if gene_id not in features:
        logger.error(f"Gene ID {gene_id} not found in features")
        return 0.0, 0.0, {'deleterious_count': 0, 'total_mutations': 0}
    
    # Convert nucleotide mutations to amino acid mutations
    aa_muts, _ = translate.convert_mutations(
        {gene_id: nuc_mutations},
        wolgenome,
        codon_table,
        features
    )
    

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
    total_score_with_nonsense = 0
    deleterious_count = 0
    deleterious_count_with_nonsense = 0
    total_mutations = 0
    nonsense_count = 0
    new_mutations = []
    failed_mutations = []
    
    for hgvs_mut in hgvs_dict[gene_id]['mutations']:
        if gene_id in score_db.db and hgvs_mut in score_db.db[gene_id]:
            score = score_db.get_score(gene_id, hgvs_mut)
            
            if score is None:
                failed_mutations.append(hgvs_mut)
                continue
                
            total_mutations += 1
            is_nonsense = '>*' in hgvs_mut or 'del' in hgvs_mut
            if is_nonsense:
                nonsense_count += 1
                total_score_with_nonsense += score
                if score < DELETERIOUS_THRESHOLD:
                    deleterious_count_with_nonsense += 1
            else:
                total_score += score
                total_score_with_nonsense += score
                if score < DELETERIOUS_THRESHOLD:
                    deleterious_count += 1
                    deleterious_count_with_nonsense += 1
        else:
            # Mutation not in database, need to calculate score
            new_mutations.append(hgvs_mut)
    
    # Calculate new scores if needed
    if new_mutations:
        # Write new variants to file
        var_file = os.path.join(paths['provpath'], f"{gene_id}.new.var")
        csv_file = os.path.join(paths['provpath'], f"{gene_id}.csv")
        provean_log = os.path.join(paths['provpath'], f"{gene_id}.provean.log")
        
        try:
            with open(var_file, 'w') as f:
                for mutation in new_mutations:
                    f.write(f"{mutation}\n")
                    
            # Get script path for running PROVEAN
            script_dir = os.path.dirname(os.path.abspath(__file__))
            subp_provean_path = os.path.join(script_dir, "subp_provean.sh")
            
            # Debug: Print nucleotide mutations for this gene
            logger.debug(f"Processing gene {gene_id} with mutations:")
            logger.debug(f"  {new_mutations}")
            
            # Add this:
            logger.info(f"Calling PROVEAN for gene {gene_id} with {len(new_mutations)} new mutations")
            
            # Run PROVEAN with output redirected to log file
            with open(provean_log, 'w') as log_file:
                cmd = [
                    subp_provean_path,
                    gene_id,
                    paths['provpath'],
                    str(provean_config['num_threads']),
                    provean_config['data_dir'],
                    provean_config['executable']
                ]
                subprocess.run(cmd, 
                             check=True,
                             stdout=log_file,
                             stderr=subprocess.STDOUT)
            
            # Check for AA mismatch errors
            with open(provean_log, 'r') as log_file:
                log_content = log_file.read()
                if "reference AA does not match" in log_content:
                    for line in log_content.split('\n'):
                        if "reference AA does not match" in line:
                            # Parse the problematic mutation (e.g., "Q136_L276del")
                            problem_mutation = line.split(':')[0].strip()
                            logger.error(f"Gene {gene_id}: Mutation {problem_mutation} failed - {line.strip()}")
                            
                            # Only mark this specific mutation as failed
                            if problem_mutation in new_mutations:
                                failed_mutations.append(problem_mutation)
                                # Add to database as None
                                score_db.add_scores_batch(gene_id, {problem_mutation: None})
                                # Remove from new_mutations
                                new_mutations.remove(problem_mutation)
            
            # Only process scores if mutations didn't fail
            if new_mutations:
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
                                    new_scores[mutation] = score
                                    total_mutations += 1
                                    # Check new mutations for nonsense/deletions
                                    is_nonsense = '>*' in mutation or 'del' in mutation
                                    if is_nonsense:
                                        nonsense_count += 1
                                        total_score_with_nonsense += score
                                    else:
                                        total_score += score
                                        total_score_with_nonsense += score
                                    if score < DELETERIOUS_THRESHOLD:
                                        deleterious_count += 1
                                        deleterious_count_with_nonsense += 1
                
                # Add successful scores to database
                if new_scores:
                    score_db.add_scores_batch(gene_id, new_scores)
                
                # Any mutations that didn't get scores are failed
                scored_mutations = set(new_scores.keys())
                failed_mutations.extend(set(new_mutations) - scored_mutations)
                
                # Add failed mutations to database as None
                failed_scores = {mut: None for mut in failed_mutations}
                if failed_scores:
                    score_db.add_scores_batch(gene_id, failed_scores)
            
        except Exception as e:
            logger.error(f"Error running PROVEAN for gene {gene_id}: {e}")
            return 0, 0, {'deleterious_count': 0, 'total_mutations': 0}
        finally:
            # Cleanup temporary files
            for file in [var_file, csv_file, provean_log]:
                if os.path.exists(file):
                    try:
                        os.remove(file)
                    except:
                        pass
    
    stats = {
        'deleterious_count': deleterious_count,
        'total_mutations': total_mutations,
        'failed_mutations': len(failed_mutations),
        'nonsense_count': nonsense_count
    }
    
    return total_score, total_score_with_nonsense, stats

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
) -> Dict[str, Any]:
    """Randomly place nucleotide mutations and calculate PROVEAN scores."""
    
    permuted_scores = []
    permuted_deleterious_counts = []
    
    gene_length = nuc_mutations['gene_len']
    nuc_muts = list(nuc_mutations['mutations'].keys())
    n_mutations = len(nuc_muts)
    
    # Get gene sequence
    gene_seq = str(wolgenome.genome[features[gene_id].start:features[gene_id].end])
    is_reverse = features[gene_id].strand == -1
    if is_reverse:
        gene_seq = str(Seq(gene_seq).reverse_complement())
    
    # Store original valid positions for each base type
    original_valid_positions = {'C': [], 'G': []}
    for pos in range(gene_length):
        if pos not in filtered_positions:  # Skip filtered positions
            if pos < len(gene_seq):  # Ensure position is within gene sequence
                base = gene_seq[pos]
                if base in original_valid_positions:
                    original_valid_positions[base].append(pos)
    
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
                
                # Verify the reference base at this position
                if new_pos < len(gene_seq) and gene_seq[new_pos] == ref_base:
                    # For reverse strand genes, we need to adjust the mutation type
                    if is_reverse:
                        if mut_type == 'C>T':
                            mut_type = 'G>A'
                        elif mut_type == 'G>A':
                            mut_type = 'C>T'
                    
                    # Use 1-based position for mutation key
                    # For reverse strand, we need to convert the position
                    if is_reverse:
                        # Convert position to genomic coordinates
                        genomic_pos = len(gene_seq) - new_pos
                    else:
                        genomic_pos = new_pos + 1
                    
                    permuted_muts['mutations'][f"{genomic_pos}_{mut_type}"] = 1
                else:
                    # Skip this mutation if reference base doesn't match
                    logger.debug(f"Reference mismatch at position {new_pos}: expected {ref_base}, found {gene_seq[new_pos] if new_pos < len(gene_seq) else 'out of bounds'}")
                    continue
        
        if permuted_muts['mutations']:
            score, score_with_nonsense, mutation_stats = nuc_to_provean_score(
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
            permuted_deleterious_counts.append(mutation_stats['deleterious_count'])
        
        if pbar is not None:
            pbar.update(1)
    
    return {
        'scores': permuted_scores,
        'deleterious_counts': permuted_deleterious_counts,
        'mean': np.mean(permuted_scores) if permuted_scores else 0,
        'std': np.std(permuted_scores) if permuted_scores else 0,
        'mean_deleterious': np.mean(permuted_deleterious_counts) if permuted_deleterious_counts else 0,
        'std_deleterious': np.std(permuted_deleterious_counts) if permuted_deleterious_counts else 0
    }

def process_gene_permutations(
    gene_id: str,
    sample_mutations: Dict[str, Dict[str, Any]],
    sample_filtered_positions: Dict[str, List[int]],
    score_db_path: str,
    provean_config: Dict[str, Any],
    paths: Dict[str, str],
    features: Dict[str, Any],
    wolgenome: SeqContext,
    codon_table: str,
    n_permutations: int = 1000,
    min_mutations: int = 3
) -> Tuple[str, Dict[str, Any]]:
    """Process permutations for a single gene across all samples."""
    # Re-initialize DB in each process
    score_db = ProveanScoreDB(score_db_path)
    try:
        total_mutations = sum(
            len(mutations['mutations']) 
            for mutations in sample_mutations.values()
        )
        
        if total_mutations < min_mutations:
            return gene_id, create_empty_result(sample_mutations, total_mutations)
        
        process_id = os.getpid()
        gene_temp_dir = os.path.join(paths['provpath'], f"{gene_id}_{process_id}")
        os.makedirs(gene_temp_dir, exist_ok=True)
        gene_paths = paths.copy()
        gene_paths['provpath'] = gene_temp_dir

        # --- Timing: Original score calculation ---
        t0 = time.time()
        original_total_score = 0
        original_total_score_with_nonsense = 0
        original_sample_scores = {}
        original_total_stats = {'deleterious_count': 0, 'total_mutations': 0, 'nonsense_count': 0}
        
        for sample, mutations in sample_mutations.items():
            score, score_with_nonsense, stats = nuc_to_provean_score(
                mutations,
                gene_id,
                score_db,
                provean_config,
                gene_paths,
                features,
                wolgenome,
                codon_table
            )
            original_sample_scores[sample] = (score, score_with_nonsense, stats)
            original_total_score += score
            original_total_score_with_nonsense += score_with_nonsense
            original_total_stats['deleterious_count'] += stats['deleterious_count']
            original_total_stats['total_mutations'] += stats['total_mutations']
            original_total_stats['nonsense_count'] += stats.get('nonsense_count', 0)
        t1 = time.time()
        logger.info(f"[{gene_id}] Original score calculation took {t1-t0:.2f} seconds.")

        # --- Timing: Permutation loop ---
        permuted_total_scores = []
        permuted_total_scores_with_nonsense = []
        permuted_total_deleterious = []
        t2 = time.time()
        for perm_idx in range(n_permutations):
            perm_start = time.time()
            perm_total_score = 0
            perm_total_score_with_nonsense = 0
            perm_total_deleterious = 0
            
            for sample, mutations in sample_mutations.items():
                filtered_pos = sample_filtered_positions.get(sample, [])
                permuted_muts = permute_single_sample(
                    mutations,
                    gene_id,
                    filtered_pos,
                    features,
                    wolgenome
                )
                # --- Timing: Score calculation for permutation ---
                score_start = time.time()
                if permuted_muts['mutations']:
                    score, score_with_nonsense, stats = nuc_to_provean_score(
                        permuted_muts,
                        gene_id,
                        score_db,
                        provean_config,
                        gene_paths,
                        features,
                        wolgenome,
                        codon_table
                    )
                    perm_total_score += score
                    perm_total_score_with_nonsense += score_with_nonsense
                    perm_total_deleterious += stats['deleterious_count']
                score_end = time.time()
                # Optional: log per-sample score time
                # logger.debug(f"[{gene_id}] Perm {perm_idx} sample {sample} score: {score_end-score_start:.4f}s")
            permuted_total_scores.append(perm_total_score)
            permuted_total_scores_with_nonsense.append(perm_total_score_with_nonsense)
            permuted_total_deleterious.append(perm_total_deleterious)
            perm_end = time.time()
            if perm_idx % 100 == 0:
                logger.info(f"[{gene_id}] Permutation {perm_idx} took {perm_end-perm_start:.3f}s")
        t3 = time.time()
        logger.info(f"[{gene_id}] All {n_permutations} permutations took {t3-t2:.2f} seconds.")

        # Calculate p-value based on summed scores
        p_value = sum(s <= original_total_score for s in permuted_total_scores) / len(permuted_total_scores)
        p_value_with_nonsense = sum(s <= original_total_score_with_nonsense for s in permuted_total_scores_with_nonsense) / len(permuted_total_scores_with_nonsense)
        
        # Get gene length from first sample (should be same for all)
        first_sample = next(iter(sample_mutations.values()))
        gene_len = first_sample['gene_len']

        result = {
            'effect': original_total_score,
            'effect_with_nonsense': original_total_score_with_nonsense,
            'total_mutations': total_mutations,
            'gene_len': gene_len,  # Keep gene length
            'mutation_stats': {
                'deleterious_count': original_total_stats['deleterious_count'],
                'total_mutations': original_total_stats['total_mutations'],
                'nonsense_count': original_total_stats['nonsense_count']
            },
            'permutation': {
                'scores': permuted_total_scores,
                'scores_with_nonsense': permuted_total_scores_with_nonsense,
                'deleterious_counts': permuted_total_deleterious,
                'mean': np.mean(permuted_total_scores) if permuted_total_scores else 0,
                'mean_with_nonsense': np.mean(permuted_total_scores_with_nonsense) if permuted_total_scores_with_nonsense else 0,
                'std': np.std(permuted_total_scores) if permuted_total_scores else 0,
                'std_with_nonsense': np.std(permuted_total_scores_with_nonsense) if permuted_total_scores_with_nonsense else 0,
                'mean_deleterious': np.mean(permuted_total_deleterious) if permuted_total_deleterious else 0,
                'std_deleterious': np.std(permuted_total_deleterious) if permuted_total_deleterious else 0,
                'pvalue': p_value,
                'pvalue_with_nonsense': p_value_with_nonsense
            }
        }
        
        return gene_id, result
        
    finally:
        # Clean up
        try:
            shutil.rmtree(gene_temp_dir)
        except:
            pass

def permute_single_sample(
    nuc_mutations: Dict[str, Any],
    gene_id: str,
    filtered_positions: List[int],
    features: Dict[str, Any],
    wolgenome: SeqContext
) -> Dict[str, Any]:
    """Randomly place mutations for a single sample."""
    
    gene_length = nuc_mutations['gene_len']
    nuc_muts = list(nuc_mutations['mutations'].keys())
    
    # Get gene sequence
    gene_seq = str(wolgenome.genome[features[gene_id].start:features[gene_id].end])
    is_reverse = features[gene_id].strand == -1
    if is_reverse:
        gene_seq = str(Seq(gene_seq).reverse_complement())
    
    # Store original valid positions for each base type
    original_valid_positions = {'C': [], 'G': []}
    for pos in range(gene_length):
        if pos not in filtered_positions:
            if pos < len(gene_seq):
                base = gene_seq[pos]
                if base in original_valid_positions:
                    original_valid_positions[base].append(pos)
    
    # Reset valid positions
    valid_positions = {
        'C': original_valid_positions['C'][:],
        'G': original_valid_positions['G'][:]
    }
    
    permuted_muts = {
        'mutations': {},
        'gene_len': nuc_mutations['gene_len'],
        'avg_cov': nuc_mutations['avg_cov']
    }
    
    # Place each mutation
    for orig_mut in nuc_muts:
        ref_base = orig_mut.split('_')[1].split('>')[0]
        if valid_positions[ref_base]:
            pos_idx = random.randrange(len(valid_positions[ref_base]))
            new_pos = valid_positions[ref_base].pop(pos_idx)
            mut_type = orig_mut.split('_')[1]
            
            if new_pos < len(gene_seq) and gene_seq[new_pos] == ref_base:
                if is_reverse:
                    if mut_type == 'C>T':
                        mut_type = 'G>A'
                    elif mut_type == 'G>A':
                        mut_type = 'C>T'
                
                if is_reverse:
                    genomic_pos = len(gene_seq) - new_pos
                else:
                    genomic_pos = new_pos + 1
                
                permuted_muts['mutations'][f"{genomic_pos}_{mut_type}"] = 1
    
    return permuted_muts

def is_ems_sample(sample_name: str) -> bool:
    """Determine if a sample is a valid EMS treated sample.
    
    Excludes 7d samples and controls.
    """
    return 'EMS' in sample_name and '7d' not in sample_name

def is_control_sample(sample_name: str) -> bool:
    """Determine if a sample is a control sample."""
    return 'EMS' not in sample_name

def calculate_provean_scores_parallel(
    mutation_jsons: List[str],
    score_db: ProveanScoreDB,
    paths: Dict[str, str],
    provean_config: Dict[str, Any],
    features: Dict[str, Any],
    wolgenome: SeqContext,
    codon_table: str,
    mpileup_dir: str,
    max_workers: int = None,
    min_mutations: int = 3
) -> None:
    '''Calculate PROVEAN scores and perform permutation tests in parallel.'''
    
    # Separate EMS and control samples
    ems_jsons = []
    control_jsons = []
    
    for js in mutation_jsons:
        sample = js.split('/')[-1].split('.')[0]
        if is_ems_sample(sample):
            ems_jsons.append(js)
        elif is_control_sample(sample):
            control_jsons.append(js)
    
    logger.info(f"Found {len(ems_jsons)} EMS samples and {len(control_jsons)} control samples")
    
    # Process EMS samples
    if ems_jsons:
        logger.info("Processing EMS samples...")
        process_sample_group(
            ems_jsons,
            score_db,
            paths,
            provean_config,
            features,
            wolgenome,
            codon_table,
            mpileup_dir,
            max_workers,
            min_mutations,
            "ems_merged_results.json"
        )
    
    # Process control samples
    if control_jsons:
        logger.info("Processing control samples...")
        process_sample_group(
            control_jsons,
            score_db,
            paths,
            provean_config,
            features,
            wolgenome,
            codon_table,
            mpileup_dir,
            max_workers,
            min_mutations,
            "control_merged_results.json"
        )

def process_sample_group(
    sample_jsons: List[str],
    score_db: ProveanScoreDB,
    paths: Dict[str, str],
    provean_config: Dict[str, Any],
    features: Dict[str, Any],
    wolgenome: SeqContext,
    codon_table: str,
    mpileup_dir: str,
    max_workers: int,
    min_mutations: int,
    output_filename: str
) -> None:
    """Process a group of samples (either EMS or control) together."""
    
    # First, collect all mutations and filtered positions by gene
    gene_mutations = defaultdict(dict)  # gene -> sample -> mutations
    gene_filtered = defaultdict(dict)   # gene -> sample -> filtered positions
    
    # Process each sample
    for js in tqdm(sample_jsons, desc="Loading samples"):
        sample = js.split('/')[-1].split('.')[0]
        
        # Load filtered positions
        filtered_positions = load_filtered_positions(sample, paths, mpileup_dir)
        
        # Load mutations
        with open(js) as f:
            nuc_mutations = json.load(f)
            
        # Group by gene
        for gene, mutations in nuc_mutations.items():
            gene_mutations[gene][sample] = mutations
            gene_filtered[gene][sample] = filtered_positions.get(gene, [])
    
    logger.info(f"Processing {len(gene_mutations)} genes for {len(sample_jsons)} samples")
    
    # Process genes in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for gene in gene_mutations:
            future = executor.submit(
                process_gene_permutations,
                gene,
                gene_mutations[gene],
                gene_filtered[gene],
                score_db.db_path,
                provean_config,
                paths,
                features,
                wolgenome,
                codon_table,
                1000,
                min_mutations
            )
            futures[future] = gene
        
        # Collect results
        merged_results = {}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing genes"):
            gene = futures[future]
            try:
                gene_id, result = future.result()
                merged_results[gene_id] = result
            except Exception as e:
                logger.error(f"Error processing gene {gene}: {e}")
        
        # Save merged results
        merged_results_file = os.path.join(paths['results'], output_filename)
        with open(merged_results_file, 'w') as f:
            json.dump(merged_results, f, indent=2)

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

def create_empty_result(sample_mutations: Dict[str, Dict[str, Any]], total_mutations: int) -> Dict[str, Any]:
    first_sample = next(iter(sample_mutations.values()))
    gene_len = first_sample['gene_len']
    
    return {
        'effect': 0,
        'effect_with_nonsense': 0,
        'total_mutations': total_mutations,
        'gene_len': gene_len,
        'mutation_stats': {
            'deleterious_count': 0,
            'total_mutations': total_mutations,
            'nonsense_count': 0
        },
        'permutation': {
            'scores': [],
            'scores_with_nonsense': [],
            'deleterious_counts': [],
            'mean': 0,
            'mean_with_nonsense': 0,
            'std': 0,
            'std_with_nonsense': 0,
            'mean_deleterious': 0,
            'std_deleterious': 0,
            'pvalue': 1.0,
            'pvalue_with_nonsense': 1.0
        }
    }

def calculate_sample_statistics(
    nuc_mutations: Dict[str, Dict[str, Any]], 
    sample_name: str
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    """Calculate mutation statistics for a sample."""
    
    # Sample-level statistics
    sample_stats = {
        'sample_name': sample_name,
        'total_mutations': 0,
        'total_genes_with_mutations': 0,
        'mutation_types': {},
        'genes_processed': []
    }
    
    # Gene-level statistics
    gene_stats = {}
    
    for gene_id, gene_data in nuc_mutations.items():
        mutations = gene_data['mutations']
        gene_mutation_count = sum(mutations.values())
        
        if gene_mutation_count > 0:
            sample_stats['total_genes_with_mutations'] += 1
            sample_stats['genes_processed'].append(gene_id)
        
        sample_stats['total_mutations'] += gene_mutation_count
        
        # Count mutation types
        gene_mutation_types = {}
        for mut_key, count in mutations.items():
            mut_type = mut_key.split('_')[1]
            gene_mutation_types[mut_type] = gene_mutation_types.get(mut_type, 0) + count
            sample_stats['mutation_types'][mut_type] = sample_stats['mutation_types'].get(mut_type, 0) + count
        
        # Store gene-level stats
        gene_stats[gene_id] = {
            'gene_id': gene_id,
            'total_mutations': gene_mutation_count,
            'gene_length': gene_data['gene_len'],
            'average_coverage': gene_data['avg_cov'],
            'mutation_types': gene_mutation_types,
            'mutations_per_kb': (gene_mutation_count / gene_data['gene_len']) * 1000 if gene_data['gene_len'] > 0 else 0
        }
    
    return sample_stats, gene_stats

def normalize_context_counts(
    context_counts: Dict[str, Dict[str, int]], 
    wolgenome: SeqContext
) -> Dict[str, Dict[str, Any]]:
    """Normalize existing context counts against genome background."""
    
    # Count genome 3-mers once
    genome_kmers = {}
    for i in range(len(wolgenome.genome) - 2):  # 3-mer = k-1
        kmer = str(wolgenome.genome[i:i+3]).upper()
        genome_kmers[kmer] = genome_kmers.get(kmer, 0) + 1
    
    total_genome_kmers = sum(genome_kmers.values())
    
    normalized_contexts = {}
    
    for sample, contexts in context_counts.items():
        if sample == 'kmer_3mer_analysis':  # Skip if already processed
            continue
            
        total_mutations = sum(contexts.values())
        normalized_contexts[sample] = {}
        
        for context, mut_count in contexts.items():
            genome_count = genome_kmers.get(context, 0)
            if genome_count > 0 and total_mutations > 0:
                normalized_freq = (mut_count / total_mutations) / (genome_count / total_genome_kmers)
                normalized_contexts[sample][context] = {
                    'mutation_count': mut_count,
                    'genome_count': genome_count,
                    'normalized_frequency': normalized_freq
                }
    
    # Add genome-wide kmer counts to the output
    normalized_contexts['genome_kmer_counts'] = genome_kmers
    
    return normalized_contexts

def calculate_genome_base_counts(wolgenome: SeqContext) -> Dict[str, int]:
    """Calculate total counts of each base in the genome."""
    base_counts = {'A': 0, 'T': 0, 'G': 0, 'C': 0}
    
    for base in str(wolgenome.genome).upper():
        if base in base_counts:
            base_counts[base] += 1
    
    return base_counts

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
    score_db = ProveanScoreDB(refs['prov_score_json_db'])
    
    # Log configuration settings
    logger.info("Configuration settings:")
    logger.info(f"Reference files:")
    logger.info(f"  Genome: {refs['genomic_fna']}")
    logger.info(f"  Annotation: {refs['annotation']}")
    logger.info(f"  Codon table: {refs['codon_table']}")
    logger.info(f"  PROVEAN score database: {refs['prov_score_json_db']}")
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
    
    # Limit max_workers to avoid "too many open files" error
    # A reasonable default is 20-30 workers depending on the system
    max_workers = min(1, os.cpu_count())
    min_mutations = 3
    calculate_provean_scores_parallel(
        mutation_jsons,
        score_db,
        paths,
        provean_config,
        features,
        wolgenome,
        refs['codon_table'],
        args.mpileups,
        max_workers=max_workers,  # Use limited number of workers
        min_mutations=min_mutations  # Add minimum mutation parameter
    )

    

    logger.success("Pipeline completed successfully")

if __name__ == '__main__':
    main()