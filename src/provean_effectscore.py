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
import copy
import logging
import multiprocessing

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
            - only_parse (bool): Flag to only parse mpileup files
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
    parser.add_argument('-p', '--only_parse', action='store_true',
                        help='Only parse mpileup files')
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
            context_counts[sample],
            7  # context_size changed to 7
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
    
    # After processing all samples, collect genome-wide k-mer counts
    genome_kmer_counts = normalize_context_counts(context_counts, wolgenome, context_size=7)

    # Save genome-wide k-mer counts to file
    with open(f"{outdir}/results/genome_kmer_counts.json", 'w') as of:
        json.dump(genome_kmer_counts, of)
    
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
    aa_mutations_dict: Optional[Dict] = None,
    collect_new_scores: Optional[dict] = None
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
            new_mutations.append(hgvs_mut)
    
    # Calculate new scores if needed
    new_scores = {}
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
                
                # Do not write to DB here; collect for batch write
                # Any mutations that didn't get scores are failed
                scored_mutations = set(new_scores.keys())
                failed_mutations.extend(set(new_mutations) - scored_mutations)
                failed_scores = {mut: None for mut in failed_mutations}
                if failed_scores:
                    new_scores.update(failed_scores)
            
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
    
    # If collecting, add new_scores to the collector
    if collect_new_scores is not None and new_scores:
        collect_new_scores.update(new_scores)
    stats = {
        'deleterious_count': deleterious_count,
        'total_mutations': total_mutations,
        'failed_mutations': len(failed_mutations),
        'nonsense_count': nonsense_count
    }
    return total_score, total_score_with_nonsense, stats

def is_ems_sample(sample_name: str) -> bool:
    """Determine if a sample is a valid EMS treated sample.
    
    Excludes 7d samples and controls.
    """
    return 'EMS' in sample_name and '7d' not in sample_name

def is_control_sample(sample_name: str) -> bool:
    """Determine if a sample is a control sample."""
    return 'EMS' not in sample_name

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
        
        # Count unique mutated sites
        unique_mutated_sites = len(mutations)
        
        if unique_mutated_sites > 0:
            sample_stats['total_genes_with_mutations'] += 1
            sample_stats['genes_processed'].append(gene_id)
        
        sample_stats['total_mutations'] += unique_mutated_sites
        
        # Count mutation types
        gene_mutation_types = {}
        for mut_key, count in mutations.items():
            mut_type = mut_key.split('_')[1]
            gene_mutation_types[mut_type] = gene_mutation_types.get(mut_type, 0) + 1  # Count each site once
            sample_stats['mutation_types'][mut_type] = sample_stats['mutation_types'].get(mut_type, 0) + 1
        
        # Store gene-level stats
        gene_stats[gene_id] = {
            'gene_id': gene_id,
            'total_mutations': unique_mutated_sites,
            'gene_length': gene_data['gene_len'],
            'average_coverage': gene_data['avg_cov'],
            'mutation_types': gene_mutation_types,
            'mutations_per_kb': (unique_mutated_sites / gene_data['gene_len']) * 1000 if gene_data['gene_len'] > 0 else 0
        }
    
    return sample_stats, gene_stats

def normalize_context_counts(
    context_counts: Dict[str, Dict[str, int]], 
    wolgenome: SeqContext,
    context_size: int = 7
) -> Dict[str, int]:
    """Collect genome-wide k-mer counts for the specified context size."""
    genome_kmers = {}
    for i in range(len(wolgenome.genome) - context_size + 1):
        kmer = str(wolgenome.genome[i:i+context_size]).upper()
        genome_kmers[kmer] = genome_kmers.get(kmer, 0) + 1
    return genome_kmers

def calculate_genome_base_counts(wolgenome: SeqContext) -> Dict[str, int]:
    """Calculate total counts of each base in the genome."""
    base_counts = {'A': 0, 'T': 0, 'G': 0, 'C': 0}
    
    for base in str(wolgenome.genome).upper():
        if base in base_counts:
            base_counts[base] += 1
    
    return base_counts

def collect_observed_5mer_contexts(
    mutation_jsons: List[str],
    wolgenome: SeqContext,
    features: Dict[str, Any],
    kmer_size: int = 5
) -> Dict[str, int]:
    """Collect the 5-mer context of all observed genic mutations across all samples.
    Only considers mutations within genes and with a C or G as the center base.
    Returns a dictionary of {5mer: count}.
    """
    kmer_counts = {}
    flank = kmer_size // 2
    genome_seq = str(wolgenome.genome).upper()
    
    for js in mutation_jsons:
        with open(js) as jf:
            mut_dict = json.load(jf)
            for gene_id, gene_data in mut_dict.items():
                if gene_id not in features:
                    continue
                gene_start = features[gene_id].start
                gene_end = features[gene_id].end
                gene_strand = features[gene_id].strand
                for mut_key in gene_data['mutations']:
                    # Extract position and ref>alt
                    pos_str, mut_type = mut_key.split('_')
                    ref_base = mut_type.split('>')[0]
                    # Only C or G mutations
                    if ref_base not in ('C', 'G'):
                        continue
                    # 1-based position in gene
                    gene_pos = int(pos_str)
                    # Convert to genomic position (0-based)
                    if gene_strand == 1:
                        genome_pos = gene_start + gene_pos - 1
                    else:
                        genome_pos = gene_end - gene_pos
                    # Extract k-mer context
                    if genome_pos - flank < 0 or genome_pos + flank >= len(genome_seq):
                        continue  # skip if k-mer would go out of bounds
                    kmer = genome_seq[genome_pos - flank: genome_pos + flank + 1]
                    if len(kmer) != kmer_size:
                        continue
                    # For reverse strand, reverse complement the kmer
                    if gene_strand == -1:
                        kmer = str(Seq(kmer).reverse_complement())
                    # Only count k-mers with C or G at center
                    if kmer[flank] not in ('C', 'G'):
                        continue
                    kmer_counts[kmer] = kmer_counts.get(kmer, 0) + 1
    logger.info(f"Collected {len(kmer_counts)} unique 5-mer contexts from observed genic mutations.")
    total_mutations = sum(kmer_counts.values())
    logger.info(f"Total observed genic mutations (C/G center): {total_mutations}")
    return kmer_counts

# --- New: Collect all 5-mer sites with C/G center base within genes ---
def collect_genic_5mer_sites(
    wolgenome: SeqContext,
    features: Dict[str, Any],
    kmer_size: int = 5
) -> Dict[str, list]:
    """Collect all 5-mer sites with a C or G center base within genes.
    Returns a dictionary mapping each 5-mer to a list of (gene_id, gene_pos, genome_pos, strand).
    """
    kmer_sites = {}
    flank = kmer_size // 2
    genome_seq = str(wolgenome.genome).upper()
    for gene_id, feat in features.items():
        gene_start = feat.start
        gene_end = feat.end
        gene_strand = feat.strand
        gene_len = gene_end - gene_start
        for i in range(gene_len):
            if gene_strand == 1:
                genome_pos = gene_start + i
                gene_pos = i + 1
            else:
                genome_pos = gene_end - 1 - i
                gene_pos = gene_end - genome_pos  # 1-based, from 3' end
            # Check k-mer bounds
            if genome_pos - flank < 0 or genome_pos + flank >= len(genome_seq):
                continue
            kmer = genome_seq[genome_pos - flank: genome_pos + flank + 1]
            if len(kmer) != kmer_size:
                continue
            kmer_for_dict = kmer
            if gene_strand == -1:
                kmer_for_dict = str(Seq(kmer).reverse_complement())
            if kmer_for_dict[flank] not in ('C', 'G'):
                continue
            kmer_sites.setdefault(kmer_for_dict, []).append((gene_id, gene_pos, genome_pos, gene_strand))
    total_sites = sum(len(v) for v in kmer_sites.values())
    logger.info(f"Collected {len(kmer_sites)} unique 5-mer contexts with C/G center in genes.")
    logger.info(f"Total genic 5-mer positions (C/G center): {total_sites}")
    return kmer_sites

# --- New: Generate a random mutation profile for a single permutation ---
def generate_random_mutation_profile(
    observed_kmer_counts: Dict[str, int],
    kmer_site_pool: Dict[str, list],
    rng: random.Random = random
) -> Dict[str, dict]:
    """
    For a single permutation, randomly place observed mutations at matching 5-mer sites.
    Args:
        observed_kmer_counts: {kmer: count} for observed mutations
        kmer_site_pool: {kmer: [(gene_id, gene_pos, genome_pos, strand), ...]}
        rng: random number generator (for reproducibility)
    Returns:
        Dict[gene_id, dict]: per-gene mutation profile in the format expected by the pipeline
    """
    # Make a copy of the pool so we can exhaust it for this permutation
    pool = {k: v[:] for k, v in kmer_site_pool.items()}
    per_gene_mutations = {}
    for kmer, count in observed_kmer_counts.items():
        if kmer not in pool or len(pool[kmer]) < count:
            # Not enough sites to place all mutations of this kmer
            continue
        # Randomly select 'count' positions without replacement
        selected_sites = rng.sample(pool[kmer], count)
        # Remove selected sites from pool
        for site in selected_sites:
            pool[kmer].remove(site)
            gene_id, gene_pos, genome_pos, strand = site
            # Infer mutation type from center base of k-mer
            center_base = kmer[2]  # 5-mer center is at index 2
            if center_base == 'C':
                mut_type = 'C>T'
            elif center_base == 'G':
                mut_type = 'G>A'
            else:
                continue  # Skip if center base is not C or G
            # Build mutation key as in original code: f"{gene_pos}_{mut_type}"
            mut_key = f"{gene_pos}_{mut_type}"
            if gene_id not in per_gene_mutations:
                per_gene_mutations[gene_id] = {
                    'mutations': {},
                    'gene_len': abs(gene_pos) if 'gene_len' not in per_gene_mutations else 0,  # will be set later
                    'avg_cov': 0  # can be set to 0 or ignored for permutations
                }
            per_gene_mutations[gene_id]['mutations'][mut_key] = 1
    # Set gene_len for each gene (use the max gene_pos seen for that gene)
    for gene_id in per_gene_mutations:
        gene_lens = [int(k.split('_')[0]) for k in per_gene_mutations[gene_id]['mutations']]
        per_gene_mutations[gene_id]['gene_len'] = max(gene_lens) if gene_lens else 0
    return per_gene_mutations

# Move _single_perm outside of generate_permutation_profiles to avoid the AttributeError when pickling the local function
def _single_perm(seed, observed_kmer_counts, kmer_site_pool):
    rng = random.Random(seed)
    return generate_random_mutation_profile(observed_kmer_counts, kmer_site_pool, rng=rng)

def generate_permutation_profiles(
    observed_kmer_counts: Dict[str, int],
    kmer_site_pool: Dict[str, list],
    n_permutations: int = 20,
    random_seed: int = None,
    max_workers: int = None
) -> list:
    """
    Generate N random mutation profiles in parallel.
    Args:
        observed_kmer_counts: {kmer: count} for observed mutations
        kmer_site_pool: {kmer: [(gene_id, gene_pos, genome_pos, strand), ...]}
        n_permutations: number of permutations to generate
        random_seed: seed for reproducibility
        max_workers: number of parallel workers
    Returns:
        List[Dict[gene_id, dict]]: one per permutation
    """
    logger.info(f"Generating {n_permutations} random mutation profiles for permutation test.")
    # Use different seeds for each permutation for reproducibility
    seeds = [(random_seed + i) if random_seed is not None else None for i in range(n_permutations)]
    profiles = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(_single_perm, seeds, [observed_kmer_counts] * n_permutations, [kmer_site_pool] * n_permutations)
        for idx, profile in enumerate(results, 1):
            profiles.append(profile)
            if idx % 50 == 0 or idx == n_permutations:
                logger.info(f"Random mutation profile generation progress: {idx}/{n_permutations} permutations generated.")
    logger.info("Random mutation profile generation complete.")
    return profiles

def _process_gene(gene_id, shared_profiles, score_db_path, provean_config, paths, features, genomic_fna, annotation, codon_table, db_lock):
    from modules import parse
    from modules.provean_db import ProveanScoreDB
    score_db = ProveanScoreDB(score_db_path)
    wolgenome = parse.SeqContext(genomic_fna, annotation)
    gene_mut_profiles = [profile.get(gene_id, None) for profile in shared_profiles]
    gene_mut_profiles = [m for m in gene_mut_profiles if m is not None]
    perm_scores = []
    all_new_scores = {}
    for mut_profile in gene_mut_profiles:
        score, score_with_nonsense, stats = nuc_to_provean_score(
            mut_profile,
            gene_id,
            score_db,
            provean_config,
            paths,
            features,
            wolgenome,
            codon_table,
            collect_new_scores=all_new_scores
        )
        perm_scores.append({
            'effect': score,
            'effect_with_nonsense': score_with_nonsense,
            'mutation_stats': stats
        })
    # After all permutations, write all new scores for this gene in one batch
    if all_new_scores:
        with db_lock:
            score_db.add_scores_batch(gene_id, all_new_scores)
    effects = [s['effect'] for s in perm_scores]
    effects_with_nonsense = [s['effect_with_nonsense'] for s in perm_scores]
    result = {
        'permutation_effects': effects,
        'permutation_effects_with_nonsense': effects_with_nonsense,
        'mean': np.mean(effects) if effects else 0,
        'std': np.std(effects) if effects else 0,
        'mean_with_nonsense': np.mean(effects_with_nonsense) if effects_with_nonsense else 0,
        'std_with_nonsense': np.std(effects_with_nonsense) if effects_with_nonsense else 0,
    }
    return gene_id, result

def process_permuted_provean_scores(
    permutation_profiles: list,
    score_db: ProveanScoreDB,
    provean_config: dict,
    paths: dict,
    features: dict,
    genomic_fna: str,
    annotation: str,
    codon_table: str,
    max_workers: int = None
) -> dict:
    gene_ids = set()
    for profile in permutation_profiles:
        gene_ids.update(profile.keys())
    gene_ids = list(gene_ids)
    logger.info(f"Starting permutation test scoring for {len(gene_ids)} genes.")
    results = {}
    processed = 0
    with multiprocessing.Manager() as manager:
        shared_profiles = manager.list(permutation_profiles)
        db_lock = manager.Lock()
        with multiprocessing.Pool(processes=max_workers) as pool:
            args = [
                (gene_id, shared_profiles, score_db.db_path, provean_config, paths, features, genomic_fna, annotation, codon_table, db_lock)
                for gene_id in gene_ids
            ]
            for gene_id, result in pool.starmap(_process_gene, args):
                results[gene_id] = result
                processed += 1
                if processed % 10 == 0 or processed == len(gene_ids):
                    logger.info(f"Permutation test progress: {processed}/{len(gene_ids)} genes processed.")
    logger.info("Permutation test scoring complete.")
    return results

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
    logger.info(f"  Only parse: {args.only_parse}")
    
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
    if args.only_parse:
        logger.info("Only parsing mpileup files")
        return
    
    # Process mutations
    #nuc_mut_files = glob(paths['mutpath'] + '/*.json')
    #process_mutations(nuc_mut_files, wolgenome, refs['codon_table'], features, paths)
    
    # Calculate PROVEAN scores with permutation testing
    mutation_jsons = glob(paths['mutpath'] + '/*.json')
    
    # Set max_workers for parallelization
    max_workers = min(20, os.cpu_count())
    
    # Collect 5-mer context of all observed genic mutations
    collect_observed_5mer_contexts(
        mutation_jsons,
        wolgenome,
        features,
        kmer_size=5
    )

    # Collect all 5-mer sites with C/G center base within genes
    collect_genic_5mer_sites(
        wolgenome,
        features,
        kmer_size=5
    )

    # Calculate permuted PROVEAN scores
    permutation_profiles = generate_permutation_profiles(
        collect_observed_5mer_contexts(
            mutation_jsons,
            wolgenome,
            features,
            kmer_size=5
        ),
        collect_genic_5mer_sites(
            wolgenome,
            features,
            kmer_size=5
        ),
        n_permutations=1000,
        random_seed=None,
        max_workers=max_workers
    )
    permuted_results = process_permuted_provean_scores(
        permutation_profiles,
        score_db,
        provean_config,
        paths,
        features,
        refs['genomic_fna'],
        refs['annotation'],
        refs['codon_table'],
        max_workers=max_workers
    )

    logger.success("Pipeline completed successfully")

if __name__ == '__main__':
    main()