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
from Bio.Data import CodonTable
import pandas as pd

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
    
    # Add console handler with INFO level
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
        sample (str): Sample name (usually stem of nuc_muts JSON)
        paths (Dict[str, str]): Dictionary containing output paths
        mpileup_dir (str): Directory containing mpileup files and filtered positions
        
    Returns:
        Dict[str, List[int]]: Dictionary mapping gene IDs to 0-based relative filtered positions
    """
    # Support both naming schemes to avoid double "_filtered" in filenames
    candidates = []
    # Common expected names
    candidates.append(os.path.join(mpileup_dir, f"{sample}_filtered_positions.json"))
    candidates.append(os.path.join(mpileup_dir, f"{sample}_positions.json"))
    # If sample already ends with _filtered, prefer *_positions.json and also try base without trailing _filtered
    if sample.endswith('_filtered'):
        base = sample[:-9]
        candidates.insert(0, os.path.join(mpileup_dir, f"{sample}_positions.json"))
        candidates.append(os.path.join(mpileup_dir, f"{base}_filtered_positions.json"))
    for filtered_file in candidates:
        if os.path.exists(filtered_file):
            with open(filtered_file) as f:
                raw_positions = json.load(f)
            # Convert from 1-based relative to 0-based relative positions
            corrected_positions = {}
            for gene_id, positions in raw_positions.items():
                corrected_positions[gene_id] = [pos - 1 for pos in positions]
            return corrected_positions
    return {}

# Helper: build union of callable positions per gene for a set of samples

def build_union_callable_positions(sample_jsons: List[str], paths: Dict[str, str], mpileup_dir: str) -> Dict[str, set]:
    """Given sample mutation JSON paths, load each sample's filtered positions and
    return a dict mapping gene_id -> set of 0-based relative positions that are callable
    in any of the samples (union)."""
    union_map: Dict[str, set] = {}
    for js in sample_jsons:
        sample_name = os.path.basename(js).replace('.json', '')
        filtered = load_filtered_positions(sample_name, paths, mpileup_dir)
        for gene_id, pos_list in filtered.items():
            if gene_id not in union_map:
                union_map[gene_id] = set()
            union_map[gene_id].update(pos_list)
    return union_map

# New: build intersection of callable positions across samples

def build_intersection_callable_positions(sample_jsons: List[str], paths: Dict[str, str], mpileup_dir: str) -> Dict[str, set]:
    """Return gene_id -> set of 0-based positions callable in all provided samples (intersection).
    If a gene is missing in any sample's filtered map, its intersection becomes empty and is dropped."""
    inter_map: Dict[str, set] = {}
    first = True
    for js in sample_jsons:
        sample_name = os.path.basename(js).replace('.json', '')
        filtered = load_filtered_positions(sample_name, paths, mpileup_dir)
        if first:
            # seed with first sample's callable positions
            inter_map = {g: set(pos_list) for g, pos_list in filtered.items()}
            first = False
        else:
            # intersect existing genes; genes absent in this sample become empty
            current_genes = set(inter_map.keys())
            for gene_id in list(current_genes):
                if gene_id in filtered:
                    inter_map[gene_id] &= set(filtered[gene_id])
                else:
                    inter_map[gene_id] = set()
        # Early stop if everything empty
        if not any(len(s) for s in inter_map.values()):
            break
    # Drop empty entries
    inter_map = {g: s for g, s in inter_map.items() if len(s) > 0}
    return inter_map

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
    aa_muts, genome_stats, mismatch_info = translate.convert_mutations(
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
            score = score_db.get_score(gene_id, hgvs_mut)  # This is in-memory, but for SQLite, the key is 'gene'
            
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
        'nonsense_count': nonsense_count,
        'non_syn_muts': genome_stats.get('non_syn_muts', 0),
        'syn_muts': genome_stats.get('syn_muts', 0)
    }
    return total_score, total_score_with_nonsense, stats, mismatch_info

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
    Deduplicates strictly by site (gene_id, 0-based gene-relative position).
    Returns a dictionary of {5mer: count}.
    """
    kmer_counts = {}
    flank = kmer_size // 2
    genome_seq = str(wolgenome.genome).upper()
    # Deduplicate by site only (gene_id, rel0)
    seen_sites_global: set = set()
    
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
                    # 0-based position in gene
                    rel0 = int(pos_str)
                    # De-duplicate by site only
                    site_id = (gene_id, rel0)
                    if site_id in seen_sites_global:
                        continue
                    seen_sites_global.add(site_id)
                    # Convert to genomic position (0-based)
                    if gene_strand == 1:
                        genome_pos = gene_start + rel0
                    else:
                        genome_pos = gene_end - 1 - rel0
                    # Extract k-mer context
                    if genome_pos - flank < 0 or genome_pos + flank >= len(genome_seq):
                        continue  # skip if k-mer would go out of bounds
                    kmer = genome_seq[genome_pos - flank: genome_pos + flank + 1]
                    if len(kmer) != kmer_size:
                        continue
                    # For reverse strand, reverse complement the kmer to coding orientation
                    if gene_strand == -1:
                        kmer = str(Seq(kmer).reverse_complement())
                    # Only count k-mers with C or G at center
                    if kmer[flank] not in ('C', 'G'):
                        continue
                    kmer_counts[kmer] = kmer_counts.get(kmer, 0) + 1
    logger.info(f"Collected {len(kmer_counts)} unique 5-mer contexts from observed genic mutations (site-deduped).")
    total_mutations = sum(kmer_counts.values())
    logger.info(f"Total observed genic unique mutated sites (C/G center): {total_mutations}")
    return kmer_counts

# --- New: Collect observed 5-mer + codon triplet strata counts ---

def collect_observed_5mer_codon_contexts(
    mutation_jsons: List[str],
    wolgenome: SeqContext,
    features: Dict[str, Any],
    kmer_size: int = 5
) -> Dict[tuple, int]:
    """Collect observed counts per (5-mer, codon_triplet) for C/G-centered genic mutations.
    Deduplicates strictly by site (gene_id, 0-based gene-relative position).
    Returns {(kmer, codon_triplet): count} in coding orientation.
    """
    kmer_codon_counts: Dict[tuple, int] = {}
    flank = kmer_size // 2
    genome_seq = str(wolgenome.genome).upper()
    # Deduplicate by site only
    seen_sites_global: set = set()
    for js in mutation_jsons:
        with open(js) as jf:
            mut_dict = json.load(jf)
        for gene_id, gene_data in mut_dict.items():
            if gene_id not in features:
                continue
            feat = features[gene_id]
            gene_start = feat.start
            gene_end = feat.end
            strand = feat.strand
            for mut_key in gene_data['mutations']:
                pos_str, mut_type = mut_key.split('_')
                ref_base = mut_type.split('>')[0]
                if ref_base not in ('C', 'G'):
                    continue
                # 0-based gene-relative position
                rel0 = int(pos_str)
                # Deduplicate by site only
                site_id = (gene_id, rel0)
                if site_id in seen_sites_global:
                    continue
                seen_sites_global.add(site_id)
                # Genomic coordinate (0-based)
                if strand == 1:
                    genome_pos = gene_start + rel0
                else:
                    genome_pos = gene_end - 1 - rel0
                if genome_pos - flank < 0 or genome_pos + flank >= len(genome_seq):
                    continue
                kmer = genome_seq[genome_pos - flank: genome_pos + flank + 1]
                # Determine coding-orientation 5-mer and codon
                frame = rel0 % 3  # (gene_pos_1b - 1) % 3
                if strand == -1:
                    kmer = str(Seq(kmer).reverse_complement())
                    codon_start = genome_pos - (2 - frame)
                    if codon_start < 0 or codon_start + 3 > len(genome_seq):
                        continue
                    codon_genomic = genome_seq[codon_start: codon_start + 3]
                    codon_coding = str(Seq(codon_genomic).reverse_complement())
                else:
                    # Forward strand: keep kmer as is (coding orientation)
                    codon_start = genome_pos - frame
                    if codon_start < 0 or codon_start + 3 > len(genome_seq):
                        continue
                    codon_coding = genome_seq[codon_start: codon_start + 3]
                if kmer[flank] not in ('C', 'G'):
                    continue
                key = (kmer, codon_coding)
                kmer_codon_counts[key] = kmer_codon_counts.get(key, 0) + 1
    logger.info(f"Collected {len(kmer_codon_counts)} unique (5-mer, codon) strata from observed mutations (site-deduped).")
    return kmer_codon_counts

# --- New: Collect all 5-mer sites with C/G center base within genes, stratified by codon triplet and optional callable restriction ---

def collect_genic_5mer_sites(
    wolgenome: SeqContext,
    features: Dict[str, Any],
    kmer_size: int = 5,
    callable_positions: Dict[str, set] = None
) -> Dict[tuple, list]:
    """Collect all 5-mer sites with a C or G center base within genes.
    Returns a dictionary mapping (5-mer, codon_triplet) -> list of (gene_id, gene_pos, genome_pos, strand).
    If callable_positions is provided, restricts to positions present in callable_positions[gene_id] (0-based relative to gene).
    """
    kmer_sites: Dict[tuple, list] = {}
    flank = kmer_size // 2
    genome_seq = str(wolgenome.genome).upper()
    for gene_id, feat in features.items():
        gene_start = feat.start
        gene_end = feat.end
        gene_strand = feat.strand
        gene_len = gene_end - gene_start
        allowed = None
        if callable_positions and gene_id in callable_positions:
            allowed = callable_positions[gene_id]
        for i in range(gene_len):
            if gene_strand == 1:
                genome_pos = gene_start + i
                gene_pos = i + 1  # 1-based
            else:
                genome_pos = gene_end - 1 - i
                gene_pos = gene_end - genome_pos  # 1-based from 3' end but consistent with frame calc
            if allowed is not None and (gene_pos - 1) not in allowed:
                continue
            if genome_pos - flank < 0 or genome_pos + flank >= len(genome_seq):
                continue
            kmer = genome_seq[genome_pos - flank: genome_pos + flank + 1]
            if len(kmer) != kmer_size:
                continue
            frame = (gene_pos - 1) % 3
            if gene_strand == -1:
                kmer_key = str(Seq(kmer).reverse_complement())
                codon_start = genome_pos - (2 - frame)
                if codon_start < 0 or codon_start + 3 > len(genome_seq):
                    continue
                codon_genomic = genome_seq[codon_start: codon_start + 3]
                codon_coding = str(Seq(codon_genomic).reverse_complement())
            else:
                kmer_key = kmer
                codon_start = genome_pos - frame
                if codon_start < 0 or codon_start + 3 > len(genome_seq):
                    continue
                codon_coding = genome_seq[codon_start: codon_start + 3]
            if kmer_key[flank] not in ('C', 'G'):
                continue
            kmer_sites.setdefault((kmer_key, codon_coding), []).append((gene_id, gene_pos, genome_pos, gene_strand))
    total_sites = sum(len(v) for v in kmer_sites.values())
    logger.info(f"Collected {len(kmer_sites)} unique (5-mer, codon) strata with C/G center in genes.")
    logger.info(f"Total genic 5-mer positions (C/G center): {total_sites}")
    return kmer_sites

# --- New: Summarize observed vs capacity per (5-mer, codon) stratum ---

def summarize_strata_capacity(
    observed_kmer_codon_counts: Dict[tuple, int],
    kmer_site_pool: Dict[tuple, list],
    group_label: str,
    results_dir: str
) -> None:
    """Write a CSV summarizing, per (5-mer, codon) stratum, the observed unique sites used
    to seed permutations vs the available site capacity in the pool.

    Columns: stratum, observed, capacity, deficit (observed - capacity if > 0)
    Safely no-ops if inputs are empty.
    """
    try:
        if not observed_kmer_codon_counts or not kmer_site_pool:
            logger.warning(f"summarize_strata_capacity: empty inputs for {group_label}; skipping summary")
            return
        rows = []
        total_obs = 0
        total_cap = 0
        for key, obs in observed_kmer_codon_counts.items():
            cap = len(kmer_site_pool.get(key, []))
            total_obs += int(obs)
            total_cap += int(cap)
            rows.append({
                'stratum': f"{key[0]}|{key[1]}",
                'observed': int(obs),
                'capacity': int(cap),
                'deficit': max(0, int(obs) - int(cap))
            })
        outp = os.path.join(results_dir, f'strata_capacity_summary_{group_label}.csv')
        pd.DataFrame(rows, columns=['stratum','observed','capacity','deficit']).to_csv(outp, index=False)
        logger.info(f"Wrote strata capacity summary ({group_label}): {outp} (obs={total_obs} cap={total_cap})")
    except Exception as e:
        logger.warning(f"summarize_strata_capacity failed for {group_label}: {e}")

# --- New: Generate a random mutation profile for a single permutation (with replacement, stratified) ---

def generate_random_mutation_profile(
    observed_kmer_codon_counts: Dict[tuple, int],
    kmer_site_pool: Dict[tuple, list],
    rng: random.Random = random
) -> Dict[str, dict]:
    """
    For a single permutation, randomly place observed mutations at matching (5-mer, codon) sites.
    Sampling is without replacement per site to avoid multiple placements on the same site.
    Returns per-gene mutation profile.
    """
    per_gene_mutations: Dict[str, dict] = {}
    for key, count in observed_kmer_codon_counts.items():
        if key not in kmer_site_pool or len(kmer_site_pool[key]) == 0 or count <= 0:
            continue
        sites = kmer_site_pool[key]
        # Sample without replacement per site; cap by available unique sites
        if count >= len(sites):
            selected_sites = list(sites)
        else:
            selected_sites = rng.sample(sites, k=count)
        for site in selected_sites:
            gene_id, gene_pos, genome_pos, strand = site
            center_base = key[0][2]  # kmer center in coding orientation
            if center_base == 'C':
                mut_type = 'C>T'
            elif center_base == 'G':
                mut_type = 'G>A'
            else:
                continue
            mut_key = f"{gene_pos - 1}_{mut_type}"
            if gene_id not in per_gene_mutations:
                per_gene_mutations[gene_id] = {
                    'mutations': {},
                    'gene_len': 0,
                    'avg_cov': 0
                }
            # Place at most one hit per site (no duplicate placements)
            if mut_key not in per_gene_mutations[gene_id]['mutations']:
                per_gene_mutations[gene_id]['mutations'][mut_key] = 1
    # Set gene_len for each gene (use the max gene_pos seen for that gene)
    for gene_id in per_gene_mutations:
        gene_lens = [int(k.split('_')[0]) for k in per_gene_mutations[gene_id]['mutations']]
        per_gene_mutations[gene_id]['gene_len'] = max(gene_lens) if gene_lens else 0
    return per_gene_mutations

# Update helper for ProcessPool

def _single_perm(seed, observed_kmer_codon_counts, kmer_site_pool):
    rng = random.Random(seed)
    return generate_random_mutation_profile(observed_kmer_codon_counts, kmer_site_pool, rng=rng)

def generate_permutation_profiles(
    observed_kmer_codon_counts: Dict[tuple, int],
    kmer_site_pool: Dict[tuple, list],
    n_permutations: int = 20,
    random_seed: int = None,
    max_workers: int = None
) -> list:
    """
    Generate N random mutation profiles in parallel using with-replacement sampling
    stratified by (5-mer, codon) to better match observed deleteriousness potential.
    """
    logger.info(f"Generating {n_permutations} random mutation profiles for permutation test (with replacement, stratified).")
    seeds = [(random_seed + i) if random_seed is not None else None for i in range(n_permutations)]
    profiles = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(_single_perm, seeds, [observed_kmer_codon_counts] * n_permutations, [kmer_site_pool] * n_permutations)
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
    # Ensure we preserve the global permutation index for this gene by filling
    # permutations with no placements with an empty profile (0 mutations)
    num_perms = len(shared_profiles)
    gene_len = (features[gene_id].end - features[gene_id].start) if gene_id in features else 0
    gene_mut_profiles = []
    for idx in range(num_perms):
        prof = shared_profiles[idx].get(gene_id)
        if prof is None:
            prof = {
                'mutations': {},
                'gene_len': gene_len,
                'avg_cov': 0
            }
        gene_mut_profiles.append(prof)
    perm_scores = []
    all_new_scores = {}
    placed_counts = []
    counted_counts = []
    mismatch_counts = []
    for idx, mut_profile in enumerate(gene_mut_profiles):
        score, score_with_nonsense, stats, mismatch_info = nuc_to_provean_score(
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
        # Diagnostics: placed vs counted and mismatch tally
        placed_counts.append(len(mut_profile.get('mutations', {})))
        counted_counts.append((stats.get('non_syn_muts', 0) or 0) + (stats.get('syn_muts', 0) or 0))
        mismatch_counts.append(mismatch_info['summary']['total_mismatches'] if mismatch_info and 'summary' in mismatch_info else 0)
    # After all permutations, write all new scores for this gene in one batch
    if all_new_scores:
        with db_lock:
            score_db.add_scores_batch(gene_id, all_new_scores)
    effects = [s['effect'] for s in perm_scores]
    effects_with_nonsense = [s['effect_with_nonsense'] for s in perm_scores]
    deleterious_counts = [s['mutation_stats']['deleterious_count'] for s in perm_scores]
    total_mutations = [s['mutation_stats']['total_mutations'] for s in perm_scores]
    non_syn_counts = [s['mutation_stats'].get('non_syn_muts', 0) for s in perm_scores]
    syn_counts = [s['mutation_stats'].get('syn_muts', 0) for s in perm_scores]
    result = {
        'permutation_effects': effects,
        'permutation_effects_with_nonsense': effects_with_nonsense,
        'deleterious_counts': deleterious_counts,
        'total_mutations': total_mutations,
        'non_syn_counts': non_syn_counts,
        'syn_counts': syn_counts,
        'placed_counts': placed_counts,
        'counted_counts': counted_counts,
        'mismatch_counts': mismatch_counts,
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

def compute_potential_synonymous_sites(
    features: Dict[str, Any],
    wolgenome: SeqContext,
    callable_positions: Dict[str, set]
) -> Dict[str, Dict[str, int]]:
    """For each gene, count potential synonymous C/G->T/A sites among callable positions.
    Returns {gene_id: {'potential_syn_sites': int}}.
    """
    result: Dict[str, Dict[str, int]] = {}
    genome_seq = str(wolgenome.genome).upper()
    for gene_id, feat in features.items():
        allowed = callable_positions.get(gene_id, None)
        if allowed is None or len(allowed) == 0:
            continue
        gene_start = feat.start
        gene_end = feat.end
        strand = feat.strand
        potential = 0
        for rel0 in allowed:  # 0-based relative within gene
            # Map to genomic position
            if strand == 1:
                genome_pos = gene_start + rel0
                gene_pos_1b = rel0 + 1
            else:
                genome_pos = gene_end - 1 - rel0
                gene_pos_1b = gene_end - genome_pos
            # Determine coding base
            base = genome_seq[genome_pos]
            if base not in ('C', 'G'):
                continue
            # Build codon
            frame = (gene_pos_1b - 1) % 3
            if strand == 1:
                codon_start = genome_pos - frame
                if codon_start < 0 or codon_start + 3 > len(genome_seq):
                    continue
                codon = genome_seq[codon_start: codon_start + 3]
            else:
                codon_start = genome_pos - (2 - frame)
                if codon_start < 0 or codon_start + 3 > len(genome_seq):
                    continue
                codon_genomic = genome_seq[codon_start: codon_start + 3]
                codon = str(Seq(codon_genomic).reverse_complement())
            # Simulate C/G->T/A at the codon position
            pos_in_codon = (gene_pos_1b - 1) % 3
            codon_list = list(codon)
            if base == 'C':
                mutated = 'T'
            elif base == 'G':
                mutated = 'A'
            else:
                continue
            # Only proceed if codon center base in coding orientation matches
            if codon_list[pos_in_codon] != base:
                continue
            codon_list[pos_in_codon] = mutated
            mutated_codon = ''.join(codon_list)
            try:
                std_table = CodonTable.unambiguous_dna_by_id[11]  # bacterial/archaea standard
            except KeyError:
                std_table = CodonTable.unambiguous_dna_by_name['Bacterial, Archaeal and Plant Plastid']
            # Translate codons (simple map via table)
            def aa_of(c):
                return '*' if c in std_table.stop_codons else std_table.forward_table.get(c, 'X')
            if aa_of(codon) == aa_of(mutated_codon):
                potential += 1
        result[gene_id] = {'potential_syn_sites': potential}
    return result

# Helper: fallback to all genic positions per gene when callable sets are unavailable

def build_all_genic_positions(features: Dict[str, Any]) -> Dict[str, set]:
    """Return all 0-based relative positions per gene as callable."""
    all_pos: Dict[str, set] = {}
    for gene_id, feat in features.items():
        gene_len = feat.end - feat.start
        if gene_len > 0:
            all_pos[gene_id] = set(range(gene_len))
    return all_pos

# Update: count truly synonymous EMS-type observed hits (unique sites)

def compute_observed_synonymous_hits(
    mutation_jsons: List[str],
    features: Dict[str, Any] = None,
    wolgenome: SeqContext = None,
    callable_positions: Dict[str, set] = None
) -> Dict[str, Dict[str, int]]:
    """For each gene, count unique callable positions that had a synonymous EMS-type (C>T/G>A) hit.
    Returns {gene_id: {'observed_syn_sites': int}}. If features/wolgenome provided, verifies synonymous by translation.
    """
    if features is None or wolgenome is None:
        # Backward-compat: count EMS-type sites without verifying synonymous (not preferred)
        observed_syn_sites: Dict[str, set] = defaultdict(set)
        for js in mutation_jsons:
            with open(js) as jf:
                mut_dict = json.load(jf)
            for gene_id, gene_data in mut_dict.items():
                for mut_key in gene_data.get('mutations', {}).keys():
                    pos_str, mut_type = mut_key.split('_')
                    if mut_type not in ('C>T', 'G>A'):
                        continue
                    rel0 = int(pos_str)
                    if callable_positions and rel0 not in callable_positions.get(gene_id, set()):
                        continue
                    observed_syn_sites[gene_id].add(rel0)
        return {g: {'observed_syn_sites': len(pos_set)} for g, pos_set in observed_syn_sites.items()}

    genome_seq = str(wolgenome.genome).upper()
    try:
        std_table = CodonTable.unambiguous_dna_by_id[11]
    except KeyError:
        std_table = CodonTable.unambiguous_dna_by_name['Bacterial, Archaeal and Plant Plastid']
    def aa_of(c: str) -> str:
        return '*' if c in std_table.stop_codons else std_table.forward_table.get(c, 'X')

    observed_syn_sites: Dict[str, set] = defaultdict(set)
    for js in mutation_jsons:
        with open(js) as jf:
            mut_dict = json.load(jf)
        for gene_id, gene_data in mut_dict.items():
            feat = features.get(gene_id)
            if feat is None:
                continue
            allowed = callable_positions.get(gene_id, None) if callable_positions else None
            gene_start = feat.start
            gene_end = feat.end
            strand = feat.strand
            for mut_key in gene_data.get('mutations', {}).keys():
                pos_str, mut_type = mut_key.split('_')
                if mut_type not in ('C>T', 'G>A'):
                    continue
                rel0 = int(pos_str)
                if allowed is not None and rel0 not in allowed:
                    continue
                if strand == 1:
                    genome_pos = gene_start + rel0
                    gene_pos_1b = rel0 + 1
                else:
                    genome_pos = gene_end - 1 - rel0
                    gene_pos_1b = gene_end - genome_pos
                base = genome_seq[genome_pos]
                if base not in ('C', 'G'):
                    continue
                frame = (gene_pos_1b - 1) % 3
                if strand == 1:
                    codon_start = genome_pos - frame
                    if codon_start < 0 or codon_start + 3 > len(genome_seq):
                        continue
                    codon = genome_seq[codon_start: codon_start + 3]
                else:
                    codon_start = genome_pos - (2 - frame)
                    if codon_start < 0 or codon_start + 3 > len(genome_seq):
                        continue
                    codon_genomic = genome_seq[codon_start: codon_start + 3]
                    codon = str(Seq(codon_genomic).reverse_complement())
                pos_in_codon = (gene_pos_1b - 1) % 3
                codon_list = list(codon)
                if codon_list[pos_in_codon] != base:
                    continue
                mutated = 'T' if base == 'C' else 'A'
                codon_list[pos_in_codon] = mutated
                mutated_codon = ''.join(codon_list)
                if aa_of(codon) == aa_of(mutated_codon):
                    observed_syn_sites[gene_id].add(rel0)
    return {g: {'observed_syn_sites': len(pos_set)} for g, pos_set in observed_syn_sites.items()}

def write_syn_site_fraction_csv(
    label: str,
    output_results_dir: str,
    potential_syn: Dict[str, Dict[str, int]],
    observed_syn: Dict[str, Dict[str, int]]
) -> None:
    rows = []
    for gene_id, pot in potential_syn.items():
        pot_n = pot.get('potential_syn_sites', 0)
        obs_n = observed_syn.get(gene_id, {}).get('observed_syn_sites', 0)
        frac = (obs_n / pot_n) if pot_n > 0 else np.nan
        rows.append({
            'gene_id': gene_id,
            'potential_syn_sites': pot_n,
            'observed_syn_sites': obs_n,
            'observed_fraction_of_potential_syn': frac
        })
    columns = ['gene_id', 'potential_syn_sites', 'observed_syn_sites', 'observed_fraction_of_potential_syn']
    df = pd.DataFrame(rows, columns=columns)
    out_path = os.path.join(output_results_dir, f'{label}_observed_vs_potential_syn_sites.csv')
    df.to_csv(out_path, index=False)
    logger.info(f'Wrote per-gene observed vs potential synonymous site fractions: {out_path} ({len(df)} rows)')

def compute_expected_nonsyn_fraction(
    features: Dict[str, Any],
    wolgenome: SeqContext,
    callable_positions: Dict[str, set]
) -> Dict[str, Dict[str, float]]:
    """For each gene, compute counts of callable C/G sites that would be non-syn vs syn under C->T/G->A,
    and the expected non-syn fraction = non_syn_sites / (syn + non_syn).
    Returns {gene_id: {'expected_nonsyn_sites': int, 'expected_syn_sites': int, 'expected_nonsyn_fraction': float}}.
    """
    result: Dict[str, Dict[str, float]] = {}
    genome_seq = str(wolgenome.genome).upper()
    try:
        std_table = CodonTable.unambiguous_dna_by_id[11]
    except KeyError:
        std_table = CodonTable.unambiguous_dna_by_name['Bacterial, Archaeal and Plant Plastid']
    def aa_of(c: str) -> str:
        return '*' if c in std_table.stop_codons else std_table.forward_table.get(c, 'X')
    for gene_id, feat in features.items():
        allowed = callable_positions.get(gene_id, None)
        if not allowed:
            continue
        gene_start = feat.start
        gene_end = feat.end
        strand = feat.strand
        non_syn = 0
        syn = 0
        for rel0 in allowed:
            if strand == 1:
                genome_pos = gene_start + rel0
                gene_pos_1b = rel0 + 1
            else:
                genome_pos = gene_end - 1 - rel0
                gene_pos_1b = gene_end - genome_pos
            base = genome_seq[genome_pos]
            if base not in ('C', 'G'):
                continue
            frame = (gene_pos_1b - 1) % 3
            if strand == 1:
                codon_start = genome_pos - frame
                if codon_start < 0 or codon_start + 3 > len(genome_seq):
                    continue
                codon = genome_seq[codon_start: codon_start + 3]
            else:
                codon_start = genome_pos - (2 - frame)
                if codon_start < 0 or codon_start + 3 > len(genome_seq):
                    continue
                codon_genomic = genome_seq[codon_start: codon_start + 3]
                codon = str(Seq(codon_genomic).reverse_complement())
            pos_in_codon = (gene_pos_1b - 1) % 3
            codon_list = list(codon)
            if codon_list[pos_in_codon] != base:
                continue
            mutated = 'T' if base == 'C' else 'A'
            codon_list[pos_in_codon] = mutated
            mutated_codon = ''.join(codon_list)
            if aa_of(codon) == aa_of(mutated_codon):
                syn += 1
            else:
                non_syn += 1
        total = syn + non_syn
        frac = (non_syn / total) if total > 0 else float('nan')
        result[gene_id] = {
            'expected_nonsyn_sites': non_syn,
            'expected_syn_sites': syn,
            'expected_nonsyn_fraction': frac
        }
    return result


def compute_observed_nonsyn_fraction(
    mutation_jsons: List[str],
    features: Dict[str, Any],
    wolgenome: SeqContext,
    callable_positions: Dict[str, set]
) -> Dict[str, Dict[str, float]]:
    """For each gene, compute observed counts of mutated callable C/G sites that would be non-syn vs syn under C->T/G->A.
    Restrict to EMS-type (C->T/G->A) and deduplicate sites across samples. Returns
    {gene_id: {'observed_nonsyn_sites': int, 'observed_syn_sites': int, 'observed_nonsyn_fraction': float, 'observed_total_cg_sites': int}}.
    """
    genome_seq = str(wolgenome.genome).upper()
    try:
        std_table = CodonTable.unambiguous_dna_by_id[11]
    except KeyError:
        std_table = CodonTable.unambiguous_dna_by_name['Bacterial, Archaeal and Plant Plastid']
    def aa_of(c: str) -> str:
        return '*' if c in std_table.stop_codons else std_table.forward_table.get(c, 'X')
    # Collect unique mutated positions per gene, restricted to callable and EMS-types
    mutated_positions: Dict[str, set] = defaultdict(set)
    for js in mutation_jsons:
        with open(js) as jf:
            mut_dict = json.load(jf)
        for gene_id, gene_data in mut_dict.items():
            allowed = callable_positions.get(gene_id, None)
            if allowed is None or len(allowed) == 0:
                continue
            for mut_key in gene_data.get('mutations', {}).keys():
                pos_str, mut_type = mut_key.split('_')
                if mut_type not in ('C>T', 'G>A'):
                    continue
                rel0 = int(pos_str)
                if rel0 in allowed:
                    mutated_positions[gene_id].add(rel0)
    # Classify mutated sites as syn / non-syn
    result: Dict[str, Dict[str, float]] = {}
    for gene_id, pos_set in mutated_positions.items():
        feat = features.get(gene_id)
        if feat is None:
            continue
        gene_start = feat.start
        gene_end = feat.end
        strand = feat.strand
        non_syn = 0
        syn = 0
        for rel0 in pos_set:
            if strand == 1:
                genome_pos = gene_start + rel0
                gene_pos_1b = rel0 + 1
            else:
                genome_pos = gene_end - 1 - rel0
                gene_pos_1b = gene_end - genome_pos
            base = genome_seq[genome_pos]
            if base not in ('C', 'G'):
                continue
            frame = (gene_pos_1b - 1) % 3
            if strand == 1:
                codon_start = genome_pos - frame
                if codon_start < 0 or codon_start + 3 > len(genome_seq):
                    continue
                codon = genome_seq[codon_start: codon_start + 3]
            else:
                codon_start = genome_pos - (2 - frame)
                if codon_start < 0 or codon_start + 3 > len(genome_seq):
                    continue
                codon_genomic = genome_seq[codon_start: codon_start + 3]
                codon = str(Seq(codon_genomic).reverse_complement())
            pos_in_codon = (gene_pos_1b - 1) % 3
            codon_list = list(codon)
            if codon_list[pos_in_codon] != base:
                continue
            mutated = 'T' if base == 'C' else 'A'
            codon_list[pos_in_codon] = mutated
            mutated_codon = ''.join(codon_list)
            if aa_of(codon) == aa_of(mutated_codon):
                syn += 1
            else:
                non_syn += 1
        total_mut = syn + non_syn
        frac = (non_syn / total_mut) if total_mut > 0 else float('nan')
        result[gene_id] = {
            'observed_nonsyn_sites': non_syn,
            'observed_syn_sites': syn,
            'observed_total_cg_sites': total_mut,
            'observed_nonsyn_fraction': frac
        }
    return result


def write_expected_vs_observed_nonsyn_csv(
    label: str,
    output_results_dir: str,
    expected: Dict[str, Dict[str, float]],
    observed: Dict[str, Dict[str, float]]
) -> None:
    rows = []
    for gene_id, exp in expected.items():
        e_non = exp.get('expected_nonsyn_sites', 0)
        e_syn = exp.get('expected_syn_sites', 0)
        e_frac = exp.get('expected_nonsyn_fraction', float('nan'))
        obs = observed.get(gene_id, {})
        o_non = obs.get('observed_nonsyn_sites', 0)
        o_syn = obs.get('observed_syn_sites', 0)
        o_frac = obs.get('observed_nonsyn_fraction', float('nan'))
        rows.append({
            'gene_id': gene_id,
            'expected_nonsyn_sites': e_non,
            'expected_syn_sites': e_syn,
            'expected_nonsyn_fraction': e_frac,
            'observed_nonsyn_sites': o_non,
            'observed_syn_sites': o_syn,
            'observed_nonsyn_fraction': o_frac,
            'observed_minus_expected_fraction': (o_frac - e_frac) if (not np.isnan(o_frac) and not np.isnan(e_frac)) else float('nan')
        })
    columns = [
        'gene_id',
        'expected_nonsyn_sites',
        'expected_syn_sites',
        'expected_nonsyn_fraction',
        'observed_nonsyn_sites',
        'observed_syn_sites',
        'observed_nonsyn_fraction',
        'observed_minus_expected_fraction'
    ]
    df = pd.DataFrame(rows, columns=columns)
    out_path = os.path.join(output_results_dir, f'{label}_expected_vs_observed_nonsyn_fraction.csv')
    df.to_csv(out_path, index=False)
    logger.info(f'Wrote per-gene expected vs observed non-syn fractions: {out_path} ({len(df)} rows)')

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
    logger.info("Initializing PROVEAN score database (SQLite3)")
    score_db = ProveanScoreDB(refs['prov_score_sql_db'])
    
    # Log configuration settings
    logger.info("Configuration settings:")
    logger.info(f"Reference files:")
    logger.info(f"  Genome: {refs['genomic_fna']}")
    logger.info(f"  Annotation: {refs['annotation']}")
    logger.info(f"  Codon table: {refs['codon_table']}")
    logger.info(f"  PROVEAN score database: {refs['prov_score_sql_db']}")
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
    
    # Separate control and treated sample mutation files
    control_jsons = []
    treated_jsons = []
    for js in mutation_jsons:
        sample_name = os.path.basename(js).replace('.json', '')
        if is_control_sample(sample_name):
            control_jsons.append(js)
        elif is_ems_sample(sample_name):
            treated_jsons.append(js)

    # Build callable unions up-front so we can filter observed mutations with the SAME exposure as permutations
    callable_union_ctrl = build_union_callable_positions(control_jsons, paths, args.mpileups)
    callable_union_treat = build_union_callable_positions(treated_jsons, paths, args.mpileups)
    callable_inter_ctrl = build_intersection_callable_positions(control_jsons, paths, args.mpileups)
    callable_inter_treat = build_intersection_callable_positions(treated_jsons, paths, args.mpileups)
    # Logging: site totals union vs intersection
    def count_sites(pos_map: Dict[str, set]) -> int:
        return sum(len(s) for s in pos_map.values())
    union_ctrl_n = count_sites(callable_union_ctrl)
    union_treat_n = count_sites(callable_union_treat)
    inter_ctrl_n = count_sites(callable_inter_ctrl)
    inter_treat_n = count_sites(callable_inter_treat)
    logger.info(f"Callable sites (controls): union={union_ctrl_n:,} intersection={inter_ctrl_n:,}")
    logger.info(f"Callable sites (treated): union={union_treat_n:,} intersection={inter_treat_n:,}")
    # Use union of callable positions for placement; fallback to all genic if union empty
    callable_ctrl = callable_union_ctrl
    callable_treat = callable_union_treat
    if not any(len(v) for v in callable_ctrl.values()):
        logger.warning("Callable (controls) union empty; falling back to all genic positions.")
        callable_ctrl = build_all_genic_positions(features)
    if not any(len(v) for v in callable_treat.values()):
        logger.warning("Callable (treated) union empty; falling back to all genic positions.")
        callable_treat = build_all_genic_positions(features)

    # Set max_workers for parallelization
    max_workers = min(20, os.cpu_count())
    
    # --- Original mutation data processing for controls AND treated ---
    logger.info("=== STARTING ORIGINAL MUTATION DATA PROCESSING ===")
    print("=== STARTING ORIGINAL MUTATION DATA PROCESSING ===")
    
    # --- Process controls ---
    logger.info("Processing original mutation data for controls...")
    
    # Collect all mismatch info
    all_mismatch_info = {
        'summary': {
            'total_genes_with_mismatches': 0,
            'total_mismatches': 0
        },
        'genes_with_mismatches': {},
        'reverse_strand_genes': [],
        'forward_strand_genes': []
    }
    # Diagnostics: observed placed vs counted per gene
    obs_dropout_rows: list[dict] = []
    
    # --- Merge all observed mutations for controls ---
    merged_control_mutations = {}
    if control_jsons:
        for js in control_jsons:
            with open(js) as jf:
                mut_dict = json.load(jf)
                for gene_id, gene_data in mut_dict.items():
                    if gene_id not in merged_control_mutations:
                        merged_control_mutations[gene_id] = {
                            'mutations': {},
                            'gene_len': gene_data['gene_len'],
                            'avg_cov': 0,
                            'sample_count': 0
                        }
                    # Merge mutations (presence/absence per site), restrict to EMS-type only (no callable filtering)
                    for mut_key in gene_data['mutations'].keys():
                        pos_str, mut_type = mut_key.split('_')
                        if mut_type not in ('C>T', 'G>A'):
                            continue
                        merged_control_mutations[gene_id]['mutations'][mut_key] = 1
                    merged_control_mutations[gene_id]['avg_cov'] += gene_data.get('avg_cov', 0)
                    merged_control_mutations[gene_id]['sample_count'] += 1
     # Average coverage
    for gene_id in merged_control_mutations:
        sc = merged_control_mutations[gene_id]['sample_count']
        if sc > 0:
            merged_control_mutations[gene_id]['avg_cov'] /= sc
    # Log observed pool size (unique mutated sites across genes) for controls
    if merged_control_mutations:
        observed_ctrl_total_sites = sum(len(g['mutations']) for g in merged_control_mutations.values())
        logger.info(f"Observed unique sites (controls, EMS-only): {observed_ctrl_total_sites:,}")
    # Calculate observed effect and stats for merged controls
    observed_control_results = {}
    if merged_control_mutations:
        for gene_id, gene_data in merged_control_mutations.items():
            effect, effect_with_nonsense, stats, gene_mismatch_info = nuc_to_provean_score(
                gene_data,
                gene_id,
                score_db,
                provean_config,
                paths,
                features,
                wolgenome,
                refs['codon_table']
            )
            
            # Collect mismatch info
            if gene_mismatch_info and gene_mismatch_info['summary']['total_mismatches'] > 0:
                all_mismatch_info['genes_with_mismatches'].update(gene_mismatch_info['genes_with_mismatches'])
                all_mismatch_info['summary']['total_genes_with_mismatches'] += gene_mismatch_info['summary']['total_genes_with_mismatches']
                all_mismatch_info['summary']['total_mismatches'] += gene_mismatch_info['summary']['total_mismatches']
                all_mismatch_info['reverse_strand_genes'].extend(gene_mismatch_info['reverse_strand_genes'])
                all_mismatch_info['forward_strand_genes'].extend(gene_mismatch_info['forward_strand_genes'])
            # Diagnostics: observed placed vs counted for this gene
            obs_dropout_rows.append({
                'group': 'controls',
                'gene_id': gene_id,
                'observed_placed_sites': len(gene_data['mutations']),
                'observed_counted_sites': (stats.get('non_syn_muts', 0) or 0) + (stats.get('syn_muts', 0) or 0),
                'observed_mismatch_count': gene_mismatch_info['summary']['total_mismatches'] if gene_mismatch_info and 'summary' in gene_mismatch_info else 0
            })
            # Compose output in original format (permutation data will be added later)
            observed_control_results[gene_id] = {
                'effect': effect,
                'effect_with_nonsense': effect_with_nonsense,
                'total_mutations': len(gene_data['mutations']),
                'gene_len': gene_data['gene_len'],
                'mutation_stats': stats,
                'permutation': {
                    'scores': [],
                    'scores_with_nonsense': [],
                    'deleterious_counts': [],
                    'total_mutations': [],
                    'mean': 0,
                    'std': 0,
                    'mean_with_nonsense': 0,
                    'std_with_nonsense': 0
                }
            }
    
    # --- Process treated ---
    logger.info("Processing original mutation data for treated...")
    
    # --- Merge all observed mutations for treated ---
    merged_treated_mutations = {}
    for js in treated_jsons:
        with open(js) as jf:
            mut_dict = json.load(jf)
            for gene_id, gene_data in mut_dict.items():
                if gene_id not in merged_treated_mutations:
                    merged_treated_mutations[gene_id] = {
                        'mutations': {},
                        'gene_len': gene_data['gene_len'],
                        'avg_cov': 0,
                        'sample_count': 0
                    }
                # Merge mutations (presence/absence per site), restrict to EMS-type only (no callable filtering)
                for mut_key in gene_data['mutations'].keys():
                    pos_str, mut_type = mut_key.split('_')
                    if mut_type not in ('C>T', 'G>A'):
                        continue
                    merged_treated_mutations[gene_id]['mutations'][mut_key] = 1
                merged_treated_mutations[gene_id]['avg_cov'] += gene_data.get('avg_cov', 0)
                merged_treated_mutations[gene_id]['sample_count'] += 1
    # Average coverage
    for gene_id in merged_treated_mutations:
        sc = merged_treated_mutations[gene_id]['sample_count']
        if sc > 0:
            merged_treated_mutations[gene_id]['avg_cov'] /= sc
    # Log observed pool size (unique mutated sites across genes) for treated
    observed_treat_total_sites = sum(len(g['mutations']) for g in merged_treated_mutations.values())
    logger.info(f"Observed unique sites (treated, EMS-only): {observed_treat_total_sites:,}")
    # Calculate observed effect and stats for merged treated
    observed_treated_results = {}
    for gene_id, gene_data in merged_treated_mutations.items():
        effect, effect_with_nonsense, stats, gene_mismatch_info = nuc_to_provean_score(
            gene_data,
            gene_id,
            score_db,
            provean_config,
            paths,
            features,
            wolgenome,
            refs['codon_table']
        )
        
        # Collect mismatch info
        if gene_mismatch_info and gene_mismatch_info['summary']['total_mismatches'] > 0:
            all_mismatch_info['genes_with_mismatches'].update(gene_mismatch_info['genes_with_mismatches'])
            all_mismatch_info['summary']['total_genes_with_mismatches'] += gene_mismatch_info['summary']['total_genes_with_mismatches']
            all_mismatch_info['summary']['total_mismatches'] += gene_mismatch_info['summary']['total_mismatches']
            all_mismatch_info['reverse_strand_genes'].extend(gene_mismatch_info['reverse_strand_genes'])
            all_mismatch_info['forward_strand_genes'].extend(gene_mismatch_info['forward_strand_genes'])
        # Diagnostics: observed placed vs counted for this gene
        obs_dropout_rows.append({
            'group': 'treated',
            'gene_id': gene_id,
            'observed_placed_sites': len(gene_data['mutations']),
            'observed_counted_sites': (stats.get('non_syn_muts', 0) or 0) + (stats.get('syn_muts', 0) or 0),
            'observed_mismatch_count': gene_mismatch_info['summary']['total_mismatches'] if gene_mismatch_info and 'summary' in gene_mismatch_info else 0
        })
        # Compose output in original format (permutation data will be added later)
        observed_treated_results[gene_id] = {
            'effect': effect,
            'effect_with_nonsense': effect_with_nonsense,
            'total_mutations': len(gene_data['mutations']),
            'gene_len': gene_data['gene_len'],
            'mutation_stats': stats,
            'permutation': {
                'scores': [],
                'scores_with_nonsense': [],
                'deleterious_counts': [],
                'total_mutations': [],
                'mean': 0,
                'std': 0,
                'mean_with_nonsense': 0,
                'std_with_nonsense': 0
            }
        }
    
    logger.info("=== FINISHED ORIGINAL MUTATION DATA PROCESSING ===")
    print("=== FINISHED ORIGINAL MUTATION DATA PROCESSING ===")
    
    # Write final mismatch debug file
    if all_mismatch_info['summary']['total_mismatches'] > 0:
        # Remove duplicates from gene lists
        all_mismatch_info['reverse_strand_genes'] = list(set(all_mismatch_info['reverse_strand_genes']))
        all_mismatch_info['forward_strand_genes'] = list(set(all_mismatch_info['forward_strand_genes']))
        
        # Save to debug file
        debug_file = "mismatch_debug.json"
        with open(debug_file, 'w') as f:
            json.dump(all_mismatch_info, f, indent=2)
        
        print(f"Detailed mismatch info saved to {debug_file}")
        logger.info(f"Detailed mismatch info saved to {debug_file}")
        
        # Print final summary
        print(f"FINAL MISMATCH SUMMARY: {all_mismatch_info['summary']['total_genes_with_mismatches']} genes with mismatches, {all_mismatch_info['summary']['total_mismatches']} total mismatches")
        logger.info(f"FINAL MISMATCH SUMMARY: {all_mismatch_info['summary']['total_genes_with_mismatches']} genes with mismatches, {all_mismatch_info['summary']['total_mismatches']} total mismatches")
        
        # Print summary by strand
        reverse_count = len(all_mismatch_info['reverse_strand_genes'])
        forward_count = len(all_mismatch_info['forward_strand_genes'])
        print(f"  Reverse strand genes with mismatches: {reverse_count}")
        print(f"  Forward strand genes with mismatches: {forward_count}")
        logger.info(f"  Reverse strand genes with mismatches: {reverse_count}")
        logger.info(f"  Forward strand genes with mismatches: {forward_count}")
    else:
        print("No mismatches found in original data")
        logger.info("No mismatches found in original data")
    
    # --- Permutation processing for controls AND treated ---
    # --- Compute expected vs observed non-syn fractions BEFORE permutations ---
    # (callable_union_ctrl and callable_union_treat already computed above and reused here)
    expected_nonsyn_control = compute_expected_nonsyn_fraction(features, wolgenome, callable_ctrl)
    observed_nonsyn_control = compute_observed_nonsyn_fraction(control_jsons, features, wolgenome, callable_ctrl)
    write_expected_vs_observed_nonsyn_csv("control", paths['results'], expected_nonsyn_control, observed_nonsyn_control)
    expected_nonsyn_treated = compute_expected_nonsyn_fraction(features, wolgenome, callable_treat)
    observed_nonsyn_treated = compute_observed_nonsyn_fraction(treated_jsons, features, wolgenome, callable_treat)
    write_expected_vs_observed_nonsyn_csv("treated", paths['results'], expected_nonsyn_treated, observed_nonsyn_treated)

    logger.info("=== STARTING PERMUTATION PROCESSING ===")
    print("=== STARTING PERMUTATION PROCESSING ===")
    
    # --- Generate permutation profiles for controls ---
    logger.info(f"Running permutation test for pooled control samples: {len(control_jsons)} files.")
    observed_kmers_codon_control = collect_observed_5mer_codon_contexts(
        control_jsons,
        wolgenome,
        features,
        kmer_size=5
    )
    # Log total observed sites feeding permutations (deduped by site)
    obs_seed_ctrl_total = sum(observed_kmers_codon_control.values())
    logger.info(f"[controls] observed EMS unique sites for permutation seeding: {obs_seed_ctrl_total:,}")
    # reuse callable_union_ctrl built above
    kmer_sites_control = collect_genic_5mer_sites(
        wolgenome,
        features,
        kmer_size=5,
        callable_positions=callable_ctrl
    )
    summarize_strata_capacity(observed_kmers_codon_control, kmer_sites_control, 'controls', paths['results'])
    permutation_profiles_control = generate_permutation_profiles(
        observed_kmers_codon_control,
        kmer_sites_control,
        n_permutations=1000,
        random_seed=None,
        max_workers=max_workers
    )
    # Global placed totals per permutation (pre-scoring)
    placed_totals_ctrl = [sum(len(g.get('mutations', {})) for g in prof.values()) for prof in permutation_profiles_control]
    if placed_totals_ctrl:
        logger.info(f"[controls] permutation placed totals: mean={np.mean(placed_totals_ctrl):.1f} min={np.min(placed_totals_ctrl)} max={np.max(placed_totals_ctrl)} n={len(placed_totals_ctrl)}")
        try:
            out_csv = os.path.join(paths['results'], 'placed_totals_permutation_controls.csv')
            with open(out_csv, 'w') as f:
                f.write('perm_index,placed_total\n')
                for i, v in enumerate(placed_totals_ctrl, 1):
                    f.write(f"{i},{v}\n")
            logger.info(f"Wrote permutation placed totals: {out_csv}")
        except Exception as e:
            logger.warning(f"Failed writing placed totals CSV (controls): {e}")
    permuted_results_control = process_permuted_provean_scores(
        permutation_profiles_control,
        score_db,
        provean_config,
        paths,
        features,
        refs['genomic_fna'],
        refs['annotation'],
        refs['codon_table'],
        max_workers=max_workers
    )
    
    # --- Generate permutation profiles for treated ---
    logger.info(f"Running permutation test for pooled treated samples: {len(treated_jsons)} files.")
    observed_kmers_codon_treated = collect_observed_5mer_codon_contexts(
        treated_jsons,
        wolgenome,
        features,
        kmer_size=5
    )
    obs_seed_treat_total = sum(observed_kmers_codon_treated.values())
    logger.info(f"[treated] observed EMS unique sites for permutation seeding: {obs_seed_treat_total:,}")
    # reuse callable_union_treat built above
    kmer_sites_treated = collect_genic_5mer_sites(
        wolgenome,
        features,
    kmer_size=5,
    callable_positions=callable_treat
    )
    summarize_strata_capacity(observed_kmers_codon_treated, kmer_sites_treated, 'treated', paths['results'])
    permutation_profiles_treated = generate_permutation_profiles(
        observed_kmers_codon_treated,
        kmer_sites_treated,
        n_permutations=1000,
        random_seed=None,
        max_workers=max_workers
    )
    # Global placed totals per permutation (pre-scoring)
    placed_totals_treated = [sum(len(g.get('mutations', {})) for g in prof.values()) for prof in permutation_profiles_treated]
    if placed_totals_treated:
        logger.info(f"[treated] permutation placed totals: mean={np.mean(placed_totals_treated):.1f} min={np.min(placed_totals_treated)} max={np.max(placed_totals_treated)} n={len(placed_totals_treated)}")
        try:
            out_csv = os.path.join(paths['results'], 'placed_totals_permutation_treated.csv')
            with open(out_csv, 'w') as f:
                f.write('perm_index,placed_total\n')
                for i, v in enumerate(placed_totals_treated, 1):
                    f.write(f"{i},{v}\n")
            logger.info(f"Wrote permutation placed totals: {out_csv}")
        except Exception as e:
            logger.warning(f"Failed writing placed totals CSV (treated): {e}")
    permuted_results_treated = process_permuted_provean_scores(
        permutation_profiles_treated,
        score_db,
        provean_config,
        paths,
        features,
        refs['genomic_fna'],
        refs['annotation'],
        refs['codon_table'],
        max_workers=max_workers
    )
    
    logger.info("=== FINISHED PERMUTATION PROCESSING ===")
    print("=== FINISHED PERMUTATION PROCESSING ===")
    
    # --- Combine results ---
    logger.info("=== COMBINING RESULTS ===")
    print("=== COMBINING RESULTS ===")
    
    # --- Add permutation results to observed control results ---
    for gene_id, gene_data in observed_control_results.items():
        perm = permuted_results_control.get(gene_id, {})
        observed_control_results[gene_id]['permutation'] = {
            'scores': perm.get('permutation_effects', []),
            'scores_with_nonsense': perm.get('permutation_effects_with_nonsense', []),
            'deleterious_counts': perm.get('deleterious_counts', []),
            'total_mutations': perm.get('total_mutations', []),
            'non_syn_counts': perm.get('non_syn_counts', []),
            'syn_counts': perm.get('syn_counts', []),
            'mean': perm.get('mean', 0),
            'std': perm.get('std', 0),
            'mean_with_nonsense': perm.get('mean_with_nonsense', 0),
            'std_with_nonsense': perm.get('std_with_nonsense', 0)
        }
    
    with open(f"{paths['results']}/permuted_provean_results_controls.json", "w") as f:
        json.dump(observed_control_results, f, indent=2)
    
    # --- Add permutation results to observed treated results ---
    for gene_id, gene_data in observed_treated_results.items():
        perm = permuted_results_treated.get(gene_id, {})
        observed_treated_results[gene_id]['permutation'] = {
            'scores': perm.get('permutation_effects', []),
            'scores_with_nonsense': perm.get('permutation_effects_with_nonsense', []),
            'deleterious_counts': perm.get('deleterious_counts', []),
            'total_mutations': perm.get('total_mutations', []),
            'non_syn_counts': perm.get('non_syn_counts', []),
            'syn_counts': perm.get('syn_counts', []),
            'mean': perm.get('mean', 0),
            'std': perm.get('std', 0),
            'mean_with_nonsense': perm.get('mean_with_nonsense', 0),
            'std_with_nonsense': perm.get('std_with_nonsense', 0)
        }
    
    with open(f"{paths['results']}/permuted_provean_results_treated.json", "w") as f:
        json.dump(observed_treated_results, f, indent=2)
    
    # --- Compute potential synonymous sites for controls ---
    potential_syn_control = compute_potential_synonymous_sites(features, wolgenome, callable_ctrl)
    observed_syn_hits_control = compute_observed_synonymous_hits(control_jsons, features=features, wolgenome=wolgenome, callable_positions=callable_ctrl)
    write_syn_site_fraction_csv("control", paths['results'], potential_syn_control, observed_syn_hits_control)

    # --- Compute potential synonymous sites for treated ---
    potential_syn_treated = compute_potential_synonymous_sites(features, wolgenome, callable_treat)
    observed_syn_hits_treated = compute_observed_synonymous_hits(treated_jsons, features=features, wolgenome=wolgenome, callable_positions=callable_treat)
    write_syn_site_fraction_csv("treated", paths['results'], potential_syn_treated, observed_syn_hits_treated)

    logger.success("Pipeline completed successfully")

    # Write observed diagnostics CSV (placed vs counted and mismatches per gene)
    if obs_dropout_rows:
        obs_df = pd.DataFrame(obs_dropout_rows)
        obs_path = os.path.join(paths['results'], 'dropout_diagnostics_observed.csv')
        obs_df.to_csv(obs_path, index=False)
        logger.info(f"Wrote observed dropout diagnostics: {obs_path}")

    # Write per-permutation placed vs counted diagnostics for controls and treated
    def write_perm_dropout_diag(group_name: str, perm_results: dict):
        if not perm_results:
            return
        # Determine number of permutations from the first gene present
        first_gene = next(iter(perm_results.keys()), None)
        if first_gene is None:
            return
        num_perms = len(perm_results[first_gene].get('placed_counts', []))
        placed_totals = np.zeros(num_perms, dtype=int)
        counted_totals = np.zeros(num_perms, dtype=int)
        mismatch_totals = np.zeros(num_perms, dtype=int)
        for gene_id, vals in perm_results.items():
            pc = np.array(vals.get('placed_counts', []), dtype=int)
            cc = np.array(vals.get('counted_counts', []), dtype=int)
            mc = np.array(vals.get('mismatch_counts', []), dtype=int)
            if pc.size != num_perms:
                # pad if needed (should not happen due to alignment)
                pc = np.pad(pc, (0, max(0, num_perms - pc.size)))
            if cc.size != num_perms:
                cc = np.pad(cc, (0, max(0, num_perms - cc.size)))
            if mc.size != num_perms:
                mc = np.pad(mc, (0, max(0, num_perms - mc.size)))
            placed_totals += pc
            counted_totals += cc
            mismatch_totals += mc
        dropped = placed_totals - counted_totals
        df = pd.DataFrame({
            'perm_index': np.arange(1, num_perms + 1),
            'placed_total': placed_totals,
            'counted_total': counted_totals,
            'dropped_total': dropped,
            'mismatch_total': mismatch_totals
        })
        outp = os.path.join(paths['results'], f'dropout_diagnostics_permutations_{group_name}.csv')
        df.to_csv(outp, index=False)
        logger.info(f"Wrote permutation dropout diagnostics: {outp}")

    write_perm_dropout_diag('controls', permuted_results_control)
    write_perm_dropout_diag('treated', permuted_results_treated)

if __name__ == '__main__':
    main()