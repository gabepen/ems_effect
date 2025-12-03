#!/usr/bin/env python3
"""
Enhanced mutation rate estimator with statistical improvements.

This script implements multiple mutation rate estimation methods:
- Simple rates: Low estimate (mutated positions / total depth) and High estimate (total alt alleles / total depth)
- Alpha correction: Background false positive rate estimation from controls
- GLM analysis: Poisson/Negative Binomial regression for better uncertainty quantification
- Coverage-dependent analysis: Rate estimation across coverage bins

Processes only EMS mutations: C>T and G>A at G/C sites.
Includes sample name simplification and comprehensive ranking plots.
"""
import argparse
import glob
import csv
import os
import math
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter
import numpy as np
import re
import scipy.stats as stats
from scipy.stats import binom
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import NegativeBinomial as NBDiscrete
import gzip
from collections import defaultdict
import pickle
from Bio.Seq import Seq



def proportion_ci(successes, total, alpha=0.05):
    """Return estimate, lower, upper using Wald normal approximation."""
    if total == 0:
        return 0.0, 0.0, 0.0
    p_hat = successes / total
    se = math.sqrt(p_hat * (1 - p_hat) / total)
    z = 1.96
    lower = max(0.0, p_hat - z * se)
    upper = min(1.0, p_hat + z * se)
    return p_hat, lower, upper


def derive_sample_label(sample_name, treated_str='EMS', control_str='NT'):
    """Derive cleaner sample labels using the logic from ems_mutation_rate.py"""
    s_str = str(sample_name)
    
    if treated_str in s_str:
        # Capture group number with optional separators and optional leading zeros, anywhere in string
        m = re.search(f'(?i){treated_str}[\\s\\-_]*0*(\\d+)', s_str)
        base = f"{treated_str}{int(m.group(1))}" if m else None
    elif control_str in s_str:
        m = re.search(f'(?i){control_str}[\\s\\-_]*0*(\\d+)', s_str)
        base = f"{control_str}{int(m.group(1))}" if m else None
    if not base:
        return s_str
    # Capture 3d/7d anywhere (case-insensitive), with flexible separators
    d = re.search(r'(?i)(?:^|[^A-Za-z0-9])(3|7)[\s\-_]*d(?:[^A-Za-z0-9]|$)', s_str)
    if d:
        return f"{base}_{d.group(1).lower()}d"
    return base


def extract_treatment_day(sample_name: str) -> str:
    """Extract treatment day from sample name.
    
    Returns:
        '3d', '7d', or 'no_label' for treated samples
        'control' for control samples
    """
    s_str = str(sample_name)
    
    # Check if control
    if "NT" in s_str:
        return "control"
    
    # Check for 3d or 7d
    d = re.search(r'(?i)(?:^|[^A-Za-z0-9])(3|7)[\s\-_]*d(?:[^A-Za-z0-9]|$)', s_str)
    if d:
        return f"{d.group(1).lower()}d"
    
    # If treated but no day label
    if "EMS" in s_str:
        return "no_label"
    
    # Default
    return "unknown"


def load_exclusion_mask(mask_file):
    """Load exclusion mask from file."""
    excluded_sites = set()
    with open(mask_file) as f:
        header = next(f, None)  # Skip header
        for line in f:
            chrom, pos = line.strip().split('\t')
            excluded_sites.add((chrom, pos))
    return excluded_sites


def estimate_alpha_from_controls(df):
    """Estimate background false positive rate from control samples using weighted average."""
    controls = df[df['is_control']]
    if len(controls) == 0:
        return 0.0
    
    # Weight by total depth (more reads = more reliable estimate)
    total_mutations = (controls['finite_sites_mutations']).sum()
    total_depth = controls['total_depth'].sum()
    alpha = total_mutations / total_depth if total_depth > 0 else 0.0
    return alpha


def estimate_alpha_from_control_sites(control_files, min_alt=1, excluded_sites=None):
    """Estimate alpha from raw control site data (most accurate approach)."""
    total_control_mutations = 0
    total_control_depth = 0
    
    for filepath in control_files:
        with open(filepath) as f:
            reader = csv.reader(f, delimiter="\t")
            header = next(reader, None)
            for row in reader:
                if len(row) < 9:
                    continue
                chrom, pos, ref, ref_count, A_count, C_count, G_count, T_count, depth = row
                
                # Convert to appropriate types
                ref_count = int(ref_count)
                A_count = int(A_count)
                C_count = int(C_count)
                G_count = int(G_count)
                T_count = int(T_count)
                depth = int(depth)
                
                # Only consider G/C sites (EMS targets)
                if ref not in {"G", "C"}:
                    continue
                
                # Apply exclusion mask if provided
                if excluded_sites and (chrom, pos) in excluded_sites:
                    continue
                    
                total_control_depth += depth
                
                # Calculate EMS mutations: C>T and G>A only
                if ref == "C":
                    ems_count = T_count  # C>T mutations
                elif ref == "G":
                    ems_count = A_count  # G>A mutations
                else:
                    ems_count = 0
                
                if ems_count >= min_alt:
                    total_control_mutations += ems_count
    
    alpha = total_control_mutations / total_control_depth if total_control_depth > 0 else 0.0
    return alpha


def parse_gff(gff_file):
    """Parse GFF file to extract gene regions."""
    gene_regions = []
    with open(gff_file) as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            fields = line.strip().split('\t')
            if len(fields) >= 9 and fields[2] == 'gene':
                chrom = fields[0]
                start = int(fields[3])
                end = int(fields[4])
                gene_id = fields[8].split(';')[0].replace('ID=', '') if 'ID=' in fields[8] else f"gene_{len(gene_regions)}"
                gene_regions.append((chrom, start, end, gene_id))
    return gene_regions


def parse_gff_cds(gff_file):
    """Parse GFF file to extract CDS regions for synonymous/non-synonymous classification."""
    cds_regions = []
    with open(gff_file) as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            fields = line.strip().split('\t')
            if len(fields) >= 9 and fields[2] == 'CDS':
                chrom = fields[0]
                start = int(fields[3])
                end = int(fields[4])
                strand = fields[6]
                # Extract gene ID from attributes
                attrs = fields[8]
                gene_id = None
                for attr in attrs.split(';'):
                    if attr.startswith('Parent=') or attr.startswith('gene_id=') or attr.startswith('ID='):
                        gene_id = attr.split('=')[1] if '=' in attr else None
                        break
                if gene_id is None:
                    gene_id = f"gene_{len(cds_regions)}"
                cds_regions.append((chrom, start, end, strand, gene_id))
    return cds_regions


def load_genome_sequences(genome_fasta):
    """Load genome sequences from FASTA file."""
    seqs = {}
    current = None
    def _read(handle):
        nonlocal current
        for raw in handle:
            line = raw.rstrip('\n')
            if not line:
                continue
            if line.startswith('>'):
                current = line[1:].split()[0]
                seqs[current] = []
            else:
                if current is not None:
                    seqs[current].append(line.upper())
    if genome_fasta.endswith('.gz'):
        with gzip.open(genome_fasta, 'rt') as fh:
            _read(fh)
    else:
        with open(genome_fasta, 'r') as fh:
            _read(fh)
    for k in list(seqs.keys()):
        seqs[k] = ''.join(seqs[k])
    return seqs


def get_codon_context(seqs, chrom, pos_1based, strand='+'):
    """Get the codon containing a position and determine if mutation is synonymous.
    
    For reverse strand genes, the CDS coordinates in GFF are in forward strand coordinates,
    but we need to reverse complement the sequence to get the actual codon.
    """
    if chrom not in seqs:
        return None, None, None
    
    seq = seqs[chrom]
    pos_0based = pos_1based - 1
    
    if pos_0based < 0 or pos_0based >= len(seq):
        return None, None, None
    
    # For forward strand, extract codon directly
    if strand == '+':
        codon_start = (pos_0based // 3) * 3
        codon_pos = pos_0based % 3
        if codon_start + 3 > len(seq):
            return None, None, None
        codon = seq[codon_start:codon_start + 3]
    else:
        # For reverse strand, CDS coordinates are in forward strand but sequence is reverse complemented
        # We need to find the codon in the reverse complemented sequence
        # First, get the codon boundaries in forward strand coordinates
        codon_start = (pos_0based // 3) * 3
        codon_pos = pos_0based % 3
        
        if codon_start + 3 > len(seq):
            return None, None, None
        
        # Extract codon from forward strand
        codon_fwd = seq[codon_start:codon_start + 3]
        
        # Reverse complement to get actual codon
        complement = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G', 'N': 'N'}
        codon = ''.join([complement.get(b, 'N') for b in codon_fwd[::-1]])
        # Position within codon is reversed
        codon_pos = 2 - codon_pos
    
    if len(codon) != 3 or 'N' in codon:
        return None, None, None
    
    return codon, codon_pos, codon_start


# Cached genetic code dictionary (codon -> amino acid)
_genetic_code_cache = None

def _load_genetic_code_from_json(codon_table_path=None):
    """Load genetic code from JSON file and return codon -> amino acid mapping."""
    global _genetic_code_cache
    
    if _genetic_code_cache is not None:
        return _genetic_code_cache
    
    import json
    
    if codon_table_path is None:
        # Default: look for 11.json in data/references relative to workspace root
        # Try to find workspace root by looking for common markers
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Go up from ems_effect/src/rate_modeling to workspace root
        workspace_root = os.path.join(current_dir, '..', '..', '..', '..')
        workspace_root = os.path.abspath(workspace_root)
        codon_table_path = os.path.join(workspace_root, 'data', 'references', '11.json')
    
    genetic_code = {}
    
    try:
        with open(codon_table_path, 'r') as f:
            aa_to_codons = json.load(f)
        
        # Invert the mapping: amino acid -> codons becomes codon -> amino acid
        # First, process all non-start codons
        for aa, codons in aa_to_codons.items():
            if aa != "starts":
                for codon in codons:
                    genetic_code[codon.upper()] = aa
        
        # Then handle start codons - they code for M (or their existing amino acid if already mapped)
        if "starts" in aa_to_codons:
            for codon in aa_to_codons["starts"]:
                codon_upper = codon.upper()
                # Start codons code for M, but preserve existing mapping if already set
                # (e.g., ATG is already M, TTG/CTG are already L)
                if codon_upper not in genetic_code:
                    genetic_code[codon_upper] = 'M'
        
        _genetic_code_cache = genetic_code
    except FileNotFoundError:
        # Fallback: minimal genetic code if file does not exist
        raise ValueError(f"Genetic code file {codon_table_path} not found")
        sys.exit(1)
    except Exception as e:
        raise ValueError(f"Failed to load genetic code from {codon_table_path}: {e}")
        sys.exit(1)
        
    return genetic_code

def is_synonymous_mutation(codon, codon_pos, ref_base, alt_base, codon_table_path=None):
    """Determine if a mutation is synonymous (doesn't change amino acid).
    
    Args:
        codon: Original codon (3 bases)
        codon_pos: Position within codon (0-based)
        ref_base: Reference base at that position
        alt_base: Alternate base (mutated base)
        codon_table_path: Optional path to JSON codon table file
    
    Returns:
        True if synonymous, False if non-synonymous, None if invalid
    """
    genetic_code = _load_genetic_code_from_json(codon_table_path)
    
    codon = codon.upper()
    alt_base = alt_base.upper()
    
    if codon not in genetic_code:
        return None
    
    # Get original amino acid
    orig_aa = genetic_code[codon]
    
    # Create mutated codon
    codon_list = list(codon)
    codon_list[codon_pos] = alt_base
    mutated_codon = ''.join(codon_list)
    
    if mutated_codon not in genetic_code:
        return None
    
    # Get mutated amino acid
    mutated_aa = genetic_code[mutated_codon]
    
    # Check if synonymous
    return orig_aa == mutated_aa

def genomic_gc_content(genome_fasta_or_seqs, chrom=None, start=None, end=None, exclusion_mask=None):
    """Calculate GC content for a genome sequence or entire genome.
    
    Since the data uses a single chromosome, chrom parameter is optional.
    If chrom is None, automatically uses the single chromosome in the genome.
    
    Args:
        genome_fasta_or_seqs: Either a file path (str) to FASTA file, or a loaded 
                              dictionary from load_genome_sequences() where keys are 
                              sequence identifiers and values are sequence strings
        chrom: Sequence identifier (str). If None, automatically uses the single chromosome
        start: Start position (0-based). If None, uses 0
        end: End position (0-based, exclusive). If None, uses end of sequence
        exclusion_mask: Set of (chrom, pos) tuples to exclude from GC calculation.
                        pos should be a string (1-based position from exclusion mask file).
                        If None, uses empty set (no exclusions).
    
    Returns:
        GC content as float (0.0 to 1.0)
    """
    # Initialize exclusion_mask to empty set if None (faster than checking None in loop)
    if exclusion_mask is None:
        exclusion_mask = set()
    
    # Load genome if file path provided, otherwise use provided dictionary
    if isinstance(genome_fasta_or_seqs, str):
        seqs = load_genome_sequences(genome_fasta_or_seqs)
    else:
        seqs = genome_fasta_or_seqs
    
    # If chrom not specified, use the single chromosome (assumes only one)
    if chrom is None:
        if len(seqs) == 0:
            raise ValueError("Genome dictionary is empty")
        if len(seqs) > 1:
            raise ValueError(f"Multiple sequences found ({len(seqs)}). Specify chrom parameter.")
        chrom = list(seqs.keys())[0]
    
    # Get sequence
    if chrom not in seqs:
        raise ValueError(f"Sequence identifier '{chrom}' not found in genome")
    
    seq = seqs[chrom]
    if start is None:
        start = 0
    if end is None:
        end = len(seq)
    
    # Ensure valid range
    start = max(0, start)
    end = min(len(seq), end)
    
    if start >= end:
        return 0.0
    
    # Calculate GC content, excluding masked sites
    gc_count = 0
    total_count = 0
    
    for pos_0based in range(start, end):
        pos_1based = pos_0based + 1
        
        # Check if this position is excluded (membership check in empty set is fast)
        if (chrom, str(pos_1based)) in exclusion_mask:
            continue  # Skip excluded sites
        
        base = seq[pos_0based]
        if base in ('G', 'C'):
            gc_count += 1
        total_count += 1
    
    if total_count == 0:
        return 0.0
    
    return gc_count / total_count

def create_genomic_bins(gene_regions, bin_size=2000):
    """Create 2kb bins for intergenic regions."""
    bins = []
    
    # Group genes by chromosome
    chrom_genes = {}
    for chrom, start, end, gene_id in gene_regions:
        if chrom not in chrom_genes:
            chrom_genes[chrom] = []
        chrom_genes[chrom].append((start, end, gene_id))
    
    # Create bins for each chromosome
    for chrom, genes in chrom_genes.items():
        genes.sort()  # Sort by start position
        
        # Add gene regions
        for start, end, gene_id in genes:
            bins.append((chrom, start, end, f"gene_{gene_id}"))
        
        # Add intergenic bins
        for i in range(len(genes) - 1):
            current_end = genes[i][1]
            next_start = genes[i + 1][0]
            
            # Create 2kb bins in the intergenic region
            pos = current_end
            while pos < next_start:
                bin_end = min(pos + bin_size, next_start)
                if bin_end - pos >= 100:  # Only create bins with at least 100bp
                    bins.append((chrom, pos, bin_end, f"intergenic_{chrom}_{pos}"))
                pos = bin_end
    
    return bins


def estimate_f_clonal_from_sites(counts_file, genomic_regions, min_alt=5, vaf_threshold=0.01, min_cov=10, excluded_sites=None):
    """Estimate f-clonal for each genomic region from high-confidence mutations."""
    region_f_clonal = {}
    
    # Collect VAFs for each region
    region_vafs = {region[3]: [] for region in genomic_regions}
    
    with open(counts_file) as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)
        for row in reader:
            if len(row) < 9:
                continue
            chrom, pos, ref, ref_count, A_count, C_count, G_count, T_count, depth = row
            
            # Convert to appropriate types
            ref_count = int(ref_count)
            A_count = int(A_count)
            C_count = int(C_count)
            G_count = int(G_count)
            T_count = int(T_count)
            depth = int(depth)
            pos = int(pos)
            
            # Only consider G/C sites (EMS targets)
            if ref not in {"G", "C"}:
                continue
            
            # Apply exclusion mask if provided
            if excluded_sites and (chrom, pos) in excluded_sites:
                continue
            
            # Check coverage
            if depth < min_cov:
                continue
            
            # Calculate EMS mutations: C>T and G>A only
            if ref == "C":
                ems_count = T_count  # C>T mutations
            elif ref == "G":
                ems_count = A_count  # G>A mutations
            else:
                ems_count = 0
            
            # Only consider high-confidence mutations
            if ems_count >= min_alt:
                vaf = ems_count / depth
                if vaf >= vaf_threshold:
                    # Find which region this position belongs to
                    for region_chrom, region_start, region_end, region_id in genomic_regions:
                        if region_chrom == chrom and region_start <= pos <= region_end:
                            region_vafs[region_id].append(vaf)
                            break
    
    # Estimate f-clonal for each region
    for region_id, vafs in region_vafs.items():
        if len(vafs) >= 3:  # Need at least 3 mutations for reliable estimate
            # Use median VAF as f-clonal estimate
            f_clonal = np.median(vafs)
            region_f_clonal[region_id] = f_clonal
        else:
            # Not enough data, use default
            region_f_clonal[region_id] = 0.5
    
    return region_f_clonal


def calculate_detection_probability(depth, f_clonal=0.5, vaf_threshold=0.01, min_alt=1):
    """Calculate per-site detection probability s(n) given coverage."""
    if depth == 0:
        return 0.0
    
    # Calculate minimum alt reads needed to call mutation
    k_vaf = int(np.ceil(vaf_threshold * depth))
    k_thresh = max(k_vaf, min_alt)
    
    # Probability that Bin(n, f_clonal) >= k_thresh
    return binom.sf(k_thresh - 1, depth, f_clonal)


def apply_alpha_correction(df, alpha):
    """Apply alpha correction to existing rate estimates."""
    df = df.copy()
    
    # Apply alpha correction: adjusted_rate = max(0, raw_rate - alpha)
    df['low_rate_alpha_corrected'] = np.maximum(0, df['low_rate'] - alpha)
    df['high_rate_alpha_corrected'] = np.maximum(0, df['high_rate'] - alpha)
    
    # Adjust confidence intervals (subtract alpha from bounds)
    df['low_CI_low_alpha_corrected'] = np.maximum(0, df['low_CI_low'] - alpha)
    df['low_CI_high_alpha_corrected'] = np.maximum(0, df['low_CI_high'] - alpha)
    df['high_CI_low_alpha_corrected'] = np.maximum(0, df['high_CI_low'] - alpha)
    df['high_CI_high_alpha_corrected'] = np.maximum(0, df['high_CI_high'] - alpha)
    
    # Store alpha used for reference
    df['alpha_used'] = alpha
    
    return df


def estimate_rates_from_counts(filename, min_alt=1, excluded_sites=None):
    """
    Estimate mutation rates from a .counts file (output from collect_mutation_counts.py).
    
    Returns:
        dict with rate estimates and metadata
    """
    total_depth = 0
    infinite_sites_mutations = 0  # count of positions with mutations
    finite_sites_mutations = 0    # total count of alt alleles
    
    with open(filename) as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)
        for row in reader:
            if len(row) < 9:
                continue
            chrom, pos, ref, ref_count, A_count, C_count, G_count, T_count, depth = row
            
            # Convert to appropriate types
            ref_count = int(ref_count)
            A_count = int(A_count)
            C_count = int(C_count)
            G_count = int(G_count)
            T_count = int(T_count)
            depth = int(depth)
            
            # Only consider G/C sites (EMS targets)
            if ref not in {"G", "C"}:
                continue
            
            # Apply exclusion mask if provided
            if excluded_sites and (chrom, pos) in excluded_sites:
                continue
                
            total_depth += depth
            
            # Calculate EMS mutations: C>T and G>A only
            if ref == "C":
                ems_count = T_count  # C>T mutations
            elif ref == "G":
                ems_count = A_count  # G>A mutations
            else:
                ems_count = 0
            
            if ems_count >= min_alt:
                infinite_sites_mutations += 1
                finite_sites_mutations += ems_count

    # Calculate rate estimates
    inf_est, inf_lo, inf_hi = proportion_ci(infinite_sites_mutations, total_depth)
    fin_est, fin_lo, fin_hi = proportion_ci(finite_sites_mutations, total_depth)

    basename = os.path.basename(filename)
    sample_name = basename.replace(".counts", "")
    label = derive_sample_label(sample_name, treated_str='EMS', control_str='NT')

    return {
        "sample": sample_name,
        "label": label,
        "low_rate": inf_est,
        "low_CI_low": inf_lo,
        "low_CI_high": inf_hi,
        "high_rate": fin_est,
        "high_CI_low": fin_lo,
        "high_CI_high": fin_hi,
        "total_depth": total_depth,
        "infinite_sites_mutations": infinite_sites_mutations,
        "finite_sites_mutations": finite_sites_mutations,
        "is_control": "NT" in basename,
    }


def create_ranking_plots(df, output_prefix):
    """Create clear ranking plots for mutation rates."""
    
    # Sort by high rate for ranking
    df_sorted = df.sort_values('high_rate', ascending=False)
    
    # Create figure with subplots - much taller figure to give more room for y-axis labels
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 20))
    
    # Adjust subplot spacing to give more room for y-axis labels
    plt.subplots_adjust(left=0.15, right=0.95, top=0.92, bottom=0.08, hspace=0.2, wspace=0.25)
    
    # 1) High rate ranking (horizontal bar plot)
    colors = ['#FF7F0E' if not is_ctrl else '#1F77B4' for is_ctrl in df_sorted['is_control']]
    bars1 = ax1.barh(range(len(df_sorted)), df_sorted['high_rate'], color=colors, alpha=0.7)
    ax1.set_yticks(range(len(df_sorted)))
    ax1.set_yticklabels(df_sorted['label'], fontsize=12)
    ax1.set_xlabel('Per-base mutation probability (high estimate)', fontsize=14)
    fmt = ScalarFormatter(useMathText=True)
    fmt.set_scientific(True)
    fmt.set_powerlimits((-2, 2))
    ax1.xaxis.set_major_formatter(fmt)
    ax1.set_title('Mutation Rate Ranking - High Estimate', fontsize=16)
    ax1.invert_yaxis()
    
    # Add error bars for high rate
    y_pos = range(len(df_sorted))
    ax1.errorbar(df_sorted['high_rate'], y_pos, 
                xerr=[df_sorted['high_rate'] - df_sorted['high_CI_low'], 
                      df_sorted['high_CI_high'] - df_sorted['high_rate']],
                fmt='none', color='black', capsize=2, alpha=0.6)
    
    # 2) Low rate ranking (horizontal bar plot)
    bars2 = ax2.barh(range(len(df_sorted)), df_sorted['low_rate'], color=colors, alpha=0.7)
    ax2.set_yticks(range(len(df_sorted)))
    ax2.set_yticklabels(df_sorted['label'], fontsize=12)
    ax2.set_xlabel('Per-base mutation probability (low estimate)', fontsize=14)
    ax2.xaxis.set_major_formatter(fmt)
    ax2.set_title('Mutation Rate Ranking - Low Estimate', fontsize=16)
    ax2.invert_yaxis()
    
    # Add error bars for low rate
    ax2.errorbar(df_sorted['low_rate'], y_pos,
                xerr=[df_sorted['low_rate'] - df_sorted['low_CI_low'],
                      df_sorted['low_CI_high'] - df_sorted['low_rate']],
                fmt='none', color='black', capsize=2, alpha=0.6)
    
    # 3) Scatter plot: Low vs High rates
    control_mask = df['is_control']
    sample_mask = ~control_mask
    
    if control_mask.any():
        ax3.scatter(df.loc[control_mask, 'low_rate'], df.loc[control_mask, 'high_rate'], 
                   color='#1F77B4', s=100, alpha=0.7, label='Controls')
    
    if sample_mask.any():
        ax3.scatter(df.loc[sample_mask, 'low_rate'], df.loc[sample_mask, 'high_rate'], 
                   color='#FF7F0E', s=100, alpha=0.7, label='Samples')
    
    # Add error bars to scatter plot
    for _, row in df.iterrows():
        color = '#1F77B4' if row['is_control'] else '#FF7F0E'
        ax3.errorbar(row['low_rate'], row['high_rate'],
                    xerr=[[row['low_rate'] - row['low_CI_low']], [row['low_CI_high'] - row['low_rate']]],
                    yerr=[[row['high_rate'] - row['high_CI_low']], [row['high_CI_high'] - row['high_rate']]],
                    fmt='none', color=color, alpha=0.5, capsize=2)
    
    ax3.set_xlabel('Per-base mutation probability (low)', fontsize=14)
    ax3.set_ylabel('Per-base mutation probability (high)', fontsize=14)
    ax3.xaxis.set_major_formatter(fmt)
    ax3.yaxis.set_major_formatter(fmt)
    ax3.set_title('Low vs High Rate Estimates', fontsize=16)
    ax3.legend()
    
    
    # 4) Alpha-corrected rates (if available, otherwise rate difference)
    if 'high_rate_alpha_corrected' in df_sorted.columns:
        # Use alpha-corrected high rate
        corrected_rates = df_sorted['high_rate_alpha_corrected']
        colors_corrected = ['#FF7F0E' if not is_ctrl else '#1F77B4' for is_ctrl in df_sorted['is_control']]
        bars4 = ax4.barh(range(len(df_sorted)), corrected_rates, color=colors_corrected, alpha=0.7)
        ax4.set_xlabel('Per-base mutation probability (alpha-corrected)')
        ax4.xaxis.set_major_formatter(fmt)
        ax4.set_title('Alpha-Corrected Mutation Rates')
        
        # Add error bars for alpha-corrected rates if available
        if 'high_CI_low_alpha_corrected' in df_sorted.columns:
            ax4.errorbar(corrected_rates, range(len(df_sorted)),
                        xerr=[corrected_rates - df_sorted['high_CI_low_alpha_corrected'],
                              df_sorted['high_CI_high_alpha_corrected'] - corrected_rates],
                        fmt='none', color='black', capsize=2, alpha=0.6)
    else:
        # Fallback to rate difference if alpha correction not available
        df_sorted['rate_diff'] = df_sorted['high_rate'] - df_sorted['low_rate']
        colors_diff = ['#FF7F0E' if not is_ctrl else '#1F77B4' for is_ctrl in df_sorted['is_control']]
        bars4 = ax4.barh(range(len(df_sorted)), df_sorted['rate_diff'], color=colors_diff, alpha=0.7)
        ax4.set_xlabel('Rate Difference (High - Low)')
        ax4.xaxis.set_major_formatter(fmt)
        ax4.set_title('Difference Between High and Low Estimates')
        ax4.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    
    ax4.set_yticks(range(len(df_sorted)))
    ax4.set_yticklabels(df_sorted['label'], fontsize=12)
    ax4.invert_yaxis()
    
    # Add legend
    legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor='#FF7F0E', 
                             markersize=10, label='Samples'),
                      Line2D([0], [0], marker='o', color='w', markerfacecolor='#1F77B4', 
                             markersize=10, label='Controls')]
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))
    
    # Save plot
    plot_path = f"{output_prefix}_mutation_rate_ranking.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Simple rate ranking plots saved to {plot_path}")
    
    return df_sorted


def save_individual_simple_plots(df, output_prefix):
    """Save individual plots for simple rate analysis."""
    
    # 1) Low rate ranking
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    df_sorted = df.sort_values('low_rate', ascending=False)
    colors = ['#E74C3C' if not is_ctrl else '#3498DB' for is_ctrl in df_sorted['is_control']]
    bars = ax.barh(range(len(df_sorted)), df_sorted['low_rate'], color=colors, alpha=0.7)
    ax.set_yticks(range(len(df_sorted)))
    ax.set_yticklabels(df_sorted['label'], fontsize=10)
    ax.set_xlabel('Low Estimate Rate')
    ax.set_title('Low Rate Ranking')
    ax.invert_yaxis()
    
    # Add error bars if available
    if 'low_CI_low' in df_sorted.columns:
        y_pos = range(len(df_sorted))
        ax.errorbar(df_sorted['low_rate'], y_pos, 
                   xerr=[df_sorted['low_rate'] - df_sorted['low_CI_low'], 
                         df_sorted['low_CI_high'] - df_sorted['low_rate']],
                   fmt='none', color='black', capsize=2, alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_low_rate_ranking.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Low rate ranking saved to {output_prefix}_low_rate_ranking.png")
    
    # 2) High rate ranking
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    df_sorted = df.sort_values('high_rate', ascending=False)
    colors = ['#E74C3C' if not is_ctrl else '#3498DB' for is_ctrl in df_sorted['is_control']]
    bars = ax.barh(range(len(df_sorted)), df_sorted['high_rate'], color=colors, alpha=0.7)
    ax.set_yticks(range(len(df_sorted)))
    ax.set_yticklabels(df_sorted['label'], fontsize=10)
    ax.set_xlabel('High Estimate Rate')
    ax.set_title('High Rate Ranking')
    ax.invert_yaxis()
    
    # Add error bars if available
    if 'high_CI_low' in df_sorted.columns:
        y_pos = range(len(df_sorted))
        ax.errorbar(df_sorted['high_rate'], y_pos, 
                   xerr=[df_sorted['high_rate'] - df_sorted['high_CI_low'], 
                         df_sorted['high_CI_high'] - df_sorted['high_rate']],
                   fmt='none', color='black', capsize=2, alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_high_rate_ranking.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"High rate ranking saved to {output_prefix}_high_rate_ranking.png")
    
    # 3) Scatter plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    control_mask = df['is_control']
    sample_mask = ~control_mask
    
    if control_mask.any():
        ax.scatter(df.loc[control_mask, 'low_rate'], df.loc[control_mask, 'high_rate'], 
                   color='#3498DB', s=100, alpha=0.7, label='Controls')
    
    if sample_mask.any():
        ax.scatter(df.loc[sample_mask, 'low_rate'], df.loc[sample_mask, 'high_rate'], 
                   color='#E74C3C', s=100, alpha=0.7, label='Samples')
    
    # Add error bars to scatter plot
    for _, row in df.iterrows():
        color = '#3498DB' if row['is_control'] else '#E74C3C'
        ax.errorbar(row['low_rate'], row['high_rate'],
                    xerr=[[row['low_rate'] - row['low_CI_low']], [row['low_CI_high'] - row['low_rate']]],
                    yerr=[[row['high_rate'] - row['high_CI_low']], [row['high_CI_high'] - row['high_rate']]],
                    fmt='none', color=color, alpha=0.5, capsize=2)
    
    ax.set_xlabel('Low Estimate Rate')
    ax.set_ylabel('High Estimate Rate')
    ax.set_title('Low vs High Rate Estimates')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_rate_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Rate comparison scatter plot saved to {output_prefix}_rate_comparison.png")
    
    # 4) Alpha-corrected rate (if available)
    if 'high_rate_alpha_corrected' in df.columns:
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        df_sorted = df.sort_values('high_rate_alpha_corrected', ascending=False)
        colors = ['#E74C3C' if not is_ctrl else '#3498DB' for is_ctrl in df_sorted['is_control']]
        bars = ax.barh(range(len(df_sorted)), df_sorted['high_rate_alpha_corrected'], color=colors, alpha=0.7)
        ax.set_yticks(range(len(df_sorted)))
        ax.set_yticklabels(df_sorted['label'], fontsize=10)
        ax.set_xlabel('Alpha-Corrected High Rate')
        ax.set_title('Alpha-Corrected Rate Ranking')
        ax.invert_yaxis()
        
        # Add error bars if available
        if 'high_rate_alpha_corrected_CI_low' in df_sorted.columns:
            y_pos = range(len(df_sorted))
            ax.errorbar(df_sorted['high_rate_alpha_corrected'], y_pos, 
                       xerr=[df_sorted['high_rate_alpha_corrected'] - df_sorted['high_rate_alpha_corrected_CI_low'], 
                             df_sorted['high_rate_alpha_corrected_CI_high'] - df_sorted['high_rate_alpha_corrected']],
                       fmt='none', color='black', capsize=2, alpha=0.6)
        
        plt.tight_layout()
        plt.savefig(f"{output_prefix}_alpha_corrected_ranking.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Alpha-corrected rate ranking saved to {output_prefix}_alpha_corrected_ranking.png")


def create_site_level_glm_plots(df, output_prefix):
    """Create plots for per-sample site-level GLM rates."""
    if 'site_glm_rate' not in df.columns or df['site_glm_rate'].notna().sum() == 0:
        print("No per-sample site-level GLM data available for plotting")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Ranking plot
    df_sorted = df.sort_values('site_glm_rate', ascending=False)
    colors = ['#FF7F0E' if not is_ctrl else '#1F77B4' for is_ctrl in df_sorted['is_control']]
    ax1.barh(range(len(df_sorted)), df_sorted['site_glm_rate'], color=colors, alpha=0.7)
    ax1.set_yticks(range(len(df_sorted)))
    ax1.set_yticklabels(df_sorted['label'], fontsize=12)
    ax1.set_xlabel('Per-base mutation probability (site-level GLM)', fontsize=14)
    fmt = ScalarFormatter(useMathText=True)
    fmt.set_scientific(True)
    fmt.set_powerlimits((-2, 2))
    ax1.xaxis.set_major_formatter(fmt)
    ax1.set_title('Site-Level GLM Rate Ranking', fontsize=16)
    ax1.invert_yaxis()

    # Error bars if available
    if 'site_glm_CI_low' in df_sorted.columns and 'site_glm_CI_high' in df_sorted.columns:
        y_pos = range(len(df_sorted))
        xerr_low = df_sorted['site_glm_rate'] - df_sorted['site_glm_CI_low']
        xerr_high = df_sorted['site_glm_CI_high'] - df_sorted['site_glm_rate']
        ax1.errorbar(df_sorted['site_glm_rate'], y_pos,
                     xerr=[xerr_low, xerr_high], fmt='none', color='black', capsize=2, alpha=0.6)

    # Comparison vs per-sample GLM if available
    if 'glm_rate' in df.columns and df['glm_rate'].notna().any():
        control_mask = df['is_control']
        sample_mask = ~control_mask
        if control_mask.any():
            ax2.scatter(df.loc[control_mask, 'glm_rate'], df.loc[control_mask, 'site_glm_rate'],
                        color='#1F77B4', s=100, alpha=0.7, label='Controls')
        if sample_mask.any():
            ax2.scatter(df.loc[sample_mask, 'glm_rate'], df.loc[sample_mask, 'site_glm_rate'],
                        color='#FF7F0E', s=100, alpha=0.7, label='Samples')
        ax2.set_xlabel('Per-base probability (per-sample GLM)', fontsize=14)
        ax2.set_ylabel('Per-base probability (site-level GLM)', fontsize=14)
        ax2.xaxis.set_major_formatter(fmt)
        ax2.yaxis.set_major_formatter(fmt)
        ax2.set_title('Per-Sample GLM vs Site-Level GLM', fontsize=16)
        ax2.legend()
    else:
        ax2.text(0.5, 0.5, 'Per-sample GLM not available', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Site-Level GLM Analysis', fontsize=16)

    plt.tight_layout()
    plot_path = f"{output_prefix}_site_glm_analysis.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Site-level GLM analysis plots saved to {plot_path}")


def save_individual_site_level_glm_plots(df, output_prefix):
    """Save individual ranking plot for per-sample site-level GLM rates."""
    if 'site_glm_rate' not in df.columns or df['site_glm_rate'].notna().sum() == 0:
        return
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    df_sorted = df.sort_values('site_glm_rate', ascending=False)
    colors = ['#FF7F0E' if not is_ctrl else '#1F77B4' for is_ctrl in df_sorted['is_control']]
    ax.barh(range(len(df_sorted)), df_sorted['site_glm_rate'], color=colors, alpha=0.7)
    ax.set_yticks(range(len(df_sorted)))
    ax.set_yticklabels(df_sorted['label'], fontsize=10)
    ax.set_xlabel('Per-base mutation probability (site-level GLM)')
    fmt = ScalarFormatter(useMathText=True)
    fmt.set_scientific(True)
    fmt.set_powerlimits((-2, 2))
    ax.xaxis.set_major_formatter(fmt)
    ax.set_title('Site-Level GLM Rate Ranking')
    ax.invert_yaxis()
    if 'site_glm_CI_low' in df_sorted.columns and 'site_glm_CI_high' in df_sorted.columns:
        y_pos = range(len(df_sorted))
        xerr_low = df_sorted['site_glm_rate'] - df_sorted['site_glm_CI_low']
        xerr_high = df_sorted['site_glm_CI_high'] - df_sorted['site_glm_rate']
        ax.errorbar(df_sorted['site_glm_rate'], y_pos,
                    xerr=[xerr_low, xerr_high], fmt='none', color='black', capsize=2, alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_site_glm_ranking.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Site-level GLM ranking saved to {output_prefix}_site_glm_ranking.png")

def load_site_level_data_no_context(counts_dir: str, exclusion_mask: str | None = None):
    """Load site-level rows across all samples without sequence-context features.

    Returns a DataFrame with columns: chrom, pos, treatment (0 control, 1 treated),
    depth, ems_count, sample.
    """
    excluded_sites = None
    if exclusion_mask and os.path.exists(exclusion_mask):
        excluded_sites = load_exclusion_mask(exclusion_mask)

    rows = []
    counts_files = sorted(glob.glob(os.path.join(counts_dir, "*.counts")))
    for path in counts_files:
        sample_name = os.path.basename(path).replace(".counts", "")
        is_control = ("NT" in sample_name)
        treatment = 0 if is_control else 1
        with open(path) as f:
            reader = csv.reader(f, delimiter="\t")
            header = next(reader, None)
            for row in reader:
                if len(row) < 9:
                    continue
                chrom, pos, ref, ref_count, A_count, C_count, G_count, T_count, depth = row
                if excluded_sites and (chrom, pos) in excluded_sites:
                    continue
                try:
                    A_i = int(A_count)
                    C_i = int(C_count)
                    G_i = int(G_count)
                    T_i = int(T_count)
                    depth_i = int(depth)
                except Exception:
                    continue
                if depth_i <= 0:
                    continue
                if ref not in {"G", "C"}:
                    continue
                # Strict EMS-only counting
                ems_count = T_i if ref == "C" else A_i
                rows.append({
                    "chrom": chrom,
                    "pos": pos,
                    "treatment": treatment,
                    "depth": depth_i,
                    "ems_count": ems_count,
                    "sample": sample_name,
                })
    return pd.DataFrame(rows)


def fit_site_level_glm_no_context(df: pd.DataFrame, method: str = "poisson", alpha: float = 0.0):
    """Fit site-level GLM with Treatment covariate and log(depth) offset.

    Optionally subtract background alpha at counts level per site (max with 0).
    Returns the fitted model result object and a small summary dict.
    """
    if df.empty:
        return None, {}

    work = df.copy()
    work["log_depth"] = np.log(work["depth"].values)

    # Apply alpha subtraction at counts level if provided
    if alpha and alpha > 0:
        work["ems_adj"] = np.maximum(0.0, work["ems_count"].values - alpha * work["depth"].values)
        y = work["ems_adj"].values
    else:
        y = work["ems_count"].values

    X = work[["treatment"]]
    design = sm.add_constant(X, has_constant='add')

    if method == "poisson":
        fam = sm.families.Poisson()
        model = sm.GLM(y, design, family=fam, offset=work["log_depth"].values)
        result = model.fit()
    else:
        # For negative binomial, estimate alpha from all data (controls + treated)
        # This is correct when using treatment covariate approach
        from statsmodels.discrete.discrete_model import NegativeBinomial as NBDiscrete
        exposure = np.exp(work["log_depth"].values)
        # Warm start with Poisson
        pois_model = sm.GLM(y, design, family=sm.families.Poisson(), offset=work["log_depth"].values)
        pois_result = pois_model.fit()
        start_params = np.r_[pois_result.params.values, 0.1]  # Add dispersion parameter
        
        nb_model = NBDiscrete(y, design, exposure=exposure)
        result = nb_model.fit(start_params=start_params, disp=False)

    # Interpret coefficients as per-base rates for control (intercept) and treated
    beta0 = result.params.get("const", np.nan)
    beta_t = result.params.get("treatment", np.nan)
    rate_control = float(np.exp(beta0)) if np.isfinite(beta0) else np.nan
    rate_treated = float(np.exp(beta0 + beta_t)) if np.isfinite(beta0) and np.isfinite(beta_t) else np.nan

    summary = {
        "aic": float(getattr(result, "aic", np.nan)),
        "bic": float(getattr(result, "bic", np.nan)) if hasattr(result, "bic") else np.nan,
        "llf": float(getattr(result, "llf", np.nan)),
        "rate_control": rate_control,
        "rate_treated": rate_treated,
        "coef_const": float(beta0) if np.isfinite(beta0) else np.nan,
        "coef_treatment": float(beta_t) if np.isfinite(beta_t) else np.nan,
    }

    return result, summary


def fit_site_level_glm_per_sample(site_df: pd.DataFrame, method: str = "poisson", alpha: float = 0.0):
    """Fit intercept-only site-level GLM separately for each sample.

    Returns a list of dicts with sample-level rate estimates and intervals.
    """
    if site_df is None or site_df.empty:
        return []

    results = []
    samples = sorted(site_df["sample"].unique())

    for sample_name in samples:
        df = site_df[site_df["sample"] == sample_name].copy()
        if df.empty:
            continue

        df["log_depth"] = np.log(df["depth"].values)

        if alpha and alpha > 0:
            df["ems_adj"] = np.maximum(0.0, df["ems_count"].values - alpha * df["depth"].values)
            y = df["ems_adj"].values
        else:
            y = df["ems_count"].values

        # Intercept-only design
        design = sm.add_constant(pd.DataFrame(index=df.index), has_constant='add')

        if method == "poisson":
            fam = sm.families.Poisson()
        else:
            fam = sm.families.NegativeBinomial()

        try:
            model = sm.GLM(y, design, family=fam, offset=df["log_depth"].values)
            result = model.fit()

            beta0 = result.params.get("const", np.nan)
            rate = float(np.exp(beta0)) if np.isfinite(beta0) else np.nan

            # Confidence interval for intercept, transformed
            try:
                ci_beta = result.conf_int().loc["const"].values
                ci_low = float(np.exp(ci_beta[0]))
                ci_high = float(np.exp(ci_beta[1]))
            except Exception:
                ci_low = np.nan
                ci_high = np.nan

            results.append({
                "sample": sample_name,
                "site_glm_rate": rate,
                "site_glm_CI_low": ci_low,
                "site_glm_CI_high": ci_high,
                "site_glm_aic": float(getattr(result, "aic", np.nan)),
                "site_glm_llf": float(getattr(result, "llf", np.nan)),
                "site_glm_rows": int(len(df)),
                "site_glm_depth": int(df["depth"].sum()),
                "site_glm_mutations": float(df["ems_count"].sum()),
            })
        except Exception as e:
            print(f"Warning: Site-level GLM failed for sample {sample_name}: {e}")
            results.append({
                "sample": sample_name,
                "site_glm_rate": np.nan,
                "site_glm_CI_low": np.nan,
                "site_glm_CI_high": np.nan,
                "site_glm_aic": np.nan,
                "site_glm_llf": np.nan,
                "site_glm_rows": int(len(df)),
                "site_glm_depth": int(df["depth"].sum()),
                "site_glm_mutations": float(df["ems_count"].sum()),
            })

    return results


def load_site_level_with_ref(counts_dir: str, exclusion_mask: str | None = None) -> pd.DataFrame:
    """Load site-level rows with reference base retained for min-alt logic.

    Columns: chrom, pos(int), ref, depth, ems_count, sample, is_control(bool)
    """
    excluded_sites = None
    if exclusion_mask and os.path.exists(exclusion_mask):
        excluded_sites = load_exclusion_mask(exclusion_mask)

    rows = []
    counts_files = sorted(glob.glob(os.path.join(counts_dir, "*.counts")))
    for path in counts_files:
        sample = os.path.basename(path).replace(".counts", "")
        is_ctrl = ("NT" in sample)
        with open(path) as f:
            reader = csv.reader(f, delimiter="\t")
            _ = next(reader, None)
            for row in reader:
                if len(row) < 9:
                    continue
                chrom, pos, ref, ref_count, A_count, C_count, G_count, T_count, depth = row
                if ref not in {"C", "G"}:
                    continue
                if excluded_sites and (chrom, pos) in excluded_sites:
                    continue
                try:
                    A_i = int(A_count); C_i = int(C_count); G_i = int(G_count); T_i = int(T_count)
                    depth_i = int(depth); pos_i = int(pos)
                except Exception:
                    continue
                if depth_i <= 0:
                    continue
                ems = T_i if ref == "C" else A_i
                rows.append({
                    "chrom": chrom,
                    "pos": pos_i,
                    "ref": ref,
                    "depth": depth_i,
                    "ems_count": ems,
                    "sample": sample,
                    "is_control": is_ctrl,
                })
    return pd.DataFrame(rows)


def load_site_level_with_category(counts_dir: str, gff_file: str, genome_fasta: str, 
                                   exclusion_mask: str | None = None, codon_table_path: str | None = None) -> pd.DataFrame:
    """Load site-level rows with category annotations (intergenic, synonymous, non-synonymous).
    
    Columns: chrom, pos(int), ref, depth, ems_count, sample, is_control(bool), category(str)
    """
    excluded_sites = None
    if exclusion_mask and os.path.exists(exclusion_mask):
        excluded_sites = load_exclusion_mask(exclusion_mask)
    
    # Parse GFF to get CDS regions
    print("Parsing GFF file for CDS regions...")
    cds_regions = parse_gff_cds(gff_file)
    print(f"Found {len(cds_regions)} CDS regions")
    
    # Build CDS lookup by chromosome
    cds_by_chrom = defaultdict(list)
    for chrom, start, end, strand, gene_id in cds_regions:
        cds_by_chrom[chrom].append((start, end, strand, gene_id))
    
    # Sort CDS regions by start position for each chromosome
    for chrom in cds_by_chrom:
        cds_by_chrom[chrom].sort(key=lambda x: x[0])
    
    # Load genome sequences
    print("Loading genome sequences...")
    seqs = load_genome_sequences(genome_fasta)
    print(f"Loaded sequences for {len(seqs)} chromosomes")
    
    def classify_site(chrom: str, pos: int, ref: str) -> str:
        """Classify a site as intergenic, synonymous, or non-synonymous."""
        # Check if position is in any CDS region
        if chrom not in cds_by_chrom:
            return "intergenic"
        
        cds_list = cds_by_chrom[chrom]
        for start, end, strand, gene_id in cds_list:
            if start <= pos <= end:
                # Position is in CDS, determine if mutation is synonymous
                # EMS mutations: C>T or G>A
                if ref == "C":
                    alt_base = "T"
                elif ref == "G":
                    alt_base = "A"
                else:
                    return "intergenic"  # Shouldn't happen for EMS sites
                
                # Get codon context
                codon, codon_pos, codon_start = get_codon_context(seqs, chrom, pos, strand)
                if codon is None:
                    return "non_synonymous"  # Default to non-synonymous if can't determine
                
                # Check if synonymous
                is_syn = is_synonymous_mutation(codon, codon_pos, ref, alt_base, codon_table_path=codon_table_path)
                if is_syn is None:
                    return "non_synonymous"  # Default to non-synonymous if can't determine
                
                return "synonymous" if is_syn else "non_synonymous"
        
        return "intergenic"
    
    # Load site-level data
    rows = []
    counts_files = sorted(glob.glob(os.path.join(counts_dir, "*.counts")))
    print(f"Processing {len(counts_files)} count files...")
    
    for path in counts_files:
        sample = os.path.basename(path).replace(".counts", "")
        is_ctrl = ("NT" in sample)
        with open(path) as f:
            reader = csv.reader(f, delimiter="\t")
            _ = next(reader, None)
            for row in reader:
                if len(row) < 9:
                    continue
                chrom, pos, ref, ref_count, A_count, C_count, G_count, T_count, depth = row
                if ref not in {"C", "G"}:
                    continue
                if excluded_sites and (chrom, pos) in excluded_sites:
                    continue
                try:
                    A_i = int(A_count); C_i = int(C_count); G_i = int(G_count); T_i = int(T_count)
                    depth_i = int(depth); pos_i = int(pos)
                except Exception:
                    continue
                if depth_i <= 0:
                    continue
                ems = T_i if ref == "C" else A_i
                
                # Classify site
                category = classify_site(chrom, pos_i, ref)
                
                # Extract treatment day
                treatment_day = extract_treatment_day(sample)
                
                rows.append({
                    "chrom": chrom,
                    "pos": pos_i,
                    "ref": ref,
                    "depth": depth_i,
                    "ems_count": ems,
                    "sample": sample,
                    "is_control": is_ctrl,
                    "category": category,
                    "treatment_day": treatment_day,
                })
    
    df = pd.DataFrame(rows)
    print(f"Loaded {len(df):,} sites")
    print(f"Category distribution:")
    print(df['category'].value_counts())
    print(f"Treatment day distribution:")
    print(df['treatment_day'].value_counts())
    return df


def fit_category_rates_nb_glm(site_df: pd.DataFrame, use_treatment_covariate: bool = True, 
                               nb_alpha: float | None = None):
    """Fit negative binomial GLM for rates across genomic regions (intergenic, synonymous, non-synonymous).
    
    Returns fitted model result and summary statistics.
    """
    if site_df is None or site_df.empty:
        print("No site-level data available")
        return None, None
    
    # Prepare data
    df = site_df.copy()
    df = df[df['depth'] > 0].copy()
    df['log_depth'] = np.log(df['depth'].values)
    df['treatment'] = (~df['is_control']).astype(int).values
    
    # Create category dummy variables (intergenic as reference)
    categories = ['intergenic', 'synonymous', 'non_synonymous']
    for cat in categories[1:]:  # Skip intergenic (reference)
        df[f'cat_{cat}'] = (df['category'] == cat).astype(int)
    
    # Prepare design matrix
    if use_treatment_covariate:
        # Model with interactions: log(E[Y]) = log(depth) + β₀ + β_treatment·treatment + 
        # β_syn·cat_synonymous + β_nonsyn·cat_non_synonymous + 
        # β_treatment_syn·treatment·cat_synonymous + β_treatment_nonsyn·treatment·cat_non_synonymous
        # This allows category effects to differ between controls and treated
        X_cols = ['treatment'] + [f'cat_{cat}' for cat in categories[1:]]
        # Add interaction terms: treatment × category
        for cat in categories[1:]:
            df[f'treatment_x_cat_{cat}'] = df['treatment'] * df[f'cat_{cat}']
            X_cols.append(f'treatment_x_cat_{cat}')
        X = df[X_cols]
    else:
        # Without treatment covariate, just category effects
        X = df[[f'cat_{cat}' for cat in categories[1:]]]
    
    design = sm.add_constant(X, has_constant='add')
    y = df['ems_count'].values
    offset = df['log_depth'].values
    
    # Fit negative binomial GLM
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            if nb_alpha is not None:
                # Use provided NB alpha
                fam = sm.families.NegativeBinomial(alpha=nb_alpha)
                model = sm.GLM(y, design, family=fam, offset=offset)
                result = model.fit()
                return result, df
            else:
                # Estimate NB alpha from all data (controls + treated) using discrete NB model
                # This is the correct approach when using treatment covariate
                from statsmodels.discrete.discrete_model import NegativeBinomial as NBDiscrete
                exposure = np.exp(offset)
                # Warm start with Poisson
                pois_model = sm.GLM(y, design, family=sm.families.Poisson(), offset=offset)
                pois_result = pois_model.fit()
                start_params = np.r_[pois_result.params.values, 0.1]  # Add dispersion parameter
                
                nb_model = NBDiscrete(y, design, exposure=exposure)
                result = nb_model.fit(start_params=start_params, disp=False)
                return result, df
            
    except Exception as e:
        print(f"Error fitting negative binomial GLM: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def _fit_absolute_effect_model(df, categories, method, alpha, log_5mer_offset=None):
    """
    Fit constant absolute effect model.
    
    Tests if treatment adds the same absolute number of mutations per base
    across all categories (not proportional to baseline).
    
    Step 1: Estimate control rates by category (with optional 5mer normalization)
    Step 2: Calculate excess mutations = observed - expected_control
    Step 3: Fit GLM to excess mutations with identity link
    Step 4: LRT comparing simple vs interaction models
    
    Args:
        df: DataFrame with site-level data
        categories: List of category names
        method: GLM method ('poisson' or 'negative_binomial')
        alpha: Background rate subtraction
        log_5mer_offset: Optional Series of log(expected_rate_5mer) for sequence context normalization
    
    Returns:
        dict with absolute_effect_result or None if failed
    """
    print("\n=== Using absolute effect model ===")
    print("Step 1: Fitting control-only model to get category-specific baseline rates")
    
    control_df = df[df['is_control']].copy()
    if len(control_df) == 0:
        print("Warning: No control samples found")
        return None
    
    # Fit control model
    X_control = control_df[[f'cat_{cat}' for cat in categories[1:]]]
    design_control = sm.add_constant(X_control, has_constant='add')
    y_control = control_df['ems_count'].values
    offset_control = control_df['log_depth'].values
    
    # Add 5mer offset for control samples if provided
    if log_5mer_offset is not None:
        valid_5mer_control = ~pd.isna(log_5mer_offset.loc[control_df.index])
        if valid_5mer_control.sum() > 0:
            offset_control = offset_control + log_5mer_offset.loc[control_df.index].fillna(0.0).values
            print(f"  Using 5mer normalization for {valid_5mer_control.sum()} control sites")
    
    if alpha > 0:
        y_control = np.maximum(0.0, y_control - alpha * control_df['depth'].values)
    
    try:
        if method == "poisson":
            model_control = sm.GLM(y_control, design_control, family=sm.families.Poisson(), offset=offset_control)
            result_control = model_control.fit()
        else:
            exposure_control = np.exp(offset_control)
            pois_model = sm.GLM(y_control, design_control, family=sm.families.Poisson(), offset=offset_control)
            pois_result = pois_model.fit()
            start_params = np.r_[pois_result.params.values, 0.1]
            nb_model = NBDiscrete(y_control, design_control, exposure=exposure_control)
            result_control = nb_model.fit(start_params=start_params, disp=False)
        
        # Extract control rates
        beta0_control = result_control.params.get('const', np.nan)
        beta_syn_control = result_control.params.get('cat_synonymous', 0.0)
        beta_nonsyn_control = result_control.params.get('cat_non_synonymous', 0.0)
        
        print(f"  Control rates: intergenic={np.exp(beta0_control):.2e}, "
              f"syn={np.exp(beta0_control + beta_syn_control):.2e}, "
              f"nonsyn={np.exp(beta0_control + beta_nonsyn_control):.2e}")
        
        # Step 2: Calculate excess mutations for ALL samples (control and treated)
        print("Step 2: Calculating excess mutations (observed - expected_control)")
        df['expected_control_count'] = 0.0
        for idx, row in df.iterrows():
            cat = row['category']
            if cat == 'intergenic':
                log_rate = beta0_control
            elif cat == 'synonymous':
                log_rate = beta0_control + beta_syn_control
            elif cat == 'non_synonymous':
                log_rate = beta0_control + beta_nonsyn_control
            else:
                log_rate = beta0_control
            
            expected_rate = np.exp(log_rate)
            df.loc[idx, 'expected_control_count'] = expected_rate * row['depth']
        
        df['excess_count'] = df['ems_count'] - df['expected_control_count']
        
        print(f"  Mean excess in controls: {df[df['is_control']]['excess_count'].mean():.2e}")
        print(f"  Mean excess in treated: {df[~df['is_control']]['excess_count'].mean():.2e}")
        
        # Step 3: Fit models to excess counts using Gaussian family with identity link
        # (excess can be negative, so we use Gaussian not Poisson)
        print("Step 3: Fitting models to excess mutations")
        
        # We model excess as a function of treatment and depth
        # Model: excess = β_treatment × depth × treatment + interactions
        
        # Create weighted depth variable for each category in treated samples
        df['depth_treatment'] = df['depth'] * df['treatment']
        df['depth_treatment_syn'] = df['depth'] * df['treatment'] * df['cat_synonymous']
        df['depth_treatment_nonsyn'] = df['depth'] * df['treatment'] * df['cat_non_synonymous']
        
        # Simple model: excess = β × depth × treatment (no intercept, constant effect)
        X_simple = df[['depth_treatment']]
        y_excess = df['excess_count'].values
        
        # Fit with Gaussian family (identity link) - no intercept needed
        model_simple = sm.GLM(y_excess, X_simple, family=sm.families.Gaussian())
        result_simple = model_simple.fit()
        
        beta_simple = result_simple.params.get('depth_treatment', np.nan)
        se_simple = result_simple.bse.get('depth_treatment', np.nan)
        print(f"  Simple model: β_treatment = {beta_simple:.2e} (SE={se_simple:.2e})")
        print(f"    → Treatment adds {beta_simple:.2e} mutations/base uniformly")
        
        # Interaction model: excess = β × depth × treatment + interactions
        X_interaction = df[['depth_treatment', 'depth_treatment_syn', 'depth_treatment_nonsyn']]
        model_interaction = sm.GLM(y_excess, X_interaction, family=sm.families.Gaussian())
        result_interaction = model_interaction.fit()
        
        beta_inter = result_interaction.params.get('depth_treatment', np.nan)
        beta_syn_inter = result_interaction.params.get('depth_treatment_syn', np.nan)
        beta_nonsyn_inter = result_interaction.params.get('depth_treatment_nonsyn', np.nan)
        
        print(f"  Interaction model:")
        print(f"    Intergenic: {beta_inter:.2e} mutations/base")
        print(f"    Synonymous: {beta_inter + beta_syn_inter:.2e} mutations/base")
        print(f"    Non-synonymous: {beta_inter + beta_nonsyn_inter:.2e} mutations/base")
        
        # Step 4: Likelihood ratio test
        # For Gaussian models, LR = (RSS_simple - RSS_interaction) / (RSS_interaction / (n - p_interaction))
        # But we'll use deviance which is analogous
        lr_stat = result_simple.deviance - result_interaction.deviance
        df_diff = 2  # 2 additional parameters in interaction model
        lr_pvalue = 1 - stats.chi2.cdf(lr_stat, df=df_diff)
        
        print(f"\n=== Absolute effect model results ===")
        print(f"LR test: LR={lr_stat:.4f}, p={lr_pvalue:.4e}")
        if lr_pvalue > 0.05:
            print(f"  → p > 0.05: Absolute treatment effect is CONSTANT across categories")
        else:
            print(f"  → p < 0.05: Absolute treatment effect DIFFERS across categories")
        print("="*60)
        
        return {
            'result_simple': result_simple,
            'result_interaction': result_interaction,
            'result_control': result_control,
            'beta0_control': beta0_control,
            'beta_syn_control': beta_syn_control,
            'beta_nonsyn_control': beta_nonsyn_control,
            'lr_stat': lr_stat,
            'lr_pvalue': lr_pvalue,
            'beta_simple': beta_simple,
            'se_simple': se_simple,
            'beta_intergenic': beta_inter,
            'beta_synonymous': beta_inter + beta_syn_inter,
            'beta_nonsyn': beta_inter + beta_nonsyn_inter,
            'df_modified': df
        }
        
    except Exception as e:
        print(f"Warning: Absolute effect model failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def _fit_control_baseline_model(df, categories, method, alpha, log_5mer_offset=None):
    """
    Fit control-baseline model: first estimate control rates, then test treatment effects.
    
    Args:
        df: DataFrame with site-level data
        categories: List of category names
        method: GLM method ('poisson' or 'negative_binomial')
        alpha: Background rate subtraction
        log_5mer_offset: Optional Series of log(expected_rate_5mer) for sequence context normalization
    
    Returns:
        dict with control_baseline_result or None if failed
    """
    print("\n=== Using control-baseline approach ===")
    print("Step 1: Fitting control-only model to get category-specific baseline rates")
    
    control_df = df[df['is_control']].copy()
    if len(control_df) == 0:
        print("Warning: No control samples found")
        return None
    
    # Fit control model
    X_control = control_df[[f'cat_{cat}' for cat in categories[1:]]]
    design_control = sm.add_constant(X_control, has_constant='add')
    y_control = control_df['ems_count'].values
    offset_control = control_df['log_depth'].values
    
    # Add 5mer offset for control samples if provided
    if log_5mer_offset is not None:
        valid_5mer_control = ~pd.isna(log_5mer_offset.loc[control_df.index])
        if valid_5mer_control.sum() > 0:
            offset_control = offset_control + log_5mer_offset.loc[control_df.index].fillna(0.0).values
            print(f"  Using 5mer normalization for {valid_5mer_control.sum()} control sites")
    
    if alpha > 0:
        y_control = np.maximum(0.0, y_control - alpha * control_df['depth'].values)
    
    try:
        if method == "poisson":
            model_control = sm.GLM(y_control, design_control, family=sm.families.Poisson(), offset=offset_control)
            result_control = model_control.fit()
        else:
            exposure_control = np.exp(offset_control)
            pois_model = sm.GLM(y_control, design_control, family=sm.families.Poisson(), offset=offset_control)
            pois_result = pois_model.fit()
            start_params = np.r_[pois_result.params.values, 0.1]
            nb_model = NBDiscrete(y_control, design_control, exposure=exposure_control)
            result_control = nb_model.fit(start_params=start_params, disp=False)
        
        # Extract control rates
        beta0_control = result_control.params.get('const', np.nan)
        beta_syn_control = result_control.params.get('cat_synonymous', 0.0)
        beta_nonsyn_control = result_control.params.get('cat_non_synonymous', 0.0)
        
        print(f"  Control rates: intergenic={np.exp(beta0_control):.2e}, "
              f"syn={np.exp(beta0_control + beta_syn_control):.2e}, "
              f"nonsyn={np.exp(beta0_control + beta_nonsyn_control):.2e}")
        
        # Calculate expected control rate for each site
        df['log_expected_control_rate'] = np.nan
        for idx, row in df.iterrows():
            cat = row['category']
            if cat == 'intergenic':
                log_rate = beta0_control
            elif cat == 'synonymous':
                log_rate = beta0_control + beta_syn_control
            elif cat == 'non_synonymous':
                log_rate = beta0_control + beta_nonsyn_control
            else:
                log_rate = beta0_control
            df.loc[idx, 'log_expected_control_rate'] = log_rate
        
        print("Step 2: Fitting treatment model with control rates as baseline offset")
        
        # Fit treatment models (simple and interaction)
        X_treatment = df[['treatment']]
        design_treatment = sm.add_constant(X_treatment, has_constant='add')
        y_all = df['ems_count'].values
        offset_treatment = df['log_depth'].values + df['log_expected_control_rate'].values
        
        # Add 5mer offset for all samples if provided
        if log_5mer_offset is not None:
            valid_5mer_all = ~pd.isna(log_5mer_offset)
            if valid_5mer_all.sum() > 0:
                offset_treatment = offset_treatment + log_5mer_offset.fillna(0.0).values
                print(f"  Using 5mer normalization for {valid_5mer_all.sum()} sites in treatment model")
        
        if alpha > 0:
            y_all = np.maximum(0.0, y_all - alpha * df['depth'].values)
        
        # Simple model
        if method == "poisson":
            model_simple = sm.GLM(y_all, design_treatment, family=sm.families.Poisson(), offset=offset_treatment)
            result_simple = model_simple.fit()
        else:
            exposure = np.exp(offset_treatment)
            pois_model = sm.GLM(y_all, design_treatment, family=sm.families.Poisson(), offset=offset_treatment)
            pois_result = pois_model.fit()
            start_params = np.r_[pois_result.params.values, 0.1]
            nb_model = NBDiscrete(y_all, design_treatment, exposure=exposure)
            result_simple = nb_model.fit(start_params=start_params, disp=False)
        
        beta_treatment_simple = result_simple.params.get('treatment', np.nan)
        beta_treatment_se_simple = result_simple.bse.get('treatment', np.nan)
        
        print("Step 3: Fitting model WITH category×treatment interactions")
        
        # Interaction model
        X_interaction = df[['treatment']].copy()
        for cat in categories[1:]:
            df[f'treatment_x_cat_{cat}'] = df['treatment'] * df[f'cat_{cat}']
            X_interaction = pd.concat([X_interaction, df[[f'treatment_x_cat_{cat}']]], axis=1)
        
        design_interaction = sm.add_constant(X_interaction, has_constant='add')
        
        if method == "poisson":
            model_interaction = sm.GLM(y_all, design_interaction, family=sm.families.Poisson(), offset=offset_treatment)
            result_interaction = model_interaction.fit()
        else:
            nb_model = NBDiscrete(y_all, design_interaction, exposure=exposure)
            pois_model = sm.GLM(y_all, design_interaction, family=sm.families.Poisson(), offset=offset_treatment)
            pois_result = pois_model.fit()
            pois_coefs = pois_result.params.values
            alpha_val = 0.1
            start_params = np.r_[pois_coefs, alpha_val]
            result_interaction = nb_model.fit(start_params=start_params, disp=False)
        
        # Likelihood ratio test
        lr_stat = 2 * (result_interaction.llf - result_simple.llf)
        lr_pvalue = 1 - stats.chi2.cdf(lr_stat, df=2)
        
        print(f"\n=== Control-baseline model results ===")
        print(f"Treatment effect (constant): β={beta_treatment_simple:.4f} (SE={beta_treatment_se_simple:.4f})")
        print(f"Rate ratio: exp(β)={np.exp(beta_treatment_simple):.4f}")
        print(f"LR test: LR={lr_stat:.4f}, p={lr_pvalue:.4e}")
        if lr_pvalue > 0.05:
            print(f"  → p > 0.05: Treatment effect is SAME across categories")
        else:
            print(f"  → p < 0.05: Treatment effect DIFFERS across categories")
        print("="*60)
        
        return {
            'result_simple': result_simple,
            'result_interaction': result_interaction,
            'result_control': result_control,
            'beta0_control': beta0_control,
            'beta_syn_control': beta_syn_control,
            'beta_nonsyn_control': beta_nonsyn_control,
            'lr_stat': lr_stat,
            'lr_pvalue': lr_pvalue,
            'beta_treatment_simple': beta_treatment_simple,
            'beta_treatment_se_simple': beta_treatment_se_simple,
            'df_modified': df
        }
        
    except Exception as e:
        print(f"Warning: Control-baseline model failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def _fit_standard_model(df, categories, method, alpha, use_treatment_covariate, log_5mer_offset=None):
    """
    Fit standard interaction model.
    
    Args:
        df: DataFrame with site-level data
        categories: List of category names
        method: GLM method ('poisson' or 'negative_binomial')
        alpha: Background rate subtraction
        use_treatment_covariate: Include treatment covariate
        log_5mer_offset: Optional Series of log(expected_rate_5mer) for sequence context normalization
    
    Returns:
        fitted model result or None if failed
    """
    if use_treatment_covariate:
        X_cols = ['treatment'] + [f'cat_{cat}' for cat in categories[1:]]
        for cat in categories[1:]:
            df[f'treatment_x_cat_{cat}'] = df['treatment'] * df[f'cat_{cat}']
            X_cols.append(f'treatment_x_cat_{cat}')
        X = df[X_cols]
    else:
        X = df[[f'cat_{cat}' for cat in categories[1:]]]
    
    design = sm.add_constant(X, has_constant='add')
    y = df['ems_count'].values
    offset = df['log_depth'].values
    
    # Add 5mer offset if provided: log(E[Y]) = log(depth) + log(expected_rate_5mer) + X*beta
    if log_5mer_offset is not None:
        valid_5mer = ~pd.isna(log_5mer_offset)
        if valid_5mer.sum() > 0:
            offset = offset + log_5mer_offset.fillna(0.0).values
            print(f"  Using 5mer normalization for {valid_5mer.sum()} sites")
    
    if alpha > 0:
        y = np.maximum(0.0, y - alpha * df['depth'].values)
    
    try:
        if method == "poisson":
            model = sm.GLM(y, design, family=sm.families.Poisson(), offset=offset)
            result = model.fit()
        else:
            exposure = np.exp(offset)
            pois_model = sm.GLM(y, design, family=sm.families.Poisson(), offset=offset)
            pois_result = pois_model.fit()
            start_params = np.r_[pois_result.params.values, 0.1]
            nb_model = NBDiscrete(y, design, exposure=exposure)
            result = nb_model.fit(start_params=start_params, disp=False)
        
        return result
        
    except Exception as e:
        print(f"Error fitting standard model: {e}")
        import traceback
        traceback.print_exc()
        return None


def _calculate_rates_and_cis(result, categories, use_treatment_covariate, control_baseline_result=None, mean_expected_rates=None):
    """
    Calculate rates and confidence intervals from fitted model.
    
    If control_baseline_result is provided, uses control-baseline approach:
    - Control rates from Step 1 (result_control)
    - Treated rates as control_rate × exp(treatment_effects) from Step 2 (result_interaction)
    
    Otherwise, extracts rates from standard interaction model.
    
    If mean_expected_rates is provided (from 5mer normalization), the rates are multiplied
    by the appropriate mean expected rate to convert from relative (exp(beta)) to absolute
    per-base rates.
    
    Args:
        result: Fitted GLM result
        categories: List of category names
        use_treatment_covariate: Whether treatment covariate was used
        control_baseline_result: Optional control-baseline model results
        mean_expected_rates: Optional dict mapping 'category_treatment' to mean expected 5mer rate
    
    Returns:
        tuple of (rates dict, cis dict)
    """
    rates = {}
    cis = {}
    
    # Control-baseline approach: extract rates differently
    if control_baseline_result is not None:
        # Check if this is an absolute effect model (identity link) or control-baseline (log link)
        is_absolute_effect_model = 'beta_intergenic' in control_baseline_result
        
        if is_absolute_effect_model:
            print("Extracting rates from ABSOLUTE EFFECT model (identity link)")
        else:
            print("Extracting rates from control-baseline model (log link)")
        
        # Step 1: Get control rates from control-only model
        result_control = control_baseline_result['result_control']
        beta0_control = control_baseline_result['beta0_control']
        beta_syn_control = control_baseline_result['beta_syn_control']
        beta_nonsyn_control = control_baseline_result['beta_nonsyn_control']
        
        # Control rates
        # If 5mer normalization was used, multiply by mean expected rate to get actual rates
        control_rates = {
            'intergenic': np.exp(beta0_control),
            'synonymous': np.exp(beta0_control + beta_syn_control),
            'non_synonymous': np.exp(beta0_control + beta_nonsyn_control)
        }
        
        # Apply 5mer normalization scaling if available
        if mean_expected_rates is not None:
            for cat in control_rates:
                key = f'{cat}_control'
                if key in mean_expected_rates and np.isfinite(mean_expected_rates[key]):
                    control_rates[cat] = control_rates[cat] * mean_expected_rates[key]
        
        # Control CIs (from result_control)
        try:
            vcov_control = result_control.cov_params()
            for cat in categories:
                if cat == 'intergenic':
                    var = vcov_control.loc['const', 'const']
                    se = np.sqrt(var) if var > 0 else np.nan
                    ci_low = control_rates[cat] * np.exp(-1.96 * se) if np.isfinite(se) else np.nan
                    ci_high = control_rates[cat] * np.exp(1.96 * se) if np.isfinite(se) else np.nan
                elif cat == 'synonymous':
                    var = (vcov_control.loc['const', 'const'] + 
                           vcov_control.loc['cat_synonymous', 'cat_synonymous'] +
                           2 * vcov_control.loc['const', 'cat_synonymous'])
                    se = np.sqrt(var) if var > 0 else np.nan
                    ci_low = control_rates[cat] * np.exp(-1.96 * se) if np.isfinite(se) else np.nan
                    ci_high = control_rates[cat] * np.exp(1.96 * se) if np.isfinite(se) else np.nan
                elif cat == 'non_synonymous':
                    var = (vcov_control.loc['const', 'const'] + 
                           vcov_control.loc['cat_non_synonymous', 'cat_non_synonymous'] +
                           2 * vcov_control.loc['const', 'cat_non_synonymous'])
                    se = np.sqrt(var) if var > 0 else np.nan
                    ci_low = control_rates[cat] * np.exp(-1.96 * se) if np.isfinite(se) else np.nan
                    ci_high = control_rates[cat] * np.exp(1.96 * se) if np.isfinite(se) else np.nan
                
                rates[f'{cat}_control'] = control_rates[cat]
                cis[f'{cat}_control'] = (ci_low, ci_high)
        except Exception as e:
            print(f"Warning: Failed to calculate control CIs: {e}")
            for cat in categories:
                rates[f'{cat}_control'] = control_rates[cat]
                cis[f'{cat}_control'] = (np.nan, np.nan)
        
        # Step 2: Get treatment effects
        if is_absolute_effect_model:
            # ABSOLUTE EFFECT MODEL (identity link): treated_rate = control_rate + β_absolute
            # Get absolute treatment effects directly from the model
            beta_abs_intergenic = control_baseline_result.get('beta_intergenic', np.nan)
            beta_abs_synonymous = control_baseline_result.get('beta_synonymous', np.nan)
            beta_abs_nonsyn = control_baseline_result.get('beta_nonsyn', np.nan)
            
            # Get SEs from the interaction model
            result_interaction = control_baseline_result['result_interaction']
            try:
                vcov_interaction = result_interaction.cov_params()
            except:
                vcov_interaction = None
            
            # Store absolute effects for reporting
            rates['absolute_effect_intergenic'] = beta_abs_intergenic
            rates['absolute_effect_synonymous'] = beta_abs_synonymous
            rates['absolute_effect_nonsyn'] = beta_abs_nonsyn
            
            for cat in categories:
                control_rate = control_rates[cat]
                
                # Get absolute treatment effect for this category
                if cat == 'intergenic':
                    abs_effect = beta_abs_intergenic
                    if vcov_interaction is not None and 'depth_treatment' in vcov_interaction.index:
                        var_treatment = vcov_interaction.loc['depth_treatment', 'depth_treatment']
                    else:
                        var_treatment = np.nan
                elif cat == 'synonymous':
                    abs_effect = beta_abs_synonymous
                    if vcov_interaction is not None:
                        try:
                            var_treatment = (vcov_interaction.loc['depth_treatment', 'depth_treatment'] +
                                           vcov_interaction.loc['depth_treatment_syn', 'depth_treatment_syn'] +
                                           2 * vcov_interaction.loc['depth_treatment', 'depth_treatment_syn'])
                        except:
                            var_treatment = np.nan
                    else:
                        var_treatment = np.nan
                elif cat == 'non_synonymous':
                    abs_effect = beta_abs_nonsyn
                    if vcov_interaction is not None:
                        try:
                            var_treatment = (vcov_interaction.loc['depth_treatment', 'depth_treatment'] +
                                           vcov_interaction.loc['depth_treatment_nonsyn', 'depth_treatment_nonsyn'] +
                                           2 * vcov_interaction.loc['depth_treatment', 'depth_treatment_nonsyn'])
                        except:
                            var_treatment = np.nan
                    else:
                        var_treatment = np.nan
                
                # IDENTITY LINK: treated_rate = control_rate + abs_effect
                treated_rate = control_rate + abs_effect
                
                # Apply 5mer normalization to treated rate if available
                if mean_expected_rates is not None:
                    key = f'{cat}_treated'
                    if key in mean_expected_rates and np.isfinite(mean_expected_rates[key]):
                        # For absolute model with 5mer: multiply the absolute effect by the 5mer ratio
                        # This accounts for different 5mer compositions in treated vs control
                        pass  # The abs_effect is already in the right units
                
                # CI for identity link: treated_rate ± 1.96 × SE
                se_treatment = np.sqrt(var_treatment) if np.isfinite(var_treatment) and var_treatment > 0 else np.nan
                if np.isfinite(se_treatment):
                    ci_low = treated_rate - 1.96 * se_treatment
                    ci_high = treated_rate + 1.96 * se_treatment
                else:
                    ci_low = np.nan
                    ci_high = np.nan
                
                rates[f'{cat}_treated'] = treated_rate
                cis[f'{cat}_treated'] = (ci_low, ci_high)
                
                print(f"  {cat}: control={control_rate:.2e}, treated={treated_rate:.2e} (abs_effect={abs_effect:.2e})")
            
        else:
            # CONTROL-BASELINE MODEL (log link): treated_rate = control_rate × exp(β_treatment)
            result_interaction = control_baseline_result['result_interaction']
            beta_treatment = result_interaction.params.get('treatment', np.nan)
            beta_treatment_syn = result_interaction.params.get('treatment_x_cat_synonymous', 0.0)
            beta_treatment_nonsyn = result_interaction.params.get('treatment_x_cat_non_synonymous', 0.0)
            
            try:
                vcov_interaction = result_interaction.cov_params()
            except:
                vcov_interaction = None
            
            for cat in categories:
                control_rate = control_rates[cat]
                
                if cat == 'intergenic':
                    treatment_effect = beta_treatment
                    if vcov_interaction is not None:
                        var_treatment = vcov_interaction.loc['treatment', 'treatment']
                    else:
                        var_treatment = np.nan
                elif cat == 'synonymous':
                    treatment_effect = beta_treatment + beta_treatment_syn
                    if vcov_interaction is not None:
                        var_treatment = (vcov_interaction.loc['treatment', 'treatment'] +
                                       vcov_interaction.loc['treatment_x_cat_synonymous', 'treatment_x_cat_synonymous'] +
                                       2 * vcov_interaction.loc['treatment', 'treatment_x_cat_synonymous'])
                    else:
                        var_treatment = np.nan
                elif cat == 'non_synonymous':
                    treatment_effect = beta_treatment + beta_treatment_nonsyn
                    if vcov_interaction is not None:
                        var_treatment = (vcov_interaction.loc['treatment', 'treatment'] +
                                       vcov_interaction.loc['treatment_x_cat_non_synonymous', 'treatment_x_cat_non_synonymous'] +
                                       2 * vcov_interaction.loc['treatment', 'treatment_x_cat_non_synonymous'])
                    else:
                        var_treatment = np.nan
                
                treated_rate = control_rate * np.exp(treatment_effect)
                
                # Approximate CI (ignoring covariance between control and treatment estimates)
                se_treatment = np.sqrt(var_treatment) if np.isfinite(var_treatment) and var_treatment > 0 else np.nan
                if np.isfinite(se_treatment):
                    ci_low = treated_rate * np.exp(-1.96 * se_treatment)
                    ci_high = treated_rate * np.exp(1.96 * se_treatment)
                else:
                    ci_low = np.nan
                    ci_high = np.nan
                
                rates[f'{cat}_treated'] = treated_rate
                cis[f'{cat}_treated'] = (ci_low, ci_high)
        
        return rates, cis
    
    # Standard approach: extract rates from interaction model
    beta0 = result.params.get('const', np.nan)
    beta_treatment = result.params.get('treatment', 0.0) if use_treatment_covariate else 0.0
    beta_syn = result.params.get('cat_synonymous', 0.0)
    beta_nonsyn = result.params.get('cat_non_synonymous', 0.0)
    beta_treatment_syn = result.params.get('treatment_x_cat_synonymous', 0.0) if use_treatment_covariate else 0.0
    beta_treatment_nonsyn = result.params.get('treatment_x_cat_non_synonymous', 0.0) if use_treatment_covariate else 0.0
    
    try:
        vcov = result.cov_params()
    except:
        vcov = None
    
    for cat in categories:
        for is_control in [True, False]:
            linear_pred = beta0
            
            if cat == 'synonymous':
                linear_pred += beta_syn
            elif cat == 'non_synonymous':
                linear_pred += beta_nonsyn
            
            if not is_control and use_treatment_covariate:
                linear_pred += beta_treatment
                if cat == 'synonymous':
                    linear_pred += beta_treatment_syn
                elif cat == 'non_synonymous':
                    linear_pred += beta_treatment_nonsyn
            
            rate = np.exp(linear_pred) if np.isfinite(linear_pred) else np.nan
            
            # Apply 5mer normalization scaling if available
            # This converts from relative rate (exp(beta)) to absolute per-base rate
            treatment_label = 'control' if is_control else 'treated'
            key = f"{cat}_{treatment_label}"
            if mean_expected_rates is not None and key in mean_expected_rates:
                expected_rate = mean_expected_rates[key]
                if np.isfinite(expected_rate) and expected_rate > 0:
                    rate = rate * expected_rate
            
            # Calculate variance using delta method
            var_linear = _calc_var_linear(cat, is_control, vcov, use_treatment_covariate)
            se_linear = np.sqrt(var_linear) if np.isfinite(var_linear) and var_linear > 0 else np.nan
            
            if np.isfinite(se_linear) and se_linear > 0:
                ci_low = rate * np.exp(-1.96 * se_linear)
                ci_high = rate * np.exp(1.96 * se_linear)
            else:
                ci_low = np.nan
                ci_high = np.nan
            
            rates[key] = rate
            cis[key] = (ci_low, ci_high)
    
    return rates, cis


def _calc_var_linear(cat, is_control, vcov, use_treatment_covariate):
    """Calculate variance of linear predictor using delta method."""
    if vcov is None:
        return np.nan
    
    var = 0.0
    
    if 'const' in vcov.index:
        var += vcov.loc['const', 'const']
    
    if not is_control and use_treatment_covariate and 'treatment' in vcov.index:
        var += vcov.loc['treatment', 'treatment']
        if 'const' in vcov.index:
            var += 2 * vcov.loc['const', 'treatment']
    
    if cat == 'synonymous' and 'cat_synonymous' in vcov.index:
        var += vcov.loc['cat_synonymous', 'cat_synonymous']
        if 'const' in vcov.index:
            var += 2 * vcov.loc['const', 'cat_synonymous']
        if not is_control and use_treatment_covariate:
            if 'treatment' in vcov.index:
                var += 2 * vcov.loc['treatment', 'cat_synonymous']
            if 'treatment_x_cat_synonymous' in vcov.index:
                var += vcov.loc['treatment_x_cat_synonymous', 'treatment_x_cat_synonymous']
                if 'const' in vcov.index:
                    var += 2 * vcov.loc['const', 'treatment_x_cat_synonymous']
                if 'treatment' in vcov.index:
                    var += 2 * vcov.loc['treatment', 'treatment_x_cat_synonymous']
                var += 2 * vcov.loc['cat_synonymous', 'treatment_x_cat_synonymous']
    
    elif cat == 'non_synonymous' and 'cat_non_synonymous' in vcov.index:
        var += vcov.loc['cat_non_synonymous', 'cat_non_synonymous']
        if 'const' in vcov.index:
            var += 2 * vcov.loc['const', 'cat_non_synonymous']
        if not is_control and use_treatment_covariate:
            if 'treatment' in vcov.index:
                var += 2 * vcov.loc['treatment', 'cat_non_synonymous']
            if 'treatment_x_cat_non_synonymous' in vcov.index:
                var += vcov.loc['treatment_x_cat_non_synonymous', 'treatment_x_cat_non_synonymous']
                if 'const' in vcov.index:
                    var += 2 * vcov.loc['const', 'treatment_x_cat_non_synonymous']
                if 'treatment' in vcov.index:
                    var += 2 * vcov.loc['treatment', 'treatment_x_cat_non_synonymous']
                var += 2 * vcov.loc['cat_non_synonymous', 'treatment_x_cat_non_synonymous']
    
    return var if np.isfinite(var) and var > 0 else np.nan


def _run_statistical_tests(result, df, rates, cis, use_treatment_covariate, control_baseline_result, absolute_effect_result=None):
    """
    Run all statistical tests.
    
    Returns:
        list of test result dicts
    """
    test_results = []
    
    # Add absolute effect model test if available (PRIMARY TEST)
    if absolute_effect_result is not None:
        test_results.append({
            'test': 'Absolute effect model: Treatment adds constant mutations/base across categories?',
            'coefficient': 'Likelihood ratio test (simple vs interaction model on excess counts)',
            'estimate': absolute_effect_result['lr_stat'],
            'SE': np.nan,
            'z': np.nan,
            'p_value': absolute_effect_result['lr_pvalue'],
            'rate_ratio': np.nan,
            'rate_ratio_CI_low': np.nan,
            'rate_ratio_CI_high': np.nan,
            'absolute_effect_intergenic': absolute_effect_result['beta_intergenic'],
            'absolute_effect_synonymous': absolute_effect_result['beta_synonymous'],
            'absolute_effect_nonsyn': absolute_effect_result['beta_nonsyn'],
            'interpretation': 'Tests if treatment adds the same ABSOLUTE mutations/base across categories. p > 0.05 means absolute effect is constant (category does NOT matter).'
        })
        print(f"✓ Added absolute effect model test: p={absolute_effect_result['lr_pvalue']:.4e}")
    
    # Add control-baseline test if available
    if control_baseline_result is not None:
        test_results.append({
            'test': 'Control-baseline: Treatment effect constant across categories?',
            'coefficient': 'Likelihood ratio test (simple vs interaction model)',
            'estimate': control_baseline_result['lr_stat'],
            'SE': np.nan,
            'z': np.nan,
            'p_value': control_baseline_result['lr_pvalue'],
            'rate_ratio': np.exp(control_baseline_result['beta_treatment_simple']),
            'rate_ratio_CI_low': np.exp(control_baseline_result['beta_treatment_simple'] - 1.96 * control_baseline_result['beta_treatment_se_simple']),
            'rate_ratio_CI_high': np.exp(control_baseline_result['beta_treatment_simple'] + 1.96 * control_baseline_result['beta_treatment_se_simple']),
            'interpretation': 'Tests if treatment has same PROPORTIONAL effect across categories (p > 0.05 means proportional effect is constant)'
        })
        print(f"✓ Added control-baseline test: p={control_baseline_result['lr_pvalue']:.4e}")
    
    if not use_treatment_covariate:
        return test_results
    
    # Get coefficients
    beta_treatment = result.params.get('treatment', 0.0)
    beta_syn = result.params.get('cat_synonymous', 0.0)
    beta_nonsyn = result.params.get('cat_non_synonymous', 0.0)
    beta_treatment_syn = result.params.get('treatment_x_cat_synonymous', 0.0)
    beta_treatment_nonsyn = result.params.get('treatment_x_cat_non_synonymous', 0.0)
    
    try:
        vcov = result.cov_params()
    except:
        vcov = None
    
    # Test: Absolute treatment effect (coding vs intergenic)
    if 'intergenic_control' in rates and 'intergenic_treated' in rates:
        test_results.extend(_test_absolute_treatment_effect(df, rates, cis))
    
    # Test: Category effects in treated
    if 'cat_synonymous' in result.params.index and 'treatment_x_cat_synonymous' in result.params.index:
        test_results.extend(_test_category_effects_in_treated(result, beta_syn, beta_nonsyn, beta_treatment_syn, beta_treatment_nonsyn, vcov))
    
    # Test: Treatment effect
    if 'treatment' in result.params.index:
        beta_treatment_se = result.bse.get('treatment', np.nan)
        if np.isfinite(beta_treatment) and np.isfinite(beta_treatment_se) and beta_treatment_se > 0:
            z_treatment = beta_treatment / beta_treatment_se
            p_treatment = 2 * (1 - stats.norm.cdf(abs(z_treatment)))
            test_results.append({
                'test': 'treatment effect (intergenic)',
                'coefficient': 'β_treatment',
                'estimate': beta_treatment,
                'SE': beta_treatment_se,
                'z': z_treatment,
                'p_value': p_treatment,
                'rate_ratio': np.exp(beta_treatment),
                'rate_ratio_CI_low': np.exp(beta_treatment - 1.96 * beta_treatment_se),
                'rate_ratio_CI_high': np.exp(beta_treatment + 1.96 * beta_treatment_se),
            })
    
    # Test: Interaction terms (EMS effect on category differences)
    test_results.extend(_test_interaction_terms(result, beta_treatment_syn, beta_treatment_nonsyn, vcov))
    
    # Test: Category differences within controls
    test_results.extend(_test_category_effects_in_controls(result, beta_syn, beta_nonsyn, vcov))
    
    return test_results


def _test_absolute_treatment_effect(df, rates, cis):
    """Test if absolute treatment effect is the same for coding and intergenic."""
    tests = []
    
    if 'synonymous_control' not in rates or 'non_synonymous_control' not in rates:
        return tests
    
    treated_df = df[df['treatment'] == 1].copy()
    n_syn = treated_df[treated_df['category'] == 'synonymous']['depth'].sum()
    n_nonsyn = treated_df[treated_df['category'] == 'non_synonymous']['depth'].sum()
    n_coding = n_syn + n_nonsyn
    
    if n_coding == 0:
        return tests
    
    try:
        # Calculate weighted average rates
        syn_control = rates['synonymous_control']
        nonsyn_control = rates['non_synonymous_control']
        syn_treated = rates['synonymous_treated']
        nonsyn_treated = rates['non_synonymous_treated']
        
        coding_control = (n_syn * syn_control + n_nonsyn * nonsyn_control) / n_coding
        coding_treated = (n_syn * syn_treated + n_nonsyn * nonsyn_treated) / n_coding
        
        abs_effect_intergenic = rates['intergenic_treated'] - rates['intergenic_control']
        abs_effect_coding = coding_treated - coding_control
        abs_diff = abs_effect_coding - abs_effect_intergenic
        
        # Calculate SE
        ci_ig_c = cis['intergenic_control']
        ci_ig_t = cis['intergenic_treated']
        ci_syn_c = cis['synonymous_control']
        ci_syn_t = cis['synonymous_treated']
        ci_ns_c = cis['non_synonymous_control']
        ci_ns_t = cis['non_synonymous_treated']
        
        se_ig_c = (ci_ig_c[1] - ci_ig_c[0]) / (2 * 1.96)
        se_ig_t = (ci_ig_t[1] - ci_ig_t[0]) / (2 * 1.96)
        se_syn_c = (ci_syn_c[1] - ci_syn_c[0]) / (2 * 1.96)
        se_syn_t = (ci_syn_t[1] - ci_syn_t[0]) / (2 * 1.96)
        se_ns_c = (ci_ns_c[1] - ci_ns_c[0]) / (2 * 1.96)
        se_ns_t = (ci_ns_t[1] - ci_ns_t[0]) / (2 * 1.96)
        
        se_coding_c = np.sqrt((n_syn**2 * se_syn_c**2 + n_nonsyn**2 * se_ns_c**2) / (n_coding**2))
        se_coding_t = np.sqrt((n_syn**2 * se_syn_t**2 + n_nonsyn**2 * se_ns_t**2) / (n_coding**2))
        
        se_abs_ig = np.sqrt(se_ig_t**2 + se_ig_c**2)
        se_abs_coding = np.sqrt(se_coding_t**2 + se_coding_c**2)
        se_abs_diff = np.sqrt(se_abs_coding**2 + se_abs_ig**2)
        
        if se_abs_diff > 0 and np.isfinite(se_abs_diff):
            z = abs_diff / se_abs_diff
            p = 2 * (1 - stats.norm.cdf(abs(z)))
            
            tests.append({
                'test': 'Absolute treatment effect: coding vs intergenic (same absolute increase?)',
                'coefficient': '(treated_coding - control_coding) - (treated_intergenic - control_intergenic)',
                'estimate': abs_diff,
                'SE': se_abs_diff,
                'z': z,
                'p_value': p,
                'absolute_effect_intergenic': abs_effect_intergenic,
                'absolute_effect_coding': abs_effect_coding,
                'interpretation': 'Tests if treatment adds the same ABSOLUTE amount to coding and intergenic rates. p > 0.05 means absolute treatment effect is the same (category does NOT matter for absolute treatment effect).'
            })
    except Exception as e:
        print(f"Warning: Failed absolute treatment effect test: {e}")
    
    return tests


def _test_category_effects_in_treated(result, beta_syn, beta_nonsyn, beta_treatment_syn, beta_treatment_nonsyn, vcov):
    """Test category effects within treated samples."""
    tests = []
    
    # Synonymous vs intergenic in treated
    cat_effect = beta_syn + beta_treatment_syn
    if vcov is not None:
        try:
            var = (vcov.loc['cat_synonymous', 'cat_synonymous'] + 
                  vcov.loc['treatment_x_cat_synonymous', 'treatment_x_cat_synonymous'] +
                  2 * vcov.loc['cat_synonymous', 'treatment_x_cat_synonymous'])
            se = np.sqrt(var) if var > 0 else np.nan
            if np.isfinite(se) and se > 0:
                z = cat_effect / se
                p = 2 * (1 - stats.norm.cdf(abs(z)))
                tests.append({
                    'test': 'synonymous vs intergenic (IN TREATED)',
                    'coefficient': 'β_syn + β_treatment_syn',
                    'estimate': cat_effect,
                    'SE': se,
                    'z': z,
                    'p_value': p,
                    'rate_ratio': np.exp(cat_effect),
                    'rate_ratio_CI_low': np.exp(cat_effect - 1.96 * se),
                    'rate_ratio_CI_high': np.exp(cat_effect + 1.96 * se),
                })
        except:
            pass
    
    # Non-synonymous vs intergenic in treated
    cat_effect = beta_nonsyn + beta_treatment_nonsyn
    if vcov is not None:
        try:
            var = (vcov.loc['cat_non_synonymous', 'cat_non_synonymous'] + 
                  vcov.loc['treatment_x_cat_non_synonymous', 'treatment_x_cat_non_synonymous'] +
                  2 * vcov.loc['cat_non_synonymous', 'treatment_x_cat_non_synonymous'])
            se = np.sqrt(var) if var > 0 else np.nan
            if np.isfinite(se) and se > 0:
                z = cat_effect / se
                p = 2 * (1 - stats.norm.cdf(abs(z)))
                tests.append({
                    'test': 'non_synonymous vs intergenic (IN TREATED)',
                    'coefficient': 'β_nonsyn + β_treatment_nonsyn',
                    'estimate': cat_effect,
                    'SE': se,
                    'z': z,
                    'p_value': p,
                    'rate_ratio': np.exp(cat_effect),
                    'rate_ratio_CI_low': np.exp(cat_effect - 1.96 * se),
                    'rate_ratio_CI_high': np.exp(cat_effect + 1.96 * se),
                })
        except:
            pass
    
    return tests


def _test_interaction_terms(result, beta_treatment_syn, beta_treatment_nonsyn, vcov):
    """Test interaction terms (EMS effect on category differences)."""
    tests = []
    
    # Test β_treatment_syn
    if 'treatment_x_cat_synonymous' in result.params.index:
        se = result.bse.get('treatment_x_cat_synonymous', np.nan)
        if np.isfinite(beta_treatment_syn) and np.isfinite(se) and se > 0:
            z = beta_treatment_syn / se
            p = 2 * (1 - stats.norm.cdf(abs(z)))
            tests.append({
                'test': 'EMS effect on category difference: synonymous vs intergenic (treated - controls)',
                'coefficient': 'β_treatment_syn',
                'estimate': beta_treatment_syn,
                'SE': se,
                'z': z,
                'p_value': p,
                'rate_ratio': np.exp(beta_treatment_syn),
                'rate_ratio_CI_low': np.exp(beta_treatment_syn - 1.96 * se),
                'rate_ratio_CI_high': np.exp(beta_treatment_syn + 1.96 * se),
                'interpretation': 'Tests if EMS causes synonymous vs intergenic difference that differs from controls'
            })
    
    # Test β_treatment_nonsyn
    if 'treatment_x_cat_non_synonymous' in result.params.index:
        se = result.bse.get('treatment_x_cat_non_synonymous', np.nan)
        if np.isfinite(beta_treatment_nonsyn) and np.isfinite(se) and se > 0:
            z = beta_treatment_nonsyn / se
            p = 2 * (1 - stats.norm.cdf(abs(z)))
            tests.append({
                'test': 'EMS effect on category difference: non_synonymous vs intergenic (treated - controls)',
                'coefficient': 'β_treatment_nonsyn',
                'estimate': beta_treatment_nonsyn,
                'SE': se,
                'z': z,
                'p_value': p,
                'rate_ratio': np.exp(beta_treatment_nonsyn),
                'rate_ratio_CI_low': np.exp(beta_treatment_nonsyn - 1.96 * se),
                'rate_ratio_CI_high': np.exp(beta_treatment_nonsyn + 1.96 * se),
                'interpretation': 'Tests if EMS causes non-synonymous vs intergenic difference that differs from controls'
            })
    
    # Test β_treatment_syn - β_treatment_nonsyn
    if vcov is not None and 'treatment_x_cat_synonymous' in result.params.index and 'treatment_x_cat_non_synonymous' in result.params.index:
        diff = beta_treatment_syn - beta_treatment_nonsyn
        try:
            var = (vcov.loc['treatment_x_cat_synonymous', 'treatment_x_cat_synonymous'] +
                   vcov.loc['treatment_x_cat_non_synonymous', 'treatment_x_cat_non_synonymous'] -
                   2 * vcov.loc['treatment_x_cat_synonymous', 'treatment_x_cat_non_synonymous'])
            se = np.sqrt(var) if var > 0 else np.nan
            if np.isfinite(se) and se > 0:
                z = diff / se
                p = 2 * (1 - stats.norm.cdf(abs(z)))
                tests.append({
                    'test': 'EMS effect on category difference: synonymous vs non_synonymous (treated - controls)',
                    'coefficient': 'β_treatment_syn - β_treatment_nonsyn',
                    'estimate': diff,
                    'SE': se,
                    'z': z,
                    'p_value': p,
                    'rate_ratio': np.exp(diff),
                    'rate_ratio_CI_low': np.exp(diff - 1.96 * se),
                    'rate_ratio_CI_high': np.exp(diff + 1.96 * se),
                    'interpretation': 'Tests if EMS causes synonymous vs non-synonymous difference that differs from controls'
                })
        except:
            pass
    
    # Test average interaction: (β_treatment_syn + β_treatment_nonsyn) / 2
    if vcov is not None and 'treatment_x_cat_synonymous' in result.params.index and 'treatment_x_cat_non_synonymous' in result.params.index:
        avg = (beta_treatment_syn + beta_treatment_nonsyn) / 2
        try:
            var = (vcov.loc['treatment_x_cat_synonymous', 'treatment_x_cat_synonymous'] +
                   vcov.loc['treatment_x_cat_non_synonymous', 'treatment_x_cat_non_synonymous'] +
                   2 * vcov.loc['treatment_x_cat_synonymous', 'treatment_x_cat_non_synonymous']) / 4
            se = np.sqrt(var) if var > 0 else np.nan
            if np.isfinite(se) and se > 0:
                z = avg / se
                p = 2 * (1 - stats.norm.cdf(abs(z)))
                tests.append({
                    'test': 'EMS effect on category difference: coding vs intergenic (treated - controls, average)',
                    'coefficient': '(β_treatment_syn + β_treatment_nonsyn) / 2',
                    'estimate': avg,
                    'SE': se,
                    'z': z,
                    'p_value': p,
                    'rate_ratio': np.exp(avg),
                    'rate_ratio_CI_low': np.exp(avg - 1.96 * se),
                    'rate_ratio_CI_high': np.exp(avg + 1.96 * se),
                    'interpretation': 'Tests if EMS causes coding vs intergenic difference (averaged across syn/nonsyn) that differs from controls'
                })
        except:
            pass
    
    return tests


def _test_category_effects_in_controls(result, beta_syn, beta_nonsyn, vcov):
    """Test category effects within control samples."""
    tests = []
    
    # Test β_syn
    if 'cat_synonymous' in result.params.index:
        se = result.bse.get('cat_synonymous', np.nan)
        if np.isfinite(beta_syn) and np.isfinite(se) and se > 0:
            z = beta_syn / se
            p = 2 * (1 - stats.norm.cdf(abs(z)))
            tests.append({
                'test': 'synonymous vs intergenic (within controls)',
                'coefficient': 'β_syn',
                'estimate': beta_syn,
                'SE': se,
                'z': z,
                'p_value': p,
                'rate_ratio': np.exp(beta_syn),
                'rate_ratio_CI_low': np.exp(beta_syn - 1.96 * se),
                'rate_ratio_CI_high': np.exp(beta_syn + 1.96 * se),
            })
    
    # Test β_nonsyn
    if 'cat_non_synonymous' in result.params.index:
        se = result.bse.get('cat_non_synonymous', np.nan)
        if np.isfinite(beta_nonsyn) and np.isfinite(se) and se > 0:
            z = beta_nonsyn / se
            p = 2 * (1 - stats.norm.cdf(abs(z)))
            tests.append({
                'test': 'non_synonymous vs intergenic (within controls)',
                'coefficient': 'β_nonsyn',
                'estimate': beta_nonsyn,
                'SE': se,
                'z': z,
                'p_value': p,
                'rate_ratio': np.exp(beta_nonsyn),
                'rate_ratio_CI_low': np.exp(beta_nonsyn - 1.96 * se),
                'rate_ratio_CI_high': np.exp(beta_nonsyn + 1.96 * se),
            })
    
    # Test β_syn - β_nonsyn
    if vcov is not None and 'cat_synonymous' in result.params.index and 'cat_non_synonymous' in result.params.index:
        diff = beta_syn - beta_nonsyn
        try:
            var = (vcov.loc['cat_synonymous', 'cat_synonymous'] +
                   vcov.loc['cat_non_synonymous', 'cat_non_synonymous'] -
                   2 * vcov.loc['cat_synonymous', 'cat_non_synonymous'])
            se = np.sqrt(var) if var > 0 else np.nan
            if np.isfinite(se) and se > 0:
                z = diff / se
                p = 2 * (1 - stats.norm.cdf(abs(z)))
                tests.append({
                    'test': 'synonymous vs non_synonymous (within controls)',
                    'coefficient': 'β_syn - β_nonsyn',
                    'estimate': diff,
                    'SE': se,
                    'z': z,
                    'p_value': p,
                    'rate_ratio': np.exp(diff),
                    'rate_ratio_CI_low': np.exp(diff - 1.96 * se),
                    'rate_ratio_CI_high': np.exp(diff + 1.96 * se),
                })
        except:
            pass
    
    return tests


def _save_results(summary_df, test_df, result, output_dir, kmer5_normalized=False, absolute_effects=None):
    """Save results to TSV files and text report."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save summary
    summary_path = os.path.join(output_dir, 'category_rates_summary.tsv')
    summary_df.to_csv(summary_path, sep='\t', index=False)
    print(f"Category rates summary saved to {summary_path}")
    
    # Save tests
    if not test_df.empty:
        test_path = os.path.join(output_dir, 'category_statistical_tests.tsv')
        test_df.to_csv(test_path, sep='\t', index=False)
        print(f"Category statistical tests saved to {test_path}")
    
    # Save text report
    report_path = os.path.join(output_dir, 'category_comparison_report.txt')
    with open(report_path, 'w') as f:
        f.write("=== CATEGORY-SPECIFIC MUTATION RATES ===\n\n")
        
        # Add 5mer normalization status
        if kmer5_normalized:
            f.write("✓ 5mer sequence context normalization: ENABLED\n")
            f.write("  Rates are absolute per-base mutation rates (5mer-normalized)\n\n")
        else:
            f.write("5mer sequence context normalization: NOT APPLIED\n")
            f.write("  Rates are relative to depth-normalized baseline\n\n")
        
        # Add absolute effect model info if available
        if absolute_effects is not None:
            f.write("✓ Absolute effect model (identity link): ENABLED\n")
            f.write("  Treatment effects are ADDITIVE: treated_rate = control_rate + β_absolute\n\n")
            
            f.write("=== ABSOLUTE TREATMENT EFFECTS (mutations/base added by EMS) ===\n\n")
            abs_inter = absolute_effects.get('absolute_effect_intergenic', np.nan)
            abs_syn = absolute_effects.get('absolute_effect_synonymous', np.nan)
            abs_nonsyn = absolute_effects.get('absolute_effect_nonsyn', np.nan)
            f.write(f"  Intergenic:     {abs_inter:.6e}\n")
            f.write(f"  Synonymous:     {abs_syn:.6e}\n")
            f.write(f"  Non-synonymous: {abs_nonsyn:.6e}\n\n")
        
        f.write(f"{'Category':<15} {'Treatment':<10} {'Rate':<15} {'CI_low':<15} {'CI_high':<15}")
        if 'absolute_effect' in summary_df.columns:
            f.write(f" {'Abs_Effect':<15}")
        f.write("\n")
        f.write("-" * (85 if 'absolute_effect' in summary_df.columns else 70) + "\n")
        for _, row in summary_df.iterrows():
            f.write(f"{row['category']:<15} {row['treatment']:<10} {row['rate']:<15.6e} {row['CI_low']:<15.6e} {row['CI_high']:<15.6e}")
            if 'absolute_effect' in summary_df.columns and row['treatment'] == 'treated':
                abs_eff = row.get('absolute_effect', np.nan)
                if np.isfinite(abs_eff):
                    f.write(f" {abs_eff:<15.6e}")
                else:
                    f.write(f" {'N/A':<15}")
            elif 'absolute_effect' in summary_df.columns:
                f.write(f" {'-':<15}")
            f.write("\n")
        
        f.write(f"\n=== STATISTICAL TESTS ===\n\n")
        if not test_df.empty:
            f.write(f"{'Test':<35} {'Coefficient':<20} {'Estimate':<12} {'SE':<12} {'z':<10} {'p_value':<12} {'Rate_Ratio':<15}\n")
            f.write("-" * 120 + "\n")
            for _, row in test_df.iterrows():
                sig = "***" if row['p_value'] < 0.001 else "**" if row['p_value'] < 0.01 else "*" if row['p_value'] < 0.05 else ""
                f.write(f"{row['test']:<35} {row['coefficient']:<20} {row['estimate']:<12.6f} {row['SE']:<12.6f} {row['z']:<10.3f} {row['p_value']:<12.6e} {row['rate_ratio']:<15.6f} {sig}\n")
        
        f.write(f"\nModel fit statistics:\n")
        f.write(f"  AIC: {result.aic:.2f}\n")
        f.write(f"  Log-likelihood: {result.llf:.2f}\n")
    
    print(f"Category rates report saved to {report_path}")


def _compute_5mer_offsets(df: pd.DataFrame, kmer5_model_path: str, genome_fasta: str):
    """
    Compute 5mer-based expected rate offsets for each site.
    
    Returns log(expected_rate_5mer) for each site, which will be added to the GLM offset.
    This normalizes for sequence context bias.
    
    Args:
        df: DataFrame with columns: chrom, pos, depth, treatment
        kmer5_model_path: Path to fitted 5mer model pickle file
        genome_fasta: Path to genome FASTA file
    
    Returns:
        Series of log(expected_rate_5mer) values, or None if failed
    """
    try:
        # Load 5mer model
        model_5mer = load_5mer_model(kmer5_model_path)
        seqs = load_genome_sequences(genome_fasta)
        
        # Extract 5mer contexts
        print("  Extracting 5mer contexts...")
        df['chrom_str'] = df['chrom'].astype(str)
        df['pos_int'] = pd.to_numeric(df['pos'], errors='coerce')
        
        kmers = []
        for chrom, pos in zip(df['chrom_str'].values, df['pos_int'].values):
            if pd.isna(pos):
                kmers.append(None)
                continue
            kmer5 = extract_kmer(seqs, chrom, int(pos), 5)
            if kmer5 is None:
                kmers.append(None)
                continue
            kmer5_canonical = canonical_kmer(kmer5)
            kmers.append(kmer5_canonical if (kmer5_canonical and len(kmer5_canonical) == 5) else None)
        
        df['kmer5'] = kmers
        
        # Compute expected rates (vectorized by unique combinations)
        print("  Predicting rates from 5mer model...")
        valid_mask = df['kmer5'].notna() & df['depth'].notna() & (df['depth'] > 0)
        
        log_expected_rates = pd.Series(np.nan, index=df.index)
        
        if valid_mask.sum() > 0:
            valid_df = df[valid_mask].copy()
            unique_combos = valid_df[['kmer5', 'treatment', 'depth']].drop_duplicates()
            
            # Build prediction dict
            pred_dict = {}
            for _, row in unique_combos.iterrows():
                kmer5 = row['kmer5']
                treatment = int(row['treatment'])
                depth = float(row['depth'])
                combo_key = (kmer5, treatment, depth)
                
                if combo_key not in pred_dict:
                    pred_rate = predict_5mer_rate(model_5mer, kmer5, treatment, depth)
                    if np.isfinite(pred_rate) and pred_rate > 0:
                        pred_dict[combo_key] = np.log(pred_rate)
                    else:
                        pred_dict[combo_key] = np.nan
            
            # Map back to all sites
            for idx in valid_df.index:
                kmer5 = df.loc[idx, 'kmer5']
                treatment = int(df.loc[idx, 'treatment'])
                depth = float(df.loc[idx, 'depth'])
                combo_key = (kmer5, treatment, depth)
                if combo_key in pred_dict:
                    log_expected_rates.loc[idx] = pred_dict[combo_key]
        
        return log_expected_rates
        
    except Exception as e:
        print(f"  Error computing 5mer offsets: {e}")
        import traceback
        traceback.print_exc()
        return None


def compare_category_rates(site_df: pd.DataFrame, output_dir: str, method: str = "poisson", 
                            use_treatment_covariate: bool = True, alpha: float = 0.0,
                            use_control_baseline: bool = False, 
                            kmer5_model_path: str = None, genome_fasta: str = None):
    """
    Compare mutation rates across categories (intergenic, synonymous, non-synonymous).
    
    Refactored version with clean separation of concerns.
    
    Args:
        site_df: Site-level DataFrame with mutation counts
        output_dir: Output directory for results
        method: GLM method ('poisson' or 'negative_binomial')
        use_treatment_covariate: Include treatment covariate in model
        alpha: Background rate subtraction
        use_control_baseline: Use control-baseline modeling approach
        kmer5_model_path: Path to fitted 5mer model (optional, for sequence context normalization)
        genome_fasta: Path to genome FASTA file (required if kmer5_model_path provided)
    """
    if site_df is None or site_df.empty:
        print("No site-level data available for category comparison")
        return None
    
    # Prepare data
    df = site_df.copy()
    df = df[df['depth'] > 0].copy()
    df['log_depth'] = np.log(df['depth'].values)
    df['treatment'] = (~df['is_control']).astype(int).values
    
    # Add 5mer normalization if model provided
    log_5mer_offset = None
    mean_expected_rates = None  # Will store mean expected rates per category/treatment
    if kmer5_model_path is not None and genome_fasta is not None:
        print("Computing 5mer-based expected rates for sequence context normalization...")
        log_5mer_offset = _compute_5mer_offsets(df, kmer5_model_path, genome_fasta)
        if log_5mer_offset is not None:
            print(f"✓ 5mer offsets computed for {(~pd.isna(log_5mer_offset)).sum()} sites")
            
            # Calculate mean expected rates per category and treatment for rate recovery
            # This is needed to convert exp(coefficients) back to actual rates
            df['log_5mer_offset'] = log_5mer_offset
            mean_expected_rates = {}
            categories_for_mean = ['intergenic', 'synonymous', 'non_synonymous']
            for cat in categories_for_mean:
                for is_control in [True, False]:
                    treatment_label = 'control' if is_control else 'treated'
                    mask = (df['category'] == cat) & (df['is_control'] == is_control) & df['log_5mer_offset'].notna()
                    if mask.sum() > 0:
                        # Weighted mean of expected rates (weighted by depth)
                        weights = df.loc[mask, 'depth'].values
                        log_offsets = df.loc[mask, 'log_5mer_offset'].values
                        # Mean of log gives geometric mean; use weighted arithmetic mean of exp
                        expected_rates = np.exp(log_offsets)
                        mean_expected_rate = np.average(expected_rates, weights=weights)
                        mean_expected_rates[f'{cat}_{treatment_label}'] = mean_expected_rate
                        print(f"  Mean expected rate for {cat} ({treatment_label}): {mean_expected_rate:.6e}")
                    else:
                        mean_expected_rates[f'{cat}_{treatment_label}'] = np.nan
            print(f"✓ Mean expected rates computed per category/treatment")
        else:
            print("⚠ Warning: 5mer offset computation failed, proceeding without normalization")
    
    # Create category dummy variables (intergenic as reference)
    categories = ['intergenic', 'synonymous', 'non_synonymous']
    for cat in categories[1:]:
        df[f'cat_{cat}'] = (df['category'] == cat).astype(int)
    
    # Try control-baseline model first if requested
    control_baseline_result = None
    absolute_effect_result = None
    result = None
    
    if use_control_baseline and use_treatment_covariate:
        # Try absolute effect model first (answers the constant absolute effect question)
        absolute_effect_result = _fit_absolute_effect_model(df, categories, method, alpha, log_5mer_offset)
        
        # Also fit control-baseline model (for comparison)
        control_baseline_result = _fit_control_baseline_model(df, categories, method, alpha, log_5mer_offset)
        
        if absolute_effect_result is not None:
            # Use absolute effect model's interaction result for rate extraction
            # (it has same structure as control-baseline interaction model)
            result = absolute_effect_result['result_interaction']
            df = absolute_effect_result['df_modified']
            print("✓ Absolute effect model succeeded")
        elif control_baseline_result is not None:
            result = control_baseline_result['result_interaction']
            df = control_baseline_result['df_modified']
            print("✓ Control-baseline model succeeded")
        else:
            print("Both absolute effect and control-baseline models failed, falling back to standard model")
    
    # Fit standard model if control-baseline not used or failed
    if result is None:
        result = _fit_standard_model(df, categories, method, alpha, use_treatment_covariate, log_5mer_offset)
        if result is None:
            print("⚠ Warning: Category comparison failed")
            return None
    
    # Calculate rates and CIs
    # Use absolute_effect_result if available, otherwise control_baseline_result
    baseline_result = absolute_effect_result if absolute_effect_result is not None else control_baseline_result
    rates, cis = _calculate_rates_and_cis(result, categories, use_treatment_covariate, baseline_result, mean_expected_rates)
    
    # Run statistical tests
    test_results = _run_statistical_tests(result, df, rates, cis, use_treatment_covariate, 
                                          control_baseline_result, absolute_effect_result)
    
    # Create summary DataFrames
    summary_rows = []
    for cat in categories:
        for treatment in ['control', 'treated']:
            key = f"{cat}_{treatment}"
            if key in rates:
                summary_rows.append({
                    'category': cat,
                    'treatment': treatment,
                    'rate': rates[key],
                    'CI_low': cis[key][0],
                    'CI_high': cis[key][1]
                })
    
    summary_df = pd.DataFrame(summary_rows)
    test_df = pd.DataFrame(test_results)
    
    # Add 5mer normalization flag to summary
    summary_df['5mer_normalized'] = mean_expected_rates is not None
    
    # Add absolute effect values to summary if available
    if 'absolute_effect_intergenic' in rates:
        summary_df['absolute_effect'] = np.nan
        for idx, row in summary_df.iterrows():
            if row['treatment'] == 'treated':
                cat = row['category']
                if cat == 'intergenic':
                    summary_df.loc[idx, 'absolute_effect'] = rates.get('absolute_effect_intergenic', np.nan)
                elif cat == 'synonymous':
                    summary_df.loc[idx, 'absolute_effect'] = rates.get('absolute_effect_synonymous', np.nan)
                elif cat == 'non_synonymous':
                    summary_df.loc[idx, 'absolute_effect'] = rates.get('absolute_effect_nonsyn', np.nan)
    
    # Save results
    _save_results(summary_df, test_df, result, output_dir, 
                  kmer5_normalized=(mean_expected_rates is not None),
                  absolute_effects=rates if 'absolute_effect_intergenic' in rates else None)
    
    return summary_df, result


def compare_treatment_day_rates(site_df: pd.DataFrame, output_dir: str, method: str = "negative_binomial", 
                                 alpha: float = 0.0):
    """Compare mutation rates across treatment days (3d, 7d, no_label) in treated samples.
    
    Fits GLMs with treatment day as covariate, extracts rates for each treatment day,
    and creates comparison plots and reports.
    
    Args:
        site_df: DataFrame with site-level data including 'treatment_day' column
        output_dir: Output directory for results
        method: GLM method ('poisson' or 'negative_binomial')
        alpha: Background rate to subtract (if > 0)
    
    Returns:
        Tuple of (summary_df, result) or None if analysis fails
    """
    if site_df is None or site_df.empty:
        print("No site-level data available for treatment day comparison")
        return None
    
    # Include both treated and control samples
    df = site_df.copy()
    df = df[df['depth'] > 0].copy()
    
    if df.empty:
        print("No samples found for treatment day comparison")
        return None
    
    # Check if treatment_day column exists
    if 'treatment_day' not in df.columns:
        print("Warning: 'treatment_day' column not found. Extracting from sample names...")
        df['treatment_day'] = df['sample'].apply(extract_treatment_day)
    
    # Filter to valid treatment days (3d, 7d, no_label) and controls
    valid_days = ['3d', '7d', 'no_label', 'control']
    df = df[df['treatment_day'].isin(valid_days)].copy()
    
    if df.empty:
        print("No valid treatment day samples found (need 3d, 7d, no_label, or control)")
        return None
    
    print(f"Treatment day distribution:")
    print(df['treatment_day'].value_counts())
    
    # Prepare data
    df['log_depth'] = np.log(df['depth'].values)
    
    # Create treatment day dummy variables (control as reference for comparison with treated days)
    # We'll fit separate models: one for treated samples only, and include controls in plots
    treatment_days = ['control', 'no_label', '3d', '7d']
    
    # For GLM: fit model on treated samples only (controls handled separately)
    df_treated = df[df['treatment_day'].isin(['no_label', '3d', '7d'])].copy()
    
    if df_treated.empty:
        print("No treated samples found for GLM fitting")
        return None
    
    # Create treatment day dummy variables for treated samples (no_label as reference)
    treated_days = ['no_label', '3d', '7d']
    for day in treated_days[1:]:  # Skip no_label (reference)
        df_treated[f'day_{day}'] = (df_treated['treatment_day'] == day).astype(int)
    
    # Prepare design matrix for treated samples
    X_cols = [f'day_{day}' for day in treated_days[1:]]
    X = df_treated[X_cols]
    design = sm.add_constant(X, has_constant='add')
    y = df_treated['ems_count'].values
    offset = df_treated['log_depth'].values
    
    # Apply alpha subtraction if provided
    if alpha and alpha > 0:
        y = np.maximum(0.0, y - alpha * df_treated['depth'].values)
    
    # Calculate control rates separately (fit intercept-only GLM on control samples)
    rates = {}
    cis = {}
    
    df_control = df[df['treatment_day'] == 'control'].copy()
    if not df_control.empty:
        try:
            df_control['log_depth'] = np.log(df_control['depth'].values)
            y_control = df_control['ems_count'].values
            if alpha and alpha > 0:
                y_control = np.maximum(0.0, y_control - alpha * df_control['depth'].values)
            offset_control = df_control['log_depth'].values
            
            design_control = sm.add_constant(pd.DataFrame(index=df_control.index), has_constant='add')
            
            if method == "poisson":
                fam = sm.families.Poisson()
                model_control = sm.GLM(y_control, design_control, family=fam, offset=offset_control)
                result_control = model_control.fit()
            else:
                from statsmodels.discrete.discrete_model import NegativeBinomial as NBDiscrete
                exposure_control = np.exp(offset_control)
                pois_model = sm.GLM(y_control, design_control, family=sm.families.Poisson(), offset=offset_control)
                pois_result = pois_model.fit()
                start_params = np.r_[pois_result.params.values, 0.1]
                nb_model = NBDiscrete(y_control, design_control, exposure=exposure_control)
                result_control = nb_model.fit(start_params=start_params, disp=False)
            
            beta0_control = result_control.params.get('const', np.nan)
            rate_control = np.exp(beta0_control) if np.isfinite(beta0_control) else np.nan
            
            # Calculate CI for control
            try:
                ci_beta0_control = result_control.conf_int().loc['const'].values
                ci_low_control = np.exp(ci_beta0_control[0])
                ci_high_control = np.exp(ci_beta0_control[1])
            except Exception:
                ci_low_control, ci_high_control = np.nan, np.nan
            
            rates['control'] = rate_control
            cis['control'] = (ci_low_control, ci_high_control)
        except Exception as e:
            print(f"Warning: Failed to calculate control rates: {e}")
            rates['control'] = np.nan
            cis['control'] = (np.nan, np.nan)
    else:
        rates['control'] = np.nan
        cis['control'] = (np.nan, np.nan)
    
    # Fit GLM for treated samples
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            if method == "poisson":
                fam = sm.families.Poisson()
                model = sm.GLM(y, design, family=fam, offset=offset)
                result = model.fit()
            else:
                # For negative binomial, estimate alpha from all data
                from statsmodels.discrete.discrete_model import NegativeBinomial as NBDiscrete
                exposure = np.exp(offset)
                # Warm start with Poisson
                pois_model = sm.GLM(y, design, family=sm.families.Poisson(), offset=offset)
                pois_result = pois_model.fit()
                start_params = np.r_[pois_result.params.values, 0.1]  # Add dispersion parameter
                
                nb_model = NBDiscrete(y, design, exposure=exposure)
                result = nb_model.fit(start_params=start_params, disp=False)
            
            # Extract coefficients
            beta0 = result.params.get('const', np.nan)
            beta_3d = result.params.get('day_3d', 0.0)
            beta_7d = result.params.get('day_7d', 0.0)
            
            # Get variance-covariance matrix for delta method
            try:
                vcov = result.cov_params()
            except Exception:
                vcov = None
            
            # Calculate rates and CIs for treated treatment days
            treated_days = ['no_label', '3d', '7d']
            for day in treated_days:
                # Calculate linear predictor
                linear_pred = beta0
                if day == '3d' and np.isfinite(beta_3d):
                    linear_pred += beta_3d
                elif day == '7d' and np.isfinite(beta_7d):
                    linear_pred += beta_7d
                
                # Calculate rate: exp(linear_pred)
                rate = np.exp(linear_pred) if np.isfinite(linear_pred) else np.nan
                
                # Calculate variance of linear predictor
                var_linear = 0.0
                if vcov is not None:
                    if 'const' in vcov.index:
                        var_linear += vcov.loc['const', 'const']
                    if day == '3d' and 'day_3d' in vcov.index:
                        var_linear += vcov.loc['day_3d', 'day_3d']
                        if 'const' in vcov.index:
                            var_linear += 2 * vcov.loc['const', 'day_3d']
                    elif day == '7d' and 'day_7d' in vcov.index:
                        var_linear += vcov.loc['day_7d', 'day_7d']
                        if 'const' in vcov.index:
                            var_linear += 2 * vcov.loc['const', 'day_7d']
                
                se_linear = np.sqrt(var_linear) if var_linear > 0 else np.nan
                
                # Delta method: 95% CI for exp(linear_pred)
                z = 1.96
                if np.isfinite(se_linear) and se_linear > 0 and np.isfinite(rate):
                    ci_low = rate * np.exp(-z * se_linear)
                    ci_high = rate * np.exp(z * se_linear)
                else:
                    ci_low = np.nan
                    ci_high = np.nan
                
                rates[day] = rate
                cis[day] = (ci_low, ci_high)
            
            # Statistical tests
            test_results = []
            
            # Test: 3d vs no_label
            if 'day_3d' in result.params.index:
                beta_3d_se = result.bse.get('day_3d', np.nan)
                if np.isfinite(beta_3d) and np.isfinite(beta_3d_se) and beta_3d_se > 0:
                    z_3d = beta_3d / beta_3d_se
                    p_3d = 2 * (1 - stats.norm.cdf(abs(z_3d)))
                    test_results.append({
                        'test': '3d vs no_label',
                        'coefficient': 'β_3d',
                        'estimate': beta_3d,
                        'SE': beta_3d_se,
                        'z': z_3d,
                        'p_value': p_3d,
                        'rate_ratio': np.exp(beta_3d),
                        'rate_ratio_CI_low': np.exp(beta_3d - 1.96 * beta_3d_se),
                        'rate_ratio_CI_high': np.exp(beta_3d + 1.96 * beta_3d_se),
                    })
            
            # Test: 7d vs no_label
            if 'day_7d' in result.params.index:
                beta_7d_se = result.bse.get('day_7d', np.nan)
                if np.isfinite(beta_7d) and np.isfinite(beta_7d_se) and beta_7d_se > 0:
                    z_7d = beta_7d / beta_7d_se
                    p_7d = 2 * (1 - stats.norm.cdf(abs(z_7d)))
                    test_results.append({
                        'test': '7d vs no_label',
                        'coefficient': 'β_7d',
                        'estimate': beta_7d,
                        'SE': beta_7d_se,
                        'z': z_7d,
                        'p_value': p_7d,
                        'rate_ratio': np.exp(beta_7d),
                        'rate_ratio_CI_low': np.exp(beta_7d - 1.96 * beta_7d_se),
                        'rate_ratio_CI_high': np.exp(beta_7d + 1.96 * beta_7d_se),
                    })
            
            # Test: 3d vs 7d
            if ('day_3d' in result.params.index and 'day_7d' in result.params.index and vcov is not None):
                diff = beta_3d - beta_7d
                try:
                    var_diff = (vcov.loc['day_3d', 'day_3d'] + 
                               vcov.loc['day_7d', 'day_7d'] -
                               2 * vcov.loc['day_3d', 'day_7d'])
                    se_diff = np.sqrt(var_diff) if var_diff > 0 else np.nan
                    if np.isfinite(se_diff) and se_diff > 0:
                        z_diff = diff / se_diff
                        p_diff = 2 * (1 - stats.norm.cdf(abs(z_diff)))
                        test_results.append({
                            'test': '3d vs 7d',
                            'coefficient': 'β_3d - β_7d',
                            'estimate': diff,
                            'SE': se_diff,
                            'z': z_diff,
                            'p_value': p_diff,
                            'rate_ratio': np.exp(diff),
                            'rate_ratio_CI_low': np.exp(diff - 1.96 * se_diff),
                            'rate_ratio_CI_high': np.exp(diff + 1.96 * se_diff),
                        })
                except Exception:
                    pass
            
            # Add tests comparing control to each treatment group
            # Since control is fit separately, we compare log rates using delta method
            if np.isfinite(rates.get('control', np.nan)) and rates['control'] > 0:
                log_rate_control = np.log(rates['control'])
                
                # Get SE for control from CI
                ci_control = cis.get('control', (np.nan, np.nan))
                if np.isfinite(ci_control[0]) and np.isfinite(ci_control[1]) and ci_control[0] > 0:
                    # Approximate SE from CI: SE ≈ (CI_high - CI_low) / (2 * 1.96)
                    se_log_control = (np.log(ci_control[1]) - np.log(ci_control[0])) / (2 * 1.96)
                else:
                    se_log_control = np.nan
                
                # Compare control to each treatment day
                for day in ['no_label', '3d', '7d']:
                    if day in rates and np.isfinite(rates[day]) and rates[day] > 0:
                        log_rate_day = np.log(rates[day])
                        
                        # Get SE for treatment day
                        if day == 'no_label':
                            # no_label is the intercept, get SE from model
                            se_log_day = result.bse.get('const', np.nan) if 'const' in result.bse.index else np.nan
                        elif day == '3d':
                            # 3d = intercept + beta_3d
                            if 'const' in result.bse.index and 'day_3d' in result.bse.index and vcov is not None:
                                var_log_3d = (vcov.loc['const', 'const'] + 
                                            vcov.loc['day_3d', 'day_3d'] +
                                            2 * vcov.loc['const', 'day_3d'])
                                se_log_day = np.sqrt(var_log_3d) if var_log_3d > 0 else np.nan
                            else:
                                se_log_day = np.nan
                        elif day == '7d':
                            # 7d = intercept + beta_7d
                            if 'const' in result.bse.index and 'day_7d' in result.bse.index and vcov is not None:
                                var_log_7d = (vcov.loc['const', 'const'] + 
                                            vcov.loc['day_7d', 'day_7d'] +
                                            2 * vcov.loc['const', 'day_7d'])
                                se_log_day = np.sqrt(var_log_7d) if var_log_7d > 0 else np.nan
                            else:
                                se_log_day = np.nan
                        
                        # If we can't get SE from model, try from CI
                        if not (np.isfinite(se_log_day) and se_log_day > 0):
                            ci_day = cis.get(day, (np.nan, np.nan))
                            if np.isfinite(ci_day[0]) and np.isfinite(ci_day[1]) and ci_day[0] > 0:
                                se_log_day = (np.log(ci_day[1]) - np.log(ci_day[0])) / (2 * 1.96)
                        
                        # Two-sample z-test for log rates
                        if np.isfinite(se_log_control) and se_log_control > 0 and np.isfinite(se_log_day) and se_log_day > 0:
                            diff_log = log_rate_day - log_rate_control
                            se_diff_log = np.sqrt(se_log_control**2 + se_log_day**2)
                            if se_diff_log > 0:
                                z_diff = diff_log / se_diff_log
                                p_diff = 2 * (1 - stats.norm.cdf(abs(z_diff)))
                                rate_ratio = np.exp(diff_log)
                                ci_low_ratio = np.exp(diff_log - 1.96 * se_diff_log)
                                ci_high_ratio = np.exp(diff_log + 1.96 * se_diff_log)
                                
                                test_results.append({
                                    'test': f'control vs {day}',
                                    'coefficient': f'log({day}) - log(control)',
                                    'estimate': diff_log,
                                    'SE': se_diff_log,
                                    'z': z_diff,
                                    'p_value': p_diff,
                                    'rate_ratio': rate_ratio,
                                    'rate_ratio_CI_low': ci_low_ratio,
                                    'rate_ratio_CI_high': ci_high_ratio,
                                })
            
            test_df = pd.DataFrame(test_results)
            
            # Create summary DataFrame
            summary_rows = []
            for day in treatment_days:
                summary_rows.append({
                    'treatment_day': day,
                    'rate': rates[day],
                    'CI_low': cis[day][0],
                    'CI_high': cis[day][1],
                })
            
            summary_df = pd.DataFrame(summary_rows)
            
            # Save summary and test results
            os.makedirs(output_dir, exist_ok=True)
            summary_path = os.path.join(output_dir, 'treatment_day_rates_summary.tsv')
            summary_df.to_csv(summary_path, sep='\t', index=False)
            print(f"Treatment day rates summary saved to {summary_path}")
            
            if not test_df.empty:
                test_path = os.path.join(output_dir, 'treatment_day_statistical_tests.tsv')
                test_df.to_csv(test_path, sep='\t', index=False)
                print(f"Treatment day statistical tests saved to {test_path}")
            
            # Calculate per-sample rates for grouped bar plot
            print("Calculating per-sample rates for grouped bar plot...")
            sample_rates = {}
            samples_by_day = {}
            
            for day in treatment_days:
                samples_by_day[day] = sorted(df[df['treatment_day'] == day]['sample'].unique())
                sample_rates[day] = {}
                
                for sample_name in samples_by_day[day]:
                    sample_df = df[(df['sample'] == sample_name) & (df['treatment_day'] == day)].copy()
                    if sample_df.empty or sample_df['depth'].sum() == 0:
                        continue
                    
                    # Fit intercept-only GLM for this sample
                    sample_df['log_depth'] = np.log(sample_df['depth'].values)
                    y_sample = sample_df['ems_count'].values
                    if alpha and alpha > 0:
                        y_sample = np.maximum(0.0, y_sample - alpha * sample_df['depth'].values)
                    offset_sample = sample_df['log_depth'].values
                    
                    try:
                        design_sample = sm.add_constant(pd.DataFrame(index=sample_df.index), has_constant='add')
                        if method == "poisson":
                            fam = sm.families.Poisson()
                            model_sample = sm.GLM(y_sample, design_sample, family=fam, offset=offset_sample)
                            result_sample = model_sample.fit()
                        else:
                            from statsmodels.discrete.discrete_model import NegativeBinomial as NBDiscrete
                            exposure_sample = np.exp(offset_sample)
                            pois_model = sm.GLM(y_sample, design_sample, family=sm.families.Poisson(), offset=offset_sample)
                            pois_result = pois_model.fit()
                            start_params = np.r_[pois_result.params.values, 0.1]
                            nb_model = NBDiscrete(y_sample, design_sample, exposure=exposure_sample)
                            result_sample = nb_model.fit(start_params=start_params, disp=False)
                        
                        beta0_sample = result_sample.params.get('const', np.nan)
                        rate_sample = np.exp(beta0_sample) if np.isfinite(beta0_sample) else np.nan
                        sample_rates[day][sample_name] = rate_sample
                    except Exception:
                        sample_rates[day][sample_name] = np.nan
            
            # Save per-sample rates
            sample_rates_rows = []
            for day in treatment_days:
                for sample_name in samples_by_day.get(day, []):
                    rate = sample_rates.get(day, {}).get(sample_name, np.nan)
                    sample_rates_rows.append({
                        'sample': sample_name,
                        'treatment_day': day,
                        'rate': rate
                    })
            if sample_rates_rows:
                sample_rates_df = pd.DataFrame(sample_rates_rows)
                sample_rates_path = os.path.join(output_dir, 'treatment_day_per_sample_rates.tsv')
                sample_rates_df.to_csv(sample_rates_path, sep='\t', index=False)
                print(f"Per-sample rates saved to {sample_rates_path}")
            
            # Create single plot with individual sample rates and aggregated rates
            fig, ax = plt.subplots(figsize=(12, 8))
            days_plot = ['control', 'no_label', '3d', '7d']
            
            # Color-blind accessible palette (Okabe-Ito inspired)
            # Control: grey, no_label: orange, 3d: sky blue, 7d: vermillion
            colors_day = {
                'control': '#808080',      # Grey
                'no_label': '#E69F00',     # Orange
                '3d': '#56B4E9',           # Sky blue
                '7d': '#D55E00'            # Vermillion
            }
            
            # X positions for each day group
            x_positions = np.arange(len(days_plot))
            
            # Calculate jitter for individual samples to avoid overlap
            max_samples_per_day = max([len(samples_by_day.get(day, [])) for day in days_plot], default=1)
            jitter_width = 0.3  # Total width for jittering samples
            
            # Plot individual sample rates as points with jitter
            for day_idx, day in enumerate(days_plot):
                day_samples = samples_by_day.get(day, [])
                if not day_samples:
                    continue
                
                x_base = x_positions[day_idx]
                sample_rates_list = []
                
                for sample_name in day_samples:
                    rate = sample_rates.get(day, {}).get(sample_name, np.nan)
                    if np.isfinite(rate) and rate > 0:
                        sample_rates_list.append(rate)
                
                if sample_rates_list:
                    # Add jitter to x positions
                    n_samples = len(sample_rates_list)
                    if n_samples == 1:
                        x_jittered = [x_base]
                    else:
                        x_jittered = x_base + np.linspace(-jitter_width/2, jitter_width/2, n_samples)
                    
                    # Plot individual samples as points
                    ax.scatter(x_jittered, sample_rates_list, 
                             color=colors_day[day], s=100, alpha=0.6, 
                             edgecolors='black', linewidths=0.5, zorder=3,
                             label=day if day_idx == 0 or day not in [d for d in days_plot[:day_idx]] else '')
            
            # Plot aggregated rates as larger points with error bars
            day_rates = [rates[d] for d in days_plot]
            day_ci_low = [cis[d][0] for d in days_plot]
            day_ci_high = [cis[d][1] for d in days_plot]
            
            day_err_low = [r - cl if np.isfinite(r) and np.isfinite(cl) and cl > 0 else 0 
                          for r, cl in zip(day_rates, day_ci_low)]
            day_err_high = [ch - r if np.isfinite(r) and np.isfinite(ch) and ch > 0 else 0 
                           for r, ch in zip(day_rates, day_ci_high)]
            
            # Plot aggregated rates as larger points
            for day_idx, day in enumerate(days_plot):
                if np.isfinite(day_rates[day_idx]) and day_rates[day_idx] > 0:
                    ax.errorbar(x_positions[day_idx], day_rates[day_idx],
                              yerr=[[day_err_low[day_idx]], [day_err_high[day_idx]]],
                              fmt='o', markersize=12, capsize=5, capthick=2,
                              color=colors_day[day], alpha=0.9, 
                              markeredgecolor='black', markeredgewidth=1.5,
                              elinewidth=2, zorder=5,
                              label=f'{day} (aggregated)' if day_idx == 0 else '')
            
            # Set labels and formatting
            ax.set_xlabel('Treatment Day', fontsize=14, fontweight='bold')
            ax.set_ylabel('Per-base mutation rate', fontsize=14, fontweight='bold')
            ax.set_title('Individual Sample Rates by Treatment Day', fontsize=16, fontweight='bold')
            ax.set_xticks(x_positions)
            ax.set_xticklabels(days_plot, fontsize=12)
            ax.set_yscale('log')
            ax.grid(alpha=0.3, axis='y', zorder=0)
            
            # Find maximum rate for positioning significance bars
            all_rates = [r for r in day_rates if np.isfinite(r) and r > 0]
            all_sample_rates = []
            for day in days_plot:
                for sample_name in samples_by_day.get(day, []):
                    rate = sample_rates.get(day, {}).get(sample_name, np.nan)
                    if np.isfinite(rate) and rate > 0:
                        all_sample_rates.append(rate)
            max_rate = max(all_rates + all_sample_rates) if (all_rates + all_sample_rates) else 1e-6
            
            # Position significance bars above the plot with much better spacing
            sig_bar_base = max_rate * 1.15
            sig_bar_spacing = max_rate * 0.25  # Much larger spacing to avoid overlaps
            
            # Collect all tests to draw
            tests_to_draw = []
            if not test_df.empty:
                day_to_xpos = {day: i for i, day in enumerate(days_plot)}
                
                # Control vs treated comparisons
                for test_name in ['control vs no_label', 'control vs 3d', 'control vs 7d']:
                    test_row = test_df[test_df['test'] == test_name]
                    if not test_row.empty:
                        p = test_row.iloc[0]['p_value']
                        parts = test_name.split(' vs ')
                        x1, x2 = day_to_xpos[parts[0]], day_to_xpos[parts[1]]
                        tests_to_draw.append({'x1': x1, 'x2': x2, 'p': p, 'test': test_name})
                
                # Treated vs treated comparisons
                for test_name in ['3d vs no_label', '7d vs no_label', '3d vs 7d']:
                    test_row = test_df[test_df['test'] == test_name]
                    if not test_row.empty:
                        p = test_row.iloc[0]['p_value']
                        parts = test_name.split(' vs ')
                        x1, x2 = day_to_xpos[parts[0]], day_to_xpos[parts[1]]
                        tests_to_draw.append({'x1': x1, 'x2': x2, 'p': p, 'test': test_name})
            
            # Smart positioning: group bars by their x-range to avoid overlaps
            # Bars that share endpoints need to be at different y-levels
            if tests_to_draw:
                # Sort tests by x1, then x2 to group similar comparisons
                tests_to_draw.sort(key=lambda t: (t['x1'], t['x2']))
                
                # Assign y-positions avoiding overlaps
                # Track which x-ranges are occupied at each y-level
                y_levels = []  # List of sets, each set contains (x1, x2) tuples for bars at that level
                
                for test in tests_to_draw:
                    x1, x2 = test['x1'], test['x2']
                    x_range = (min(x1, x2), max(x1, x2))
                    
                    # Find first y-level where this bar doesn't overlap
                    assigned_level = -1
                    for level_idx, occupied_ranges in enumerate(y_levels):
                        # Check if this bar overlaps with any bar at this level
                        overlaps = False
                        for (ox1, ox2) in occupied_ranges:
                            # Check if ranges overlap
                            if not (x_range[1] < ox1 or x_range[0] > ox2):
                                overlaps = True
                                break
                        if not overlaps:
                            assigned_level = level_idx
                            occupied_ranges.add(x_range)
                            break
                    
                    # If no level found, create new one
                    if assigned_level == -1:
                        y_levels.append(set([x_range]))
                        assigned_level = len(y_levels) - 1
                    
                    test['y_level'] = assigned_level
                
                # Adjust y-axis to accommodate significance bars
                n_levels = len(y_levels)
                max_sig_height = sig_bar_base + (n_levels - 1) * sig_bar_spacing + max_rate * 0.15
                current_ylim = ax.get_ylim()
                ax.set_ylim(top=max(current_ylim[1], max_sig_height))
                
                # Helper function to format p-value for display
                def format_pvalue(p):
                    """Format p-value in readable way."""
                    if p < 0.001:
                        return 'p<0.001'
                    elif p < 0.01:
                        return f'p={p:.3f}'
                    elif p < 0.05:
                        return f'p={p:.3f}'
                    else:
                        return f'p={p:.2f}'
                
                # Draw color-coded p-value boxes (no significance bars)
                if tests_to_draw:
                    from matplotlib.patches import FancyBboxPatch
                    
                    for test in tests_to_draw:
                        # Calculate y position for box (above the bars)
                        y_sig = sig_bar_base + test['y_level'] * sig_bar_spacing
                        
                        # Add color-coded p-value box
                        p = test['p']
                        p_str = format_pvalue(p)
                        test_name = test['test']
                        
                        # Parse test name to get the two groups
                        parts = test_name.split(' vs ')
                        if len(parts) == 2:
                            group1, group2 = parts[0], parts[1]
                            
                            # Get colors for each group
                            color1 = colors_day.get(group1, '#808080')
                            color2 = colors_day.get(group2, '#808080')
                            
                            # Position box centered between x1 and x2, above the data points
                            box_center_x = (test['x1'] + test['x2']) / 2
                            box_y = y_sig  # Use the calculated y position
                            box_width = abs(test['x2'] - test['x1']) * 0.8  # Box width based on bar span
                            if box_width < 0.3:  # Minimum width
                                box_width = 0.3
                            box_height = max_rate * 0.08  # Height proportional to rate scale
                            
                            # Create two-color box: left half color1, right half color2
                            box_left = box_center_x - box_width / 2
                            
                            # Left half
                            left_box = FancyBboxPatch(
                                (box_left, box_y - box_height / 2), box_width / 2, box_height,
                                boxstyle="round,pad=0.02",
                                facecolor=color1,
                                edgecolor='black',
                                linewidth=2,
                                zorder=7
                            )
                            ax.add_patch(left_box)
                            
                            # Right half
                            right_box = FancyBboxPatch(
                                (box_left + box_width / 2, box_y - box_height / 2), box_width / 2, box_height,
                                boxstyle="round,pad=0.02",
                                facecolor=color2,
                                edgecolor='black',
                                linewidth=2,
                                zorder=7
                            )
                            ax.add_patch(right_box)
                            
                            # Add p-value text in the center of the box
                            ax.text(box_center_x, box_y, p_str,
                                   ha='center', va='center',
                                   fontsize=10, fontweight='bold',
                                   color='white' if min(p, 0.05) < 0.05 else 'black',
                                   zorder=8,
                                   bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.3) if p < 0.05 else None)
            
            # Add legend (simplified - just show treatment days)
            handles, labels = ax.get_legend_handles_labels()
            # Remove duplicates while preserving order
            seen = set()
            unique_handles = []
            unique_labels = []
            for h, l in zip(handles, labels):
                if l not in seen:
                    seen.add(l)
                    unique_handles.append(h)
                    unique_labels.append(l)
            if unique_handles:
                ax.legend(unique_handles, unique_labels, fontsize=11, loc='upper left', framealpha=0.9)
            
            plt.tight_layout()
            plot_path = os.path.join(output_dir, 'treatment_day_rates_comparison.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Treatment day rates comparison plot saved to {plot_path}")
            
            # Create report
            report_path = os.path.join(output_dir, 'treatment_day_rates_report.txt')
            with open(report_path, 'w') as f:
                f.write("=== Mutation Rate Comparison by Treatment Day ===\n\n")
                f.write(f"Model: GLM with treatment day as covariate\n")
                f.write(f"Method: {method}\n")
                if alpha > 0:
                    f.write(f"Alpha subtracted: {alpha:.2e}\n")
                f.write(f"\nTotal sites: {len(df):,}\n")
                f.write(f"Treatment day distribution:\n")
                f.write(str(df['treatment_day'].value_counts()) + "\n\n")
                
                f.write("Model coefficients:\n")
                f.write(f"  Intercept (β₀, no_label): {beta0:.6f}\n")
                f.write(f"  3d (β_3d): {beta_3d:.6f}\n")
                f.write(f"  7d (β_7d): {beta_7d:.6f}\n")
                f.write("\n")
                
                f.write("Rates by treatment day:\n")
                f.write(f"{'Treatment Day':<15} {'Rate':<15} {'CI_low':<15} {'CI_high':<15}\n")
                f.write("-" * 60 + "\n")
                for _, row in summary_df.iterrows():
                    f.write(f"{row['treatment_day']:<15} {row['rate']:<15.6e} {row['CI_low']:<15.6e} {row['CI_high']:<15.6e}\n")
                
                f.write(f"\n=== STATISTICAL TESTS ===\n\n")
                if not test_df.empty:
                    f.write(f"{'Test':<20} {'Coefficient':<20} {'Estimate':<12} {'SE':<12} {'z':<10} {'p_value':<12} {'Rate_Ratio':<15}\n")
                    f.write("-" * 100 + "\n")
                    for _, row in test_df.iterrows():
                        sig = "***" if row['p_value'] < 0.001 else "**" if row['p_value'] < 0.01 else "*" if row['p_value'] < 0.05 else ""
                        f.write(f"{row['test']:<20} {row['coefficient']:<20} {row['estimate']:<12.6f} {row['SE']:<12.6f} {row['z']:<10.3f} {row['p_value']:<12.6e} {row['rate_ratio']:<15.6f} {sig}\n")
                    f.write("\nSignificance: *** p<0.001, ** p<0.01, * p<0.05\n\n")
                    
                    f.write("Rate ratios with 95% CI:\n")
                    for _, row in test_df.iterrows():
                        f.write(f"  {row['test']}: {row['rate_ratio']:.4f} (95% CI: {row['rate_ratio_CI_low']:.4f} - {row['rate_ratio_CI_high']:.4f})\n")
                else:
                    f.write("No statistical tests performed.\n")
                
                f.write(f"\nModel fit statistics:\n")
                f.write(f"  AIC: {result.aic:.2f}\n")
                f.write(f"  Log-likelihood: {result.llf:.2f}\n")
            
            print(f"Treatment day rates report saved to {report_path}")
            
            return summary_df, result
            
    except Exception as e:
        print(f"Error in treatment day rate comparison: {e}")
        import traceback
        traceback.print_exc()
        return None


def calculate_gc_content(seq: str) -> float:
    """Calculate GC content of a sequence."""
    if len(seq) == 0:
        return 0.0
    gc_count = seq.count('G') + seq.count('C')
    return gc_count / len(seq)


def canonical_kmer(kmer: str) -> str:
    """Canonicalize kmer: G-centered kmers are reverse-complemented to C-centered."""
    if len(kmer) % 2 == 0:
        return kmer
    center = len(kmer) // 2
    if kmer[center] == 'C':
        return kmer
    if kmer[center] == 'G':
        return str(Seq(kmer).reverse_complement())
    return kmer


def extract_kmer(seqs, chrom: str, pos_1based: int, k: int):
    """Extract kmer centered on position (1-based)."""
    if chrom not in seqs:
        return None
    s = seqs[chrom]
    i0 = int(pos_1based) - 1
    half = (k - 1) // 2
    start = i0 - half
    end = i0 + half + 1
    if start < 0 or end > len(s):
        return None
    kmer = s[start:end].upper()
    if len(kmer) != k or 'N' in kmer:
        return None
    return kmer


def load_5mer_model(model_path: str):
    """Load 5mer model from pickle file."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"5mer model file not found: {model_path}")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model


def predict_5mer_rate(model, kmer5: str, treatment: int, depth: float):
    """Predict mutation rate from 5mer model for a given site.
    
    Args:
        model: Fitted statsmodels GLM result object
        kmer5: Canonicalized 5mer context (C-centered)
        treatment: 0 for control, 1 for treated
        depth: Sequencing depth at site
    
    Returns:
        Predicted mutation rate (per base)
    """
    if kmer5 is None or len(kmer5) != 5:
        return np.nan
    
    try:
        # Get feature column names from model
        # Try exog_names first, fall back to params.index
        if hasattr(model.model, 'exog_names'):
            feature_names = list(model.model.exog_names)
        elif hasattr(model, 'params') and hasattr(model.params, 'index'):
            feature_names = list(model.params.index)
        else:
            return np.nan
        
        # Create one-hot encoded features matching model's design
        # Model expects: const, treatment, and 5mer_XXXXX columns
        X_dict = {}
        for col in feature_names:
            if col == 'const':
                X_dict[col] = 1.0
            elif col == 'treatment':
                X_dict[col] = treatment
            elif col.startswith('5mer_'):
                kmer_name = col.replace('5mer_', '')
                X_dict[col] = 1.0 if kmer_name == kmer5 else 0.0
            else:
                X_dict[col] = 0.0
        
        # Create DataFrame with same columns as model (in correct order)
        X_df = pd.DataFrame([X_dict])
        X_df = X_df.reindex(columns=feature_names, fill_value=0.0)
        
        # Predict with offset = log(depth)
        log_depth = np.log(depth) if depth > 0 else 0.0
        pred = model.predict(X_df, offset=[log_depth])
        
        # Convert predicted count to rate by dividing by depth
        pred_count = float(pred.iloc[0]) if len(pred) > 0 else np.nan
        if np.isfinite(pred_count) and depth > 0:
            return pred_count / depth
        else:
            return np.nan
    except Exception as e:
        # Return NaN if prediction fails
        return np.nan


def create_fixed_windows(
    genome_fasta: str,
    window_size: int = 5000,
    exclusion_mask: set = None
) -> pd.DataFrame:
    """
    Create fixed-size non-overlapping windows across the genome.
    
    Args:
        genome_fasta: Path to genome FASTA file
        window_size: Fixed window size in bp
        exclusion_mask: Set of (chrom, pos) tuples to exclude (not used for window creation, but kept for compatibility)
    
    Returns:
        DataFrame with columns: chrom, start, end, window_size, window_id
    """
    print(f"Creating fixed-size windows (window_size={window_size}bp)...")
    
    # Load genome sequences
    seqs = load_genome_sequences(genome_fasta)
    
    windows = []
    window_id = 0
    
    for chrom in sorted(seqs.keys()):
        seq = seqs[chrom]
        chrom_len = len(seq)
        
        # Create non-overlapping windows
        pos = 0
        while pos < chrom_len:
            window_start = pos
            window_end = min(pos + window_size, chrom_len)
            actual_size = window_end - window_start
            
            windows.append({
                'chrom': chrom,
                'start': window_start + 1,  # 1-based
                'end': window_end,
                'window_size': actual_size,
                'window_id': window_id,
            })
            window_id += 1
            pos = window_end
    
    windows_df = pd.DataFrame(windows)
    
    if len(windows_df) > 0:
        print(f"Created {len(windows_df)} fixed-size windows")
        print(f"Window size: {window_size}bp (actual range: {windows_df['window_size'].min()}-{windows_df['window_size'].max()}bp)")
    else:
        print("Warning: No windows created")
    
    return windows_df


def calculate_window_rates_5mer_normalized(
    windows_df: pd.DataFrame,
    site_df: pd.DataFrame,
    genome_fasta: str,
    model_5mer,
    method: str = "negative_binomial",
    use_treatment_covariate: bool = True
) -> pd.DataFrame:
    """
    Calculate 5mer-normalized mutation rates for each fixed window.
    
    Approach:
    1. For each window, calculate expected rate based on 5mer model (fixes the baseline)
    2. Then fit site-level GLM on top of that baseline using expected rate as offset
    3. This gives site-level rates normalized by 5mer expectations
    
    For each window:
    - Extract 5mer context for each site
    - Get predicted rate from 5mer model for each site
    - Calculate expected rate for window (weighted average by depth)
    - Fit GLM on site-level data with offset = log(expected_rate_from_5mer)
    - This gives rates that are deviations from 5mer-expected baseline
    
    Args:
        windows_df: DataFrame with columns: chrom, start, end, window_id
        site_df: Site-level mutation data (from load_site_level_data_no_context)
        genome_fasta: Path to genome FASTA file (for extracting 5mer contexts)
        model_5mer: Fitted 5mer model (statsmodels GLM result)
        method: GLM method ('poisson' or 'negative_binomial')
        use_treatment_covariate: If True, fit with treatment covariate; else intercept-only
    
    Returns:
        DataFrame with columns: window_id, chrom, start, end, window_size,
        rate_control, rate_treated, expected_rate_control, expected_rate_treated,
        CI_low_control, CI_high_control, CI_low_treated, CI_high_treated,
        n_sites, total_depth, total_mutations
    """
    print(f"Calculating 5mer-normalized mutation rates for {len(windows_df)} windows...")
    print("  Approach: Fix expected rate from 5mer model, then fit site-level GLM on top")
    print("  Pre-computing 5mer contexts and expected rates for all sites...")
    
    # Load genome sequences for 5mer extraction
    seqs = load_genome_sequences(genome_fasta)
    
    # Pre-compute 5mer contexts and expected rates for all sites (much faster than doing it per window)
    print("  Extracting 5mer contexts...")
    site_df = site_df.copy()
    site_df['chrom_str'] = site_df['chrom'].astype(str)
    site_df['pos_int'] = pd.to_numeric(site_df['pos'], errors='coerce')
    
    # Vectorized 5mer extraction (using list comprehension - faster than apply)
    print("  Extracting 5mer contexts...")
    chroms = site_df['chrom_str'].values
    poss = site_df['pos_int'].values
    kmers = []
    for chrom, pos in zip(chroms, poss):
        if pd.isna(pos):
            kmers.append(None)
            continue
        kmer5 = extract_kmer(seqs, chrom, int(pos), 5)
        if kmer5 is None:
            kmers.append(None)
            continue
        kmer5_canonical = canonical_kmer(kmer5)
        kmers.append(kmer5_canonical if (kmer5_canonical and len(kmer5_canonical) == 5) else None)
    
    site_df['kmer5'] = kmers
    
    # Pre-compute expected rates - VECTORIZED approach for speed
    print("  Computing expected rates from 5mer model (vectorized)...")
    
    # Get unique kmers and depths to batch predictions
    valid_mask = site_df['kmer5'].notna() & site_df['depth'].notna() & (site_df['depth'] > 0)
    valid_sites = site_df[valid_mask].copy()
    
    if len(valid_sites) > 0:
        # Group by kmer5 and depth to reduce prediction calls
        # For each unique (kmer5, depth) combination, predict once
        unique_combos = valid_sites[['kmer5', 'depth']].drop_duplicates()
        
        # Batch predict for control
        pred_rates_control_dict = {}
        pred_rates_treated_dict = {}
        
        # Process in batches to avoid memory issues
        batch_size = 10000
        for i in range(0, len(unique_combos), batch_size):
            batch = unique_combos.iloc[i:i+batch_size]
            for _, row in batch.iterrows():
                kmer5 = row['kmer5']
                depth = float(row['depth'])
                combo_key = (kmer5, depth)
                
                if combo_key not in pred_rates_control_dict:
                    pred_rate_control = predict_5mer_rate(model_5mer, kmer5, treatment=0, depth=depth)
                    pred_rate_treated = predict_5mer_rate(model_5mer, kmer5, treatment=1, depth=depth)
                    pred_rates_control_dict[combo_key] = pred_rate_control if np.isfinite(pred_rate_control) else np.nan
                    pred_rates_treated_dict[combo_key] = pred_rate_treated if np.isfinite(pred_rate_treated) else np.nan
        
        # Map predictions back to all sites
        site_df['expected_rate_control'] = np.nan
        site_df['expected_rate_treated'] = np.nan
        
        valid_indices = valid_sites.index
        for idx in valid_indices:
            kmer5 = site_df.loc[idx, 'kmer5']
            depth = float(site_df.loc[idx, 'depth'])
            combo_key = (kmer5, depth)
            if combo_key in pred_rates_control_dict:
                site_df.loc[idx, 'expected_rate_control'] = pred_rates_control_dict[combo_key]
                site_df.loc[idx, 'expected_rate_treated'] = pred_rates_treated_dict[combo_key]
    else:
        site_df['expected_rate_control'] = np.nan
        site_df['expected_rate_treated'] = np.nan
    
    # Filter to sites with valid expected rates
    site_df = site_df[site_df['kmer5'].notna() & 
                      site_df['expected_rate_control'].notna() & 
                      site_df['expected_rate_treated'].notna()].copy()
    
    print(f"  Processed {len(site_df)} sites with valid 5mer contexts")
    
    window_rates = []
    
    # Process windows (now much faster since 5mer contexts are pre-computed)
    # Add progress reporting
    total_windows = len(windows_df)
    print(f"  Processing {total_windows} windows...")
    
    # Use itertuples instead of iterrows for better performance
    for window_idx, window in enumerate(windows_df.itertuples(), 1):
        if window_idx % 100 == 0 or window_idx == total_windows:
            print(f"    Progress: {window_idx}/{total_windows} windows ({100*window_idx/total_windows:.1f}%)")
        
        chrom = str(window.chrom)
        start = int(window.start)  # 1-based
        end = int(window.end)  # 0-based end (exclusive)
        
        # Get sites within this window (vectorized filtering)
        window_sites = site_df[
            (site_df['chrom_str'] == chrom) & 
            (site_df['pos_int'] >= start) & 
            (site_df['pos_int'] <= end)
        ].copy()
        
        if window_sites.empty:
            continue
        
        # Calculate window-level expected rates (vectorized)
        total_expected_control = (window_sites['expected_rate_control'] * window_sites['depth']).sum()
        total_expected_treated = (window_sites['expected_rate_treated'] * window_sites['depth']).sum()
        total_depth_all = window_sites['depth'].sum()
        
        if total_depth_all == 0:
                continue
            
        expected_rate_control = total_expected_control / total_depth_all
        expected_rate_treated = total_expected_treated / total_depth_all
        
        # Quick check: if window has very few sites or mutations, skip GLM and use simple rates
        total_mutations_pre = window_sites['ems_count'].sum()
        n_sites_pre = len(window_sites)
        
        # Skip GLM for windows with < 50 sites or < 10 mutations (use simple rate instead)
        # This is more aggressive to speed up processing - GLM is only worth it for windows with substantial data
        use_simple_rate = (n_sites_pre < 50) or (total_mutations_pre < 10)
        
        # Prepare data for GLM (much faster now)
        # Ensure treatment column exists
        if 'treatment' not in window_sites.columns:
            window_sites['treatment'] = (~window_sites.get('is_control', pd.Series([True] * len(window_sites)))).astype(int)
        
        glm_df = window_sites[['pos_int', 'depth', 'ems_count', 'treatment', 
                               'expected_rate_control', 'expected_rate_treated']].copy()
        glm_df.columns = ['pos', 'depth', 'ems_count', 'treatment', 
                          'expected_rate_control', 'expected_rate_treated']
        
        # Step 2: Fit site-level GLM with 5mer-expected rate as offset
        # Skip GLM for small windows and use simple rates instead (much faster)
        
        if use_simple_rate:
            # Use simple observed rates for small windows (skip GLM fitting)
            control_sites = glm_df[glm_df['treatment'] == 0]
            treated_sites = glm_df[glm_df['treatment'] == 1]
            
            control_depth = control_sites['depth'].sum()
            treated_depth = treated_sites['depth'].sum()
            control_mutations = control_sites['ems_count'].sum()
            treated_mutations = treated_sites['ems_count'].sum()
            
            if control_depth > 0:
                rate_control = control_mutations / control_depth
            else:
                rate_control = expected_rate_control
            
            if treated_depth > 0:
                rate_treated = treated_mutations / treated_depth
            else:
                rate_treated = expected_rate_treated
            
            # Simple Poisson CI for small windows
            if control_mutations > 0:
                se_log_control = 1.0 / np.sqrt(control_mutations)
                ci_low_control = rate_control * np.exp(-1.96 * se_log_control)
                ci_high_control = rate_control * np.exp(1.96 * se_log_control)
            else:
                ci_low_control = 0.0
                ci_high_control = 3.69 / control_depth if control_depth > 0 else np.nan
            
            if treated_mutations > 0:
                se_log_treated = 1.0 / np.sqrt(treated_mutations)
                ci_low_treated = rate_treated * np.exp(-1.96 * se_log_treated)
                ci_high_treated = rate_treated * np.exp(1.96 * se_log_treated)
            else:
                ci_low_treated = 0.0
                ci_high_treated = 3.69 / treated_depth if treated_depth > 0 else np.nan
        
        elif use_treatment_covariate:
            # Fit GLM: log(E[Y]) = log(expected_rate_from_5mer) + β₀ + β_treatment × treatment
            # The offset accounts for 5mer-expected rate, so coefficients give deviations from expected
            try:
                # Prepare data
                # Offset should be log(expected_rate * depth) because:
                # E[Y] = expected_rate * depth * exp(X*beta)
                # So offset = log(expected_rate * depth) = log(expected_rate) + log(depth)
                glm_df['log_expected_control'] = np.log(glm_df['expected_rate_control'].values) + np.log(glm_df['depth'].values)
                glm_df['log_expected_treated'] = np.log(glm_df['expected_rate_treated'].values) + np.log(glm_df['depth'].values)
                
                # Use appropriate expected rate offset based on treatment
                glm_df['log_expected'] = np.where(
                    glm_df['treatment'] == 0,
                    glm_df['log_expected_control'],
                    glm_df['log_expected_treated']
                )
                
                # Design matrix
                X = glm_df[['treatment']]
                design = sm.add_constant(X, has_constant='add')
                y = glm_df['ems_count'].values
                offset = glm_df['log_expected'].values
                
                # Fit GLM
                if method == "negative_binomial":
                    # Use Poisson as warm start
                    pois_model = sm.GLM(y, design, family=sm.families.Poisson(), offset=offset)
                    pois_res = pois_model.fit()
                    start_beta = pois_res.params.values
                    
                    # Fit NB with estimated alpha
                    exposure = np.exp(offset)
                    nb_model = NBDiscrete(y, design, exposure=exposure)
                    start_params_nb = np.r_[start_beta, 0.1]
                    result = nb_model.fit(start_params=start_params_nb, disp=False)
                else:
                    # Poisson
                    pois_model = sm.GLM(y, design, family=sm.families.Poisson(), offset=offset)
                    result = pois_model.fit()
                
                # Extract rates
                beta0 = result.params.get('const', np.nan)
                beta_t = result.params.get('treatment', np.nan)
                
                # Rates are: expected_rate × exp(β₀) for control, expected_rate × exp(β₀ + β_t) for treated
                rate_control = expected_rate_control * np.exp(beta0) if np.isfinite(beta0) and np.isfinite(expected_rate_control) else np.nan
                rate_treated = expected_rate_treated * np.exp(beta0 + beta_t) if np.isfinite(beta0) and np.isfinite(beta_t) and np.isfinite(expected_rate_treated) else np.nan
                
                # Calculate CIs
                ci_low_control, ci_high_control = np.nan, np.nan
                ci_low_treated, ci_high_treated = np.nan, np.nan
                
                try:
                    if np.isfinite(beta0):
                        ci_beta0 = result.conf_int().loc['const'].values if 'const' in result.conf_int().index else None
                        if ci_beta0 is not None and len(ci_beta0) == 2:
                            ci_low_control = expected_rate_control * np.exp(ci_beta0[0])
                            ci_high_control = expected_rate_control * np.exp(ci_beta0[1])
                    
                    if np.isfinite(beta0) and np.isfinite(beta_t):
                        vcov = result.cov_params()
                        var_sum = (vcov.loc['const', 'const'] + 
                                  vcov.loc['treatment', 'treatment'] +
                                  2 * vcov.loc['const', 'treatment'])
                        se_sum = np.sqrt(var_sum) if var_sum > 0 else np.nan
                        if np.isfinite(se_sum) and se_sum > 0:
                            z = 1.96
                            linear_pred_treated = beta0 + beta_t
                            ci_low_treated = expected_rate_treated * np.exp(linear_pred_treated - z * se_sum)
                            ci_high_treated = expected_rate_treated * np.exp(linear_pred_treated + z * se_sum)
                except Exception:
                    pass
                
            except Exception as e:
                # Fallback: simple rate calculation
                rate_control = expected_rate_control
                rate_treated = expected_rate_treated
                ci_low_control, ci_high_control = np.nan, np.nan
                ci_low_treated, ci_high_treated = np.nan, np.nan
        else:
            # Intercept-only: fit separately for control and treated
            rate_control, ci_low_control, ci_high_control = expected_rate_control, np.nan, np.nan
            rate_treated, ci_low_treated, ci_high_treated = expected_rate_treated, np.nan, np.nan
            
            control_sites = glm_df[glm_df['treatment'] == 0]
            treated_sites = glm_df[glm_df['treatment'] == 1]
            
            if not control_sites.empty:
                try:
                    X_ctrl = pd.DataFrame({'const': [1.0] * len(control_sites)})
                    design_ctrl = sm.add_constant(X_ctrl, has_constant='add')
                    y_ctrl = control_sites['ems_count'].values
                    # Offset should include depth: log(expected_rate * depth)
                    offset_ctrl = np.log(control_sites['expected_rate_control'].values) + np.log(control_sites['depth'].values)
                    
                    if method == "negative_binomial":
                        exposure_ctrl = np.exp(offset_ctrl)
                        nb_model_ctrl = NBDiscrete(y_ctrl, design_ctrl, exposure=exposure_ctrl)
                        result_ctrl = nb_model_ctrl.fit(disp=False)
                    else:
                        pois_model_ctrl = sm.GLM(y_ctrl, design_ctrl, family=sm.families.Poisson(), offset=offset_ctrl)
                        result_ctrl = pois_model_ctrl.fit()
                    
                    beta0_ctrl = result_ctrl.params.get('const', 0.0)
                    rate_control = expected_rate_control * np.exp(beta0_ctrl) if np.isfinite(beta0_ctrl) else expected_rate_control
                except Exception:
                    pass
            
            if not treated_sites.empty:
                try:
                    X_treat = pd.DataFrame({'const': [1.0] * len(treated_sites)})
                    design_treat = sm.add_constant(X_treat, has_constant='add')
                    y_treat = treated_sites['ems_count'].values
                    # Offset should include depth: log(expected_rate * depth)
                    offset_treat = np.log(treated_sites['expected_rate_treated'].values) + np.log(treated_sites['depth'].values)
                    
                    if method == "negative_binomial":
                        exposure_treat = np.exp(offset_treat)
                        nb_model_treat = NBDiscrete(y_treat, design_treat, exposure=exposure_treat)
                        result_treat = nb_model_treat.fit(disp=False)
                    else:
                        pois_model_treat = sm.GLM(y_treat, design_treat, family=sm.families.Poisson(), offset=offset_treat)
                        result_treat = pois_model_treat.fit()
                    
                    beta0_treat = result_treat.params.get('const', 0.0)
                    rate_treated = expected_rate_treated * np.exp(beta0_treat) if np.isfinite(beta0_treat) else expected_rate_treated
                except Exception:
                    pass
        
        # Aggregate statistics
        total_mutations = glm_df['ems_count'].sum()
        total_depth = glm_df['depth'].sum()
        
        # Calculate simple observed rates for sanity check (only if we used GLM)
        # Get control/treated depths and mutations (needed for sanity check and zero-mutation handling)
        control_depth = glm_df[glm_df['treatment'] == 0]['depth'].sum()
        treated_depth = glm_df[glm_df['treatment'] == 1]['depth'].sum()
        control_mutations = glm_df[glm_df['treatment'] == 0]['ems_count'].sum()
        treated_mutations = glm_df[glm_df['treatment'] == 1]['ems_count'].sum()
        
        if not use_simple_rate:
            # Calculate simple rates for sanity check when using GLM
            simple_rate_control = control_mutations / control_depth if control_depth > 0 else 0.0
            simple_rate_treated = treated_mutations / treated_depth if treated_depth > 0 else 0.0
        else:
            # Already calculated above for simple rate case
            simple_rate_control = rate_control
            simple_rate_treated = rate_treated
        
        # If no mutations observed, set rates to 0
        if total_mutations == 0:
            rate_control = 0.0
            rate_treated = 0.0
            ci_low_control = 0.0
            ci_high_control = 0.0
            ci_low_treated = 0.0
            ci_high_treated = 0.0
        elif not use_simple_rate:
            # Sanity check: if calculated rate is way off from simple rate, use simple rate
            # This can happen if GLM fails or gives unreasonable coefficients
            if control_depth > 0 and np.isfinite(rate_control) and np.isfinite(simple_rate_control):
                if rate_control > simple_rate_control * 10 or rate_control < simple_rate_control / 10:
                    # Rate is way off, use simple rate instead
                    rate_control = simple_rate_control
                    ci_low_control = np.nan
                    ci_high_control = np.nan
            
            if treated_depth > 0 and np.isfinite(rate_treated) and np.isfinite(simple_rate_treated):
                if rate_treated > simple_rate_treated * 10 or rate_treated < simple_rate_treated / 10:
                    # Rate is way off, use simple rate instead
                    rate_treated = simple_rate_treated
                    ci_low_treated = np.nan
                    ci_high_treated = np.nan
        
        window_rates.append({
            'window_id': window.window_id,
            'chrom': chrom,
            'start': start,
            'end': end,
            'window_size': getattr(window, 'window_size', end - start + 1),
            'rate_control': rate_control,
            'rate_treated': rate_treated,
            'expected_rate_control': expected_rate_control,
            'expected_rate_treated': expected_rate_treated,
            'CI_low_control': ci_low_control,
            'CI_high_control': ci_high_control,
            'CI_low_treated': ci_low_treated,
            'CI_high_treated': ci_high_treated,
            'n_sites': len(glm_df),
            'total_depth': total_depth,
            'total_mutations': total_mutations,
        })
    
    rates_df = pd.DataFrame(window_rates)
    print(f"Calculated 5mer-normalized rates for {len(rates_df)} windows")
    
    return rates_df


def compare_regional_rates_gc_normalized(
    windows_rates_df: pd.DataFrame,
    region_type: str = "chromosome",
    output_dir: str = None,
    method: str = "negative_binomial",
    use_treatment_covariate: bool = True
) -> pd.DataFrame:
    """
    Compare mutation rates across genomic regions using GC-normalized windows.
    
    Since windows are GC-normalized, GC is controlled when comparing regions.
    
    Args:
        windows_rates_df: DataFrame with window rates (from calculate_window_rates)
        region_type: Type of region ('chromosome' or 'custom')
        output_dir: Output directory for results
        method: GLM method
        use_treatment_covariate: Whether to use treatment covariate
    
    Returns:
        DataFrame with regional rate comparisons
    """
    print(f"Comparing rates across regions (region_type={region_type})...")
    
    # For now, use chromosome as region
    if region_type == "chromosome":
        windows_rates_df['region'] = windows_rates_df['chrom']
    
    # Prepare data for GLM
    # Aggregate window-level data for regional comparison
    # Model: log(E[Y]) = log(depth) + β₀ + β_treatment·treatment + β_region·region
    
    # For each window, we have aggregated data
    # We'll fit a GLM with region as covariate
    
    # Prepare site-level data grouped by window and region
    # This is a bit tricky - we need to restructure the data
    
    # For now, let's do a simpler approach: compare mean rates across regions
    # with statistical tests
    
    regional_summary = []
    
    for region in sorted(windows_rates_df['region'].unique()):
        region_windows = windows_rates_df[windows_rates_df['region'] == region]
        
        if use_treatment_covariate:
            # Calculate mean rates
            control_rates = region_windows['rate_control'].dropna()
            treated_rates = region_windows['rate_treated'].dropna()
            
            regional_summary.append({
                'region': region,
                'n_windows': len(region_windows),
                'mean_rate_control': control_rates.mean() if len(control_rates) > 0 else np.nan,
                'mean_rate_treated': treated_rates.mean() if len(treated_rates) > 0 else np.nan,
                'median_rate_control': control_rates.median() if len(control_rates) > 0 else np.nan,
                'median_rate_treated': treated_rates.median() if len(treated_rates) > 0 else np.nan,
                'total_sites': region_windows['n_sites'].sum(),
                'total_depth': region_windows['total_depth'].sum(),
            })
        else:
            # Single rate per window
            rates = region_windows['rate_control'].dropna()
            regional_summary.append({
                'region': region,
                'n_windows': len(region_windows),
                'mean_rate': rates.mean() if len(rates) > 0 else np.nan,
                'median_rate': rates.median() if len(rates) > 0 else np.nan,
                'total_sites': region_windows['n_sites'].sum(),
                'total_depth': region_windows['total_depth'].sum(),
            })
    
    regional_df = pd.DataFrame(regional_summary)
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        regional_path = os.path.join(output_dir, 'regional_rates_summary.tsv')
        regional_df.to_csv(regional_path, sep='\t', index=False)
        print(f"Regional rates summary saved to {regional_path}")
    
    return regional_df


def analyze_gene_hits_in_windows(
    windows_rates_df: pd.DataFrame,
    gff_file: str,
    rate_threshold_multiplier: float = 1.5,
    use_treated_rate: bool = True
) -> pd.DataFrame:
    """
    Identify windows with mutation rates far above standard and find overlapping genes.
    
    Args:
        windows_rates_df: DataFrame with window rates (from calculate_window_rates)
        gff_file: Path to GFF file with gene annotations
        rate_threshold_multiplier: Multiplier for median rate to define "far above" (default: 2.0)
        use_treated_rate: If True, use treated rate; else use control rate
    
    Returns:
        DataFrame with columns: window_index, window_length, window_gc_content, 
        genes_included, gene_coverage_percentage
    """
    print(f"Analyzing gene hits in high-rate windows...")
    
    # Determine which rate column to use
    if use_treated_rate and 'rate_treated' in windows_rates_df.columns:
        rate_col = 'rate_treated'
    else:
        rate_col = 'rate_control'
    
    # Calculate standard mutation rate (median of all windows)
    valid_rates = windows_rates_df[rate_col].dropna()
    if len(valid_rates) == 0:
        print("Warning: No valid rates found for gene hit analysis")
        return pd.DataFrame()
    
    median_rate = valid_rates.median()
    threshold_rate = median_rate * rate_threshold_multiplier
    
    print(f"Median mutation rate: {median_rate:.6e}")
    print(f"Threshold (>{rate_threshold_multiplier}x median): {threshold_rate:.6e}")
    
    # Identify high-rate windows
    high_rate_windows = windows_rates_df[
        (windows_rates_df[rate_col] > threshold_rate) & 
        windows_rates_df[rate_col].notna()
    ].copy()
    
    print(f"Found {len(high_rate_windows)} windows with rates above threshold")
    
    if len(high_rate_windows) == 0:
        return pd.DataFrame(columns=['window_index', 'window_length', 'window_gc_content', 
                                    'genes_included', 'gene_coverage_percentage'])
    
    # Load genes from GFF file
    try:
        gene_regions = parse_gff(gff_file)
        print(f"Loaded {len(gene_regions)} genes from GFF file")
    except Exception as e:
        print(f"Error loading GFF file: {e}")
        return pd.DataFrame(columns=['window_index', 'window_length', 'window_gc_content', 
                                    'genes_included', 'gene_coverage_percentage'])
    
    # Organize genes by chromosome for faster lookup
    genes_by_chrom = defaultdict(list)
    for chrom, start, end, gene_id in gene_regions:
        genes_by_chrom[str(chrom)].append((start, end, gene_id))
    
    # For each high-rate window, find overlapping genes
    gene_hits = []
    
    for idx, window in high_rate_windows.iterrows():
        chrom = str(window['chrom'])
        window_start = int(window['start'])
        window_end = int(window['end'])
        window_length = window_end - window_start + 1
        window_gc = window.get('gc_content', np.nan)
        # Try to get window_id, fall back to index-based name
        if 'window_id' in window.index:
            window_id = str(window['window_id'])
        else:
            window_id = f'window_{idx}'
        
        # Find overlapping genes
        overlapping_genes = []
        coverage_percentages = []
        
        if chrom in genes_by_chrom:
            for gene_start, gene_end, gene_id in genes_by_chrom[chrom]:
                # Calculate overlap
                overlap_start = max(window_start, gene_start)
                overlap_end = min(window_end, gene_end)
                overlap_length = max(0, overlap_end - overlap_start + 1)
                
                if overlap_length > 0:
                    # Calculate coverage percentage (percentage of window covered by gene)
                    coverage_pct = (overlap_length / window_length) * 100
                    overlapping_genes.append(gene_id)
                    coverage_percentages.append(coverage_pct)
        
        # Format genes and coverage percentages as comma-separated strings
        if overlapping_genes:
            genes_str = ','.join(overlapping_genes)
            coverage_str = ','.join([f'{pct:.2f}' for pct in coverage_percentages])
        else:
            genes_str = 'none'
            coverage_str = '0.00'
        
        gene_hits.append({
            'window_index': window_id,
            'window_length': window_length,
            'window_gc_content': window_gc,
            'genes_included': genes_str,
            'gene_coverage_percentage': coverage_str
        })
    
    result_df = pd.DataFrame(gene_hits)
    print(f"Gene hit analysis complete: {len(result_df)} high-rate windows analyzed")
    
    return result_df


def plot_regional_rates_manhattan(
    windows_rates_df: pd.DataFrame,
    output_path: str,
    use_treatment_covariate: bool = True,
    significance_threshold: float = 0.05,
    gff_file: str = None,
    rate_threshold_multiplier: float = 2.0
):
    """
    Create Manhattan plot showing mutation rates across the genome.
    
    Windows with significantly different rates are colored differently.
    
    Args:
        windows_rates_df: DataFrame with window rates
        output_path: Path to save plot
        use_treatment_covariate: Whether to show treated rates
        significance_threshold: p-value threshold for significance
    """
    print(f"Creating Manhattan plot...")
    
    # Calculate genomic position (cumulative across chromosomes)
    chrom_order = sorted(windows_rates_df['chrom'].unique())
    chrom_lengths = {}
    chrom_offsets = {}
    
    cumulative_pos = 0
    for chrom in chrom_order:
        chrom_windows = windows_rates_df[windows_rates_df['chrom'] == chrom]
        chrom_max = chrom_windows['end'].max() if len(chrom_windows) > 0 else 0
        chrom_lengths[chrom] = chrom_max
        chrom_offsets[chrom] = cumulative_pos
        cumulative_pos += chrom_max
    
    # Calculate x positions
    windows_rates_df = windows_rates_df.copy()
    windows_rates_df['x_pos'] = windows_rates_df.apply(
        lambda row: chrom_offsets[row['chrom']] + (row['start'] + row['end']) / 2,
        axis=1
    )
    
    # Determine significance using statistical tests
    # Test if each window's rate is significantly different from the overall mean rate
    # This identifies regions with unusually high or low mutation rates (hotspots/coldspots)
    if use_treatment_covariate:
        # Use treated rate for comparison (since that's what we're interested in)
        rate_col = 'rate_treated'
        
        # Calculate overall mean treated rate across all windows
        overall_mean = windows_rates_df[rate_col].dropna().mean()
        
        print(f"Comparing each window's treated rate to overall mean treated rate: {overall_mean:.6e}")
        
        # For each window, test if treated rate differs from overall mean
        windows_rates_df['p_value'] = np.nan
        
        for idx, row in windows_rates_df.iterrows():
            p_val = np.nan
            
            # Test if window rate differs from overall mean using CI-based test
            if (not pd.isna(row[rate_col]) and not pd.isna(overall_mean) and
                row[rate_col] > 0 and
                not pd.isna(row.get('CI_low_treated')) and not pd.isna(row.get('CI_high_treated'))):
                
                # Calculate SE on log scale from CI
                # For log-normal: SE_log ≈ (log(CI_high) - log(CI_low)) / (2 * 1.96)
                se_log_rate = (np.log(row['CI_high_treated']) - np.log(row['CI_low_treated'])) / (2 * 1.96)
                
                if se_log_rate > 0 and np.isfinite(se_log_rate):
                    # Test if log(rate) differs from log(overall_mean)
                    log_rate = np.log(row[rate_col])
                    log_mean = np.log(overall_mean)
                    log_diff = log_rate - log_mean
                    
                    # SE of the difference (assuming independence)
                    # Approximate SE of overall mean (using median absolute deviation as proxy)
                    # For simplicity, use the window's own SE (conservative)
                    se_log_diff = se_log_rate
                    
                    if se_log_diff > 0 and np.isfinite(se_log_diff):
                        z = log_diff / se_log_diff
                        if np.isfinite(z):
                            p_val = 2 * (1 - stats.norm.cdf(abs(z)))
            
            # Store p-value (significance will be determined after FDR correction)
            if np.isfinite(p_val) and not pd.isna(p_val):
                windows_rates_df.at[idx, 'p_value'] = p_val
        
        # Calculate fold-change relative to overall mean (for coloring)
        windows_rates_df['fold_change'] = (
            windows_rates_df[rate_col] / overall_mean
        ).replace([np.inf, -np.inf], np.nan)
    else:
        # Single rate per window - compare to overall mean
        overall_mean = windows_rates_df['rate_control'].dropna().mean()
        windows_rates_df['p_value'] = np.nan
        
        for idx, row in windows_rates_df.iterrows():
            if (not pd.isna(row['rate_control']) and not pd.isna(overall_mean) and
                not pd.isna(row['CI_low_control']) and not pd.isna(row['CI_high_control'])):
                # Approximate SE from CI
                se = (row['CI_high_control'] - row['CI_low_control']) / (2 * 1.96)
                if se > 0 and np.isfinite(se):
                    z = (row['rate_control'] - overall_mean) / se
                    if np.isfinite(z):
                        p_val = 2 * (1 - stats.norm.cdf(abs(z)))
                        windows_rates_df.at[idx, 'p_value'] = p_val
        
        rate_col = 'rate_control'
    
    # Apply FDR correction (Benjamini-Hochberg) to p-values
    from statsmodels.stats.multitest import multipletests
    
    # Get all valid p-values
    valid_pvalues = windows_rates_df['p_value'].dropna()
    if len(valid_pvalues) > 1:
        # Apply FDR correction
        pvalues_array = valid_pvalues.values
        rejected, pvalues_corrected, alpha_sidak, alpha_bonf = multipletests(
            pvalues_array, 
            alpha=significance_threshold, 
            method='fdr_bh'  # Benjamini-Hochberg FDR correction
        )
        
        # Store corrected p-values
        windows_rates_df['p_value_fdr_corrected'] = np.nan
        windows_rates_df.loc[valid_pvalues.index, 'p_value_fdr_corrected'] = pvalues_corrected
        
        # Update significance based on FDR-corrected p-values
        windows_rates_df['is_significant'] = False
        windows_rates_df.loc[valid_pvalues.index[rejected], 'is_significant'] = True
        
        n_significant = rejected.sum()
        n_tested = len(valid_pvalues)
        print(f"Significance testing (FDR-corrected): {n_significant} of {n_tested} windows significant at FDR<{significance_threshold}")
        print(f"  Applied Benjamini-Hochberg FDR correction for multiple comparisons")
    else:
        # If only one or no p-values, no correction needed
        n_significant = windows_rates_df['is_significant'].sum()
        n_tested = windows_rates_df['p_value'].notna().sum()
        windows_rates_df['p_value_fdr_corrected'] = windows_rates_df['p_value']
        print(f"Significance testing: {n_significant} of {n_tested} windows significant at p<{significance_threshold}")
        if n_tested <= 1:
            print(f"  No multiple comparisons correction needed (only {n_tested} test(s))")
    
    # Create plot
    fig, ax = plt.subplots(figsize=(16, 6))
    
    # Plot non-significant windows
    non_sig = windows_rates_df[~windows_rates_df['is_significant']]
    if len(non_sig) > 0:
        ax.scatter(non_sig['x_pos'], non_sig[rate_col], 
                  c='gray', alpha=0.5, s=20, label='Not significant')
    
    # Plot significant windows
    sig = windows_rates_df[windows_rates_df['is_significant']]
    if len(sig) > 0:
        # Color by whether rate is above or below overall mean (identifies hotspots/coldspots)
        if use_treatment_covariate:
            overall_mean = windows_rates_df[rate_col].dropna().mean()
        else:
            overall_mean = windows_rates_df[rate_col].dropna().mean()
        
            increased = sig[sig[rate_col] > overall_mean]
            decreased = sig[sig[rate_col] < overall_mean]
            
            if len(increased) > 0:
                ax.scatter(increased['x_pos'], increased[rate_col],
                      c='red', alpha=0.7, s=30, 
                      label=f'Hotspot: significantly above mean (FDR<{significance_threshold})')
            if len(decreased) > 0:
                ax.scatter(decreased['x_pos'], decreased[rate_col],
                      c='blue', alpha=0.7, s=30, 
                      label=f'Coldspot: significantly below mean (FDR<{significance_threshold})')
    
    # Add chromosome boundaries
    for i, chrom in enumerate(chrom_order):
        offset = chrom_offsets[chrom]
        if i > 0:
            ax.axvline(x=offset, color='black', linestyle='--', alpha=0.3, linewidth=0.5)
    
    # Add horizontal line at overall mean
    if use_treatment_covariate:
        overall_mean = windows_rates_df[rate_col].dropna().mean()
    else:
        overall_mean = windows_rates_df[rate_col].dropna().mean()
    ax.axhline(y=overall_mean, color='green', linestyle='--', linewidth=1.5, 
               alpha=0.7, label=f'Overall mean: {overall_mean:.2e}')
    
    # Labels
    ax.set_xlabel('Genomic Position', fontsize=12)
    ax.set_ylabel('Mutation Rate (per base)', fontsize=12)
    if use_treatment_covariate:
        ax.set_title('Treated Mutation Rates Across Genome\n(compared to overall mean; FDR-corrected)', 
                    fontsize=14, fontweight='bold')
    else:
        ax.set_title('Mutation Rates Across Genome\n(compared to overall mean; FDR-corrected)', 
                    fontsize=14, fontweight='bold')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(alpha=0.3, axis='y')
    
    # Chromosome labels on x-axis
    chrom_centers = [chrom_offsets[chrom] + chrom_lengths[chrom] / 2 for chrom in chrom_order]
    ax.set_xticks(chrom_centers)
    ax.set_xticklabels(chrom_order, rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Manhattan plot saved to {output_path}")
    
    # Perform gene hit analysis if GFF file is provided
    if gff_file and os.path.exists(gff_file):
        print(f"Performing gene hit analysis with GFF file: {gff_file}")
        use_treated = use_treatment_covariate and 'rate_treated' in windows_rates_df.columns
        
        gene_hits_df = analyze_gene_hits_in_windows(
            windows_rates_df=windows_rates_df,
            gff_file=gff_file,
            rate_threshold_multiplier=rate_threshold_multiplier,
            use_treated_rate=use_treated
        )
        
        if not gene_hits_df.empty:
            # Save TSV to same directory as manhattan plot
            output_dir = os.path.dirname(output_path)
            if output_dir:
                gene_hits_path = os.path.join(output_dir, 'gene_hits_high_rate_windows.tsv')
            else:
                # If output_path is just a filename, save in current directory
                base_name = os.path.splitext(os.path.basename(output_path))[0]
                gene_hits_path = f'{base_name}_gene_hits.tsv'
            
            gene_hits_df.to_csv(gene_hits_path, sep='\t', index=False)
            print(f"Gene hits TSV saved to {gene_hits_path}")
        else:
            print("No high-rate windows found for gene hit analysis")
    elif gff_file:
        print(f"Warning: GFF file specified but not found: {gff_file}")


def fit_site_level_glm_intercept_only(df_sites: pd.DataFrame, method: str = "poisson", alpha: float = 0.0) -> tuple[float, float, float]:
    """Fit intercept-only site-level GLM on provided site rows for a single sample.

    Expects columns: depth, y (counts to model). Applies alpha subtraction at counts level.
    Returns (rate, ci_low, ci_high) in per-base units.
    """
    if df_sites is None or df_sites.empty:
        return (float("nan"), float("nan"), float("nan"))

    work = df_sites.copy()
    work["log_depth"] = np.log(work["depth"].values)

    if alpha and alpha > 0:
        y = np.maximum(0.0, work["y"].values - alpha * work["depth"].values)
    else:
        y = work["y"].values

    design = sm.add_constant(pd.DataFrame(index=work.index), has_constant='add')

    try:
        if method == "poisson":
            fam = sm.families.Poisson()
            model = sm.GLM(y, design, family=fam, offset=work["log_depth"].values)
            res = model.fit()
        else:
            # For negative binomial, estimate alpha from all data
            # This is correct when using treatment covariate approach
            from statsmodels.discrete.discrete_model import NegativeBinomial as NBDiscrete
            exposure = np.exp(work["log_depth"].values)
            # Warm start with Poisson
            pois_model = sm.GLM(y, design, family=sm.families.Poisson(), offset=work["log_depth"].values)
            pois_result = pois_model.fit()
            start_params = np.r_[pois_result.params.values, 0.1]  # Add dispersion parameter
            
            nb_model = NBDiscrete(y, design, exposure=exposure)
            res = nb_model.fit(start_params=start_params, disp=False)
        b0 = res.params.get("const", np.nan)
        rate = float(np.exp(b0)) if np.isfinite(b0) else np.nan
        try:
            ci_b = res.conf_int().loc["const"].values
            ci_low = float(np.exp(ci_b[0])); ci_high = float(np.exp(ci_b[1]))
        except Exception:
            ci_low = np.nan; ci_high = np.nan
        return (rate, ci_low, ci_high)
    except Exception:
        return (float("nan"), float("nan"), float("nan"))

def main():
    parser = argparse.ArgumentParser(
        description="Estimate mutation rates with enhanced statistical methods"
    )
    parser.add_argument("--counts-dir", type=str, required=True,
                        help="Directory containing .counts files")
    parser.add_argument("--output-prefix", type=str, default="simple_mutation_rates",
                        help="Prefix for output files (default: simple_mutation_rates)")
    parser.add_argument("--min-alt", type=int, default=1,
                        help="Minimum alt allele count to include site in numerator (default 1)")
    parser.add_argument("--exclude-controls", action="store_true",
                        help="Exclude control samples from analysis")
    parser.add_argument("--exclusion-mask", type=str, default=None,
                        help="Path to exclusion mask file (optional)")
    
    # Statistical enhancement arguments (all enabled by default)
    parser.add_argument("--no-alpha-correction", action="store_true",
                        help="Skip alpha correction for simple rates (GLM methods use treatment covariate by default)")
    parser.add_argument("--use-alpha-correction", action="store_true",
                        help="Use alpha correction approach instead of treatment covariate for GLM methods (legacy)")
    parser.add_argument("--gff-file", type=str, default=None,
                        help="GFF file for gene annotations (required for category analysis)")
    parser.add_argument("--genome-fasta", type=str, default=None,
                        help="Reference genome FASTA file (required for category analysis)")
    parser.add_argument("--codon-table-path", type=str, default=None,
                        help="Path to codon table JSON file (optional, defaults to data/references/11.json)")
    parser.add_argument("--compare-categories", action="store_true",
                        help="Compare mutation rates across categories (intergenic, synonymous, non-synonymous)")
    parser.add_argument("--compare-treatment-days", action="store_true",
                        help="Compare mutation rates across treatment days (3d, 7d, no_label) in treated samples")
    parser.add_argument("--kmer5-normalized-windows", action="store_true",
                        help="Analyze mutation rates using 5mer-normalized fixed-size windows")
    parser.add_argument("--window-size", type=int, default=5000,
                        help="Fixed window size in bp for 5mer-normalized analysis (default: 5000)")
    parser.add_argument("--kmer5-model-path", type=str, default=None,
                        help="Path to fitted 5mer model pickle file (required for --kmer5-normalized-windows)")
    parser.add_argument("--min-alt-c", type=int, default=None,
                        help="Minimum alt reads at C sites (C>T) to count a site; overrides --min-alt for C")
    parser.add_argument("--min-alt-g", type=int, default=None,
                        help="Minimum alt reads at G sites (G>A) to count a site; overrides --min-alt for G (unused in C-only mode)")
    parser.add_argument("--site-glm", action="store_true",
                        help="Run site-level GLM without sequence-context covariates")
    parser.add_argument("--site-glm-method", type=str, default="poisson",
                        choices=["poisson", "negative_binomial"],
                        help="GLM family for site-level model (default: poisson)")
    # NB alpha is estimated from all data (controls + treated) when using treatment covariate approach

    # Sweep options (site-level GLM)
    parser.add_argument("--sweep", action="store_true",
                        help="Run a min-alt sweep using site-level per-sample GLM; writes TSVs and plots similar to the sweep script")
    parser.add_argument("--min-alt-start", type=int, default=2,
                        help="Sweep start threshold (inclusive) for min-alt")
    parser.add_argument("--min-alt-end", type=int, default=5,
                        help="Sweep end threshold (inclusive) for min-alt")
    parser.add_argument("--sweep-target", type=str, default="C", choices=["C", "G"],
                        help="Which base to sweep (C or G); the other base fixed at --other-min-alt")
    parser.add_argument("--other-min-alt", type=int, default=1,
                        help="Fixed min-alt to apply to the non-swept base during sweep (default 1)")
    parser.add_argument("--use-split-alpha", action="store_true",
                        help="During sweep, attempt separate alpha for genic vs intergenic (requires --gff-file); otherwise use overall alpha")
    
    args = parser.parse_args()

    # Find all .counts files
    counts_files = sorted(glob.glob(os.path.join(args.counts_dir, "*.counts")))
    if not counts_files:
        print(f"No .counts files found in {args.counts_dir}")
        return

    print(f"Found {len(counts_files)} .counts files")

    # Load exclusion mask if provided
    excluded_sites = None
    if args.exclusion_mask:
        if os.path.exists(args.exclusion_mask):
            excluded_sites = load_exclusion_mask(args.exclusion_mask)
            print(f"Loaded exclusion mask: {len(excluded_sites)} sites to exclude")
        else:
            print(f"Warning: Exclusion mask file not found: {args.exclusion_mask}")

    # Step 1: Basic rate estimation (existing)
    results = []
    for filepath in counts_files:
        result = estimate_rates_from_counts(filepath, min_alt=args.min_alt, excluded_sites=excluded_sites)
        results.append(result)

    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Optionally exclude controls
    if args.exclude_controls:
        df = df[~df['is_control']].reset_index(drop=True)
        print(f"Excluded controls, analyzing {len(df)} samples")

    # Step 2: Determine approach for handling controls
    # Default: Treatment covariate approach (controls as covariate in GLM)
    # Legacy: Alpha correction approach (subtract background rate from counts)
    alpha = 0.0
    nb_alpha_controls = None
    use_treatment_covariate = not args.use_alpha_correction  # Default to True unless user requests alpha correction
    
    if df['is_control'].any():
        if use_treatment_covariate:
            print("Using treatment covariate approach (controls as covariate in GLM)")
            print("  This fits a pooled model: log(E[Y]) = log(depth) + β₀ + β₁·treatment")
            print("  Control rate = exp(β₀), Treated rate = exp(β₀ + β₁)")
        else:
            print("Using alpha correction approach (legacy method)")
            print("  This subtracts background rate from mutation counts before fitting")
        
        # Estimate alpha for reference or for simple rate alpha correction
        control_files = [f for f in counts_files if "NT" in os.path.basename(f)]
        if control_files:
            alpha = estimate_alpha_from_control_sites(control_files, min_alt=args.min_alt, excluded_sites=excluded_sites)
            print(f"Estimated alpha from pooled control sites (background rate): {alpha:.2e}")
        else:
            alpha = estimate_alpha_from_controls(df)
            print(f"Estimated alpha from weighted control rates (background rate): {alpha:.2e}")
    else:
        print("No control samples found, using intercept-only models")
        use_treatment_covariate = False

    # Step 3: Apply alpha correction to simple rates (if not skipped)
    # Note: GLM methods use treatment covariate approach by default
    if alpha > 0 and not args.no_alpha_correction:
        df = apply_alpha_correction(df, alpha)
        print("Applied alpha correction to simple rate estimates")

    # NB alpha is estimated from all data (controls + treated) when fitting negative binomial GLMs
    # No need to estimate separately from controls since treatment covariate approach uses all data

    # Step 4: GLM analysis removed - only site-level GLM analysis is now supported
    # Use --site-glm for site-level GLM analysis

    # Optional: Site-level GLM without sequence-context covariates
    if args.site_glm:
        print(f"Running site-level GLM (method={args.site_glm_method}) without sequence-context covariates")
        site_df = load_site_level_data_no_context(args.counts_dir, args.exclusion_mask)
        if site_df is None or site_df.empty:
            print("No site-level data loaded; skipping site-level GLM")
        else:
            site_result, site_summary = fit_site_level_glm_no_context(site_df, method=args.site_glm_method, alpha=alpha)

            # Save a concise report
            site_dir = args.output_prefix
            os.makedirs(site_dir, exist_ok=True)
            report_path = os.path.join(site_dir, "site_level_glm_report.txt")
            with open(report_path, "w") as fh:
                fh.write("=== Site-Level GLM (no context covariates) ===\n\n")
                fh.write(f"Method: {args.site_glm_method}\n")
                fh.write(f"Rows: {len(site_df)}\n")
                fh.write(f"Alpha subtracted: {alpha:.2e}\n\n")
                if site_result is not None:
                    fh.write(f"AIC: {site_summary.get('aic', float('nan'))}\n")
                    fh.write(f"BIC: {site_summary.get('bic', float('nan'))}\n")
                    fh.write(f"LogLik: {site_summary.get('llf', float('nan'))}\n\n")
                    fh.write("Coefficients (log-rate scale):\n")
                    fh.write(f"  const: {site_summary.get('coef_const', float('nan'))}\n")
                    fh.write(f"  treatment: {site_summary.get('coef_treatment', float('nan'))}\n\n")
                    fh.write("Implied per-base rates:\n")
                    fh.write(f"  control: {site_summary.get('rate_control', float('nan')):.6e}\n")
                    fh.write(f"  treated: {site_summary.get('rate_treated', float('nan')):.6e}\n")
                else:
                    fh.write("Model fitting failed.\n")
            print(f"Site-level GLM report saved to {report_path}")

            # Per-sample site-level GLM (intercept-only per sample)
            print("Fitting per-sample site-level GLMs (intercept-only)...")
            per_sample_rows = fit_site_level_glm_per_sample(site_df, method=args.site_glm_method, alpha=alpha)
            per_sample_df = pd.DataFrame(per_sample_rows)
            tsv_path = os.path.join(site_dir, "site_level_glm_per_sample.tsv")
            per_sample_df.to_csv(tsv_path, sep="\t", index=False)
            print(f"Per-sample site-level GLM rates saved to {tsv_path}")

            # Merge per-sample site GLM rates into main results table
            if not per_sample_df.empty:
                df = df.merge(per_sample_df[["sample", "site_glm_rate", "site_glm_CI_low", "site_glm_CI_high"]], on="sample", how="left")

            # Create per-sample site-level GLM plots
            create_site_level_glm_plots(df, os.path.join(site_dir, "site_glm"))
            save_individual_site_level_glm_plots(df, os.path.join(site_dir, "site_glm"))

    # Optional: Site-level GLM sweep (overall + optional genic/intergenic)
    if args.sweep:
        print(f"Running site-level GLM sweep: {args.sweep_target} min_alt {args.min_alt_start}..{args.min_alt_end}, other={args.other_min_alt}")
        sweep_dir = os.path.join(args.output_prefix, "sweep")
        os.makedirs(sweep_dir, exist_ok=True)

        # Load site-level rows with reference base
        site_all = load_site_level_with_ref(args.counts_dir, args.exclusion_mask)
        if site_all is None or site_all.empty:
            print("No site-level rows found; skipping sweep")
        else:
            thresholds = list(range(args.min_alt_start, args.min_alt_end + 1))

            # Prepare genic regions (optional)
            gene_regions = None
            if args.gff_file and os.path.exists(args.gff_file):
                print("Parsing GFF for genic/intergenic split...")
                gene_regions = parse_gff(args.gff_file)

            # Helper to test genic membership (simple scan over gene list by chrom)
            chrom_to_genes = {}
            if gene_regions:
                for chrom, start, end, gid in gene_regions:
                    chrom_to_genes.setdefault(chrom, []).append((start, end))
                for chrom in chrom_to_genes:
                    chrom_to_genes[chrom].sort()

            def is_genic(chrom: str, pos: int) -> bool:
                if not chrom_to_genes:
                    return False
                genes = chrom_to_genes.get(chrom)
                if not genes:
                    return False
                # Linear scan (acceptable for moderate sizes); could be optimized with bisect
                for s, e in genes:
                    if pos < s:
                        break
                    if s <= pos <= e:
                        return True
                return False

            # Alpha per threshold (overall); split alpha optional if genic available
            control_files = [f for f in counts_files if "NT" in os.path.basename(f)]

            overall_wide_frames = []
            genic_wide_frames = []
            inter_wide_frames = []
            trends_rows = []

            for thr in thresholds:
                thr_c = thr if args.sweep_target == "C" else args.other_min_alt
                thr_g = thr if args.sweep_target == "G" else args.other_min_alt

                # Estimate alpha overall for this threshold from controls
                alpha_thr = 0.0
                if control_files and not args.no_alpha_correction:
                    alpha_thr = estimate_alpha_from_control_sites(control_files, min_alt=min(thr_c, thr_g), excluded_sites=excluded_sites)

                # Prepare thresholded counts y
                work = site_all.copy()
                mask_c = (work["ref"] == "C") & (work["ems_count"] >= thr_c)
                mask_g = (work["ref"] == "G") & (work["ems_count"] >= thr_g)
                work["y"] = np.where(mask_c | mask_g, work["ems_count"], 0)

                # Per-sample overall site-level GLM
                samples = sorted(work["sample"].unique())
                overall_rows = []
                genic_rows = []
                inter_rows = []
                for s in samples:
                    sub = work[work["sample"] == s]
                    is_ctrl = bool(("NT" in s))

                    r_o, lo_o, hi_o = fit_site_level_glm_intercept_only(sub, method=args.site_glm_method, alpha=alpha_thr)
                    overall_rows.append({"sample": s, "rate": r_o, "lo": lo_o, "hi": hi_o})

                    # Genic/intergenic split if GFF available
                    if gene_regions:
                        mask_gx = sub.apply(lambda r: is_genic(r["chrom"], int(r["pos"])), axis=1)
                        sub_g = sub[mask_gx]
                        sub_i = sub[~mask_gx]

                        # If requested, we could compute split alphas; for now use overall alpha_thr
                        r_g, lo_g, hi_g = fit_site_level_glm_intercept_only(sub_g, method=args.site_glm_method, alpha=alpha_thr)
                        r_i, lo_i, hi_i = fit_site_level_glm_intercept_only(sub_i, method=args.site_glm_method, alpha=alpha_thr)
                        genic_rows.append({"sample": s, "rate": r_g, "lo": lo_g, "hi": hi_g})
                        inter_rows.append({"sample": s, "rate": r_i, "lo": lo_i, "hi": hi_i})

                    # Trends row
                    trends_rows.append({
                        "sample": s,
                        "min_alt": thr,
                        "rate": r_o,
                        "is_control": is_ctrl,
                    })

                label = f"{args.sweep_target}_minAlt{thr}"
                over_df = pd.DataFrame(overall_rows)
                overall_wide_frames.append(over_df[["sample", "rate"]].rename(columns={"rate": f"rate_{label}"}))

                if gene_regions:
                    gdf = pd.DataFrame(genic_rows)
                    idf = pd.DataFrame(inter_rows)
                    genic_wide_frames.append(gdf[["sample", "rate", "lo", "hi"]].rename(columns={
                        "rate": f"genic_rate_{label}", "lo": f"genic_CI_low_{label}", "hi": f"genic_CI_high_{label}"
                    }))
                    inter_wide_frames.append(idf[["sample", "rate", "lo", "hi"]].rename(columns={
                        "rate": f"intergenic_rate_{label}", "lo": f"intergenic_CI_low_{label}", "hi": f"intergenic_CI_high_{label}"
                    }))

            # Merge and save TSVs
            if overall_wide_frames:
                rates_wide = overall_wide_frames[0]
                for f in overall_wide_frames[1:]:
                    rates_wide = rates_wide.merge(f, on="sample", how="outer")
                rates_wide_path = os.path.join(sweep_dir, "site_level_overall_rates_wide.tsv")
                rates_wide.to_csv(rates_wide_path, sep="\t", index=False)
                print(f"Saved sweep overall wide TSV to {rates_wide_path}")

                rates_long = rates_wide.melt(id_vars=["sample"], var_name="threshold", value_name="rate")
                rates_long["min_alt"] = rates_long["threshold"].str.extract(r"(\d+)").astype(int)
                rates_long_path = os.path.join(sweep_dir, "site_level_overall_rates_long.tsv")
                rates_long.sort_values(["sample", "min_alt"]).to_csv(rates_long_path, sep="\t", index=False)
                print(f"Saved sweep overall long TSV to {rates_long_path}")

                # Trend plot
                trend_df = pd.DataFrame(trends_rows)
                if not trend_df.empty:
                    plt.figure(figsize=(10, 7))
                    control_color = "#1f77b4"; treated_color = "#ff7f0e"
                    for s, sub in trend_df.groupby("sample"):
                        sub = sub.sort_values("min_alt")
                        color = control_color if bool(("NT" in s)) else treated_color
                        plt.plot(sub["min_alt"], sub["rate"], marker="o", alpha=0.55, linewidth=1, color=color)
                    plt.yscale("log")
                    plt.xlabel("min_alt"); plt.ylabel("Per-base mutation probability (site-level GLM)")
                    plt.title("Per-sample site-level GLM rate trends vs min_alt")
                    legend_elements = [Line2D([0],[0], color=treated_color, lw=2, label="Treated"), Line2D([0],[0], color=control_color, lw=2, label="Control")]
                    plt.legend(handles=legend_elements, loc="best")
                    trend_path = os.path.join(sweep_dir, "site_level_trends_overall.png")
                    plt.tight_layout(); plt.savefig(trend_path, dpi=300); plt.close()
                    print(f"Saved sweep overall trend plot to {trend_path}")

            if gene_regions and genic_wide_frames and inter_wide_frames:
                genic_wide = genic_wide_frames[0]
                for f in genic_wide_frames[1:]:
                    genic_wide = genic_wide.merge(f, on="sample", how="outer")
                inter_wide = inter_wide_frames[0]
                for f in inter_wide_frames[1:]:
                    inter_wide = inter_wide.merge(f, on="sample", how="outer")
                genic_wide_path = os.path.join(sweep_dir, "site_level_genic_rates_wide.tsv")
                inter_wide_path = os.path.join(sweep_dir, "site_level_intergenic_rates_wide.tsv")
                genic_wide.to_csv(genic_wide_path, sep="\t", index=False)
                inter_wide.to_csv(inter_wide_path, sep="\t", index=False)
                print(f"Saved sweep genic/intergenic wide TSVs to {sweep_dir}")

                # Genic vs intergenic scatter per threshold
                for thr in thresholds:
                    label = f"{args.sweep_target}_minAlt{thr}"
                    gcol = f"genic_rate_{label}"; icol = f"intergenic_rate_{label}"
                    g_lo = f"genic_CI_low_{label}"; g_hi = f"genic_CI_high_{label}"
                    i_lo = f"intergenic_CI_low_{label}"; i_hi = f"intergenic_CI_high_{label}"
                    if gcol not in genic_wide.columns or icol not in inter_wide.columns:
                        continue
                    join_df = genic_wide[["sample", gcol, g_lo, g_hi]].merge(inter_wide[["sample", icol, i_lo, i_hi]], on="sample", how="inner")
                    if join_df.empty:
                        continue
                    join_df["g_err_low"] = join_df[gcol] - join_df[g_lo]
                    join_df["g_err_high"] = join_df[g_hi] - join_df[gcol]
                    join_df["i_err_low"] = join_df[icol] - join_df[i_lo]
                    join_df["i_err_high"] = join_df[i_hi] - join_df[icol]
                    plt.figure(figsize=(7,7))
                    # Control vs treated coloring
                    mask_ctrl = join_df["sample"].str.contains("NT")
                    treated_df = join_df[~mask_ctrl]; control_df = join_df[mask_ctrl]
                    if not treated_df.empty:
                        plt.errorbar(treated_df[gcol], treated_df[icol], xerr=[treated_df["g_err_low"], treated_df["g_err_high"]], yerr=[treated_df["i_err_low"], treated_df["i_err_high"]], fmt='o', color="#ff7f0e", alpha=0.6, elinewidth=1, capsize=2, label="Treated")
                    if not control_df.empty:
                        plt.errorbar(control_df[gcol], control_df[icol], xerr=[control_df["g_err_low"], control_df["g_err_high"]], yerr=[control_df["i_err_low"], control_df["i_err_high"]], fmt='o', color="#1f77b4", alpha=0.6, elinewidth=1, capsize=2, label="Control")
                    plt.xscale("log"); plt.yscale("log")
                    plt.xlabel(f"Genic per-base mutation probability (min_alt={thr})")
                    plt.ylabel(f"Intergenic per-base mutation probability (min_alt={thr})")
                    plt.title("Site-level GLM: Genic vs Intergenic rates (95% CI)")
                    # Diagonal
                    vals = pd.concat([join_df[gcol], join_df[icol]]).replace([np.inf, -np.inf], np.nan).dropna()
                    if not vals.empty:
                        lo_v, hi_v = vals.min(), vals.max()
                        if lo_v > 0 and np.isfinite(hi_v):
                            plt.plot([lo_v, hi_v], [lo_v, hi_v], linestyle="--", color="grey", alpha=0.6)
                    plt.legend(loc="best"); plt.grid(alpha=0.2)
                    scatter_path = os.path.join(sweep_dir, f"site_level_genic_vs_intergenic_minAlt{thr}.png")
                    plt.tight_layout(); plt.savefig(scatter_path, dpi=300); plt.close()
                    print(f"Saved sweep genic vs intergenic scatter for min_alt={thr} to {scatter_path}")

    # Save results table
    # Step 4: Create output directory
    output_dir = args.output_prefix
    os.makedirs(output_dir, exist_ok=True)
    print(f"Created output directory: {output_dir}")
    
    # Step 5: Save results to file
    out_table = os.path.join(output_dir, "simple_rates.tsv")
    df.to_csv(out_table, sep="\t", index=False)
    print(f"Simple rates saved to {out_table}")

    # Step 6: Create ranking plots with enhanced data
    df_ranked = create_ranking_plots(df, os.path.join(output_dir, "simple_rates"))
    
    # Save individual simple rate plots immediately
    save_individual_simple_plots(df_ranked, os.path.join(output_dir, "simple_rates"))
    
    # GLM plotting removed - only site-level GLM plotting remains
    
    # Print summary statistics
    print("\n=== SUMMARY STATISTICS ===")
    print(f"Total samples analyzed: {len(df)}")
    print(f"Controls: {df['is_control'].sum()}")
    print(f"Samples: {(~df['is_control']).sum()}")
    
    # Check for depth bias
    print("\n=== DEPTH BIAS ANALYSIS ===")
    if 'mean_coverage' in df.columns and 'high_rate' in df.columns:
        # Calculate correlation between depth and mutation rate
        valid_mask = df['mean_coverage'].notna() & df['high_rate'].notna()
        if valid_mask.sum() > 3:
            correlation, p_value = stats.pearsonr(df.loc[valid_mask, 'mean_coverage'], 
                                                df.loc[valid_mask, 'high_rate'])
            print(f"Depth vs High Rate correlation: {correlation:.3f} (p={p_value:.3f})")
            
            if abs(correlation) > 0.3 and p_value < 0.05:
                print("⚠️  WARNING: Significant depth bias detected!")
                print("   Consider using depth-based GLM or alpha correction")
            else:
                print("✅ No significant depth bias detected")
    
    # Site-level GLM analysis summary
    if 'site_glm_rate' in df.columns and df['site_glm_rate'].notna().any():
        print("✅ Site-level GLM analysis completed")
    
    if alpha > 0:
        print(f"Alpha correction applied: {alpha:.2e}")
    
    if args.site_glm:
        print(f"Site-level GLM analysis: {args.site_glm_method} method")
    else:
        print("Site-level GLM analysis: disabled (use --site-glm to enable)")
    
    # Optional: Category comparison analysis
    if args.compare_categories:
        if not args.gff_file or not args.genome_fasta:
            print("Warning: --compare-categories requires --gff-file and --genome-fasta")
            print("Skipping category comparison analysis")
        else:
            print("\n" + "="*80)
            print("RUNNING CATEGORY COMPARISON ANALYSIS")
            print("="*80)
            print("Loading site-level data with category annotations...")
            try:
                category_df = load_site_level_with_category(
                    args.counts_dir, args.gff_file, args.genome_fasta, args.exclusion_mask, args.codon_table_path
                )
                if category_df is not None and not category_df.empty:
                    category_dir = os.path.join(output_dir, "category_comparison")
                    os.makedirs(category_dir, exist_ok=True)
                    
                    # Fit negative binomial GLM for category rates
                    print("Fitting negative binomial GLM for category rates...")
                    # NB alpha will be estimated from all data (controls + treated) in the function
                    nb_result, nb_df = fit_category_rates_nb_glm(
                        category_df,
                        use_treatment_covariate=use_treatment_covariate,
                        nb_alpha=None  # Let it estimate from all data
                    )
                    
                    if nb_result is not None:
                        # Extract and save results
                        beta0 = nb_result.params.get('const', np.nan)
                        beta_treatment = nb_result.params.get('treatment', 0.0) if use_treatment_covariate else 0.0
                        beta_syn = nb_result.params.get('cat_synonymous', 0.0)
                        beta_nonsyn = nb_result.params.get('cat_non_synonymous', 0.0)
                        # Interaction terms (allow category effects to differ between controls and treated)
                        beta_treatment_syn = nb_result.params.get('treatment_x_cat_synonymous', 0.0) if use_treatment_covariate else 0.0
                        beta_treatment_nonsyn = nb_result.params.get('treatment_x_cat_non_synonymous', 0.0) if use_treatment_covariate else 0.0
                        
                        # Get variance-covariance matrix for proper CI calculation
                        try:
                            vcov = nb_result.cov_params()
                        except Exception:
                            vcov = None
                        
                        # Helper function to calculate variance of linear predictor (same as in compare_category_rates)
                        def calc_var_linear_simple(cat, is_control, vcov):
                            """Calculate variance of linear predictor for a given category and treatment."""
                            if vcov is None:
                                return np.nan
                            var = 0.0
                            if 'const' in vcov.index:
                                var += vcov.loc['const', 'const']
                            if not is_control and use_treatment_covariate and 'treatment' in vcov.index:
                                var += vcov.loc['treatment', 'treatment']
                                if 'const' in vcov.index:
                                    var += 2 * vcov.loc['const', 'treatment']
                            if cat == 'synonymous' and 'cat_synonymous' in vcov.index:
                                var += vcov.loc['cat_synonymous', 'cat_synonymous']
                                if 'const' in vcov.index:
                                    var += 2 * vcov.loc['const', 'cat_synonymous']
                                if not is_control and use_treatment_covariate and 'treatment' in vcov.index:
                                    var += 2 * vcov.loc['treatment', 'cat_synonymous']
                                if not is_control and use_treatment_covariate and 'treatment_x_cat_synonymous' in vcov.index:
                                    var += vcov.loc['treatment_x_cat_synonymous', 'treatment_x_cat_synonymous']
                                    if 'const' in vcov.index:
                                        var += 2 * vcov.loc['const', 'treatment_x_cat_synonymous']
                                    if 'treatment' in vcov.index:
                                        var += 2 * vcov.loc['treatment', 'treatment_x_cat_synonymous']
                                    var += 2 * vcov.loc['cat_synonymous', 'treatment_x_cat_synonymous']
                            elif cat == 'non_synonymous' and 'cat_non_synonymous' in vcov.index:
                                var += vcov.loc['cat_non_synonymous', 'cat_non_synonymous']
                                if 'const' in vcov.index:
                                    var += 2 * vcov.loc['const', 'cat_non_synonymous']
                                if not is_control and use_treatment_covariate and 'treatment' in vcov.index:
                                    var += 2 * vcov.loc['treatment', 'cat_non_synonymous']
                                if not is_control and use_treatment_covariate and 'treatment_x_cat_non_synonymous' in vcov.index:
                                    var += vcov.loc['treatment_x_cat_non_synonymous', 'treatment_x_cat_non_synonymous']
                                    if 'const' in vcov.index:
                                        var += 2 * vcov.loc['const', 'treatment_x_cat_non_synonymous']
                                    if 'treatment' in vcov.index:
                                        var += 2 * vcov.loc['treatment', 'treatment_x_cat_non_synonymous']
                                    var += 2 * vcov.loc['cat_non_synonymous', 'treatment_x_cat_non_synonymous']
                            return var if np.isfinite(var) and var > 0 else np.nan
                        
                        # Calculate rates for each category with proper interaction terms
                        categories = ['intergenic', 'synonymous', 'non_synonymous']
                        summary_rows = []
                        
                        for cat in categories:
                            for is_control in [True, False]:
                                # Calculate linear predictor with interactions:
                                # Controls: β₀ + β_cat·cat
                                # Treated: β₀ + β_treatment + β_cat·cat + β_treatment_cat·cat
                                linear_pred = beta0
                                
                                # Add category effect
                                if cat == 'synonymous' and np.isfinite(beta_syn):
                                    linear_pred += beta_syn
                                elif cat == 'non_synonymous' and np.isfinite(beta_nonsyn):
                                    linear_pred += beta_nonsyn
                                
                                # Add treatment effect and interaction if treated
                                if not is_control and use_treatment_covariate:
                                    if np.isfinite(beta_treatment):
                                        linear_pred += beta_treatment
                                    # Add interaction term (allows category effect to differ in treated)
                                    if cat == 'synonymous' and np.isfinite(beta_treatment_syn):
                                        linear_pred += beta_treatment_syn
                                    elif cat == 'non_synonymous' and np.isfinite(beta_treatment_nonsyn):
                                        linear_pred += beta_treatment_nonsyn
                                
                                # Calculate rate: exp(linear_pred)
                                rate = np.exp(linear_pred) if np.isfinite(linear_pred) else np.nan
                                
                                # Calculate proper CI using delta method
                                var_linear = calc_var_linear_simple(cat, is_control, vcov)
                                se_linear = np.sqrt(var_linear) if np.isfinite(var_linear) and var_linear > 0 else np.nan
                                z = 1.96
                                if np.isfinite(se_linear) and se_linear > 0 and np.isfinite(rate):
                                    ci_low = rate * np.exp(-z * se_linear)
                                    ci_high = rate * np.exp(z * se_linear)
                                else:
                                    ci_low, ci_high = np.nan, np.nan
                                
                                summary_rows.append({
                                    'category': cat,
                                    'treatment': 'control' if is_control else 'treated',
                                    'rate': rate,
                                    'CI_low': ci_low,
                                    'CI_high': ci_high,
                                })
                        
                        # Add coding category - calculate directly from site-level data (same as other categories)
                        for is_control in [True, False]:
                            treatment_mask = category_df['is_control'] if is_control else ~category_df['is_control']
                            subset_df = category_df[treatment_mask].copy()

                            # Get all coding sites (synonymous OR non_synonymous)
                            coding_mask = (subset_df['category'] == 'synonymous') | (subset_df['category'] == 'non_synonymous')
                            coding_df = subset_df[coding_mask]

                            if len(coding_df) > 0:
                                mutations_coding = coding_df['ems_count'].sum()
                                bases_coding = coding_df['depth'].sum()

                                if bases_coding > 0:
                                    coding_rate = mutations_coding / bases_coding
                                    
                                    # Calculate CI using Poisson approximation (same as simple rate)
                                    if mutations_coding > 0:
                                        se_log_rate = 1.0 / np.sqrt(mutations_coding)
                                        z = 1.96
                                        coding_ci_low = coding_rate * np.exp(-z * se_log_rate)
                                        coding_ci_high = coding_rate * np.exp(z * se_log_rate)
                                    else:
                                        coding_ci_low = 0.0
                                        coding_ci_high = 3.69 / bases_coding  # Poisson 95% CI for 0 events

                                    summary_rows.append({
                                        'category': 'coding',
                                        'treatment': 'control' if is_control else 'treated',
                                        'rate': coding_rate,
                                        'CI_low': coding_ci_low,
                                        'CI_high': coding_ci_high,
                                })
                        
                        summary_df = pd.DataFrame(summary_rows)
                        summary_path = os.path.join(category_dir, 'category_rates_nb_glm.tsv')
                        summary_df.to_csv(summary_path, sep='\t', index=False)
                        print(f"Negative binomial GLM results saved to {summary_path}")
                        
                        # Print summary
                        print("\nNegative Binomial GLM Results:")
                        print("-" * 70)
                        print(f"{'Category':<15} {'Treatment':<10} {'Rate':<15} {'CI_low':<15} {'CI_high':<15}")
                        print("-" * 70)
                        for _, row in summary_df.iterrows():
                            print(f"{row['category']:<15} {row['treatment']:<10} {row['rate']:<15.6e} {row['CI_low']:<15.6e} {row['CI_high']:<15.6e}")
                        print(f"\nModel AIC: {nb_result.aic:.2f}")
                        print(f"Model Log-likelihood: {nb_result.llf:.2f}")
                        
                        print("✓ Negative binomial GLM category analysis complete")
                    else:
                        print("⚠ Warning: Negative binomial GLM fitting failed")
                    
                    # Also run full comparison analysis
                    print("\nRunning full category comparison analysis...")
                    category_results = compare_category_rates(
                        category_df, category_dir, 
                        method="negative_binomial",
                        use_treatment_covariate=use_treatment_covariate,
                        alpha=alpha if not use_treatment_covariate else 0.0,
                        use_control_baseline=True,  # Use control-baseline approach to test if category matters
                        kmer5_model_path=args.kmer5_model_path,  # 5mer normalization for sequence context
                        genome_fasta=args.genome_fasta
                    )
                    if category_results:
                        print("✓ Category comparison analysis complete")
                    else:
                        print("⚠ Warning: Category comparison failed")
                else:
                    print("⚠ Warning: No site-level data loaded for category comparison")
            except Exception as e:
                print(f"⚠ Warning: Category comparison analysis failed: {e}")
                import traceback
                traceback.print_exc()
    
    # Optional: Treatment day comparison analysis
    if args.compare_treatment_days:
        if not args.gff_file or not args.genome_fasta:
            print("Warning: --compare-treatment-days requires --gff-file and --genome-fasta")
            print("Skipping treatment day comparison analysis")
        else:
            print("\n" + "="*80)
            print("RUNNING TREATMENT DAY COMPARISON ANALYSIS")
            print("="*80)
            print("Loading site-level data with treatment day annotations...")
            try:
                # Load site-level data with categories (which includes treatment_day)
                category_df = load_site_level_with_category(
                    args.counts_dir, args.gff_file, args.genome_fasta, args.exclusion_mask, args.codon_table_path
                )
                if category_df is not None and not category_df.empty:
                    treatment_day_dir = os.path.join(output_dir, "treatment_day_comparison")
                    os.makedirs(treatment_day_dir, exist_ok=True)
                    
                    # Run treatment day comparison
                    print("Comparing mutation rates across treatment days...")
                    treatment_day_results = compare_treatment_day_rates(
                        category_df, treatment_day_dir,
                        method="negative_binomial",
                        alpha=alpha if not use_treatment_covariate else 0.0
                    )
                    if treatment_day_results:
                        print("✓ Treatment day comparison analysis complete")
                    else:
                        print("⚠ Warning: Treatment day comparison failed")
                else:
                    print("⚠ Warning: No site-level data loaded for treatment day comparison")
            except Exception as e:
                print(f"⚠ Warning: Treatment day comparison analysis failed: {e}")
                import traceback
                traceback.print_exc()
    
    # Optional: 5mer-normalized window analysis
    if getattr(args, 'kmer5_normalized_windows', False):
        if not args.genome_fasta:
            print("Warning: --kmer5-normalized-windows requires --genome-fasta")
            print("Skipping 5mer-normalized window analysis")
        elif not getattr(args, 'kmer5_model_path', None):
            print("Warning: --kmer5-normalized-windows requires --kmer5-model-path")
            print("Skipping 5mer-normalized window analysis")
        else:
            print("\n" + "="*80)
            print("RUNNING 5MER-NORMALIZED WINDOW ANALYSIS")
            print("="*80)
            try:
                # Load 5mer model
                print(f"Loading 5mer model from {args.kmer5_model_path}...")
                model_5mer = load_5mer_model(args.kmer5_model_path)
                print("✓ 5mer model loaded")
                
                # Create fixed-size windows
                excluded_sites_set = excluded_sites if excluded_sites else None
                windows_df = create_fixed_windows(
                    args.genome_fasta,
                    window_size=args.window_size,
                    exclusion_mask=excluded_sites_set
                )
                
                if windows_df is not None and not windows_df.empty:
                    # Load site-level data
                    print("Loading site-level mutation data...")
                    site_df = load_site_level_data_no_context(args.counts_dir, args.exclusion_mask)
                    
                    if site_df is not None and not site_df.empty:
                        # Add treatment column if not present
                        if 'treatment' not in site_df.columns:
                            site_df['treatment'] = (~site_df['is_control']).astype(int)
                        
                        # Calculate 5mer-normalized rates per window
                        windows_rates_df = calculate_window_rates_5mer_normalized(
                            windows_df,
                            site_df,
                            args.genome_fasta,
                            model_5mer,
                            method="negative_binomial",
                            use_treatment_covariate=use_treatment_covariate
                        )
                        
                        if windows_rates_df is not None and not windows_rates_df.empty:
                            # Save window rates
                            window_dir = os.path.join(output_dir, "5mer_normalized_windows")
                            os.makedirs(window_dir, exist_ok=True)
                            
                            windows_rates_path = os.path.join(window_dir, "window_rates.tsv")
                            windows_rates_df.to_csv(windows_rates_path, sep='\t', index=False)
                            print(f"Window rates saved to {windows_rates_path}")
                            
                            # Create Manhattan plot (using normalized rates)
                            manhattan_path = os.path.join(window_dir, "manhattan_plot.png")
                            plot_regional_rates_manhattan(
                                windows_rates_df,
                                manhattan_path,
                                use_treatment_covariate=use_treatment_covariate,
                                gff_file=args.gff_file if hasattr(args, 'gff_file') else None,
                                rate_threshold_multiplier=2.0
                            )
                            
                            # Regional comparison
                            regional_df = compare_regional_rates_gc_normalized(
                                windows_rates_df,
                                region_type="chromosome",
                                output_dir=window_dir,
                                method="negative_binomial",
                                use_treatment_covariate=use_treatment_covariate
                            )
                            
                            print("✓ 5mer-normalized window analysis complete")
                        else:
                            print("⚠ Warning: No window rates calculated")
                    else:
                        print("⚠ Warning: No site-level data loaded")
                else:
                    print("⚠ Warning: No fixed windows created")
            except Exception as e:
                print(f"⚠ Warning: 5mer-normalized window analysis failed: {e}")
                import traceback
                traceback.print_exc()
    
    if not df.empty:
        print(f"\nHigh Rate Range: {df['high_rate'].min():.2e} - {df['high_rate'].max():.2e}")
        print(f"Low Rate Range: {df['low_rate'].min():.2e} - {df['low_rate'].max():.2e}")
        
        if 'high_rate_alpha_corrected' in df.columns:
            print(f"Alpha-Corrected High Rate Range: {df['high_rate_alpha_corrected'].min():.2e} - {df['high_rate_alpha_corrected'].max():.2e}")
        
        if 'site_glm_rate' in df.columns and df['site_glm_rate'].notna().any():
            print(f"Site-Level GLM Rate Range: {df['site_glm_rate'].min():.2e} - {df['site_glm_rate'].max():.2e}")
        
        # Top 5 samples by high rate
        top_samples = df.nlargest(5, 'high_rate')[['label', 'high_rate', 'low_rate']]
        print(f"\nTop 5 samples by high rate:")
        for _, row in top_samples.iterrows():
            print(f"  {row['label']}: High={row['high_rate']:.2e}, Low={row['low_rate']:.2e}")
            
        # Top 5 samples by alpha-corrected rate (if available)
        if 'high_rate_alpha_corrected' in df.columns:
            top_corrected = df.nlargest(5, 'high_rate_alpha_corrected')[['label', 'high_rate_alpha_corrected']]
            print(f"\nTop 5 samples by alpha-corrected high rate:")
            for _, row in top_corrected.iterrows():
                print(f"  {row['label']}: Alpha-corrected={row['high_rate_alpha_corrected']:.2e}")


if __name__ == "__main__":
    main()
