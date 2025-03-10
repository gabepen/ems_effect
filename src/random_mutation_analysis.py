import argparse
from pathlib import Path
import json
import csv
from typing import Dict, List, Tuple, Set
import random
import numpy as np
from scipy import stats
from loguru import logger
from tqdm import tqdm
from Bio.Seq import Seq
from collections import defaultdict

from modules.parse import SeqContext
from plotting import plot_dnds_analysis  # Import from the main plotting module

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Generate and analyze randomized mutation datasets')
    parser.add_argument('-i', '--input_dir', required=True,
                        help='Directory containing mutation JSON files')
    parser.add_argument('-o', '--output', required=True,
                        help='Output directory')
    parser.add_argument('-c', '--config', required=True,
                        help='Path to config file with reference paths')
    parser.add_argument('-n', '--iterations', type=int, default=5,
                        help='Number of randomized datasets to generate per sample')
    parser.add_argument('-p', '--pattern', default='*.json',
                        help='Pattern to match mutation files (default: *.json)')
    return parser.parse_args()

def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    import yaml
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config

def setup_directories(output_dir: str) -> Dict[str, str]:
    """Create output directories."""
    paths = {
        'base': output_dir,
        'results': f"{output_dir}/results"
    }
    
    for path in paths.values():
        Path(path).mkdir(parents=True, exist_ok=True)
        
    return paths

def lookup_aa(codon: str, codon_table: Dict) -> str:
    """Look up amino acid for a given codon."""
    for aa in codon_table.keys():
        if aa != 'starts' and codon in codon_table[aa]:
            return aa
    return '@'  # Unknown/invalid codon

def randomize_mutations_preserve_type(
    mut_dict: Dict, 
    seqobject: SeqContext,
    features: Dict,
    codon_table: Dict
) -> Dict:
    """Generate one randomized mutation dataset preserving mutation types."""
    
    # Count total mutations by type
    mutations_by_type = {'C>T': 0, 'G>A': 0}
    for gene in mut_dict:
        for mut, count in mut_dict[gene]['mutations'].items():
            mut_type = mut.split('_')[1]
            if mut_type in mutations_by_type:
                mutations_by_type[mut_type] += 1
    
    # Collect all valid C/G sites
    valid_positions = {'C': [], 'G': []}
    
    for gene_id, feature in features.items():
        seq = str(seqobject.genome[feature.start:feature.end])
        if feature.strand == -1:
            seq = str(Seq(seq).reverse_complement())
        
        # Find start codon
        start = -1
        for i in range(len(seq)-2):
            if seq[i:i+3] in codon_table['starts']:
                start = i
                break
        
        if start == -1:
            continue
            
        # Process each codon
        for codon_pos in range(start, len(seq)-2, 3):
            codon = seq[codon_pos:codon_pos+3]
            if len(codon) != 3:
                continue
                
            # Check each position in codon
            for base_pos in range(3):
                abs_pos = codon_pos + base_pos
                ref_base = codon[base_pos]
                
                if ref_base not in ['C', 'G']:
                    continue
                    
                    genome_pos = feature.start + abs_pos
                if not seqobject.overlap_mask[genome_pos]:
                    continue
                
                # Classify site as synonymous or non-synonymous
                mutated = list(codon)
                mutated[base_pos] = 'T' if ref_base == 'C' else 'A'
                mutated = ''.join(mutated)
                
                ref_aa = lookup_aa(codon, codon_table)
                mut_aa = lookup_aa(mutated, codon_table)
                is_synonymous = (ref_aa == mut_aa)
                
                # Store position with classification
                valid_positions[ref_base].append((gene_id, abs_pos, genome_pos, is_synonymous))
    
    # Count sites by type
    syn_sites = {'C': sum(1 for p in valid_positions['C'] if p[3]), 
                 'G': sum(1 for p in valid_positions['G'] if p[3])}
    non_syn_sites = {'C': sum(1 for p in valid_positions['C'] if not p[3]), 
                     'G': sum(1 for p in valid_positions['G'] if not p[3])}
    
    total_syn = sum(syn_sites.values())
    total_non_syn = sum(non_syn_sites.values())
    
    # Generate randomized dataset
    new_mut_dict = {}
    
    # Track mutations placed
    placed_syn = {'C>T': 0, 'G>A': 0}
    placed_non_syn = {'C>T': 0, 'G>A': 0}
    
    # Place mutations by type, removing used positions
    for mut_type, count in mutations_by_type.items():
        ref = 'C' if mut_type == 'C>T' else 'G'
        
        if count > len(valid_positions[ref]):
            raise ValueError(f"Not enough {ref} positions ({len(valid_positions[ref])}) "
                           f"for requested mutations ({count})")
        
        for _ in range(count):
            if not valid_positions[ref]:
                break
                
            idx = random.randrange(len(valid_positions[ref]))
            gene_id, gene_pos, _, is_synonymous = valid_positions[ref].pop(idx)
            
            if gene_id not in new_mut_dict:
                new_mut_dict[gene_id] = {
                    'mutations': {},
                    'avg_cov': mut_dict.get(gene_id, {}).get('avg_cov', 30),
                    'gene_len': features[gene_id].end - features[gene_id].start
                }
            
            mut_key = f"{gene_pos}_{mut_type}"
            new_mut_dict[gene_id]['mutations'][mut_key] = \
                new_mut_dict[gene_id]['mutations'].get(mut_key, 0) + 1
                
            # Track mutation type
            if is_synonymous:
                placed_syn[mut_type] += 1
            else:
                placed_non_syn[mut_type] += 1
    
    # Store mutation classification for later use
    for gene_id in new_mut_dict:
        new_mut_dict[gene_id]['_syn_muts'] = {}
        new_mut_dict[gene_id]['_non_syn_muts'] = {}
        
    # Record which mutations are synonymous vs non-synonymous
    for gene_id in new_mut_dict:
        for mut_key, count in new_mut_dict[gene_id]['mutations'].items():
            pos, mut_type = mut_key.split('_')
            
            # Get gene sequence
            seq = str(seqobject.genome[features[gene_id].start:features[gene_id].end])
            if features[gene_id].strand == -1:
                seq = str(Seq(seq).reverse_complement())
            
            # Find start codon
            start = -1
            for i in range(len(seq)-2):
                if seq[i:i+3] in codon_table['starts']:
                    start = i
                    break
            
            if start == -1:
                continue
            
            # Get codon position
            pos = int(pos)
            codon_pos = (pos - start) // 3 * 3 + start
            base_pos = (pos - start) % 3
            
            if codon_pos + 2 >= len(seq):
                continue
            
            codon = seq[codon_pos:codon_pos+3]
            
            # Make mutation
            mutated = list(codon)
            ref_base = codon[base_pos]
            alt_base = 'T' if ref_base == 'C' else 'A'
            mutated[base_pos] = alt_base
            mutated = ''.join(mutated)
            
            # Check if synonymous
            ref_aa = lookup_aa(codon, codon_table)
            mut_aa = lookup_aa(mutated, codon_table)
            is_synonymous = (ref_aa == mut_aa)
            
            if is_synonymous:
                new_mut_dict[gene_id]['_syn_muts'][mut_key] = count
            else:
                new_mut_dict[gene_id]['_non_syn_muts'][mut_key] = count
    
    return new_mut_dict

def calculate_dnds_confidence_interval(
    syn_muts: int,
    non_syn_muts: int,
    syn_sites: int,
    non_syn_sites: int,
    confidence: float = 0.95
) -> Tuple[float, float, float]:
    """
    Calculate confidence interval for dN/dS ratio using bootstrapping.
    
    Args:
        syn_muts: Number of synonymous mutations
        non_syn_muts: Number of non-synonymous mutations
        syn_sites: Number of potential synonymous sites
        non_syn_sites: Number of potential non-synonymous sites
        confidence: Confidence level (default: 0.95 for 95% CI)
        
    Returns:
        Tuple containing:
        - dnds: Point estimate of dN/dS
        - lower_bound: Lower bound of confidence interval
        - upper_bound: Upper bound of confidence interval
    """
    # Calculate point estimate
    dn = non_syn_muts / non_syn_sites if non_syn_sites > 0 else 0
    ds = syn_muts / syn_sites if syn_sites > 0 else 0
    dnds = dn / ds if ds > 0 else 0
    
    # For small counts, use analytical approximation
    if syn_muts < 20 or non_syn_muts < 20:
        # Calculate standard errors using Poisson approximation
        se_syn = np.sqrt(syn_muts) / syn_sites if syn_muts > 0 else 0
        se_non_syn = np.sqrt(non_syn_muts) / non_syn_sites if non_syn_muts > 0 else 0
        
        # Calculate standard error of the ratio using delta method
        if ds > 0:
            se_ratio = dnds * np.sqrt((se_non_syn/dn)**2 + (se_syn/ds)**2)
            
            # Calculate confidence interval
            z_score = stats.norm.ppf((1 + confidence) / 2)
            lower_bound = max(0, dnds - z_score * se_ratio)
            upper_bound = dnds + z_score * se_ratio
        else:
            # If ds is zero, can't calculate CI analytically
            lower_bound = 0
            upper_bound = float('inf')
    else:
        # For larger counts, use bootstrap resampling
        n_bootstrap = 1000
        bootstrap_dnds = []
        
        for _ in range(n_bootstrap):
            # Resample mutation counts
            syn_resample = np.random.poisson(syn_muts)
            non_syn_resample = np.random.poisson(non_syn_muts)
            
            # Calculate dN/dS for this resample
            dn_boot = non_syn_resample / non_syn_sites if non_syn_sites > 0 else 0
            ds_boot = syn_resample / syn_sites if syn_sites > 0 else 0
            dnds_boot = dn_boot / ds_boot if ds_boot > 0 else 0
            
            bootstrap_dnds.append(dnds_boot)
        
        # Calculate confidence interval from bootstrap distribution
        lower_percentile = (1 - confidence) / 2 * 100
        upper_percentile = (1 + confidence) / 2 * 100
        lower_bound = max(0, np.percentile(bootstrap_dnds, lower_percentile))
        upper_bound = np.percentile(bootstrap_dnds, upper_percentile)
    
    return dnds, lower_bound, upper_bound

def analyze_original_mutations(
    mut_dict: Dict,
    wolgenome: SeqContext,
    features: Dict,
    codon_table: Dict,
    syn_sites: int,
    non_syn_sites: int
) -> Dict:
    """Analyze original mutations to calculate dN/dS."""
    # Count synonymous and non-synonymous mutations
    syn_muts = 0  # Total synonymous mutation occurrences
    non_syn_muts = 0  # Total non-synonymous mutation occurrences
    total_muts = 0  # Total mutation occurrences
    
    # Track unique mutation sites
    syn_sites_mutated = set()  # Unique sites with synonymous mutations
    non_syn_sites_mutated = set()  # Unique sites with non-synonymous mutations
    
    # Track mutation types for debugging
    mutation_types = {}
    
    for gene_id, gene_data in mut_dict.items():
        if gene_id not in features:
            continue
            
        seq = str(wolgenome.genome[features[gene_id].start:features[gene_id].end])
        if features[gene_id].strand == -1:
            seq = str(Seq(seq).reverse_complement())
        
        # Find start codon
        start = -1
        for i in range(len(seq)-2):
            if seq[i:i+3] in codon_table['starts']:
                start = i
                break
        
        if start == -1:
            continue
            
        # Process each mutation
        for mut_key, count in gene_data['mutations'].items():
            # Parse mutation
            pos_str, mut_type = mut_key.split('_')
            pos = int(pos_str)
            
            # Track mutation types
            mutation_types[mut_type] = mutation_types.get(mut_type, 0) + 1
            
            # Filter for only EMS mutations (C>T and G>A)
            if mut_type not in ['C>T', 'G>A']:
                continue
                
            total_muts += count
            
            # Skip positions before start codon
            if pos < start:
                continue
                
            # Get codon position
            codon_pos = (pos - start) // 3 * 3 + start
            base_pos = (pos - start) % 3
            
            if codon_pos + 2 >= len(seq):
                continue
                
            codon = seq[codon_pos:codon_pos+3]
            
            # Make mutation
            mutated = list(codon)
            ref_base = codon[base_pos]
            
            # Extract alt_base from mutation type
            alt_base = mut_type.split('>')[1]
                
            mutated[base_pos] = alt_base
            mutated = ''.join(mutated)
            
            # Check if synonymous
            ref_aa = lookup_aa(codon, codon_table)
            mut_aa = lookup_aa(mutated, codon_table)
            is_synonymous = (ref_aa == mut_aa)
            
            # Create a unique site key
            site_key = f"{gene_id}_{pos}"
            
            # Track total mutation occurrences
            if is_synonymous:
                syn_muts += count
                syn_sites_mutated.add(site_key)
            else:
                non_syn_muts += count
                non_syn_sites_mutated.add(site_key)
    
    # Print mutation type statistics for debugging
    print(f"Mutation types found: {mutation_types}")
    print(f"Total EMS mutations (C>T, G>A): {total_muts}")
    
    # Calculate dN/dS using unique sites
    unique_syn = len(syn_sites_mutated)
    unique_non_syn = len(non_syn_sites_mutated)
    
    # Calculate dN/dS using unique sites
    dn = unique_non_syn / non_syn_sites if non_syn_sites > 0 else 0
    ds = unique_syn / syn_sites if syn_sites > 0 else 0
    dnds = dn / ds if ds > 0 else 0
    
    # Calculate confidence intervals for dN/dS
    if syn_muts < 20 or non_syn_muts < 20:
        # Calculate standard errors using Poisson approximation
        se_syn = np.sqrt(syn_muts) / syn_sites if syn_muts > 0 else 0
        se_non_syn = np.sqrt(non_syn_muts) / non_syn_sites if non_syn_muts > 0 else 0
        
        # Calculate standard error of the ratio using delta method
        if ds > 0:
            se_ratio = dnds * np.sqrt((se_non_syn/dn)**2 + (se_syn/ds)**2)
            
            # Calculate 95% confidence interval
            z_score = stats.norm.ppf(0.975)  # 95% CI
            lower_ci = max(0, dnds - z_score * se_ratio)
            upper_ci = dnds + z_score * se_ratio
        else:
            lower_ci = 0
            upper_ci = float('inf')
    else:
        # For larger counts, use bootstrap resampling
        n_bootstrap = 1000
        bootstrap_dnds = []
        
        for _ in range(n_bootstrap):
            # Resample mutation counts
            syn_resample = np.random.poisson(syn_muts)
            non_syn_resample = np.random.poisson(non_syn_muts)
            
            # Calculate dN/dS for this resample
            dn_boot = non_syn_resample / non_syn_sites if non_syn_sites > 0 else 0
            ds_boot = syn_resample / syn_sites if syn_sites > 0 else 0
            dnds_boot = dn_boot / ds_boot if ds_boot > 0 else 0
            
            bootstrap_dnds.append(dnds_boot)
        
        # Calculate confidence interval from bootstrap distribution
        lower_percentile = (1 - 0.95) / 2 * 100
        upper_percentile = (1 + 0.95) / 2 * 100
        lower_ci = max(0, np.percentile(bootstrap_dnds, lower_percentile))
        upper_ci = np.percentile(bootstrap_dnds, upper_percentile)
    
    return {
        'total_muts': total_muts,
        'syn_muts': syn_muts,
        'non_syn_muts': non_syn_muts,
        'unique_syn_sites': unique_syn,
        'unique_non_syn_sites': unique_non_syn,
        'dnds_raw': dnds,
        'dnds_lower_ci': lower_ci,
        'dnds_upper_ci': upper_ci
    }

def count_potential_sites(
    seqobject: SeqContext,
    features: Dict,
    codon_table: Dict
) -> Tuple[int, int]:
    """Count potential synonymous and non-synonymous sites in the genome."""
    syn_sites = 0
    non_syn_sites = 0
    
    for gene_id, feature in features.items():
        seq = str(seqobject.genome[feature.start:feature.end])
        if feature.strand == -1:
            seq = str(Seq(seq).reverse_complement())
        
        # Find start codon
        start = -1
        for i in range(len(seq)-2):
            if seq[i:i+3] in codon_table['starts']:
                start = i
                break
        
        if start == -1:
            continue
            
        # Process each codon
        for codon_pos in range(start, len(seq)-2, 3):
            codon = seq[codon_pos:codon_pos+3]
            if len(codon) != 3:
                continue
                
            # Check each position in codon
            for base_pos in range(3):
                abs_pos = codon_pos + base_pos
                ref_base = codon[base_pos]
                
                if ref_base not in ['C', 'G']:
                    continue
                    
                genome_pos = feature.start + abs_pos
                if not seqobject.overlap_mask[genome_pos]:
                    continue
                
                # Classify site as synonymous or non-synonymous
                mutated = list(codon)
                mutated[base_pos] = 'T' if ref_base == 'C' else 'A'
                mutated = ''.join(mutated)
                
                ref_aa = lookup_aa(codon, codon_table)
                mut_aa = lookup_aa(mutated, codon_table)
                is_synonymous = (ref_aa == mut_aa)
                
                if is_synonymous:
                    syn_sites += 1
                else:
                    non_syn_sites += 1
    
    return syn_sites, non_syn_sites

def calculate_dnds(
    mut_dict: Dict,
    syn_sites: int,
    non_syn_sites: int
) -> Dict:
    """Calculate dN/dS ratio from mutation dictionary."""
    # Count synonymous and non-synonymous mutations
    syn_muts = 0
    non_syn_muts = 0
    
    for gene_id, gene_data in mut_dict.items():
        if '_syn_muts' in gene_data:
            for mut_key, count in gene_data['_syn_muts'].items():
                syn_muts += count
        
        if '_non_syn_muts' in gene_data:
            for mut_key, count in gene_data['_non_syn_muts'].items():
                non_syn_muts += count
    
    # Calculate dN/dS
    dn = non_syn_muts / non_syn_sites if non_syn_sites > 0 else 0
    ds = syn_muts / syn_sites if syn_sites > 0 else 0
    dnds = dn / ds if ds > 0 else 0
    
    return {
        'syn_sites': syn_sites,
        'non_syn_sites': non_syn_sites,
        'syn_muts': syn_muts,
        'non_syn_muts': non_syn_muts,
        'dnds_raw': dnds
    }

def process_sample(
    sample_name: str,
    mut_file: Path,
    output_dir: Path,
    wolgenome: SeqContext,
    features: Dict,
    codon_table: Dict,
    syn_sites: int,
    non_syn_sites: int,
    iterations: int
) -> Dict:
    """Process a single sample and return its statistics."""
    # Create sample output directory
    sample_dir = output_dir / sample_name
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    # Load mutations
    with open(mut_file) as f:
        original_muts = json.load(f)
    
    # Analyze original mutations
    original_stats = analyze_original_mutations(
        original_muts,
        wolgenome,
        features,
        codon_table,
        syn_sites,
        non_syn_sites
    )
    
    # Generate and analyze randomized datasets
    all_genome_stats = []
    
    for i in range(iterations):
        # Generate randomized mutations
        random_muts = randomize_mutations_preserve_type(
            original_muts, 
            wolgenome,
            features,
            codon_table
        )
        
        # Calculate dN/dS
        genome_stats = calculate_dnds(random_muts, syn_sites, non_syn_sites)
        all_genome_stats.append(genome_stats)
    
    # Calculate random mean and confidence interval
    random_values = [stats['dnds_raw'] for stats in all_genome_stats]
    random_mean = np.mean(random_values)
    random_std = np.std(random_values)
    random_lower_ci = max(0, random_mean - 1.96 * random_std)
    random_upper_ci = random_mean + 1.96 * random_std
    
    # Calculate ratio and significance
    ratio = original_stats['dnds_raw'] / random_mean if random_mean > 0 else 0
    
    # Determine if original dN/dS is significantly different from random
    is_significant = (original_stats['dnds_raw'] < random_lower_ci or 
                     original_stats['dnds_raw'] > random_upper_ci)
    
    # Save results
    results = {
        'sample': sample_name,
        'original': original_stats,
        'random': {
            'mean': random_mean,
            'std': random_std,
            'lower_ci': random_lower_ci,
            'upper_ci': random_upper_ci,
            'values': random_values
        },
        'ratio': ratio,
        'significant': is_significant
    }
    
    with open(sample_dir / 'dnds_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return results

def main():
    args = parse_args()
    
    # Setup
    config = load_config(args.config)
    refs = config['references']
    paths = setup_directories(args.output)
    
    # Load codon table
    with open(refs['codon_table']) as f:
        codon_table = json.load(f)
    
    # Initialize genome context
    wolgenome = SeqContext(refs['genomic_fna'], refs['annotation'])
    features = wolgenome.genome_features()
    
    # Count potential sites once
    print("Counting potential mutation sites...")
    syn_sites, non_syn_sites = count_potential_sites(wolgenome, features, codon_table)
    print(f"Potential sites: Synonymous={syn_sites}, Non-synonymous={non_syn_sites}")
    
    # Find all mutation files
    input_dir = Path(args.input_dir)
    mutation_files = list(input_dir.glob(args.pattern))
    
    if not mutation_files:
        print(f"No mutation files found in {input_dir} matching pattern {args.pattern}")
        return
    
    print(f"Found {len(mutation_files)} mutation files to process")
    
    # Process each sample
    all_results = []
    
    for mut_file in tqdm(mutation_files, desc="Processing samples"):
        sample_name = mut_file.stem
        print(f"\nProcessing sample: {sample_name}")
        
        try:
            result = process_sample(
                sample_name,
                mut_file,
                Path(paths['results']),
                wolgenome,
                features,
                codon_table,
                syn_sites,
                non_syn_sites,
                args.iterations
            )
            all_results.append(result)
            
        except Exception as e:
            print(f"Error processing {sample_name}: {e}")
            continue
    
    # Generate summary table
    summary_rows = []
    
    for result in all_results:
        summary_rows.append({
            'Sample': result['sample'],
            'Total_Mutations': result['original']['total_muts'],
            'Syn_Mutations': result['original']['syn_muts'],
            'NonSyn_Mutations': result['original']['non_syn_muts'],
            'Original_dNdS': result['original']['dnds_raw'],
            'Original_dNdS_Lower_CI': result['original']['dnds_lower_ci'],
            'Original_dNdS_Upper_CI': result['original']['dnds_upper_ci'],
            'Random_dNdS': result['random']['mean'],
            'Random_dNdS_Lower_CI': result['random']['lower_ci'],
            'Random_dNdS_Upper_CI': result['random']['upper_ci'],
            'Ratio': result['ratio'],
            'Significant': 'Yes' if result['significant'] else 'No'
        })
    
    # Write summary CSV
    csv_path = f"{paths['results']}/dnds_summary.csv"
    with open(csv_path, 'w', newline='') as f:
        fieldnames = ['Sample', 'Total_Mutations', 'Syn_Mutations', 'NonSyn_Mutations', 
                     'Original_dNdS', 'Original_dNdS_Lower_CI', 'Original_dNdS_Upper_CI',
                     'Random_dNdS', 'Random_dNdS_Lower_CI', 'Random_dNdS_Upper_CI',
                     'Ratio', 'Significant']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)
    
    print(f"\nSummary table written to {csv_path}")
    
    # Also save as JSON for further processing
    with open(f"{paths['results']}/dnds_summary.json", 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Generate plots
    try:
        print("Generating plots...")
        plots_dir = f"{paths['results']}/plots"
        Path(plots_dir).mkdir(parents=True, exist_ok=True)
        
        plot_dnds_analysis(
            results_csv=csv_path,
            output_dir=plots_dir,
            title="dN/dS Analysis of EMS Mutations"
        )
        print(f"Plots saved to {plots_dir}")
    except Exception as e:
        print(f"Error generating plots: {e}")

if __name__ == '__main__':
    main() 