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
    parser.add_argument('--c-only', action='store_true',
                        help='Only analyze C>T mutations')
    parser.add_argument('--g-only', action='store_true',
                        help='Only analyze G>A mutations')
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
        'results': f"{output_dir}/results",
        'logs': f"{output_dir}/logs"
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

def collect_valid_positions(
    seqobject: SeqContext,
    features: Dict,
    codon_table: Dict,
    c_only: bool = False,
    g_only: bool = False
) -> Dict:
    """Collect valid positions for EMS mutations."""
    
    # Initialize dictionaries based on mode
    bases_to_track = ['C'] if c_only else ['G'] if g_only else ['C', 'G']
    valid_positions = {base: [] for base in bases_to_track}
    codon_pos_counts = {base: {0: 0, 1: 0, 2: 0} for base in bases_to_track}
    syn_by_pos = {base: {0: 0, 1: 0, 2: 0} for base in bases_to_track}
    
    print("\nCollecting valid positions:")
    
    for gene_id, feature in features.items():
        seq = str(seqobject.genome[feature.start:feature.end])
        is_reverse = feature.strand == -1
        
        if is_reverse:
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
                
                # Skip if not tracking this base type
                if ref_base not in bases_to_track:
                    continue
                    
                genome_pos = feature.start + abs_pos
                if not seqobject.overlap_mask[genome_pos]:
                    continue
                
                # Count position in codon
                codon_pos_counts[ref_base][base_pos] += 1
                
                # Classify site
                mutated = list(codon)
                mutated[base_pos] = 'T' if ref_base == 'C' else 'A'
                mutated = ''.join(mutated)
                
                ref_aa = lookup_aa(codon, codon_table)
                mut_aa = lookup_aa(mutated, codon_table)
                is_synonymous = (ref_aa == mut_aa)
                
                if is_synonymous:
                    syn_by_pos[ref_base][base_pos] += 1
                
                valid_positions[ref_base].append((gene_id, abs_pos, genome_pos, is_synonymous))
    
    # Count sites by type
    syn_sites = sum(sum(1 for p in valid_positions[base] if p[3]) for base in bases_to_track)
    non_syn_sites = sum(sum(1 for p in valid_positions[base] if not p[3]) for base in bases_to_track)
    
    # Verify initial site distribution
    print("\nVerifying initial site distribution:")
    for base in bases_to_track:
        total_sites = len(valid_positions[base])
        syn_count = sum(1 for p in valid_positions[base] if p[3])
        non_syn_count = sum(1 for p in valid_positions[base] if not p[3])
        print(f"{base}: {total_sites} total sites ({syn_count} synonymous, {non_syn_count} non-synonymous)")
    
    return valid_positions, syn_sites, non_syn_sites

def randomize_mutations_preserve_type(
    mut_dict: Dict, 
    seqobject: SeqContext,
    features: Dict,
    codon_table: Dict,
    valid_positions: Dict,
    c_only: bool = False,
    g_only: bool = False
) -> Dict:
    """Generate one randomized mutation dataset preserving mutation types."""
    
    # Make deep copy of valid positions since we'll be modifying it
    valid_positions = {
        base: valid_positions[base][:] for base in valid_positions.keys()
    }
    
    # Count total mutations by type
    mutations_by_type = {'C>T': 0} if c_only else {'G>A': 0} if g_only else {'C>T': 0, 'G>A': 0}
    for gene in mut_dict:
        for mut, count in mut_dict[gene]['mutations'].items():
            mut_type = mut.split('_')[1]
            if mut_type in mutations_by_type:
                mutations_by_type[mut_type] += 1
    
    # Generate randomized dataset
    new_mut_dict = {}
    
    # Place mutations by type, removing used positions
    for mut_type, count in mutations_by_type.items():
        ref = 'C' if mut_type == 'C>T' else 'G'
        
        for _ in range(count):
            if len(valid_positions[ref]) == 0:
                logger.warning(f"Ran out of valid positions for {ref}")
                break
                
            idx = random.randrange(len(valid_positions[ref]))
            gene_id, gene_pos, _, is_synonymous = valid_positions[ref].pop(idx)
            
            if gene_id not in new_mut_dict:
                new_mut_dict[gene_id] = {
                    'mutations': {},
                    'avg_cov': mut_dict.get(gene_id, {}).get('avg_cov', 30),
                    'gene_len': features[gene_id].end - features[gene_id].start
                }
            
            mut_key = f"{gene_pos + 1}_{mut_type}"
            new_mut_dict[gene_id]['mutations'][mut_key] = \
                new_mut_dict[gene_id]['mutations'].get(mut_key, 0) + 1
    
    return new_mut_dict

def calculate_dnds_confidence_interval(
    syn_muts: int,
    non_syn_muts: int,
    syn_sites: int,
    non_syn_sites: int,
    confidence: float = 0.95,
    bootstrap_iterations: int = 1000
) -> Tuple[float, float, float]:
    """
    Calculate confidence interval for dN/dS ratio using bootstrapping.
    
    Args:
        syn_muts: Number of synonymous mutations
        non_syn_muts: Number of non-synonymous mutations
        syn_sites: Number of potential synonymous sites
        non_syn_sites: Number of potential non-synonymous sites
        confidence: Confidence level (default: 0.95 for 95% CI)
        bootstrap_iterations: Number of bootstrap iterations
        
    Returns:
        Tuple containing:
            - float: dN/dS ratio
            - float: Lower bound of confidence interval
            - float: Upper bound of confidence interval
    """
    # Calculate raw dN/dS
    if syn_muts == 0 or syn_sites == 0:
        return 0.0, 0.0, 0.0
    
    dn = non_syn_muts / non_syn_sites if non_syn_sites > 0 else 0
    ds = syn_muts / syn_sites if syn_sites > 0 else 0
    dnds = dn / ds if ds > 0 else 0
    
    # For small counts, use analytical approach based on Poisson distribution
    if syn_muts < 10 or non_syn_muts < 10:
        # Calculate standard errors for dN and dS
        se_dn = np.sqrt(non_syn_muts) / non_syn_sites if non_syn_muts > 0 and non_syn_sites > 0 else 0
        se_ds = np.sqrt(syn_muts) / syn_sites if syn_muts > 0 and syn_sites > 0 else 0
        
        # Calculate standard error for dN/dS using error propagation
        if ds > 0 and se_ds > 0 and se_dn > 0:
            se_dnds = dnds * np.sqrt((se_dn/dn)**2 + (se_ds/ds)**2)
            
            # Calculate confidence interval
            z_score = stats.norm.ppf(1 - (1 - confidence) / 2)
            lower_ci = max(0, dnds - z_score * se_dnds)
            upper_ci = dnds + z_score * se_dnds
            
            return dnds, lower_ci, upper_ci
    
    # For larger counts, use bootstrapping
    bootstrap_results = []
    
    for _ in range(bootstrap_iterations):
        # Resample mutation counts from Poisson distribution
        bootstrap_syn = np.random.poisson(syn_muts)
        bootstrap_non_syn = np.random.poisson(non_syn_muts)
        
        # Calculate dN/dS for this bootstrap sample
        bootstrap_dn = bootstrap_non_syn / non_syn_sites if non_syn_sites > 0 else 0
        bootstrap_ds = bootstrap_syn / syn_sites if syn_sites > 0 else 0
        bootstrap_dnds = bootstrap_dn / bootstrap_ds if bootstrap_ds > 0 else 0
        
        bootstrap_results.append(bootstrap_dnds)
    
    # Calculate confidence interval from bootstrap distribution
    lower_percentile = (1 - confidence) / 2 * 100
    upper_percentile = (1 - (1 - confidence) / 2) * 100
    
    lower_ci = max(0, np.percentile(bootstrap_results, lower_percentile))
    upper_ci = np.percentile(bootstrap_results, upper_percentile)
    
    return dnds, lower_ci, upper_ci

def analyze_mutations(
    mut_dict: Dict,
    wolgenome: SeqContext,
    features: Dict,
    codon_table: Dict,
    syn_sites: int,
    non_syn_sites: int,
    filter_value: str,
    mutkey_is_reverse: bool = False,
    c_only: bool = False,
    g_only: bool = False
) -> Dict:
    """Analyze original mutations to calculate dN/dS."""
    # Add strand-specific tracking
    strand_stats = {
        '+': {'syn': 0, 'non_syn': 0},
        '-': {'syn': 0, 'non_syn': 0}
    }
    
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
        
        strand = '+' if features[gene_id].strand == 1 else '-'
        
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
            
            # Skip if count is less than filter value
            if count < int(filter_value):
                continue
            
            # Parse mutation
            pos_str, mut_type = mut_key.split('_')
            pos = int(pos_str) - 1
        
            # Skip mutations based on mode
            if c_only and mut_type != 'C>T':
                continue
            if g_only and mut_type != 'G>A':
                continue
            
            # Track mutation types
            mutation_types[mut_type] = mutation_types.get(mut_type, 0) + 1
            
            if features[gene_id].strand == -1 and not mutkey_is_reverse:
                pos = len(seq) - pos - 1
                if mut_type == 'C>T':
                    mut_type = 'G>A'
                elif mut_type == 'G>A':
                    mut_type = 'C>T'
            
            # Filter for only EMS mutations (C>T and G>A)
            if mut_type not in ['C>T', 'G>A']:
                continue
                
            total_muts += count
            
            # Skip positions before start codon
            if pos < start:
                print(f"BEFORE START CODON")
                continue
                
            # Get codon position
            codon_pos = (pos - start) // 3 * 3 + start
            base_pos = (pos - start) % 3
            
            if codon_pos + 2 >= len(seq):
                print(f"AFTER END OF GENE")
                print(f"Gene: {gene_id}")
                print(f"Position: {pos}")
                print(f"Codon position: {codon_pos}")
                print(f"Base position: {base_pos}")
                print(f"Sequence: {seq}")
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
                strand_stats[strand]['syn'] += count
            else:
                non_syn_muts += count
                non_syn_sites_mutated.add(site_key)
                strand_stats[strand]['non_syn'] += count
    
    # Calculate dN/dS using unique sites
    unique_syn = len(syn_sites_mutated)
    unique_non_syn = len(non_syn_sites_mutated)
    
    # Calculate dN/dS using unique sites
    dn = unique_non_syn / non_syn_sites if non_syn_sites > 0 else None
    ds = unique_syn / syn_sites if syn_sites > 0 else None
    dnds = dn / ds if ds > 0 else None
    
    # Calculate confidence intervals using unique sites
    dnds, lower_ci, upper_ci = calculate_dnds_confidence_interval(
        len(syn_sites_mutated),      # Using unique sites
        len(non_syn_sites_mutated),  # Using unique sites
        syn_sites,
        non_syn_sites,
        confidence=0.95
    )
    
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

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, bool):  # Add handling for boolean values
            return str(obj)
        elif isinstance(obj, np.bool_):  # Add handling for boolean values
            return str(obj)
        return super(NumpyEncoder, self).default(obj)

def process_sample(
    sample_name: str,
    mut_file: Path,
    output_dir: Path,
    wolgenome: SeqContext,
    features: Dict,
    codon_table: Dict,
    syn_sites: int,
    non_syn_sites: int,
    iterations: int,
    filter_value: str,
    valid_positions: Dict,
    c_only: bool = False,
    g_only: bool = False
) -> Dict:
    """Process a single sample and return its statistics."""
    # Create sample output directory
    sample_dir = output_dir / sample_name
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    # Load mutations
    with open(mut_file) as f:
        original_muts = json.load(f)
    
    # Analyze original mutations
    original_stats = analyze_mutations(
        original_muts,
        wolgenome,
        features,
        codon_table,
        syn_sites,
        non_syn_sites,
        filter_value,
        False,
        c_only,
        g_only
    )
    
    # Generate and analyze randomized datasets
    all_genome_stats = []
    
    for i in range(iterations):
        # Generate randomized mutations
        random_muts = randomize_mutations_preserve_type(
            original_muts, 
            wolgenome,
            features,
            codon_table,
            valid_positions,
            c_only,
            g_only
        )
       
        # Analyze random mutations
        random_stats = analyze_mutations(
            random_muts,
            wolgenome,
            features,
            codon_table,
            syn_sites,
            non_syn_sites,
            0,
            True,
            c_only,
            g_only
        )
        
        all_genome_stats.append(random_stats)
    
    # Calculate random mean and confidence interval
    random_values = [stats['dnds_raw'] for stats in all_genome_stats]
    random_mean = np.mean(random_values)
    
    # Calculate confidence interval using mean mutation counts
    mean_syn_muts = np.mean([stats['syn_muts'] for stats in all_genome_stats])
    mean_non_syn_muts = np.mean([stats['non_syn_muts'] for stats in all_genome_stats])
    _, random_lower_ci, random_upper_ci = calculate_dnds_confidence_interval(
        mean_syn_muts,
        mean_non_syn_muts,
        syn_sites,
        non_syn_sites,
        confidence=0.95
    )
    
    # Calculate ratio and significance
    ratio = original_stats['dnds_raw'] / random_mean if random_mean > 0 else None
    is_significant = original_stats['dnds_raw'] < random_lower_ci or original_stats['dnds_raw'] > random_upper_ci
    
    # Prepare results dictionary
    results = {
        'sample': sample_name,
        'original': {
            'total_muts': original_stats['total_muts'],
            'syn_muts': original_stats['syn_muts'],
            'non_syn_muts': original_stats['non_syn_muts'],
            'unique_syn_sites': original_stats['unique_syn_sites'],
            'unique_non_syn_sites': original_stats['unique_non_syn_sites'],
            'dnds_raw': original_stats['dnds_raw'],
            'dnds_lower_ci': original_stats['dnds_lower_ci'],
            'dnds_upper_ci': original_stats['dnds_upper_ci']
        },
        'random': {
            'mean': random_mean,
            'lower_ci': random_lower_ci,
            'upper_ci': random_upper_ci,
            'values': random_values
        },
        'ratio': ratio,
        'significant': is_significant
    }
    
    # Save results
    with open(sample_dir / 'dnds_results.json', 'w') as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
        
    return results

def write_summary_csv(results, output_path):
    """Write summary of dN/dS analysis to CSV file."""
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Sample', 
            'Original_dNdS', 
            'Original_dNdS_Lower', 
            'Original_dNdS_Upper',
            'Random_dNdS', 
            'Random_dNdS_Lower', 
            'Random_dNdS_Upper',
            'Ratio', 
            'Significant',
            'Syn_Mutations',  # Changed from total to unique synonymous sites
            'NonSyn_Mutations',  # Changed from total to unique non-synonymous sites
            'Total_Mutations'  # Changed to sum of unique sites
        ])
        
        for result in results:
            writer.writerow([
                result['sample'],
                result['original']['dnds_raw'],
                result['original']['dnds_lower_ci'],
                result['original']['dnds_upper_ci'],
                result['random']['mean'],
                result['random']['lower_ci'],
                result['random']['upper_ci'],
                result['ratio'],
                result['significant'],
                result['original']['unique_syn_sites'],  # Changed from syn_muts to unique_syn_sites
                result['original']['unique_non_syn_sites'],  # Changed from non_syn_muts to unique_non_syn_sites
                result['original']['unique_syn_sites'] + result['original']['unique_non_syn_sites']  # Sum of unique sites
            ])

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
    valid_positions, syn_sites, non_syn_sites = collect_valid_positions(
        wolgenome, 
        features, 
        codon_table,
        args.c_only,
        args.g_only
    )
    print(f"Potential sites: Synonymous={syn_sites}, Non-synonymous={non_syn_sites}")
    
    # Find all mutation files
    input_dir = Path(args.input_dir)
    mutation_files = list(input_dir.glob(args.pattern))
    
    if not mutation_files:
        print(f"No mutation files found in {input_dir} matching pattern {args.pattern}")
        return
    
    print(f"Found {len(mutation_files)} mutation files to process")
    
    # Define output directory
    output_dir = Path(paths['results'])
    
    # Process all samples
    for filter_value in [0,2,3,5,7,10]:
        all_results = []
        for mut_file in tqdm(mutation_files, desc="Processing samples"):
            sample_name = mut_file.stem  # Extract sample name from filename
            try:
                result = process_sample(
                    sample_name,
                    mut_file,
                    output_dir,
                    wolgenome,
                    features,
                    codon_table,
                    syn_sites,
                    non_syn_sites,
                    args.iterations,
                    filter_value,
                    valid_positions,
                    args.c_only,
                    args.g_only
                )
                all_results.append(result)
            except Exception as e:
                print(f"Error processing {sample_name}: {e}")
                import traceback
                print(f"Traceback:\n{traceback.format_exc()}")
        
        # Write summary CSV
        csv_path = output_dir / f'dnds_summary_{filter_value}_debug.csv'
        write_summary_csv(all_results, str(csv_path))
        
        print(f"\nSummary table written to {csv_path}")
        
        # Also save as JSON for further processing
        with open(f"{paths['results']}/dnds_summary_{filter_value}_debug.json", 'w') as f:
            json.dump(all_results, f, indent=2, cls=NumpyEncoder)
        
        # Generate plots
        try:
            print("Generating plots...")
            plots_dir = f"{paths['results']}/plots_{filter_value}"
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