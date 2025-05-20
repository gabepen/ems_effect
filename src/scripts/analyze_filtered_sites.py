import os
import json
import argparse
from collections import defaultdict
from typing import Dict, Set, List, Tuple
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger
import yaml

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Analyze filtered positions across treated samples')
    parser.add_argument('-m', '--mpileups', required=True,
                       help='Directory containing mpileup files and filtered positions')
    parser.add_argument('-o', '--output', required=True,
                       help='Output directory for analysis results')
    parser.add_argument('-c', '--config', required=True,
                       help='Path to config file with reference paths')
    return parser.parse_args()

def load_config(config_path: str) -> Dict[str, str]:
    '''Load reference paths from config file.'''
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config['references']

def get_gene_lengths(gff_path: str) -> Dict[str, int]:
    """Extract gene lengths from GFF file.
    
    Returns:
        Dict mapping gene IDs (from Dbxref) to their lengths
    """
    gene_lengths = {}
    with open(gff_path) as f:
        for line in f:
            if line.startswith('#'):
                continue
            fields = line.strip().split('\t')
            if len(fields) < 9 or fields[2] != 'gene':
                continue
            
            # Extract attributes
            attrs = dict(
                attr.split('=', 1) 
                for attr in fields[8].split(';') 
                if '=' in attr
            )
                
            # Get reference ID from Dbxref
            ref_id = attrs['Dbxref'].split(':')[-1]
            
            start = int(fields[3])
            end = int(fields[4])
            gene_lengths[ref_id] = end - start + 1
    
    if not gene_lengths:
        logger.error(f"No gene lengths extracted from GFF file: {gff_path}")
    else:
        logger.info(f"Extracted lengths for {len(gene_lengths)} genes")
    
    return gene_lengths

def is_treated_sample(sample_name: str) -> bool:
    """Determine if a sample is treated with EMS based on its name.
    
    Treated samples contain 'EMS' in their name.
    """
    return 'EMS' in sample_name and '7d' not in sample_name and '7d' not in sample_name

def load_filtered_positions(mpileup_dir: str) -> Dict[str, Dict[str, Set[int]]]:
    """Load filtered positions for treated samples only.
    
    Returns:
        Dict mapping sample names to Dict mapping gene IDs to sets of filtered positions
    """
    filtered_positions = {}
    
    # Find all position files
    for filename in os.listdir(mpileup_dir):
        if filename.endswith('_positions.json'):
            sample = filename.replace('_positions.json', '')
            
            # Only process treated samples
            if not is_treated_sample(sample):
                continue
                
            filepath = os.path.join(mpileup_dir, filename)
            
            with open(filepath) as f:
                # Convert lists to sets for efficient operations
                sample_positions = {
                    gene: set(positions) 
                    for gene, positions in json.load(f).items()
                }
                filtered_positions[sample] = sample_positions
                
    if not filtered_positions:
        logger.warning("No treated samples found! Check that sample names contain 'EMS'")
    else:
        logger.info(f"Found {len(filtered_positions)} treated samples")
    
    return filtered_positions

def analyze_filtered_sites(
    filtered_positions: Dict[str, Dict[str, Set[int]]],
    gene_lengths: Dict[str, int]
) -> Dict:
    """Analyze intersection and union of filtered sites across treated samples."""
    
    # Get all unique genes and log some examples
    all_genes = set()
    for sample_data in filtered_positions.values():
        all_genes.update(sample_data.keys())
    
    # Debug: Print first few entries from filtered positions
    logger.debug("First few entries from filtered_positions:")
    sample = next(iter(filtered_positions))
    for gene_id in list(filtered_positions[sample].keys())[:5]:
        logger.debug(f"  {gene_id}")
    
    logger.debug(f"Example genes from filtered positions: {list(all_genes)[:5]}")
    logger.debug(f"Example genes from gene_lengths: {list(gene_lengths.keys())[:5]}")
    
    # Check gene ID overlap
    genes_not_found = [g for g in all_genes if g not in gene_lengths]
    if genes_not_found:
        logger.warning(f"{len(genes_not_found)} genes not found in GFF file")
        logger.debug(f"First few missing genes: {genes_not_found[:5]}")
    
    results = {
        'per_gene': {},
        'summary': {
            'total_genes': len(all_genes),
            'treated_samples': len(filtered_positions),
            'genes_with_filtered_sites': 0,
            'total_unique_filtered_sites': 0,
            'samples': list(filtered_positions.keys())
        }
    }
    
    # Analyze each gene
    for gene in all_genes:
        gene_length = gene_lengths.get(gene, 0)
        if gene_length == 0:
            logger.warning(f"Gene {gene} not found in GFF file")
            continue
            
        gene_results = {
            'samples_with_filtered_sites': 0,
            'filtered_sites_per_sample': {},
            'percent_coverage_per_sample': {},  # Percentage of gene length filtered per sample
            'union_size': 0,
            'intersection_size': 0,
            'unique_sites': 0,
            'gene_length': gene_length,
            'merged_coverage_percent': 0.0,  # Percentage after merging all samples
            'avg_coverage_percent': 0.0  # Average coverage across samples
        }
        
        # Get samples that have filtered sites for this gene
        samples_with_gene = [
            sample for sample, data in filtered_positions.items()
            if gene in data and data[gene]
        ]
        
        if samples_with_gene:
            gene_results['samples_with_filtered_sites'] = len(samples_with_gene)
            
            # Calculate coverage per sample
            coverage_percentages = []
            for sample in samples_with_gene:
                sites = filtered_positions[sample].get(gene, set())
                num_sites = len(sites)
                coverage_percent = (num_sites / gene_length) * 100
                
                gene_results['filtered_sites_per_sample'][sample] = num_sites
                gene_results['percent_coverage_per_sample'][sample] = coverage_percent
                coverage_percentages.append(coverage_percent)
            
            # Calculate average coverage across samples
            gene_results['avg_coverage_percent'] = sum(coverage_percentages) / len(coverage_percentages)
            
            # Calculate merged coverage
            all_sites = set().union(*(
                filtered_positions[sample][gene]
                for sample in samples_with_gene
            ))
            
            common_sites = set.intersection(*(
                filtered_positions[sample][gene]
                for sample in samples_with_gene
            ))
            
            gene_results['union_size'] = len(all_sites)
            gene_results['intersection_size'] = len(common_sites)
            gene_results['unique_sites'] = len(all_sites)
            gene_results['merged_coverage_percent'] = (len(all_sites) / gene_length) * 100
            
            # Update summary stats
            if gene_results['unique_sites'] > 0:
                results['summary']['genes_with_filtered_sites'] += 1
                results['summary']['total_unique_filtered_sites'] += gene_results['unique_sites']
        
        results['per_gene'][gene] = gene_results
    
    return results

def plot_filtered_sites_distribution(results: Dict, output_dir: str):
    """Create visualizations of filtered sites distribution."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot 1: Distribution of unique filtered sites per gene
    unique_sites = [
        data['unique_sites'] 
        for data in results['per_gene'].values()
        if data['unique_sites'] > 0
    ]
    
    plt.figure(figsize=(10, 6))
    sns.histplot(unique_sites, bins=50)
    plt.title('Distribution of Unique Filtered Sites per Gene\n(Treated Samples Only)')
    plt.xlabel('Number of Filtered Sites')
    plt.ylabel('Count')
    plt.savefig(os.path.join(output_dir, 'filtered_sites_distribution.png'))
    plt.close()
    
    # Plot 2: Distribution of average coverage percentage
    avg_coverages = [
        data['avg_coverage_percent']
        for data in results['per_gene'].values()
        if data['unique_sites'] > 0
    ]
    
    plt.figure(figsize=(10, 6))
    sns.histplot(avg_coverages, bins=50)
    plt.title('Distribution of Average Coverage Percentage per Gene\n(Treated Samples Only)')
    plt.xlabel('Average Percentage of Gene Length Filtered')
    plt.ylabel('Count')
    plt.savefig(os.path.join(output_dir, 'average_coverage_distribution.png'))
    plt.close()
    
    # Plot 3: Distribution of merged coverage percentage
    merged_coverages = [
        data['merged_coverage_percent']
        for data in results['per_gene'].values()
        if data['unique_sites'] > 0
    ]
    
    plt.figure(figsize=(10, 6))
    sns.histplot(merged_coverages, bins=50)
    plt.title('Distribution of Merged Coverage Percentage per Gene\n(Treated Samples Only)')
    plt.xlabel('Percentage of Gene Length Filtered After Merging')
    plt.ylabel('Count')
    plt.savefig(os.path.join(output_dir, 'merged_coverage_distribution.png'))
    plt.close()

def main():
    args = parse_args()
    
    # Setup logging
    logger.add(os.path.join(args.output, "filtered_sites_analysis.log"))
    logger.info("Starting filtered sites analysis for treated samples")
    
    # Load config and get gene lengths
    refs = load_config(args.config)
    gene_lengths = get_gene_lengths(refs['annotation'])
    logger.info(f"Loaded lengths for {len(gene_lengths)} genes")
    
    # Load and analyze filtered positions
    filtered_positions = load_filtered_positions(args.mpileups)
    logger.info(f"Loaded filtered positions from {len(filtered_positions)} treated samples")
    
    # Analyze filtered sites
    results = analyze_filtered_sites(filtered_positions, gene_lengths)
    
    # Save results
    os.makedirs(args.output, exist_ok=True)
    with open(os.path.join(args.output, 'filtered_sites_analysis.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create visualizations
    plot_filtered_sites_distribution(results, args.output)
    
    # Log summary statistics
    logger.info("Analysis Summary:")
    logger.info(f"Total genes analyzed: {results['summary']['total_genes']}")
    logger.info(f"Genes with filtered sites: {results['summary']['genes_with_filtered_sites']}")
    logger.info(f"Total unique filtered sites: {results['summary']['total_unique_filtered_sites']}")
    logger.info(f"Treated samples analyzed: {results['summary']['treated_samples']}")

if __name__ == '__main__':
    main() 