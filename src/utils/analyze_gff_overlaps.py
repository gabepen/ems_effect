from BCBio import GFF
import argparse
from collections import defaultdict
from typing import Dict, List, Tuple, Set
import matplotlib.pyplot as plt
from pathlib import Path

def analyze_overlaps(gff_file: str) -> Tuple[Dict, List[Tuple], int]:
    '''Analyze overlapping genes in GFF file.
    
    Args:
        gff_file: Path to GFF annotation file
        
    Returns:
        Tuple containing:
            - Dict: Statistics about overlaps
            - List[Tuple]: List of overlapping gene pairs
            - int: Total bases in overlapping regions
    '''
    overlap_stats = {
        'total_genes': 0,
        'overlapping_genes': 0,
        'total_overlap_bases': 0,
        'total_genic_bases': 0,  # Track total bases in genes
        'overlap_lengths': [],
        'genes_by_strand': {'+': 0, '-': 0},
        'overlaps_by_strand': {
            'same_strand': 0,
            'opposite_strand': 0
        }
    }
    
    # Store gene locations
    gene_locations = []
    overlapping_pairs = []
    
    # First pass - collect all gene positions and total genic bases
    genic_bases = set()  # Track all positions within genes
    with open(gff_file) as handle:
        for rec in GFF.parse(handle):
            for feat in rec.features:
                if feat.type == 'gene':
                    gene_id = feat.qualifiers['Dbxref'][0].split(':')[-1]
                    strand = '+' if feat.strand == 1 else '-'
                    overlap_stats['genes_by_strand'][strand] += 1
                    
                    # Add all positions in this gene to set
                    genic_bases.update(range(feat.location.start, feat.location.end))
                    
                    gene_locations.append((
                        feat.location.start,
                        feat.location.end,
                        gene_id,
                        strand
                    ))
    
    overlap_stats['total_genes'] = len(gene_locations)
    overlap_stats['total_genic_bases'] = len(genic_bases)
    
    # Sort by start position
    gene_locations.sort(key=lambda x: x[0])
    
    # Find overlapping genes
    overlapping_bases = set()
    overlapping_gene_ids = set()
    
    for i in range(len(gene_locations)-1):
        current = gene_locations[i]
        j = i + 1
        while j < len(gene_locations) and gene_locations[j][0] <= current[1]:
            next_gene = gene_locations[j]
            
            # Calculate overlap
            overlap_start = max(current[0], next_gene[0])
            overlap_end = min(current[1], next_gene[1])
            
            if overlap_end > overlap_start:  # Genes overlap
                overlap_length = overlap_end - overlap_start
                overlap_stats['overlap_lengths'].append(overlap_length)
                
                # Add overlapping bases to set
                overlapping_bases.update(range(overlap_start, overlap_end))
                
                # Track overlapping genes
                overlapping_gene_ids.add(current[2])
                overlapping_gene_ids.add(next_gene[2])
                
                # Track strand relationships
                if current[3] == next_gene[3]:
                    overlap_stats['overlaps_by_strand']['same_strand'] += 1
                else:
                    overlap_stats['overlaps_by_strand']['opposite_strand'] += 1
                
                overlapping_pairs.append((
                    current[2], 
                    next_gene[2], 
                    overlap_length,
                    f"{current[3]}/{next_gene[3]}"  # Strand orientation
                ))
            j += 1
    
    overlap_stats['overlapping_genes'] = len(overlapping_gene_ids)
    overlap_stats['total_overlap_bases'] = len(overlapping_bases)
    
    # Calculate percentage of genic bases that would be masked
    overlap_stats['percent_genic_bases_masked'] = (
        len(overlapping_bases) / overlap_stats['total_genic_bases'] * 100
        if overlap_stats['total_genic_bases'] > 0 else 0
    )
    
    return overlap_stats, overlapping_pairs, len(overlapping_bases)

def plot_overlap_distribution(overlap_lengths: List[int], output_dir: str):
    '''Create histogram of overlap lengths.'''
    plt.figure(figsize=(10, 6))
    plt.hist(overlap_lengths, bins=50)
    plt.xlabel('Overlap Length (bp)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Gene Overlap Lengths')
    plt.savefig(Path(output_dir) / 'gene_overlap_distribution.png')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Analyze gene overlaps in GFF file')
    parser.add_argument('-g', '--gff', required=True,
                      help='Path to GFF annotation file')
    parser.add_argument('-o', '--output', required=True,
                      help='Output directory for results')
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Analyze overlaps
    stats, pairs, total_overlap = analyze_overlaps(args.gff)
    
    # Print expanded summary
    print("\nGene Overlap Analysis:")
    print(f"Total genes: {stats['total_genes']}")
    print(f"Total bases within genes: {stats['total_genic_bases']}")
    print(f"Genes with overlaps: {stats['overlapping_genes']}")
    print(f"Total bases in overlapping regions: {total_overlap}")
    print(f"Percentage of genic bases masked: {stats['percent_genic_bases_masked']:.2f}%")
    print("\nStrand distribution:")
    print(f"Forward strand genes: {stats['genes_by_strand']['+']}")
    print(f"Reverse strand genes: {stats['genes_by_strand']['-']}")
    print("\nOverlap types:")
    print(f"Same strand overlaps: {stats['overlaps_by_strand']['same_strand']}")
    print(f"Opposite strand overlaps: {stats['overlaps_by_strand']['opposite_strand']}")
    
    # Save detailed results
    with open(output_dir / 'overlap_pairs.tsv', 'w') as f:
        f.write("Gene1\tGene2\tOverlap_Length\tStrand_Orientation\n")
        for pair in pairs:
            f.write(f"{pair[0]}\t{pair[1]}\t{pair[2]}\t{pair[3]}\n")
    
    # Plot overlap length distribution
    if stats['overlap_lengths']:
        plot_overlap_distribution(stats['overlap_lengths'], args.output)

if __name__ == '__main__':
    main() 