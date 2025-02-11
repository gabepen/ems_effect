import json
import argparse
from typing import Dict, Set, Any, Tuple, Counter
from pathlib import Path

def load_json(file_path: str) -> Dict[str, Any]:
    '''Load a JSON file and return its contents.
    
    Args:
        file_path (str): Path to JSON file
        
    Returns:
        Dict[str, Any]: Loaded JSON data
    '''
    with open(file_path, 'r') as file:
        return json.load(file)

def get_mutation_stats(mutation_data: Dict[str, Dict[str, Any]]) -> Tuple[Dict[str, Counter], Dict[str, int]]:
    '''Get mutation type frequencies and counts per gene.
    
    Args:
        mutation_data (Dict[str, Dict[str, Any]]): Mutation data from JSON
        
    Returns:
        Tuple containing:
            - Dict[str, Counter]: Mutation type frequencies per gene
            - Dict[str, int]: Total mutation count per gene
    '''
    mutation_types = {}  # Gene -> Counter of mutation types
    mutation_counts = {}  # Gene -> total count
    
    for gene in mutation_data:
        type_counter = Counter()
        total_count = 0
        
        for mutation in mutation_data[gene]['mutations']:
            mut_type = mutation.split('_')[1]  # Get mutation type (e.g., 'C>T')
            count = mutation_data[gene]['mutations'][mutation]
            type_counter[mut_type] += count
            total_count += count
            
        mutation_types[gene] = type_counter
        mutation_counts[gene] = total_count
        
    return mutation_types, mutation_counts

def compare_mutations(json1: str, json2: str, output_dir: str) -> None:
    '''Compare two mutation JSON files and output similarity metrics.
    
    Args:
        json1 (str): Path to first mutation JSON file
        json2 (str): Path to second mutation JSON file
        output_dir (str): Directory to write comparison results
    '''
    # Load both JSON files
    data1 = load_json(json1)
    data2 = load_json(json2)
    
    # Get sample names from filenames
    sample1 = Path(json1).stem
    sample2 = Path(json2).stem
    
    # Get mutation statistics
    types1, counts1 = get_mutation_stats(data1)
    types2, counts2 = get_mutation_stats(data2)
    
    # Get overall mutation type frequencies
    total_types1 = Counter()
    total_types2 = Counter()
    for gene in types1:
        total_types1.update(types1[gene])
    for gene in types2:
        total_types2.update(types2[gene])
    
    # Compare gene sets
    genes_in_both = set(counts1.keys()) & set(counts2.keys())
    genes_only_1 = set(counts1.keys()) - set(counts2.keys())
    genes_only_2 = set(counts2.keys()) - set(counts1.keys())
    
    # Find genes with significant differences
    count_differences = {}
    type_differences = {}
    for gene in genes_in_both:
        count_diff = abs(counts1[gene] - counts2[gene])
        if count_diff > 0:
            count_differences[gene] = {
                'sample1': counts1[gene],
                'sample2': counts2[gene],
                'diff': count_diff,
                'types1': dict(types1[gene]),
                'types2': dict(types2[gene])
            }
    
    # Calculate overall similarity metrics
    gene_similarity = len(genes_in_both) / len(set(counts1.keys()) | set(counts2.keys()))
    total_count1 = sum(counts1.values())
    total_count2 = sum(counts2.values())
    count_similarity = min(total_count1, total_count2) / max(total_count1, total_count2)
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Write summary report
    with open(Path(output_dir) / "comparison_summary.txt", 'w') as f:
        f.write(f"Mutation Comparison Summary\n")
        f.write(f"==========================\n\n")
        f.write(f"Sample 1: {sample1}\n")
        f.write(f"Sample 2: {sample2}\n\n")
        
        f.write(f"Overall Statistics:\n")
        f.write(f"-----------------\n")
        f.write(f"Total mutation count in {sample1}: {total_count1:,}\n")
        f.write(f"Total mutation count in {sample2}: {total_count2:,}\n")
        f.write(f"Count similarity ratio: {count_similarity:.3f}\n")
        f.write(f"Gene overlap ratio: {gene_similarity:.3f}\n\n")
        
        f.write(f"Mutation Type Frequencies:\n")
        f.write(f"-----------------------\n")
        f.write(f"{'Mutation Type':<10} {'Sample 1':<12} {'Sample 2':<12} {'Ratio':<8}\n")
        all_types = sorted(set(total_types1.keys()) | set(total_types2.keys()))
        for mut_type in all_types:
            count1 = total_types1[mut_type]
            count2 = total_types2[mut_type]
            ratio = count1/count2 if count2 != 0 else float('inf')
            f.write(f"{mut_type:<10} {count1:<12} {count2:<12} {ratio:.3f}\n")
        f.write("\n")
        
        f.write(f"Gene Coverage:\n")
        f.write(f"-------------\n")
        f.write(f"Genes in both samples: {len(genes_in_both):,}\n")
        f.write(f"Genes only in {sample1}: {len(genes_only_1):,}\n")
        f.write(f"Genes only in {sample2}: {len(genes_only_2):,}\n\n")
        
        if count_differences:
            f.write(f"Genes with Different Mutation Counts:\n")
            f.write(f"----------------------------------\n")
            f.write(f"{'Gene ID':<15} {'Sample 1':<10} {'Sample 2':<10} {'Difference':<10} {'Main Types':<20}\n")
            
            # Sort by difference magnitude
            for gene in sorted(count_differences, key=lambda x: count_differences[x]['diff'], reverse=True):
                diff = count_differences[gene]
                # Get most common mutation types
                main_types1 = sorted(diff['types1'].items(), key=lambda x: x[1], reverse=True)[:2]
                main_types2 = sorted(diff['types2'].items(), key=lambda x: x[1], reverse=True)[:2]
                main_types_str = f"{main_types1[0][0]}:{main_types1[0][1]}"
                
                f.write(f"{gene:<15} {diff['sample1']:<10} {diff['sample2']:<10} "
                       f"{diff['diff']:<10} {main_types_str:<20}\n")
        else:
            f.write("No significant differences in mutation counts found between samples.\n")

def main() -> None:
    parser = argparse.ArgumentParser(description='Compare two nucleotide mutation JSON files')
    parser.add_argument('-1', '--json1', required=True,
                      help='Path to first mutation JSON file')
    parser.add_argument('-2', '--json2', required=True,
                      help='Path to second mutation JSON file')
    parser.add_argument('-o', '--output', required=True,
                      help='Output directory for comparison results')
    
    args = parser.parse_args()
    compare_mutations(args.json1, args.json2, args.output)

if __name__ == '__main__':
    main() 