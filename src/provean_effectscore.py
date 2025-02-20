import sys
import os
import argparse
from typing import Dict, List, Any, Tuple
from argparse import Namespace
from pathlib import Path

# Add src to path 
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from modules import parse
from modules import translate
from mimetypes import guess_type 
from glob import glob
from tqdm import tqdm
import getopt
import json
import random
import subprocess
import time

def parse_args() -> Namespace:
    '''Parse command line arguments.
    
    Returns:
        Namespace: Parsed command line arguments containing:
            - mpileups (str): Path to mpileup files
            - output (str): Working directory path
            - exclude (bool): Flag for EMS mutations only
            - skip_parse (bool): Flag to skip parsing step
    '''
    parser = argparse.ArgumentParser(description='Process mpileup files and calculate PROVEAN effect scores')
    parser.add_argument('-m', '--mpileups', required=True,
                        help='Path to mpileup files')
    parser.add_argument('-o', '--output', required=True,
                        help='Working directory')
    parser.add_argument('-e', '--exclude', action='store_true',
                        help='EMS mutations only')
    parser.add_argument('-s', '--skip_parse', action='store_true',
                        help='Skip parsing step')
    return parser.parse_args()

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
        f"{outdir}/results"
    ]
    for directory in directories:
        if not os.path.isdir(directory):
            os.mkdir(directory)
    
    return {
        'mutpath': f"{outdir}/nuc_muts",
        'aapath': f"{outdir}/aa_muts",
        'provpath': f"{outdir}/provean_files"
    }

def process_mpileups(pile: str, mutpath: str, outdir: str, wolgenome: Any, ems_only: bool) -> None:
    '''Process mpileup files and generate nucleotide mutation JSONs.
    
    Args:
        pile (str): Path to directory containing mpileup files
        mutpath (str): Output directory path for mutation JSON files
        outdir (str): Main output directory for summary files
        wolgenome (SeqContext): Genome context object containing sequence and annotation info
        ems_only (bool): Flag to process only EMS mutations
    '''
    piles = glob(pile + '/*_variants.txt')
    base_counts: Dict[str, Dict[str, int]] = {}
    context_counts: Dict[str, Dict[str, int]] = {}  # Store contexts per sample
    
    for mpileup in piles: 
        samp = mpileup.split('/')[-1]
        sample = samp.split('.')[0]
        base_counts[sample] = {'A':0, 'T':0, 'G':0, 'C':0}
        context_counts[sample] = {}
        
        # Process mutations and count bases/contexts
        nuc_muts, contexts, intergenic_counts = parse.parse_mpile(
            mpileup, 
            wolgenome, 
            ems_only, 
            base_counts[sample],
            context_counts[sample]
        )
        
        # Save mutation data
        with open(f"{mutpath}/{sample}.json", 'w') as of:
            json.dump(nuc_muts, of)
    
    # Save summary files to main results directory
    with open(f"{outdir}/results/basecounts.json", 'w') as of:
        json.dump(base_counts, of)
        
    with open(f"{outdir}/results/contextcounts.json", 'w') as of:
        json.dump(context_counts, of)
        
    with open(f"{outdir}/results/intergeniccounts.json", 'w') as of:
        json.dump(intergenic_counts, of)

def process_mutations(
    nuc_mut_files: List[str],
    wolgenome: Any,
    codon_table: str,
    features: Dict[str, Any],
    paths: Dict[str, str]
) -> None:
    
    '''Convert nucleotide mutations to amino acid format and HGVS format.
    
    Args:
        nuc_mut_files (List[str]): List of paths to nucleotide mutation JSON files
        wolgenome (SeqContext): Genome context object containing sequence and annotation info
        codon_table (str): Path to codon table JSON file
        features (Dict[str, Any]): Dictionary mapping gene IDs to genome feature locations
        paths (Dict[str, str]): Dictionary containing paths for output files
            - 'aapath': Path for amino acid mutation files
            - 'provpath': Path for PROVEAN input files
            
    Returns:
        None: Results are written to JSON files in the specified output paths
    '''
    
    for js in nuc_mut_files:
        sample = js.split('/')[-1].split('.')[0]
        with open(js) as jf:
            mut_dict = json.load(jf)

            # Convert mutations to aa format and save to json
            aa_muts = translate.convert_mutations(mut_dict, wolgenome, codon_table, features)
            with open(f"{paths['aapath']}/{sample}.json", 'w') as of:
                json.dump(aa_muts, of)

            # Convert mutations to hgvs format and save to json 
            hgvs_dict = translate.prep_provean(aa_muts, wolgenome, sample, paths['provpath'], features)
            with open(f"{paths['provpath']}/{sample}.json", 'w') as of:
                json.dump(hgvs_dict, of)

def calculate_provean_scores(
    provean_jsons: List[str],
    prov_score_table_path: str,
    paths: Dict[str, str]
) -> Dict[str, Dict[str, Any]]:
    '''Calculate PROVEAN scores for mutations and normalize results.
    
    Args:
        provean_jsons (List[str]): List of paths to PROVEAN input JSON files
        prov_score_table_path (str): Path to PROVEAN score lookup table
        paths (Dict[str, str]): Dictionary containing output paths
            - provpath: Path for PROVEAN input files
            
    Returns:
        Dict[str, Dict[str, Any]]: Nested dictionary containing normalized scores:
            - First level keys are sample names
            - Second level keys are gene IDs
            - Values contain effect scores and metadata
    '''
    with open(prov_score_table_path) as jf:
        prov_score_table = json.load(jf)
    
    results: Dict[str, Dict[str, Any]] = {}
    
    for js in provean_jsons:
        sample = js.split('/')[-1].split('.')[0]
        with open(js) as jf:
            hgvsmut_dict = json.load(jf)
            
            if sample not in results:
                results[sample] = {}
                
            results = process_gene_mutations(
                sample, hgvsmut_dict, results, prov_score_table, 
                paths['provpath'], prov_score_table_path
            )
    
    return results

def process_gene_mutations(
    sample: str,
    hgvsmut_dict: Dict[str, Any],
    results: Dict[str, Dict[str, Any]],
    prov_score_table: Dict[str, Dict[str, float]],
    provpath: str,
    prov_score_table_path: str
) -> Dict[str, Dict[str, Any]]:
    '''Process mutations for each gene and calculate scores.
    
    Args:
        sample (str): Sample identifier
        hgvsmut_dict (Dict[str, Any]): Dictionary containing HGVS mutations by gene
        results (Dict[str, Dict[str, Any]]): Accumulated results dictionary
        prov_score_table (Dict[str, Dict[str, float]]): PROVEAN score lookup table
        provpath (str): Path to PROVEAN input files
        prov_score_table_path (str): Path to save updated score table
        
    Returns:
        Dict[str, Dict[str, Any]]: Updated results dictionary with new scores
    '''
    for gene in hgvsmut_dict:
        if gene not in results[sample]:
            results[sample][gene] = {
                'effect': 0,
                'gene_len': hgvsmut_dict[gene]['gene_len'],
                'avg_cov': hgvsmut_dict[gene]['avg_cov']
            }
        if gene not in prov_score_table:
            prov_score_table[gene] = {}
            
        results, prov_score_table = calculate_gene_scores(
            sample, gene, hgvsmut_dict, results, 
            prov_score_table, provpath
        )
        
    normalize_scores(sample, results)
    
    # Save updated score table
    with open(prov_score_table_path, 'w') as of:
        json.dump(prov_score_table, of)
        
    return results

def calculate_gene_scores(
    sample: str,
    gene: str,
    hgvsmut_dict: Dict[str, Any],
    results: Dict[str, Dict[str, Any]],
    prov_score_table: Dict[str, Dict[str, float]],
    provpath: str
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, float]]]:
    '''Calculate PROVEAN scores for a specific gene.
    
    Args:
        sample (str): Sample identifier
        gene (str): Gene identifier
        hgvsmut_dict (Dict[str, Any]): Dictionary containing mutations for the gene
        results (Dict[str, Dict[str, Any]]): Accumulated results dictionary
        prov_score_table (Dict[str, Dict[str, float]]): PROVEAN score lookup table
        provpath (str): Path to PROVEAN input files
        
    Returns:
        Tuple containing:
            - Dict[str, Dict[str, Any]]: Updated results dictionary
            - Dict[str, Dict[str, float]]: Updated PROVEAN score table
    '''
    var_path = f"{provpath}/{sample}_vars"
    new_variants = open(f"{var_path}/{gene}.new.var", 'w')
    no_score = False
    
    with open(f"{var_path}/{gene}.var", 'r') as var_file:
        for hgvs_mut in var_file:
            hgvs_mut = hgvs_mut.strip()
            if hgvs_mut in prov_score_table[gene]:
                score = float(prov_score_table[gene][hgvs_mut]) * hgvsmut_dict[gene]['mutations'][hgvs_mut]
                results[sample][gene]['effect'] += score
            else:
                no_score = True
                new_variants.write(f"{hgvs_mut}\n")
    new_variants.close()
    
    if no_score:
        results, prov_score_table = run_provean_calculation(
            sample, gene, var_path, results, 
            prov_score_table, hgvsmut_dict
        )
    
    return results, prov_score_table

def run_provean_calculation(
    sample: str,
    gene: str,
    var_path: str,
    results: Dict[str, Dict[str, Any]],
    prov_score_table: Dict[str, Dict[str, float]],
    hgvsmut_dict: Dict[str, Any]
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, float]]]:
    '''Run PROVEAN calculation for new variants.
    
    Args:
        sample (str): Sample identifier
        gene (str): Gene identifier
        var_path (str): Path to variant files
        results (Dict[str, Dict[str, Any]]): Accumulated results dictionary
        prov_score_table (Dict[str, Dict[str, float]]): PROVEAN score lookup table
        hgvsmut_dict (Dict[str, Any]): Dictionary containing mutations for the gene
        
    Returns:
        Tuple containing:
            - Dict[str, Dict[str, Any]]: Updated results dictionary
            - Dict[str, Dict[str, float]]: Updated PROVEAN score table
    '''
    print(f'Provean called. GID: {gene} Sample: {sample}')
    subprocess.run(["/home/gabe/wol_py/subp_provean.sh", gene, var_path])
    
    with open(f"{var_path}/{gene}.csv", 'r') as scores:
        for line in scores:
            hgvs_mut, score = line.strip().split(',')
            results[sample][gene]['effect'] += (float(score) * hgvsmut_dict[gene]['mutations'][hgvs_mut])
            prov_score_table[gene][hgvs_mut] = score
    
    return results, prov_score_table

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
    
    # Define reference paths
    genomic_fna = '/storage1/gabe/ems_effect_code/data/references/GCF_000008025.1_ASM802v1_genomic.fna.gz'
    annotation = '/storage1/gabe/ems_effect_code/data/references/GCF_000008025.1_ASM802v1_genomic.gff'
    codon_table = '/storage1/gabe/ems_effect_code/data/references/11.json'
    prov_score_table_path = '/storage1/gabe/ems_effect_code/data/provean_tables/score_table_2.json'
    
    # Initialize genome context
    wolgenome = parse.SeqContext(genomic_fna, annotation)
    gene_seqs = wolgenome.gene_seqs()
    features = wolgenome.genome_features()
    
    # Process mpileups if not skipped
    if not args.skip_parse:
        process_mpileups(args.mpileups, paths['mutpath'], args.output.rstrip('/'), wolgenome, args.exclude)
        print('nuc_mutation jsons generated')
        input()
    
    # Process mutations
    nuc_mut_files = glob(paths['mutpath'] + '/*.json')
    process_mutations(nuc_mut_files, wolgenome, codon_table, features, paths)
    
    # Calculate PROVEAN scores
    provean_jsons = glob(paths['provpath'] + '/*.json')
    results = calculate_provean_scores(provean_jsons, prov_score_table_path, paths)
    
    # Save final results
    with open(f"{args.output.rstrip('/')}/results/normalized_scores.json", 'w') as of:
        json.dump(results, of)

if __name__ == '__main__':
    main()