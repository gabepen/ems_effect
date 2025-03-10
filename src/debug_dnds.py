import argparse
from pathlib import Path
import json
from typing import Dict, List, Tuple, Set
import random
from Bio import SeqIO
from Bio.Seq import Seq
from BCBio import GFF
from collections import defaultdict
import pandas as pd
from tqdm import tqdm
import yaml  # Add import at top
import gzip
from functools import partial
from mimetypes import guess_type

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Debug dN/dS calculation')
    parser.add_argument('-c', '--config', required=True,
                      help='Path to config file with reference paths')
    parser.add_argument('-o', '--output', required=True,
                      help='Output directory')
    parser.add_argument('-n', '--num_mutations', type=int, default=140000,
                      help='Number of random mutations to generate')
    return parser.parse_args()

def load_references(config_path: str) -> Dict[str, str]:
    """Load reference file paths from config."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config['references']

def load_genome(fasta_path: str) -> Seq:
    """Load genome sequence."""
    # Handle gzipped files
    encoding = guess_type(fasta_path)[1]
    _open = partial(gzip.open, mode='rt') if encoding == 'gzip' else open
    
    with _open(fasta_path) as f:
        for record in SeqIO.parse(f, 'fasta'):
            return record.seq
    raise ValueError("No sequence found in FASTA file")

def load_codon_table(table_path: str) -> Dict:
    """Load codon table."""
    with open(table_path) as f:
        return json.load(f)

def get_amino_acid(codon: str, codon_table: Dict) -> str:
    """Get amino acid for a codon."""
    for aa, codons in codon_table.items():
        if aa != 'starts' and codon in codons:
            return aa
    return 'X'  # Unknown/invalid codon

def classify_site(
    gene_seq: str,
    codon_pos: int,
    base_pos: int,
    codon_table: Dict
) -> Tuple[bool, str, str]:
    """Classify a site as synonymous or non-synonymous when mutated.
    
    Args:
        gene_seq: Full gene sequence
        codon_pos: Start position of codon in gene
        base_pos: Position within codon (0,1,2)
        codon_table: Codon to amino acid mapping
        
    Returns:
        Tuple of:
        - is_synonymous: True if mutation is synonymous
        - ref_aa: Reference amino acid
        - mut_aa: Mutated amino acid
    """
    codon = gene_seq[codon_pos:codon_pos+3]
    ref_base = codon[base_pos]
    
    if ref_base not in ['C', 'G']:
        return None, None, None
        
    # Make EMS mutation
    mutated = list(codon)
    mutated[base_pos] = 'T' if ref_base == 'C' else 'A'
    mutated = ''.join(mutated)
    
    ref_aa = get_amino_acid(codon, codon_table)
    mut_aa = get_amino_acid(mutated, codon_table)
    
    return ref_aa == mut_aa, ref_aa, mut_aa

def build_site_table(
    genome: Seq,
    gff_path: str,
    codon_table: Dict
) -> pd.DataFrame:
    """Build table of all C/G sites in genes with their classifications."""
    
    sites = []
    
    for rec in GFF.parse(gff_path):
        for feature in tqdm(rec.features):
            if feature.type != 'gene':
                continue
                
            gene_id = feature.qualifiers['Dbxref'][0].split(':')[-1]
            gene_seq = str(genome[feature.location.start:feature.location.end])
            
            if feature.strand == -1:
                gene_seq = str(Seq(gene_seq).reverse_complement())
            
            # Find start codon
            start = -1
            for i in range(len(gene_seq)-2):
                if gene_seq[i:i+3] in codon_table['starts']:
                    start = i
                    break
            
            if start == -1:
                continue
                
            # Process each codon
            for codon_pos in range(start, len(gene_seq)-2, 3):
                for base_pos in range(3):
                    abs_pos = codon_pos + base_pos
                    genome_pos = feature.location.start + abs_pos
                    
                    is_syn, ref_aa, mut_aa = classify_site(
                        gene_seq, codon_pos, base_pos, codon_table
                    )
                    
                    if is_syn is not None:  # Only store C/G sites
                        sites.append({
                            'gene_id': gene_id,
                            'genome_pos': genome_pos,
                            'gene_pos': abs_pos,
                            'codon_pos': base_pos,
                            'ref_base': gene_seq[abs_pos],
                            'is_synonymous': is_syn,
                            'ref_aa': ref_aa,
                            'mut_aa': mut_aa,
                            'strand': feature.strand
                        })
    
    return pd.DataFrame(sites)

def simulate_mutations(
    site_table: pd.DataFrame,
    num_mutations: int
) -> Dict:
    """Randomly select sites and calculate dN/dS."""
    
    # Get total sites
    total_syn = len(site_table[site_table.is_synonymous])
    total_non_syn = len(site_table[~site_table.is_synonymous])
    
    # Randomly select sites
    selected_sites = site_table.sample(n=num_mutations)
    
    # Count mutations
    syn_muts = len(selected_sites[selected_sites.is_synonymous])
    non_syn_muts = len(selected_sites[~selected_sites.is_synonymous])
    
    # Calculate dN/dS
    dn = non_syn_muts / total_non_syn if total_non_syn > 0 else 0
    ds = syn_muts / total_syn if total_syn > 0 else 0
    dnds = dn/ds if ds > 0 else 0
    
    return {
        'total_syn_sites': total_syn,
        'total_non_syn_sites': total_non_syn,
        'syn_mutations': syn_muts,
        'non_syn_mutations': non_syn_muts,
        'dn': dn,
        'ds': ds,
        'dnds': dnds
    }

def main():
    args = parse_args()
    
    # Create output directory
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load references
    refs = load_references(args.config)
    genome = load_genome(refs['genomic_fna'])
    codon_table = load_codon_table(refs['codon_table'])
    
    # Build site table
    print("Building site table...")
    site_table = build_site_table(genome, refs['annotation'], codon_table)
    
    # Save site table
    site_table.to_csv(out_dir / 'site_table.csv', index=False)
    
    print("\nSite Statistics:")
    print(f"Total C/G sites: {len(site_table)}")
    print(f"Synonymous sites: {len(site_table[site_table.is_synonymous])}")
    print(f"Non-synonymous sites: {len(site_table[~site_table.is_synonymous])}")
    
    # Run simulation
    print("\nSimulating random mutations...")
    results = simulate_mutations(site_table, args.num_mutations)
    
    print("\nSimulation Results:")
    for key, value in results.items():
        print(f"{key}: {value}")
        
    # Save results
    with open(out_dir / 'simulation_results.json', 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == '__main__':
    main() 