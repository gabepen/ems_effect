import os 
import json
import gzip
import multiprocessing as mp
from tqdm import tqdm
from functools import partial
from mimetypes import guess_type 
import numpy as np
from Bio import SeqIO
from BCBio import GFF
from modules.parse import SeqContext
from typing import Dict, List, Tuple, Optional, Any

def table_lookup(codon: str) -> str:
    '''Look up amino acid for a given codon using global codon table.
    
    Args:
        codon (str): Three letter nucleotide codon
        
    Returns:
        str: Single letter amino acid code, or '@' if codon not found
    '''
    for aa in table.keys():
        if codon in table[aa] and aa != 'starts':
            return aa
    return '@'

def ems_immune_codons(codon_table: Dict[str, List[str]]) -> List[str]:
    '''Find codons that only produce synonymous mutations from EMS changes.
    
    Args:
        codon_table: Dictionary mapping amino acids to their codons
        
    Returns:
        List[str]: List of codons that can only produce synonymous mutations 
                  from C>T or G>A changes
    '''
    # Create reverse lookup (codon -> amino acid)
    codon_to_aa = {}
    for aa, codons in codon_table.items():
        if aa != 'starts':  # Skip start codons list
            for codon in codons:
                codon_to_aa[codon] = aa
    
    immune_codons = []
    
    for codon in codon_to_aa:
        original_aa = codon_to_aa[codon]
        is_immune = True
        has_ems_site = False
        
        # Check each position in the codon
        for i, base in enumerate(codon):
            if base == 'C':
                has_ems_site = True
                # Test C>T mutation
                mutated = codon[:i] + 'T' + codon[i+1:]
                if mutated in codon_to_aa and codon_to_aa[mutated] != original_aa:
                    is_immune = False
                    break
            elif base == 'G':
                has_ems_site = True
                # Test G>A mutation
                mutated = codon[:i] + 'A' + codon[i+1:]
                if mutated in codon_to_aa and codon_to_aa[mutated] != original_aa:
                    is_immune = False
                    break
        
        if is_immune and has_ems_site:
            immune_codons.append(codon)
            
    return sorted(immune_codons)


def hgvs_notation(mut):
    '''
    Converts mutations from mut_jsons to hgvs notation
    '''
    pos = int(mut.split('_')[0]) + 1
    residues = mut.split('_')[1]
    ref = residues.split('>')[0]
    alt = residues.split('>')[1]
    return ref+str(pos)+alt
    
def single_flip(base: str) -> str:
    '''Convert nucleotide to its complement.
    
    Args:
        base (str): Single nucleotide character (A,T,C,G)
        
    Returns:
        str: Complementary nucleotide
    '''
    if base == 'A':
        return 'T'
    elif base == 'T':
        return 'A'
    elif base == 'C':
        return 'G'
    elif base == 'G':
        return 'C'

def translate(seq: str, fasta_bool: bool) -> Tuple[Optional[str], int]:
    '''Translate nucleotide sequence to amino acid sequence.
    
    Args:
        seq (str): Nucleotide sequence
        fasta_bool (bool): Whether to format output as FASTA with line breaks
        
    Returns:
        Tuple containing:
            - Optional[str]: Amino acid sequence, or None if no start codon found
            - int: Length of translated sequence, or 1 if translation failed
    '''
    aa_seq = ''
    start = -1
    
    # Find start codon
    for i in range(len(seq)):
        if seq[i:i+3] in table['starts']:
            start = i
            break
            
    if start == -1:
        print(seq)
        return None, 1
        
    # Split into codons and translate
    codons = []
    for pos1, pos2, pos3 in zip(seq[start::3], seq[start+1::3], seq[start+2::3]):
       codons.append(''.join([pos1, pos2, pos3]))
       
    line_count = 0
    length = 0
    for codon in codons:
        line_count += 1
        length += 1
        aa_seq += table_lookup(codon)
        if fasta_bool and line_count >= 50:
            aa_seq += '\n'
            line_count = 0

    return aa_seq, length

def convertor(seq: str, mutations: Dict[str, int], rc: bool, immune_codons: List[str]) -> Tuple[Dict[str, Any], int, Dict[str, float]]:
    '''Convert nucleotide mutations to amino acid mutations and calculate dN/dS ratios.'''
    
    # Initialize empty rates dictionary with all required keys
    empty_rates = {
        'raw': 0,
        'normalized': 0,
        'non_syn_sites': 0,
        'syn_sites': 0,
        'syn_sites_mutated': 0,
        'non_syn_sites_mutated': 0,
        'non_syn_muts': 0,
        'syn_muts': 0
    }
    
    # Find start codon
    start = -1
    for i in range(len(seq)):
        if seq[i:i+3] in table['starts']:
            start = i
            break
            
    if start == -1:
        print(seq)
        return None, 1, empty_rates
    
    # Count potential mutation sites
    non_syn_sites, syn_sites = count_ems_sites(seq, immune_codons)
    
    # Track unique mutated sites
    mutated_sites = set()  # Keep track of which sites have been mutated
    syn_sites_mutated = set()  # Sites with synonymous mutations
    non_syn_sites_mutated = set()  # Sites with non-synonymous mutations
    
    # Count observed mutations
    syn = 0  # Total synonymous mutations (including multiple at same site)
    nonsyn = 0  # Total non-synonymous mutations (including multiple at same site)
    aa_mutations = {}
    
    match = 0
    mis = 0
    
    # Split into codons
    codons = []
    for pos1, pos2, pos3 in zip(seq[start::3], seq[start+1::3], seq[start+2::3]):
       codons.append(''.join([pos1, pos2, pos3]))
       
    # Convert mutations
    for mut in mutations.keys():
        if rc:
            old_mut = mut.split('_')
            new_loc = len(seq) - int(old_mut[0])
            new_ref = single_flip(old_mut[1].split('>')[0])
            new_alt = single_flip(old_mut[1].split('>')[1])
            new_mut = str(new_loc+1) +'_'+new_ref+'>'+new_alt
            
        new_mut = mut.split('_')
        bp = (int(new_mut[0])-1) - start
        if bp < 1:  # Not in frame
            continue
            
        # Get reference base from the split mutation
        ref = new_mut[1].split('>')[0]
        
        try:
            if ref == seq[bp+start]:
                match += 1
            else:
                mis += 1
        except IndexError:
            print("BPIndexError")
            print(f"Sequence length: {len(seq)}")
            print(f"Position: {bp+start}")
            print(f"Mutation: {mut}")
            continue  # Skip this mutation and continue processing
            
        alt = new_mut[1].split('>')[1]
        aa_pos = (bp//3)
        
        try:
            codon = codons[aa_pos]
        except IndexError:
            return aa_mutations, 0, empty_rates
            
        new = codon[:(bp%3)] + alt + codon[(bp%3)+1:]
        ref_aa = table_lookup(codon)
        alt_aa = table_lookup(new)
        
        site_key = f"{aa_pos}_{bp%3}"  # Unique key for this mutation site
        
        if ref_aa == alt_aa:
            syn += 1
            if site_key not in mutated_sites:
                syn_sites_mutated.add(site_key)
                mutated_sites.add(site_key)
            continue
        else:
            nonsyn += 1
            if site_key not in mutated_sites:
                non_syn_sites_mutated.add(site_key)
                mutated_sites.add(site_key)
            
        aa_mut = str(aa_pos) + '_' + ref_aa + '>' + alt_aa
        if aa_mut in aa_mutations:
            aa_mutations[aa_mut] += mutations[mut]
        else:
            aa_mutations[aa_mut] = mutations[mut]

    # Filter low count mutations
    filtered = []
    for mut in aa_mutations:
        if aa_mutations[mut] < 1:
            filtered.append(mut)
    for mut in filtered:
        aa_mutations.pop(mut)
        
    # Check for genes with no synonymous sites (shouldn't happen)
    if syn_sites == 0:
        print(f"WARNING: Found gene with no potential synonymous sites.")
        print(f"Sequence length: {len(seq)}")
        print(f"Number of C/G sites: {sum(1 for base in seq if base in ['C', 'G'])}")
        print(f"Immune codons found: {sum(1 for i in range(0, len(seq)-2, 3) if seq[i:i+3] in immune_codons)}")
    
    dnds_rates = {
        'raw': 0, 
        'normalized': 0,
        'non_syn_sites': non_syn_sites,
        'syn_sites': syn_sites,
        'syn_sites_mutated': len(syn_sites_mutated),
        'non_syn_sites_mutated': len(non_syn_sites_mutated),
        'non_syn_muts': nonsyn,
        'syn_muts': syn
    }
    
    try:
        # Raw dN/dS - use minimum of 1 synonymous mutation
        dnds_rates['raw'] = nonsyn / max(syn, 1)
        
        # Normalized dN/dS - use minimum of 1 synonymous site
        dn = nonsyn / non_syn_sites if non_syn_sites > 0 else 0
        ds = syn / max(syn_sites, 1)  # Use at least 1 synonymous site
        dnds_rates['normalized'] = dn/ds if ds > 0 else 0
            
    except ZeroDivisionError:
        print(f"WARNING: Unexpected division by zero in dN/dS calculation")
        print(f"nonsyn: {nonsyn}, syn: {syn}")
        print(f"non_syn_sites: {non_syn_sites}, syn_sites: {syn_sites}")

    return aa_mutations, 0, dnds_rates

def count_shared(jsons):
    '''
    parses nuc_mutation dictionaries for expirement and counts total mutation 
    occurence for all mutation in control samples
    '''
   
    total_counts = {}
    for jp in jsons:
        sample = jp.split('/')[-1]
        if 'NT' not in sample:
            continue
        with open(jp) as jf:
            mut_dict = json.load(jf)
        for key in tqdm(mut_dict):
            for mut in mut_dict[key]['mutations']:
                gene_mut = key + ';' + mut
                if gene_mut not in total_counts:
                    total_counts[gene_mut] = mut_dict[key]['mutations'][mut]
                else:
                    total_counts[gene_mut] += mut_dict[key]['mutations'][mut]
        
    return total_counts
        
def prep_provean(mut_dict, seqobject, sample, output, features):
    '''
    Create HGVS mutation format json and var files for use in PROVEAN effect scoring.
    Intended to be called after convert_mutations() using features dictionary output 
    as input for getting aa_seqs.
    '''

    refseq = seqobject.genome

    provean_json = {}
    for gene in mut_dict:
        #get nuc sequence for gene
        seq = refseq[features[gene].start:features[gene].end]
        if features[gene].strand == -1:
            #get reverse comp of gene seq 
            seq = seq.reverse_complement()
        #convert to aa sequence without fasta_bool
        aa_seq, aa_len = translate(seq, False)
        #structure for storing mutations 
        provean_json[gene] = {'mutations':{},
                            'gene_len':mut_dict[gene]['gene_len'],
                            'avg_cov':mut_dict[gene]['avg_cov']}
        #create vars path and .var file 
        path = output+'/'+sample+'_vars/'
        os.makedirs(os.path.dirname(path), exist_ok=True)
        variants = open(path+gene+'.var','w')
        
        for mut in mut_dict[gene]['mutations']:
            hgvs_mut = hgvs_notation(mut)
            if '*' not in hgvs_mut[1:]:
                provean_json[gene]['mutations'][hgvs_mut] = mut_dict[gene]['mutations'][mut]
            else:
                deletion = '_'
                if aa_seq[-1] != '*':
                    deletion += aa_seq[-1] + str(aa_len) + 'del'
                else:
                    deletion += aa_seq[-2] + str(aa_len-1) + 'del'
                hgvs_mut = hgvs_mut.replace('*',deletion)
                provean_json[gene]['mutations'][hgvs_mut] = mut_dict[gene]['mutations'][mut]
            variants.write(hgvs_mut+'\n')
        variants.close()

    return provean_json

def generate_fasta(seq, name, codon_table):
    '''save nucleic acid seq as protein fasta string'''
    global table
    with open(codon_table) as jf:
        table = json.load(jf)

    aa_seq, aa_len = translate(seq, True)
    fasta_str = '>'+name+'\n'
    fasta_str += aa_seq
    return fasta_str

def convert_mutations(mut_dict, seqobject, codon_table, features) -> Tuple[Dict, Dict]:
    '''Convert nucleotide mutations to amino acid mutations.
    
    Args:
        mut_dict: Dictionary of mutations per gene
        seqobject: Genome sequence context
        codon_table: Path to codon table file
        features: Dictionary of genome features
        
    Returns:
        Tuple containing:
            - Dict: Mutation dictionary with amino acid mutations per gene
            - Dict: Genome-wide dN/dS statistics including:
                - non_syn_sites: Total potential non-synonymous sites
                - syn_sites: Total potential synonymous sites
                - non_syn_muts: Total non-synonymous mutations observed
                - syn_muts: Total synonymous mutations observed
                - syn_sites_mutated: Number of unique synonymous sites with mutations
                - non_syn_sites_mutated: Number of unique non-synonymous sites with mutations
                - dnds_raw: Raw genome-wide dN/dS ratio
                - dnds_norm: Coverage-normalized genome-wide dN/dS ratio
    '''
    global table
    with open(codon_table) as jf:
        table = json.load(jf)

    refseq = seqobject.genome
    
    # Get immune codons
    immune_codons = ems_immune_codons(table)
    
    # Track genome-wide stats
    genome_stats = {
        'non_syn_sites': 0,
        'syn_sites': 0,
        'non_syn_muts': 0,
        'syn_muts': 0,
        'syn_sites_mutated': 0,
        'non_syn_sites_mutated': 0,
        'total_coverage': 0,
        'gene_count': 0  # Track number of genes for averaging coverage
    }
    
    for key in mut_dict:
        seq = refseq[features[key].start:features[key].end]
        if features[key].strand == -1:
            seq = seq.reverse_complement()
            rc = True
        else:
            rc = False

        aa_muts, err, dnds_rates = convertor(seq, mut_dict[key]['mutations'], rc, immune_codons)
            
        if err != 0:
            if err == 1:
                print('Conversion Error: No start codon in gene: ' + key)
            if err == 2: 
                print('Conversion Error: ref does not match at bp  ' + str(aa_muts) + ' in gene: ' + key)
            continue
        else:
            mut_dict[key]['mutations'] = aa_muts
            mut_dict[key]['dnds_raw'] = dnds_rates['raw']
            mut_dict[key]['dnds_norm'] = dnds_rates['normalized']
            
            # Accumulate genome-wide stats
            genome_stats['non_syn_sites'] += dnds_rates['non_syn_sites']
            genome_stats['syn_sites'] += dnds_rates['syn_sites']
            genome_stats['non_syn_muts'] += dnds_rates['non_syn_muts']
            genome_stats['syn_muts'] += dnds_rates['syn_muts']
            genome_stats['syn_sites_mutated'] += dnds_rates['syn_sites_mutated']
            genome_stats['non_syn_sites_mutated'] += dnds_rates['non_syn_sites_mutated']
            
            # Track coverage
            genome_stats['total_coverage'] += mut_dict[key]['avg_cov']
            genome_stats['gene_count'] += 1
    
    # Calculate average coverage across all genes
    avg_coverage = genome_stats['total_coverage'] / genome_stats['gene_count'] if genome_stats['gene_count'] > 0 else 1
    
    # Calculate raw genome-wide dN/dS
    dn_raw = genome_stats['non_syn_muts'] / genome_stats['non_syn_sites'] if genome_stats['non_syn_sites'] > 0 else 0
    ds_raw = genome_stats['syn_muts'] / genome_stats['syn_sites'] if genome_stats['syn_sites'] > 0 else 0
    genome_stats['dnds_raw'] = dn_raw/ds_raw if ds_raw > 0 else 0
    
    # Calculate coverage-normalized genome-wide dN/dS
    normalized_non_syn_muts = genome_stats['non_syn_muts'] / avg_coverage
    normalized_syn_muts = genome_stats['syn_muts'] / avg_coverage
    
    dn_norm = normalized_non_syn_muts / genome_stats['non_syn_sites'] if genome_stats['non_syn_sites'] > 0 else 0
    ds_norm = normalized_syn_muts / genome_stats['syn_sites'] if genome_stats['syn_sites'] > 0 else 0
    genome_stats['dnds_norm'] = dn_norm/ds_norm if ds_norm > 0 else 0
    
    # Store all values for reference
    genome_stats['avg_coverage'] = avg_coverage
    genome_stats['normalized_non_syn_muts'] = normalized_non_syn_muts
    genome_stats['normalized_syn_muts'] = normalized_syn_muts
    genome_stats['dn_raw'] = dn_raw
    genome_stats['ds_raw'] = ds_raw
    genome_stats['dn_norm'] = dn_norm
    genome_stats['ds_norm'] = ds_norm
            
    return mut_dict, genome_stats

def count_ems_sites(seq: str, immune_codons: List[str]) -> Tuple[int, int]:
    '''Count potential EMS mutation sites in a sequence.
    
    Args:
        seq (str): DNA sequence
        immune_codons (List[str]): List of codons immune to non-synonymous EMS mutations
        
    Returns:
        Tuple containing:
            - int: Number of sites that could cause non-synonymous mutations
            - int: Number of sites that could only cause synonymous mutations
    '''
    non_syn_sites = 0
    syn_sites = 0
    
    # Process sequence in codons
    for i in range(0, len(seq)-2, 3):
        codon = seq[i:i+3]
        # Check if any position in codon is masked
        codon_positions = range(start_pos + i, start_pos + i + 3)
        if not all(mask[pos] for pos in codon_positions):
            continue
            
        # For immune codons, all C/G sites can only cause synonymous mutations
        if codon in immune_codons:
            for base in codon:
                if base in ['C', 'G']:
                    syn_sites += 1
        # For non-immune codons, at least one C/G site can cause non-synonymous mutations
        else:
            for base in codon:
                if base in ['C', 'G']:
                    non_syn_sites += 1
                    
    return non_syn_sites, syn_sites