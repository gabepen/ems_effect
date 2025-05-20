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

def table_lookup(codon: str, debug_info: Optional[Dict] = None) -> str:
    '''Look up amino acid for a given codon using global codon table.
    
    Args:
        codon (str): Three letter nucleotide codon
        debug_info (Optional[Dict]): Dictionary containing mutation context for debugging
        
    Returns:
        str: Single letter amino acid code, or '@' if codon not found
    '''
    for aa in table.keys():
        if codon in table[aa] and aa != 'starts':
            return aa
            
    # Debug output for unknown codons
    if codon not in ['---', '...']:  # Skip common placeholders
        print("\nERROR: Unknown codon found!")
        print(f"Codon: {codon}")
        if debug_info:
            print("\nDebug context:")
            print(f"  Gene ID: {debug_info.get('gene_id', 'unknown')}")
            print(f"  Original mutation: {debug_info.get('mutation', 'unknown')}")
            print(f"  Position: {debug_info.get('position', 'unknown')}")
            print(f"  Original codon: {debug_info.get('original_codon', 'unknown')}")
            print(f"  Reference AA: {debug_info.get('ref_aa', 'unknown')}")
            print(f"  Sequence context: {debug_info.get('sequence', 'unknown')}")
        import pdb; pdb.set_trace()  # Break for debugging
        
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

def count_ems_sites(seq: str, immune_codons: List[str], start_pos: int = 0, mask: List[bool] = None):
    '''Count potential EMS mutation sites in a sequence.'''
    non_syn_sites = 0
    syn_sites = 0
    
    # Find start codon first
    start = -1
    for i in range(len(seq)-2):
        if seq[i:i+3] in table['starts']:
            start = i
            break
            
    if start == -1:
        return 0, 0
        
    # Process by codons from start position
    for i in range(start, len(seq)-2, 3):
        codon = seq[i:i+3]
        
        # Check each position in codon
        for j, base in enumerate(codon):
            if base not in ['C', 'G']:
                continue
                
            # Check masking for this specific position
            genome_pos = start_pos + i + j
            if mask and not mask[genome_pos]:
                continue
                
            # Make mutation
            mutated = list(codon)
            mutated[j] = 'T' if base == 'C' else 'A'
            mutated = ''.join(mutated)
            
            # Check if mutation is synonymous
            if table_lookup(codon) == table_lookup(mutated):
                syn_sites += 1
            else:
                non_syn_sites += 1
                
    return non_syn_sites, syn_sites

def convertor(seq: str, seq_id: str, mutations: Dict[str, int], rc: bool, immune_codons: List[str], start_pos: int = 0, mask: List[bool] = None):
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
    
    # Count potential mutation sites
    non_syn_sites, syn_sites = count_ems_sites(seq, immune_codons, start_pos, mask)
    
    # Track sites independently
    syn_sites_mutated = set()  # Unique sites with synonymous mutations
    non_syn_sites_mutated = set()  # Unique sites with non-synonymous mutations
    
    # Count observed mutations
    syn = 0  # Total synonymous mutations
    nonsyn = 0  # Total non-synonymous mutations
    aa_mutations = {}
    
    # Find start codon first
    start = -1
    for i in range(len(seq)-2):
        if seq[i:i+3] in table['starts']:
            start = i
            break
            
    if start == -1:
        return None, 1, empty_rates
    
    for mut_key, count in mutations.items():
        # Parse mutation
        pos_str, mut_type = mut_key.split('_')
        pos = int(pos_str) - 1  # Convert to 0-based immediately
        
        # Handle reverse strand
        if rc:
            pos = len(seq) - pos - 1
            ref_base_key, alt_base_key = mut_type.split('>')
            if mut_type == 'C>T':
                mut_type = 'G>A'
                ref_base_key = 'G'
                alt_base_key = 'A'
            elif mut_type == 'G>A':
                mut_type = 'C>T'
                ref_base_key = 'C'
                alt_base_key = 'T'
        else:
            ref_base_key, alt_base_key = mut_type.split('>')
        
        # Skip positions before start codon
        if pos < start:
            print(f"Skipping - position before start codon")
            continue
            
        # Calculate codon position relative to start
        codon_pos = (pos - start) // 3 * 3 + start
        base_pos = (pos - start) % 3

        if codon_pos + 2 >= len(seq):
            print(f"Skipping - position beyond sequence end")
            continue
            
        codon = seq[codon_pos:codon_pos+3]
        
        # Make mutation
        ref_base = codon[base_pos]
        alt_base = alt_base_key
        
        # Verify reference base matches
        if ref_base != ref_base_key:
            print(f"\n=== REFERENCE MISMATCH DETAILS ===")
            print(f"Gene ID: {seq_id}")
            print(f"Position: {pos} (0-based)")
            print(f"Original position: {int(pos_str) - 1} (0-based)")
            print(f"Mutation: {mut_key} ({mut_type})")
            print(f"Expected ref base: {ref_base_key}")
            print(f"Actual ref base: {ref_base}")
            print(f"Codon: {codon} at position {codon_pos}")
            print(f"Codon position in gene: {codon_pos//3}")
            print(f"Base position in codon: {base_pos}")
            print(f"Reverse complement: {rc}")
            print(f"Sequence around position: {seq[max(0, pos-5):min(len(seq), pos+6)]}")
            print(f"Position in sequence: {pos - max(0, pos-5)}")
            print(f"=== END MISMATCH DETAILS ===\n")
            input()
            continue
            
        # Create mutated codon
        mutated = list(codon)
        mutated[base_pos] = alt_base
        mutated = ''.join(mutated)
        
        # Check translation
        ref_aa = table_lookup(codon)
        mut_aa = table_lookup(mutated)
        
        # Verify reference base matches
        if ref_base != ref_base_key:
            continue
            
        # Check if synonymous with debug info
        debug_info = {
            'gene_id': seq_id,
            'mutation': mut_key,
            'position': pos,
            'original_codon': codon,
            'ref_aa': table_lookup(codon),
            'sequence': seq[max(0, codon_pos-6):min(len(seq), codon_pos+9)]  # Show 2 codons before/after
        }
        ref_aa = table_lookup(codon, debug_info)
        mut_aa = table_lookup(mutated, debug_info)
        
        # Check if synonymous
        is_synonymous = (ref_aa == mut_aa)
        
        # Track mutation
        site_key = f"{seq_id}_{codon_pos}_{base_pos}"
        
        if is_synonymous:
            syn += count
            syn_sites_mutated.add(site_key)
            
            # Add to amino acid mutations if it's a synonymous change
            aa_key = f"{codon_pos//3}_{ref_aa}>{ref_aa}"
            if aa_key not in aa_mutations:
                aa_mutations[aa_key] = 0
            aa_mutations[aa_key] += count
        else:
            nonsyn += count
            non_syn_sites_mutated.add(site_key)
            
            # Add to amino acid mutations
            aa_key = f"{codon_pos//3}_{ref_aa}>{mut_aa}"
            if aa_key not in aa_mutations:
                aa_mutations[aa_key] = 0
            aa_mutations[aa_key] += count
    
    # Check for genes with no synonymous sites (shouldn't happen)
    if syn_sites == 0:  # Check potential sites, not observed mutations
        print(f"WARNING: Found gene with no potential synonymous sites.")
        print(f"Sequence length: {len(seq)}")
        print(f"Number of C/G sites: {sum(1 for base in seq if base in ['C', 'G'])}")
        print(f"Immune codons found: {sum(1 for i in range(0, len(seq)-2, 3) if seq[i:i+3] in immune_codons)}")
    
    dnds_rates = {
        'raw': 0, 
        'normalized': 0,
        'non_syn_sites': non_syn_sites,  # Use counted sites
        'syn_sites': syn_sites,  # Use counted sites
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
        ds = syn / syn_sites if syn_sites > 0 else 0
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
                # Regular mutation, no stop codon
                provean_json[gene]['mutations'][hgvs_mut] = mut_dict[gene]['mutations'][mut]
            elif hgvs_mut[0] == '*':
                # stop codon is being mutated to something else
                continue
            else:
                # Handle premature stop codon with proper deletion format
                pos = int(mut.split('_')[0])
                
                # Get the amino acid before the stop codon
                prev_aa = aa_seq[pos-1] if pos > 0 and pos-1 < len(aa_seq) else 'X'
                
                # Get the last amino acid in the sequence
                last_aa = aa_seq[-1] if aa_seq[-1] != '*' else aa_seq[-2]
                
                # Format as P4_S100del
                deletion = f"_{last_aa}{aa_len-1}del"
                
                # Replace the stop codon with the proper deletion format
                hgvs_mut = hgvs_mut.replace('*', deletion)
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

def convert_mutations(
    nuc_mutations: Dict[str, Dict[str, Any]],
    seqobject: SeqContext,
    codon_table: str,
    features: Dict[str, Any]
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    '''Convert nucleotide mutations to amino acid mutations.'''
    
    # Load codon table
    global table
    with open(codon_table) as jf:
        table = json.load(jf)
    
    # Initialize dictionaries
    aa_mutations = {}
    genome_stats = {
        'non_syn_sites': 0,
        'syn_sites': 0,
        'non_syn_muts': 0,
        'syn_muts': 0,
        'syn_sites_mutated': 0,
        'non_syn_sites_mutated': 0
    }
    
    # Process each gene
    for key in nuc_mutations.keys():
        # Skip genes not in features
        if key not in features:
            print(f"Gene {key} not found in features")
            continue
            
        # Get gene sequence
        seq = seqobject.genome[features[key].start:features[key].end]
        
        # Check if reverse complement
        rc = features[key].strand == -1
        if rc:
            seq = seq.reverse_complement()
            
        # Get immune codons
        immune_codons = []
        
        # Convert mutations
        aa_muts, err, dnds_rates = convertor(
            str(seq),
            key,  # Pass the correct gene ID
            nuc_mutations[key]['mutations'], 
            rc, 
            immune_codons,
            start_pos=features[key].start,
            mask=seqobject.overlap_mask
        )
            
        if err != 0:
            if err == 1:
                #print('Conversion Error: No start codon in gene: ' + key)
                pass
            if err == 2: 
                #print('Conversion Error: ref does not match at bp  ' + str(aa_muts) + ' in gene: ' + key)
                pass
            continue
            
        # Store amino acid mutations in new dictionary
        aa_mutations[key] = {
            'mutations': aa_muts,
            'gene_len': nuc_mutations[key]['gene_len'],
            'avg_cov': nuc_mutations[key]['avg_cov']
        }
        
        # Accumulate genome-wide stats
        genome_stats['non_syn_sites'] += dnds_rates['non_syn_sites']
        genome_stats['syn_sites'] += dnds_rates['syn_sites']
        genome_stats['non_syn_muts'] += dnds_rates['non_syn_muts']
        genome_stats['syn_muts'] += dnds_rates['syn_muts']
        genome_stats['syn_sites_mutated'] += dnds_rates['syn_sites_mutated']
        genome_stats['non_syn_sites_mutated'] += dnds_rates['non_syn_sites_mutated']
    
    return aa_mutations, genome_stats