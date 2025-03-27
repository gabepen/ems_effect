import os 
import json
from typing import Dict, List, Any, Tuple, Optional
from Bio.Seq import Seq
from Bio.SeqFeature import FeatureLocation
from tqdm import tqdm 
import gzip
from Bio import SeqIO
from BCBio.GFF import GFFExaminer
from BCBio import GFF
from mimetypes import guess_type 
from selectors import EpollSelector
from functools import partial
from itertools import islice
import multiprocessing as mp
from glob import glob
from pathlib import Path
import numpy as np

class SeqContext:
    '''A class to handle genomic sequence context and annotations.
    
    This class loads and provides access to genome sequence data and its annotations.
    
    Attributes:
        genome (Seq): The complete genome sequence
        annot (str): Path to the genome annotation file
    '''
    
    def __init__(self, fna: str, annot: str) -> None:
        '''Initialize SeqContext with genome and annotation files.
        
        Args:
            fna (str): Path to the genome FASTA file
            annot (str): Path to the genome annotation file (GFF format)
        '''
        #load genome fasta
        fna_encoding = guess_type(fna)[1]
        _open = partial(gzip.open, mode='rt') if fna_encoding == 'gzip' else open
        genome_fasta = SeqIO.parse(_open(fna), 'fasta')
        for record in genome_fasta:
            self.genome = record.seq
    
        #load annotation
        self.annot = annot

        # Create overlap mask
        self.overlap_mask = self._create_overlap_mask()

    def _create_overlap_mask(self) -> List[bool]:
        '''Create binary mask for overlapping regions.'''
        mask = [True] * len(self.genome)  # Initialize all positions as valid
        
        # Get sorted gene locations
        gene_locations = []
        with open(self.annot) as handle:
            for rec in GFF.parse(handle):
                for feat in rec.features:
                    if feat.type == 'gene':
                        gene_locations.append((
                            feat.location.start,
                            feat.location.end
                        ))
        
        # Mark overlapping regions as False
        gene_locations.sort(key=lambda x: x[0])
        for i in range(len(gene_locations)-1):
            current = gene_locations[i]
            j = i + 1
            while j < len(gene_locations) and gene_locations[j][0] <= current[1]:
                overlap_start = max(current[0], gene_locations[j][0])
                overlap_end = min(current[1], gene_locations[j][1])
                if overlap_end > overlap_start:
                    mask[overlap_start:overlap_end] = [False] * (overlap_end - overlap_start)
                j += 1
                
        return mask

    def gene_seqs(self) -> Dict[str, Seq]:
        '''Extract sequences for all protein-coding genes in the genome.
        
        Returns:
            Dict[str, Seq]: Dictionary mapping gene IDs to their sequences.
                Keys are gene reference IDs (from Dbxref)
                Values are gene sequences (including reverse complement for - strand)
        '''
        seqs: Dict[str, Seq] = {}
        for rec in GFF.parse(self.annot):
            for feat in rec.features:
                if (feat.type == 'gene' and 
                    'gene_biotype' in feat.qualifiers and
                    feat.qualifiers['gene_biotype'][0] == 'protein_coding'):
                    
                    refid = feat.qualifiers['Dbxref'][0].split(':')[-1]
                    # Skip genes that are entirely masked
                    if not any(self.overlap_mask[feat.location.start:feat.location.end]):
                        continue
                    seq = self.genome[feat.location.start:feat.location.end]
                    if feat.strand == -1:
                        seq = seq.reverse_complement()
                    seqs[refid] = seq
        return seqs
    
    def genome_features(self) -> Dict[str, FeatureLocation]:
        '''Get locations for all protein-coding genes in the genome.'''
        features: Dict[str, FeatureLocation] = {}
        for rec in GFF.parse(self.annot):
            for feat in rec.features:
                if (feat.type == 'gene' and
                    'gene_biotype' in feat.qualifiers and 
                    feat.qualifiers['gene_biotype'][0] == 'protein_coding'):
                    features[feat.qualifiers['Dbxref'][0].split(':')[-1]] = feat.location
        return features 
                    

def get_sequence_context(
    position: int, 
    ref_base: str,
    genome_seq: str,
    context_size: int = 3
) -> str:
    '''Get sequence context around a mutation site.
    
    Args:
        position (int): Position of mutation in genome (1-based)
        ref_base (str): Reference base at mutation site
        genome_seq (str): Full genome sequence
        context_size (int): Size of context window (3 or 5, default=3)
        
    Returns:
        str: Sequence context with mutation site in center (e.g., 'ATC' for 3mer)
    '''
    if context_size not in [3, 5]:
        raise ValueError("Context size must be 3 or 5")
        
    pos_0based = position - 1
    flank = context_size // 2
    start = max(0, pos_0based - flank)
    end = min(len(genome_seq), pos_0based + flank + 1)
    
    return str(genome_seq[start:end])

def check_mutations(
    entry: List[str], 
    genestart: int, 
    sample: str, 
    ems_only: bool, 
    base_counts: Dict[str, int],
    genome_seq: str,
    context_counts: Dict[str, int],
    context_size: int = 3
) -> Tuple[Dict[str, int], int, Dict[str, int]]:
    '''Analyze a mpileup entry to identify mutations and count contexts.
    
    Args:
        entry (List[str]): A line from mpileup file split into fields
        genestart (int): Start position of the gene in the genome
        sample (str): Sample identifier
        ems_only (bool): If True, only return EMS canonical mutations (C>T or G>A)
        base_counts (Dict[str, int]): Dictionary to track base counts
        genome_seq (str): Full genome sequence for context
        context_counts (Dict[str, int]): Dictionary to track context counts
        context_size (int): Size of sequence context window (3 or 5)
        
    Returns:
        Tuple containing:
            - Dict[str, int]: Dictionary of mutations and their counts
            - int: Read depth at this position
            - Dict[str, int]: Dictionary of base counts
    '''
    muts: Dict[str, int] = {}    
    bp = (int(entry[1]) - int(genestart))
    bp = str(bp) + '_'
    
    ref = entry[2]
    depth = int(entry[3])
    position = int(entry[1])

    # Get sequence context
    context = get_sequence_context(position, ref, genome_seq, context_size)

    # Count reference base occurrences
    local_base_counts = {'A': 0, 'T': 0, 'G': 0, 'C': 0}
    local_base_counts[ref] += depth

    reads = entry[4]
    reads = enumerate(reads)
    r_iter = iter(reads)
    for i, r in reads:
        if r != '.' and r != 'N' and r != '$' and r != '*':
            # Handle insertions
            if r == '+':
                insert = next(r_iter)[1]
                mut = bp+ref+'>'
                for c in islice(r_iter,int(insert)):
                    mut+=c[1]
            # Handle deletions
            elif r == '-':
                delete = next(r_iter)[1]
                mut = bp+'del'
                for c in islice(r_iter,int(delete)):
                    mut+=c[1]
            # Handle point mutations
            else:
                mut = f"{ref}>{r}"
                if ems_only and mut not in ['C>T', 'G>A']:
                    continue
                    
                # Only count contexts for point mutations
                context_counts[context] = context_counts.get(context, 0) + 1
                
                mut = bp + mut

            # Record all mutations (including indels)
            muts[mut] = muts.get(mut, 0) + 1

    return muts, depth, local_base_counts

def process_mpileup_chunk(args: Tuple[str, List[str], 'SeqContext', bool, List[Tuple], int, int, int]) -> Tuple[str, Dict, Dict, Dict]:
    '''Process a chunk of mpileup lines.
    
    Args:
        args (Tuple): Contains:
            - sample (str): Sample name
            - lines (List[str]): Chunk of mpileup lines
            - seqobject (SeqContext): Genome context object
            - ems_only (bool): Whether to only process EMS mutations
            - gff_features (List[Tuple]): Sorted list of gene features
            - context_size (int): Size of context window
            - start_idx (int): Starting line index
            - chunk_size (int): Size of chunk
            
    Returns:
        Tuple containing:
            - str: Sample name
            - Dict: Mutations per gene
            - Dict: Base counts
            - Dict: Intergenic counts
    '''
    sample, lines, seqobject, ems_only, gff_features, context_size, start_idx, chunk_size = args
    
    mutations = {}
    base_counts = {'A': 0, 'T': 0, 'G': 0, 'C': 0}  # Initialize local base counts
    context_counts = {}
    
    # Initialize intergenic counts before combining chunks
    intergenic_counts = {
        'mutation_types': {'C>T': 0, 'G>A': 0},  # Only track EMS mutation counts
        'unique_sites': 0,  # Count of unique positions mutated
        'gc_sites': 0,  # Total G/C sites
        'total_sites': 0,  # Total sites processed
        'coverage_depths': [],  # For calculating average coverage
        'count_bins': {str(i): 0 for i in range(1, 16)}  # Bins for mutation counts 1-15, with 15+ in bin '15'
    }
    
    current_gene_idx = 0
    current_gene = None if not gff_features else gff_features[0]
    
    for line in lines:
        entry = line.split()
        pos = int(entry[1]) - 1
        ref = entry[2]
        depth = int(entry[3])
        
        # Skip if position is in overlapping region
        if not seqobject.overlap_mask[pos]:
            continue
            
        # Count bases directly here instead of in check_mutations
        base_counts[ref] += depth
        
        # Move to next gene if we're past the current one
        while current_gene and pos > current_gene[1]:
            current_gene_idx += 1
            if current_gene_idx < len(gff_features):
                current_gene = gff_features[current_gene_idx]
            else:
                current_gene = None
                break
        
        muts, depth, chunk_base_counts = check_mutations(
            entry,
            current_gene[0] if current_gene and int(entry[1]) >= current_gene[0] else 0,
            sample,
            ems_only,
            base_counts,
            str(seqobject.genome),
            context_counts,
            context_size
        )
        
        # Add mutations to appropriate collection
        if current_gene and current_gene[0] + 1 <= int(entry[1]) < current_gene[1] + 1:
            # Gene mutation handling remains the same
            gene_id = current_gene[2]
            if gene_id not in mutations:
                mutations[gene_id] = {
                    'mutations': {},
                    'avg_cov': 0,
                    'gene_len': current_gene[1] - current_gene[0],
                    'depths': []
                }
            mutations[gene_id]['mutations'].update(muts)
            mutations[gene_id]['depths'].append(depth)
        else:
            # Intergenic region - track mutations, GC content, and coverage
            if ref in ['G', 'C']:
                intergenic_counts['gc_sites'] += 1
            intergenic_counts['total_sites'] += 1
            intergenic_counts['coverage_depths'].append(depth)
            
            # Track mutation types in intergenic regions
            for mut in muts:
                mut_type = mut.split('_')[1]  # Get mutation type (e.g., 'C>T')
                position = int(mut.split('_')[0])  # Get position
                count = muts[mut]  # Get mutation count
                
                if mut_type in ['C>T', 'G>A']:  # Only track EMS mutations
                    intergenic_counts['mutation_types'][mut_type] += count
                    
                    # Track unique sites
                    if position not in intergenic_counts.get('_temp_sites', set()):
                        if '_temp_sites' not in intergenic_counts:
                            intergenic_counts['_temp_sites'] = set()
                        intergenic_counts['_temp_sites'].add(position)
                        intergenic_counts['unique_sites'] += 1
                    
                    # Track count bins for mutations (how many times each mutation was observed)
                    count_bin = str(min(15, count))  # Cap at 15+
                    intergenic_counts['count_bins'][count_bin] += 1
    
    # Remove temporary set before returning
    if '_temp_sites' in intergenic_counts:
        del intergenic_counts['_temp_sites']
    
    return sample, mutations, base_counts, intergenic_counts

def get_chunk_boundaries(mpileup_file, chunk_size: int, gff_features: List[Tuple]) -> List[Tuple[int, int]]:
    '''Determine chunk boundaries that don't split genes.
    
    Args:
        mpileup_file: Open file handle to mpileup
        chunk_size: Target size for chunks
        gff_features: Sorted list of gene features
        
    Returns:
        List[Tuple[int, int]]: List of (start_line, end_line) for each chunk
    '''
    boundaries = []
    current_start = 0
    current_lines = []
    current_pos = 0
    
    for i, line in enumerate(mpileup_file):
        current_lines.append(line)
        pos = int(line.split()[1])
        
        # Check if we've hit our target chunk size
        if len(current_lines) >= chunk_size:
            # Find the next gene boundary
            for gene_start, gene_end, _, _ in gff_features:
                if gene_start > pos:
                    # Found next gene - use this as boundary
                    while current_lines and int(current_lines[-1].split()[1]) >= gene_start:
                        current_lines.pop()
                    boundaries.append((current_start, current_start + len(current_lines)))
                    current_start += len(current_lines)
                    current_lines = []
                    break
            
            # If we didn't find a gene boundary, use chunk size
            if current_lines:
                boundaries.append((current_start, current_start + len(current_lines)))
                current_start += len(current_lines)
                current_lines = []
                
    # Handle remaining lines
    if current_lines:
        boundaries.append((current_start, current_start + len(current_lines)))
    
    return boundaries

def parse_mpile(mpile: str, seqobject: 'SeqContext', ems_only: bool, base_counts: Dict[str, int],
                context_counts: Dict[str, int], context_size: int = 3,
                chunk_size: int = 100000) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, int], Dict[str, Dict[str, int]]]:
    '''Parse a single mpileup file to identify mutations.
    
    Args:
        mpile (str): Path to single mpileup file
        seqobject (SeqContext): Genome context object
        ems_only (bool): Whether to only process EMS mutations
        base_counts (Dict[str, int]): Dictionary to store base counts
        context_counts (Dict[str, int]): Dictionary to store context counts
        context_size (int): Size of context window
        chunk_size (int): Size of chunks for parallel processing
        
    Returns:
        Tuple containing:
            - Dict: Mutations per gene
            - Dict: Context counts
            - Dict: Intergenic counts
    '''
    # Load and sort GFF features once, filtering for protein-coding genes only
    gff_features = []
    with open(seqobject.annot) as gff_file:
        for rec in GFF.parse(gff_file):
            for feat in rec.features:
                if (feat.type == 'gene' and
                    'gene_biotype' in feat.qualifiers and 
                    feat.qualifiers['gene_biotype'][0] == 'protein_coding'):
                    gff_features.append((
                        feat.location.start,
                        feat.location.end,
                        feat.qualifiers['Dbxref'][0].split(':')[-1],
                        feat.location
                    ))
    gff_features.sort(key=lambda x: x[0])

    # Process single mpileup in chunks
    chunk_args = []
    sample = Path(mpile).stem
    
    # Create chunks for parallel processing
    with open(mpile) as f:
        lines = f.readlines()
        for i in range(0, len(lines), chunk_size):
            chunk = lines[i:i + chunk_size]
            chunk_args.append((sample, chunk, seqobject, ems_only, gff_features, context_size, i, chunk_size))
    
    # Initialize intergenic counts BEFORE processing chunks
    intergenic_counts = {
        'mutation_types': {'C>T': 0, 'G>A': 0},  # Only track EMS mutation counts
        'unique_sites': 0,  # Count of unique positions mutated
        'gc_sites': 0,  # Total G/C sites
        'total_sites': 0,  # Total sites processed
        'coverage_depths': [],  # For calculating average coverage
        'count_bins': {str(i): 0 for i in range(1, 16)}  # Bins for mutation counts 1-15, with 15+ in bin '15'
    }
    
    # Process chunks in parallel
    with mp.Pool() as pool:
        chunk_results = list(pool.imap(process_mpileup_chunk, chunk_args))
    
    # Combine chunk results
    sample_mutations = {}
    
    for _, mutations, chunk_base_counts, chunk_intergenic in chunk_results:
        # Combine base counts from chunk
        for base, count in chunk_base_counts.items():
            base_counts[base] += count  # Add chunk counts to main base_counts
            
        # Combine mutations
        for gene_id, gene_data in mutations.items():
            if gene_id not in sample_mutations:
                sample_mutations[gene_id] = {
                    'mutations': {},
                    'avg_cov': 0,
                    'gene_len': gene_data['gene_len'],
                    'depths': []
                }
            sample_mutations[gene_id]['mutations'].update(gene_data['mutations'])
            sample_mutations[gene_id]['depths'].extend(gene_data['depths'])
        
        # Combine intergenic data
        if 'mutation_types' in chunk_intergenic:
            for mut_type, count in chunk_intergenic['mutation_types'].items():
                # Combine total mutations
                intergenic_counts['mutation_types'][mut_type] = \
                    intergenic_counts['mutation_types'].get(mut_type, 0) + count
        
        intergenic_counts['gc_sites'] += chunk_intergenic['gc_sites']
        intergenic_counts['total_sites'] += chunk_intergenic['total_sites']
        intergenic_counts['unique_sites'] += chunk_intergenic['unique_sites']
        intergenic_counts['coverage_depths'].extend(chunk_intergenic['coverage_depths'])
        
        # Combine count bins
        if 'count_bins' in chunk_intergenic:
            for count, bin_count in chunk_intergenic['count_bins'].items():
                intergenic_counts['count_bins'][count] = \
                    intergenic_counts['count_bins'].get(count, 0) + bin_count
    
    # Calculate final averages
    for gene_id in sample_mutations:
        depths = sample_mutations[gene_id]['depths']
        sample_mutations[gene_id]['avg_cov'] = sum(depths) / len(depths) if depths else 0
        del sample_mutations[gene_id]['depths']
    
    # Calculate average coverage for intergenic regions
    if intergenic_counts['coverage_depths']:
        intergenic_counts['avg_coverage'] = \
            sum(intergenic_counts['coverage_depths']) / len(intergenic_counts['coverage_depths'])
    del intergenic_counts['coverage_depths']
    
    return sample_mutations, context_counts, intergenic_counts
                            