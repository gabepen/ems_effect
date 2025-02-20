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

    def gene_seqs(self) -> Dict[str, Seq]:
        '''Extract sequences for all genes in the genome.
        
        Returns:
            Dict[str, Seq]: Dictionary mapping gene IDs to their sequences.
                Keys are gene reference IDs (from Dbxref)
                Values are gene sequences (including reverse complement for - strand)
        '''
        seqs: Dict[str, Seq] = {}
        annot_encoding = guess_type(self.annot)[1]
        _open = partial(gzip.open, mode='rt') if annot_encoding == 'gzip' else open
        for rec in GFF.parse(_open(self.annot)):
            for feat in rec.features:
                if feat.type == 'gene':
                    refid = feat.qualifiers['Dbxref'][0].split(':')[-1]
                    seq = self.genome[feat.location.start:feat.location.end]
                    if feat.strand == -1:
                        #get reverse comp of gene seq 
                        seq = seq.reverse_complement()
                    seqs[refid] = seq
        return seqs
    
    def genome_features(self) -> Dict[str, FeatureLocation]:
        '''Get locations for all features in the genome.
        
        Returns:
            Dict[str, FeatureLocation]: Dictionary mapping feature IDs to their locations.
                Keys are feature reference IDs (from Dbxref)
                Values are BioPython FeatureLocation objects
        '''
        features: Dict[str, FeatureLocation] = {}
        for rec in GFF.parse(self.annot):
            for feat in rec.features:
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
) -> Tuple[Dict[str, int], int]:
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
    base_counts[ref] += depth

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

    return muts, depth

def process_mpileup_chunk(args: Tuple[str, List[str], SeqContext, bool, List[Tuple], int, int, int]) -> Tuple[str, Dict, Dict, Dict]:
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
            - Dict: Intergenic counts with:
                - mutations: Dict of mutation types and counts
                - gc_sites: Count of G/C bases in intergenic regions
                - total_sites: Total bases in intergenic regions
                - coverage: List of coverage values for averaging
    '''
    sample, lines, seqobject, ems_only, gff_features, context_size, start_idx, chunk_size = args
    
    mutations = {}
    base_counts = {'A': 0, 'T': 0, 'G': 0, 'C': 0}
    context_counts = {}
    intergenic_counts = {
        sample: {
            'mutations': {},  # Store mutation types and counts
            'gc_sites': 0,    # Count of G and C bases in intergenic regions
            'total_sites': 0, # Total bases in intergenic regions
            'coverage': [],   # List to track coverage for averaging
        }
    }
    
    current_gene_idx = 0
    current_gene = None if not gff_features else gff_features[0]
    
    for line in lines:
        entry = line.split()
        pos = int(entry[1]) - 1
        ref = entry[2]
        depth = int(entry[3])
        
        # Move to next gene if we're past the current one
        while current_gene and pos > current_gene[1]:
            current_gene_idx += 1
            if current_gene_idx < len(gff_features):
                current_gene = gff_features[current_gene_idx]
            else:
                current_gene = None
                break
        
        # Process mutations based on position
        muts, depth = check_mutations(
            entry,
            current_gene[0] if current_gene and pos >= current_gene[0] else 0,
            sample,
            ems_only,
            base_counts,
            str(seqobject.genome),
            context_counts,
            context_size
        )
        
        # Add mutations to appropriate collection
        if current_gene and current_gene[0] <= pos < current_gene[1]:
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
            intergenic_counts[sample]['total_sites'] += 1
            intergenic_counts[sample]['coverage'].append(depth)
            if ref in ['G', 'C']:
                intergenic_counts[sample]['gc_sites'] += 1
            
            for mut in muts:
                mut_type = mut.split('_')[1]
                if 'mutations' not in intergenic_counts[sample]:
                    intergenic_counts[sample]['mutations'] = {}
                intergenic_counts[sample]['mutations'][mut_type] = \
                    intergenic_counts[sample]['mutations'].get(mut_type, 0) + muts[mut]
    
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

def parse_mpile(mpile: str, 
                seqobject: SeqContext, 
                ems_only: bool, 
                base_counts: Dict[str, int],
                context_counts: Dict[str, int], 
                context_size: int = 3,
                chunk_size: int = 100000
                ) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, int], Dict[str, Dict[str, int]]]:
    '''Parse mpileup files in parallel to identify mutations.'''
    # Load and sort GFF features once
    gff_features = []
    with open(seqobject.annot) as gff_file:
        for rec in GFF.parse(gff_file):
            for feat in rec.features:
                if feat.type == 'gene':
                    gff_features.append((
                        feat.location.start,
                        feat.location.end,
                        feat.qualifiers['Dbxref'][0].split(':')[-1],
                        feat.location
                    ))
    gff_features.sort(key=lambda x: x[0])
    
    # Get list of mpileup files
    if os.path.isdir(mpile):
        mpileup_files = glob(os.path.join(mpile, "*.mpileup*"))
    else:
        mpileup_files = [mpile]
    
    # Process each mpileup file in chunks
    all_mutations = {}
    all_intergenic = {}
    
    for mpileup in mpileup_files:
        sample = Path(mpileup).stem
        print(f"Processing {sample}...")
        
        encoding = guess_type(mpileup)[1]
        _open = partial(gzip.open, mode='rt') if encoding == 'gzip' else open
        
        with _open(mpileup) as f:
            # First pass to get chunk boundaries
            chunk_boundaries = get_chunk_boundaries(f, chunk_size, gff_features)
            f.seek(0)  # Reset file pointer
            
            # Read file in chunks based on boundaries
            chunk_args = []
            for start_idx, end_idx in chunk_boundaries:
                current_chunk = list(islice(f, end_idx - start_idx))
                if current_chunk:
                    chunk_args.append((
                        sample, current_chunk, seqobject, ems_only, 
                        gff_features, context_size, start_idx, len(current_chunk)
                    ))
        
        # Process chunks in parallel
        with mp.Pool() as pool:
            chunk_results = list(pool.imap(process_mpileup_chunk, chunk_args))
        
        # Combine chunk results
        sample_mutations = {}
        sample_intergenic = {
            'mutations': {},
            'gc_sites': 0,
            'total_sites': 0,
            'avg_coverage': 0,
            'coverage_depths': []
        }
        sample_base_counts = {'A': 0, 'T': 0, 'G': 0, 'C': 0}
        
        for _, mutations, counts, intergenic in chunk_results:
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
            chunk_intergenic = intergenic[sample]
            if 'mutations' in chunk_intergenic:
                for mut_type, count in chunk_intergenic['mutations'].items():
                    sample_intergenic['mutations'][mut_type] = \
                        sample_intergenic['mutations'].get(mut_type, 0) + count
            
            sample_intergenic['gc_sites'] += chunk_intergenic['gc_sites']
            sample_intergenic['total_sites'] += chunk_intergenic['total_sites']
            sample_intergenic['coverage_depths'].extend(chunk_intergenic['coverage'])
            
            # Combine base counts
            for base, count in counts.items():
                sample_base_counts[base] += count
        
        # Calculate final averages
        for gene_id in sample_mutations:
            depths = sample_mutations[gene_id]['depths']
            sample_mutations[gene_id]['avg_cov'] = sum(depths) / len(depths) if depths else 0
            del sample_mutations[gene_id]['depths']
        
        # For intergenic regions
        if sample_intergenic['coverage_depths']:
            sample_intergenic['avg_coverage'] = \
                sum(sample_intergenic['coverage_depths']) / len(sample_intergenic['coverage_depths'])
        del sample_intergenic['coverage_depths']  # Clean up temporary list
        
        # Update global results
        all_mutations.update(sample_mutations)
        all_intergenic[sample] = sample_intergenic
        for base, count in sample_base_counts.items():
            base_counts[base] = base_counts.get(base, 0) + count
    
    return all_mutations, context_counts, all_intergenic
                            