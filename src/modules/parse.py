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
                    
def check_mutations(
    entry: List[str], 
    genestart: int, 
    sample: str, 
    ems_only: bool, 
    c_filter: int = 0
) -> Tuple[Dict[str, int], int]:
    '''Analyze a mpileup entry to identify mutations.
    
    Given a position in a mpileup file (entry), position in genome (genestart),
    and sample name determines nucleotide mutation and count.
    
    Args:
        entry (List[str]): A line from mpileup file split into fields
        genestart (int): Start position of the gene in the genome
        sample (str): Sample identifier
        ems_only (bool): If True, only return EMS canonical mutations (C>T or G>A)
        c_filter (int, optional): Minimum count filter for mutations. Defaults to 0
        
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

    #running total bp count
    #counts[sample][ref] += depth

    reads = entry[4]
    reads = enumerate(reads)
    r_iter = iter(reads)
    for i, r in reads:
        if r != '.' and r != 'N' and r != '$' and r != '*':
            #catch insertions 
            if r == '+':
                insert = next(r_iter)[1]
                mut = bp+ref+'>'
                for c in islice(r_iter,int(insert)):
                    mut+=c[1]
            #catch deletions
            elif r == '-':
                delete = next(r_iter)[1]
                mut = bp+'del'
                for c in islice(r_iter,int(delete)):
                    mut+=c[1]
            #normal point muts
            else:
                change = ref+'>'+r    
                mut = bp+change

            #filter out non ems canon mutations    
            if ems_only:
                if mut.split('_')[1] == 'C>T' or mut.split('_')[1] == 'G>A':
                    if mut not in muts:
                        muts[mut] = 1
                    else:
                        muts[mut] += 1
                else:
                    continue
            else:
                if mut not in muts:
                    muts[mut] = 1
                else:
                    muts[mut] += 1
    #filter low count mutations
    
    rmkeys = []
    for key in muts:
        if muts[key] < c_filter:
            rmkeys.append(key)
    for key in rmkeys:
        del muts[key]
    

    return muts, depth

def parse_mpile(
    mpile: str, 
    seqobject: SeqContext, 
    ems_only: bool
) -> Dict[str, Dict[str, Any]]:
    '''Parse an mpileup file to identify mutations in genes.
    
    Iterates over an mpileup file calling check_mutations() for each position.
    Collects mutations into a dictionary object to be returned.
    
    Args:
        mpile (str): Path to mpileup file
        seqobject (SeqContext): Genomic context object containing sequence and annotations
        ems_only (bool): If True, only process EMS canonical mutations
        
    Returns:
        Dict[str, Dict[str, Any]]: Nested dictionary containing:
            - First level keys are gene IDs
            - Second level contains:
                - 'mutations': Dictionary of mutations and their counts
                - 'avg_cov': Average coverage for the gene
                - 'gene_len': Length of the gene
    '''
    annot_encoding = guess_type(seqobject.annot)[1]
    _open = partial(gzip.open, mode='rt') if annot_encoding == 'gzip' else open 
    for rec in GFF.parse(_open(seqobject.annot)):
        mutations: Dict[str, Dict[str, Any]] = {}
        encoding = guess_type(mpile)[1] 
        _open = partial(gzip.open, mode='rt') if encoding == 'gzip' else open

        samp = mpile.split('/')[-1]
        samp = samp.split('.')[0]
        #counts[samp] = {'A':0,'T':0,'G':0,'C':0}

        with _open(mpile) as mf:
            #iterate over genes in gtf
            print('wol_parse | parsing mpileup: ' + mpile)
            for feat in tqdm(rec.features):
                if feat.type == 'gene':
                    refid = feat.qualifiers['Dbxref'][0].split(':')[-1]
                    mutations[refid] = {
                        'mutations': {},
                        'avg_cov': 0,  #avg and len are used for normalizing effect score
                        'gene_len': 0
                    }
                    loc = feat.location
                    depths: List[int] = []
                    #iterate over lines in mpileup 
                    for l in mf:
                        entry = l.split()
                        #check if mpileup entry is in the gene 
                        if int(entry[1])-1 in loc: 
                            muts, depth = check_mutations(entry, loc.start, samp, ems_only)
                            depths.append(depth)
                            mutations[refid]['mutations'].update(muts)
                        elif int(entry[1]) > loc.end:
                            break
                    try:
                        mutations[refid]['avg_cov'] = sum(depths) / len(depths)
                    except ZeroDivisionError:
                        mutations[refid]['avg_cov'] = 0
                    
                    mutations[refid]['gene_len'] = loc.end - loc.start
        return mutations
                            