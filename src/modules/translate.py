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
from wol_parse import SeqContext

def table_lookup(codon):
    '''
    for use with table in dict format 
    '''
    for aa in table.keys():
        if codon in table[aa] and aa != 'starts':
            return aa
    return '@'

def hgvs_notation(mut):
    '''
    Converts mutations from mut_jsons to hgvs notation
    '''
    pos = int(mut.split('_')[0]) + 1
    residues = mut.split('_')[1]
    ref = residues.split('>')[0]
    alt = residues.split('>')[1]
    return ref+str(pos)+alt
    
def single_flip(base):
    '''
    Flips to paired base
    '''
    if base == 'A':
        return 'T'
    elif base == 'T':
        return 'A'
    elif base == 'C':
        return 'G'
    elif base == 'G':
        return 'C'

def translate(seq, fasta_bool):
    '''
    Given nucleic acid sequence, converts to coresponding AA seq
    '''
    aa_seq = ''
    for i in range(len(seq)):
        if seq[i:i+3] in table['starts']:
            start = i
            break
    if start == -1:
        print(seq)
        return None, 1
    #split seq into codons
    codons = []
    for pos1, pos2, pos3 in zip(seq[start::3], seq[start+1::3], seq[start+2::3]):
       codons.append(''.join([pos1, pos2, pos3]))
    #line count
    line_count = 0
    length = 0
    for codon in codons:
        line_count += 1
        length += 1
        aa_seq += table_lookup(codon)
        if fasta_bool:
            if line_count >= 50:
                aa_seq += '\n'
                line_count = 0

    return aa_seq, length

def convertor(seq, mutations, rc): 
    #find first start codon
    match = 0
    mis = 0
    start = -1
    nonsyn = 0
    syn = 0
    for i in range(len(seq)):
        if seq[i:i+3] in table['starts']:
            start = i
            break
    if start == -1:
        print(seq)
        return None, 1
    #split seq into codons
    codons = []
    for pos1, pos2, pos3 in zip(seq[start::3], seq[start+1::3], seq[start+2::3]):
       codons.append(''.join([pos1, pos2, pos3])) 
    #determine aa change
    aa_mutations = {}
    for mut in mutations.keys():
        if rc:
            old_mut = mut.split('_')
            new_loc = len(seq) - int(old_mut[0])
            new_ref = single_flip(old_mut[1].split('>')[0])
            new_alt = single_flip(old_mut[1].split('>')[1])
            new_mut = str(new_loc+1) +'_'+new_ref+'>'+new_alt
        new_mut = mut.split('_')
        bp = (int(new_mut[0])-1) - start
        if bp < 1: # not in frame 
            continue
        ref = mut[1].split('>')[0]
        
        try:
            if ref == seq[bp+start]:
                match += 1
            else:
                mis += 1
        except IndexError:
            print("BPIndexError")
            print(len(seq))
            print(bp)
        alt = new_mut[1].split('>')[1]
        aa_pos = (bp//3)
        try:
            codon = codons[aa_pos]
        except IndexError:
            return aa_mutations, 0, 0
        new = codon[:(bp%3)] + alt + codon[(bp%3)+1:]
        ref_aa = table_lookup(codon)
        alt_aa = table_lookup(new)
        if ref_aa == alt_aa:
            syn += 1
            continue
        else:
            nonsyn += 1
        aa_mut = str(aa_pos) + '_' + ref_aa + '>' + alt_aa
        if aa_mut in aa_mutations.keys():
            aa_mutations[aa_mut] += mutations[mut]
        else:
            aa_mutations[aa_mut] = mutations[mut]

    filtered = []
    for mut in aa_mutations.keys():
        if aa_mutations[mut] < 1:
            filtered.append(mut)
    for mut in filtered:
        aa_mutations.pop(mut)
    try:
        synRatio = nonsyn / syn
    except ZeroDivisionError:
        synRatio = 0

    return aa_mutations, 0, synRatio

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

def convert_mutations(mut_dict, seqobject, codon_table, features):
    ''''''
    global table
    with open(codon_table) as jf:
        table = json.load(jf)

    refseq = seqobject.genome

    # generate dictionary of GID keys and coordinate values
    '''
    features = {}
    for rec in GFF.parse(seqobject.annot):
        for feat in rec.features:
            features[feat.qualifiers['Dbxref'][0].split(':')[-1]] = feat.location
    '''
    for key in mut_dict:
        seq = refseq[features[key].start:features[key].end]
        if features[key].strand == -1:
            #get reverse comp of gene seq 
            seq = seq.reverse_complement()
            rc = True
        else:
            rc = False

        aa_muts, err, syn_ratio = convertor(seq, mut_dict[key]['mutations'], rc)
            
        if err != 0:
            if err == 1:
                print('Conversion Error: No start codon in gene: ' + key)
            if err == 2: 
                print('Conversion Error: ref does not match at bp  ' + str(aa_muts) + ' in gene: ' + key)
                pp.pprint(mut_dict[key])
        else:
            mut_dict[key]['mutations'] = aa_muts
            mut_dict[key]['syn_ratio'] = syn_ratio
    return mut_dict