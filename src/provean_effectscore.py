import sys
import os 

# Add src to path 
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import parse 
import translate
from mimetypes import guess_type 
from glob import glob
from tqdm import tqdm
import getopt
import json
import random
import subprocess
import time

def main(argv):
    PERMUTATIONS=0
    PILE=''
    EMS_ONLY=False
    SKIP=False
    try:
        opts, args = getopt.getopt(argv,'hm:p:o:es',['mpileups=','output=','exclude=','skip_parse='])
    except getopt.GetoptError:
        print('provean_effectscore.py -m <path to mpileup files> -o <working dir>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('provean_effectscore.py -m <path to mpileup files> -o <working dir>')
            sys.exit()
        elif opt in ('-m', '--mpileups'):
            PILE = arg
        elif opt in ('-e', '--exclude'):
            EMS_ONLY = True
        elif opt in ('-s', '--skip_parse'):
            SKIP = True
        elif opt in ('-o', '--output'):
            OUTDIR = arg
            if OUTDIR[-1] == '/':
                OUTDIR = OUTDIR[:-1]
    #references 
    genomic_fna = '/storage1/gabe/ems_effect_code/data/references/GCF_016584425.1_ASM1658442v1_genomic.fna.gz'
    annotation = '/storage1/gabe/ems_effect_code/data/references/GCF_016584425.1_ASM1658442v1_genomic.gff'
    codon_table = '/storage1/gabe/ems_effect_code/data/references/11.json'
    prov_score_table_path = '/storage1/gabe/ems_effect_code/data/provean_tables/score_table_2.json'
    #create output directory 
    if not os.path.isdir(OUTDIR):
        os.mkdir(OUTDIR)
    if not os.path.isdir(OUTDIR+'/nuc_muts'):
        os.mkdir(OUTDIR+'/nuc_muts')
    if not os.path.isdir(OUTDIR+'/aa_muts'):
        os.mkdir(OUTDIR+'/aa_muts')
    if not os.path.isdir(OUTDIR+'/provean_files'):
        os.mkdir(OUTDIR+'/provean_files')
    if not os.path.isdir(OUTDIR+'/results'):
        os.mkdir(OUTDIR+'/results')

    results = {}
    mutpath = OUTDIR+'/nuc_muts'
    aapath = OUTDIR+'/aa_muts'
    provpath = OUTDIR+'/provean_files'
    
    piles = glob(PILE + '/*_variants.txt')
    #initalize SeqContext object and parse mpileups into nucleotide mutations 
    wolgenome = parse.SeqContext(genomic_fna,annotation)
    gene_seqs = wolgenome.gene_seqs()
    features = wolgenome.genome_features()

    #convert mpileups to nucleotide mutation format jsons
    #print('skipping pileup')
    ems_only=False
    
    if not SKIP:
        for mpileup in piles: 
            samp = mpileup.split('/')[-1]
            sample = samp.split('.')[0]
            nuc_muts = parse.parse_mpile(mpileup, wolgenome, ems_only)
            with open(mutpath+'/'+sample+'.json', 'w') as of:
                json.dump(nuc_muts, of)
        
    nuc_muts = glob(mutpath + '/*.json')
    
    print('nuc_mutation jsons generated')
    input()
    #convert nuc_muts to amino acid format jsons
    for js in nuc_muts:
        sample = js.split('/')[-1].split('.')[0]
        with open(js) as jf:
            mut_dict = json.load(jf)

            #convert mutations to aa format and save to json
            aa_muts = translate.convert_mutations(mut_dict, wolgenome, codon_table, features)
            with open(aapath+'/'+sample+'.json', 'w') as of:
                json.dump(aa_muts, of)

            #convert mutations to hgvs format and save to json 
            hgvs_dict = translate.prep_provean(aa_muts, wolgenome, sample, provpath, features)
            with open(provpath+'/'+sample+'.json', 'w') as of:
                json.dump(hgvs_dict, of)

    #provean scoring
    with open(prov_score_table_path) as jf:
        prov_score_table = json.load(jf)

    #collect mutations for scoring
    provean_jsons = glob(provpath+'/*.json')

    for js in provean_jsons:
        sample = js.split('/')[-1].split('.')[0]
        with open(js) as jf:
            hgvsmut_dict = json.load(jf)

            #for each sample sample json store results
            if sample not in results:
                results[sample] = {}

            #score each gene in hgvs formated mutation dict
            for gene in hgvsmut_dict:
                if gene not in results:
                    results[sample][gene] = {'effect':0,
                                             'gene_len':hgvsmut_dict[gene]['gene_len'],
                                             'avg_cov':hgvsmut_dict[gene]['avg_cov']}
                if gene not in prov_score_table:
                    prov_score_table[gene] = {}
                
                #check variants in .var for calculated provean score in lookup table
                no_score = False
                var_path = provpath+'/'+sample+'_vars'
                new_variants = open(var_path+'/'+gene+'.new.var','w')
                with open(var_path+'/'+gene+'.var','r') as var_file:
                    for hgvs_mut in var_file:
                        hgvs_mut = hgvs_mut.strip()
                        if hgvs_mut in prov_score_table[gene]:
                            score = float(prov_score_table[gene][hgvs_mut]) * hgvsmut_dict[gene]['mutations'][hgvs_mut] #factor mut count
                            results[sample][gene]['effect'] += score
                        else: #save variants not calculated for that gene as new 
                            no_score = True
                            new_variants.write(hgvs_mut+'\n')
                new_variants.close()
                #if gene var file contained new mutations
                if no_score:
                    #call provean subprocess subp_provean.sh 
                    print('Provean called. GID: ' + gene + ' Sample: '+ sample)
                    subprocess.run(["/home/gabe/wol_py/subp_provean.sh", gene, var_path])
                    #get scores from generated csv
                    with open(var_path+'/'+gene+'.csv','r') as scores:
                        for line in scores:
                            hgvs_mut, score = line.strip().split(',')
                            #save to results 
                            results[sample][gene]['effect'] += (float(score) * hgvsmut_dict[gene]['mutations'][hgvs_mut])
                            #update lookuptable 
                            prov_score_table[gene][hgvs_mut] = score

            #normalize scores 
            for gene in results[sample]:
                effect_score = results[sample][gene]['effect'] * 1000
                _score = effect_score / results[sample][gene]['gene_len']
                if results[sample][gene]['avg_cov'] != 0:
                    results[sample][gene]['effect'] = _score / results[sample][gene]['avg_cov']
                else:
                    results[sample][gene]['effect'] = 0

        #save lookuptable 
        with open(prov_score_table_path, 'w') as of:
            json.dump(prov_score_table, of)                   

    #save results      
    with open(OUTDIR+'/results/normalized_scores.json', 'w') as of:
        json.dump(results, of) 
                        
if __name__ == '__main__':
    main(sys.argv[1:])