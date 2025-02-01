from ast import GeneratorExp
import pprint as pp
import sys, getopt 
from glob import glob
import json
import math
import csv
import os
#from tkinter import E
from tqdm import tqdm
import numpy as np
#import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import entrezpy.esearch.esearcher
from Bio import Entrez
Entrez.email = 'gapenunu@ucsc.edu'


def retrieve_annotation(id_list):

    """Annotates Entrez Gene IDs using Bio.Entrez, in particular epost (to
    submit the data to NCBI) and esummary to retrieve the information.
    Returns a list of dictionaries with the annotations."""

    request = Entrez.epost("gene", id=",".join(id_list))
    try:
        result = Entrez.read(request)
    except RuntimeError as e:
    
        return 'NA'

    webEnv = result["WebEnv"]
    queryKey = result["QueryKey"]
    data = Entrez.esummary(db="gene", webenv=webEnv, query_key=queryKey)
    annotations = Entrez.read(data)

    #print("Retrieved %d annotations for %d genes" % (len(annotations), len(id_list)))
    return annotations


def collectAnnos(effectFilter, jsons):

    with open('/home/gabe/wolbachia/transposons.csv', '+w') as csvf:
        writer = csv.writer(csvf)
        writer.writerow(['Annotation','Effect','GID'])
        for js in jsons:
            with open(js) as jf:
                effects = json.load(jf)
            for gene in effects:
                try:
                    effect = math.log(float(effects[gene]['total'])) 
                except ValueError:
                    effect = 0
                if effect == effectFilter:
                    anno = retrieve_annotation([gene])
                    writer.writerow([anno['DocumentSummarySet']['DocumentSummary'][0]['Description'],effect,gene])
                    print(anno['DocumentSummarySet']['DocumentSummary'][0]['Description'])

def parse_effects(result_dict, threshold, exclude):
    '''
        Takes in a dictionary object of one sample containing results of effect scoring.
        Threshold is passed to determine when a gene should be annotated.
        Exclude list contains GID of genes to ignore.
        Returns dataframe object for plotting and list of rows for csv output.
    '''
    
    #load annotation table
    with open('/scratch/home/gabe/EMS_data/anno_tables/anno_table1.json', 'r') as at:
        anno_table = json.load(at)
    

    #gene_df = pd.DataFrame({'gene':[],
                            #'total':[]})
    '''
    with open('/home/gabe/wolbachia/data_csv/genes/'+analysis+'/'+sample+'_genes_eff'+str(effFilter)+'.csv', '+w') as csvf:
        writer = csv.writer(csvf)
        writer.writerow(['Annotation','Effect','GID'])
    '''
    scores = []
    anno_scores = []
    for gene in result_dict:
        if gene not in exclude: #and gene not in transposons:
            try:
                effect = float(result_dict[gene]['effect'])
                #effect = math.log(float(result_dict[gene]['effect'])) 
            except ValueError:
                effect = 0

            #gene_df.loc[len(gene_df.index)] = [gene, effect]
            scores.append([gene, effect]) 

    sorted_scores = sorted(scores, key=lambda x:x[1])  

    lowest = len(sorted_scores) - (len(sorted_scores) * threshold)
    
    for i in range(int(lowest), len(sorted_scores)):
        gene_id = sorted_scores[i][0]
        
        
        if gene_id in anno_table.keys():
            gene_anno = anno_table[gene_id]
        else:
            pulled_anno = (retrieve_annotation(sorted_scores[i][0]))
            if pulled_anno != 'NA':  
                gene_anno = pulled_anno['DocumentSummarySet']['DocumentSummary'][0]['Description']
            else:
                gene_anno = 'not_found'
            anno_table[gene_id] = gene_anno
        
        anno_scores.append([sorted_scores[i][1],gene_id])

    #save anno table 
    with open('/scratch/home/gabe/EMS_data/anno_tables/anno_table1.json', 'w') as of:
        json.dump(anno_table, of)
        
    return anno_scores

def annotate_gid(gid):
    pulled_anno = retrieve_annotation(gid)
    if pulled_anno != 'NA':  
        gene_anno = pulled_anno['DocumentSummarySet']['DocumentSummary'][0]['Description']
    else:
        gene_anno = 'not_found'
    return gene_anno


def score_pvalue(results, simulated):
    '''
    Determines pvalue of gene effect scores for given project.
    '''
    pvalues = []
    samples = list(results.keys())
    header = samples
    header.insert(0, 'gene')
    pvalues.append(header) #samples as header
    
    for gene in results[samples[1]]:
        raw_scores = simulated[samples[1]][gene]
        # remove nonsense scores from dist
        scores = list(filter(lambda val: val > -100, raw_scores))
        scores.sort()

        if gene != '69724419':
            sns.kdeplot(data=scores, label=gene)
            plt.legend(loc='upper left')
            plt.title('Simulated PROVEAN Effect Score Distributions')
            plt.xlabel('Effect Score')
            plt.ylabel('Density')
            plt.savefig(gene + '_simscores.png')
            input()
        
        row = [gene]
        for sample in samples[1:]:
            for i in range(len(scores)):
                
                if results[sample][gene]['effect'] < scores[i]:
                    p_val = i / len(scores)
                    row.append(p_val)
                    break
                
                elif i == len(scores)-1:
                    p_val = i / len(scores)
                    row.append(p_val)
                    break
                #print(i, len(scores))
        
        pvalues.append(row)
        
    return pvalues

