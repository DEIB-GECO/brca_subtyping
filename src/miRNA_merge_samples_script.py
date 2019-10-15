import pandas as pd
import numpy as np
import urllib.request, urllib.parse, urllib.error
import sys
import os

metadata = pd.read_csv("../data/miRNA/miRNA_otherCancers_metadata_silvia.csv")

bed_files = os.listdir('../data/miRNA/genosurf_download/') 
names = []
pattern = "meq.bed"

for entry in bed_files:
    if pattern in entry :
        names.append(entry[0:-4])
print("There are {} TCGA non-brca miRNA samples".format(len(names)))


set_up = pd.read_csv("../data/miRNA/genosurf_download/9970426e-551e-4061-8f22-9e60f356da38-meq.bed", sep="\t")
set_up.columns=['chr','start','end', 'strand', 'mirna_id', 'raw', 'reads_per_million_mirna_mapped', 'cross-mapped', 'entrez-gene ID', 'gene_symbol']
miRNA_final_columns = np.append(set_up['mirna_id'].values, ['gdc_id', 'tcga_id'])
miRNA_data_final = pd.DataFrame(columns=miRNA_final_columns)

i=1
for name in names:
    miRNA_data_tmp = pd.read_csv("../data/miRNA/genosurf_download/"+name+".bed", sep="\t")
    miRNA_data_tmp.columns= ['chr','start','end', 'strand', 'mirna_id', 'raw', 'reads_per_million_mirna_mapped', 'cross-mapped', 'entrez-gene ID', 'gene_symbol']
    tcga_id_temp = metadata[metadata["item_source_id"]==name]["alt_donor_source_id"].item()

    aux = pd.DataFrame(np.array([miRNA_data_tmp['raw'].values]), columns=miRNA_data_tmp['mirna_id'].values)
    aux["gdc_id"] = name
    aux["tcga_id"] = tcga_id_temp
    miRNA_data_final = miRNA_data_final.append(aux)
    print(i)
    i+=1
miRNA_data_final.to_pickle("../data/miRNA_raw_no_brca.pkl")