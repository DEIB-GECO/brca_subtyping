import os
os.environ["OMP_NUM_THREADS"] = "20"
from collections import defaultdict
import pandas as pd
import numpy as np
from tqdm import tqdm

folder = "../data/cna/genosurf_download/files/"


files = os.listdir(folder)
files = filter(lambda x:x.endswith(".bed"), files)
files = list(files)
s = set()
d = defaultdict(int)

d_new = defaultdict(int)

final_cna_df = pd.DataFrame()


check = False

max_ = 0
min_ = 0
min_pos = 10000000
max_neg = -10000000

counts = defaultdict(int)

for file_name in tqdm(files):
    file_name = folder + file_name
    if check:
        pre_chr = None
    with open(file_name) as file:
#         print(file_name)
        file = map(lambda x:x.strip().split("\t"), file)
        file = map(lambda x:(x[0],int(x[1]), int(x[2]),x[3],int(float(x[4])),float(x[5]),x[6]), file)
        file = filter(lambda x:x[6] == 'Y' , file)
        
        if check:
            file = sorted(file)
                
        for (chrom, start,stop,strand,count,value,nocnv) in file:
#             print(file_name,(chrom, start,stop,strand,count,value,nocnv))
            if check: # check is there any overlap !!!!
                if pre_chr != chrom:
                    pre_chr = chrom
                else:
                    if start < pre_stop:
                        print(pre_stop,"********************************************************************************")
                        print(chrom, start,stop,strand,count,value,nocnv)

                pre_stop = stop
            
            s.add((chrom,start))
            s.add((chrom,stop))
            d[(chrom, start,stop)] = d[(chrom, start,stop)] + 1
            
            d_new[(chrom,start)] = d_new[(chrom,start)] + 1
            d_new[(chrom,stop)] = d_new[(chrom,stop)] - 1
            
            max_ = max(max_, value)

            min_ = min(min_, value)
            if value > 0:
                min_pos = min(min_pos, value )
            if value < 0:
                max_neg = max(max_neg, value)
                
            if value == 0:
                counts['value == 0'] = counts['value == 0']  + 1
            
            if value < 0:
                counts['value < 0'] = counts['value < 0']  + 1
                
            if value > 0:
                counts['value > 0'] = counts['value > 0']  + 1
            
            if value > 1:
                counts['value > 1'] = counts['value > 1']  + 1
            if value < -1:
                counts['value < -1'] = counts['value < -1']  + 1
                
            if value > 2:
                counts['value > 2'] = counts['value > 2']  + 1
            if value < -2:
                counts['value < -2'] = counts['value < -2']  + 1
            
            counts['all'] = counts['all']  + 1
            
        
#         break

# in order to use the positions, it is enough sets in the previous cell!!!!

chromosomes = ['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8', 'chr9', 'chr10', 'chr11', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17', 'chr18', 'chr19', 'chr20', 'chr21', 'chr22', 'chrX']

for chr_idx in tqdm(list(range(1,23))+list("X")):
    
    th = 0.6 if chr_idx == "X" else 0.8
    
    chr_now = 'chr'+str(chr_idx)
    chr_positions =sorted(map(lambda x : x[1], filter(lambda x : x[0] == chr_now, d_new.keys())))
    df = pd.DataFrame(index = chr_positions)
    indexes = df.index
    all_series = []
    
    for file_name in tqdm(files):
        new_series = pd.Series(np.nan, index=indexes, name=file_name)

        file_name = folder + file_name
        if check:
            pre_chr = None
        with open(file_name) as file:
            file = map(lambda x:x.strip().split("\t"), file)
            file = map(lambda x:(x[0],int(x[1]), int(x[2]),x[3],int(float(x[4])),float(x[5]),x[6]), file)
            file = filter(lambda x:x[6] == 'Y' , file)

            file = sorted(file)

            for (chrom, start,stop,strand,count,value,nocnv) in file:
                if chrom != chr_now:
                    continue
                new_series[(start<=indexes) & (indexes<=stop)] = value
        all_series.append(new_series)
        
    all_series = pd.concat(all_series, axis=1).T
      
    corrs = []
    has_nas = []
    
    from scipy.stats import pearsonr
    for x,y in tqdm(list(zip(all_series.columns[:-1],all_series.columns[1:]))):
        # in order understand the if the corrs change because of the nulls    
        has_na =  (sum(all_series[x].isna()) + sum(all_series[y].isna()) > 0)
        has_nas.append(has_na) 

        corr, z = pearsonr(all_series[x].fillna(0), all_series[y].fillna(0))
        #  print(x,y,corr, z)

        corrs.append(corr)
        
    df = pd.DataFrame([corrs,has_nas]).T
    all_series.fillna(0, inplace=True)

    init_idx = 0
        
    for col_idx in tqdm(df[(df[0] < th)].index):
        print(col_idx)
        
        final_cna_df[chr_now+"_"+str(all_series.columns[init_idx] + np.sign(init_idx) )+":"+str(all_series.columns[col_idx])] = all_series.iloc[:,init_idx:col_idx].mean(axis=1)
        init_idx = col_idx
        
    final_cna_df[chr_now+"_"+str(all_series.columns[init_idx])+":"+str(all_series.columns[-1])] = all_series.iloc[:,init_idx:].mean(axis=1)

    
print(final_cna_df.shape)
final_cna_df.to_pickle("../data/cna/tcga_cna_raw_all_samples_all_chr_0.8_threshold_0.6_X.pkl")
    
