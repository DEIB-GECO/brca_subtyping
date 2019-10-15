import pandas as pd
import numpy as np
import urllib.request, urllib.parse, urllib.error
import sys

with open("../data/cna/cna_download_links.txt") as f:
    i=0
    for line in f:
        split = urllib.parse.urlsplit(line)
        file_type = split.path.split("/")[-1]
        filename = split.path.split("/")[-2]
        
        try:
            if(file_type == "region"):
                urllib.request.urlretrieve(line, '../data/cna/genosurf_download/'+filename+'region.bed')
            #else:
                #urllib.request.urlretrieve(line, '../data/cna/genosurf_download/'+filename+'metadata.bed')
        except (urllib.error.HTTPError, urllib.error.URLError) as err:
            print(err)
            print("Error code: {}".format(err.code))
        if i%10==0: print(i)
        i+=1
    print(i)