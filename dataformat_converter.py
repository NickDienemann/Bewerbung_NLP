"""
This file holds functionality to convert the tsv data into csv files
"""

#imports 
import csv
import codecs
import pandas as pd

#open file and read lines
"""
with codecs.open("data/v1/v1/dev.tsv","r","utf-8") as f:
    tsv_file=csv.reader(f,delimiter="\t")
    i=0
    for line in tsv_file:
        print(line)
        i+=1
        if i==4:
            break
"""

tsv_data = pd.read_csv("data/v1/v1/dev.tsv", sep= "\t")
print(tsv_data.head())
