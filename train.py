# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 20:46:27 2021

@author: DP
"""

import numpy as np 
import matplotlib.pyplot as plt
import os
from sklearn.cluster import KMeans
import math
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeRegressor
from sklearn import datasets, preprocessing,feature_extraction
from sklearn import linear_model, svm, metrics, tree, ensemble
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LogisticRegression
#import boto3
import pandas as pd
from sklearn.naive_bayes import GaussianNB
import sys
from io import StringIO
import csv


unswtrain=pd.read_csv("train-nids.csv")
unswtrain=pd.DataFrame(unswtrain.head(120000))



with open('normal.csv', 'w', newline='') as w:
    writer=csv.writer(w)
    writer.writerow(unswtrain.columns.values)
    for (p,row) in unswtrain.iterrows():
        if (row.loc['attack_cat']=='Normal'):
            writer.writerow(row)
            
with open('worms.csv', 'w', newline='') as w:
    writer=csv.writer(w)
    writer.writerow(unswtrain.columns.values)
    for (p,row) in unswtrain.iterrows():
        if (row.loc['attack_cat']=='Worms'):
            writer.writerow(row)            


with open('Shellcode.csv', 'w', newline='') as w:
    writer=csv.writer(w)
    writer.writerow(unswtrain.columns.values)
    for (p,row) in unswtrain.iterrows():
        if (row.loc['attack_cat']=='Shellcode'):
            writer.writerow(row)    


with open('Backdoor.csv', 'w', newline='') as w:
    writer=csv.writer(w)
    writer.writerow(unswtrain.columns.values)
    for (p,row) in unswtrain.iterrows():
        if (row.loc['attack_cat']=='Backdoor'):
            writer.writerow(row)    


with open('Analysis.csv', 'w', newline='') as w:
    writer=csv.writer(w)
    writer.writerow(unswtrain.columns.values)
    for (p,row) in unswtrain.iterrows():
        if (row.loc['attack_cat']=='Analysis'):
            writer.writerow(row)    

with open('DoS.csv', 'w', newline='') as w:
    writer=csv.writer(w)
    writer.writerow(unswtrain.columns.values)
    for (p,row) in unswtrain.iterrows():
        if (row.loc['attack_cat']=='DoS'):
            writer.writerow(row)    

with open('Reconnaissance.csv', 'w', newline='') as w:
    writer=csv.writer(w)
    writer.writerow(unswtrain.columns.values)
    for (p,row) in unswtrain.iterrows():
        if (row.loc['attack_cat']=='Reconnaissance'):
            writer.writerow(row)    

with open('Fuzzers.csv', 'w', newline='') as w:
    writer=csv.writer(w)
    writer.writerow(unswtrain.columns.values)
    for (p,row) in unswtrain.iterrows():
        if (row.loc['attack_cat']=='Fuzzers'):
            writer.writerow(row)    

with open('Exploits.csv', 'w', newline='') as w:
    writer=csv.writer(w)
    writer.writerow(unswtrain.columns.values)
    for (p,row) in unswtrain.iterrows():
        if (row.loc['attack_cat']=='Exploits'):
            writer.writerow(row)    

with open('Generic.csv', 'w', newline='') as w:
    writer=csv.writer(w)
    writer.writerow(unswtrain.columns.values)
    for (p,row) in unswtrain.iterrows():
        if (row.loc['attack_cat']=='Generic'):
            writer.writerow(row)    
            

            
    