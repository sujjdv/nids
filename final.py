# -*- coding: utf-8 -*-
"""
Created on Tue May  8 17:24:30 2018

@author: Sahana Hegde
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


#read dataset frm cloud


#s3=boto3.client('s3')
#obj=s3.get_object(Bucket='nids123',Key='test-nids.csv')
#obj1=s3.get_object(Bucket='nids123',Key='train-nids.csv')
#ini=obj['Body']
#ini1=obj1['Body']
#csv_test=ini.read().decode('utf-8')
#csv_train=ini1.read().decode('utf-8')
unswtest = pd.read_csv("test-nids.csv")
unswtrain = pd.read_csv("train-nids.csv")
unswtest1 = pd.read_csv("test-nids.csv")
unswtrain1 = pd.read_csv("train-nids.csv")

#categorizing into attacks and normal


print(unswtrain['attack_cat'].value_counts(),"\n")
print(unswtest['attack_cat'].value_counts())


unswtrain.loc[(unswtrain['attack_cat'] !='Normal'),'attack_cat'] = 1
unswtrain.loc[(unswtrain['attack_cat'] =='Normal'),'attack_cat'] = 0

print(unswtrain['attack_cat'].value_counts())

unswtest.loc[(unswtest['attack_cat'] !='Normal'),'attack_cat'] = 1
unswtest.loc[(unswtest['attack_cat'] =='Normal'),'attack_cat'] = 0

print(unswtest['attack_cat'].value_counts())

#feature extraction

attribute_encoder = feature_extraction.DictVectorizer(sparse=False)
label_encoder = preprocessing.LabelEncoder()


train_data= attribute_encoder.fit_transform(unswtrain.iloc[:,:-1].T.to_dict().values())
train_data_decoded = pd.DataFrame(train_data)
train_target_decoded =unswtrain['attack_cat']

 
test_data = attribute_encoder.transform(unswtest.iloc[:,:-1].T.to_dict().values())
test_data_decoded = pd.DataFrame(test_data)
test_target_decoded = unswtest['attack_cat']






#data-preprocessing/normalizing


standard_scaler = preprocessing.StandardScaler()
train_ratio_normalized_scaled_values = standard_scaler.fit_transform(train_data_decoded.values)
train_data_scaled_normalized = pd.DataFrame(train_ratio_normalized_scaled_values)
test_ratio_normalized_scaled_values = standard_scaler.fit_transform(test_data_decoded.values)
test_data_scaled_normalized = pd.DataFrame(test_ratio_normalized_scaled_values)
		
		


dos_train_x=train_data_scaled_normalized
dos_train_y=train_target_decoded
dos_train_y=dos_train_y.astype('int')
dos_test_x=test_data_scaled_normalized 
dos_test_y=test_target_decoded
dos_test_y=dos_test_y.astype('int')

#ALGORITHMS/VISUALIZATION
"""
def calc(mod):
     model=mod
     plt.plot(dos_test_y)
     plt.show()
     model.fit(dos_test_x,dos_test_y)
     predicted= model.predict(dos_train_x.head(100000))
     print(predicted)
     print("\nMean absolute error:")
     print(mean_absolute_error(dos_train_y.head(100000), predicted))
     plt.plot(predicted)
     plt.show()
     print("\n    Confusion matrix:")
     cm = confusion_matrix(dos_train_y.head(100000), predicted)
     print (cm)



print("SVM")
calc(svm.SVC(kernel='linear'))

print("\nNAIVE BAYES")
#calc(GaussianNB())

print("\n LOGISTIC REGRESSION")
#calc(LogisticRegression())


print("\n DECISION TREES")
#calc(DecisionTreeRegressor())


  """  
def unsup():
    clf = KMeans(n_clusters = 2 , random_state=2)
    clf.fit(dos_train_x.head(1000))
    predicted = clf.labels_
    print(predicted)
    print("\nMean absolute error:")
    #print(dos_train_y[0].head(1000))
    print(mean_absolute_error(dos_train_y.head(1000), predicted))
    plt.plot(predicted)
    plt.show()
    print("\n    Confusion matrix:")
    cm = confusion_matrix(dos_test_y.head(1000), predicted)
    print (cm)


unsup()   
###################################################################

# RECORDS INTO DIFFERENT ATTACKS

unsw1=unswtest1['attack_cat'].value_counts()


attribute_encoder = feature_extraction.DictVectorizer(sparse=False)
label_encoder = preprocessing.LabelEncoder()

train_data= attribute_encoder.fit_transform(unswtrain1.iloc[:,:-1].T.to_dict().values())
train_target=label_encoder.fit_transform(unswtrain1.iloc[:,-2])


train_data_decoded = pd.DataFrame(train_data)
train_target_decoded = pd.DataFrame(train_target)

test_data = attribute_encoder.transform(unswtest1.iloc[:,:-1].T.to_dict().values())
test_target= label_encoder.transform(unswtest1.iloc[:,-2])
		
test_data_decoded = pd.DataFrame(test_data)
test_target_decoded =pd.DataFrame(test_target)




#data-preprocessing/normalizing


standard_scaler = preprocessing.StandardScaler()
train_ratio_normalized_scaled_values = standard_scaler.fit_transform(train_data_decoded.values)
train_data_scaled_normalized = pd.DataFrame(train_ratio_normalized_scaled_values)
test_ratio_normalized_scaled_values = standard_scaler.fit_transform(test_data_decoded.values)
test_data_scaled_normalized = pd.DataFrame(test_ratio_normalized_scaled_values)
		

dos_train_x=train_data_scaled_normalized
dos_train_y=train_target_decoded
dos_test_x=test_data_scaled_normalized 
dos_test_y=test_target_decoded





"""
model_nb=GaussianNB()
model_nb.fit(dos_train_x.head(100000),dos_train_y[0].head(100000))
predicted= model_nb.predict(dos_test_x)
n1=g1=e1=f1=d1=r1=a1=b1=s1=w1=0  
for i in predicted:
    if i==0:
        n1+=1
    if i==1:
        g1+=1
    if i==2:
        e1+=1


model_tree=DecisionTreeRegressor()
model_tree.fit(dos_train_x.head(100000),dos_train_y[0].head(100000))
predicted= model_tree.predict(dos_test_x)


n=g=e=f=d=r=a=b=s=w=0      
for i in predicted:
    if i==0:
        n+=1
    if i==1:
        g+=1
    if i==2:
        e+=1
    if i==3:
        f+=1
    if i==4:
        d+=1
    if i==5:
        r+=1
    if i==6:
        a+=1
    if i==7:
        b+=1
    if i==8:
        s+=1
    if i==9:
        w+=1

     

objects = ('Analysis', 'Backdoor', 'DoS', 'Exploits', 'Fuzzers', 'Generic','Normal','Reconnaissance','Shellcode','Worms')
y_pos = np.arange(len(objects))
performance = [n,g1,e1,f,d,r,a,b,s,w]
 
plt.barh(y_pos, performance, align='center', alpha=0.5)
plt.yticks(y_pos, objects)
plt.xlabel('record count')
plt.title('No. of attacks')
 
plt.show()
print("predicted records:")
print('\nNormal:',a,'\nGeneric:',r,'\nExploits:',f,'\nFuzzers:', d,'\nDoS:',e1,'\nReconnaissance:',b,'\nAanalysis:', n,'\nBackdoor:',g1, '\nShellcode:',s,'\nWorms:',w,'\n')
print(unsw1)

"""