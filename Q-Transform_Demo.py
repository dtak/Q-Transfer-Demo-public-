#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 10:51:41 2018

@author: arjumand
"""

#%%

import numpy as np 
from sklearn.decomposition import NMF

# Dimensions of Small Data Set to learn Transformation Matrices QA, QW from
D1 = 12
R1 = 5
N1 = 12

#Generate Basis, Weights and Data
A1 = np.random.rand(D1,R1)
W1 = np.random.rand(R1,N1)
Data1 = np.dot(A1, W1) 

# Transfer learning rank parameters
R_T = 3
R_SVD = 3
R_NMF = 3


#Compute SVD of Data
u,s,v = np.linalg.svd(Data1, full_matrices=False)
S = np.diag(np.sqrt(s))
A_SVD_1 = np.dot(u[:,0:R_SVD],S[0:R_SVD,0:R_SVD])
W_SVD_1 = np.dot(S[0:R_SVD,0:R_SVD],v[0:R_SVD,:])


QA_list = []
QW_list = []
for i in range(5):
    #Compute NMFs of Data
    model =  NMF(n_components=R_T, init = 'random', solver = 'cd', max_iter = 200, beta_loss = 'frobenius')
    model.fit(Data1)
    A_NMF_1 = model.transform(Data1)
    W_NMF_1 = model.components_
    
    #Relate SVD and NMF using Transformation Matrices QA, QW
    QA = np.linalg.lstsq(A_SVD_1,A_NMF_1)[0]
    QW = np.linalg.lstsq(W_SVD_1.T,W_NMF_1.T)[0].T

    QA_list.append(QA)
    QW_list.append(QW)
    

# New Larger Dataset Dimensions
D2 = 500
R2 = 3
N2 = 500

# In this demo the new dataset is synthetic so we generate it
A2 = np.random.rand(D2,R2)
W2 = np.random.rand(R2,N2)
Data2 = np.dot(A2, W2) 

#Compute SVD of New Data
u,s,v = np.linalg.svd(Data2, full_matrices=False)
S = np.diag(np.sqrt(s))
A_SVD_2 = np.dot(u[:,0:R_SVD],S[0:R_SVD,0:R_SVD])
W_SVD_2 = np.dot(S[0:R_SVD,0:R_SVD],v[0:R_SVD,:])

#Find Q-Transform Initializations
Normalized_Reconstruction_Q = []
for i in range(len(QA_list)):
    QA = QA_list[i]
    QW = QW_list[i]
    AQ = np.dot(A_SVD_2, QA)
    WQ = np.dot(QW, W_SVD_2)
    
    A0 = np.clip(AQ, 0, np.inf)
    W0 = np.clip(WQ, 0, np.inf)
    Normalized_Reconstruction_Q.append(np.linalg.norm(np.dot(A0,W0) - Data2)/np.linalg.norm(Data2))

print "[Q-Transform] Init Error Range: [",np.min(Normalized_Reconstruction_Q), ', ', np.max(Normalized_Reconstruction_Q), ']'  

#Compare to Random Initializations
Normalized_Reconstruction_Random = []
for i in range(len(QA_list)):
    avg = np.sqrt(np.mean(Data2) / R_NMF)
    A_rand = np.abs(avg * np.random.randn(D2, R_NMF))
    W_rand = np.abs(avg * np.random.randn(R_NMF, N2))
    Normalized_Reconstruction_Random.append(np.linalg.norm(np.dot(A_rand,W_rand) - Data2)/np.linalg.norm(Data2))

print "[Random] Init Error Range: [",np.min(Normalized_Reconstruction_Random), ', ', np.max(Normalized_Reconstruction_Random), ']'  


