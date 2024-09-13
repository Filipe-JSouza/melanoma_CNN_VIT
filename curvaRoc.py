import os
import sys
import numpy as np
import scipy.io as sio
import tensorflow as tf
import matplotlib.pyplot as plt
import yagmail
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm, metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


dirname1=('/tcc/complete_dataset/embedding_melanoma')
dirname2=('/tcc/complete_dataset/embedding_naevus')

features_melanoma=sio.loadmat(dirname1 +'/features.mat')
features_melanoma=features_melanoma['features']
features_naevus=sio.loadmat(dirname2+'/features.mat')
features_naevus=features_naevus['features']

#Separando conjuntos
melanoma_train, melanoma_teste=train_test_split(features_melanoma,test_size=0.2,random_state=10)
naevus_train, naevus_teste=train_test_split(features_naevus,test_size=0.2, random_state=10)


#Criando Rotulos
linhas=len(melanoma_train)
label1=np.ones(linhas)
linhas=len(naevus_train)
label2=np.zeros(linhas)
xlabel=np.concatenate((label1, label2),axis=0)
xtrain=np.concatenate((melanoma_train, naevus_train),axis=0)


linhas=len(melanoma_teste)
label1=np.ones(linhas)
linhas=len(naevus_teste)
label2=np.zeros(linhas)
ylabel=np.concatenate((label1, label2),axis=0)
yteste=np.concatenate((melanoma_teste, naevus_teste),axis=0)


#---------------------------------------------------------------------------------------
#scalertrain
scaler = StandardScaler()
xtrain = scaler.fit_transform(xtrain)
yteste = scaler.transform(yteste)



#---------------------------------------------------------------------------------------
#SVM hiperparametros
Kernel = 'rbf'
c= 10
gam = 0.001
#SVM(Support Vector Machine)
clf=svm.SVC(C=c, kernel=Kernel, gamma=gam,  probability=True)
clf.fit(xtrain,xlabel)
SVMproba = clf.predict_proba(yteste)
SVMauc = roc_auc_score(ylabel, SVMproba[:,-1])

#---------------------------------------------------------------------------------------
#KNN hiperparametros
n=7
#KNN(K-Nearest Neighbors)
knn=KNeighborsClassifier(n_neighbors=n)
knn.fit(xtrain,xlabel)
KNNproba= knn.predict_proba(yteste)
KNNauc=roc_auc_score(ylabel,KNNproba[:,-1])

#---------------------------------------------------------------------------------------
#Naive Bayes
nb=GaussianNB()
nb.fit(xtrain,xlabel)
NBproba=clf.predict_proba(yteste)
NBauc=roc_auc_score(ylabel,NBproba[:,-1])

#---------------------------------------------------------------------------------------
#LDA hiperparametro
#LDA(Linear Discriminat Analysis)
lda=LinearDiscriminantAnalysis(solver='svd', shrinkage=None, n_components=1)
lda.fit(xtrain,xlabel)
LDAproba=lda.predict_proba(yteste)
LDAauc= roc_auc_score(ylabel,LDAproba[:,-1])


#---------------------------------------------------------------------------------------
#roc curve

fpr_SVM, tpr_SVM, th_SVM= roc_curve(ylabel, SVMproba[:,1])
fpr_KNN, tpr_KNN, th_KNN = roc_curve(ylabel, KNNproba[:,1])
fpr_NB, tpr_NB, th_NB= roc_curve(ylabel, NBproba[:,1])
fpr_LDA, tpr_LDA, th_LDA= roc_curve(ylabel, LDAproba[:,1])

print('t')
roc_auc_SVM = auc(fpr_SVM, tpr_SVM)
roc_auc_KNN = auc(fpr_KNN, tpr_KNN)
roc_auc_NB = auc(fpr_NB, tpr_NB)
roc_auc_LDA = auc(fpr_LDA, tpr_LDA)

plt.figure(figsize=(8, 6))
plt.plot(fpr_SVM, tpr_SVM, color='darkorange', lw=2, label=f'SVM (AUC = {roc_auc_SVM:.2f})')
#plt.plot(fpr_KNN, tpr_KNN, color='green', lw=2, label=f'KNN (AUC = {roc_auc_KNN:.2f})')
#plt.plot(fpr_NB, tpr_NB, color='blue', lw=2, label=f'NB (AUC = {roc_auc_NB:.2f})')
#plt.plot(fpr_LDA, tpr_LDA, color='purple', lw=2, label=f'LDA (AUC = {roc_auc_LDA:.2f})')

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taxa de Falso Positivo')
plt.ylabel('Taxa de Verdadeiro Positivo')
plt.title('Curva ROC para Modelos')
plt.legend(loc='lower right')

plt.savefig('imgs_curvas/curva_roc.png')

