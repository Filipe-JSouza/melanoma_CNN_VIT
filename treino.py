import os
import sys
import numpy as np
import scipy.io as sio
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm, metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


dirname1=('/tcc/complete_dataset/embedding_melanoma')
dirname2=('/tcc/complete_dataset/embedding_naevus')

features_melanoma=sio.loadmat(dirname1 +'/features.mat')
features_melanoma=features_melanoma['features']
features_naevus=sio.loadmat(dirname2+'/features.mat')
features_naevus=features_naevus['features']

#Separando conjuntos
melanoma_train, melanoma_teste=train_test_split(features_melanoma,test_size=0.3,random_state=10)
naevus_train, naevus_teste=train_test_split(features_naevus,test_size=0.3, random_state=10)

#Separando conjuntos
melanoma_teste, melanoma_validation=train_test_split(melanoma_teste,test_size=0.3, random_state=10)
naevus_teste, naevus_validation=train_test_split(naevus_teste,test_size=0.3, random_state=10)


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


linhas=len(melanoma_validation)
label1=np.ones(linhas)
linhas=len(naevus_validation)
label2=np.zeros(linhas)
zlabel=np.concatenate((label1, label2),axis=0)
zvalidation=np.concatenate((melanoma_validation, naevus_validation), axis=0)



#-----------------------------------------------------------------------------------------
#scalertrain
scaler = StandardScaler()
xtrain = scaler.fit_transform(xtrain)
yteste = scaler.transform(yteste)
zvalidation = scaler.transform(zvalidation)



#----------------------------------------------------------------------------------------
#SVM hiperparametros
Kernel = 'rbf'
Co=[0.01, 0.1, 1, 10, 100]
Gamma = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
results_SVM = np.zeros((len(Co), len(Gamma)))
#SVM(Support Vector Machine)
for c_idx, c in enumerate(Co):
 for gam_idx, gam in enumerate(Gamma):
  clf=svm.SVC(C=c, kernel=Kernel, gamma=gam, probability=True)
  clf.fit(xtrain,xlabel)
  SVMpred=clf.predict(zvalidation)
  SVMproba = clf.predict_proba(zvalidation)
#  SVMaccuracy=accuracy_score(zlabel, SVMpred)
  SVMauc = roc_auc_score(zlabel, SVMproba[:,-1])
#  results_SVM[c_idx, gam_idx]=SVMaccuracy
  results_SVM[c_idx, gam_idx]=SVMauc

posmax_SVM = results_SVM.argmax()
lmax = posmax_SVM//len(Gamma)
cmax = posmax_SVM%len(Gamma)
Copt = Co[lmax]
Gammaopt = Gamma[cmax]
#print('SVM')
#print(Copt, Gammaopt)
#----------------------------------------------------------------------------------------
#KNN hiperparametros
N_neighbros=[1,3,5,7,9]
results_KNN=np.zeros(len(N_neighbros))
#KNN(K-Nearest Neighbors)
for n_idx, n in enumerate(N_neighbros):
 knn=KNeighborsClassifier(n_neighbors=n)
 knn.fit(xtrain,xlabel)
# KNNpred=knn.predict(zvalidation)
 KNNproba= knn.predict_proba(zvalidation)
# KNNaccuracy=accuracy_score(zlabel,KNNpred)
 KNNauc=roc_auc_score(zlabel,KNNproba[:,-1])
 results_KNN[n_idx]=KNNauc

posmax_KNN = results_KNN.argmax()
#print('KNN')
#print(N_neighbros[posmax_KNN])
#--------------------------------------------------------------------------------------
#Naive Bayes
nb=GaussianNB()
nb.fit(xtrain,xlabel)
NBproba=clf.predict_proba(zvalidation)
#NBaccuracy=accuracy_score(zlabel,NBpred)
NBauc=roc_auc_score(zlabel,NBproba[:,-1])
#print('NBayes')
#print(NBauc)
#-------------------------------------------------------------------------------------
#LDA hiperparametro
#LDA(Linear Discriminat Analysis)
lda=LinearDiscriminantAnalysis(solver='svd', shrinkage=None, n_components=1)
lda.fit(xtrain,xlabel)
LDAproba=lda.predict_proba(zvalidation)
#LDAaccuracy=accuracy_score(zlabel,LDApred)
LDAauc= roc_auc_score(zlabel,LDAproba[:,-1])
#print('LDA')
#print(LDAauc)
