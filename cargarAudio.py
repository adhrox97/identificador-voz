#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from wav_rw import wavread
from scipy.fftpack import fft
from python_speech_features import mfcc
from pandas import Series
from sklearn.linear_model import LogisticRegression
from sklearn import svm
import pickle

def vad(logE,trh):
    naverage = 10
    s=Series(logE)
    logE = (s.rolling(window=naverage)).mean().values
    logE[0:naverage-1]=logE[naverage:].min()
    logE=np.roll(logE, -int(naverage/2), axis=0)    
    wvad=logE>trh
    return wvad

def center(X3):
    Mu =np.mean(X3, axis = 0)
    newX = X2 - Mu
    return newX, Mu

def standardize(X2):
    sigma = np.std(X2, axis = 0)
    newX = center(X3)/sigma
    return newX

lista='Datos/lista_voces.txt'
hf=open(lista,'r')
lines=hf.readlines()
hf.close()

X=[]
caract=[]
veCar=[]
veLab=[]
'''    
audio,label=lines[0].split('\t')
(r,s)=wavread(audio)
MFCC=mfcc(s,samplerate=r,winlen=0.025,winstep=0.01,numcep=13,nfilt=27,nfft=512,lowfreq=0,preemph=0.97,ceplifter=22,appendEnergy=True,winfunc=np.hamming)
x=MFCC.mean(axis=0)
X.append(x)
E=MFCC[:,0]
act=vad(E, -5)
MFCC=MFCC[act,:]
matriz=[]
caract=abs(act*E)
plt.subplot(211)
plt.plot(s)
plt.subplot(212)
plt.plot(x/abs(x).max())

pr=np.zeros((3,3))
  
cova=np.cov(MFCC,rowvar=False)
vecc=cova[np.triu_indices(13, k = 0)]

'''
for i in range(len(lines)):
    audio,label=lines[i].split('\t')
    (r,s)=wavread(audio)
    MFCC=mfcc(s,samplerate=r,winlen=0.025,winstep=0.01,numcep=20,nfilt=27,nfft=512,lowfreq=0,preemph=0.97,ceplifter=22,appendEnergy=True,winfunc=np.hamming)
    #x=MFCC.mean(axis=0)
    #X.append(x)
    #E=MFCC[:,0]
    #act=vad(E, -5)
    #MFCC=MFCC[act,:]
    #caract=abs(E*act)
    #veCar.append(caract)
    veLab.append(int(label[:-1]))
    cova=np.cov(MFCC,rowvar=False)
    vecc=cova[np.triu_indices(13, k = 0)]
    x=vecc
    X.append(x)
C=np.array(X)
L=np.array(veLab)

#normalizmos
for ii in range(len(lines)):
    C[ii]=C[ii]/(abs(C[ii]).max())
  
#C0=C[(L==0),:]
C1=C[(L==1),:]
C2=C[(L==2),:]
C3=C[(L==3),:]
#C4=C[(L==4),:]
#C5=C[(L==5),:]
#C6=C[(L==6),:]
#C7=C[(L==7),:]
#C8=C[(L==8),:]
#C9=C[(L==9),:]
#LB0=L[(L==0)]
LB1=L[(L==1)]
LB2=L[(L==2)]
LB3=L[(L==3)]
#LB4=L[(L==4)]
#LB5=L[(L==5)]
#LB6=L[(L==6)]
#LB7=L[(L==7)]
#LB8=L[(L==8)]
#LB9=L[(L==9)]



X_train=np.vstack((C1[:32],C2[:32],C3[:32]))
Y_train=np.hstack((LB1[:32],LB2[:32],LB3[:32]))

X_test=np.vstack((C1[32:],C2[32:],C3[32:]))
Y_test=np.hstack((LB1[32:],LB2[32:],LB3[32:]))

#X_train=np.vstack((C1,C2,C3))
#Y_train=np.hstack((LB0,LB1,LB2))

#X_test=np.vstack((C0[32:],C1[32:],C2[32:],C3[32:],C4[32:],C5[32:],C6[32:],C7[32:],C8[32:],C9[32:]))
#Y_test=np.hstack((LB0[32:],LB1[32:],LB2[32:],LB3[32:],LB4[32:],LB5[32:],LB6[32:],LB7[32:],LB8[32:],LB9[32:]))


#Eleccion de los mejores parametros con el mayor porcentaje de acierto
'''
Ce=np.array([0.01, 0.03, 0.1,0.3, 1,3,10,30])       
gamma=np.array([0.01, 0.03, 0.1,0.3, 1,3,10,30])
nC = 0
nG = 0
porAnt = 0
porBest = 0
porAct = 0
for c in Ce:
    for g in gamma:
        clf_rbf = svm.SVC(kernel='rbf', gamma=g, C=c)
        clf_rbf.fit(X_train,Y_train)
        Y_pred = clf_rbf.predict(X_test)
        porAct = (Y_test==Y_pred).sum()/Y_test.size*100
        if (porAct>porAnt):
            porBest=porAct
            nG=g
            nC=c
            porAnt=porAct
            porAct=None
print("el mejor porcentaje de acierto es =",porBest)
print("el mejor valor de gamma es =",nG)
print("el mejor valor de C es =",nC)
'''
nG=0.3
nC=3
clf_rbf = svm.SVC(kernel='rbf', gamma=nG, C=nC)
#entrenar
clf_rbf.fit(X_train,Y_train)
#probar
Y_pred = clf_rbf.predict(X_test)
porAct = (Y_test==Y_pred).sum()/Y_test.size*100
print("El porcentaje de acierto es =",porAct)
pickle.dump(clf_rbf, open("SVM.p","wb"))
plt.show()

