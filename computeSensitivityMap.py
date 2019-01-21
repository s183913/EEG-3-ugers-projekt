
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC
import os
from scipy.spatial import distance
import scipy.io as sio

# Author: Greta Tuckute, grtu@dtu.dk 
# Modified by: Alex Németh, Anders Henriksen and Mikkel Danielsen @DTU January 2019

###### DATA LOAD ######
_dir='/Users/mikkeldanielsen/Desktop/preprocessed_EEG/'
filename = 'sub3_preprocessed_EEG.mat'
categories = 'sub3_categories.mat'
    
DATA=sio.loadmat(_dir+filename)
eeg=DATA['eeg_events']
eeg=np.reshape(eeg,(32*550,1200)).T

cat=sio.loadmat(_dir+categories)
cat=cat['cat']

cat[cat == "indoor       "] = -1
cat[cat == "female       "] = 1

eeg = [eeg[:,k] for k in range(0,17600,10)]
eeg = np.array([eeg])
eeg = eeg[0].T

X = eeg
y = cat
C_val = 1
gamma_val = 2.5*10**(-6)
no_channels = 32
no_timepoints = 55

###### SENSITIVITY MAP ######
### Compute SVM classifier ###
y = np.squeeze(y)
classifier = SVC(C=C_val, gamma=gamma_val)
clf = classifier.fit(X, y)

### Extract classifier model coefficients and add zero indices ### 
coefficients = clf.dual_coef_
support_array = clf.support_

coefficients = np.squeeze(coefficients)

trials = len(X[:,0])
features = len(X[0])
alpha = np.zeros(trials)
alpha[support_array] = coefficients
alpha = np.squeeze(alpha)

no_zero_indices = trials - len(support_array)

### Compute training kernal matrix, K ###
M = distance.pdist(X,'euclidean')

M_exp = np.exp(gamma_val*(-(np.square(M))))
K = distance.squareform(M_exp) 

### Compute sensitivity map ###
X = np.transpose(X) # Obtain training examples in columns for further computation

mapping = np.matmul(X,np.matmul(np.diag(alpha),K))-(np.matmul(X,(np.diag(np.matmul(alpha,K)))))
s = np.sum(np.square(mapping),axis=1)/np.size(alpha) 

s_matrix = np.reshape(s,[no_channels,no_timepoints])

### Generation of sensitivity map plot ###
channel_vector = ['P7','P4','Cz','Pz','P3','P8','O1','O2','T8','F8','C4','F4',
              'Fp2','Fz','C3','F3','Fp1','T7','F7','Oz','PO3','AF3','FC5',
              'FC1','CP5','CP1','CP2','CP6','AF4','FC2','FC6','PO4']

time_vector = ['-100','0','100','200','300','400','500','600','700','800','900','1000']

plt.matshow(s_matrix,aspect = 1)
plt.xlabel('Time (ms)')
plt.xticks(np.arange(0,55,5),time_vector)
plt.yticks(np.arange(no_channels),channel_vector)
plt.ylabel('EEG channels')
plt.colorbar()
plt.title('Sensitivity map SVM RBF kernel',y=1.05)

plt.rcParams['figure.figsize'] = 5000,100
plt.savefig('sub2.jpg')

print('Sensitivity map computed. Number of support vectors for the classifier: {0}.'.format(len(support_array)))

