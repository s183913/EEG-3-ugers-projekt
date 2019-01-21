from sklearn.model_selection import train_test_split
import numpy as np  
import scipy.io as sio
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import datetime
import numpy as np  
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt

####### DATA LOAD #######

# Author: Greta Tuckute, grtu@dtu.dk 
# Modified by: Alex Németh, Anders Henriksen and Mikkel Danielsen @DTU January 2019

_dir='/Users/mikkeldanielsen/Desktop/preprocessed_EEG/'
filename = 'sub1_preprocessed_EEG.mat'
categories = 'sub1_categories.mat'

# male = 9 mellemrum
# female = 7 mellemrum
gender = 'male         '

DATA=sio.loadmat(_dir+filename)
eeg=DATA['eeg_events']
eeg=np.reshape(eeg,(32*550,1200)).T

cat=sio.loadmat(_dir+categories)
cat=cat['cat']

cat[cat == "indoor       "] = 0
cat[cat == gender] = 1

x = eeg
y = cat.ravel()

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size = 0.2)

####### HYPER PARAMETER OPTIMIZATION #######

# Authors: Alex Németh, Anders Henriksen and Mikkel Danielsen @DTU January 2019

Cs=[1,1.5,1.75,2.5,5,10,20,40,80,160]
gammas =[6.8*10**(-10), 2.1*10**(-9),6.8*10**(-9),2.1*10**(-8),6.8*10**(-8),2.5*10**(-7),5*10**(-7),2.5*10**(-6),5*10**(-6),2.5*10**(-5)]
number_of_folds = 5

time_before = datetime.datetime.now()
print(time_before)

tunedParameters = [{"kernel": ["rbf"], "gamma": gamma, "C": C}]

clf = GridSearchCV(SVC(), tunedParameters, cv = folds)

print("Best: ", clf.best_params_)  

scores=[]
for param, score in zip(clf.cv_results_["params"], clf.cv_results_["mean_test_score"]):
    scores.append(score)
print(scores)

time_after = datetime.datetime.now()
print(time_after - time_before)

####### TEST MODEL ACCURACY #######

# Authors:  Alex Németh, Anders Henriksen and Mikkel Danielsen @DTU January 2019

svclassifier = SVC(kernel='rbf',gamma=gamma,C=C)
svclassifier.fit(x_train,y_train)

y_pred = svclassifier.predict(x_val)

print(confusion_matrix(y_val,y_pred))
print(classification_report(y_val, y_pred))
    

####### HEAT MAP #######

# Author: Greta Tuckute, grtu@dtu.dk 
# Modified by: Alex Németh, Anders Henriksen and Mikkel Danielsen @DTU January 2019

title = 'Subject 1 - Heat Map'
  
C_vals = tunedParameters[0]['C']*10

num_gamma = len(tunedParameters[0]['gamma'])
num_c = len(tunedParameters[0]['C'])

gamma_vals = np.array([])
for i in range(num_gamma):
    gamma_vals = np.append(gamma_vals,np.repeat(tunedParameters[0]['gamma'][i],num_gamma ))

reshape_scores = np.reshape(scores,[num_c,num_gamma])
reshape_scores = np.flip(reshape_scores, axis = 0)

random_chance = 0.5
C_range = list(reversed(C_vals[0:num_c])) # Add list with C vals here
gamma_range = gamma_vals[0::num_gamma]


class MidpointNormalize(Normalize):

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


plt.figure(figsize=(8, 6))
plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
plt.imshow(reshape_scores, interpolation='nearest', cmap=plt.cm.RdBu_r, # evt. cmap=plt.cm.RdBu_r 
           norm=MidpointNormalize(vmin=random_chance - 0.1, midpoint=random_chance),vmax=random_chance+0.1)
plt.xlabel('gamma')
plt.ylabel('C',rotation=360)
plt.colorbar()
plt.xticks(np.arange(len(gamma_range)), 
np.round(gamma_range,20),rotation=45)
plt.yticks(np.arange(len(C_range)), C_range)
plt.title(title)
plt.show()
