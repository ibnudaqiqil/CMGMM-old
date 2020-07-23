from abelab.EvolvingGMM2 import *
from abelab.Helper import *
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from pycm import *

import pandas as pd
import numpy as np
import os
import pickle   as pkl
import time
from pycm import *
import os


# ## Parameter Setting
 


#parameter setting
DETECTION_METHOD="mcWindow"
ADAPT_METHOD="isocombine"
WINDOW_SIZE=45
DELTA=0.0001
WARNING_ZONE_SIZE=10


# ## Load and Build Dataset


#Build dataset

df= pd.read_pickle("dataset/exported_800.pickle")
LABEL = df.label.unique()
trainDataset = df[df['status']==1]
testDataset = df[df['status']==2]
df.head()
testDataset_d= pd.read_pickle("dataset/exported_800_t1.pickle")




m0 =testDataset.mfcc.to_numpy()
m1 =testDataset_d.mfcc_t1.to_numpy()
l0 = testDataset.label.to_numpy()
l1 =testDataset_d.label.to_numpy()





#loaf base model
#model = pkl.load( open( "models/base_model.pkl", "rb" ) )
model = EvolvingGMM2( n_components_max=10,detection_method=DETECTION_METHOD,label=LABEL,threshold=WINDOW_SIZE,delta=DELTA,adapt_method=ADAPT_METHOD)
model.train("base",trainDataset,"mfcc")


DATASET = np.append(m0,m1)
LABELDATA = np.append(l0,l1)

print("Active Prediction Base Model Using {0}".format(ADAPT_METHOD))
start_time = time.time()
result = model.activePrediction("base",DATASET,LABELDATA,warningZoneLimit=WARNING_ZONE_SIZE)
print("Testing Time :  %s seconds ---" % (time.time() - start_time))
acc=[]
acc_window=[]
cm3 = ConfusionMatrix(actual_vector=LABELDATA, predict_vector=result)
cm3.save_html(os.path.join("report","XACTIVE-ADAPT-{0}-{1}-{2}".format(ADAPT_METHOD,DETECTION_METHOD,WINDOW_SIZE)))

for x in range(int(len(result)/100)):
	akhir = (x+1)*100
	window = (x)*100
	
	acc2 = accuracy_score(LABELDATA[0:akhir], np.array(result[0:akhir]))
	accw2 = accuracy_score(LABELDATA[window:akhir], np.array(result[window:akhir]))
	acc.append(acc2)
	acc_window.append(accw2)

dfx= pd.DataFrame(acc)
dfx['acc_window'] = acc_window;
dfx.to_excel(os.path.join("report","xACTIVE-ADAPT-{0}-{1}-{2}.xls".format(ADAPT_METHOD,ADAPT_METHOD,WINDOW_SIZE)))


