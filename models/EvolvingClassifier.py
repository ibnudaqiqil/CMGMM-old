import warnings
warnings.filterwarnings("ignore")
from sklearn.mixture import GaussianMixture
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils import check_random_state
from sklearn.mixture.base import _check_X
from sklearn.mixture.gaussian_mixture import _compute_precision_cholesky
from skmultiflow.drift_detection import PageHinkley
from skmultiflow.drift_detection.adwin import ADWIN
from skmultiflow.drift_detection.eddm import EDDM
from skmultiflow.drift_detection import DDM
from scipy.stats import multivariate_normal as MVN
#import warnings
import numpy as np
import pandas as pd
from collections import defaultdict
from numpy.linalg import det

import copy

class EvolvingClassifier():
	model= defaultdict()
	ph_test=defaultdict()
	drift_detector =defaultdict()
	index=defaultdict()
	prevdDriftData={}
	def __init__(self, 
			strategy="active",#active, passive
			adaptor= None ,
			detector = None,
			label=["a","b"]):
		'''
		Detection Method : [pageHinkley, adwin,eddm,ddm]
		'''
		self.strategy = strategy
		self.setLabel(label)

		self.setAdaptor(adaptor)
		self.use_detector = False
		if detector is not None:
			self.use_detector = True
			self.setDetector(detector)

	
		self.driftData = {}
		columns = ['label','index','diff', 'diff_sum','status']
		self.driftLog = pd.DataFrame(columns=columns)
	

	def setDetector(self,detector):
		for scene_label in self.label:	  
			self.drift_detector[scene_label] = copy.deepcopy(detector)   
	def setAdaptor(self,adaptor):
		for scene_label in self.label:
			#print("set adaptoer :",scene_label)	  
			self.model[scene_label] = copy.deepcopy(adaptor)   
	def setLabel(self,label):
		self.label = label


	def print(self, text):
		return 0
		print(text)

	def resetDriftData(self):
		for scene_label in self.label:
			self.driftData[scene_label]=[]
			self.prevdDriftData[scene_label] =[]

	def train(self,data,column_label,column_data):
	
	
			
		for scene_label in self.label:
			#print(self.model[scene_label])
			self.model[scene_label].fit(np.vstack( data[data[column_label]==scene_label][column_data].to_numpy())) 		
	



	def predict(self, data,column_label,column_data):
		predicted=[]
		wrong=0
		
		for index, row in data.iterrows():

			predicted_label,highest_prob = self.score(row[column_data])	
			predicted.append(predicted_label)
			if (self.use_detector):
				self.drift_detector[row['label']].add_element(highest_prob)
				   

			if(row[column_label]!=predicted_label):
				wrong=wrong+1

		
		return predicted

	def score(self, data):
		'''
		 Memperediksi satu data mfcc
		'''
		highest_prob=-np.inf
		for scene_label in self.label:
			#compute likelihood to the labeled model
			logls = self.model[scene_label].score([data])
			
			#select the highest likelihood as the predicted	
			if(highest_prob<logls):
				highest_prob=logls
				predicted_label = scene_label
			
		return predicted_label,highest_prob

	def activeLearning(self,data,label, warningZoneLimit=4):

		predicted=[]
		wrong={}
		warningZone=False
		jumlah_element={}
		warningZoneCount={}
		logDrift=[]
		self.resetDriftData()
		for index  in  range(data.shape[0]):
			row = data[index]
			key_label = label[index]
			predicted_label,highest_prob = self.score(row)

			predicted.append(predicted_label)

			
			self.drift_detector[key_label].add_element(highest_prob)
			isDetected = self.drift_detector[key_label].detected_change()

			if(warningZone==True):
				#add data along warning zone
				self.driftData[key_label].append(row)
				 
		
			salah=0
			if(label[index]!=predicted_label):
				salah=1
				
			wrong[key_label] = wrong.get(key_label, 0) +salah
			jumlah_element[key_label] = jumlah_element.get(key_label, 0) +1
			if(isDetected):
				#print("DETECTED:",key_label," at ",index)			
				warningZone=True
				warningZoneCount[key_label] = warningZoneCount.get(key_label, 0) + 1
				if(warningZoneCount[key_label] == warningZoneLimit):
					#print(key_label," adapted :",index,"  jumlah_element : ",jumlah_element[key_label],"  wrong : ",wrong[key_label])
					#do adaptation
					dd= np.array(self.driftData[key_label])
					
					self.model[key_label].fit(dd)
					logDrift.append(index) 
					#flush data and market
					self.prevdDriftData[key_label]= copy.deepcopy(self.driftData[key_label])
					if(len(self.driftData[key_label])>=500):
						self.driftData[key_label]=[]
					warningZoneCount[key_label]=0
					warningZone=False
			

		
		return predicted, logDrift



	