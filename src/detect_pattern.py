import numpy as np
import sys
from sklearn.cluster import DBSCAN
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
import glob


class LearnPattern:
	def __init__(self):
		self.learningFiles = [file for file in glob.glob("../trainData/generated/*.csv")]
		self.timestamp = []
		self.situation =[]
		self.latitude = []
		self.longitude = []
		self.angle = []
		self.number_samples = 0
		self.situationList = []
		self.macros_number = []
		self.mapping_macros_situation = {}


	def read(self,pattern_name):
		filename = "../globalParams/"+pattern_name
		count = 0
		for line in open(filename):
			if len(line.strip()) ==0:
				continue
			if '#' in line:
				count += 1
				continue
			if count == 1:
				self.situationList = [entry.strip() for entry in line.split(',')]
			if count == 2:
				key = line.split(':')[0].strip()
				situations = line.split(':')[1].strip().split(',')
				self.macros_number.append(key)
				self.mapping_macros_situation[key]= [situation.strip()for situation in situations]
		return

	def learn(self):
		for file in self.learningFiles:
			df = pd.read_csv(file,skipinitialspace=True, index_col = False)
			self.timestamp += [entry for entry in df.Timestamp]
			self.situation += [self.relabel(entry) for entry in df.Situation]
			self.latitude += [entry for entry in df.Latitude]
			self.longitude += [entry for entry in df.Longitude]
			self.angle += [entry for entry in df.Angle]
		self.number_samples = len(self.angle)

		self.generate_characteristics( "../patternLearned/" )

	def generate_characteristics(self, filename):
		situation_label=[]
		situation_mean=[]
		situation_stddev =[]
		for situ in set(self.situation):
			Situation_timestamp = []
			for entry in range(self.number_samples):
				if self.situation[entry]== situ :
					Situation_timestamp += [self.timestamp[entry] ]

			Situation_timestamp = np.array(Situation_timestamp).reshape(len(Situation_timestamp), 1)
			db = DBSCAN(eps=1000, min_samples=10, metric = 'euclidean').fit_predict(Situation_timestamp)
			number_clusters = len(set(db)) - (1 if -1 in set(db) else 0 )

			cluster_dictionary = {}
			for key in range(number_clusters):
				cluster_dictionary[key] = []
			for index in range(len(db)):
				if db[index] < 0:
					continue
				cluster_dictionary[db[index]].append(Situation_timestamp[index])
			for key in cluster_dictionary.keys():
				situation_label .append(situ)
				situation_mean.append(int(np.mean(cluster_dictionary[key])))
				situation_stddev.append(int(np.std(cluster_dictionary[key])))

		raw_data = {'Situation': situation_label,
		        'Mean': situation_mean,
		        'Std_dev':situation_stddev}
		df = pd.DataFrame(raw_data, columns = [ 'Situation','Mean','Std_dev'])
		df.to_csv(filename+"Situation-timestamp.csv", index = False)

		#club situatuon angles

		situation_label=[]
		angle_mean = []
		for situ in set(self.situation):
			Situation_angle = []
			for entry in range(self.number_samples):
				if self.situation[entry]== situ :
					Situation_angle += [self.angle[entry] ]
			# print (situ, set(self.situation))
			assert len(Situation_angle)  > 0
			Situation_angle = np.array(Situation_angle).reshape(len(Situation_angle), 1)
			db = DBSCAN(eps=8 , min_samples=10, metric = 'euclidean').fit_predict(Situation_angle)
			number_clusters = len(set(db)) - (1 if -1 in set(db) else 0 )

			cluster_dictionary = {}
			for key in range(number_clusters):
				cluster_dictionary[key] = []
			for index in range(len(db)):
				if db[index] < 0:
					continue
				cluster_dictionary[db[index]].append(Situation_angle[index])
			for key in cluster_dictionary.keys():
				situation_label .append(situ)
				angle_mean.append(int(np.mean(cluster_dictionary[key])))

		raw_data = {'Situation': situation_label,
		        'Angle': angle_mean}
		df = pd.DataFrame(raw_data, columns = [ 'Situation','Angle'])
		df.to_csv(filename+"Situation-angle", index = False)

	def relabel(self, situation):
		relabel = situation
		try:
			inv_macro_situation_map = {}
			for key in self.mapping_macros_situation.keys():
				item = self.mapping_macros_situation[key]
				try:
					inv_macro_situation_map[item] = key
				except:
					for entry in item:
						inv_macro_situation_map[entry] = key
			relabel = (self.macros_number.index(inv_macro_situation_map[self.situationList[situation]]))
		except:
			print ("Error in Relabelling")
		return relabel


if __name__ == "__main__":
	lp = LearnPattern()
	lp.read("model1")
	lp.learn()
