import networkx as nx
import random as rnd
import pandas as pd
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity


class Pattern:
	"""docstring for Pattern"""
	def __init__(self):
		self.macros_number = []
		self.mapping_macros_situation={}
		self.graph = []
		self.testFiles = 5
		self.trainFiles = 80
		self.randomFiles = 5
		self.number_events = 500
		self.weight_macros = []
		self.situationList = []
		self.macro_mean_time = {}
		self.macro_standard_deviation ={}
		self.test_deviation_multiplier = 1.00
		self.train_deviation_multiplier = 1.00
		self.location_situation = {}



	def readPattern(self,pattern_name):
		filename = "../artificialDataParams/"+pattern_name
		# self.username ="user-" + pattern_name.strip('-')[-1]
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

			if count == 3:
				self.graph = line.split()

			if count == 4:
				key = line.split('$')[0].strip()
				mean_sd = line.split('$')[1].strip().split(',')
				for entry in mean_sd:
					self.macro_mean_time[key]=int(entry.split(':')[0].strip())    #in seconds
					self.macro_standard_deviation[key]=  int(entry.split(':')[1].strip())    #in seconds

			if count == 5:
				self.weight_macros = [float(entry)  for entry in line.split()]

			if count == 6:
				key = line.split()[0]
				angle = -1
				value = line.split()[1].split(',')
				lat = (float)(value[0].strip())
				lon = (float)(value[1].strip())
				if (len(value)>2):
					angle = (float)(value[2].strip())
				self.location_situation[key] = [lat,lon, angle]


	def auxillary_pattern_data(self, filename):
		timestamps = []
		situation = []
		latitude =[]
		longitude = []
		angle = []
		multiplier = 1.00
		if 'test' in filename:
			multiplier = self.test_deviation_multiplier
		else:
			multiplier = self.train_deviation_multiplier

		events_per_macro = np.random.binomial(self.number_events, self.weight_macros, len(self.weight_macros))
		for index in range(len(events_per_macro)):
			key = self.graph[index]   # only applicable in a linear model
			requirement = events_per_macro[index]
			satisfied_timestamp_counter = 0
			while requirement > 0:
				temp_timestamps = np.random.normal(self.macro_mean_time[key], multiplier * self.macro_standard_deviation[key], events_per_macro[index])
				for entry in temp_timestamps:
					if int(entry) in range(0,86400) :#and int(entry) not in timestamps:
						timestamps.append(int(entry))
						satisfied_timestamp_counter += 1
						if satisfied_timestamp_counter == events_per_macro[index]:
							break

				requirement = events_per_macro[index] - satisfied_timestamp_counter
			situation += [self.situationList.index(rnd.choice(self.mapping_macros_situation[key.split('*')[0]]))]*satisfied_timestamp_counter
			if key in self.location_situation.keys():
				latitude += [self.location_situation[key][0] + rnd.random() for l in range(satisfied_timestamp_counter)]
				longitude += [self.location_situation[key][1] + rnd.random() for l in range(satisfied_timestamp_counter)]
				if self.location_situation[key][2] != -1:
					angle += [self.location_situation[key][2] + rnd.random() for l in range(satisfied_timestamp_counter)]
				else:
					angle += [-1]*satisfied_timestamp_counter
			else:
				latitude += np.linspace(self.location_situation[self.graph[index -1 ]][0],self.location_situation[self.graph[index + 1 ]][0], satisfied_timestamp_counter ).tolist()
				longitude += np.linspace(self.location_situation[self.graph[index -1 ]][0],self.location_situation[self.graph[index + 1 ]][0], satisfied_timestamp_counter ).tolist()
				angle += [-1]*(satisfied_timestamp_counter)
		sort_scenario = np.array([[x,w,z,a] for (y,x,w,z,a) in sorted(zip(timestamps,situation, latitude, longitude, angle))])
		situation = sort_scenario[:,0]
		situation = [int(entry) for entry in list(situation)]
		latitude = sort_scenario[:,1]
		longitude = sort_scenario[:,2]
		angle = sort_scenario[:,3]
		timestamps.sort()
		self.write_to_csv(timestamps, situation, latitude,longitude, angle,  filename+".csv")

	def generate_pattern_data(self, filePrefix):
		for iteration in range(self.testFiles):
			self.auxillary_pattern_data("../testData/generated/test-"+filePrefix+str(iteration))

		for iteration in range(self.trainFiles):
			self.auxillary_pattern_data("../trainData/generated/train-"+filePrefix+str(iteration))

		for iteration in range(self.randomFiles):
			timestamp, situations ,latitude,longitude, angle = self.randomize_situation()
			self.write_to_csv(timestamp, situations, latitude,longitude, angle, "../testData/generated/rn-"+filePrefix+str(iteration)+".csv")

	def write_to_csv(self, timestamp, situation, latitude, longitude, angle, filename):
		raw_data = {'Timestamp': timestamp,
	        'Situation': situation,
			'Latitude': latitude,
			'Longitude': longitude,
			'Angle': angle}
		df = pd.DataFrame(raw_data, columns = ['Timestamp', 'Situation', 'Latitude', 'Longitude', 'Angle'])
		df.to_csv(filename, index = False)

	def randomize_situation(self):
		situations = [rnd.randint(0, len(self.situationList)-1) for i in range(self.number_events)]
		timestamp = [ int(86400*rnd.random()) for i in range(self.number_events)]
		timestamp.sort()
		latitude = [rnd.randint(0, 180) for i in range(self.number_events)]
		longitude = [rnd.randint(0, 180) for i in range(self.number_events)]
		angle = [rnd.randint(-1, 180) for i in range(self.number_events)]
		return timestamp, situations, latitude,longitude, angle

	def plot_kernel(situation, timestamps, filename):
		fig, ax = plt.subplots()
		X_plot =  np.linspace(0, 86400, 86400)[:, np.newaxis]
		for entry in self.macros_number:
			specific_timestamp = []
			for index in range(len(timestamps)):
				if self.situationList[situation[index]] in self.mapping_macros_situation[entry]:
					specific_timestamp.append(timestamps[index])

			X = np.array(specific_timestamp)[:, np.newaxis]
			kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(X)
			log_dens = kde.score_samples(X_plot)
			ax.plot(X_plot[:, 0], np.exp(log_dens), '-',
			        label=entry)

		ax.legend(loc='upper left')
		plt.savefig("../generatedDataVisualisation/"+filename.split('/')[-1]+".png")
		plt.clf()
		plt.close()
