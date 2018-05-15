import datetime, math
import numpy as np
import matplotlib.pyplot as plt
import sys
import glob
import math as m
import networkx as nx


def getData(filename):
	f=  open(filename)
	f.readline()
	record=[]
	for line in f:
		timestamp= int(line.strip().split(',')[0])
		situation= int(line.strip().split(',')[1])
		latitude = float(line.strip().split(',')[2])
		longtitude = float(line.strip().split(',')[3])
		angle = float(line.strip().split(',')[4])
		record.append(Situation(timestamp, situation, latitude, longtitude))
	f.close()
	return record

def parseTime(timestamp):
	t=datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
	year = int(t.split()[0].split('-')[0])
	month = int(t.split()[0].split('-')[1])
	day = int(t.split()[0].split('-')[2])
	hour= int(t.split()[1].split(':')[0])
	min= int(t.split()[1].split(':')[1])
	sec= int(t.split()[1].split(':')[2])
	return (year, month,day, hour, min, sec)


class Situation:
	def __init__(self, timestamp,situation,latitude,longitude):
		self.timestamp = timestamp
		self.situation = situation
		year, month,day, hour, min, sec =  parseTime(timestamp)
		self.year = year
		self.month = month
		self.day = day
		self.hour = hour
		self.min = min
		self.sec = sec
		self.latitude = latitude
		self.longitude = longitude


class Histogram:
	def __init__(self,  timerange, situationList ):
		self.timerange = timerange
		self.situationList = situationList
		self.number_situations = len(situationList)
		self.hist = np.zeros(shape=(timerange,self.number_situations))
		self.location = np.zeros(shape=(timerange, 2)) # for latitude and longtitude in order


	def fillHistogram(self, record):
		width = 1440/self.timerange
		for timestamp in range(self.timerange):
			self.hist[timestamp]+=(self.exert_influence_situation(record, timestamp, width ))
		for entry in record:
			flag = 0
			timestamp = int((entry.hour*60+entry.min)/width)
			if self.location[timestamp][0] == 0.0 and self.location[timestamp][1] == 0.0:
				flag = 1
			if flag == 1:
				self.location[timestamp][0] = entry.latitude
				self.location[timestamp][1] = entry.longitude
			else:
				self.location[timestamp][0] = (entry.latitude + self.location[timestamp][0])/2
				self.location[timestamp][1] = (entry.longitude + self.location[timestamp][1])/2


	def __getitem__(self, key):
		return self.hist[key]

	def aggregate(self, mapping_macros_situation, macros_number):
		aggregatedHist = np.zeros(shape=(self.timerange, len(mapping_macros_situation)))
		inv_macro_situation_map = {}
		for key in mapping_macros_situation.keys():
			item = mapping_macros_situation[key]
			try:
				inv_macro_situation_map[item] = key
			except:
				for entry in item:
					inv_macro_situation_map[entry] = key

		for timestamp in range(self.timerange):
			for situation in  range(len(self.hist[timestamp])):
				aggregatedHist[timestamp][macros_number.index(inv_macro_situation_map[self.situationList[situation]])] += self.hist[timestamp][situation]

		return aggregatedHist



	def plot(self, timestamp, index, label):
		plt.scatter(timestamp,self.hist[:,index], label=label )
		plt.show()


	def printParams(self):
		for entry in self.hist:
			print (entry)

	def callingSupport(self,filenames):
		for filename in filenames:
			#print ("Reading "+ filename)
			record = getData(filename)
			self.fillHistogram(record)

	def calculate_influence(self,center, rv, deviation ):
		return m.exp(-1*((center - rv)/(max(1, deviation)))**2 )


	def exert_influence_situation(self, samples, timestamp,width):
		predicates = [0.0]*(self.number_situations )
		for entry in samples:
			predicates[entry.situation] += self.calculate_influence((entry.hour*60+entry.min)/width, timestamp, 30 )
		predicates = [1 if entry==max(predicates) else 0 for entry in predicates]  #argmax is implemented
		return predicates



def resolve(hist):
	resolvedList = []
	for index in range(len(hist)):
		resolvedList.append(np.argmax(hist[index]))
	return resolvedList
