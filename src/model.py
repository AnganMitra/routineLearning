
import histogram as hst
import glob
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt




def richard_function(x, x0, a,k,b,c,q,g):
    val = a + (k - a)/(c+ q*np.exp(-b*(x- x0)))**(1/g)
    return val

class Model:
    def __init__(self, width,modelFile, dataFiles):

        self.width = width
        self.timerange = int(1440/width)
        self.transitionTable = []
        self.persistentTable = []
        self.macros_number = []	# macro
        self.mapping_macros_situation={}   # macro : list of situations
        self.situationList = []
        self.read(modelFile)
        self.hist = hst.Histogram(self.timerange, self.situationList)
        self.number_situations = len(self.mapping_macros_situation)
        self.transitionTable = ( np.zeros(shape=(self.timerange, self.number_situations, self.number_situations)) )
        self.persistentTable = np.zeros(shape=(self.timerange,self.number_situations))
        self.dataFiles = dataFiles
        self.hist.callingSupport(self.dataFiles)
        self.clustering_angle = {}
        self.cluster_outward_angle()
        self.persistentProbability()
        # self.transitionProbability()

    def read(self,pattern_name):
        filename = "../artificialDataParams/"+pattern_name
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



    def cluster_outward_angle(self ):
        situation = []
        angle = []
        for file in self.dataFiles:
            df = pd.read_csv(file,skipinitialspace=True, index_col = False)
            situation += [self.relabel(entry) for entry in df.Situation]
            angle += [entry for entry in df.Angle]

        dataPoints = [[]]*len(set(situation))
        for index in range(len(situation)):
            entry = angle[index]
            if entry != -1:
                dataPoints[situation[index]].append(entry)
        for index in range(len(dataPoints)):
            self.clustering_angle[index] = self.cluster(dataPoints[index])
        return

    def cluster(self, dataset):
        return np.mean(dataset)

    def transitionProbability(self):
        for file in self.dataFiles:
            temp_hist = hst.Histogram(self.timerange, self.situationList)
            temp_hist.callingSupport([file])
            aggregatedHist = temp_hist.aggregate(self.mapping_macros_situation, self.macros_number)
            situation = [0]+  hst.resolve(aggregatedHist)
            for timestamp in range(1,self.timerange):
                sList = situation [timestamp-1 : timestamp + 1]
                self.transitionTable[timestamp][sList[0]][sList[1]] += 1
        self.updateTransitionProbability()


    def updateTransitionProbability(self):
        for timestamp in range(self.timerange):
            for index in range(len(self.transitionTable[0])):
                for entry in range(len(self.transitionTable[timestamp][index])):
                    self.transitionTable[timestamp][index][entry] = richard_function(self.transitionTable[timestamp][index][entry],6,0,1,0.5,1,1,1  )
                total = float(np.sum(self.transitionTable[timestamp][index]))
                if total != 0.0:
                    self.transitionTable[timestamp][index] /= total



    def persistentProbability(self):
        aggregatedHist = self.hist.aggregate(self.mapping_macros_situation, self.macros_number)
        for timestamp in range(self.timerange):
            self.persistentTable[timestamp] = np.array([richard_function(situation,6,0,1,0.5,1,1,1) for situation in aggregatedHist[timestamp]])
            total = np.sum(self.persistentTable[timestamp])
            self.persistentTable[timestamp] = np.array([situation/max(1,total) for situation in self.persistentTable[timestamp]])

    def printParams(self, marker):
        print ("------------Writing Transition Probabilities------------")
        for timestamp in range(self.timerange):
            df = pd.DataFrame(self.transitionTable[timestamp])
            df.to_csv("../modelOutput/transition"+marker+"-"+str(timestamp)+".csv")

        print ("------------Writing Persistent Probabilities------------")
        time = [entry for entry in range(self.timerange)]
        array = np.array(self.persistentTable)
        raw_data = {'Timestamp': time}
        for index in range(len(self.persistentTable[0])):
            raw_data.update({self.macros_number[index]:self.persistentTable[:,index]} )
        df = pd.DataFrame(raw_data, columns =["Timestamp"] + self.macros_number)
        df.to_csv("../modelOutput/persistent"+marker+".csv", index = False)


    def test(self,testData):
        (testDataTimestamp, testDataSituation, testDataLatitude, testDataLongitude, testDataAngle) = (testData)
        probability = [0]*self.timerange
        index = 0
        for index in range(len(testDataTimestamp)):
            timestamp = testDataTimestamp[index]
            curr = testDataSituation[index]
            assert self.persistentTable[index][curr] <= 1.00
            location_factor =  self.proximity(testDataSituation[index], testDataLatitude[index], testDataLongitude[index], testDataAngle[index])
            assert timestamp < 1440/self.width
            probability[timestamp] = richard_function( self.persistentTable[timestamp][curr] + location_factor, 0.41, -1,1,5,1,1,1)

        timestamp = [i for i in range(len(probability))]
        drift = 0.0
        for index in range(1,len(probability)):
            if probability[index] == 0.0:
                drift = -0.05
            else:
                drift = 0.0
            probability[index] += probability[index-1] + drift

        range_min = min (probability)
        range_max = max(probability)
        new_range_min = -1.00
        new_range_max = 1.00
        if range_max <= 0.0:
            new_range_max = 0.0
        if range_min+range_max >= 0.00:
            new_range_min = 0.0
        for index in range(len(probability)):
            probability[index] = new_range_min + (probability[index] - range_min)*(new_range_max - new_range_min)/(range_max - range_min)
        return timestamp,probability


    def proximity(self,testSituation, testLatitude, testLongtitude, testAngle ):
        proximity = 0.0
        if self.clustering_angle[testSituation] != -1 and testAngle != -1:
            if abs(self.clustering_angle[testSituation] - testAngle) < 10:
                proximity = 1.00
            else:
                proximity = -1.00
        # print (testSituation, testAngle, proximity)
        return proximity

    def transformTestData (self, testData):
        timestamp = testData [0]
        situation = testData [1]
        latitude = testData [2]
        longitude = testData [3]
        angle = testData [4]
        for index in range(len(timestamp)):
            timestamp[index] = math.floor(timestamp[index]/(60*self.width))
            assert timestamp[index] <1440/self.width
            situation[index] = self.relabel(situation[index])
        return (timestamp,situation,latitude, longitude, angle)

    def relabel(self, situation):
        inv_macro_situation_map = {}
        for key in self.mapping_macros_situation.keys():
        	item = self.mapping_macros_situation[key]
        	try:
        		inv_macro_situation_map[item] = key
        	except:
        		for entry in item:
        			inv_macro_situation_map[entry] = key
        relabel = (self.macros_number.index(inv_macro_situation_map[self.situationList[situation]]))
        return relabel



def readTestFiles(filename):
    df = pd.read_csv(filename,skipinitialspace=True, index_col = False)
    timestamp = [entry for entry in df.Timestamp]
    situation = [entry for entry in df.Situation]
    latitude = [entry for entry in df.Latitude]
    longitude = [entry for entry in df.Longitude]
    angle = [entry for entry in df.Angle]
    return (timestamp, situation, latitude, longitude, angle)

def plot_confidence(timestamp, probability, filename):
    plt.plot(timestamp,probability)
    plt.xlabel("Time Segment")
    plt.ylabel("Confidence Factor")
    plt.savefig(filename)
    plt.clf()
