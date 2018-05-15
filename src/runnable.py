import glob
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import generate_pattern as gnp
from model import *

if __name__ == "__main__":   # calculate what can be the best parameters or cross validate

    print ("-----------------GENERATING PATTERN DATA----------------")

    patternFiles = ["pattern1", "pattern2", "pattern3" ]
    for file in patternFiles:
        pattern = gnp.Pattern()
        pattern.readPattern(file)
        pattern.generate_pattern_data(file+"-")
        print (file ," data generated")

    print ("----------------Creating MODEL -------------------------")
    width =  1
    modelParamFile = "model1"
    for pattern_index in range(len(patternFiles)):
        print("------------- Working on Pattern "+patternFiles[pattern_index]+" --------------")

        modelFiles = [file for file in glob.glob("../trainData/generated/train-"+patternFiles[pattern_index]+"-*.csv")]
        m = Model(width, "../globalParams/"+modelParamFile, modelFiles)
        m.printParams(patternFiles[pattern_index])
        print ("-----------------Running on TEST  files-----------------")

        testFiles = [file for file in glob.glob("../testData/generated/*.csv")] # files which have the data
        for file in testFiles:
            print ("Testing on ", file)# testData = (situation,timestamp, latitude, longitude)
            testData = readTestFiles(file)
            resolvedTestData = m.transformTestData(testData)
            (timestamp,probability) = m.test(resolvedTestData)
            raw_data ={'Timestamp': timestamp,  'ConfidenceFactor':probability}
            df = pd.DataFrame(raw_data, columns = ['Timestamp','ConfidenceFactor'])
            df.to_csv("../comparisons/csv/"+ patternFiles[pattern_index]+"-"+file.split("/")[-1], index = False)
            plot_confidence(timestamp, probability, "../comparisons/plot/"+ patternFiles[pattern_index]+"-"+file.split("/")[-1].split('.')[0]+".png")
