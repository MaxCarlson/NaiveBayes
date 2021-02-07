import csv
import numpy as np
















def processData():
    #with open('spamebase.data', 'r') as csvfile:
    #    csv.reader(csvfile, delimiter=',', lineterminator='\n')

    data = np.genfromtxt('spambase.csv', dtype=float, delimiter=',')

    spam    = data[0:1813]
    notSpam = data[1813:]
    np.random.shuffle(spam)
    np.random.shuffle(notSpam)

    training = notSpam[0:int(len(notSpam)/2)]
    test     = notSpam[int(len(notSpam)/2):]

    training = np.vstack([training, spam[0:int(len(spam)/2)]])
    test     = np.vstack([test, spam[int(len(spam)/2):]])

    np.random.shuffle(training)
    np.random.shuffle(test)

    a = 5


processData()