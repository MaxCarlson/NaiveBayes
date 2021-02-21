import csv
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import LogNorm
import copy


class NaiveBayes:

    def __init__(self, path):
        data = np.genfromtxt(path, dtype=float, delimiter=',')

        # Split into spam/not spam
        spam    = data[0:1813]
        notSpam = data[1813:]
        np.random.seed(1)
        np.random.shuffle(spam)
        np.random.shuffle(notSpam)
        
        
        # Used for calculating means and stds for model
        self.trainingData = [notSpam[0:int(len(notSpam)/2)], spam[0:int(len(spam)/2)]]
        trainingCount = len(self.trainingData[0]) + len(self.trainingData[1])

        # Calcualte Priors
        self.priors = [len(self.trainingData[0]) / trainingCount, 
                       len(self.trainingData[1]) / trainingCount]
        self.priors = [math.log(x) for x in self.priors]

        # Calculate means and stds
        self.stats()

        # Split data
        self.training = np.vstack([self.trainingData[0], self.trainingData[1]])
        self.test     = np.vstack([notSpam[int(len(notSpam)/2):], spam[int(len(spam)/2):]])

        np.random.shuffle(self.training)
        np.random.shuffle(self.test)

        self.classify(self.test[:,:-1], self.test[:,-1:])


    def stats(self):

        self.mean   = [np.mean(x[:,:-1], axis=0) for x in self.trainingData]
        self.std    = [np.std(x[:,:-1], axis=0)  for x in self.trainingData]
        self.std    = [np.where(x==0.0, 0.0001, x) for x in self.std]

    def classify(self, input, labels):
        norm = [1 / (math.sqrt(2 * math.pi) * self.std[i]) for i in range(2)]
        power = [(np.square(input - self.mean[i]) / (2 * np.square(self.std[i]))) 
                 for i in range(2)]

        likelihoods = [np.log(np.exp(-power[i]) + 10e-10) * norm[i] for i in range(2)]

        preds = [self.priors[i] + np.sum(likelihoods[i], axis=1) for i in range(2)]
        preds = [0 if x >= v else 1 for x, v in zip(preds[0], preds[1])]

        labels  = np.reshape(labels, (len(labels)))
        results = (labels == preds) * 1

        self.confusionMatrix(preds, labels)


    def confusionMatrix(self, preds, labels):
        confusion = np.zeros((2,2))
        for i in range(len(labels)):
            j, k = int(labels[i]), preds[i]
            confusion[j,k] += 1

        self.characteristics(confusion)
        my_cmap = copy.copy(matplotlib.cm.get_cmap('viridis')) # copy the default cmap
        my_cmap.set_bad((0,0,0))
        plt.matshow(confusion, norm=LogNorm(), interpolation='nearest', cmap=my_cmap)
        plt.colorbar()
        plt.show()

    def characteristics(self, confusion):
        tp = confusion[1,1]
        tn = confusion[0,0]
        fp = confusion[0,1]
        fn = confusion[1,0]

        accuracy = (tp + tn) / (tp + fp + tn + fn)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)

        print('Accuracy:', accuracy)
        print('Precision:', precision)
        print('Recall:', recall)

NaiveBayes('spambase.data')