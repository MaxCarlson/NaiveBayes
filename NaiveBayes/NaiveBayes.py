import csv
import numpy as np
import math



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

        # TODO: Calculate Priors from training ONLY!!!
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

        #t = np.exp(power[0])

        likelihoods = [np.log(np.exp(-power[i]) + 10e-10) * norm[i] for i in range(2)]

        preds = [self.priors[i] + np.sum(likelihoods[i], axis=1) for i in range(2)]
        preds = [0 if x >= v else 1 for x, v in zip(preds[0], preds[1])]

        labels  = np.reshape(labels, (len(labels)))
        results = (labels == preds) * 1
        accuracy = np.sum(results) / len(results)

        a=5

    #def prob(self, )






NaiveBayes('spambase.data')