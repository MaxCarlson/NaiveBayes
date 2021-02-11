import csv
import numpy as np
import math



class NaiveBayes:

    def __init__(self, path):
        data = np.genfromtxt(path, dtype=float, delimiter=',')

        # Split into spam/not spam
        spam    = data[0:1813]
        notSpam = data[1813:]
        np.random.shuffle(spam)
        np.random.shuffle(notSpam)
        
        # Calcualte Priors
        self.priors = [len(notSpam) / len(data), 1813 / len(data)]
        self.priors = [math.log(x) for x in self.priors]
        
        # Used for calculating means and stds for model
        self.trainingData = [notSpam[0:int(len(notSpam)/2)], spam[0:int(len(spam)/2)]]

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

        # stddv & mean of columns
        #self.std     = np.std(self.training[:,:-1], axis=0)
        #self.std     = np.where(self.std==0, 0.0001, self.std)
        #self.mean    = np.mean(self.training[:,:-1], axis=0)

    def classify(self, input, labels):
        norm = [1.0 / ((math.sqrt(2 * math.pi) * self.std[i])) for i in range(2)]
        power = [-(np.square(input - self.mean[i]) / 2 * np.square(self.std[i])) 
                 for i in range(2)]

        likelihoods = [np.log(norm[i] * np.exp(power[i])) for i in range(2)]

        preds = [self.priors[i] + np.sum(likelihoods[i], axis=1) for i in range(2)]

        a=5

    #def prob(self, )






NaiveBayes('spambase.data')