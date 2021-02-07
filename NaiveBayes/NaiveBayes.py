import csv
import numpy as np




class NaiveBayes:

    def __init__(self, path):
        data = np.genfromtxt(path, dtype=float, delimiter=',')

        spam    = data[0:1813]
        notSpam = data[1813:]
        np.random.shuffle(spam)
        np.random.shuffle(notSpam)
        
        self.spamPrior = 1813 / len(data)
        self.notSpamPrior = len(notSpam) / len(data)

        self.training = notSpam[0:int(len(notSpam)/2)]
        self.test     = notSpam[int(len(notSpam)/2):]

        self.training = np.vstack([self.training, spam[0:int(len(spam)/2)]])
        self.test     = np.vstack([self.test, spam[int(len(spam)/2):]])

        np.random.shuffle(self.training)
        np.random.shuffle(self.test)

        self.stats()

    def stats(self):

        # stddv & mean of columns
        self.std     = np.std(self.training[:,:-1], axis=0)
        self.std     = np.where(self.std==0, 0.0001, self.std)
        self.mean    = np.mean(self.training[:,:-1], axis=0)

    def prob(self, )






NaiveBayes('spambase.data')