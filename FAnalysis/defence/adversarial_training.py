import numpy as np
from sklearn.metrics import accuracy_score


class AdversarialTraining:

    def __init__(self, xTrain, xTrainAdv, xTestAdv, yTrain,yTest, model):
        self.xTrain = xTrain
        self.xTrainAdv = xTrainAdv
        self.yTrain = yTrain
        self.xTestAdv = xTestAdv
        self.yTest = yTest
        self.model = model
        self.dataAmount = 100

    def defence(self):

        new_x_train = np.append(self.xTrain[:self.dataAmount], self.xTrainAdv, axis=0)
        new_y_train = np.append(self.yTrain[:self.dataAmount], self.yTrain[:self.dataAmount], axis=0)
        self.model.fit(new_x_train, new_y_train)
        new_yPred = self.model.predict(self.xTestAdv)
        acc = accuracy_score(self.yTest[:self.dataAmount], new_yPred)

        return acc