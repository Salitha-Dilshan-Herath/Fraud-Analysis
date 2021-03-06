from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score, matthews_corrcoef
from sklearn.metrics import confusion_matrix


class RandomForest:

    def __init__(self, xTrain, yTrain, yTest, xTest):
        self.xTrain = xTrain
        self.yTrain = yTrain
        self.yTest = yTest
        self.xTest = xTest

    def model_train(self):
        # random forest model creation
        rfc = RandomForestClassifier()
        rfc.fit(self.xTrain, self.yTrain)

        # predictions
        print("------------Start predication in normal environment-------------")
        yPred = rfc.predict(self.xTest)

        print("The model used is Random Forest classifier")

        acc = accuracy_score(self.yTest, yPred)
        print("The accuracy is {}".format(acc))

        prec = precision_score(self.yTest, yPred)
        print("The precision is {}".format(prec))

        rec = recall_score(self.yTest, yPred)
        print("The recall is {}".format(rec))

        f1 = f1_score(self.yTest, yPred)
        print("The F1-Score is {}".format(f1))

        MCC = matthews_corrcoef(self.yTest, yPred)
        print("The Matthews correlation coefficient is {}".format(MCC))

        print("------------End predication in normal environment-------------")

        return rfc, acc, prec, rec, f1
