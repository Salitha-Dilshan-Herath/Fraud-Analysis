from enum import Enum
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split

from models.logistic_regression import LRegression
from defence.adversarial_training import AdversarialTraining
from attack.zoo_attack import ZooAdvAttack
from models.random_forest import RandomForest


class LearningModel(Enum):
    RANDOMFOREST = 1
    LOGISTICREGRESSION = 2


class Attacks(Enum):
    ZOO = 1
    FGM = 2


class Defence(Enum):
    ADVERSARIALTRAINING = 1


class FAnalysis:
    df = pd.DataFrame()
    trainModel = ""
    X_train_var, X_test_var, yTrain, yTest, x_train_adv, x_test_adv = [], [], [], [], [], []

    def __init__(self, dataset_path: str = "", learning_model: LearningModel = LearningModel.LOGISTICREGRESSION,
                 label_parameter: str = "", use_percentage_data: float = 100):
        self.dataset_path = dataset_path
        self.learning_model = learning_model
        self.label_parameter = label_parameter
        self.percentage_data = use_percentage_data

    def initiate(self):
        self.df = pd.read_csv(self.dataset_path)
        self.df = self.df.sample(frac=self.percentage_data)
        self.data_frame_describe()
        self.implement_feature_selection()

    def data_frame_describe(self):
        fraud = self.df[self.df[self.label_parameter] == 1]
        valid = self.df[self.df[self.label_parameter] == 0]
        outlier_fraction = len(fraud) / float(len(valid))
        fraud_case = 'Fraud Cases: {}'.format(len(self.df[self.df[self.label_parameter] == 1]))
        valid_case = 'Valid Transactions: {}'.format(len(self.df[self.df[self.label_parameter] == 0]))

        print("----------Dataset Description-------------")
        print(fraud_case)
        print(valid_case)
        print("------------------------------------------")

    def implement_feature_selection(self):
        X = self.df.drop([self.label_parameter], axis=1)
        Y = self.df[self.label_parameter]

        # getting just the values for the sake of processing
        # (its a numpy array with no columns)
        xData = X.values
        yData = Y.values

        sm = SMOTE(k_neighbors=2)

        X_train_over, y_train_over = sm.fit_resample(xData, yData)

        # split the data into training and testing sets
        xTrain, xTest, yTrain, yTest = train_test_split(X_train_over, y_train_over, test_size=0.2, random_state=42)

        var = VarianceThreshold(threshold=.5)
        var.fit(xTrain, yTrain)
        X_train_var = var.transform(xTrain)
        X_test_var = var.transform(xTest)

        self.X_train_var = X_train_var
        self.X_test_var = X_test_var
        self.yTest = yTest
        self.yTrain = yTrain

    def train_model(self):

        if self.learning_model == LearningModel.RANDOMFOREST:
            rf_obj = RandomForest(self.X_train_var, self.yTrain, self.yTest, self.X_test_var)
            rfc, acc, prec, rec, f1 = rf_obj.model_train()
            self.trainModel = rfc
        elif self.learning_model == LearningModel.LOGISTICREGRESSION:
            model_obj = LRegression(self.X_train_var, self.yTrain, self.yTest, self.X_test_var)
            rfc, acc, prec, rec, f1 = model_obj.model_train()
            self.trainModel = rfc

    def attack(self, attack_type=Attacks.ZOO):

        if attack_type == Attacks.ZOO:
            at_obj = ZooAdvAttack(self.X_train_var, self.yTrain, self.yTest, self.X_test_var, self.trainModel,
                                  self.percentage_data)
            score_train, score_test, self.x_train_adv, self.x_test_adv = at_obj.generate_attack()

            return score_train, score_test

    def defence(self, defence_type = Defence.ADVERSARIALTRAINING):

        if defence_type == Defence.ADVERSARIALTRAINING:
            df_obj = AdversarialTraining(self.X_train_var, self.x_train_adv, self.x_test_adv, self.yTrain, self.yTest,
                                         self.trainModel)
            acc = df_obj.defence()

            print("Defence accuracy " + str(acc))

            return acc
