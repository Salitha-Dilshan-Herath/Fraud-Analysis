from art.estimators.classification import SklearnClassifier
from art.attacks.evasion import ZooAttack


class ZooAdvAttack:

    def __init__(self, xTrain, yTrain, yTest, xTest, model,amount):
        self.xTrain = xTrain
        self.yTrain = yTrain
        self.yTest = yTest
        self.xTest = xTest
        self.model = model
        self.dataAmount = 100

    def generate_attack(self):
        art_classifier = SklearnClassifier(model=self.model)

        zoo = ZooAttack(classifier=art_classifier, confidence=0.0, targeted=False, learning_rate=1e-1, max_iter=30,
                        binary_search_steps=20, initial_const=1e-3, abort_early=True, use_resize=False,
                        use_importance=False, nb_parallel=10, batch_size=1, variable_h=0.25)

        print("start attack")
        x_train_adv = zoo.generate(self.xTrain[:self.dataAmount])

        score_train = self.model.score(x_train_adv, self.yTrain[:self.dataAmount])
        print("Adversarial Training Score: %.4f" % score_train)

        x_test_adv = zoo.generate(self.xTest[:self.dataAmount])

        score_test = self.model.score(x_test_adv, self.yTest[:self.dataAmount])
        print("Adversarial Test Score: %.4f" % score_test)

        return score_train, score_test , x_train_adv, x_test_adv
