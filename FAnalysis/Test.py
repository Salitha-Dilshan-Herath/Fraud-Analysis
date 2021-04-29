from main import FAnalysis, LearningModel, Attacks, Defence

test = FAnalysis("creditcard.csv", LearningModel.RANDOMFOREST, "Class", .1)

test.initiate()
test.train_model()
test.attack(Attacks.ZOO)
test.defence(Defence.ADVERSARIALTRAINING)