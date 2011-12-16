import Orange
table = Orange.data.Table("iris")

knnLearner = Orange.classification.knn.kNNLearner()
knnLearner.k = 10
knnClassifier = knnLearner(table)
