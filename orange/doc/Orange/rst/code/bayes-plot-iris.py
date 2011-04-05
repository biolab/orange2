import Orange
import pylab

iris = Orange.data.Table("iris.tab")
bayes = Orange.classification.bayes.NaiveLearner(iris)

sepal_length, probabilities = zip(*bayes.conditional_distributions[0].items())
probability_setosa, probability_versicolor, probability_virginica = zip(*probabilities)

pylab.xlabel("sepal length")
pylab.ylabel("probability")
pylab.plot(sepal_length, probability_setosa, label="setosa", linewidth=2)
pylab.plot(sepal_length, probability_versicolor, label="versicolor", linewidth=2)
pylab.plot(sepal_length, probability_virginica, label="virginica", linewidth=2)

pylab.legend(loc="best")
pylab.savefig("bayes-iris.png")