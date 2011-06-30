import Orange
import pylab

iris = Orange.data.Table("iris.tab")
nb = Orange.classification.bayes.NaiveLearner(iris)

sepal_length, probabilities = zip(*nb.conditional_distributions[0].items())
p_setosa, p_versicolor, p_virginica = zip(*probabilities)

pylab.xlabel("sepal length")
pylab.ylabel("probability")
pylab.plot(sepal_length, p_setosa, label="setosa", linewidth=2)
pylab.plot(sepal_length, p_versicolor, label="versicolor", linewidth=2)
pylab.plot(sepal_length, p_virginica, label="virginica", linewidth=2)

pylab.legend(loc="best")
pylab.savefig("bayes-iris.png")