# Description: Shows how to use preprocessors for discretization
# Category:    preprocessing, discretization, categorization
# Classes:     Preprocessor, Preprocessor_discretize
# Uses:        iris
# Referenced:  preprocessing.htm

import orange
iris = orange.ExampleTable("iris")

pp = orange.Preprocessor_discretize()
pp.method = orange.EquiDistDiscretization(numberOfIntervals = 5)
data2 = pp(iris)

for ex in data2[:10]:
    print ex

pp.attributes = [iris.domain["petal length"], iris.domain["sepal length"]]
data2 = pp(iris)

for ex in data2[:10]:
    print ex
