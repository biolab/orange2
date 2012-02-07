# Description: projecting data with trained PCA
# Category:    projection
# Uses:        iris
# Referenced:  orngPCA.htm
# Classes:     orngPCA.PCA

import orange, orngPCA

data = orange.ExampleTable("iris.tab")

attributes = ['sepal length', 'sepal width', 'petal length', 'petal width']
pca = orngPCA.PCA(data, attributes = attributes, standardize = True)

projected = pca(data)
print "Projection on first two components:"
for d in projected[:5]:
    print d
