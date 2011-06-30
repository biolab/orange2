# Description: PCA with attribute and row selection
# Category:    projection
# Uses:        iris
# Referenced:  orngPCA.htm
# Classes:     orngPCA.PCA

import orange, orngPCA

data = orange.ExampleTable("iris.tab")

pca = orngPCA.PCA(data, standardize = True)
print "PCA on all data:"
print pca

attributes = ['sepal length', 'sepal width', 'petal length', 'petal width']
pca = orngPCA.PCA(data, standardize = True, attributes = attributes)
print "PCA on attributes sepal.length, sepal.width, petal.length, petal.width:"
print pca

rows = [1, 0] * (len(data) / 2)
pca = orngPCA.PCA(data, standardize = True, rows = rows)
print "PCA on every second row:"
print pca