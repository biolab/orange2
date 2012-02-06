# Description: Screeplot and biplot for PCA
# Category:    projection
# Uses:        iris
# Referenced:  orngPCA.htm
# Classes:     orngPCA.PCA

import orange, orngPCA

data = orange.ExampleTable("iris.tab")

attributes = ['sepal length', 'sepal width', 'petal length', 'petal width']
pca = orngPCA.PCA(data, standardize = True, attributes = attributes)

pca.plot(filename = None)

pca(data)
pca.biplot()