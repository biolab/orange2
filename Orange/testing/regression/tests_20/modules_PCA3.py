# Description: setting number of retained components and variance covered, using generalized eigenvectors
# Category:    projection
# Uses:        iris
# Referenced:  orngPCA.htm
# Classes:     orngPCA.PCA

import orange, orngPCA

data = orange.ExampleTable("iris.tab")

attributes = ['sepal length', 'sepal width', 'petal length', 'petal width']
pca = orngPCA.PCA(data, standardize = True, attributes = attributes,
          maxNumberOfComponents = -1, varianceCovered = 1.0)
print "Retain all vectors and full variance:"
print pca

pca = orngPCA.PCA(data, standardize = True, maxNumberOfComponents = -1,
                  varianceCovered = 1.0, useGeneralizedVectors = 1)
print "As above, only with generalized vectors:"
print pca

