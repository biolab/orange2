# Description: Pca on iris data set
# Category:    projection
# Uses:        iris
# Referenced:  Orange.projection.pca
# Classes:     Orange.projection.pca.Pca, Orange.projection.pca.PcaClassifier

import Orange
iris = Orange.data.Table("iris.tab")

pca = Orange.projection.pca.Pca(iris)
transformed_data = pca(iris)

print pca
