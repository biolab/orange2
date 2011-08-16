# Description: Pca on iris data set
# Category:    projection
# Uses:        iris
# Referenced:  Orange.projection.pca
# Classes:     Orange.projection.pca.Pca, Orange.projection.pca.PcaClassifier

import Orange
table = Orange.data.Table("iris.tab")

pca = Orange.projection.pca.Pca(table)
transformed_data = pca(data)

print pca
