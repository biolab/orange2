# Description: Scree plot of PCA on iris data set
# Category:    projection
# Uses:        iris
# Referenced:  Orange.projection.pca
# Classes:     Orange.projection.pca.Pca, Orange.projection.pca.PcaClassifier

import Orange
iris = Orange.data.Table("iris.tab")

pca = Orange.projection.pca.Pca()(iris)
pca.scree_plot("pca-scree.png")
