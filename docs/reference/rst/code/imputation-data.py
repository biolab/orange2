# Description: Shows how to impute missing values in data Table.
# Category:    imputation
# Uses:        bridges
# Referenced:  Orange.feature.imputation.html
# Classes:     Orange.feature.imputation.AverageConstructor

import Orange
bridges = Orange.data.Table("bridges")
imputed_bridges = Orange.data.imputation.ImputeTable(bridges,
    method=Orange.feature.imputation.AverageConstructor())

print "Original data set:"
for e in bridges[:3]:
    print e

print "Imputed data set:"
for e in imputed_bridges[:3]:
    print e
