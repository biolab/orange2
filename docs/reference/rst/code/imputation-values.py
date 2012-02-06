# Description: Shows how to impute missing values in data Table.
# Category:    imputation
# Uses:        bridges
# Referenced:  Orange.feature.imputation.html
# Classes:     Orange.feature.imputation.ImputerConstructor_minimal

import Orange
bridges = Orange.data.Table("bridges")

imputer = Orange.feature.imputation.ImputerConstructor_minimal()
imputer = imputer(bridges)

print "Example with missing values"
print bridges[10]
print "Imputed values:"
print imputer(bridges[10])

imputed_bridges = imputer(bridges)
print imputed_bridges[10]
