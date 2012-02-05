# Description: Import from Weka ARFF format
# Category:    data
# Uses:        iris
# Referenced:  Orange.data.io
# Classes:     Orange.data.io.loadARFF

import Orange
iris = Orange.data.io.loadARFF('iris.arff')
print iris.attribute_load_status
print iris.domain
print iris.domain.attributes
print "\n".join(["\t".join([str(value) for value in row]) for row in iris])
print

iris = Orange.data.Table('iris.arff')
print iris.attribute_load_status
print iris.domain
print iris.domain.attributes
print "\n".join(["\t".join([str(value) for value in row]) for row in iris])