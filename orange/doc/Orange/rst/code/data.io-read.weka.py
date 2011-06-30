# Description: Import from Weka ARFF format
# Category:    data
# Uses:        iris
# Referenced:  Orange.data.io
# Classes:     Orange.data.io.loadARFF

import Orange
table = Orange.data.io.loadARFF('iris.arff')
print table.attribute_load_status
print table.domain
print table.domain.attributes
print "\n".join(["\t".join([str(value) for value in row]) for row in table])
print

table = Orange.data.Table('iris.arff')
print table.attribute_load_status
print table.domain
print table.domain.attributes
print "\n".join(["\t".join([str(value) for value in row]) for row in table])