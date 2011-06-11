# Description: Export to Weka ARFF format
# Category:    data
# Uses:        iris
# Referenced:  Orange.data.io
# Classes:     Orange.data.io.toARFF

import Orange
table = Orange.data.Table('iris.tab')
Orange.data.io.toARFF('iris.testsave.arff', table)
table.save('iris.testsave.arff')
f = open('iris.testsave.arff')
for line in f:
    print line.strip()
f.close()