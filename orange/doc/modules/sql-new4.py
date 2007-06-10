# Description: Writes a data set to and reads from an SQL database
# Category:    file formats
# Classes:     ExampleTable, orngSQL.SQLReader, orngSQL.SQLWriter
# Uses:        iris.tab
# Referenced:  orngSQL.htm

import orange, orngSQL, orngTree

data = orange.ExampleTable("iris")
print "Input data domain:"
for a in data.domain.variables:
    print a
w = orngSQL.SQLWriter('mysql://user:somepass@localhost/test')
w.create('iris', data)

r = orngSQL.SQLReader('mysql://user:somepass@puhek/test')
r.execute('SELECT "petal width", "petal length" FROM iris WHERE "sepal length"<5.0')
data = r.data()
print "\n%d instances returned" % len(data)
print "Output data domain:"
for a in data.domain.variables:
    print a
