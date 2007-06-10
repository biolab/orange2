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

r = orngSQL.SQLReader('mysql://user:somepass@localhost/test')

t.write('iris', data, overwrite=True)

sel = t.query("SELECT petal_width, petal_length FROM iris WHERE sepal_length<5.0")
print "\n%d instances returned" % len(sel)
print "Output data domain:"
for a in sel.domain.variables:
    print a
