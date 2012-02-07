# Description: Writes a data set to and reads from MySQL
# Category:    file formats
# Classes:     ExampleTable, orngMySQL.Connect
# Uses:        iris.tab
# Referenced:  orngMySQL.htm

import orange, orngMySQL, orngTree

data = orange.ExampleTable("iris")
print "Input data domain:"
for a in data.domain.variables:
    print a

t = orngMySQL.Connect('localhost','root','','test')
t.write('iris', data, overwrite=True)

sel = t.query("SELECT petal_width, petal_length FROM iris WHERE sepal_length<5.0")
print "\n%d instances returned" % len(sel)
print "Output data domain:"
for a in sel.domain.variables:
    print a
