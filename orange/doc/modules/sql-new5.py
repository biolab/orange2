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
w = orngSQL.SQLWriter('mysql://user:somepass@localhost/test')
# the following line only works with mysql because it uses the enum type.
w.create('iris', data, 
    renameDict = {'sepal length':'seplen',
        'sepal width':'sepwidth',
        'petal length':'petlen',
        'petal width':'petwidth'},
    typeDict = {'iris':"""enum('Iris-setosa', 'Iris-versicolor', 'Iris-virginica')"""})


r.execute("SELECT petwidth, petlen FROM iris WHERE seplen<5.0;")
data = r.data()
print "\n%d instances returned" % len(data)
print "Output data domain:"
for a in data.domain.variables:
    print a
