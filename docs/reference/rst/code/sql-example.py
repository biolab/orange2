from Orange.data.sql import *

# load dataset and save it into sqlite database
data = Orange.data.Table("iris")
w = SQLWriter('sqlite://iris.db/')
w.create('iris', data)

# create sql reader
from Orange.data.sql import *
r = SQLReader()
r.connect('sqlite://iris.db/')

# read iris dataset from database and convert it to Orange Table
r.execute("SELECT *  FROM iris;")
d = r.data()
print "\n%d instances returned" % len(d)
print "Output data domain:"
for a in d.domain.variables:
    print a
print "First instance :", d[0],

r.execute("SELECT `petal width`, `petal length` FROM iris WHERE `sepal length` < 5.0")
sel = r.data()
print "\n%d instances returned" % len(sel)

r.disconnect()