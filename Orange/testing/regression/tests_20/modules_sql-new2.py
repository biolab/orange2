# Description: Reads data from a database
# Category:    file formats
# Classes:     orngSQL.SQLReader
# Referenced:  orngSQL.htm

import orange, orngSQL

r = orngSQL.SQLReader('mysql://user:somepass@localhost/test')
r.query = "SELECT * FROM bus WHERE line='10';"
r.discreteNames = ['weather', 'arrival', 'daytime']
r.metaNames = ['id']
r.className = 'arrival'
r.update()
print r.metaNames
print r.discreteNames
data = r.data()
for x in data:
    print x

print
for a in data.domain.attributes:
    print a

print 'Class:', data.domain.classVar
