# Description: Reads a data base from MySQL
# Category:    file formats
# Classes:     orngMySQL.Connect
# Referenced:  orngMySQL.htm

import orange
import orngMySQL

t = orngMySQL.Connect('localhost','root','','test')

data = t.query("SELECT * FROM busclass WHERE line='10'")
for x in data:
    print x

print
for a in data.domain.attributes:
    print a
print 'Class:', data.domain.classVar
