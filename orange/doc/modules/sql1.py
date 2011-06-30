# Description: Reads a data base from MySQL
# Category:    file formats
# Classes:     orngMySQL.Connect
# Referenced:  orngMySQL.htm

import orange, orngMySQL

t = orngMySQL.Connect('localhost','root','','test')
data = t.query ("SELECT * FROM bus WHERE line='10'")
for x in data:
    print x

print
for a in data.domain.attributes:
    print a
