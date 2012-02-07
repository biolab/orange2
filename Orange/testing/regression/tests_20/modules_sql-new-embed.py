# Description: Reads data from a database
# Category:    file formats
# Classes:     orngSQL.SQLReader
# Referenced:  orngSQL.htm

import orange, orngSQLFile


orange.registerFileType("SQL", orngSQLFile.loadSQL, None, ".sql")
data = orange.ExampleTable('sql-new-embed.sql')
for x in data:
    print x

print
for a in data.domain.attributes:
    print a
