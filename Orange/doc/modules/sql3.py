# Description: Reads a data from MySQL data base and constructs a classification tree
# Category:    file formats
# Classes:     orngMySQL.Connect
# Referenced:  orngMySQL.htm

import orange, orngMySQL, orngTree

t = orngMySQL.Connect('localhost','root','','test')
data = t.query("SELECT * FROM busclass")
tree = orngTree.TreeLearner(data)
orngTree.printTxt(tree, nodeStr="%V (%1.0N)", leafStr="%V (%1.0N)")
