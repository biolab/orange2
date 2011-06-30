# Description: Shows how to use graphs
# Category:    general
# Classes:     Graph
# Uses:        
# Referenced:  graph.htm

import Orange.network

graph = Orange.network.GraphAsMatrix(4, 0, objectsOnEdges = 1)
graph.objects = ["Age", "Gender", "Height", "Weight"]

graph["Age", "Gender"] = "a string"
# commented out: causes differences in regression tests between machines
#graph["Age", "Height"] = orange
graph["Gender", "Height"] = [1, 2, 3]
print graph.get_edges()

print graph[1, 2]
print graph["Gender", "Height"]
print graph["Age", "Gender"]
print graph["Age", "Height"]

graph.objects = {}
graph.objects["Age"] = 0
graph.objects["Gender"] = 1
graph.objects["Height"] = 2

print graph["Age", "Gender"]
try:
    print graph["Gender", "Height"]
except:
    print 'graph["Gender", "Height"] failed'

print graph.get_neighbours("Age")
graph.returnIndices = 1
print graph.get_neighbours("Age")


graph = Orange.network.GraphAsMatrix(5, 0, 3)
print graph[4, 1]
graph[4, 1, 1]=12
print graph[4, 1, 1]
print graph[4, 1]

print graph.edge_exists(4, 1)
print graph.edge_exists(4, 2)
print graph.edge_exists(4, 1, 1)
print graph.edge_exists(4, 1, 2)

e = graph[4, 1]
e[1]
e[2]
e[1] = None
e[2] = 3
print graph.edge_exists(4, 1, 1)
print graph.edge_exists(4, 1, 2)
print e
graph[4, 1]=None
print e
