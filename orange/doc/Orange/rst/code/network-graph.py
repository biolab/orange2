# Description: Shows how to use graphs
# Category:    general
# Classes:     Graph
# Uses:        
# Referenced:  graph.htm

import Orange.network

graph = Orange.network.GraphAsMatrix(4, 0)
graph.objects = ["Age", "Gender", "Height", "Weight"]

graph["Age", "Gender"] = 0.1
graph["Age", "Height"] = 1.2
graph["Gender", "Height"] = 0.3
print graph.getEdges()

print graph[1, 2]
print graph["Gender", "Height"]

graph.objects = {}
graph.objects["Age"] = 0
graph.objects["Gender"] = 1
graph.objects["Height"] = 2

print graph["Age", "Gender"]
try:
    print graph["Gender", "Height"]
except:
    print 'graph["Gender", "Height"] failed'

print graph.getNeighbours("Age")
graph.returnIndices = 1
print graph.getNeighbours("Age")


graph = Orange.network.GraphAsMatrix(5, 0, 3)
print graph[4, 1]
graph[4, 1, 1]=12
print graph[4, 1, 1]
print graph[4, 1]

print graph.edgeExists(4, 1)
print graph.edgeExists(4, 2)
print graph.edgeExists(4, 1, 1)
print graph.edgeExists(4, 1, 2)

e = graph[4, 1]
e[1]
e[2]
e[1] = None
e[2] = 3
print graph.edgeExists(4, 1, 1)
print graph.edgeExists(4, 1, 2)
print e
graph[4, 1]=None
print e
