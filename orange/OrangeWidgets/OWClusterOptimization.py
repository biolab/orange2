from OWBaseWidget import *
import os
import orange, orngTest #, orngStat
from copy import copy
from math import sqrt
import OWGUI, OWDlgs
import OWVisAttrSelection
import Numeric, RandomArray

VALUE = 0
CLOSURE = 1
VERTICES = 2
ATTR_LIST = 3
CLASS = 4
TRY_INDEX = 5
STR_LIST = 6

contMeasures = [("None", None), ("ReliefF", orange.MeasureAttribute_relief()), ("Fisher discriminant", OWVisAttrSelection.MeasureFisherDiscriminant())]
discMeasures = [("None", None), ("ReliefF", orange.MeasureAttribute_relief()), ("Gain ratio", orange.MeasureAttribute_gainRatio()), ("Gini index", orange.MeasureAttribute_gini())]

VALUE = 0
CLUSTER = 1
DISTANCE = 2

# is the point (xTest, yTest) on the left or right side of the line through (x1,y1), (x2,y2)
def getPosition(x1, y1, x2, y2, xTest, yTest):
    if x1 == x2:
        if xTest > x1: return 1
        else: return -1
    k = (y2-y1)/float(x2-x1)
    c = y1 - k*x1
    if k*xTest - yTest > -c : return 1
    else: return -1

# compute distances between all connected vertices and store it in the DISTANCE field
def computeDistances(graph):
    for (i,j) in graph.getEdges():
        xdiff = graph.objects[i][0] - graph.objects[j][0]
        ydiff = graph.objects[i][1] - graph.objects[j][1]
        graph[i,j][DISTANCE] = sqrt(xdiff*xdiff + ydiff*ydiff)


# remove edges to different class values
def removeEdgesToDifferentClasses(graph, newVal):
    if newVal == None:
        for (i,j) in graph.getEdges():
            if graph.objects[i].getclass() != graph.objects[j].getclass(): graph[i,j] = None
    else:
        for (i,j) in graph.getEdges():
            if graph.objects[i].getclass() != graph.objects[j].getclass(): graph[i,j][VALUE] = newVal


def getPointsWithDifferentClassValue(graph, closureDict):
    differentClassPoinsDict = {}
    for key in closureDict:
        points = []
        vertices = [i for (i,j) in closureDict[key]] + [j for (i,j) in closureDict[key]]
        vertices.sort()
        for i in range(len(vertices)-1)[::-1]:
            if vertices[i] == vertices[i+1]: vertices.remove(vertices[i])

        for v in vertices:
            for n in graph.getNeighbours(v):
                if graph.objects[v].getclass() == graph.objects[n].getclass() or n in points: continue
                points.append(n)
        differentClassPoinsDict[key] = points
    return differentClassPoinsDict


# remove edges, that do not connect to triangles
def removeSingleLines(graph, edgesDict, verticesDict, minimumValidIndex, newVal):
    for (i,j) in graph.getEdges():
        if graph[i,j][VALUE] < minimumValidIndex: continue
        merged = graph.getNeighbours(i) + graph.getNeighbours(j)
        merged.sort()
        k=0; found = 0;
        for k in range(len(merged)-1):
            if merged[k] == merged[k+1] and graph[i, merged[k]][VALUE] >= minimumValidIndex and graph[j, merged[k]][VALUE] >= minimumValidIndex:
                found = 1
                break
        # if the edge does not form a triangle, check if both points are accessible by a different route
        if not found:
            if newVal == None: graph[i,j] = None            # delete the edge
            else:
                graph[i,j][VALUE] = newVal   # set the edge value
                graph[i,j][CLUSTER] = -1
            # remove the edge from the edgesDict
            if edgesDict and verticesDict:
                index = max(verticesDict[j], verticesDict[i])
                if index == -1: continue
                if (i,j) in edgesDict[index]: edgesDict[index].remove((i,j))
                elif (j,i) in edgesDict[index]: edgesDict[index].remove((j,i))
                verticesDict[j] = verticesDict[i] = -1
            """
            verticesToBrowse = graph.getNeighbours(j)
            verticesToBrowse.remove(i)
            for v in verticesToBrowse[::-1]:
                if graph[j,v][VALUE] < minimumValidIndex: verticesToBrowse.remove(v)
            searchedVertices = [j]
            while not found and verticesToBrowse != []:
                vertex = verticesToBrowse.pop()
                searchedVertices.append(vertex)
                neighbors = graph.getNeighbours(vertex)
                for v in neighbors[::-1]:
                    if graph[vertex,v][VALUE] < minimumValidIndex or v in searchedVertices + verticesToBrowse: neighbors.remove(v)
                if i in neighbors: found = 1
                else: verticesToBrowse += neighbors
            if not found:
                if newVal == None: graph[i,j] = None            # delete the edge
                else:              graph[i,j][VALUE] = newVal   # set the edge value
            """

# remove single lines and triangles that are now treated as their own clusters because they are probably insignificant
def removeSingleTrianglesAndLines(graph, edgesDict, verticesDict, newVal):
    for key in edgesDict.keys():
        if len(edgesDict[key]) < 4:
            for (i,j) in edgesDict[key]:
                graph[i,j][VALUE] = newVal
                graph[i,j][CLUSTER] = -1
                verticesDict[i] = -1; verticesDict[j] = -1;
            edgesDict.pop(key)

# make clusters smaller by removing points on the hull that lie closer to points in opposite class
def removeDistantPointsFromClusters(graph, edgesDict, verticesDict, closureDict, newVal):
    for key in closureDict.keys():
        edges = closureDict[key]
        vertices = [i for (i,j) in edges] + [j for (i,j) in edges]
        vertices.sort()
        for i in range(len(vertices)-1)[::-1]:
            if vertices[i] == vertices[i+1]: vertices.remove(vertices[i])
        for vertex in vertices:
            correctClass = int(graph.objects[vertex].getclass())
            correct = []; incorrect = []
            for n in graph.getNeighbours(vertex):
                if int(graph.objects[n].getclass()) == correctClass and graph[vertex,n][CLUSTER] == -1: continue
                if int(graph.objects[n].getclass()) == correctClass: correct.append(graph[vertex,n][DISTANCE])
                else: incorrect.append(graph[vertex,n][DISTANCE])

            if correct == [] or (len(incorrect) > 0 and min(incorrect) < min(correct)):
                for n in graph.getNeighbours(vertex):
                    if int(graph.objects[n].getclass()) != correctClass or graph[vertex,n][CLUSTER] == -1: continue
                    if newVal == None:  graph[vertex,n] = None
                    else:               graph[vertex,n][VALUE] = newVal
                    if (n, vertex) in edgesDict[graph[vertex,n][CLUSTER]]: edgesDict[graph[vertex,n][CLUSTER]].remove((n,vertex))
                    elif (vertex, n) in edgesDict[graph[vertex,n][CLUSTER]]: edgesDict[graph[vertex,n][CLUSTER]].remove((vertex, n))
                    verticesDict[n] = verticesDict[vertex] = -1
                    graph[vertex,n][CLUSTER] = -1

    """
    for vertex in verticesDict.keys():
        classes = [[] for j in range(len(graph.objects.domain.classVar.values))]
        correctClass = int(graph.objects[vertex].getclass())
        for n in graph.getNeighbours(vertex):
            if int(graph.objects[n].getclass()) == correctClass and graph[vertex,n][CLUSTER] == -1: continue
            classes[int(graph.objects[n].getclass())].append(graph[vertex,n][DISTANCE])

        if classes[correctClass] == []: minCorrectClassDistance = 1e20
        else:                           minCorrectClassDistance = sum(classes[correctClass])/float(len(classes[correctClass]))
        #else:                           minCorrectClassDistance = min(classes[correctClass])
        count = 0; dist = 0.0;
        for i in range(len(classes)):
            if i == correctClass: continue
            dist += sum(classes[i]); count += len(classes[i])
        
        if count > 0 and dist/float(count) < minCorrectClassDistance:
            for n in graph.getNeighbours(vertex):
                if int(graph.objects[n].getclass()) != correctClass or graph[vertex,n][CLUSTER] == -1: continue
                if newVal == None:  graph[vertex,n] = None
                else:               graph[vertex,n][VALUE] = newVal
                if (n, vertex) in edgesDict[graph[vertex,n][CLUSTER]]: edgesDict[graph[vertex,n][CLUSTER]].remove((n,vertex))
                elif (vertex, n) in edgesDict[graph[vertex,n][CLUSTER]]: edgesDict[graph[vertex,n][CLUSTER]].remove((vertex, n))
                verticesDict[n] = verticesDict[vertex] = -1
                graph[vertex,n][CLUSTER] = -1
    """
    """
    minWrongClassDistance = 2*minCorrectClassDistance
    for key in classes.keys():
        if key == correctClass or classes[key] == []: continue
        minWrongClassDistance = min(minWrongClassDistance, min(classes[key]))
    # if the distance to the wrong class is smaller than to the correct class then delete the edge
    if minWrongClassDistance < minCorrectClassDistance:
        for n in graph.getNeighbours(i):
            if graph.objects[n].getclass().value ==  correctClass: graph[i,n][VALUE] = newVal
    """

# mark edges on the closure with the outerEdgeValue and edges inside a cluster with a innerEdgeValue
# algorithm: for each edge, find all triangles that contain this edge. if the number of triangles is 1, then this edge is on the closure.
# if there is a larger number of triangles, then they must all lie on the same side of this edge, otherwise this edge is not on the closure
def computeClosure(graph, edgesDict, minimumValidIndex, innerEdgeValue, outerEdgeValue):
    closureDict = {}
    for key in edgesDict.keys(): closureDict[key] = []  # create dictionary where each cluster will contain all edges that lie on the closure
    for (i,j) in graph.getEdges():
        if graph[i,j][VALUE] < minimumValidIndex: continue
        if graph[i,j][CLUSTER] == -1:
            #graph[i,j][VALUE] = innerEdgeValue
            continue
        merged = graph.getNeighbours(i) + graph.getNeighbours(j)
        merged.sort()
        k=0; found = [];
        for k in range(len(merged)-1):
            if merged[k] == merged[k+1] and graph[i, merged[k]][VALUE] >= minimumValidIndex and graph[j, merged[k]][VALUE] >= minimumValidIndex: found.append(merged[k])
        graph[i,j][VALUE] = outerEdgeValue
        closureDict[graph[i,j][CLUSTER]].append((i,j))
        if found == []:
            graph[i,j][VALUE] = innerEdgeValue
            closureDict[graph[i,j][CLUSTER]].remove((i,j))
        elif len(found) != 1:
            dir = getPosition(graph.objects[i][0].value, graph.objects[i][1].value, graph.objects[j][0].value, graph.objects[j][1].value, graph.objects[found[0]][0].value, graph.objects[found[0]][1].value)
            for val in found[1:]:
                dir2 = getPosition(graph.objects[i][0].value, graph.objects[i][1].value, graph.objects[j][0].value, graph.objects[j][1].value, graph.objects[val][0].value, graph.objects[val][1].value)
                if dir != dir2:
                    graph[i,j][VALUE] = innerEdgeValue
                    closureDict[graph[i,j][CLUSTER]].remove((i,j))
                    break
    return closureDict

def restoreLinesToOutliers(graph, removedValue, restoredValue):
    for vertex in range(graph.nVertices):
        neighbors = graph.getNeighbours(vertex)
        if len(neighbors) < 3: continue

        # were all lines to vertex removed
        allRemoved = 1
        for n in neighbors:
            if graph[vertex, n][VALUE] != removedValue: allRemoved = 0; break
        if not allRemoved: continue
        c = graph.objects[neighbors[0]].getclass()

        # do all lines lead to the same class value
        allSameClass = 1
        for n in neighbors[1:]:
            if graph.objects[n].getclass() != allSameClass:
                allSameClass = 0; break
        if not allSameClass: continue
        
        for n in neighbors: # restore the lines since it is obviously just one outlier
            graph[vertex, n][VALUE] = restoredValue
    


# set a different value for each cluster
# minimumValidValue - edges that have smaller value than this will be ignored and will represent boundary between different clusters
def enumerateClusters(graph, minimumValidValue):    
    clusterIndex = 1
    verticesDict = {}
    edgesDict = {}
    for (i,j) in graph.getEdges(): graph[i,j][CLUSTER] = -1   # initialize class cluster
    for i in range(graph.nVertices): verticesDict[i] = -1
    
    for i in range(graph.nVertices):
        if graph.getNeighbours(i) == [] or verticesDict[i] != -1: continue
        edgesDict[clusterIndex] = []
        verticesToSearch = [i]
        while verticesToSearch != []:
            current = verticesToSearch.pop()
            if verticesDict[current] != -1: continue
            verticesDict[current] = clusterIndex
            for n in graph.getNeighbours(current):
                if graph[current,n][VALUE] < minimumValidValue: continue
                if verticesDict[n] == -1:
                    verticesToSearch.append(n)
                    edgesDict[clusterIndex].append((n, current))
                    graph[current, n][CLUSTER] = clusterIndex
        clusterIndex += 1
    
    return (edgesDict, verticesDict, clusterIndex-1)

# for edges that have too small number of vertices set edge value to insignificantEdgeValue
# for edges that fall apart because of the distance between points set value to removedEdgeValue
def computeAlphaShapes(graph, edgesDict, insignificantEdgeValue, removedEdgeValue):
    for key in edgesDict.keys():
        edges = edgesDict[key]
        if len(edges) < 6:
            for (i,j) in edges: graph[i,j][VALUE] = insignificantEdgeValue
            continue
        
        lengths = [graph[i,j][DISTANCE] for (i,j) in edges]

        # remove edges that are of lenghts average + standard deviation
        ave = sum(lengths) / len(lengths)
        std = sqrt(sum([(x-ave)*(x-ave) for x in lengths])/len(lengths))
        for index in range(len(lengths)):
            if lengths[index] > ave + std: graph[edges[index][0], edges[index][1]][VALUE] = removedEdgeValue
            #if lengths[index] > ave: graph[edges[index][0], edges[index][1]][VALUE] = removedEdgeValue
        

# alphashapes removed some edges. if after clustering two poins of the edge still belong to the same cluster, restore the edge
def fixDeletedEdges(graph, verticesDict, edgesDict, deletedEdgeValue, repairValue, deleteValue):
    for (i,j) in graph.getEdges():
        if graph[i,j][VALUE] == deletedEdgeValue:
            if verticesDict[i] == verticesDict[j]:
                graph[i,j][VALUE] = repairValue             # restore the value of the edge
                graph[i,j][CLUSTER] = verticesDict[i]       # reset the edge value
                edgesDict[verticesDict[i]].append((i,j))    # re-add the edge to the list of edges
            else:
                if deleteValue == None: graph[i,j] = None
                else:                   graph[i,j][VALUE] = deleteValue
        

# compute the area of the polygons that are not necessarily convex  
def computeAreas(graph, closureDict, outerEdgeValue):
    areaDict = {}
    for key in closureDict.keys():
        currArea = 0.0
        edges = closureDict[key] # select outer edges
        coveredEdges = []
        while len(coveredEdges) < len(edges):
            for (e1, e2) in edges:
                if (e1, e2) not in coveredEdges and (e2, e1) not in coveredEdges: break
            polygons = computePolygons(graph, (e1, e2), outerEdgeValue)
            for poly in polygons:
                currArea += computeAreaOfPolygon(graph, poly)
                for i in range(len(poly)-2): coveredEdges.append((poly[i], poly[i+1]))
                coveredEdges.append((poly[0], poly[-1]))
        areaDict[key] = currArea
    return areaDict

# for a given graph and starting edge return a set of polygons that represent the boundaries. in the simples case there will be only one polygon.
# However there can be also subpolygons and connected polygons
# this method is unable to guaranty correct computation of polygons in case there is an ambiguity - such as in case of 3 connected triangles where each shares two vertices with the other two triangles.
def computePolygons(graph, startingEdge, outerEdgeValue):
    waitingEdges = [startingEdge]
    takenEdges = []
    polygons = []
    while waitingEdges != []:
        (start, end) = waitingEdges.pop()
        if (start, end) in takenEdges or (end,start) in takenEdges: continue
        subpath = [start]
        while end != subpath[0]:
            neighbors = graph.getNeighbours(end)
            outerEdges = []
            for n in neighbors:
                if graph[end, n][VALUE] == outerEdgeValue and n != start and (n, end) not in takenEdges and (end, n) not in takenEdges:
                    outerEdges.append(n)
            if end in subpath:
                i = subpath.index(end)
                polygons.append(subpath[i:])
                subpath = subpath[:i]
            subpath.append(end)
            takenEdges.append((start, end))
            start = end; end = outerEdges[0]
            outerEdges.remove(end)
            if len(outerEdges) > 0:
                for e in outerEdges:
                    if (start, e) not in waitingEdges and (e, start) not in waitingEdges: waitingEdges.append((start, e))
        takenEdges.append((subpath[-1], subpath[0]))
        polygons.append(subpath)
    return polygons

# fast computation of the area of the polygon
# formula can be found at http://en.wikipedia.org/wiki/Polygon
def computeAreaOfPolygon(graph, polygon):
    polygon = polygon + [polygon[0]]
    tempArea = 0.0
    for i in range(len(polygon)-1):
        tempArea += graph.objects[polygon[i]][0]*graph.objects[polygon[i+1]][1] - graph.objects[polygon[i+1]][0]*graph.objects[polygon[i]][1]
    return abs(tempArea)


def getVerticesInPolygons(verticesDict):
    polygonVerticesDict = {}
    for v in verticesDict.keys():
        if verticesDict[v] != -1:
            if not polygonVerticesDict.has_key(verticesDict[v]): polygonVerticesDict[verticesDict[v]] = []
            polygonVerticesDict[verticesDict[v]].append(v)
    for key in polygonVerticesDict.keys():
        if len(polygonVerticesDict[key]) < 2: polygonVerticesDict.pop(key)
    return polygonVerticesDict

"""
def pointInsideCluster(data, closure, xTest, yTest):
    p1 = closure[0][0]
    count = 0
    for (p2, p3) in closure[1:]:
        pos = getPosition(data[p1][0], data[p1][1], data[p2][0], data[p2][1], xTest, yTest)
        pos2 = getPosition(data[p1][0], data[p1][1], data[p3][0], data[p3][1], xTest, yTest)
        if pos2 != pos: continue
        pos3 = getPosition(data[p2][0], data[p2][1], data[p3][0], data[p3][1], xTest, yTest)
        if pos3 == pos: count += 1
    if count % 2 == 0: return 0
    else: return 1
"""


def pointInsideCluster(data, closure, xTest, yTest):
    p1 = closure[0][0]
    count = 0
    for (p2, p3) in closure[1:]:
        x1 = data[p2][0] - xTest; y1 = data[p2][1] - yTest
        x2 = data[p3][0] - xTest; y2 = data[p3][1] - yTest
        if (y1 > 0 and y2 <= 0) or (y2 > 0 and y1 <= 0):
            x = (x1 * y2 - x2 * y1) / (y2 - y1)
            if x > 0: count += 1
    if count % 2 == 1: return 1
    else: return 0
        

class ClusterOptimization(OWBaseWidget):
    EXACT_NUMBER_OF_ATTRS = 0
    MAXIMUM_NUMBER_OF_ATTRS = 1

    settingsList = ["resultListLen", "minExamples", "lastSaveDirName", "attrCont", "attrDisc", "showRank", "showValue", "jitterDataBeforeTriangulation", "createSnapshots"]
    resultsListLenNums = [ 100 ,  250 ,  500 ,  1000 ,  5000 ,  10000, 20000, 50000, 100000, 500000 ]
    resultsListLenList = [str(x) for x in resultsListLenNums]

    def __init__(self, parentWidget = None, graph = None):
        OWBaseWidget.__init__(self, None, "Cluster Dialog")

        self.parentWidget = parentWidget
        self.setCaption("Qt Cluster Dialog")
        self.topLayout = QVBoxLayout( self, 10 ) 
        self.grid=QGridLayout(5,2)
        self.topLayout.addLayout( self.grid, 10 )

        self.graph = graph
        self.minExamples = 0
        self.resultListLen = 1000
        self.maxResultListLen = self.resultsListLenNums[len(self.resultsListLenNums)-1]
        self.onlyOnePerSubset = 1    # used in radviz and polyviz
        self.widgetDir = os.path.realpath(os.path.dirname(__file__)) + "/"
        self.parentName = "Projection"
        self.lastSaveDirName = os.getcwd() + "/"
        self.attrCont = 1
        self.attrDisc = 1
        self.rawdata = None
        self.subsetdata = None
        self.arguments = []
        self.selectedClasses = []
        self.optimizationType = 0
        self.jitterDataBeforeTriangulation = 0

        self.showRank = 0
        self.showValue = 1
        self.allResults = []
        self.shownResults = []
        self.attrLenDict = {}
        self.datasetName = ""
        self.dataset = None
        self.cancelOptimization = 0
        self.cancelArgumentation = 0
        self.pointStability = None
        self.pointStabilityCount = None
        self.argumentationClassValue = 0
        self.createSnapshots = 1

        self.loadSettings()

        self.tabs = QTabWidget(self, 'tabWidget')
        
        self.MainTab = QVGroupBox(self)
        self.SettingsTab = QVGroupBox(self)
        self.ArgumentationTab = QVGroupBox(self)
        self.ManageTab = QVGroupBox(self)
        
        self.tabs.insertTab(self.MainTab, "Main")
        self.tabs.insertTab(self.SettingsTab, "Settings")
        self.tabs.insertTab(self.ArgumentationTab, "Argumentation")
        self.tabs.insertTab(self.ManageTab, "Manage & Save")
        
        # main tab
        self.optimizationBox = OWGUI.widgetBox(self.MainTab, " Evaluate ")
        self.resultsBox = OWGUI.widgetBox(self.MainTab, " Projection List, Most Interesting First ")
        self.resultsDetailsBox = OWGUI.widgetBox(self.MainTab, " Shown Details in Projections List " , orientation = "horizontal")

        # settings tab
        self.jitteringBox = OWGUI.checkBox(self.SettingsTab, self, 'jitterDataBeforeTriangulation', 'Use data jittering', box = " Jittering options ", tooltip = "Use jittering if you get an exception when evaluating clusters. \nIt adds a small random noise to poins which fixes the triangluation problems.")
        self.heuristicsSettingsBox = OWGUI.widgetBox(self.SettingsTab, " Heuristics for Attribute Ordering ")
        self.miscSettingsBox = OWGUI.widgetBox(self.SettingsTab, " Miscellaneous Settings ")
        #self.miscSettingsBox.hide()

        # argumentation tab        
        self.argumentationStartBox = OWGUI.widgetBox(self.ArgumentationTab, " Arguments ")
        self.findArgumentsButton = OWGUI.button(self.argumentationStartBox, self, "Find arguments", callback = self.findArguments)
        f = self.findArgumentsButton.font(); f.setBold(1);  self.findArgumentsButton.setFont(f)
        self.stopArgumentationButton = OWGUI.button(self.argumentationStartBox, self, "Stop searching", callback = self.stopArgumentationClick)
        self.stopArgumentationButton.setFont(f)
        self.stopArgumentationButton.hide()
        self.createSnapshotCheck = OWGUI.checkBox(self.argumentationStartBox, self, 'createSnapshots', 'Create snapshots of projections (a bit slower)', tooltip = "Show each argument with a projections screenshot.\nTakes a bit more time, since the projection has to be created.")
        self.classValueList = OWGUI.comboBox(self.ArgumentationTab, self, "argumentationClassValue", box = " Arguments for class: ", tooltip = "Select the class value that you wish to see arguments for", callback = self.argumentationClassChanged)
        self.argumentBox = OWGUI.widgetBox(self.ArgumentationTab, " Arguments for the selected class value ")
        self.argumentList = QListBox(self.argumentBox)
        self.argumentList.setMinimumSize(200,200)
        self.connect(self.argumentList, SIGNAL("selectionChanged()"),self.argumentSelected)


        # manage tab
        self.classesBox = OWGUI.widgetBox(self.ManageTab, " Class values in data set ")   
        self.manageResultsBox = OWGUI.widgetBox(self.ManageTab, " Manage Projections ")        
        self.evaluateBox = OWGUI.widgetBox(self.ManageTab, " Evaluate Current Projection / Classifier ")
        
        # ###########################
        # MAIN TAB
        self.buttonBox = OWGUI.widgetBox(self.optimizationBox, orientation = "horizontal")
        self.label1 = QLabel('Projections with ', self.buttonBox)
        self.optimizationTypeCombo = OWGUI.comboBox(self.buttonBox, self, "optimizationType", items = ["    exactly    ", "  maximum  "] )
        self.attributeCountCombo = OWGUI.comboBox(self.buttonBox, self, "attributeCountIndex", tooltip = "Evaluate only projections with exactly (or maximum) this number of attributes")
        self.attributeLabel = QLabel(' attributes', self.buttonBox)

        self.startOptimizationButton = OWGUI.button(self.optimizationBox, self, "Start evaluating projections")
        f = self.startOptimizationButton.font()
        f.setBold(1)
        self.startOptimizationButton.setFont(f)
        self.stopOptimizationButton = OWGUI.button(self.optimizationBox, self, "Stop evaluation", callback = self.stopOptimizationClick)
        self.stopOptimizationButton.setFont(f)
        self.stopOptimizationButton.hide()
        self.optimizeGivenProjectionButton = OWGUI.button(self.optimizationBox, self, "Optimize current projection")
        self.optimizeGivenProjectionButton.hide()

        for i in range(3,15):
            self.attributeCountCombo.insertItem(str(i))
        self.attributeCountCombo.insertItem("ALL")
        self.attributeCountIndex = 0

        self.resultList = QListBox(self.resultsBox)
        #self.resultList.setSelectionMode(QListBox.Extended)   # this would be nice if could be enabled, but it has a bug - currentItem doesn't return the correct value if this is on
        self.resultList.setMinimumSize(200,200)

        self.showRankCheck = OWGUI.checkBox(self.resultsDetailsBox, self, 'showRank', 'Rank', callback = self.updateShownProjections, tooltip = "Show projection ranks")
        self.showValueCheck = OWGUI.checkBox(self.resultsDetailsBox, self, 'showValue', 'Cluster Value', callback = self.updateShownProjections, tooltip = "Show the cluster value")

        # ##########################
        # SETTINGS TAB
        #OWGUI.radioButtonsInBox(self.heuristicsSettingsBox, self, "attrCont", [val for (val, measure) in contMeasures], box = " Ordering of Continuous Attributes")
        #OWGUI.radioButtonsInBox(self.heuristicsSettingsBox, self, "attrDisc", [val for (val, measure) in discMeasures], box = " Ordering of Discrete Attributes")
        contHeuristic = OWGUI.widgetBox(self.heuristicsSettingsBox, " Ordering of Continuous Attributes", orientation = "vertical")
        OWGUI.comboBox(contHeuristic, self, "attrCont", items = [val for (val, m) in contMeasures])
        OWGUI.comboBox(self.heuristicsSettingsBox, self, "attrDisc", box = " Ordering of Discrete Attributes", items = [val for (val, m) in discMeasures])

        self.resultListCombo = OWGUI.comboBoxWithCaption(self.miscSettingsBox, self, "resultListLen", "Maximum length of projection list:   ", tooltip = "Maximum length of the list of interesting projections. This is also the number of projections that will be saved if you click Save button.", items = self.resultsListLenNums, callback = self.updateShownProjections, sendSelectedValue = 1, valueType = int)
        self.minTableLenEdit = OWGUI.lineEdit(self.miscSettingsBox, self, "minExamples", "Minimum examples in data set:        ", orientation = "horizontal", tooltip = "Due to missing values, different subsets of attributes can have different number of examples. Projections with less than this number of examples will be ignored.", valueType = int)

        # ##########################
        # SAVE & MANAGE TAB

        self.classesCaption = QLabel('Select classes you wish to separate:', self.classesBox)
        self.classesList = QListBox(self.classesBox)
        self.classesList.setSelectionMode(QListBox.Multi)
        self.classesList.setMinimumSize(60,60)
        self.connect(self.classesList, SIGNAL("selectionChanged()"), self.updateShownProjections)

        self.buttonBox3 = OWGUI.widgetBox(self.evaluateBox, orientation = "horizontal")
        self.clusterStabilityButton = OWGUI.button(self.buttonBox3, self, 'Show cluster stability', self.evaluatePointsInClusters)
        self.clusterStabilityButton.setToggleButton(1)
        #self.saveProjectionButton = OWGUI.button(self.buttonBox3, self, 'Save projection')
        self.saveBestButton = OWGUI.button(self.buttonBox3, self, "Save best graphs", self.exportMultipleGraphs)

        self.attrLenCaption = QLabel('Number of concurrently visualized attributes:', self.manageResultsBox)
        self.attrLenList = QListBox(self.manageResultsBox)
        self.attrLenList.setSelectionMode(QListBox.Multi)
        self.attrLenList.setMinimumSize(60,60)
        self.connect(self.attrLenList, SIGNAL("selectionChanged()"), self.attrLenListChanged)

        self.buttonBox6 = OWGUI.widgetBox(self.manageResultsBox, orientation = "horizontal")
        self.buttonBox7 = OWGUI.widgetBox(self.manageResultsBox, orientation = "horizontal")
        self.loadButton = OWGUI.button(self.buttonBox6, self, "Load", self.load)
        self.saveButton = OWGUI.button(self.buttonBox6, self, "Save", self.save)
        self.clearButton = OWGUI.button(self.buttonBox7, self, "Clear results", self.clearResults)
        self.closeButton = OWGUI.button(self.buttonBox7, self, "Close", self.hide)
        self.resize(350,550)
        self.setMinimumWidth(350)
        self.tabs.setMinimumWidth(350)

    # ##############################################################
    # EVENTS
    # ##############################################################
    
    # result list can contain projections with different number of attributes
    # user clicked in the listbox that shows possible number of attributes of result list
    # result list must be updated accordingly
    def attrLenListChanged(self):
        # check which attribute lengths do we want to show
        self.attrLenDict = {}
        for i in range(self.attrLenList.count()):
            intVal = int(str(self.attrLenList.text(i)))
            selected = self.attrLenList.isSelected(i)
            self.attrLenDict[intVal] = selected
        self.updateShownProjections()

    def clearResults(self):
        self.allResults = []
        self.shownResults = []
        self.resultList.clear()
        self.attrLenDict = {}
        self.attrLenList.clear()
        self.pointStability = None

    def getSelectedClassValues(self):
        selectedClasses = []
        for i in range(self.classesList.count()-1):
            if self.classesList.isSelected(i): selectedClasses.append(i)
        if self.classesList.isSelected(self.classesList.count()-1):
            selectedClasses.append(-1)      # "all clusters" is represented by -1 in the selectedClasses array
        return selectedClasses

    # ##############################################################
    # ##############################################################
    def evaluateClusters(self, data):
        #fullgraph = orange.triangulate(data,3)
        graph = orange.triangulate(data,3)
        graph.returnIndices = 1
        computeDistances(graph)
        removeEdgesToDifferentClasses(graph, -3)    # None
        removeSingleLines(graph, None, None, 0, -1)
        edgesDict, verticesDict, count = enumerateClusters(graph, 0)

        # find points that define the edge of a cluster with one class value
        # these points used when we search for nearest poinst of a specific cluster
        closureDict = computeClosure(graph, edgesDict, 1, 1, 1)
        bigPolygonVerticesDict = getVerticesInPolygons(verticesDict)  # compute which points lie in which cluster
        otherClassDict = getPointsWithDifferentClassValue(graph, closureDict)

        removeSingleTrianglesAndLines(graph, edgesDict, verticesDict, -1)
        computeAlphaShapes(graph, edgesDict, -1, 0)                      # try to break too empty clusters into more clusters
        edgesDict, verticesDict, count = enumerateClusters(graph, 1)     # create the new clustering
        fixDeletedEdges(graph, verticesDict, edgesDict, 0, 1, -1)  # None       # restore edges that were deleted with computeAlphaShapes and did not result breaking the cluster
        closureDict = computeClosure(graph, edgesDict, 1, 1, 2)
        removeDistantPointsFromClusters(graph, edgesDict, verticesDict, closureDict, -1)
        removeSingleLines(graph, None, None, 1, -1)   # None
        edgesDict, verticesDict, count = enumerateClusters(graph, 1)     # reevaluate clusters - now some clusters might have disappeared
        removeSingleTrianglesAndLines(graph, edgesDict, verticesDict, -1)
        # add edges that were removed by removing single points with different class value
        closureDict = computeClosure(graph, edgesDict, 1, 1, 2)
        polygonVerticesDict = getVerticesInPolygons(verticesDict)  # compute which points lie in which cluster

        areaDict = computeAreas(graph, closureDict, 2)

        valueDict = {}
        otherDict = {}
        for key in closureDict.keys():
            if polygonVerticesDict[key] < 6: continue            # if the cluster has less than 6 points ignore it
            points = len(polygonVerticesDict[key]) - len(closureDict[key])    # number of points in the interior
            if points < 3: continue     # happens when there are two or more polygons that are connected in only one point
            #print "vertices ", polygonVerticesDict[key]
            #print "closure ", closureDict[key]

            """
            vertices = [i for (i,j) in closureDict[key]] + [j for (i,j) in closureDict[key]]
            vertices.sort()
            for i in range(len(vertices)-1)[::-1]:
                if vertices[i] == vertices[i+1]: vertices.remove(vertices[i])

            sameClass = []; diffClass = []
            for vertex in vertices:
                for n in graph.getNeighbours(vertex):
                    if graph[n, vertex][OWClusterOptimization.VALUE] >= 0: continue
                    if graph.objects[n].getclass() == graph.objects[vertex].getclass(): sameClass.append(graph[n,vertex][OWClusterOptimization.DISTANCE])
                    else: diffClass.append(graph[n,vertex][OWClusterOptimization.DISTANCE])
            sameClass.sort()
            diffClass.sort()
            sameClass = [i*i for i in sameClass[:5]]
            diffClass = [i*i for i in diffClass[:5]]
            dist = 3 * sum(diffClass)/max(1, len(diffClass)) + 1 * sum(sameClass)/max(1,len(sameClass))
            dist /= max(1, len(sameClass) + len(diffClass))
            """

            index = -1
            for bigKey in bigPolygonVerticesDict.keys():
                if polygonVerticesDict[key][0] in bigPolygonVerticesDict[bigKey]:
                    index = bigKey
                    break
            if index != -1:
                # compute the center of the current cluster
                xAve = sum([graph.objects[i][0] for i in polygonVerticesDict[key]]) / len(polygonVerticesDict[key])
                yAve = sum([graph.objects[i][1] for i in polygonVerticesDict[key]]) / len(polygonVerticesDict[key])
                diffClass = []
                for v in otherClassDict[index]:
                    d = sqrt((graph.objects[v][0]-xAve)*(graph.objects[v][0]-xAve) + (graph.objects[v][1]-yAve)*(graph.objects[v][1]-yAve))
                    diffClass.append(d)
                diffClass.sort()
                diffClass = [i*i for i in diffClass[:5]]
                dist = sum(diffClass) / float(max(1, len(diffClass)))
            
            else:
                print "index is -1"
                continue

            points = sqrt(points)       # make a smaller effect of the number of points
            area = areaDict[key]
            area = sqrt(area)
            value = points * dist / area
            valueDict[key] = value
            otherDict[key] = (graph.objects[polygonVerticesDict[key][0]].getclass().value, value, points, dist, area)
        
        return graph, valueDict, closureDict, polygonVerticesDict, otherDict


    def evaluatePointsInClusters(self):
        if not self.clusterStabilityButton.isOn(): return
        self.pointStability = Numeric.zeros(len(self.rawdata), Numeric.Float)
        self.pointStabilityCount = [0 for i in range(len(self.rawdata.domain.classVar.values))]       # for each class value create a counter that will count the number of clusters for it

        for i in range(len(self.allResults)):
            if type(self.allResults[i][CLASS]) != list: continue    # ignore all projections except the ones that show all clusters in the picture
            for j in range(len(self.allResults[i][VERTICES])):
                self.pointStabilityCount[self.allResults[i][CLASS][j]] += 1
                vertices = self.allResults[i][VERTICES][j]
                validData = self.graph.getValidList([self.graph.attributeNames.index(self.allResults[i][ATTR_LIST][0]), self.graph.attributeNames.index(self.allResults[i][ATTR_LIST][1])])
                indices = Numeric.compress(validData, Numeric.array(range(len(self.rawdata))))
                indicesToOriginalTable = Numeric.take(indices, vertices)
                tempArray = Numeric.zeros(len(self.rawdata))
                Numeric.put(tempArray, indicesToOriginalTable, Numeric.ones(len(indicesToOriginalTable)))
                self.pointStability += tempArray

        for i in range(len(self.rawdata)):
            self.pointStability[i] /= float(self.pointStabilityCount[int(self.rawdata[i].getclass())])
        #self.pointStability = [1.0 - val for val in self.pointStability]

            
    def updateShownProjections(self, *args):
        self.resultList.clear()
        self.shownResults = []
        self.selectedClasses = self.getSelectedClassValues()
        i = 0
    
        while self.resultList.count() < self.resultListLen and i < len(self.allResults):
            if self.attrLenDict[len(self.allResults[i][ATTR_LIST])] != 1: i+=1; continue
            if type(self.allResults[i][CLASS]) == list and -1 not in self.selectedClasses: i+=1; continue
            if type(self.allResults[i][CLASS]) == int and self.allResults[i][CLASS] not in self.selectedClasses: i+=1; continue

            string = ""
            if self.showRank: string += str(i+1) + ". "
            if self.showValue: string += "%.2f - " % (self.allResults[i][VALUE])

            if self.allResults[i][STR_LIST] != "": string += self.allResults[i][STR_LIST]
            else: string += self.buildAttrString(self.allResults[i][ATTR_LIST])
            
            self.resultList.insertItem(string)
            self.shownResults.append(self.allResults[i])
            i+=1
        if self.resultList.count() > 0: self.resultList.setCurrentItem(0)        


    # save input dataset, get possible class values, ...
    def setData(self, data):
        if hasattr(data, "name"): self.datasetName = data.name
        else: self.datasetName = ""
        self.rawdata = data
        self.selectedClasses = []
        self.classesList.clear()
        self.classValueList.clear()
        self.argumentList.clear()
        self.arguments = []
        self.clearResults()
                
        if not data: return
        if not (data.domain.classVar and data.domain.classVar.varType == orange.VarTypes.Discrete): return

        # add class values
        for i in range(len(data.domain.classVar.values)):
            self.classesList.insertItem(data.domain.classVar.values[i])
            self.classValueList.insertItem(data.domain.classVar.values[i])
        self.classesList.insertItem("All clusters")
        self.classesList.selectAll(1)
        if len(data.domain.classVar.values) > 0: self.classValueList.setCurrentItem(0)

    # save subsetdata. first example from this dataset can be used with argumentation - it can find arguments for classifying the example to the possible class values
    def setSubsetData(self, subsetdata):
        self.subsetdata = subsetdata
        self.arguments = []
        self.argumentList.clear()
    
                
    # given a dataset return a list of (val, attrName) where val is attribute "importance" and attrName is name of the attribute
    def getEvaluatedAttributes(self, data):
        return OWVisAttrSelection.evaluateAttributes(data, contMeasures[self.attrCont][1], discMeasures[self.attrDisc][1])
    
    def addResult(self, value, closure, vertices, attrList, classValue, tryIndex, strList = ""):
        self.insertItem(value, closure, vertices, attrList, classValue, self.findTargetIndex(value), tryIndex, strList)
        qApp.processEvents()        # allow processing of other events

    # use bisection to find correct index
    def findTargetIndex(self, value):
        top = 0; bottom = len(self.allResults)

        while (bottom-top) > 1:
            mid  = (bottom + top)/2
            if max(value, self.allResults[mid][VALUE]) == value: bottom = mid
            else: top = mid

        if len(self.allResults) == 0: return 0
        if max(value, self.allResults[top][VALUE]) == value:
            return top
        else: 
            return bottom

    # insert new result - give parameters: value of the cluster, closure, list of attributes.
    # parameter strList can be a pre-formated string containing attribute list (used by polyviz)
    def insertItem(self, value, closure, vertices, attrList, classValue, index, tryIndex, strList = ""):
        if index < self.maxResultListLen:
            self.allResults.insert(index, (value, closure, vertices, attrList, classValue, tryIndex, strList))
        if index < self.resultListLen:
            string = ""
            if self.showRank: string += str(index+1) + ". "
            if self.showValue: string += "%.2f - " % (value)

            if strList != "": string += strList
            else: string += self.buildAttrString(attrList)

            self.resultList.insertItem(string, index)
            self.shownResults.insert(index, (value, closure, vertices, attrList, classValue, tryIndex, strList))

        # remove worst projection if list is too long
        if self.resultList.count() > self.resultListLen:
            self.resultList.removeItem(self.resultList.count()-1)
            self.shownResults.pop()
    
    def finishedAddingResults(self):
        self.cancelOptimization = 0
        
        self.attrLenList.clear()
        self.attrLenDict = {}
        maxLen = -1
        for i in range(len(self.shownResults)):
            if len(self.shownResults[i][ATTR_LIST]) > maxLen:
                maxLen = len(self.shownResults[i][ATTR_LIST])
        if maxLen == -1: return
        if maxLen == 2: vals = [2]
        else: vals = range(3, maxLen+1)
        for val in vals:
            self.attrLenList.insertItem(str(val))
            self.attrLenDict[val] = 1
        self.attrLenList.selectAll(1)
        self.resultList.setCurrentItem(0)

    
    # ##############################################################
    # Loading and saving projection files
    # ##############################################################

    # save the list into a file - filename can be set if you want to call this function without showing the dialog
    def save(self, filename = None):
        if filename == None:
            # get file name
            if self.datasetName != "":
                filename = "%s - %s" % (os.path.splitext(os.path.split(self.datasetName)[1])[0], self.parentName)
            else:
                filename = "%s" % (self.parentName)
            qname = QFileDialog.getSaveFileName( self.lastSaveDirName + "/" + filename, "Interesting clusters (*.clu)", self, "", "Save Clusters")
            if qname.isEmpty(): return
            name = str(qname)
        else:
            name = filename

        # take care of extension
        if os.path.splitext(name)[1] != ".clu":
            name = name + ".clu"

        dirName, shortFileName = os.path.split(name)
        self.lastSaveDirName = dirName

        # open, write and save file
        file = open(name, "wt")
        attrs = ["resultListLen", "parentName"]
        dict = {}
        for attr in attrs: dict[attr] = self.__dict__[attr]
        file.write("%s\n" % (str(dict)))
        file.write("%s\n" % str(self.selectedClasses))
        for (value, closure, vertices, attrList, classValue, tryIndex, strList) in self.shownResults:
            if type(classValue) != list: continue
            s = "(%s, %s, %s, %s, %s, %s, '%s')\n" % (str(value), str(closure), str(vertices), str(attrList), str(classValue), str(tryIndex), strList)
            file.write(s)
        file.flush()
        file.close()


    # load projections from a file
    def load(self):
        self.clearResults()
        if self.rawdata == None:
            QMessageBox.critical(None,'Load','There is no data. First load a data set and then load a cluster file',QMessageBox.Ok)
            return
                
        name = QFileDialog.getOpenFileName( self.lastSaveDirName, "Interesting clusters (*.clu)", self, "", "Open Clusters")
        if name.isEmpty(): return
        name = str(name)

        dirName, shortFileName = os.path.split(name)
        self.lastSaveDirName = dirName

        file = open(name, "rt")
        settings = eval(file.readline()[:-1])
        if settings.has_key("parentName") and settings["parentName"] != self.parentName:
            QMessageBox.critical( None, "Cluster Dialog", 'Unable to load cluster file. It was saved for %s method'%(settings["parentName"]), QMessageBox.Ok)
            file.close()
            return

        self.setSettings(settings)

        # find if it was computed for specific class values
        line = file.readline()[:-1];
        selectedClasses = eval(line)
        for i in range(len(self.rawdata.domain.classVar.values)):
            self.classesList.setSelected(i, i in selectedClasses)
        if -1 in selectedClasses: self.classesList.setSelected(self.classesList.count()-1, 1)

        line = file.readline()[:-1];
        while (line != ""):
            (value, closure, vertices, attrList, classValue, tryIndex, strList) = eval(line)
            self.addResult(value, closure, vertices, attrList, classValue, tryIndex, strList)
            for i in range(len(classValue)):
                self.addResult(tryIndex[i][1], closure[i], vertices[i], attrList, classValue[i], tryIndex[i], strList)
            line = file.readline()[:-1]
        file.close()

        # update loaded results
        self.finishedAddingResults()


    # disable all controls while evaluating projections
    def disableControls(self):
        self.optimizationTypeCombo.setEnabled(0)
        self.attributeCountCombo.setEnabled(0)
        self.startOptimizationButton.hide()
        self.stopOptimizationButton.show()
        self.manageResultsBox.setEnabled(0)
        self.evaluateBox.setEnabled(0)
        self.heuristicsSettingsBox.setEnabled(0)
        self.classesList.setEnabled(0)
        self.resultsDetailsBox.setEnabled(0)
        self.miscSettingsBox.setEnabled(0)
        self.optimizeGivenProjectionButton.setEnabled(0)
        self.argumentBox.setEnabled(0)
        self.classValueList.setEnabled(0)
        self.argumentationStartBox.setEnabled(0)

    def enableControls(self):    
        self.optimizationTypeCombo.setEnabled(1)
        self.attributeCountCombo.setEnabled(1)
        self.startOptimizationButton.show()
        self.stopOptimizationButton.hide()
        self.manageResultsBox.setEnabled(1)
        self.evaluateBox.setEnabled(1)
        self.heuristicsSettingsBox.setEnabled(1)
        self.classesList.setEnabled(1)
        self.resultsDetailsBox.setEnabled(1)
        self.miscSettingsBox.setEnabled(1)
        self.optimizeGivenProjectionButton.setEnabled(1)
        self.argumentBox.setEnabled(1)
        self.classValueList.setEnabled(1)
        self.argumentationStartBox.setEnabled(1)

    # ##############################################################
    # exporting multiple pictures
    # ##############################################################
    def exportMultipleGraphs(self):
        (text, ok) = QInputDialog.getText('Qt Graph count', 'How many of the best projections do you wish to save?')
        if not ok: return
        self.bestGraphsCount = int(str(text))

        self.sizeDlg = OWDlgs.OWChooseImageSizeDlg(self.graph)
        self.sizeDlg.disconnect(self.sizeDlg.okButton, SIGNAL("clicked()"), self.sizeDlg.accept)
        self.sizeDlg.connect(self.sizeDlg.okButton, SIGNAL("clicked()"), self.saveToFileAccept)
        self.sizeDlg.exec_loop()

    def saveToFileAccept(self):
        fileName = str(QFileDialog.getSaveFileName("Graph","Portable Network Graphics (*.PNG);;Windows Bitmap (*.BMP);;Graphics Interchange Format (*.GIF)", None, "Save to..", "Save to.."))
        if fileName == "": return
        (fil,ext) = os.path.splitext(fileName)
        ext = ext.replace(".","")
        if ext == "":	
        	ext = "PNG"  	# if no format was specified, we choose png
        	fileName = fileName + ".png"
        ext = ext.upper()

        (fil, extension) = os.path.splitext(fileName)
        size = self.sizeDlg.getSize()
        for i in range(1, min(self.resultList.count(), self.bestGraphsCount+1)):
            self.resultList.setSelected(i-1, 1)
            self.graph.replot()
            name = fil + " (%02d)" % i + extension
            self.sizeDlg.saveToFileDirect(name, ext, size)
        QDialog.accept(self.sizeDlg)


    # ######################################################
    # Auxiliary functions
    # ######################################################
    def getOptimizationType(self):
        return self.optimizationType
   
    # from a list of attributes build a nice string with attribute names
    def buildAttrString(self, attrList):
        if len(attrList) == 0: return ""
        strList = attrList[0]
        for item in attrList[1:]:
            strList = strList + ", " + item
        return strList

    def getAllResults(self):
        return self.allResults

    def getShownResults(self):
        return self.shownResults

    def getSelectedCluster(self):
        if self.resultList.count() == 0: return None
        return self.shownResults[self.resultList.currentItem()]

    def resizeEvent(self, ev):
        self.tabs.resize(ev.size().width(), ev.size().height())

    def stopOptimizationClick(self):
        self.cancelOptimization = 1

    def isOptimizationCanceled(self):
        return self.cancelOptimization

    def destroy(self, dw, dsw):
        self.saveSettings()


    # ######################################################
    # Argumentation functions
    # ######################################################
    def findArguments(self):
        self.cancelArgumentation = 0
        self.arguments = [[] for i in range(self.classValueList.count())]
        self.argumentList.clear()
        snapshots = self.createSnapshots
        
        if self.subsetdata == None:
            QMessageBox.information( None, "Argumentation", 'To find arguments you first have to provide a new example that you wish to classify. \nYou can do this by sending the example to the visualization widget through the "Example Subset" signal.', QMessageBox.Ok + QMessageBox.Default)
            return
        if len(self.shownResults) == 0:
            QMessageBox.information( None, "Argumentation", 'To find arguments you first have to find clusters in some projections by clicking "Find arguments" in the Main tab.', QMessageBox.Ok + QMessageBox.Default)
            return

        example = self.subsetdata[0]
        testExample = []
        for attr in example.domain.attributes:
            if example[attr].isSpecial(): testExample.append(-1)
            else: testExample.append((example[attr].value - float(self.graph.attrValues[attr.name][0])) / float(self.graph.attrValues[attr.name][1]-self.graph.attrValues[attr.name][0]))
        self.findArgumentsButton.hide()
        self.stopArgumentationButton.show()
        if snapshots: self.parentWidget.setMinimalGraphProperties()

        for index in range(len(self.allResults)):
            if self.cancelArgumentation: break
            (value, closure, vertices, attrList, classValue, tryIndex, strList) = self.allResults[index]

            qApp.processEvents()
            
            if type(classValue) == list: continue       # the projection contains several clusters
            possiblyInside = 1
            for attr in attrList:
                if testExample[self.graph.attributeNames.index(attr)] < 0.0 or testExample[self.graph.attributeNames.index(attr)] > 1.0:
                    possiblyInside = 0
                    break
            if not possiblyInside: continue

            [xTest, yTest] = self.graph.getProjectedPointPosition(attrList, [testExample[self.graph.attributeNames.index(attrList[i])] for i in range(len(attrList))])
            array = self.graph.createProjectionAsNumericArray([self.graph.attributeNames.index(attr) for attr in attrList])
            short = Numeric.take(array, vertices)
            m = min(short); M = max(short)
            if xTest < m[0] or xTest > M[0] or yTest < m[1] or yTest > M[1]: continue       # the point is definitely not inside the cluster

            if not pointInsideCluster(array, closure, xTest, yTest): continue

            pic = None
            if snapshots:
                # if the point lies inside a cluster -> save this figure into a pixmap
                self.parentWidget.showAttributes(attrList, clusterClosure = closure)
                painter = QPainter()
                pic = QPixmap(QSize(120,120))
                painter.begin(pic)
                painter.fillRect(pic.rect(), QBrush(Qt.white)) # make background same color as the widget's background
                self.graph.printPlot(painter, pic.rect())
                painter.flush()
                painter.end()

            self.arguments[classValue].append((pic, value, attrList, index))
            if classValue == self.classValueList.currentItem():
                if snapshots: self.argumentList.insertItem(pic, "%.2f - %s" %(value, attrList))
                else:         self.argumentList.insertItem("%.2f - %s" %(value, attrList))

        self.stopArgumentationButton.hide()
        self.findArgumentsButton.show()
        self.parentWidget.restoreGraphProperties()
       
        
    def stopArgumentationClick(self):
        self.cancelArgumentation = 1
    
    def argumentationClassChanged(self):
        self.argumentList.clear()
        if len(self.arguments) == 0: return
        ind = self.classValueList.currentItem()
        for i in range(len(self.arguments[ind])):
            val = self.arguments[ind][i]
            if val[0] != None:  self.argumentList.insertItem(val[0], "%.2f - %s" %(val[1], val[2]))
            else:               self.argumentList.insertItem("%.2f - %s" %(val[1], val[2]))

    def argumentSelected(self):
        ind = self.argumentList.currentItem()
        classInd = self.classValueList.currentItem()
        self.parentWidget.showAttributes(self.arguments[classInd][ind][2], clusterClosure = self.allResults[self.arguments[classInd][ind][3]][CLOSURE])
        


#test widget appearance
if __name__=="__main__":
    import sys
    a = QApplication(sys.argv)
    ow = ClusterOptimization()
    a.setMainWidget(ow)
    ow.show()
    a.exec_loop()        