CIRCLE=0
SQUARE=1
ROUND_RECT=2

NOTHING = 0
ZOOMING = 1
SELECT_RECTANGLE = 2
SELECT_POLYGON = 3
MOVE_SELECTION = 100

import copy

from OWGraph import *
from Numeric import *
from orngScaleScatterPlotData import *
from OWGraphTools import UnconnectedLinesCurve

class OWGraphDrawerCanvas(OWGraph):
    def __init__(self, master, parent = None, name = "None"):
        OWGraph.__init__(self, parent, name)
        self.master = master
        self.parent = parent
        self.labelText = []
        self.tooltipText = []
        self.vertices = {}         # slovar vozlisc oblike  orngIndex: vertex_objekt
        self.edges = {}            # slovar povezav oblike  curveKey: edge_objekt
        self.indexPairs = {}       # slovar oblike CurveKey: orngIndex   (za vozlisca)
        self.selection = []        # seznam izbranih vozlisc (njihovih indeksov)
        self.selectionStyles = {}  # slovar stilov izbranih vozlisc
        self.colorIndex = -1
        self.visualizer = None
        self.selectedCurve = None
        self.selectedVertex = None
        self.vertexDegree = []     # seznam vozlisc oblike (vozlisce, stevilo povezav), sortiran po stevilu povezav
        self.edgesKey = -1
        self.vertexSize = 6
        self.nVertices = 0
        self.enableXaxis(0)
        self.enableYLaxis(0)
        self.state = NOTHING  #default je rocno premikanje
        self.hiddenNodes = []
        self.markedNodes = set()
        self.markWithRed = False
        self.circles = []
        self.tooltipNeighbours = 2
        self.selectionNeighbours = 2
        self.freezeNeighbours = False
        
        self.enableWheelZoom = 1

        
    def setHiddenNodes(self, nodes):
        self.hiddenNodes = nodes
        self.updateData()
        self.updateCanvas()
        
       
    def addSelection(self, ndx, replot = True):
        #print("add selection")
        change = False
        if hasattr(ndx, "__iter__"):
            for v in ndx:
                if not v in self.selection and not v in self.hiddenNodes:
                    (key, neighbours) = self.vertices[v]
                    self.selectionStyles[v] = self.curve(key).symbol().brush().color().name()
                    newSymbol = QwtSymbol(QwtSymbol.Ellipse, QBrush(QColor(self.selectionStyles[v])), QPen(Qt.yellow, 3), QSize(10, 10))
                    self.setCurveSymbol(key, newSymbol)
                    self.selection.append(v);
                    change = True
        else:
            if not ndx in self.selection and not ndx in self.hiddenNodes:
                (key, neighbours) = self.vertices[ndx]
                color = self.curve(key).symbol().pen().color().name()
                self.selectionStyles[int(ndx)] = color
                newSymbol = QwtSymbol(QwtSymbol.Ellipse, QBrush(QColor(self.selectionStyles[ndx])), QPen(Qt.yellow, 3), QSize(10, 10))
                self.setCurveSymbol(key, newSymbol)
                self.selection.append(ndx);
                #self.visualizer.filter[ndx] = True
                change = True

        if change:
            if replot:
                self.replot()
                
            self.markSelectionNeighbours()
        
        return change

        
    def removeVertex(self, v):
        if v in self.selection:
            (key, neighbours) = self.vertices[v]
            newSymbol = QwtSymbol(QwtSymbol.Ellipse, QBrush(), QPen(QColor(self.selectionStyles[v])), QSize(6, 6))
            self.setCurveSymbol(key, newSymbol)
            selection.remove(v)
            del self.selectionStyles[v]
            return True
        return False
        
    def removeSelection(self, ndx = None, replot = True):
        #print("remove selection")
        change = False
        if ndx is None:
            for v in self.selection:
                (key, neighbours) = self.vertices[v]
                color = self.selectionStyles[v]
                #print color
                newSymbol = QwtSymbol(QwtSymbol.Ellipse, QBrush(), QPen(QColor(color)), QSize(6, 6))
                self.setCurveSymbol(key, newSymbol)
            
            self.selection = []
            #self.visualizer.unselectAll()
            self.selectionStyles = {}
            change = True
            
        elif isinstance(ndx, list):
            for v in ndx:
                change = self.removeVertex(v) or change
                    
        else:
            change = self.removeVertex(ndx)
        
        if change:
            if replot:
                self.replot()
                
            self.markSelectionNeighbours()
        
            
    def selectConnectedNodes(self, distance):
        if distance <= 0:
            return
        
        #print "distance: " + str(distance)
        sel = set(self.selection)
        for v in self.selection:
            neighbours = set(self.visualizer.graph.getNeighbours(v))
            #print neighbours
            self.selectNeighbours(sel, neighbours - sel, 1, distance);
            
        self.removeSelection()
        for ndx in sel:
            (key, neighbours) = self.vertices[ndx]
            self.selectionStyles[ndx] = self.curve(key).symbol().brush().color().name()
            newSymbol = QwtSymbol(QwtSymbol.Ellipse, QBrush(QColor(self.selectionStyles[ndx])), QPen(Qt.yellow, 3), QSize(10, 10))
            self.setCurveSymbol(key, newSymbol)
            self.selection.append(ndx);
        
        self.replot()
    
    def selectNeighbours(self, sel, nodes, depth, maxdepth):
        #print "list: " + str(sel)
        #print "nodes: " + str(nodes)
        sel.update(nodes)
        if depth < maxdepth:
            for i in nodes:
                neighbours = set(self.visualizer.graph.getNeighbours(i))
                #print "neighbours: " + str(neighbours)
                self.selectNeighbours(sel, neighbours - sel, depth+1, maxdepth)
        
    def getSelectedExamples(self):
        if len(self.selection) == 0:
            return None
        
        indeces = self.visualizer.nVertices() * [0]
        
        for v in self.selection:
            indeces[v] = v + 1

        if self.visualizer.graph.items != None:
            return self.visualizer.graph.items.select(indeces)
        else:
            return None

    def getSelectedGraph(self):
        if len(self.selection) == 0:
            return None
        
        graph = orange.GraphAsList(len(self.selection), 0)
        
        for e in range(self.nEdges):
            (key,i,j) = self.edges[e]
            
            if (i in self.selection) and (j in self.selection):
                graph[self.selection.index(i), self.selection.index(j)] = 1
        
        indeces = self.visualizer.nVertices() * [0]
        
        for v in self.selection:
            indeces[v] = v + 1

        if self.visualizer.graph.items != None:
            graph.setattr("items", self.visualizer.graph.items.select(indeces))
            
        return graph

            
    def moveVertex(self, pos):
        # ce ni nic izbrano
        if self.selectedCurve == None:
            return
        #curve = self.curve(self.vertices[self.selectedVertex])  #self.selectedCurve je key
        #newX = self.invTransform(curve.xAxis(), pos.x())
        #newY = self.invTransform(curve.yAxis(), pos.y())

        newX = self.invTransform(2, pos.x())
        newY = self.invTransform(0, pos.y())

        oldX = self.visualizer.coors[self.selectedVertex][0]
        oldY = self.visualizer.coors[self.selectedVertex][1]
        
        self.visualizer.coors[self.selectedVertex][0] = newX
        self.visualizer.coors[self.selectedVertex][1] = newY
        
        (key, neighbours) = self.vertices[self.selectedVertex]
        self.setCurveData(key, [newX], [newY])
        
        edgesCurve = self.curve(self.edgesKey)

        for e in neighbours:
            if (oldX == edgesCurve.xData[e*2]) and (oldY == edgesCurve.yData[e*2]):
                edgesCurve.xData[e*2] = newX
                edgesCurve.yData[e*2] = newY
            elif (oldX == edgesCurve.xData[e*2 + 1]) and (oldY == edgesCurve.yData[e*2 + 1]):
                edgesCurve.xData[e*2 + 1] = newX
                edgesCurve.yData[e*2 + 1] = newY
            
#        for e in range(self.nEdges):
#            (key,i,j) = self.edges[e]
#            edgesCurve = self.curve(self.edgesKey)
#            
#            if i == self.selectedVertex:
#                edgesCurve.xData[key*2] = newX
#                edgesCurve.yData[key*2] = newY
#                #currEdgeObj = self.curve(key)
#                #self.setCurveData(key, [newX, currEdgeObj.x(1)], [newY, currEdgeObj.y(1)])                    
#            elif j == self.selectedVertex:
#                edgesCurve.xData[key*2 + 1] = newX
#                edgesCurve.yData[key*2 + 1] = newY
#                #currEdgeObj = self.curve(key)
#                #self.setCurveData(key, [currEdgeObj.x(0), newX], [currEdgeObj.y(0), newY])
        self.setCurveData(self.edgesKey, edgesCurve.xData, edgesCurve.yData)
    
    def getNeighboursUpTo(self, ndx, dist):
        newNeighbours = neighbours = set([ndx])
        for d in range(dist):
            tNewNeighbours = set()
            for v in newNeighbours:
                tNewNeighbours |= set(self.visualizer.graph.getNeighbours(v))
            newNeighbours = tNewNeighbours - neighbours
            neighbours |= newNeighbours
        return neighbours
     
    def onMouseMoved(self, event):
        if self.mouseCurrentlyPressed and self.state == MOVE_SELECTION:
            if len(self.selection) > 0:
                border = self.vertexSize / 2
                maxx = max(take(self.visualizer.coors[:,0], self.selection))
                maxy = max(take(self.visualizer.coors[:,1], self.selection))
                minx = min(take(self.visualizer.coors[:,0], self.selection))
                miny = min(take(self.visualizer.coors[:,1], self.selection))
                #relativni premik v pikslih
                dx = event.pos().x() - self.GMmouseStartEvent.x()
                dy = event.pos().y() - self.GMmouseStartEvent.y()

                maxx = self.transform(self.xBottom, maxx) + border + dx
                maxy = self.transform(self.yLeft, maxy) + border + dy
                minx = self.transform(self.xBottom, minx) - border + dx
                miny = self.transform(self.yLeft, miny) - border + dy
                maxx = self.invTransform(self.xBottom, maxx)
                maxy = self.invTransform(self.yLeft, maxy)
                minx = self.invTransform(self.xBottom, minx)
                miny = self.invTransform(self.yLeft, miny)

                if maxx >= self.axisScale(self.xBottom).hBound():
                    return
                if minx <= self.axisScale(self.xBottom).lBound():
                    return
                if maxy >= self.axisScale(self.yLeft).hBound():
                    return
                if miny <=self.axisScale(self.yLeft).lBound():
                    return

                for ind in self.selection:
                    self.selectedVertex = ind
                    (key, neighbours) = self.vertices[ind]
                    self.selectedCurve = key

                    vObj = self.curve(self.selectedCurve)
                    tx = self.transform(vObj.xAxis(), vObj.x(0)) + dx
                    ty = self.transform(vObj.yAxis(), vObj.y(0)) + dy
                    
                    tempPoint = QPoint(tx, ty)
                    self.moveVertex(tempPoint)

                self.GMmouseStartEvent.setX(event.pos().x())  #zacetni dogodek postane trenutni
                self.GMmouseStartEvent.setY(event.pos().y())
                self.replot()
        else:
            OWGraph.onMouseMoved(self, event)

        if not self.freezeNeighbours and self.tooltipNeighbours:
            px = self.invTransform(2, event.x())
            py = self.invTransform(0, event.y())   
            ndx, mind = self.visualizer.closestVertex(px, py)
            if ndx != -1 and mind < 50:
                toMark = set(self.getNeighboursUpTo(ndx, self.tooltipNeighbours))
                toMark -= set(self.selection)
                self.setMarkedNodes(toMark)
            else:
                self.setMarkedNodes([])
        
        
    def markSelectionNeighbours(self):
        if not self.freezeNeighbours and self.selectionNeighbours:
            toMark = set()
            for ndx in self.selection:
                toMark |= self.getNeighboursUpTo(ndx, self.selectionNeighbours)
            toMark -= set(self.selection)
            self.setMarkedNodes(toMark)
        
    def setMarkedNodes(self, marked):
        if not isinstance(marked, set):
            marked = set(marked)
        if marked == self.markedNodes:
            return

        redColor = self.markWithRed and Qt.red
        markedSize = self.markWithRed and 9 or 6
        # mark
        for m in marked - self.markedNodes:
            (key, neighbours) = self.vertices[m]
            newSymbol = QwtSymbol(QwtSymbol.Ellipse, QBrush(redColor or self.nodeColor[m]), QPen(self.nodeColor[m]), QSize(markedSize, markedSize))
            self.setCurveSymbol(key, newSymbol)
#            self.curve(key).setBrush(QBrush(redColor or self.nodeColor[m]))
        # unmark
        for m in self.markedNodes - marked - set(self.selection):
            (key, neighbours) = self.vertices[m]
            newSymbol = QwtSymbol(QwtSymbol.Ellipse, QBrush(), QPen(self.nodeColor[m]), QSize(6, 6))
            self.setCurveSymbol(key, newSymbol)
    
        self.markedNodes = marked
        self.master.nMarked = len(self.markedNodes)
        self.replot()
        
        
    def activateMoveSelection(self):
        self.state = MOVE_SELECTION


    def onMousePressed(self, event):
        if self.state == MOVE_SELECTION:
            self.mouseCurrentlyPressed = 1
            if self.isPointSelected(self.invTransform(self.xBottom, event.pos().x()), self.invTransform(self.yLeft, event.pos().y())) and self.selection!=[]:
                self.GMmouseStartEvent = QPoint(event.pos().x(), event.pos().y())
            else:
                #pritisk na gumb izven izbranega podrocja ali pa ni izbranega podrocja
                self.selectVertex(event.pos())
                self.GMmouseStartEvent = QPoint(event.pos().x(), event.pos().y())

            #self.removeAllSelections()
            #self.selectionCurveKeyList=[]
        else:
            OWGraph.onMousePressed(self, event)     


    def onMouseReleased(self, event):  
        if self.state == MOVE_SELECTION:
            self.mouseCurrentlyPressed = 0
            
            self.selectedCurve= None
            self.selectedVertex=None
            self.moveGroup=False
            #self.selectedVertices=[]
            self.GMmouseStartEvent=None
            
        elif self.state == SELECT_RECTANGLE:
            self.selectVertices()
            OWGraph.onMouseReleased(self, event)
            self.removeAllSelections()

        elif self.state == SELECT_POLYGON:
                OWGraph.onMouseReleased(self, event)
                if self.tempSelectionCurve == None:   #ce je OWVisGraph zakljucil poligon
                    self.selectVertices()
        else:
            OWGraph.onMouseReleased(self, event)
        
    
    def selectVertices(self):
        #print "selecting vertices.."
        for vertexKey in self.indexPairs.keys():
            vObj = self.curve(vertexKey)
            
            if self.isPointSelected(vObj.x(0), vObj.y(0)):
                self.addSelection(self.indexPairs[vertexKey], False)
                
        self.replot()
                
    def selectVertex(self, pos):
        #print "select vertex"
        #key, dist, xVal, yVal, index = self.closestCurve(pos.x(), pos.y())
        #curve = self.curve(key)
        min = 1000000
        ndx = -1
        #print "x: " + str(pos.x()) + " y: " + str(pos.y()) 
        px = self.invTransform(2, pos.x())
        py = self.invTransform(0, pos.y())   
        #print "xAxis: " + str(curve.xAxis()) + " yAxis: " + str(curve.yAxis())
        #print "px: " + str(px) + " py: " + str(py)
        ndx, min = self.visualizer.closestVertex(px, py)
        #print "ndx: " + str(ndx) + " min: " + str(min)
        if min < 50 and ndx != -1:
            self.addSelection(ndx) # do not replot if we replot later anyway
        else:
            self.removeSelection()
            
                
    def dist(self, s1, s2):
        return math.sqrt((s1[0]-s2[0])**2 + (s1[1]-s2[1])**2)
    
    def updateData(self):
        #print "OWGraphDrawerCanvas/updateData..."
        self.removeDrawingCurves(removeLegendItems = 0)
        self.removeMarkers()
        self.tips.removeAll()
  
        edgesCount = 0
        
        fillColor = Qt.blue#self.discPalette[classValueIndices[self.rawdata[i].getclass().value], 255*insideData[j]]
        edgeColor = Qt.blue#self.discPalette[classValueIndices[self.rawdata[i].getclass().value]]
        emptyFill = Qt.white
        
        for r in self.circles:
            #print "r: " + str(r)
            step = 2 * pi / 64;
            fi = 0
            x = []
            y = []
            for i in range(65):
                x.append(r * cos(fi) + 5000)
                y.append(r * sin(fi) + 5000)
                fi += step
                
            self.addCurve("radius", fillColor, Qt.green, 1, style = QwtCurve.Lines, xData = x, yData = y, showFilledSymbols = False)
        
        xData = []
        yData = []
        # draw edges
        for e in range(self.nEdges):
            (key,i,j) = self.edges[e]
            
            if i in self.hiddenNodes or j in self.hiddenNodes:
                continue  
            
            x1 = self.visualizer.coors[i][0]
            x2 = self.visualizer.coors[j][0]
            y1 = self.visualizer.coors[i][1]
            y2 = self.visualizer.coors[j][1]

            #key = self.addCurve(str(e), fillColor, edgeColor, 0, style = QwtCurve.Lines, xData = [x1, x2], yData = [y1, y2])
            #self.edges[e] = (key,i,j)
            xData.append(x1)
            xData.append(x2)
            yData.append(y1)
            yData.append(y2)
            # append edges to vertex descriptions
            self.edges[e] = (edgesCount,i,j)
            (key, neighbours) = self.vertices[i]
            if edgesCount not in neighbours:
                neighbours.append(edgesCount)
            self.vertices[i] = (key, neighbours)
            (key, neighbours) = self.vertices[j]
            if edgesCount not in neighbours:
                neighbours.append(edgesCount)
            self.vertices[j] = (key, neighbours)
            
            edgesCount += 1
        
        edgesCurveObject = UnconnectedLinesCurve(self, QPen(QColor(192,192,192)), xData, yData)
        edgesCurveObject.xData = xData
        edgesCurveObject.yData = yData
        self.edgesKey = self.insertCurve(edgesCurveObject)
        
        selectionX = []
        selectionY = []
        self.nodeColor = []

        # draw vertices
        for v in range(self.nVertices):
            if v in self.hiddenNodes:
                continue
            
            x1 = self.visualizer.coors[v][0]
            y1 = self.visualizer.coors[v][1]
            
            if self.colorIndex != -1:
                if self.visualizer.graph.items.domain[self.colorIndex].varType == orange.VarTypes.Continuous:
                    newColor = self.contPalette[self.noJitteringScaledData[self.colorIndex][v]]
                elif self.visualizer.graph.items.domain[self.colorIndex].varType == orange.VarTypes.Discrete:
                    newColor = self.discPalette[self.colorIndices[self.visualizer.graph.items[v][self.colorIndex].value]]
            else: 
                newColor = Qt.red #QColor(0,0,0)
            
            # This works only if there are no hidden vertices!    
            self.nodeColor.append(fillColor)
            
            selectionX.append(x1)
            selectionY.append(y1)
            key = self.addCurve(str(v), fillColor, edgeColor, 6, xData = [x1], yData = [y1], showFilledSymbols = False)
            
            if v in self.selection:
                #print "sel: " + str(v)
#                newSymbol = QwtSymbol(QwtSymbol.Ellipse, QBrush(fillColor), QPen(Qt.yellow, 3), QSize(10, 10))
                newSymbol = QwtSymbol(QwtSymbol.Ellipse, QBrush(), QPen(Qt.yellow, 3), QSize(10, 10))
                self.setCurveSymbol(key, newSymbol)
            
            (tmp, neighbours) = self.vertices[v]
            self.vertices[v] = (key, neighbours)
            self.indexPairs[key] = v
        
        #self.addCurve('vertices', fillColor, edgeColor, 6, xData=selectionX, yData=selectionY)
        
            
#        selectionX = []
#        selectionY = []
#        for v in self.selection:    
#            x1 = self.visualizer.coors[v][0]
#            y1 = self.visualizer.coors[v][1]
#            
#            selectionX.append(x1)
#            selectionY.append(y1)
#            
#        selectionCurve = self.insertCurve('selections')
#        selectionBrush = QBrush(QBrush.NoBrush)
#        selectionPen = QPen(Qt.yellow, 3)
#        selectionSymbol = QwtSymbol(QwtSymbol.Ellipse, selectionBrush, selectionPen, QSize(10, 10))
#        self.setCurveSymbol(selectionCurve, selectionSymbol)
#        self.setCurveStyle(selectionCurve, QwtCurve.NoCurve)        
#        self.setCurveData(selectionCurve, selectionX, selectionY)
        
        
        # draw markers
        if len(self.labelText) > 0:
            for v in range(self.nVertices):
                if v in self.hiddenNodes:
                    continue
                
                x1 = self.visualizer.coors[v][0]
                y1 = self.visualizer.coors[v][1]
                lbl = ""
                values = self.visualizer.graph.items[v]
                lbl = " ".join([str(values[ndx]) for ndx in self.labelText])
                if lbl:
                    mkey = self.insertMarker(lbl)
                    self.marker(mkey).setXValue(float(x1))
                    self.marker(mkey).setYValue(float(y1))
                    self.marker(mkey).setLabelAlignment(Qt.AlignCenter + Qt.AlignBottom)            
        
        # add ToolTips
        self.tooltipData = []
        self.tips.removeAll()
        if len(self.tooltipText) > 0:
            for v in range(self.nVertices):
                if v in self.hiddenNodes:
                    continue
                
                x1 = self.visualizer.coors[v][0]
                y1 = self.visualizer.coors[v][1]
                lbl = ""
                for ndx in self.tooltipText:
                    values = self.visualizer.graph.items[v]
                    lbl = lbl + str(values[ndx]) + "\n"
        
                if lbl != '':
                    lbl = lbl[:-1]
                    self.tips.addToolTip(x1, y1, lbl)
        #print "done."
            
    def setVertexColor(self, attribute):
        if attribute == "(one color)":
            self.colorIndex = -1
        else:
            i = 0
            for var in self.visualizer.graph.items.domain.variables:
                if var.name == attribute:
                    self.colorIndex = i
                    if var.varType == orange.VarTypes.Discrete: 
                        self.colorIndices = getVariableValueIndices(self.visualizer.graph.items, self.colorIndex)
                i += 1
        pass
        
    def setLabelText(self, attributes):
        self.labelText = []
        if isinstance(self.visualizer.graph.items, orange.ExampleTable):
            data = self.visualizer.graph.items
            for att in attributes:
                for i in range(len(data.domain)):
                    if data.domain[i].name == att:
                        self.labelText.append(i)
                        
                if self.visualizer.graph.items.domain.hasmeta(att):
                        self.labelText.append(self.visualizer.graph.items.domain.metaid(att))
    
    def setTooltipText(self, attributes):
        self.tooltipText = []
        if isinstance(self.visualizer.graph.items, orange.ExampleTable):
            data = self.visualizer.graph.items
            for att in attributes:
                for i in range(len(data.domain)):
                    if data.domain[i].name == att:
                        self.tooltipText.append(i)
                        
                if self.visualizer.graph.items.domain.hasmeta(att):
                        self.tooltipText.append(self.visualizer.graph.items.domain.metaid(att))
        
                            
    def edgesContainsEdge(self, i, j):
        for e in range(self.nEdges):
            (key,iTmp,jTmp) = self.edges[e]
            
            if (iTmp == i and jTmp == j) or (iTmp == j and jTmp == i):
                return True
        return False
        
    def addVisualizer(self, visualizer):
        self.visualizer = visualizer
        self.clear()
        
        self.nVertices = visualizer.graph.nVertices
        self.nEdges = 0
        self.vertexDegree = []
        
        #dodajanje vozlisc
        #print "OWGraphDrawerCanvas/addVisualizer: adding vertices..."
        self.vertices = {}
        for v in range(0, self.nVertices):
            self.vertices[v] = (None, [])
        #print "done."
        
        #dodajanje povezav
        #print "OWGraphDrawerCanvas/addVisualizer: adding edges..."
        self.edges = {}
        self.nEdges = 0
        
        for (i,j) in visualizer.graph.getEdges():
            self.edges[self.nEdges] = (None,i,j)
            
#            (key, neighbours) = self.vertices[i]
#            neighbours.append(j)
#            self.vertices[i] = (key, neighbours)
#            (key, neighbours) = self.vertices[j]
#            neighbours.append(i)
#            self.vertices[j] = (key, neighbours)
            
            self.nEdges += 1
        
        #print "done."
        
    def resetValues(self):
        self.vertices={}
        self.edges={}
        self.indexPairs={}
        #self.xCoors = None
        #self.yCoors = None
        self.n = 0
    
    def updateCanvas(self): #, xCoors, yCoors):
        #self.xCoors = xCoors
        #self.yCoors = yCoors
        
        self.setAxisAutoScaled()

        #self.drawGraph(self.nVertices);
        self.updateData()
        self.replot()
        
        # preprecimo avtomatsko primikanje plota (kadar smo odmaknili neko skrajno tocko)
        self.setAxisFixedScale()    
    
    def setAxisAutoScaled(self):
        self.setAxisAutoScale(self.xBottom)
        self.setAxisAutoScale(self.yLeft)

    def setAxisFixedScale(self):
        self.setAxisScale(self.xBottom, self.axisScale(self.xBottom).lBound(), self.axisScale(self.xBottom).hBound())
        self.setAxisScale(self.yLeft, self.axisScale(self.yLeft).lBound(), self.axisScale(self.yLeft).hBound())
        