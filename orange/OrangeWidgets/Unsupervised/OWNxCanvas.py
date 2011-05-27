CIRCLE = 0
SQUARE = 1
ROUND_RECT = 2

NOTHING = 0
ZOOMING = 1
SELECT_RECTANGLE = 2
SELECT_POLYGON = 3
MOVE_SELECTION = 100

from OWGraph import *
from numpy import *
from orngNetwork import Network
from orngScaleScatterPlotData import *

class NetworkVertex():
    def __init__(self, index=-1):
        self.index = index
        self.marked = False
        self.show = True
        self.highlight = False
        self.selected = False
        self.label = []
        self.tooltip = []
        self.uuid = None
        
        self.image = None
        self.pen = QPen(Qt.blue, 1)
        self.pen.setJoinStyle(Qt.RoundJoin)
        self.nocolor = Qt.white
        self.color = Qt.blue
        self.size = 5
        self.style = 1
    
class NetworkEdge():
    def __init__(self, u=None, v=None, weight=0, arrowu=0, arrowv=0, 
                 links_index=None, label=[]):
        self.u = u
        self.v = v
        self.links_index = links_index
        self.arrowu = arrowu
        self.arrowv = arrowv
        self.weight = weight
        self.label = label

        self.pen = QPen(Qt.lightGray, 1)
        self.pen.setCapStyle(Qt.RoundCap)

class NetworkCurve(QwtPlotCurve):
  def __init__(self, parent, pen=QPen(Qt.black), xData=None, yData=None):
      QwtPlotCurve.__init__(self, "Network Curve")

      self.coors = None
      self.vertices = []
      self.edges = []
      self.setItemAttribute(QwtPlotItem.Legend, 0)
      self.showEdgeLabels = 0

  def moveSelectedVertices(self, dx, dy):
    selected = self.getSelectedVertices()
    
    self.coors[0][selected] = self.coors[0][selected] + dx
    self.coors[1][selected] = self.coors[1][selected] + dy
      
    self.setData(self.coors[0], self.coors[1])
    return selected
  
  def setVertexColor(self, v, color):
      pen = self.vertices[v].pen
      self.vertices[v].color = color
      self.vertices[v].pen = QPen(color, pen.width())
      
  def setEdgeColor(self, index, color, nocolor=0):
      pen = self.edges[index].pen
      if nocolor:
        color.setAlpha(0)
      self.edges[index].pen = QPen(color, pen.width())
      self.edges[index].pen.setCapStyle(Qt.RoundCap)
  
  def getSelectedVertices(self):
    return [vertex.index for vertex in self.vertices if vertex.selected]

  def getUnselectedVertices(self):
    return [vertex.index for vertex in self.vertices if not vertex.selected]

  def getMarkedVertices(self):
    return [vertex.index for vertex in self.vertices if vertex.marked]
  
  def setMarkedVertices(self, vertices):
    for vertex in self.vertices:
      if vertex.index in vertices:
        vertex.marked = True
      else:
        vertex.marked = False
        
  def markToSel(self):
    for vertex in self.vertices:
      if vertex.marked == True:
          vertex.selected = True
          
  def selToMark(self):
    for vertex in self.vertices:
      if vertex.selected == True:
          vertex.selected = False
          vertex.marked = True
  
  def unMark(self):
    for vertex in self.vertices:
      vertex.marked = False
      
  def unSelect(self):
    for vertex in self.vertices:
        vertex.selected = False
        
  def setHiddenVertices(self, nodes):
    for vertex in self.vertices:
      if vertex.index in nodes:
        vertex.show = False
      else:
        vertex.show = True
      
  def hideSelectedVertices(self):
    for vertex in self.vertices:
      if vertex.selected:
        vertex.show = False
  
  def hideUnSelectedVertices(self):
    for vertex in self.vertices:
      if not vertex.selected:
        vertex.show = False
    
  def showAllVertices(self):
    for vertex in self.vertices:
      vertex.show = True
    
  def changed(self):
      self.itemChanged()

  def draw(self, painter, xMap, yMap, rect):
    for edge in self.edges:
      if edge.u.show and edge.v.show:
        painter.setPen(edge.pen)

        px1 = xMap.transform(self.coors[0][edge.u.index])   #ali pa tudi self.x1, itd
        py1 = yMap.transform(self.coors[1][edge.u.index])
        px2 = xMap.transform(self.coors[0][edge.v.index])
        py2 = yMap.transform(self.coors[1][edge.v.index])
        
        painter.drawLine(px1, py1, px2, py2)
        
        d = 12
        #painter.setPen(QPen(Qt.lightGray, 1))
        painter.setBrush(Qt.lightGray)
        if edge.arrowu:
            x = px1 - px2
            y = py1 - py2
            
            fi = math.atan2(y, x) * 180 * 16 / math.pi 

            if not fi is None:
                # (180*16) - fi - (20*16), (40*16)
                painter.drawPie(px1 - d, py1 - d, 2 * d, 2 * d, 2560 - fi, 640)
                
        if edge.arrowv:
            x = px1 - px2
            y = py1 - py2
            
            fi = math.atan2(y, x) * 180 * 16 / math.pi 
            if not fi is None:
                # (180*16) - fi - (20*16), (40*16)
                painter.drawPie(px1 - d, py1 - d, 2 * d, 2 * d, 2560 - fi, 640)
                
        if self.showEdgeLabels and len(edge.label) > 0:
            lbl = ', '.join(edge.label)
            x = (px1 + px2) / 2
            y = (py1 + py2) / 2
            
            th = painter.fontMetrics().height()
            tw = painter.fontMetrics().width(lbl)
            r = QRect(x - tw / 2, y - th / 2, tw, th)
            painter.fillRect(r, QBrush(Qt.white))
            painter.drawText(r, Qt.AlignHCenter + Qt.AlignVCenter, lbl)
    
    for vertex in self.vertices:
      if vertex.show:
        pX = xMap.transform(self.coors[0][vertex.index])   #dobimo koordinati v pikslih (tipa integer)
        pY = yMap.transform(self.coors[1][vertex.index])   #ki se stejeta od zgornjega levega kota canvasa
        if vertex.selected:    
          painter.setPen(QPen(Qt.yellow, 3))
          painter.setBrush(vertex.color)
          painter.drawEllipse(pX - (vertex.size + 4) / 2, pY - (vertex.size + 4) / 2, vertex.size + 4, vertex.size + 4)
        elif vertex.marked:
          painter.setPen(vertex.pen)
          painter.setBrush(vertex.color)
          painter.drawEllipse(pX - vertex.size / 2, pY - vertex.size / 2, vertex.size, vertex.size)
        else:
          painter.setPen(vertex.pen)
          painter.setBrush(vertex.nocolor)
          #print pX - vertex.size / 2, pY - vertex.size / 2, vertex.size
          painter.drawEllipse(pX - vertex.size / 2, pY - vertex.size / 2, vertex.size, vertex.size)
        
class OWNxCanvas(OWGraph):
    def __init__(self, master, parent=None, name="None"):
        OWGraph.__init__(self, parent, name)
        self.master = master
        self.parent = parent
        self.labelText = []
        self.tooltipText = []
        self.vertices_old = {}         # distionary of nodes (orngIndex: vertex_objekt)
        self.edges_old = {}            # distionary of edges (curveKey: edge_objekt)
        self.vertices = []
        self.edges = []
        self.indexPairs = {}       # distionary of type CurveKey: orngIndex   (for nodes)
        #self.selection = []        # list of selected nodes (indices)
        self.markerKeys = {}       # dictionary of type NodeNdx : markerCurveKey
        self.tooltipKeys = {}      # dictionary of type NodeNdx : tooltipCurveKey
        self.graph = None
        self.layout = None
        self.vertexDegree = []     # seznam vozlisc oblike (vozlisce, stevilo povezav), sortiran po stevilu povezav
        self.edgesKey = -1
        #self.vertexSize = 6
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
        self.labelsOnMarkedOnly = 0
        self.enableWheelZoom = 1
        self.optimizing = 0
        self.stopOptimizing = 0
        self.insideview = 0
        self.insideviewNeighbours = 2
        self.enableGridXB(False)
        self.enableGridYL(False)
        self.renderAntialiased = 1
        self.sendMarkedNodes = None
        self.showEdgeLabels = 0
        self.showDistances = 0
        self.showMissingValues = 0
        
        self.showWeights = 0
        self.showIndexes = 0
        self.minEdgeWeight = sys.maxint
        self.maxEdgeWeight = 0
        self.maxEdgeSize = 1
        
        self.maxVertexSize = 5
        self.minVertexSize = 5
        self.invertEdgeSize = 0
        self.showComponentAttribute = None
        self.forceVectors = None
        self.appendToSelection = 1
        self.fontSize = 12
             
        self.setAxisAutoScale(self.xBottom)
        self.setAxisAutoScale(self.yLeft)
        
        self.networkCurve = NetworkCurve(self)
        self.callbackMoveVertex = None
        self.callbackSelectVertex = None
        self.minComponentEdgeWidth = 0
        self.maxComponentEdgeWidth = 0
        self.vertexDistance = None
        self.controlPressed = False
        self.altPressed = False
        
        self.setFocusPolicy(Qt.StrongFocus)
        
    def getSelection(self):
      return self.networkCurve.getSelectedVertices()
    
    def getMarkedVertices(self):
      return self.networkCurve.getMarkedVertices()
        
    def getVertexSize(self, index):
        return 6
        
    def setHiddenVertices(self, nodes):
        self.networkCurve.setHiddenVertices(nodes)
    
    def hideSelectedVertices(self):
      self.networkCurve.hideSelectedVertices()
      self.drawPlotItems()
      
    def hideUnSelectedVertices(self):
      self.networkCurve.hideUnSelectedVertices()
      self.drawPlotItems()
      
    def showAllVertices(self):
      self.networkCurve.showAllVertices()
      self.drawPlotItems()
      
    def optimize(self, frSteps):
        qApp.processEvents()
        tolerance = 5
        initTemp = 100
        breakpoints = 20
        k = int(frSteps / breakpoints)
        o = frSteps % breakpoints
        iteration = 0
        coolFactor = exp(log(10.0 / 10000.0) / frSteps)
        #print coolFactor
        if k > 0:
            while iteration < breakpoints:
                initTemp = self.layout.fruchtermanReingold(k, initTemp, coolFactor, self.hiddenNodes)
                iteration += 1
                qApp.processEvents()
                self.updateCanvas()
    
            initTemp = self.layout.fruchtermanReingold(o, initTemp, coolFactor, self.hiddenNodes)
            qApp.processEvents()
            self.updateCanvas()
        else:
            while iteration < o:
                initTemp = self.layout.fruchtermanReingold(1, initTemp, coolFactor, self.hiddenNodes)
                iteration += 1
                qApp.processEvents()
                self.updateCanvas()
                
    def markedToSelection(self):
        self.networkCurve.markToSel()
        self.drawPlotItems()
        
    def selectionToMarked(self):
        self.networkCurve.selToMark()
        self.drawPlotItems()
        
        if self.sendMarkedNodes != None:
            self.sendMarkedNodes(self.networkCurve.getMarkedVertices())
        
    def removeSelection(self, replot=True):
        self.networkCurve.unSelect()
        
        if replot:
          self.replot()
    
    def selectNeighbours(self, sel, nodes, depth, maxdepth):
        #print "list: " + str(sel)
        #print "nodes: " + str(nodes)
        sel.update(nodes)
        if depth < maxdepth:
            for i in nodes:
                neighbours = set(self.graph.neighbors(i))
                #print "neighbours: " + str(neighbours)
                self.selectNeighbours(sel, neighbours - sel, depth + 1, maxdepth)
        
    def getSelectedExamples(self):
        selection = self.networkCurve.getSelectedVertices()
        
        if len(selection) == 0:
            return None
        
        if self.graph.items() is not None:
            return self.graph.items().getitems(selection)
        else:
            return None
        
    def getUnselectedExamples(self):
        unselection = self.networkCurve.getUnselectedVertices()
        
        if len(unselection) == 0:
            return None
        
        if self.graph.items() is not None:
            return self.graph.items().getitems(unselection)
        else:
            return None
    
    def getSelectedGraph(self):
      selection = self.networkCurve.getSelectedVertices()
      
      if len(selection) == 0:
          return None
      
      subgraph = self.graph.subgraph(selection)
      subnet = Network(subgraph)
      return subnet
    
    def getSelectedVertices(self):
      return self.networkCurve.getSelectedVertices()
    
    def getNeighboursUpTo(self, ndx, dist):
        newNeighbours = neighbours = set([ndx])
        for d in range(dist):
            tNewNeighbours = set()
            for v in newNeighbours:
                tNewNeighbours |= set(self.graph.neighbors(v))
            newNeighbours = tNewNeighbours - neighbours
            neighbours |= newNeighbours
        return neighbours
     
    def markSelectionNeighbours(self):
        if not self.freezeNeighbours and self.selectionNeighbours:
            toMark = set()
            for ndx in self.networkCurve.getSelectedVertices():
                toMark |= self.getNeighboursUpTo(ndx, self.selectionNeighbours)
            
            self.networkCurve.setMarkedVertices(toMark)
            self.drawPlotItems()
                
        elif not self.freezeNeighbours and self.selectionNeighbours == 0:
            self.networkCurve.setMarkedVertices(self.networkCurve.getSelectedVertices())
            self.drawPlotItems()
            
        if self.sendMarkedNodes != None:
            self.sendMarkedNodes(self.networkCurve.getMarkedVertices())
                
    def unMark(self):
      self.networkCurve.unMark()
      self.drawPlotItems(replot=0)
      
      if self.sendMarkedNodes != None:
            self.sendMarkedNodes([])
            
    def setMarkedVertices(self, vertices):
      self.networkCurve.setMarkedVertices(vertices)
      self.drawPlotItems(replot=0)
      
      if self.sendMarkedNodes != None:
            self.sendMarkedNodes(self.networkCurve.getMarkedVertices())
        
    def activateMoveSelection(self):
        self.state = MOVE_SELECTION
    
    def mouseMoveEvent(self, event):
        if not self.graph:
          return
          
        if self.mouseCurrentlyPressed and self.state == MOVE_SELECTION and self.GMmouseMoveEvent != None:
            newX = self.invTransform(2, event.pos().x())
            newY = self.invTransform(0, event.pos().y())
            
            dx = newX - self.invTransform(2, self.GMmouseMoveEvent.x())
            dy = newY - self.invTransform(0, self.GMmouseMoveEvent.y())
            movedVertices = self.networkCurve.moveSelectedVertices(dx, dy)
            
            self.GMmouseMoveEvent.setX(event.pos().x())  #zacetni dogodek postane trenutni
            self.GMmouseMoveEvent.setY(event.pos().y())
            
            self.drawPlotItems(replot=1, vertices=movedVertices)
            if self.callbackMoveVertex:
                self.callbackMoveVertex()
        else:
            OWGraph.mouseMoveEvent(self, event)
                
        if not self.freezeNeighbours and self.tooltipNeighbours:
            px = self.invTransform(2, event.x())
            py = self.invTransform(0, event.y())   
            ndx, mind = self.layout.closest_vertex(px, py)
            dX = self.transform(QwtPlot.xBottom, self.layout.coors[0][ndx]) - event.x()
            dY = self.transform(QwtPlot.yLeft,   self.layout.coors[1][ndx]) - event.y()
            # transform to pixel distance
            distance = math.sqrt(dX**2 + dY**2) 
              
            if ndx != -1 and distance <= self.vertices[ndx].size / 2:
                toMark = set(self.getNeighboursUpTo(ndx, self.tooltipNeighbours))
                self.networkCurve.setMarkedVertices(toMark)
                self.drawPlotItems()
                
                if self.sendMarkedNodes != None:
                    self.sendMarkedNodes(self.networkCurve.getMarkedVertices())
            else:
                self.networkCurve.unMark()
                self.drawPlotItems()
                
                if self.sendMarkedNodes != None:
                    self.sendMarkedNodes([])
        
        if self.showDistances:
            selection = self.networkCurve.getSelectedVertices()
            if len(selection) > 0:
                px = self.invTransform(2, event.x())
                py = self.invTransform(0, event.y())  
                 
                v, mind = self.layout.closest_vertex(px, py)
                dX = self.transform(QwtPlot.xBottom, self.layout.coors[0][v]) - event.x()
                dY = self.transform(QwtPlot.yLeft,   self.layout.coors[1][v]) - event.y()
                # transform to pixel distance
                distance = math.sqrt(dX**2 + dY**2)               
                if v != -1 and distance <= self.vertices[v].size / 2:
                    if self.graph.vertexDistance == None:
                        dst = 'vertex distance signal not set'
                    else:
                        dst = 0
                        for u in selection:
                            dst += self.graph.vertexDistance[u, v]
                        dst = dst / len(selection)
                        
                    self.showTip(event.pos().x(), event.pos().y(), str(dst))
                    self.replot()
    
    def mousePressEvent(self, event):
      if self.graph is None:
          return
          
      #self.grabKeyboard()
      self.mouseSelectedVertex = 0
      self.GMmouseMoveEvent = None
      
      if self.state == MOVE_SELECTION:
        self.mouseCurrentlyPressed = 1
        #if self.isPointSelected(self.invTransform(self.xBottom, event.pos().x()), self.invTransform(self.yLeft, event.pos().y())) and self.selection != []:
        #  self.GMmouseStartEvent = QPoint(event.pos().x(), event.pos().y())
        #else:
          # button pressed outside selected area or there is no area
        self.selectVertex(event.pos())
        self.GMmouseStartEvent = QPoint(event.pos().x(), event.pos().y())
        self.replot()
      elif self.state == SELECT_RECTANGLE:
          self.GMmouseStartEvent = QPoint(event.pos().x(), event.pos().y())
          
          if self.clickedSelectedOnVertex(event.pos()):
              self.mouseSelectedVertex = 1
              self.mouseCurrentlyPressed = 1
              self.state = MOVE_SELECTION
              self.GMmouseMoveEvent = QPoint(event.pos().x(), event.pos().y())
          elif self.clickedOnVertex(event.pos()):
              self.mouseSelectedVertex = 1
              self.mouseCurrentlyPressed = 1
          else:
              OWGraph.mousePressEvent(self, event)  
      else:
          OWGraph.mousePressEvent(self, event)     
    
    def mouseReleaseEvent(self, event):  
        if self.graph is None:
          return
          
        #self.releaseKeyboard()
        if self.state == MOVE_SELECTION:
            self.state = SELECT_RECTANGLE
            self.mouseCurrentlyPressed = 0
            
            self.moveGroup = False
            #self.GMmouseStartEvent=None
            
        if self.state == SELECT_RECTANGLE:
            x1 = self.invTransform(2, self.GMmouseStartEvent.x())
            y1 = self.invTransform(0, self.GMmouseStartEvent.y())
            
            x2 = self.invTransform(2, event.pos().x())
            y2 = self.invTransform(0, event.pos().y())
            
            
            if self.mouseSelectedVertex == 1 and x1 == x2 and y1 == y2 and self.selectVertex(self.GMmouseStartEvent):
                QwtPlot.mouseReleaseEvent(self, event)
            elif self.mouseSelectedVertex == 0:
                 
                selection = self.layout.get_vertices_in_rect(x1, y1, x2, y2)
    
                def selectVertex(ndx):
                    if self.vertices[ndx].show:
                        self.vertices[ndx].selected = True
                        
                map(selectVertex, selection)
                
                if len(selection) == 0 and x1 == x2 and y1 == y2:
                    self.removeSelection()
                    self.unMark()
            
                self.markSelectionNeighbours()
                OWGraph.mouseReleaseEvent(self, event)
                self.removeAllSelections()
    
        elif self.state == SELECT_POLYGON:
                OWGraph.mouseReleaseEvent(self, event)
                if self.tempSelectionCurve == None:   #if OWVisGraph closed polygon
                    self.selectVertices()
        else:
            OWGraph.mouseReleaseEvent(self, event)
            
        self.mouseCurrentlyPressed = 0
        self.moveGroup = False
            
        if self.callbackSelectVertex != None:
            self.callbackSelectVertex()
    
    def keyPressEvent(self, e):
        if self.graph is None:
          return
          
        if e.key() == 87 or e.key() == 81:
            selection = [v.index for v in self.vertices if v.selected]
            if len(selection) > 0:
                phi = [math.pi / -180 if e.key() == 87 else math.pi / 180]
                self.layout.rotate_vertices([selection], phi)
                self.drawPlotItems(replot=1)
            
        if e.key() == Qt.Key_Control:
            self.controlPressed = True
        
        elif e.key() == Qt.Key_Alt:
            self.altPressed = True
            
        if e.text() == "f":
            self.graph.freezeNeighbours = not self.graph.freezeNeighbours
        
        OWGraph.keyPressEvent(self, e)
            
    def keyReleaseEvent(self, e):
        if e.key() == Qt.Key_Control:
            self.controlPressed = False
        
        elif e.key() == Qt.Key_Alt:
            self.altPressed = False
        
        OWGraph.keyReleaseEvent(self, e)
        
    
    def clickedSelectedOnVertex(self, pos):
        min = 1000000
        ndx = -1
    
        px = self.invTransform(2, pos.x())
        py = self.invTransform(0, pos.y())   
    
        ndx, min = self.layout.closest_vertex(px, py)
        dX = self.transform(QwtPlot.xBottom, self.layout.coors[0][ndx]) - pos.x()
        dY = self.transform(QwtPlot.yLeft,   self.layout.coors[1][ndx]) - pos.y()
        # transform to pixel distance
        distance = math.sqrt(dX**2 + dY**2)
        if ndx != -1 and distance <= self.vertices[ndx].size / 2:
            return self.vertices[ndx].selected
        else:
            return False
        
    def clickedOnVertex(self, pos):
        min = 1000000
        ndx = -1
    
        px = self.invTransform(2, pos.x())
        py = self.invTransform(0, pos.y())   
    
        ndx, min = self.layout.closest_vertex(px, py)
        dX = self.transform(QwtPlot.xBottom, self.layout.coors[0][ndx]) - pos.x()
        dY = self.transform(QwtPlot.yLeft,   self.layout.coors[1][ndx]) - pos.y()
        # transform to pixel distance
        distance = math.sqrt(dX**2 + dY**2)
        if ndx != -1 and distance <= self.vertices[ndx].size / 2:
            return True
        else:
            return False
                
    def selectVertex(self, pos):
        min = 1000000
        ndx = -1
    
        px = self.invTransform(2, pos.x())
        py = self.invTransform(0, pos.y())   
    
        ndx, min = self.layout.closest_vertex(px, py)
        
        dX = self.transform(QwtPlot.xBottom, self.layout.coors[0][ndx]) - pos.x()
        dY = self.transform(QwtPlot.yLeft,   self.layout.coors[1][ndx]) - pos.y()
        # transform to pixel distance
        distance = math.sqrt(dX**2 + dY**2)
        if ndx != -1 and distance <= self.vertices[ndx].size / 2:
            if not self.appendToSelection and not self.controlPressed:
                self.removeSelection()
                      
            if self.insideview:
                self.networkCurve.unSelect()
                self.vertices[ndx].selected = not self.vertices[ndx].selected
                self.optimize(100)
                
                self.markSelectionNeighbours()
            else:
                self.vertices[ndx].selected = not self.vertices[ndx].selected
                self.markSelectionNeighbours()
            
            return True  
        else:
            return False
            self.removeSelection()
            self.unMark()
    
    def updateData(self):
        if self.graph is None:
            return
        
        self.removeDrawingCurves(removeLegendItems=0)
        self.tips.removeAll()
        
        if self.vertexDistance and self.minComponentEdgeWidth > 0 and self.maxComponentEdgeWidth > 0:          
            components = Orange.network.nx.algorithms.components.connected_components(self.graph)
            matrix = self.vertexDistance.avgLinkage(components)
            
            edges = set()
            for u in range(matrix.dim):
                neighbours = matrix.getKNN(u, 2)
                for v in neighbours:
                    if v < u:
                        edges.add((v, u))
                    else:
                        edges.add((u, v))
            edges = list(edges)
    # show 2n strongest edges
    #          vals = matrix.getValues()
    #          vals = zip(vals, range(len(vals)))
    #          count = 0
    #          add = 0
    #          for i in range(matrix.dim):
    #              add += i + 1
    #              for j in range(i+1, matrix.dim):
    #                  v, ind = vals[count]
    #                  ind += add
    #                  vals[count] = (v, ind)
    #                  count += 1
    #          vals.sort(reverse=0)
    #          vals = vals[:2 * matrix.dim]
    #          edges = [(ind / matrix.dim, ind % matrix.dim) for v, ind in vals]
    #          print "number of component edges:", len(edges), "number of components:", len(components)
            components_c = [(sum(self.layout.coors[0][c]) / len(c), sum(self.layout.coors[1][c]) / len(c)) for c in components]
            weights = [1 - matrix[u,v] for u,v in edges]
            
            max_weight = max(weights)
            min_weight = min(weights)
            span_weights = max_weight - min_weight
          
            for u,v in edges:
                x = [components_c[u][0], components_c[v][0]]
                y = [components_c[u][1], components_c[v][1]]
                w = ((1 - matrix[u,v]) - min_weight) / span_weights * (self.maxComponentEdgeWidth - self.minComponentEdgeWidth) + self.minComponentEdgeWidth
                
                pen = QPen()
                pen.setWidth(w)
                pen.setBrush(QColor(50,200,255,15))
                pen.setCapStyle(Qt.FlatCap)
                pen.setJoinStyle(Qt.RoundJoin)
                self.addCurve("component_edges", Qt.green, Qt.green, 0, style=QwtPlotCurve.Lines, symbol = QwtSymbol.NoSymbol, xData=x, yData=y, pen=pen, showFilledSymbols=False)
        
        
        self.networkCurve.setData(self.layout.coors[0], self.layout.coors[1])
        
        if self.insideview == 1:
            selection = self.networkCurve.getSelectedVertices()
            if len(selection) >= 1:
                visible = set()
                visible |= set(selection)
                visible |= self.getNeighboursUpTo(selection[0], self.insideviewNeighbours)
                self.networkCurve.setHiddenVertices(set(range(self.graph.number_of_nodes())) - visible)
    
        edgesCount = 0
        
        if self.forceVectors != None:
            for v in self.forceVectors:
                self.addCurve("force", Qt.white, Qt.green, 1, style=QwtPlotCurve.Lines, xData=v[0], yData=v[1], showFilledSymbols=False)
        
        for r in self.circles:
            step = 2 * pi / 64;
            fi = 0
            x = []
            y = []
            for i in range(65):
                x.append(r * cos(fi) + 5000)
                y.append(r * sin(fi) + 5000)
                fi += step
                
            self.addCurve("radius", Qt.white, Qt.green, 1, style=QwtPlotCurve.Lines, xData=x, yData=y, showFilledSymbols=False)
        
        if self.renderAntialiased:
            self.networkCurve.setRenderHint(QwtPlotItem.RenderAntialiased)
        else:
            self.networkCurve.setRenderHint(QwtPlotItem.RenderAntialiased, False)
      
        self.networkCurve.showEdgeLabels = self.showEdgeLabels
        self.networkCurve.attach(self)
        self.drawPlotItems(replot=0)
        
        #self.zoomExtent()
        
    def drawPlotItems(self, replot=1, vertices=[]):
        if len(vertices) > 0:
            for vertex in vertices:
                x1 = float(self.layout.coors[0][vertex])
                y1 = float(self.layout.coors[1][vertex])
                
                if vertex in self.markerKeys:
                    mkey = self.markerKeys[vertex]
                    mkey.setValue(x1, y1)
              
                if 'index ' + str(vertex) in self.markerKeys:
                    mkey = self.markerKeys['index ' + str(vertex)]
                    mkey.setValue(x1, y1)
                
                if vertex in self.tooltipKeys:
                    tkey = self.tooltipKeys[vertex]
                    self.tips.positions[tkey] = (x1, y1, 0, 0)
        else:
            self.markerKeys = {}
            self.removeMarkers()
            self.drawLabels()
            self.drawToolTips()
            self.drawWeights()
            self.drawIndexes()
            self.drawComponentKeywords()
        
        if replot:
            self.replot()
            
    def drawComponentKeywords(self):
        if self.showComponentAttribute == None:
            return
        
        if self.layout is None or self.graph is None or self.graph.items() is None:
            return
        
        if str(self.showComponentAttribute) not in self.graph.items().domain:
            self.showComponentAttribute = None
            return
        
        components = Orange.network.nx.algorithms.components.connected_components(self.graph)
        
        for component in components:
            if len(component) == 0:
                continue
            
            vertices = [vertex for vertex in component if self.vertices[vertex].show]
    
            if len(vertices) == 0:
                continue
            
            xes = [self.layout.coors[0][vertex] for vertex in vertices]  
            yes = [self.layout.coors[1][vertex] for vertex in vertices]  
                                  
            x1 = sum(xes) / len(xes)
            y1 = sum(yes) / len(yes)
            
            lbl = str(self.graph.items()[component[0]][str(self.showComponentAttribute)])
            
            mkey = self.addMarker(lbl, float(x1), float(y1), alignment=Qt.AlignCenter, size=self.fontSize)
    
    def drawToolTips(self):
      # add ToolTips
      self.tooltipData = []
      self.tooltipKeys = {}
      self.tips.removeAll()
      if len(self.tooltipText) > 0:
        for vertex in self.vertices:
          if not vertex.show:
            continue
          
          x1 = self.layout.coors[0][vertex.index]
          y1 = self.layout.coors[1][vertex.index]
          lbl = ""
          values = self.graph.items()[vertex.index]
          for ndx in self.tooltipText:
              if not ndx in self.graph.items().domain:
                  continue
              
              value = str(values[ndx])
              # wrap text
              i = 0
              while i < len(value) / 100:
                  value = value[:((i + 1) * 100) + i] + "\n" + value[((i + 1) * 100) + i:]
                  i += 1
                  
              lbl = lbl + str(value) + "\n"
    
          if lbl != '':
            lbl = lbl[:-1]
            self.tips.addToolTip(x1, y1, lbl)
            self.tooltipKeys[vertex.index] = len(self.tips.texts) - 1
                   
    def drawLabels(self):
        if len(self.labelText) > 0:
            for vertex in self.vertices:
                if not vertex.show:
                    continue
                
                if self.labelsOnMarkedOnly and not (vertex.marked):
                    continue
                                  
                x1 = self.layout.coors[0][vertex.index]
                y1 = self.layout.coors[1][vertex.index]
                lbl = ""
                values = self.graph.items()[vertex.index]
                if self.showMissingValues:
                    lbl = ", ".join([str(values[ndx]) for ndx in self.labelText])
                else:
                    lbl = ", ".join([str(values[ndx]) for ndx in self.labelText if str(values[ndx]) != '?'])
                #if not self.showMissingValues and lbl == '':
                #    continue 
                
                if lbl:
                    vertex.label = lbl
                    mkey = self.addMarker(lbl, float(x1), float(y1), alignment=Qt.AlignBottom, size=self.fontSize)
                    self.markerKeys[vertex.index] = mkey    
                     
    def drawIndexes(self):
        if self.showIndexes:
            for vertex in self.vertices:
                if not vertex.show:
                    continue
                
                if self.labelsOnMarkedOnly and not (vertex.marked):
                    continue
                                  
                x1 = self.layout.coors[0][vertex.index]
                y1 = self.layout.coors[1][vertex.index]
    
                lbl = str(vertex.index)
                mkey = self.addMarker(lbl, float(x1), float(y1), alignment=Qt.AlignTop, size=self.fontSize)
                self.markerKeys['index ' + str(vertex.index)] = mkey         
    
    def drawWeights(self):
        if self.showWeights:
            for edge in self.edges:
                if not (edge.u.show and edge.v.show):
                    continue
                
                if self.labelsOnMarkedOnly and not (edge.u.marked and edge.v.marked):
                    continue
                                  
                x1 = (self.layout.coors[0][edge.u.index] + self.layout.coors[0][edge.v.index]) / 2
                y1 = (self.layout.coors[1][edge.u.index] + self.layout.coors[1][edge.v.index]) / 2
                
                if edge.weight == None:
                    lbl = "None"
                else:
                    lbl = "%.2f" % float(edge.weight)
                
                mkey = self.addMarker(lbl, float(x1), float(y1), alignment=Qt.AlignCenter, size=self.fontSize)
                self.markerKeys[(edge.u, edge.v)] = mkey
                            
    def getColorIndeces(self, table, attribute, palette):
        colorIndices = {}
        colorIndex = None
        minValue = None
        maxValue = None
        
        if attribute[0] != "(" or attribute[ -1] != ")":
            i = 0
            for var in table.domain.variables:
                if var.name == attribute:
                    colorIndex = i
                    if var.varType == orange.VarTypes.Discrete: 
                        colorIndices = getVariableValueIndices(var, colorIndex)
                        
                i += 1
            metas = table.domain.getmetas()
            for i, var in metas.iteritems():
                if var.name == attribute:
                    colorIndex = i
                    if var.varType == orange.VarTypes.Discrete: 
                        colorIndices = getVariableValueIndices(var, colorIndex)
    
        colorIndices['?'] = len(colorIndices)
        palette.setNumberOfColors(len(colorIndices))
        
        if colorIndex != None and table.domain[colorIndex].varType == orange.VarTypes.Continuous:
            minValue = float(min([x[colorIndex].value for x in table if x[colorIndex].value != "?"] or [0.0]))
            maxValue = float(max([x[colorIndex].value for x in table if x[colorIndex].value != "?"] or [0.0]))
            
        return colorIndices, colorIndex, minValue, maxValue
    
    def setEdgeColor(self, attribute):
        if self.graph is None:
            return
        
        colorIndices, colorIndex, minValue, maxValue = self.getColorIndeces(self.graph.links(), attribute, self.discEdgePalette)
    
        for index in range(len(self.networkCurve.edges)):
            if colorIndex != None:
                links_index = self.networkCurve.edges[index].links_index
                if links_index == None:
                    continue
                
                if self.graph.links().domain[colorIndex].varType == orange.VarTypes.Continuous:
                    newColor = self.discEdgePalette[0]
                    if str(self.graph.links()[links_index][colorIndex]) != "?":
                        if maxValue == minValue:
                            newColor = self.discEdgePalette[0]
                        else:
                            value = (float(self.graph.links()[links_index][colorIndex].value) - minValue) / (maxValue - minValue)
                            newColor = self.contEdgePalette[value]
                        
                    self.networkCurve.setEdgeColor(index, newColor)
                    
                elif self.graph.links().domain[colorIndex].varType == orange.VarTypes.Discrete:
                    newColor = self.discEdgePalette[colorIndices[self.graph.links()[links_index][colorIndex].value]]
                    if self.graph.links()[links_index][colorIndex].value == "0":
                      self.networkCurve.setEdgeColor(index, newColor, nocolor=1)
                    else:
                      self.networkCurve.setEdgeColor(index, newColor)
                    
            else:
                newColor = self.discEdgePalette[0]
                self.networkCurve.setEdgeColor(index, newColor)
        
        self.replot()
    
    def setVertexColor(self, attribute):
        if self.graph is None:
            return
        
        colorIndices, colorIndex, minValue, maxValue = self.getColorIndeces(self.graph.items(), attribute, self.discPalette)
    
        for v in range(self.graph.number_of_nodes()):
            if colorIndex != None:    
                if self.graph.items().domain[colorIndex].varType == orange.VarTypes.Continuous:
                    newColor = self.discPalette[0]
                    
                    if str(self.graph.items()[v][colorIndex]) != "?":
                        if maxValue == minValue:
                            newColor = self.discPalette[0]
                        else:
                            value = (float(self.graph.items()[v][colorIndex].value) - minValue) / (maxValue - minValue)
                            newColor = self.contPalette[value]
                        
                    self.networkCurve.setVertexColor(v, newColor)
                    
                elif self.graph.items().domain[colorIndex].varType == orange.VarTypes.Discrete:
                    newColor = self.discPalette[colorIndices[self.graph.items()[v][colorIndex].value]]
                    #print newColor
                    self.networkCurve.setVertexColor(v, newColor)
                    
            else:
                newColor = self.discPalette[0]
                self.networkCurve.setVertexColor(v, newColor)
        
        self.replot()
        
    def setLabelText(self, attributes):
        self.labelText = []
        if self.layout is None or self.graph is None or self.graph.items() is None:
            return
        
        if isinstance(self.graph.items(), orange.ExampleTable):
            data = self.graph.items()
            for att in attributes:
                for i in range(len(data.domain)):
                    if data.domain[i].name == att:
                        self.labelText.append(i)
                        
                if self.graph.items().domain.hasmeta(att):
                        self.labelText.append(self.graph.items().domain.metaid(att))
    
    def setTooltipText(self, attributes):
        self.tooltipText = []
        if self.layout is None or self.graph is None or self.graph.items() is None:
            return
        
        if isinstance(self.graph.items(), orange.ExampleTable):
            data = self.graph.items()
            for att in attributes:
                for i in range(len(data.domain)):
                    if data.domain[i].name == att:
                        self.tooltipText.append(i)
                        
                if self.graph.items().domain.hasmeta(att):
                        self.tooltipText.append(self.graph.items().domain.metaid(att))
                        
    def setEdgeLabelText(self, attributes):
        self.edgeLabelText = []
        if self.layout is None or self.graph is None or self.graph.items() is None:
            return
        
    def set_graph_layout(self, graph, layout, curve=None):
        self.clear()
        self.vertexDegree = []
        self.vertices_old = {}
        self.vertices = []
        self.edges_old = {}
        self.edges = []
        self.minEdgeWeight = sys.maxint
        self.maxEdgeWeight = 0
        
        if graph is None or layout is None:
            self.graph = None
            self.layout = None
            self.networkCurve = None
            xMin = self.axisScaleDiv(QwtPlot.xBottom).interval().minValue()
            xMax = self.axisScaleDiv(QwtPlot.xBottom).interval().maxValue()
            yMin = self.axisScaleDiv(QwtPlot.yLeft).interval().minValue()
            yMax = self.axisScaleDiv(QwtPlot.yLeft).interval().maxValue()
            self.addMarker("no network", (xMax - xMin) / 2, (yMax - yMin) / 2, alignment=Qt.AlignCenter, size=self.fontSize)
            self.tooltipNeighbours = 0
            self.replot()
            return
        
        self.graph = graph
        self.layout = layout
        self.networkCurve = NetworkCurve(self) if curve is None else curve
        
        #add nodes
        self.vertices_old = [(None, []) for v in self.graph]
        self.vertices = [NetworkVertex(v) for v in self.graph]
        
        #build edge index
        row_ind = {}
        if self.graph.links() is not None and len(self.graph.links()) > 0:
          for i, r in enumerate(self.graph.links()):
              u = int(r['u'].value)
              v = int(r['v'].value)
              u_dict = row_ind.get(u, {})
              v_dict = row_ind.get(v, {})
              u_dict[v] = i
              v_dict[u] = i
              row_ind[u] = u_dict
              row_ind[v] = v_dict
              
        #add edges
        if self.graph.links() is not None and len(self.graph.links()) > 0:
            links = self.graph.links()
            links_indices = (row_ind[i + 1][j + 1] for (i, j) in self.graph.edges())
            labels = ([str(row[r].value) for r in range(2, len(row))] for row in (links[links_index] for links_index in links_indices))
            
            if self.graph.is_directed():
                self.edges = [NetworkEdge(self.vertices[i], self.vertices[j],
                    graph[i][j]['weight'], 0, 1, links_index, label) for \
                    ((i, j), links_index, label) in zip(self.graph.edges(), \
                                                        links_indices, labels)]
            else:
                self.edges = [NetworkEdge(self.vertices[i], self.vertices[j],
                    graph[i][j]['weight'], links_index, label) for \
                    ((i, j), links_index, label) in zip(self.graph.edges(), \
                                                        links_indices, labels)]
        elif self.graph.is_directed():
            self.edges = [NetworkEdge(self.vertices[i], self.vertices[j],
                                      graph[i][j]['weight'], 0, 1) for (i, j) in self.graph.edges()]
        else:
            self.edges = [NetworkEdge(self.vertices[i], self.vertices[j], 
                                      graph[i][j]['weight']) for (i, j) in self.graph.edges()]
        
        self.minEdgeWeight = min(edge.weight for edge in self.edges) if len(self.edges) > 0 else 0
        self.maxEdgeWeight = max(edge.weight for edge in self.edges) if len(self.edges) > 0 else 0
        
        if self.minEdgeWeight is None: 
            self.minEdgeWeight = 0 
        
        if self.maxEdgeWeight is None: 
            self.maxEdgeWeight = 0 
                          
        self.maxEdgeSize = 10
            
        self.setEdgesSize()
        self.setVerticesSize()
        
        self.networkCurve.coors = self.layout.coors
        self.networkCurve.vertices = self.vertices
        self.networkCurve.edges = self.edges
        self.networkCurve.changed()
        
    def setEdgesSize(self):
        if self.maxEdgeWeight > self.minEdgeWeight:
            #print 'maxEdgeSize',self.maxEdgeSize
            #print 'maxEdgeWeight',self.maxEdgeWeight
            #print 'minEdgeWeight',self.minEdgeWeight
            k = (self.maxEdgeSize - 1) / (self.maxEdgeWeight - self.minEdgeWeight)
            for edge in self.edges:
                if edge.weight == None:
                    size = 1
                    edge.pen = QPen(edge.pen.color(), size)
                    edge.pen.setCapStyle(Qt.RoundCap)
                else:
                    if self.invertEdgeSize:
                        size = (self.maxEdgeWeight - edge.weight - self.minEdgeWeight) * k + 1
                    else:
                        size = (edge.weight - self.minEdgeWeight) * k + 1
                    edge.pen = QPen(edge.pen.color(), size)
                    edge.pen.setCapStyle(Qt.RoundCap)
        else:
            for edge in self.edges:
                edge.pen = QPen(edge.pen.color(), 1)
                edge.pen.setCapStyle(Qt.RoundCap)
                
    def setVerticesSize(self, column=None, inverted=0):
        if self.layout is None or self.graph is None or self.graph.items() is None:
            return
        
        column = str(column)
        
        if column in self.graph.items().domain or (column.startswith("num of ") and column.replace("num of ", "") in self.graph.items().domain):
            values = []
            
            if column in self.graph.items().domain:
                values = [x[column].value for x in self.graph.items() if not x[column].isSpecial()]
            else:
                values = [len(x[column.replace("num of ", "")].value.split(',')) for x in self.graph.items()]
          
            minVertexWeight = float(min(values or [0]))
            maxVertexWeight = float(max(values or [0]))
            
            if maxVertexWeight - minVertexWeight == 0:
                k = 1 #doesn't matter
            else:
                k = (self.maxVertexSize - self.minVertexSize) / (maxVertexWeight - minVertexWeight)
            
            def getValue(v):
                if v.isSpecial():
                    return minVertexWeight
                else:
                    return float(v)
                 
            if inverted:
                for vertex in self.vertices:
                    if column in self.graph.items().domain:
                        vertex.size = self.maxVertexSize - ((getValue(self.graph.items()[vertex.index][column]) - minVertexWeight) * k)
                    else:
                        vertex.size = self.maxVertexSize - ((len(self.graph.items()[vertex.index][column.replace("num of ", "")].value.split(',')) - minVertexWeight) * k)
                    
                    
                    vertex.pen.setWidthF(1 + float(vertex.size) / 20)
            else:
                for vertex in self.vertices:
                    if column in self.graph.items().domain:
                        vertex.size = (getValue(self.graph.items()[vertex.index][column]) - minVertexWeight) * k + self.minVertexSize
                    else:
                        vertex.size = (float(len(self.graph.items()[vertex.index][column.replace("num of ", "")].value.split(','))) - minVertexWeight) * k + self.minVertexSize
                        
                    #print vertex.size
                    vertex.pen.setWidthF(1 + float(vertex.size) / 20)
        else:
            for vertex in self.vertices:
                vertex.size = self.maxVertexSize
                vertex.pen.setWidthF(1 + float(vertex.size) / 20)
      
    def updateCanvas(self):
        self.setAxisAutoScale(self.xBottom)
        self.setAxisAutoScale(self.yLeft)
        self.updateData()
        self.replot()  
    
    def zoomExtent(self):
        self.setAxisAutoScale(self.xBottom)
        self.setAxisAutoScale(self.yLeft)
        self.replot()
        
    def zoomSelection(self):
        selection = self.networkCurve.getSelectedVertices()
        if len(selection) > 0: 
            x = [self.layout.coors[0][v] for v in selection]
            y = [self.layout.coors[1][v] for v in selection]
    
            oldXMin = self.axisScaleDiv(QwtPlot.xBottom).interval().minValue()
            oldXMax = self.axisScaleDiv(QwtPlot.xBottom).interval().maxValue()
            oldYMin = self.axisScaleDiv(QwtPlot.yLeft).interval().minValue()
            oldYMax = self.axisScaleDiv(QwtPlot.yLeft).interval().maxValue()
            newXMin = min(x)
            newXMax = max(x)
            newYMin = min(y)
            newYMax = max(y)
            self.zoomStack.append((oldXMin, oldXMax, oldYMin, oldYMax))
            self.setAxisScale(QwtPlot.xBottom, newXMin - 100, newXMax + 100)
            self.setAxisScale(QwtPlot.yLeft, newYMin - 100, newYMax + 100)
            self.replot()
                    
