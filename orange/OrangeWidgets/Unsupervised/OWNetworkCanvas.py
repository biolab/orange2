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
from orngScaleScatterPlotData import *
from orngNetwork import Network

class NetworkVertex():
    def __init__(self):
        self.index = - 1
        self.marked = False
        self.show = True
        self.selected = False
        self.label = []
        self.tooltip = []
        
        self.pen = QPen(Qt.blue, 1)
        self.pen.setJoinStyle(Qt.RoundJoin)
        self.nocolor = Qt.white
        self.color = Qt.blue
        self.size = 5
    
class NetworkEdge():
    def __init__(self):
        self.u = None
        self.v = None
        self.links_index = None
        self.arrowu = 0
        self.arrowv = 0
        self.weight = 0
        self.label = []

        self.pen = QPen(Qt.lightGray, 1)

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
      
  def setEdgeColor(self, index, color):
      pen = self.edges[index].pen
      self.edges[index].pen = QPen(color, pen.width())
  
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
        
class OWNetworkCanvas(OWGraph):
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
      self.visualizer = None
      self.vertexDegree = []     # seznam vozlisc oblike (vozlisce, stevilo povezav), sortiran po stevilu povezav
      self.edgesKey = - 1
      #self.vertexSize = 6
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
      self.labelsOnMarkedOnly = 0
      self.enableWheelZoom = 1
      self.smoothOptimization = 0
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
      self.showComponentAttribute = None
      
      self.fontSize = 12
           
      self.setAxisAutoScale(self.xBottom)
      self.setAxisAutoScale(self.yLeft)
      
      self.networkCurve = NetworkCurve(self)
      self.callbackMoveVertex = None
      
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
              initTemp = self.visualizer.fruchtermanReingold(k, initTemp, coolFactor, self.hiddenNodes)
              iteration += 1
              qApp.processEvents()
              self.updateCanvas()

          initTemp = self.visualizer.fruchtermanReingold(o, initTemp, coolFactor, self.hiddenNodes)
          qApp.processEvents()
          self.updateCanvas()
      else:
          while iteration < o:
              initTemp = self.visualizer.fruchtermanReingold(1, initTemp, coolFactor, self.hiddenNodes)
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
              neighbours = set(self.visualizer.graph.getNeighbours(i))
              #print "neighbours: " + str(neighbours)
              self.selectNeighbours(sel, neighbours - sel, depth + 1, maxdepth)
      
  def getSelectedExamples(self):
      selection = self.networkCurve.getSelectedVertices()
      
      if len(selection) == 0:
          return None
      
      if self.visualizer.graph.items != None:
          return self.visualizer.graph.items.getitems(selection)
      else:
          return None
      
  def getUnselectedExamples(self):
      unselection = self.networkCurve.getUnselectedVertices()
      
      if len(unselection) == 0:
          return None
      
      if self.visualizer.graph.items != None:
          return self.visualizer.graph.items.getitems(unselection)
      else:
          return None

  def getSelectedGraph(self):
    selection = self.networkCurve.getSelectedVertices()
    
    if len(selection) == 0:
        return None
    
    #print self.visualizer.graph.items.domain
    subgraph = Network(self.visualizer.graph.getSubGraph(selection))
    #print subgraph.items.domain
    return subgraph
 
  def getSelectedVertices(self):
    return self.networkCurve.getSelectedVertices()
  
  def getNeighboursUpTo(self, ndx, dist):
      newNeighbours = neighbours = set([ndx])
      for d in range(dist):
          tNewNeighbours = set()
          for v in newNeighbours:
              tNewNeighbours |= set(self.visualizer.graph.getNeighbours(v))
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
          ndx, mind = self.visualizer.closestVertex(px, py)
          if ndx != - 1 and mind < 50:
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
              v, mind = self.visualizer.closestVertex(px, py)
                            
              if v != - 1 and mind < 100:
                  if self.visualizer.vertexDistance == None:
                      dst = 'vertex distance signal not set'
                  else:
                      dst = 0
                      for u in selection:
                          dst += self.visualizer.vertexDistance[u, v]
                      dst = dst / len(selection)
                      
                  self.showTip(event.pos().x(), event.pos().y(), str(dst))
                  self.replot()
                 
      if self.smoothOptimization:
          px = self.invTransform(2, event.x())
          py = self.invTransform(0, event.y())   
          ndx, mind = self.visualizer.closestVertex(px, py)
          if ndx != - 1 and mind < 30:
              if not self.optimizing:
                  self.optimizing = 1
                  initTemp = 1000
                  coolFactor = exp(log(10.0 / 10000.0) / 500)
                  
                  for i in range(10):
                      if self.stopOptimizing:
                          self.stopOptimizing = 0
                          break
                      initTemp = self.visualizer.smoothFruchtermanReingold(ndx, 50, initTemp, coolFactor)
                      qApp.processEvents()
                      self.updateData()
                      self.replot()
                  
                  self.optimizing = 0
          else:
              self.stopOptimizing = 1

  def mousePressEvent(self, event):
    self.grabKeyboard()
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
      self.releaseKeyboard()
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
              pass
          elif self.mouseSelectedVertex == 0:
          
              selection = self.visualizer.getVerticesInRect(x1, y1, x2, y2)

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

  def keyPressEvent(self, keyEvent):
      print keyEvent.key()
      if keyEvent.key() == 87:
          print "W"
          selection = [v.index for v in self.vertices if v.selected]
          #print selection
          self.visualizer.rotateVertices([selection], [ - 1])
          self.drawPlotItems(replot=1)
          
      elif keyEvent.key() == 81:
          print "Q"
          selection = [v.index for v in self.vertices if v.selected]
          #print selection
          self.visualizer.rotateVertices([selection], [1])
          self.drawPlotItems(replot=1)
      

  def clickedSelectedOnVertex(self, pos):
      min = 1000000
      ndx = - 1

      px = self.invTransform(2, pos.x())
      py = self.invTransform(0, pos.y())   

      ndx, min = self.visualizer.closestVertex(px, py)
      
      minx1 = self.invTransform(2, 0)
      miny1 = self.invTransform(0, 0)
      minx2 = self.invTransform(2, 10)
      miny2 = self.invTransform(0, 10)
      
      d = sqrt((minx2 - minx1) ** 2 + (miny2 - miny1) ** 2)
      
      if min < d and ndx != - 1:
          return self.vertices[ndx].selected
      else:
          return False
      
  def clickedOnVertex(self, pos):
      min = 1000000
      ndx = - 1

      px = self.invTransform(2, pos.x())
      py = self.invTransform(0, pos.y())   

      ndx, min = self.visualizer.closestVertex(px, py)
      
      minx1 = self.invTransform(2, 0)
      miny1 = self.invTransform(0, 0)
      minx2 = self.invTransform(2, 10)
      miny2 = self.invTransform(0, 10)
      
      d = sqrt((minx2 - minx1) ** 2 + (miny2 - miny1) ** 2)
      
      if min < d and ndx != - 1:
          return True
      else:
          return False
              
  def selectVertex(self, pos):
      min = 1000000
      ndx = - 1

      px = self.invTransform(2, pos.x())
      py = self.invTransform(0, pos.y())   

      ndx, min = self.visualizer.closestVertex(px, py)
      
      minx1 = self.invTransform(2, 0)
      miny1 = self.invTransform(0, 0)
      minx2 = self.invTransform(2, 10)
      miny2 = self.invTransform(0, 10)
      
      d = sqrt((minx2 - minx1) ** 2 + (miny2 - miny1) ** 2)
      
      if min < d and ndx != - 1:
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
      if self.visualizer == None:
          return
      
      self.removeDrawingCurves(removeLegendItems=0)
      self.tips.removeAll()
      self.networkCurve.setData(self.visualizer.network.coors[0], self.visualizer.network.coors[1])
      
      selection = self.networkCurve.getSelectedVertices()
      
      if self.insideview == 1 and len(selection) >= 1:
          visible = set()
          visible |= set(selection)
          visible |= self.getNeighboursUpTo(selection[0], self.insideviewNeighbours)
          self.networkCurve.setHiddenVertices(set(range(self.nVertices)) - visible)

      edgesCount = 0
      
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
              x1 = float(self.visualizer.network.coors[0][vertex])
              y1 = float(self.visualizer.network.coors[1][vertex])
              
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
      
      if self.visualizer == None or self.visualizer.graph == None or self.visualizer.graph.items == None:
          return
      
      if str(self.showComponentAttribute) not in self.visualizer.graph.items.domain:
          self.showComponentAttribute = None
          return
      
      components = self.visualizer.graph.getConnectedComponents()
      
      for component in components:
          if len(component) == 0:
              continue
          
          vertices = [vertex for vertex in component if self.vertices[vertex].show]

          if len(vertices) == 0:
              continue
          
          xes = [self.visualizer.network.coors[0][vertex] for vertex in vertices]  
          yes = [self.visualizer.network.coors[1][vertex] for vertex in vertices]  
                                
          x1 = sum(xes) / len(xes)
          y1 = sum(yes) / len(yes)
          
          lbl = str(self.visualizer.graph.items[component[0]][str(self.showComponentAttribute)])
          
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
        
        x1 = self.visualizer.network.coors[0][vertex.index]
        y1 = self.visualizer.network.coors[1][vertex.index]
        lbl = ""
        values = self.visualizer.graph.items[vertex.index]
        for ndx in self.tooltipText:
            if not ndx in self.visualizer.graph.items.domain:
                continue
            
            value = str(values[ndx])
            # wrap text
            i = 0
            while i < len(value) / 100:
                value = value[:((i + 1) * 100) + i] + "\n" + value[((i + 1) * 100) + i:]
                i += 1
                
            lbl = lbl + str(value) + "\n"
  
        if lbl != '':
          lbl = lbl[: - 1]
          self.tips.addToolTip(x1, y1, lbl)
          self.tooltipKeys[vertex.index] = len(self.tips.texts) - 1
                 
  def drawLabels(self):
      if len(self.labelText) > 0:
          for vertex in self.vertices:
              if not vertex.show:
                  continue
              
              if self.labelsOnMarkedOnly and not (vertex.marked):
                  continue
                                
              x1 = self.visualizer.network.coors[0][vertex.index]
              y1 = self.visualizer.network.coors[1][vertex.index]
              lbl = ""
              values = self.visualizer.graph.items[vertex.index]
              if self.showMissingValues:
                  lbl = ", ".join([str(values[ndx]) for ndx in self.labelText])
              else:
                  lbl = ", ".join([str(values[ndx]) for ndx in self.labelText if str(values[ndx]) != '?'])
              #if not self.showMissingValues and lbl == '':
              #    continue 
              
              if lbl:
                  mkey = self.addMarker(lbl, float(x1), float(y1), alignment=Qt.AlignBottom, size=self.fontSize)
                  self.markerKeys[vertex.index] = mkey    
                   
  def drawIndexes(self):
      if self.showIndexes:
          for vertex in self.vertices:
              if not vertex.show:
                  continue
              
              if self.labelsOnMarkedOnly and not (vertex.marked):
                  continue
                                
              x1 = self.visualizer.network.coors[0][vertex.index]
              y1 = self.visualizer.network.coors[1][vertex.index]

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
                                
              x1 = (self.visualizer.network.coors[0][edge.u.index] + self.visualizer.network.coors[0][edge.v.index]) / 2
              y1 = (self.visualizer.network.coors[1][edge.u.index] + self.visualizer.network.coors[1][edge.v.index]) / 2
              
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
      
      if attribute[0] != "(" or attribute[ - 1] != ")":
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
          minValue = float(min([x[colorIndex].value for x in table if x[colorIndex].value != "?"]))
          maxValue = float(max([x[colorIndex].value for x in table if x[colorIndex].value != "?"]))
          
      return colorIndices, colorIndex, minValue, maxValue
  
  def setEdgeColor(self, attribute):
      if self.visualizer == None:
          return
      
      colorIndices, colorIndex, minValue, maxValue = self.getColorIndeces(self.visualizer.graph.links, attribute, self.discEdgePalette)

      for index in range(self.nEdges):
          if colorIndex != None:
              links_index = self.networkCurve.edges[index].links_index
              if links_index == None:
                  continue
              
              if self.visualizer.graph.links.domain[colorIndex].varType == orange.VarTypes.Continuous:
                  newColor = self.discEdgePalette[0]
                  if str(self.visualizer.graph.links[links_index][colorIndex]) != "?":
                      if maxValue == minValue:
                          newColor = self.discEdgePalette[0]
                      else:
                          value = (float(self.visualizer.graph.links[links_index][colorIndex].value) - minValue) / (maxValue - minValue)
                          newColor = self.contEdgePalette[value]
                      
                  self.networkCurve.setEdgeColor(index, newColor)
                  
              elif self.visualizer.graph.links.domain[colorIndex].varType == orange.VarTypes.Discrete:
                  newColor = self.discEdgePalette[colorIndices[self.visualizer.graph.links[links_index][colorIndex].value]]
                  self.networkCurve.setEdgeColor(index, newColor)
                  
          else:
              newColor = self.discEdgePalette[0]
              self.networkCurve.setEdgeColor(index, newColor)
      
      self.replot()
  
  def setVertexColor(self, attribute):
      if self.visualizer == None:
          return
      
      colorIndices, colorIndex, minValue, maxValue = self.getColorIndeces(self.visualizer.graph.items, attribute, self.discPalette)

      for v in range(self.nVertices):
          if colorIndex != None:    
              if self.visualizer.graph.items.domain[colorIndex].varType == orange.VarTypes.Continuous:
                  newColor = self.discPalette[0]
                  
                  if str(self.visualizer.graph.items[v][colorIndex]) != "?":
                      if maxValue == minValue:
                          newColor = self.discPalette[0]
                      else:
                          value = (float(self.visualizer.graph.items[v][colorIndex].value) - minValue) / (maxValue - minValue)
                          newColor = self.contPalette[value]
                      
                  self.networkCurve.setVertexColor(v, newColor)
                  
              elif self.visualizer.graph.items.domain[colorIndex].varType == orange.VarTypes.Discrete:
                  newColor = self.discPalette[colorIndices[self.visualizer.graph.items[v][colorIndex].value]]
                  #print newColor
                  self.networkCurve.setVertexColor(v, newColor)
                  
          else:
              newColor = self.discPalette[0]
              self.networkCurve.setVertexColor(v, newColor)
      
      self.replot()
      
  def setLabelText(self, attributes):
      self.labelText = []
      if self.visualizer == None or self.visualizer.graph == None or self.visualizer.graph.items == None:
          return
      
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
      if self.visualizer == None or self.visualizer.graph == None or self.visualizer.graph.items == None:
          return
      
      if isinstance(self.visualizer.graph.items, orange.ExampleTable):
          data = self.visualizer.graph.items
          for att in attributes:
              for i in range(len(data.domain)):
                  if data.domain[i].name == att:
                      self.tooltipText.append(i)
                      
              if self.visualizer.graph.items.domain.hasmeta(att):
                      self.tooltipText.append(self.visualizer.graph.items.domain.metaid(att))
                      
  def setEdgeLabelText(self, attributes):
      self.edgeLabelText = []
      if self.visualizer == None or self.visualizer.graph == None or self.visualizer.graph.items == None:
          return
      
  def edgesContainsEdge(self, i, j):
      for e in range(self.nEdges):
          (key, iTmp, jTmp) = self.edges_old[e]
          
          if (iTmp == i and jTmp == j) or (iTmp == j and jTmp == i):
              return True
      return False
      
  def addVisualizer(self, visualizer):
      self.visualizer = visualizer
      self.clear()
      
      self.nVertices = 0
      self.nEdges = 0
      self.vertexDegree = []
      self.vertices_old = {}
      self.vertices = []
      self.edges_old = {}
      self.nEdges = 0
      self.networkCurve = NetworkCurve(self)
      self.edges = []
      self.minEdgeWeight = sys.maxint
      self.maxEdgeWeight = 0
      
      if visualizer == None:
          xMin = self.axisScaleDiv(QwtPlot.xBottom).lowerBound()
          xMax = self.axisScaleDiv(QwtPlot.xBottom).upperBound()
          yMin = self.axisScaleDiv(QwtPlot.yLeft).lowerBound()
          yMax = self.axisScaleDiv(QwtPlot.yLeft).upperBound()
          self.addMarker("no network", (xMax - xMin) / 2, (yMax - yMin) / 2, alignment=Qt.AlignCenter, size=self.fontSize)
          self.replot()
          return
      
      self.nVertices = visualizer.graph.nVertices
      #add nodes
      for v in range(0, self.nVertices):
          self.vertices_old[v] = (None, [])
          vertex = NetworkVertex()
          vertex.index = v
          self.vertices.append(vertex)
      
      #print "addVisualizer"
      
      #add edges
      for (i, j) in visualizer.graph.getEdges():
          self.edges_old[self.nEdges] = (None, i, j)
          edge = NetworkEdge()
          edge.u = self.vertices[i]
          edge.v = self.vertices[j]
          
          edge.weight = float(str(visualizer.graph[i, j]))  
              
          #print "weight:", edge.weight
          
          self.edges.append(edge)
          self.nEdges += 1
          
          if edge.weight != None and self.minEdgeWeight > edge.weight:
              self.minEdgeWeight = edge.weight
              
          elif edge.weight != None and self.maxEdgeWeight < edge.weight:
              self.maxEdgeWeight = edge.weight
            
          if visualizer.graph.directed:
              edge.arrowu = 0
              edge.arrowv = 1
              
          if visualizer.graph.links != None and len(visualizer.graph.links) > 0:
              row = visualizer.graph.links.filter(u=(i + 1, i + 1), v=(j + 1, j + 1))
              
              if len(row) == 0:
                  row = visualizer.graph.links.filter(u=(j + 1, j + 1), v=(i + 1, i + 1))
                  indexes = [k for k, x in enumerate(visualizer.graph.links) if (str(int(x[0])) == str(j + 1) and str(int(x[1])) == str(int(i + 1)))]
              else:
                  indexes = [k for k, x in enumerate(visualizer.graph.links) if (str(int(x[0])) == str(i + 1) and str(int(x[1])) == str(int(j + 1)))]
                  
              if len(row) == 1:
                  edge.label = []
                  for k in range(2, len(row[0])):
                      edge.label.append(str(row[0][k]))
              else:
                  print i, j, "not found"
              
              if len(indexes) == 1:
                  edge.links_index = indexes[0]
                        
      if self.maxEdgeWeight < 10:
          self.maxEdgeSize = self.maxEdgeWeight
      else:
          self.maxEdgeSize = 10
          
      self.setEdgesSize()
      self.setVerticesSize()
      
      self.networkCurve.coors = visualizer.network.coors
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
              else:
                  size = (edge.weight - self.minEdgeWeight) * k + 1
                  edge.pen = QPen(edge.pen.color(), size)
      else:
          for edge in self.edges:
              edge.pen = QPen(edge.pen.color(), 1)
              
  def setVerticesSize(self, column=None, inverted=0):
      if self.visualizer == None or self.visualizer.graph == None or self.visualizer.graph.items == None:
          return
      
      column = str(column)
      
      if column in self.visualizer.graph.items.domain or (column.startswith("num of ") and column.replace("num of ", "") in self.visualizer.graph.items.domain):
          values = []
          
          if column in self.visualizer.graph.items.domain:
              values = [x[column].value for x in self.visualizer.graph.items]
          else:
              values = [len(x[column.replace("num of ", "")].value.split(',')) for x in self.visualizer.graph.items]
        
          minVertexWeight = float(min(values))
          maxVertexWeight = float(max(values))
          
          if maxVertexWeight - minVertexWeight == 0:
              k = 1 #doesn't matter
          else:
              k = (self.maxVertexSize - self.minVertexSize) / (maxVertexWeight - minVertexWeight)
          
          if inverted:
              for vertex in self.vertices:
                  if column in self.visualizer.graph.items.domain:
                      vertex.size = self.maxVertexSize - ((self.visualizer.graph.items[vertex.index][column].value - minVertexWeight) * k)
                  else:
                      vertex.size = self.maxVertexSize - ((len(self.visualizer.graph.items[vertex.index][column.replace("num of ", "")].value.split(',')) - minVertexWeight) * k)
                  
                  
                  vertex.pen.setWidthF(1 + float(vertex.size) / 20)
          else:
              for vertex in self.vertices:
                  if column in self.visualizer.graph.items.domain:
                      vertex.size = (self.visualizer.graph.items[vertex.index][column].value - minVertexWeight) * k + self.minVertexSize
                  else:
                      vertex.size = (float(len(self.visualizer.graph.items[vertex.index][column.replace("num of ", "")].value.split(','))) - minVertexWeight) * k + self.minVertexSize
                      
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
          x = [self.visualizer.network.coors[0][v] for v in selection]
          y = [self.visualizer.network.coors[1][v] for v in selection]

          oldXMin = self.axisScaleDiv(QwtPlot.xBottom).lBound()
          oldXMax = self.axisScaleDiv(QwtPlot.xBottom).hBound()
          oldYMin = self.axisScaleDiv(QwtPlot.yLeft).lBound()
          oldYMax = self.axisScaleDiv(QwtPlot.yLeft).hBound()
          newXMin = min(x)
          newXMax = max(x)
          newYMin = min(y)
          newYMax = max(y)
          self.zoomStack.append((oldXMin, oldXMax, oldYMin, oldYMax))
          self.setAxisScale(QwtPlot.xBottom, newXMin - 100, newXMax + 100)
          self.setAxisScale(QwtPlot.yLeft, newYMin - 100, newYMax + 100)
          self.replot()
                  