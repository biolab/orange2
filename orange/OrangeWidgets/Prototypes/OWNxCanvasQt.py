CIRCLE = 0
SQUARE = 1
ROUND_RECT = 2

NOTHING = 0
ZOOMING = 1
SELECT_RECTANGLE = 2
SELECT_POLYGON = 3
MOVE_SELECTION = 100

import Orange
import random 

from numpy import *
from plot.owplot import *
from plot.owpoint import *
from orngScaleScatterPlotData import *
import orangeplot

class NodeItem(orangeplot.NodeItem):
    def __init__(self, index, x, y, parent=None):
        orangeplot.NodeItem.__init__(self, index, OWPoint.Ellipse, Qt.blue, 5, parent)
        self.set_x(x)
        self.set_y(y)
        
class EdgeItem(orangeplot.EdgeItem):
    def __init__(self, u=None, v=None, weight=1, links_index=0, label='', parent=None):
        orangeplot.EdgeItem.__init__(self, u, v, parent)
        self.set_u(u)
        self.set_v(v)
        self.set_weight(weight)
        self.set_links_index(links_index)

class NetworkCurve(orangeplot.NetworkCurve):
  def __init__(self, parent=None, pen=QPen(Qt.black), xData=None, yData=None):
      orangeplot.NetworkCurve.__init__(self, parent)
      self.name = "Network Curve"
      self.showEdgeLabels = 0
      
  def move_selected_nodes(self, dx, dy):
    selected = self.get_selected_nodes()
    
    self.coors[selected][0] = self.coors[0][selected] + dx
    self.coors[1][selected][1] = self.coors[1][selected] + dy
      
    self.update_properties()
    return selected
  
  def set_node_color(self, v, color):
      pen = self.vertices[v].pen
      self.vertices[v].color = color
      self.vertices[v].pen = QPen(color, pen.width())

  def set_nodes_color(self, vertices, color):
      for v in vertices:
          v.color = color
          v.pen = QPen(color, v.pen.width())
      
  def set_edge_color(self, index, color, nocolor=0):
      pen = self.edges[index].pen
      if nocolor:
        color.setAlpha(0)
      self.edges[index].pen = QPen(color, pen.width())
      self.edges[index].pen.setCapStyle(Qt.RoundCap)
  
  def get_selected_nodes(self):
    return [vertex.index for vertex in self.vertices.itervalues() if vertex.selected]

  def get_unselected_nodes(self):
    return [vertex.index for vertex in self.vertices.itervalues() if not vertex.selected]

  def get_marked_nodes(self):
    return [vertex.index for vertex in self.vertices.itervalues() if vertex.marked]
  
  def set_marked_nodes(self, vertices):
    for vertex in self.vertices.itervalues():
      if vertex.index in vertices:
        vertex.marked = True
      else:
        vertex.marked = False
        
  def mark_to_sel(self):
    for vertex in self.vertices.itervalues():
      if vertex.marked == True:
          vertex.selected = True
          
  def sel_to_mark(self):
    for vertex in self.vertices.itervalues():
      if vertex.selected == True:
          vertex.selected = False
          vertex.marked = True
  
  def unmark(self):
    for vertex in self.vertices.itervalues():
      vertex.marked = False
      
  def unselect(self):
    for vertex in self.vertices.itervalues():
        vertex.selected = False
        
  def set_hidden_nodes(self, nodes):
    for vertex in self.vertices.itervalues():
      if vertex.index in nodes:
        vertex.show = False
      else:
        vertex.show = True
      
  def hide_selected_nodes(self):
    for vertex in self.vertices.itervalues():
      if vertex.selected:
        vertex.show = False
  
  def hide_unselected_nodes(self):
    for vertex in self.vertices.itervalues():
      if not vertex.selected:
        vertex.show = False
    
  def show_all_vertices(self):
    for vertex in self.vertices.itervalues():
      vertex.show = True
    
  def changed(self):
      self.itemChanged()
#
#  def draw(self, painter, xMap, yMap, rect):
#    for edge in self.edges:
#      if edge.u.show and edge.v.show:
#        painter.setPen(edge.pen)
#
#        px1 = xMap.transform(self.coors[edge.u.index][0])   #ali pa tudi self.x1, itd
#        py1 = yMap.transform(self.coors[edge.u.index][1])
#        px2 = xMap.transform(self.coors[edge.v.index][0])
#        py2 = yMap.transform(self.coors[edge.v.index][1])
#        
#        painter.drawLine(px1, py1, px2, py2)
#        
#        d = 12
#        #painter.setPen(QPen(Qt.lightGray, 1))
#        painter.setBrush(Qt.lightGray)
#        if edge.arrowu:
#            x = px2 - px1
#            y = py2 - py1
#            
#            fi = math.atan2(y, x) * 180 * 16 / math.pi 
#
#            if not fi is None:
#                # (180*16) - fi - (20*16), (40*16)
#                rect = QRectF(px2 - d, py2 - d, 2 * d, 2 * d)
#                painter.drawPie(rect, 2560 - fi, 640)
#                
#        if edge.arrowv:
#            x = px2 - px1
#            y = py2 - py1
#            
#            fi = math.atan2(y, x) * 180 * 16 / math.pi 
#            if not fi is None:
#                # (180*16) - fi - (20*16), (40*16)
#                rect = QRectF(px2 - d, py2 - d, 2 * d, 2 * d)
#                painter.drawPie(rect, 2560 - fi, 640)
#                
#        if self.showEdgeLabels and len(edge.label) > 0:
#            lbl = ', '.join(edge.label)
#            x = (px1 + px2) / 2
#            y = (py1 + py2) / 2
#            
#            th = painter.fontMetrics().height()
#            tw = painter.fontMetrics().width(lbl)
#            r = QRect(x - tw / 2, y - th / 2, tw, th)
#            painter.fillRect(r, QBrush(Qt.white))
#            painter.drawText(r, Qt.AlignHCenter + Qt.AlignVCenter, lbl)
#    
#    for key, vertex in self.vertices.iteritems():
#      if vertex.show:
#        pX = xMap.transform(self.coors[vertex.index][0])   #dobimo koordinati v pikslih (tipa integer)
#        pY = yMap.transform(self.coors[vertex.index][1])   #ki se stejeta od zgornjega levega kota canvasa
#        if vertex.selected:    
#          painter.setPen(QPen(Qt.yellow, 3))
#          painter.setBrush(vertex.color)
#          rect = QRectF(pX - (vertex.size + 4) / 2, pY - (vertex.size + 4) / 2, vertex.size + 4, vertex.size + 4)
#          painter.drawEllipse(rect)
#        elif vertex.marked:
#          painter.setPen(vertex.pen)
#          painter.setBrush(vertex.color)
#          rect = QRectF(pX - vertex.size / 2, pY - vertex.size / 2, vertex.size, vertex.size)
#          painter.drawEllipse(rect)
#        else:
#          painter.setPen(vertex.pen)
#          painter.setBrush(vertex.nocolor)
#          rect = QRectF(pX - vertex.size / 2, pY - vertex.size / 2, vertex.size, vertex.size)
#          painter.drawEllipse(rect)
        
  def closest_node(self, px, py):
    ndx = min(self.coors, key=lambda x: abs(self.coors[x][0]-px) + abs(self.coors[x][1]-py))
    return ndx, math.sqrt((self.coors[ndx][0]-px)**2 + (self.coors[ndx][0]-px)**2)

  def get_nodes_in_rect(self, x1, y1, x2, y2):
      if x1 > x2:
          x1, x2 = x2, x1
      if y1 > y2:
          y1, y2 = y2, y1
      return [key for key in self.coors if x1 < self.coors[key][0] < x2 and y1 < self.coors[key][1] < y2]
        
class OWNxCanvas(OWPlot):
    def __init__(self, master, parent=None, name="None"):
        OWPlot.__init__(self, parent, name)
        self.master = master
        self.parent = parent
        self.labelText = []
        self.tooltipText = []
        #self.vertices_old = {}         # distionary of nodes (orngIndex: vertex_objekt)
        #self.edges_old = {}            # distionary of edges (curveKey: edge_objekt)
        #self.vertices = []
        #self.edges = []
        self.indexPairs = {}       # distionary of type CurveKey: orngIndex   (for nodes)
        #self.selection = []        # list of selected nodes (indices)
        self.markerKeys = {}       # dictionary of type NodeNdx : markerCurveKey
        self.tooltipKeys = {}      # dictionary of type NodeNdx : tooltipCurveKey
        self.graph = None
        self.layout = None
        self.vertexDegree = []     # seznam vozlisc oblike (vozlisce, stevilo povezav), sortiran po stevilu povezav
        self.edgesKey = -1
        #self.vertexSize = 6
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
             
        self.networkCurve = NetworkCurve()
        self.add_custom_curve(self.networkCurve)
        self.callbackMoveVertex = None
        self.callbackSelectVertex = None
        self.minComponentEdgeWidth = 0
        self.maxComponentEdgeWidth = 0
        self.items_matrix = None
        self.controlPressed = False
        self.altPressed = False
        self.items = None
        self.links = None
        
        self.setFocusPolicy(Qt.StrongFocus)
        
    def update_canvas(self):
        self.networkCurve.update_properties()
        #rect = self.networkCurve.dataRect()
        #self.set_axis_scale(xBottom, min, max)
        #self.set_axis_scale(yLeft, min, max)
        self.set_dirty()
        self.replot()
        
    def getSelection(self):
      return self.networkCurve.get_selected_nodes()
    
    def get_marked_nodes(self):
      return self.networkCurve.get_marked_nodes()
        
    def getVertexSize(self, index):
        return 6
        
    def set_hidden_nodes(self, nodes):
        self.networkCurve.set_hidden_nodes(nodes)
    
    def hide_selected_nodes(self):
      self.networkCurve.hide_selected_nodes()
      self.drawPlotItems()
      
    def hide_unselected_nodes(self):
      self.networkCurve.hide_unselected_nodes()
      self.drawPlotItems()
      
    def show_all_vertices(self):
      self.networkCurve.show_all_vertices()
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
        self.networkCurve.mark_to_sel()
        self.drawPlotItems()
        
    def selectionToMarked(self):
        self.networkCurve.sel_to_mark()
        self.drawPlotItems()
        
        if self.sendMarkedNodes != None:
            self.sendMarkedNodes(self.networkCurve.get_marked_nodes())
        
    def removeSelection(self, replot=True):
        self.networkCurve.unselect()
        
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
        return self.networkCurve.get_selected_nodes()
        
    def getUnselectedExamples(self):
        return self.networkCurve.get_unselected_nodes()
    
    def getSelectedGraph(self):
      selection = self.networkCurve.get_selected_nodes()
      
      if len(selection) == 0:
          return None
      
      subgraph = self.graph.subgraph(selection)
      subnet = Network(subgraph)
      return subnet
    
    def get_selected_nodes(self):
      return self.networkCurve.get_selected_nodes()
    
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
            for ndx in self.networkCurve.get_selected_nodes():
                toMark |= self.getNeighboursUpTo(ndx, self.selectionNeighbours)
            
            self.networkCurve.set_marked_nodes(toMark)
            self.drawPlotItems()
                
        elif not self.freezeNeighbours and self.selectionNeighbours == 0:
            self.networkCurve.set_marked_nodes(self.networkCurve.get_selected_nodes())
            self.drawPlotItems()
            
        if self.sendMarkedNodes != None:
            self.sendMarkedNodes(self.networkCurve.get_marked_nodes())
                
    def unmark(self):
      self.networkCurve.unmark()
      self.drawPlotItems(replot=0)
      
      if self.sendMarkedNodes != None:
            self.sendMarkedNodes([])
            
    def set_marked_nodes(self, vertices):
      self.networkCurve.set_marked_nodes(vertices)
      self.drawPlotItems(replot=0)
      
      if self.sendMarkedNodes != None:
            self.sendMarkedNodes(self.networkCurve.get_marked_nodes())
        
    def activateMoveSelection(self):
        self.state = MOVE_SELECTION
#    
#    def mouseMoveEvent(self, event):
#        if not self.graph:
#          return
#          
#        if self.mouseCurrentlyPressed and self.state == MOVE_SELECTION and self.GMmouseMoveEvent != None:
#            newX = self.invTransform(2, event.pos().x())
#            newY = self.invTransform(0, event.pos().y())
#            
#            dx = newX - self.invTransform(2, self.GMmouseMoveEvent.x())
#            dy = newY - self.invTransform(0, self.GMmouseMoveEvent.y())
#            movedVertices = self.networkCurve.move_selected_nodes(dx, dy)
#            
#            self.GMmouseMoveEvent.setX(event.pos().x())  #zacetni dogodek postane trenutni
#            self.GMmouseMoveEvent.setY(event.pos().y())
#            
#            self.drawPlotItems(replot=1, vertices=movedVertices)
#            if self.callbackMoveVertex:
#                self.callbackMoveVertex()
#        else:
#            OWPlot.mouseMoveEvent(self, event)
#                
#        if not self.freezeNeighbours and self.tooltipNeighbours:
#            px = self.invTransform(2, event.x())
#            py = self.invTransform(0, event.y())   
#            ndx, mind = self.networkCurve.closest_node(px, py)
#            dX = self.transform(QwtPlot.xBottom, self.networkCurve.coors[ndx][0]) - event.x()
#            dY = self.transform(QwtPlot.yLeft,   self.networkCurve.coors[ndx][1]) - event.y()
#            # transform to pixel distance
#            distance = math.sqrt(dX**2 + dY**2) 
#              
#            if ndx != -1 and distance <= self.networkCurve.vertices[ndx].size:
#                toMark = set(self.getNeighboursUpTo(ndx, self.tooltipNeighbours))
#                self.networkCurve.set_marked_nodes(toMark)
#                self.drawPlotItems()
#                
#                if self.sendMarkedNodes != None:
#                    self.sendMarkedNodes(self.networkCurve.get_marked_nodes())
#            else:
#                self.networkCurve.unmark()
#                self.drawPlotItems()
#                
#                if self.sendMarkedNodes != None:
#                    self.sendMarkedNodes([])
#        
#        if self.showDistances:
#            selection = self.networkCurve.get_selected_nodes()
#            if len(selection) > 0:
#                px = self.invTransform(2, event.x())
#                py = self.invTransform(0, event.y())  
#                 
#                v, mind = self.networkCurve.closest_node(px, py)
#                dX = self.transform(QwtPlot.xBottom, self.networkCurve.coors[v][0]) - event.x()
#                dY = self.transform(QwtPlot.yLeft,   self.networkCurve.coors[v][1]) - event.y()
#                # transform to pixel distance
#                distance = math.sqrt(dX**2 + dY**2)               
#                if v != -1 and distance <= self.networkCurve.vertices[v].size:
#                    if self.items_matrix == None:
#                        dst = 'vertex distance signal not set'
#                    else:
#                        dst = 0
#                        for u in selection:
#                            dst += self.items_matrix[u, v]
#                        dst = dst / len(selection)
#                        
#                    self.showTip(event.pos().x(), event.pos().y(), str(dst))
#                    self.replot()
#    
#    def mousePressEvent(self, event):
#      if self.graph is None:
#          return
#          
#      #self.grabKeyboard()
#      self.mouseSelectedVertex = 0
#      self.GMmouseMoveEvent = None
#      
#      if self.state == MOVE_SELECTION:
#        self.mouseCurrentlyPressed = 1
#        #if self.isPointSelected(self.invTransform(self.xBottom, event.pos().x()), self.invTransform(self.yLeft, event.pos().y())) and self.selection != []:
#        #  self.GMmouseStartEvent = QPoint(event.pos().x(), event.pos().y())
#        #else:
#          # button pressed outside selected area or there is no area
#        self.selectVertex(event.pos())
#        self.GMmouseStartEvent = QPoint(event.pos().x(), event.pos().y())
#        self.replot()
#      elif self.state == SELECT_RECTANGLE:
#          self.GMmouseStartEvent = QPoint(event.pos().x(), event.pos().y())
#          
#          if self.clickedSelectedOnVertex(event.pos()):
#              self.mouseSelectedVertex = 1
#              self.mouseCurrentlyPressed = 1
#              self.state = MOVE_SELECTION
#              self.GMmouseMoveEvent = QPoint(event.pos().x(), event.pos().y())
#          elif self.clickedOnVertex(event.pos()):
#              self.mouseSelectedVertex = 1
#              self.mouseCurrentlyPressed = 1
#          else:
#              OWPlot.mousePressEvent(self, event)  
#      else:
#          OWPlot.mousePressEvent(self, event)     
#    
#    def mouseReleaseEvent(self, event):  
#        if self.graph is None:
#          return
#          
#        #self.releaseKeyboard()
#        if self.state == MOVE_SELECTION:
#            self.state = SELECT_RECTANGLE
#            self.mouseCurrentlyPressed = 0
#            
#            self.moveGroup = False
#            #self.GMmouseStartEvent=None
#            
#        if self.state == SELECT_RECTANGLE:
#            x1 = self.invTransform(2, self.GMmouseStartEvent.x())
#            y1 = self.invTransform(0, self.GMmouseStartEvent.y())
#            
#            x2 = self.invTransform(2, event.pos().x())
#            y2 = self.invTransform(0, event.pos().y())
#            
#            
#            if self.mouseSelectedVertex == 1 and x1 == x2 and y1 == y2 and self.selectVertex(self.GMmouseStartEvent):
#                QwtPlot.mouseReleaseEvent(self, event)
#            elif self.mouseSelectedVertex == 0:
#                 
#                selection = self.networkCurve.get_nodes_in_rect(x1, y1, x2, y2)
#    
#                def selectVertex(ndx):
#                    if self.networkCurve.vertices[ndx].show:
#                        self.networkCurve.vertices[ndx].selected = True
#                        
#                map(selectVertex, selection)
#                
#                if len(selection) == 0 and x1 == x2 and y1 == y2:
#                    self.removeSelection()
#                    self.unmark()
#            
#                self.markSelectionNeighbours()
#                OWPlot.mouseReleaseEvent(self, event)
#                self.removeAllSelections()
#    
#        elif self.state == SELECT_POLYGON:
#                OWPlot.mouseReleaseEvent(self, event)
#                if self.tempSelectionCurve == None:   #if OWVisGraph closed polygon
#                    self.selectVertices()
#        else:
#            OWPlot.mouseReleaseEvent(self, event)
#            
#        self.mouseCurrentlyPressed = 0
#        self.moveGroup = False
#            
#        if self.callbackSelectVertex != None:
#            self.callbackSelectVertex()
#    
#    def keyPressEvent(self, e):
#        if self.graph is None:
#          return
#          
#        if e.key() == 87 or e.key() == 81:
#            selection = [v.index for v in self.networkCurve.vertices.itervalues() if v.selected]
#            if len(selection) > 0:
#                phi = [math.pi / -180 if e.key() == 87 else math.pi / 180]
#                self.layout.rotate_vertices([selection], phi)
#                self.drawPlotItems(replot=1)
#            
#        if e.key() == Qt.Key_Control:
#            self.controlPressed = True
#        
#        elif e.key() == Qt.Key_Alt:
#            self.altPressed = True
#            
#        if e.text() == "f":
#            self.graph.freezeNeighbours = not self.graph.freezeNeighbours
#        
#        OWPlot.keyPressEvent(self, e)
#            
#    def keyReleaseEvent(self, e):
#        if e.key() == Qt.Key_Control:
#            self.controlPressed = False
#        
#        elif e.key() == Qt.Key_Alt:
#            self.altPressed = False
#        
#        OWPlot.keyReleaseEvent(self, e)
#        
#    def clickedSelectedOnVertex(self, pos):
#        min = 1000000
#        ndx = -1
#    
#        px = self.invTransform(2, pos.x())
#        py = self.invTransform(0, pos.y())   
#    
#        ndx, min = self.networkCurve.closest_node(px, py)
#        dX = self.transform(QwtPlot.xBottom, self.networkCurve.coors[ndx][0]) - pos.x()
#        dY = self.transform(QwtPlot.yLeft,   self.networkCurve.coors[ndx][1]) - pos.y()
#        # transform to pixel distance
#        distance = math.sqrt(dX**2 + dY**2)
#        
#        #self.networkCurve
#        
#        if ndx != -1 and distance <= self.networkCurve.vertices[ndx].size:
#            return self.networkCurve.vertices[ndx].selected
#        else:
#            return False
#        
    def clickedOnVertex(self, pos):
        min = 1000000
        ndx = -1
    
        px = self.invTransform(2, pos.x())
        py = self.invTransform(0, pos.y())   
    
        ndx, min = self.networkCurve.closest_node(px, py)
        dX = self.transform(QwtPlot.xBottom, self.networkCurve.coors[ndx][0]) - pos.x()
        dY = self.transform(QwtPlot.yLeft,   self.networkCurve.coors[ndx][1]) - pos.y()
        # transform to pixel distance
        distance = math.sqrt(dX**2 + dY**2)
        if ndx != -1 and distance <= self.networkCurve.vertices[ndx].size:
            return True
        else:
            return False
                
    def selectVertex(self, pos):
        min = 1000000
        ndx = -1
    
        px = self.invTransform(2, pos.x())
        py = self.invTransform(0, pos.y())   
    
        ndx, min = self.networkCurve.closest_node(px, py)
        
        dX = self.transform(QwtPlot.xBottom, self.networkCurve.coors[ndx][0]) - pos.x()
        dY = self.transform(QwtPlot.yLeft,   self.networkCurve.coors[ndx][1]) - pos.y()
        # transform to pixel distance
        distance = math.sqrt(dX**2 + dY**2)
        if ndx != -1 and distance <= self.networkCurve.vertices[ndx].size:
            if not self.appendToSelection and not self.controlPressed:
                self.removeSelection()
                      
            if self.insideview:
                self.networkCurve.unselect()
                self.networkCurve.vertices[ndx].selected = not self.networkCurve.vertices[ndx].selected
                self.optimize(100)
                
                self.markSelectionNeighbours()
            else:
                self.networkCurve.vertices[ndx].selected = not self.networkCurve.vertices[ndx].selected
                self.markSelectionNeighbours()
            
            return True  
        else:
            return False
            self.removeSelection()
            self.unmark()
    
    def updateData(self):
        if self.graph is None:
            return
        
#        self.removeDrawingCurves(removeLegendItems=0)
#        self.tips.removeAll()
#        
#        if self.items_matrix and self.minComponentEdgeWidth > 0 and self.maxComponentEdgeWidth > 0:          
#            components = Orange.network.nx.algorithms.components.connected_components(self.graph)
#            matrix = self.items_matrix.avgLinkage(components)
#            
#            edges = set()
#            for u in range(matrix.dim):
#                neighbours = matrix.getKNN(u, 2)
#                for v in neighbours:
#                    if v < u:
#                        edges.add((v, u))
#                    else:
#                        edges.add((u, v))
#            edges = list(edges)
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
#            components_c = [(sum(self.networkCurve.coors[c][0]) / len(c), sum(self.networkCurve.coors[c][1]) / len(c)) for c in components]
#            weights = [1 - matrix[u,v] for u,v in edges]
#            
#            max_weight = max(weights)
#            min_weight = min(weights)
#            span_weights = max_weight - min_weight
#          
#        self.networkCurve.update_properties()
        
        if self.insideview == 1:
            selection = self.networkCurve.get_selected_nodes()
            if len(selection) >= 1:
                visible = set()
                visible |= set(selection)
                visible |= self.getNeighboursUpTo(selection[0], self.insideviewNeighbours)
                self.networkCurve.set_hidden_nodes(set(range(self.graph.number_of_nodes())) - visible)
    
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
                
            self.addCurve("radius", Qt.white, Qt.green, 1, style=NetworkCurve.Lines, xData=x, yData=y, showFilledSymbols=False)
        
        """
        if self.renderAntialiased:
            self.networkCurve.setRenderHint(QwtPlotItem.RenderAntialiased)
        else:
            self.networkCurve.setRenderHint(QwtPlotItem.RenderAntialiased, False)
        """
        
        self.networkCurve.showEdgeLabels = self.showEdgeLabels
        self.networkCurve.attach(self)
        self.drawPlotItems(replot=0)
        
        #self.zoomExtent()
        
    def drawPlotItems(self, replot=1, vertices=[]):
#        if len(vertices) > 0:
#            for vertex in vertices:
#                x1 = float(self.networkCurve.coors[vertex][0])
#                y1 = float(self.networkCurve.coors[vertex][1])
#                
#                if vertex in self.markerKeys:
#                    mkey = self.markerKeys[vertex]
#                    mkey.setValue(x1, y1)
#              
#                if 'index ' + str(vertex) in self.markerKeys:
#                    mkey = self.markerKeys['index ' + str(vertex)]
#                    mkey.setValue(x1, y1)
#                
#                if vertex in self.tooltipKeys:
#                    tkey = self.tooltipKeys[vertex]
#                    self.tips.positions[tkey] = (x1, y1, 0, 0)
#        else:
#            self.markerKeys = {}
#            self.removeMarkers()
#            self.drawLabels()
#            self.drawToolTips()
#            self.drawWeights()
#            self.drawIndexes()
#            self.drawComponentKeywords()
#        
#        if replot:
            self.replot()
            
    def drawComponentKeywords(self):
        if self.showComponentAttribute == None:
            return
        
        if self.layout is None or self.graph is None or self.items is None:
            return
        
        if str(self.showComponentAttribute) not in self.items.domain:
            self.showComponentAttribute = None
            return
        
        components = Orange.network.nx.algorithms.components.connected_components(self.graph)
        
        for component in components:
            if len(component) == 0:
                continue
            
            vertices = [vertex for vertex in component if self.networkCurve.vertices[vertex].show]
    
            if len(vertices) == 0:
                continue
            
            xes = [self.networkCurve.coors[vertex][0] for vertex in vertices]  
            yes = [self.networkCurve.coors[vertex][1] for vertex in vertices]  
                                  
            x1 = sum(xes) / len(xes)
            y1 = sum(yes) / len(yes)
            
            lbl = str(self.items[component[0]][str(self.showComponentAttribute)])
            
            mkey = self.addMarker(lbl, float(x1), float(y1), alignment=Qt.AlignCenter, size=self.fontSize)
    
    def drawToolTips(self):
      # add ToolTips
      self.tooltipData = []
      self.tooltipKeys = {}
      self.tips.removeAll()
      if len(self.tooltipText) > 0:
        for vertex in self.networkCurve.vertices.itervalues():
          if not vertex.show:
            continue
          
          x1 = self.networkCurve.coors[vertex.index][0]
          y1 = self.networkCurve.coors[vertex.index][1]
          lbl = ""
          values = self.items[vertex.index]
          for ndx in self.tooltipText:
              if not ndx in self.items.domain:
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
            for vertex in self.networkCurve.vertices.itervalues():
                if not vertex.show:
                    continue
                
                if self.labelsOnMarkedOnly and not (vertex.marked):
                    continue
                                  
                x1 = self.networkCurve.coors[vertex.index][0]
                y1 = self.networkCurve.coors[vertex.index][1]
                lbl = ""
                values = self.items[vertex.index]
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
            for vertex in self.networkCurve.vertices.itervalues():
                if not vertex.show:
                    continue
                
                if self.labelsOnMarkedOnly and not (vertex.marked):
                    continue
                                  
                x1 = self.networkCurve.coors[vertex.index][0]
                y1 = self.networkCurve.coors[vertex.index][1]
    
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
                                  
                x1 = (self.networkCurve.coors[edge.u.index][0] + self.networkCurve.coors[edge.v.index][0]) / 2
                y1 = (self.networkCurve.coors[edge.u.index][1] + self.networkCurve.coors[edge.v.index][1]) / 2
                
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
    
    def set_edge_color(self, attribute):
        if self.graph is None:
            return
        
#        colorIndices, colorIndex, minValue, maxValue = self.getColorIndeces(self.links, attribute, self.discEdgePalette)
#    
#        for index in range(len(self.networkCurve.edges)):
#            if colorIndex != None:
#                links_index = self.networkCurve.edges[index].links_index
#                if links_index == None:
#                    continue
#                
#                if self.links.domain[colorIndex].varType == orange.VarTypes.Continuous:
#                    newColor = self.discEdgePalette[0]
#                    if str(self.links[links_index][colorIndex]) != "?":
#                        if maxValue == minValue:
#                            newColor = self.discEdgePalette[0]
#                        else:
#                            value = (float(self.links[links_index][colorIndex].value) - minValue) / (maxValue - minValue)
#                            newColor = self.contEdgePalette[value]
#                        
#                    self.networkCurve.set_edge_color(index, newColor)
#                    
#                elif self.links.domain[colorIndex].varType == orange.VarTypes.Discrete:
#                    newColor = self.discEdgePalette[colorIndices[self.links[links_index][colorIndex].value]]
#                    if self.links[links_index][colorIndex].value == "0":
#                      self.networkCurve.set_edge_color(index, newColor, nocolor=1)
#                    else:
#                      self.networkCurve.set_edge_color(index, newColor)
#                    
#            else:
#                newColor = self.discEdgePalette[0]
#                self.networkCurve.set_edge_color(index, newColor)
#        
#        self.replot()
    
    def set_node_color(self, attribute, nodes=None):
#        if self.graph is None:
#            return
#        
#        if nodes is None:
#            nodes = self.networkCurve.vertices.itervalues()
#        else:
#            nodes = (self.networkCurve.vertices[nodes] for node in nodes) 
#            
#        colorIndices, colorIndex, minValue, maxValue = self.getColorIndeces(self.items, attribute, self.discPalette)
#    
#        if colorIndex is not None and self.items.domain[colorIndex].varType == orange.VarTypes.Continuous:
#            for vertex in nodes:
#                v = vertex.index
#                newColor = self.discPalette[0]
#                
#                if str(self.items[v][colorIndex]) != "?":
#                    if maxValue == minValue:
#                        newColor = self.discPalette[0]
#                    else:
#                        value = (float(self.items[v][colorIndex].value) - minValue) / (maxValue - minValue)
#                        newColor = self.contPalette[value]
#                    
#                self.networkCurve.set_node_color(v, newColor)
#                
#        elif colorIndex is not None and self.items.domain[colorIndex].varType == orange.VarTypes.Discrete:
#            for vertex in nodes:
#                v = vertex.index
#                newColor = self.discPalette[colorIndices[self.items[v][colorIndex].value]]
#                self.networkCurve.set_node_color(v, newColor)
#        else:
#            self.networkCurve.set_nodes_color(nodes, self.discPalette[0])
#            
        self.replot()
        
    def setLabelText(self, attributes):
        self.labelText = []
        if self.layout is None or self.graph is None or self.items is None:
            return
        
        if isinstance(self.items, orange.ExampleTable):
            data = self.items
            for att in attributes:
                for i in range(len(data.domain)):
                    if data.domain[i].name == att:
                        self.labelText.append(i)
                        
                if self.items.domain.hasmeta(att):
                        self.labelText.append(self.items.domain.metaid(att))
    
    def setTooltipText(self, attributes):
        self.tooltipText = []
        if self.layout is None or self.graph is None or self.items is None:
            return
        
        if isinstance(self.items, orange.ExampleTable):
            data = self.items
            for att in attributes:
                for i in range(len(data.domain)):
                    if data.domain[i].name == att:
                        self.tooltipText.append(i)
                        
                if self.items.domain.hasmeta(att):
                        self.tooltipText.append(self.items.domain.metaid(att))
                        
    def setEdgeLabelText(self, attributes):
        self.edgeLabelText = []
        if self.layout is None or self.graph is None or self.items is None:
            return
        
    def change_graph(self, newgraph, inter_nodes, add_nodes, remove_nodes):
        self.graph = newgraph
        
        [self.networkCurve.vertices.pop(key) for key in remove_nodes]
        self.networkCurve.vertices.update(dict((v, NodeItem(v)) for v in add_nodes))
        vertices = self.networkCurve.vertices
        
        #build edge index
        row_ind = {}
        if self.links is not None and len(self.links) > 0:
          for i, r in enumerate(self.links):
              u = int(r['u'].value)
              v = int(r['v'].value)
              if u in self.graph and v in self.graph:
                  u_dict = row_ind.get(u, {})
                  v_dict = row_ind.get(v, {})
                  u_dict[v] = i
                  v_dict[u] = i
                  row_ind[u] = u_dict
                  row_ind[v] = v_dict
                  
        #add edges
        if self.links is not None and len(self.links) > 0:
            links = self.links
            links_indices = (row_ind[i + 1][j + 1] for (i, j) in self.graph.edges())
            labels = ([str(row[r].value) for r in range(2, len(row))] for row in (links[links_index] for links_index in links_indices))
            
            if self.graph.is_directed():
                edges = [EdgeItem(vertices[i], vertices[j],
                    self.graph[i][j].get('weight', 1), 0, 1, links_index, label) for \
                    ((i, j), links_index, label) in zip(self.graph.edges(), \
                                                        links_indices, labels)]
            else:
                edges = [EdgeItem(vertices[i], vertices[j],
                    self.graph[i][j].get('weight', 1), links_index, label) for \
                    ((i, j), links_index, label) in zip(self.graph.edges(), \
                                                        links_indices, labels)]
        elif self.graph.is_directed():
            edges = [EdgeItem(vertices[i], vertices[j],
                                      self.graph[i][j].get('weight', 1), 0, 1) for (i, j) in self.graph.edges()]
        else:
            edges = [EdgeItem(vertices[i], vertices[j],
                                      self.graph[i][j].get('weight', 1)) for (i, j) in self.graph.edges()]
                  
        #self.minEdgeWeight = min(edge.weight for edge in edges) if len(edges) > 0 else 0
        #self.maxEdgeWeight = max(edge.weight for edge in edges) if len(edges) > 0 else 0
        
        #if self.minEdgeWeight is None: 
        #    self.minEdgeWeight = 0 
        
        #if self.maxEdgeWeight is None: 
        #    self.maxEdgeWeight = 0 
                          
        #self.maxEdgeSize = 10
            
        #self.setEdgesSize()
        #self.setVerticesSize()
        
        self.networkCurve.coors = self.layout.map_to_graph(self.graph)
        self.networkCurve.edges = edges
        self.networkCurve.changed()
        
    def set_graph(self, graph, curve=None, items=None, links=None):
        self.clear()
        self.vertexDegree = []
        #self.vertices_old = {}
        #self.vertices = []
        #self.edges_old = {}
        #self.edges = []
        self.minEdgeWeight = sys.maxint
        self.maxEdgeWeight = 0
        
        if graph is None:
            self.graph = None
            #self.layout = None
            self.networkCurve = None
            self.items = None
            self.links = None
            xMin = -1.0
            xMax = 1.0
            yMin = -1.0
            yMax = 1.0
            self.addMarker("no network", (xMax - xMin) / 2, (yMax - yMin) / 2, alignment=Qt.AlignCenter, size=self.fontSize)
            self.tooltipNeighbours = 0
            self.replot()
            return
        
        self.graph = graph
        #self.layout = layout
        self.networkCurve = NetworkCurve() if curve is None else curve
        self.items = items if items is not None else self.graph.items()
        self.links = links if links is not None else self.graph.links()
        
        #add nodes
        #self.vertices_old = [(None, []) for v in self.graph]
        vertices = dict((v, NodeItem(v, random.random(), random.random(), parent=self.networkCurve)) for v in self.graph)
        self.networkCurve.set_nodes(vertices)
                
        #build edge index
        row_ind = {}
        if self.links is not None and len(self.links) > 0:
          for i, r in enumerate(self.links):
              u = int(r['u'].value)
              v = int(r['v'].value)
              if u in self.graph and v in self.graph:
                  u_dict = row_ind.get(u, {})
                  v_dict = row_ind.get(v, {})
                  u_dict[v] = i
                  v_dict[u] = i
                  row_ind[u] = u_dict
                  row_ind[v] = v_dict
              
        #add edges
        if self.links is not None and len(self.links) > 0:
            links = self.links
            links_indices = (row_ind[i + 1][j + 1] for (i, j) in self.graph.edges())
            labels = ([str(row[r].value) for r in range(2, len(row))] for row in (links[links_index] for links_index in links_indices))
            
            if self.graph.is_directed():
                edges = [EdgeItem(vertices[i], vertices[j],
                    graph[i][j].get('weight', 1), 0, 1, links_index, label, parent=self.networkCurve) for \
                    ((i, j), links_index, label) in zip(self.graph.edges(), \
                                                        links_indices, labels)]
            else:
                edges = [EdgeItem(vertices[i], vertices[j],
                    graph[i][j].get('weight', 1), links_index, label) for \
                    ((i, j), links_index, label) in zip(self.graph.edges(), \
                                                        links_indices, labels, parent=self.networkCurve)]
        elif self.graph.is_directed():
            edges = [EdgeItem(vertices[i], vertices[j],
                                      graph[i][j].get('weight', 1), 0, 1, parent=self.networkCurve) for (i, j) in self.graph.edges()]
        else:
            edges = [EdgeItem(vertices[i], vertices[j],
                                      graph[i][j].get('weight', 1), parent=self.networkCurve) for (i, j) in self.graph.edges()]
            
        self.networkCurve.set_edges(edges)
        self.networkCurve.update_properties()
        self.replot()
#        
#        self.minEdgeWeight = min(edge.weight for edge in edges) if len(edges) > 0 else 0
#        self.maxEdgeWeight = max(edge.weight for edge in edges) if len(edges) > 0 else 0
#        
#        if self.minEdgeWeight is None: 
#            self.minEdgeWeight = 0 
#        
#        if self.maxEdgeWeight is None: 
#            self.maxEdgeWeight = 0 
#                          
#        self.maxEdgeSize = 10
#            
#        self.setEdgesSize()
#        self.setVerticesSize()
#        
#        self.networkCurve.coors = self.layout.map_to_graph(self.graph)
#        self.networkCurve.vertices = vertices
#        self.networkCurve.edges = edges
#        self.networkCurve.changed()
        
    def set_graph_layout(self, graph, layout, curve=None, items=None, links=None):
        self.clear()
        self.vertexDegree = []
        #self.vertices_old = {}
        #self.vertices = []
        #self.edges_old = {}
        #self.edges = []
        self.minEdgeWeight = sys.maxint
        self.maxEdgeWeight = 0
        
        if graph is None or layout is None:
            self.graph = None
            self.layout = None
            self.networkCurve = None
            self.items = None
            self.links = None
            xMin = -1.0
            xMax = 1.0
            yMin = -1.0
            yMax = 1.0
            self.addMarker("no network", (xMax - xMin) / 2, (yMax - yMin) / 2, alignment=Qt.AlignCenter, size=self.fontSize)
            self.tooltipNeighbours = 0
            self.replot()
            return
        
        self.graph = graph
        self.layout = layout
        self.networkCurve = NetworkCurve(self) if curve is None else curve
        self.items = items if items is not None else self.graph.items()
        self.links = links if links is not None else self.graph.links()
        
        #add nodes
        #self.vertices_old = [(None, []) for v in self.graph]
        vertices = dict((v, NodeItem(v)) for v in self.graph)
        
        #build edge index
        row_ind = {}
        if self.links is not None and len(self.links) > 0:
          for i, r in enumerate(self.links):
              u = int(r['u'].value)
              v = int(r['v'].value)
              if u in self.graph and v in self.graph:
                  u_dict = row_ind.get(u, {})
                  v_dict = row_ind.get(v, {})
                  u_dict[v] = i
                  v_dict[u] = i
                  row_ind[u] = u_dict
                  row_ind[v] = v_dict
              
        #add edges
        if self.links is not None and len(self.links) > 0:
            links = self.links
            links_indices = (row_ind[i + 1][j + 1] for (i, j) in self.graph.edges())
            labels = ([str(row[r].value) for r in range(2, len(row))] for row in (links[links_index] for links_index in links_indices))
            
            if self.graph.is_directed():
                edges = [EdgeItem(vertices[i], vertices[j],
                    graph[i][j].get('weight', 1), 0, 1, links_index, label) for \
                    ((i, j), links_index, label) in zip(self.graph.edges(), \
                                                        links_indices, labels)]
            else:
                edges = [EdgeItem(vertices[i], vertices[j],
                    graph[i][j].get('weight', 1), links_index, label) for \
                    ((i, j), links_index, label) in zip(self.graph.edges(), \
                                                        links_indices, labels)]
        elif self.graph.is_directed():
            edges = [EdgeItem(vertices[i], vertices[j],
                                      graph[i][j].get('weight', 1), 0, 1) for (i, j) in self.graph.edges()]
        else:
            edges = [EdgeItem(vertices[i], vertices[j],
                                      graph[i][j].get('weight', 1)) for (i, j) in self.graph.edges()]
        
        self.minEdgeWeight = min(edge.weight for edge in edges) if len(edges) > 0 else 0
        self.maxEdgeWeight = max(edge.weight for edge in edges) if len(edges) > 0 else 0
        
        if self.minEdgeWeight is None: 
            self.minEdgeWeight = 0 
        
        if self.maxEdgeWeight is None: 
            self.maxEdgeWeight = 0 
                          
        self.maxEdgeSize = 10
            
        self.setEdgesSize()
        self.setVerticesSize()
        
        self.networkCurve.coors = self.layout.map_to_graph(self.graph)
        self.networkCurve.vertices = vertices
        self.networkCurve.edges = edges
        self.networkCurve.changed()
        
    def setEdgesSize(self):
#        if self.maxEdgeWeight > self.minEdgeWeight:
#            #print 'maxEdgeSize',self.maxEdgeSize
#            #print 'maxEdgeWeight',self.maxEdgeWeight
#            #print 'minEdgeWeight',self.minEdgeWeight
#            k = (self.maxEdgeSize - 1) / (self.maxEdgeWeight - self.minEdgeWeight)
#            for edge in self.networkCurve.edges:
#                if edge.weight == None:
#                    size = 1
#                    edge.pen = QPen(edge.pen.color(), size)
#                    edge.pen.setCapStyle(Qt.RoundCap)
#                else:
#                    if self.invertEdgeSize:
#                        size = (self.maxEdgeWeight - edge.weight - self.minEdgeWeight) * k + 1
#                    else:
#                        size = (edge.weight - self.minEdgeWeight) * k + 1
#                    edge.pen = QPen(edge.pen.color(), size)
#                    edge.pen.setCapStyle(Qt.RoundCap)
#        else:
#            for edge in self.networkCurve.edges:
#                edge.pen = QPen(edge.pen.color(), 1)
#                edge.pen.setCapStyle(Qt.RoundCap)
        pass                
    def setVerticesSize(self, column=None, inverted=0):
        if self.layout is None or self.graph is None or self.items is None:
            return
        
        column = str(column)
        
        if column in self.items.domain or (column.startswith("num of ") and column.replace("num of ", "") in self.items.domain):
            values = []
            
            if column in self.items.domain:
                values = [self.items[x][column].value for x in self.graph if not self.items[x][column].isSpecial()]
            else:
                values = [len(self.items[x][column.replace("num of ", "")].value.split(',')) for x in self.graph]
          
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
                for key, vertex in self.networkCurve.vertices.iteritems():
                    if column in self.items.domain:
                        vertex.size = self.maxVertexSize - ((getValue(self.items[vertex.index][column]) - minVertexWeight) * k)
                    else:
                        vertex.size = self.maxVertexSize - ((len(self.items[vertex.index][column.replace("num of ", "")].value.split(',')) - minVertexWeight) * k)
                    
                    
                    vertex.pen.setWidthF(1 + float(vertex.size) / 20)
            else:
                for key, vertex in self.networkCurve.vertices.iteritems():
                    if column in self.items.domain:
                        vertex.size = (getValue(self.items[vertex.index][column]) - minVertexWeight) * k + self.minVertexSize
                    else:
                        vertex.size = (float(len(self.items[vertex.index][column.replace("num of ", "")].value.split(','))) - minVertexWeight) * k + self.minVertexSize
                        
                    #print vertex.size
                    vertex.pen.setWidthF(1 + float(vertex.size) / 20)
        else:
            for key, vertex in self.networkCurve.vertices.iteritems():
                vertex.size = self.maxVertexSize
                vertex.pen.setWidthF(1 + float(vertex.size) / 20)
      
    def updateCanvas(self):
        self.updateData()
        self.replot()  
    
    def zoomExtent(self):
        self.replot()
        
    def zoomSelection(self):
        selection = self.networkCurve.get_selected_nodes()
        if len(selection) > 0: 
            x = [self.networkCurve.coors[v][0] for v in selection]
            y = [self.networkCurve.coors[v][1] for v in selection]
    
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
                    
