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
    def __init__(self, index, x=None, y=None, parent=None):
        orangeplot.NodeItem.__init__(self, index, OWPoint.Ellipse, Qt.blue, 5, parent)
        if x is not None:
            self.set_x(x)
        if y is not None:
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
      
    def set_node_sizes(self, values={}, min_size=0, max_size=0):
        orangeplot.NetworkCurve.set_node_sizes(self, values, min_size, max_size)
      
    def move_selected_nodes(self, dx, dy):
        selected = self.get_selected_nodes()
        
        self.coors[selected][0] = self.coors[0][selected] + dx
        self.coors[1][selected][1] = self.coors[1][selected] + dy
          
        self.update_properties()
        return selected
  
    def get_selected_nodes(self):
        return [vertex.index() for vertex in self.nodes().itervalues() if vertex.is_selected()]

    def get_unselected_nodes(self):
        return [vertex.index() for vertex in self.nodes().itervalues() if not vertex.is_selected()]

    def get_marked_nodes(self):
        return [vertex.index() for vertex in self.nodes().itervalues() if vertex.is_marked()]
  
    def set_marked_nodes(self, vertices):
        n = self.nodes()
        for vertex in n.itervalues():
          vertex.set_marked(False)
        for index in vertices:
          if index in n:
            n[index].set_marked(True)
        
    def mark_to_sel(self):
        for vertex in self.nodes().itervalues():
          if vertex.is_marked():
              vertex.set_selected(True)
          
    def sel_to_mark(self):
        for vertex in self.nodes().itervalues():
          if vertex.is_selected():
              vertex.set_selected(False)
              vertex.set_marked(True)
  
    def unmark(self):
        for vertex in self.nodes().itervalues():
          vertex.set_marked(False)
      
    def unselect(self):
        for vertex in self.nodes().itervalues():
            vertex.set_selected(False)
        
    def set_hidden_nodes(self, nodes):
        for vertex in self.nodes().itervalues():
            vertex.setVisible(vertex.index() in nodes)
      
    def hide_selected_nodes(self):
        for vertex in self.nodes().itervalues():
          if vertex.selected:
            vertex.hide()
  
    def hide_unselected_nodes(self):
        for vertex in self.nodes().itervalues():
          if not vertex.selected:
            vertex.hide()
    
    def show_all_vertices(self):
        for vertex in self.nodes().itervalues():
          vertex.show()
    
    def changed(self):
        self.itemChanged()
        
    def closest_node(self, px, py):
        ndx = min(self.coors, key=lambda x: abs(self.coors[x][0]-px) + abs(self.coors[x][1]-py))
        return ndx, math.sqrt((self.coors[ndx][0]-px)**2 + (self.coors[ndx][0]-px)**2)

    def get_nodes_in_rect(self, x1, y1, x2, y2):
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1
        return [key for key in self.coors if x1 < self.coors[key][0] < x2 and y1 < self.coors[key][1] < y2]
        
    def fr(self, steps, weighted=False, smooth_cooling=False):
        orangeplot.NetworkCurve.fr(self, steps, weighted, smooth_cooling)
        
class OWNxCanvas(OWPlot):
    def __init__(self, master, parent=None, name="None"):
        OWPlot.__init__(self, parent, name)
        self.master = master
        self.parent = parent
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
        self.show_indices = True
        self.showWeights = 0
        self.minEdgeWeight = sys.maxint
        self.maxEdgeWeight = 0
        self.maxEdgeSize = 1
        
        self.invertEdgeSize = 0
        self.showComponentAttribute = None
        self.forceVectors = None
        self.appendToSelection = 1
        self.fontSize = 12
             
        self.networkCurve = NetworkCurve()
        self.add_custom_curve(self.networkCurve)
        self.callbackMoveVertex = None
        
        self.minComponentEdgeWidth = 0
        self.maxComponentEdgeWidth = 0
        self.items_matrix = None
        self.controlPressed = False
        self.altPressed = False
        self.items = None
        self.links = None
        
        self.node_label_attributes = []
        self.edge_color_attribute = None
        
        self.axis_margin = 0
        self.title_margin = 0
        self.graph_margin = 1
        self._legend_margin = QRectF(0, 0, 0, 0)
        
        self.setFocusPolicy(Qt.StrongFocus)
        
    def update_canvas(self):
        self.networkCurve.update_properties()
        
    def get_marked_nodes(self):
      return self.networkCurve.get_marked_nodes()
        
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
      
#    def optimize(self, frSteps):
#        qApp.processEvents()
#        tolerance = 5
#        initTemp = 100
#        breakpoints = 20
#        k = int(frSteps / breakpoints)
#        o = frSteps % breakpoints
#        iteration = 0
#        coolFactor = exp(log(10.0 / 10000.0) / frSteps)
#        #print coolFactor
#        if k > 0:
#            while iteration < breakpoints:
#                initTemp = self.layout.fruchtermanReingold(k, initTemp, coolFactor, self.hiddenNodes)
#                iteration += 1
#                qApp.processEvents()
#                self.updateCanvas()
#    
#            initTemp = self.layout.fruchtermanReingold(o, initTemp, coolFactor, self.hiddenNodes)
#            qApp.processEvents()
#            self.updateCanvas()
#        else:
#            while iteration < o:
#                initTemp = self.layout.fruchtermanReingold(1, initTemp, coolFactor, self.hiddenNodes)
#                iteration += 1
#                qApp.processEvents()
#                self.updateCanvas()
                
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
    
    def selected_nodes(self):
        return [p.index() for p in self.selected_points()]
    
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
    
    def updateData(self):
        if self.graph is None:
            return
        
        self.clear()
        self.tooltipData = []
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
        
        colorIndices, colorIndex, minValue, maxValue = self.getColorIndeces(self.items, attribute, self.discPalette)
        colors = []
        
        if colorIndex is not None and self.items.domain[colorIndex].varType == orange.VarTypes.Continuous and minValue == maxValue:
            colors = [self.discEdgePalette[0] for edge in self.networkCurve.edge_indices()]
        
        elif colorIndex is not None and self.items.domain[colorIndex].varType == orange.VarTypes.Continuous:
            #colors.update((v, self.contPalette[(float(self.items[v][colorIndex].value) - minValue) / (maxValue - minValue)]) 
            #              if str(self.items[v][colorIndex].value) != '?' else 
            #              (v, self.discPalette[0]) for v in nodes)
            print "TODO set continuous color"
        elif colorIndex is not None and self.items.domain[colorIndex].varType == orange.VarTypes.Discrete:
            #colors.update((v, self.discPalette[colorIndices[self.items[v][colorIndex].value]]) for v in nodes)
            print "TODO set discrete color"
        else:
            colors = [self.discEdgePalette[0] for edge in self.networkCurve.edge_indices()]
            
        self.networkCurve.set_edge_color(colors)
        self.replot()
        
#                 if self.links.domain[colorIndex].varType == orange.VarTypes.Continuous:
#                    newColor = self.discEdgePalette[0]
#                        else:
#                            value = (float(self.links[links_index][colorIndex].value) - minValue) / (maxValue - minValue)
#                            newColor = self.contEdgePalette[value]
#                elif self.links.domain[colorIndex].varType == orange.VarTypes.Discrete:
#                    newColor = self.discEdgePalette[colorIndices[self.links[links_index][colorIndex].value]]
#                    if self.links[links_index][colorIndex].value == "0":
#                      self.networkCurve.set_edge_color(index, newColor, nocolor=1)
#                    else:
#                      self.networkCurve.set_edge_color(index, newColor)
    
    def set_node_colors(self, attribute, nodes=None):
        if self.graph is None:
            return

        colorIndices, colorIndex, minValue, maxValue = self.getColorIndeces(self.items, attribute, self.discPalette)
        colors = {}
        
        if nodes is None:
            nodes = self.graph.nodes()
        
        if colorIndex is not None and self.items.domain[colorIndex].varType == orange.VarTypes.Continuous and minValue == maxValue:
            colors.update((node, self.discPalette[0]) for node in nodes)
        
        elif colorIndex is not None and self.items.domain[colorIndex].varType == orange.VarTypes.Continuous:
            colors.update((v, self.contPalette[(float(self.items[v][colorIndex].value) - minValue) / (maxValue - minValue)]) 
                          if str(self.items[v][colorIndex].value) != '?' else 
                          (v, self.discPalette[0]) for v in nodes)

        elif colorIndex is not None and self.items.domain[colorIndex].varType == orange.VarTypes.Discrete:
            colors.update((v, self.discPalette[colorIndices[self.items[v][colorIndex].value]]) for v in nodes)
            
        else:
            colors.update((node, self.discPalette[0]) for node in nodes)
        
        self.networkCurve.set_node_colors(colors)
        self.replot()
        
    def set_label_attributes(self, attributes=None):
        if self.graph is None:
            return 
        
        nodes = self.graph.nodes()
        
        if attributes is not None:
            self.node_label_attributes = attributes
        
        label_attributes = []
        if self.items is not None and isinstance(self.items, orange.ExampleTable):
            label_attributes = [self.items.domain[att] for att in \
                self.node_label_attributes if att in self.items.domain]
            
        indices = [[] for u in nodes]
        if self.show_indices:
            indices = [[str(u)] for u in nodes]
            
        self.networkCurve.set_node_labels(dict((node, ', '.join(indices[i] + \
                           [str(self.items[node][att]) for att in \
                           label_attributes])) for i, node in enumerate(nodes)))
        
        self.replot()
        
    def set_tooltip_attributes(self, attributes):
        if self.graph is None or self.items is None or \
           not isinstance(self.items, orange.ExampleTable):
            return
        
        tooltip_attributes = [self.items.domain[att] for att in \
                                 attributes if att in self.items.domain]
        self.networkCurve.set_node_tooltips(dict((node, ', '.join(str( \
                   self.items[node][att]) for att in tooltip_attributes)) \
                                                        for node in self.graph))
                        
    def setEdgeLabelText(self, attributes):
        self.edgeLabelText = []
        if self.layout is None or self.graph is None or self.items is None:
            return
        
    def change_graph(self, newgraph):
        old_nodes = set(self.graph.nodes_iter())
        new_nodes = set(newgraph.nodes_iter())
        inter_nodes = old_nodes.intersection(new_nodes)
        remove_nodes = list(old_nodes.difference(inter_nodes))
        add_nodes = new_nodes.difference(inter_nodes)
        
        self.graph = newgraph
        
        self.networkCurve.remove_nodes(list(remove_nodes))
        
        nodes = dict((v, NodeItem(v, parent=self.networkCurve)) for v in add_nodes)
        self.networkCurve.add_nodes(nodes)
        nodes = self.networkCurve.nodes()
        
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
        new_edges = self.graph.edges(add_nodes)
        
        if self.links is not None and len(self.links) > 0:
            links = self.links
            links_indices = (row_ind[i + 1][j + 1] for (i, j) in new_edges)
            labels = ([str(row[r].value) for r in range(2, len(row))] for row \
                      in (links[links_index] for links_index in links_indices))
            
            if self.graph.is_directed():
                edges = [EdgeItem(nodes[i], nodes[j],
                    self.graph[i][j].get('weight', 1), 0, 1, links_index, label, \
                    parent=self.networkCurve) for ((i, j), links_index, label) in \
                         zip(new_edges, links_indices, labels)]
            else:
                edges = [EdgeItem(nodes[i], nodes[j],
                    self.graph[i][j].get('weight', 1), links_index, label) for \
                    ((i, j), links_index, label) in zip(new_edges, \
                                        links_indices, labels, parent=self.networkCurve)]
        elif self.graph.is_directed():
            edges = [EdgeItem(nodes[i], nodes[j], self.graph[i][j].get('weight', 1), \
                    0, 1, parent=self.networkCurve) for (i, j) in new_edges]
        else:
            edges = [EdgeItem(nodes[i], nodes[j], self.graph[i][j].get('weight', 1), \
                    parent=self.networkCurve) for (i, j) in new_edges]
            
        self.networkCurve.add_edges(edges)
        
    def set_graph(self, graph, curve=None, items=None, links=None):
        self.clear()
        self.vertexDegree = []
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
        vertices = dict((v, NodeItem(v, parent=self.networkCurve)) for v in self.graph)
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
                    graph[i][j].get('weight', 1), links_index, label, parent=self.networkCurve) for \
                    ((i, j), links_index, label) in zip(self.graph.edges(), \
                                                        links_indices, labels)]
        elif self.graph.is_directed():
            edges = [EdgeItem(vertices[i], vertices[j],
                                      graph[i][j].get('weight', 1), 0, 1, parent=self.networkCurve) for (i, j) in self.graph.edges()]
        else:
            edges = [EdgeItem(vertices[i], vertices[j],
                                      graph[i][j].get('weight', 1), parent=self.networkCurve) for (i, j) in self.graph.edges()]
            
        self.networkCurve.set_edges(edges)
        self.networkCurve.update_properties()
        self.replot()
        
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
    
    def update_animations(self):
        OWPlot.use_animations(self)
        self.networkCurve.set_use_animations(True)
                    
    def replot(self):
        self.set_dirty()
        OWPlot.replot(self)
        if hasattr(self, 'networkCurve') and self.networkCurve is not None:
            self.networkCurve.update()