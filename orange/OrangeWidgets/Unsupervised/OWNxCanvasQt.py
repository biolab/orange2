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
import numpy

from plot.owplot import *
from plot.owpoint import *
from plot.owtools import *  

from orngScaleScatterPlotData import *
import orangeqt

class NodeItem(orangeqt.NodeItem):
    def __init__(self, index, x=None, y=None, parent=None):
        orangeqt.NodeItem.__init__(self, index, OWPoint.Ellipse, Qt.blue, 5, parent)
        if x is not None:
            self.set_x(x)
        if y is not None:
            self.set_y(y)
        
class EdgeItem(orangeqt.EdgeItem):
    def __init__(self, u=None, v=None, weight=1, links_index=0, arrows=None, label='', parent=None):
        orangeqt.EdgeItem.__init__(self, u, v, parent)
        self.set_weight(weight)
        self.set_links_index(links_index)
        if arrows is not None:
            self.set_arrows(arrows)

class NetworkCurve(orangeqt.NetworkCurve):
    def __init__(self, parent=None, pen=QPen(Qt.black), xData=None, yData=None):
        orangeqt.NetworkCurve.__init__(self, parent)
        self.name = "Network Curve"
        
    def layout_fr(self, steps, weighted=False, smooth_cooling=False):
        orangeqt.NetworkCurve.fr(self, steps, weighted, smooth_cooling)
      
    def set_node_sizes(self, values={}, min_size=0, max_size=0):
        orangeqt.NetworkCurve.set_node_sizes(self, values, min_size, max_size)
    
    def fragviz_callback(self, a, b, mds, mdsRefresh, components, progress_callback):
        """Refresh the UI when running  MDS on network components."""
        
        if not self.mdsStep % mdsRefresh:
            rotationOnly = False
            component_props = []
            x_mds = []
            y_mds = []
            phi = [None] * len(components)
            nodes = self.nodes()
            
            for i, component in enumerate(components):    
                
                if len(mds.points) == len(components):  # if average linkage before
                    x_avg_mds = mds.points[i][0]
                    y_avg_mds = mds.points[i][1]
                else:                                   # if not average linkage before
                    x = [mds.points[u][0] for u in component]
                    y = [mds.points[u][1] for u in component]
            
                    x_avg_mds = sum(x) / len(x) 
                    y_avg_mds = sum(y) / len(y)
                    # compute rotation angle
#                    c = [numpy.linalg.norm(numpy.cross(mds.points[u], \
#                                [nodes[u].x(), nodes[u].y()])) for u in component]
#                    
#                    n = [numpy.vdot([nodes[u].x(), nodes[u].y()], \
#                                    [nodes[u].x(), nodes[u].y()]) for u in component]
#                    phi[i] = sum(c) / sum(n)
                    
                
                x = [nodes[i].x() for i in component]
                y = [nodes[i].y() for i in component]
                
                x_avg_graph = sum(x) / len(x)
                y_avg_graph = sum(y) / len(y)
                
                x_mds.append(x_avg_mds) 
                y_mds.append(y_avg_mds)
    
                component_props.append((x_avg_graph, y_avg_graph, \
                                        x_avg_mds, y_avg_mds, phi))

            for i, component in enumerate(components):
                x_avg_graph, y_avg_graph, x_avg_mds, \
                y_avg_mds, phi = component_props[i]
                
    #            if phi[i]:  # rotate vertices
    #                #print "rotate", i, phi[i]
    #                r = numpy.array([[numpy.cos(phi[i]), -numpy.sin(phi[i])], [numpy.sin(phi[i]), numpy.cos(phi[i])]])  #rotation matrix
    #                c = [x_avg_graph, y_avg_graph]  # center of mass in FR coordinate system
    #                v = [numpy.dot(numpy.array([self.graph.coors[0][u], self.graph.coors[1][u]]) - c, r) + c for u in component]
    #                self.graph.coors[0][component] = [u[0] for u in v]
    #                self.graph.coors[1][component] = [u[1] for u in v]
                    
                # translate vertices
                if not rotationOnly:
                    self.set_node_coordinates(dict(
                       (i, ((nodes[i].x() - x_avg_graph) + x_avg_mds,  
                            (nodes[i].y() - y_avg_graph) + y_avg_mds)) \
                                  for i in component))
                    
            #if self.mdsType == MdsType.exactSimulation:
            #    self.mds.points = [[self.graph.coors[0][i], \
            #                        self.graph.coors[1][i]] \
            #                        for i in range(len(self.graph.coors))]
            #    self.mds.freshD = 0
            
            #self.update_properties()
            self.plot().replot()
            qApp.processEvents()
            
            if progress_callback is not None:
                progress_callback(a, self.mdsStep) 
            
        self.mdsStep += 1
        return 0 if self.stopMDS else 1
            
    def layout_fragviz(self, steps, distances, graph, progress_callback=None, opt_from_curr=False):
        """Position the network components according to similarities among 
        them.
        
        """

        if distances == None or graph == None or distances.dim != graph.number_of_nodes():
            self.information('invalid or no distance matrix')
            return 1
        
        p = self.plot()
        edges = self.edges()
        nodes = self.nodes()
        
        avgLinkage = True
        rotationOnly = False
        minStressDelta = 0
        mdsRefresh = int(steps / 20)
        
        self.mdsStep = 1
        self.stopMDS = False
        
        components = Orange.network.nx.algorithms.components.connected.connected_components(graph)
        distances.matrixType = Orange.core.SymMatrix.Symmetric
        
        # scale net coordinates
        if avgLinkage:
            distances = distances.avgLinkage(components)
            
        mds = Orange.projection.mds.MDS(distances)
        mds.optimize(10, Orange.projection.mds.SgnRelStress, 0)
        rect = self.data_rect()
        w_fr = rect.width()
        h_fr = rect.height()
        d_fr = math.sqrt(w_fr**2 + h_fr**2)
      
        x_mds = [mds.points[u][0] for u in range(len(mds.points))]
        y_mds = [mds.points[u][1] for u in range(len(mds.points))]
        w_mds = max(x_mds) - min(x_mds)
        h_mds = max(y_mds) - min(y_mds)
        d_mds = math.sqrt(w_mds**2 + h_mds**2)
        
        animate_points = p.animate_points
        p.animate_points = False
        
        self.set_node_coordinates(dict(
           (n, (nodes[n].x()*d_mds/d_fr, nodes[n].y()*d_mds/d_fr)) for n in nodes))
        
        #self.update_properties()
        p.replot()
        qApp.processEvents()
                     
        if opt_from_curr:
            if avgLinkage:
                for u, c in enumerate(components):
                    x = sum([nodes[n].x() for n in c]) / len(c)
                    y = sum([nodes[n].y() for n in c]) / len(c)
                    mds.points[u][0] = x
                    mds.points[u][1] = y
            else:
                for i,u in enumerate(sorted(nodes.iterkeys())):
                    mds.points[i][0] = nodes[u].x()
                    mds.points[i][1] = nodes[u].y()
        else:
            mds.Torgerson() 

        mds.optimize(steps, Orange.projection.mds.SgnRelStress, minStressDelta, 
                     progressCallback=
                         lambda a, 
                                b=None, 
                                mds=mds,
                                mdsRefresh=mdsRefresh,
                                components=components,
                                progress_callback=progress_callback: 
                                    self.fragviz_callback(a, b, mds, mdsRefresh, components, progress_callback))
        
        self.mds_callback(mds.avgStress, 0, mds, mdsRefresh, components, progress_callback)
        
        if progress_callback != None:
            progress_callback(mds.avgStress, self.mdsStep)
        
        p.animate_points = animate_points
        return 0
    
    def mds_callback(self, a, b, mds, mdsRefresh, progress_callback):
        """Refresh the UI when running  MDS."""
        
        if not self.mdsStep % mdsRefresh:
            
            self.set_node_coordinates(dict((u, (mds.points[u][0], \
                                                mds.points[u][1])) for u in \
                                           range(len(mds.points))))
            self.plot().replot()
            qApp.processEvents()
            
            if progress_callback is not None:
                progress_callback(a, self.mdsStep) 
            
        self.mdsStep += 1
        return 0 if self.stopMDS else 1
    
    def layout_mds(self, steps, distances, progress_callback=None, opt_from_curr=False):
        """Position the network components according to similarities among 
        them.
        
        """
        nodes = self.nodes()
        
        if distances == None or distances.dim != len(nodes):
            self.information('invalid or no distance matrix')
            return 1
        
        p = self.plot()
        
        minStressDelta = 0
        mdsRefresh = int(steps / 20)
        
        self.mdsStep = 1
        self.stopMDS = False
        
        distances.matrixType = Orange.core.SymMatrix.Symmetric
        mds = Orange.projection.mds.MDS(distances)
        mds.optimize(10, Orange.projection.mds.SgnRelStress, 0)
        rect = self.data_rect()
        w_fr = rect.width()
        h_fr = rect.height()
        d_fr = math.sqrt(w_fr**2 + h_fr**2)
      
        x_mds = [mds.points[u][0] for u in range(len(mds.points))]
        y_mds = [mds.points[u][1] for u in range(len(mds.points))]
        w_mds = max(x_mds) - min(x_mds)
        h_mds = max(y_mds) - min(y_mds)
        d_mds = math.sqrt(w_mds**2 + h_mds**2)
        
        animate_points = p.animate_points
        p.animate_points = False
        
        self.set_node_coordinates(dict(
           (n, (nodes[n].x()*d_mds/d_fr, nodes[n].y()*d_mds/d_fr)) for n in nodes))
        
        p.replot()
        qApp.processEvents()
                     
        if opt_from_curr:
            for i,u in enumerate(sorted(nodes.iterkeys())):
                mds.points[i][0] = nodes[u].x()
                mds.points[i][1] = nodes[u].y()
        else:
            mds.Torgerson() 

        mds.optimize(steps, Orange.projection.mds.SgnRelStress, minStressDelta, 
                     progressCallback=
                         lambda a, 
                                b=None, 
                                mds=mds,
                                mdsRefresh=mdsRefresh,
                                progress_callback=progress_callback: 
                                    self.mds_callback(a, b, mds, mdsRefresh, progress_callback))
        
        self.mds_callback(mds.avgStress, 0, mds, mdsRefresh, progress_callback)
        
        if progress_callback != None:
            progress_callback(mds.avgStress, self.mdsStep)
        
        p.animate_points = animate_points
        return 0
    
#    def move_selected_nodes(self, dx, dy):
#        selected = self.get_selected_nodes()
#        
#        self.coors[selected][0] = self.coors[0][selected] + dx
#        self.coors[1][selected][1] = self.coors[1][selected] + dy
#          
#        self.update_properties()
#        return selected
#        
#    def set_hidden_nodes(self, nodes):
#        for vertex in self.nodes().itervalues():
#            vertex.setVisible(vertex.index() in nodes)
#      
#    def hide_selected_nodes(self):
#        for vertex in self.nodes().itervalues():
#          if vertex.selected:
#            vertex.hide()
#  
#    def hide_unselected_nodes(self):
#        for vertex in self.nodes().itervalues():
#          if not vertex.selected:
#            vertex.hide()
#    
#    def show_all_vertices(self):
#        for vertex in self.nodes().itervalues():
#          vertex.show()
    
    
        
class OWNxCanvas(OWPlot):
    def __init__(self, master, parent=None, name="None"):
        OWPlot.__init__(self, parent, name, axes=[])
        self.master = master
        self.parent = parent
        self.NodeItem = NodeItem
        self.graph = None
        
        self.circles = []
        self.freezeNeighbours = False
        self.labelsOnMarkedOnly = 0

        self.show_indices = False
        self.show_weights = False
        self.trim_label_words = 0
        self.explore_distances = False
        self.show_component_distances = False
        
        self.showComponentAttribute = None
        self.forceVectors = None
        #self.appendToSelection = 1
        self.fontSize = 12
             
        self.networkCurve = NetworkCurve()
        self.add_custom_curve(self.networkCurve)
        
        self.minComponentEdgeWidth = 0
        self.maxComponentEdgeWidth = 0
        self.items_matrix = None
         
        self.items = None
        self.links = None
        self.edge_to_row = None
        
        self.node_label_attributes = []
        self.edge_label_attributes = []
        
        self.axis_margin = 0
        self.title_margin = 0
        self.graph_margin = 1
        self._legend_margin = QRectF(0, 0, 0, 0)
        
        #self.setFocusPolicy(Qt.StrongFocus)
        
    def update_canvas(self):
        self.networkCurve.update_properties()
        self.drawComponentKeywords()
        self.replot()
        
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
    
    def selected_nodes(self):
        return [vertex.index() for vertex in self.networkCurve.nodes().itervalues() if vertex.is_selected()]
        #return [p.index() for p in self.selected_points()]
        
    def not_selected_nodes(self):
        return [vertex.index() for vertex in self.networkCurve.nodes().itervalues() if not vertex.is_selected()]
        
    def marked_nodes(self):
        return [vertex.index() for vertex in self.networkCurve.nodes().itervalues() if vertex.is_marked()]
        #return [p.index() for p in self.marked_points()]
        
    def not_marked_nodes(self):
        return [vertex.index() for vertex in self.networkCurve.nodes().itervalues() if not vertex.is_marked()]    
    
    def get_neighbors_upto(self, ndx, dist):
        newNeighbours = neighbours = set([ndx])
        for d in range(dist):
            tNewNeighbours = set()
            for v in newNeighbours:
                tNewNeighbours |= set(self.graph.neighbors(v))
            newNeighbours = tNewNeighbours - neighbours
            neighbours |= newNeighbours
        return neighbours
    
    def mark_on_selection_changed(self):
        toMark = set()
        for ndx in self.selected_nodes():
            toMark |= self.get_neighbors_upto(ndx, self.mark_neighbors)
        
        self.networkCurve.clear_node_marks()
        self.networkCurve.set_node_marks(dict((i, True) for i in toMark))
    
    def mark_on_focus_changed(self, node):
        self.networkCurve.clear_node_marks()
        
        if node is not None:
            toMark = set(self.get_neighbors_upto(node.index(), self.mark_neighbors))
            self.networkCurve.set_node_marks(dict((i, True) for i in toMark))
        
    def drawComponentKeywords(self):
        self.clear_markers()
        if self.showComponentAttribute == None or self.graph is None or self.items is None:
            return
        
        if str(self.showComponentAttribute) not in self.items.domain:
            self.showComponentAttribute = None
            return
        
        components = Orange.network.nx.algorithms.components.connected_components(self.graph)
        nodes = self.networkCurve.nodes()
        
        for c in components:
            if len(c) == 0:
                continue
            
            x1 = sum(nodes[n].x() for n in c) / len(c)
            y1 = sum(nodes[n].y() for n in c) / len(c)
            lbl = str(self.items[c[0]][str(self.showComponentAttribute)])
            
            self.add_marker(lbl, x1, y1, alignment=Qt.AlignCenter, size=self.fontSize)
            
            #mkey = self.addMarker(lbl, float(x1), float(y1), alignment=Qt.AlignCenter, size=self.fontSize)
                            
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
        
    def set_node_labels(self, attributes=None):
        print "set labels"
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
            
        if self.trim_label_words > 0:
            print "trim"
            self.networkCurve.set_node_labels(dict((node, 
                ', '.join(indices[i] + 
                          [' '.join(str(self.items[node][att]).split(' ')[:min(self.trim_label_words,len(str(self.items[node][att]).split(' ')))])
                for att in label_attributes])) for i, node in enumerate(nodes)))
        else:
            print "no trim"
            self.networkCurve.set_node_labels(dict((node, ', '.join(indices[i]+\
                           [str(self.items[node][att]) for att in \
                           label_attributes])) for i, node in enumerate(nodes)))
        
        self.replot()
        
    
    def set_edge_colors(self, attribute):
        if self.graph is None:
            return
        
        colorIndices, colorIndex, minValue, maxValue = self.getColorIndeces(self.links, attribute, self.discPalette)
        colors = []
        
        if colorIndex is not None and self.links.domain[colorIndex].varType == orange.VarTypes.Continuous and minValue == maxValue:
            colors = [self.discEdgePalette[0] for edge in self.networkCurve.edge_indices()]
        
        elif colorIndex is not None and self.links.domain[colorIndex].varType == orange.VarTypes.Continuous:
            colors = [self.contPalette[(float(self.links[edge.links_index()][colorIndex].value) - minValue) / (maxValue - minValue)]
                          if str(self.links[edge.links_index()][colorIndex].value) != '?' else 
                          self.discPalette[0] for edge in self.networkCurve.edges()]
            
        elif colorIndex is not None and self.links.domain[colorIndex].varType == orange.VarTypes.Discrete:
            colors = [self.discEdgePalette[colorIndices[self.links[edge.links_index()][colorIndex].value]] for edge in self.networkCurve.edges()]
            
        else:
            colors = [self.discEdgePalette[0] for edge in self.networkCurve.edge_indices()]
            
        self.networkCurve.set_edge_colors(colors)
        self.replot()
        
    def set_edge_labels(self, attributes=None):
        if self.graph is None:
            return 
        
        edges = self.networkCurve.edge_indices()
        
        if attributes is not None:
            self.edge_label_attributes = attributes
        
        label_attributes = []
        if self.links is not None and isinstance(self.links, orange.ExampleTable):
            label_attributes = [self.links.domain[att] for att in \
                self.edge_label_attributes if att in self.links.domain]
            
        weights = [[] for ex in edges]
        if self.show_weights:
            weights = [["%.2f" % self.graph[u][v].get('weight', 1)] for u,v in edges]
            
        self.networkCurve.set_edge_labels([', '.join(weights[i] + \
                           [str(self.links[i][att]) for att in \
                           label_attributes]) for i,edge in enumerate(edges)])
        
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
                
    def change_graph(self, newgraph):
        old_nodes = set(self.graph.nodes_iter())
        new_nodes = set(newgraph.nodes_iter())
        inter_nodes = old_nodes.intersection(new_nodes)
        remove_nodes = list(old_nodes.difference(inter_nodes))
        add_nodes = list(new_nodes.difference(inter_nodes))
        
        self.graph = newgraph
        
        if len(remove_nodes) == 0 and len(add_nodes) == 0:
            return False
        
        current_nodes = self.networkCurve.nodes()
        
        center_x = numpy.average([node.x() for node in current_nodes.values()]) if len(current_nodes) > 0 else 0
        center_y = numpy.average([node.y() for node in current_nodes.values()]) if len(current_nodes) > 0 else 0
        
        def closest_nodes_with_pos(nodes):
            
            neighbors = set()
            for n in nodes:
                neighbors |= set(self.graph.neighbors(n))

            # checked all, none found            
            if len(neighbors - nodes) == 0:
                return []
            
            inter = old_nodes.intersection(neighbors)
            if len(inter) > 0:
                return inter
            else:
                return closest_nodes_with_pos(neighbors | nodes)
        
        pos = dict((n, [numpy.average(c) for c in zip(*[(current_nodes[u].x(), current_nodes[u].y()) for u in closest_nodes_with_pos(set([n]))])]) for n in add_nodes)
        
        self.networkCurve.remove_nodes(list(remove_nodes))
        
        nodes = dict((v, self.NodeItem(v, x=pos[v][0] if len(pos[v]) == 2 else center_x, y=pos[v][1] if len(pos[v]) == 2 else center_y, parent=self.networkCurve)) for v in add_nodes)
        self.networkCurve.add_nodes(nodes)
        nodes = self.networkCurve.nodes()
        
        #add edges
        new_edges = self.graph.edges(add_nodes)
        
        if self.links is not None and len(self.links) > 0:
            links = self.links
            links_indices = (self.edge_to_row[i + 1][j + 1] for (i, j) in new_edges)
            
            if self.graph.is_directed():
                edges = [EdgeItem(nodes[i], nodes[j],
                    self.graph[i][j].get('weight', 1), links_index, arrows=EdgeItem.ArrowV, \
                    parent=self.networkCurve) for ((i, j), links_index) in \
                         zip(new_edges, links_indices)]
            else:
                edges = [EdgeItem(nodes[i], nodes[j],
                    self.graph[i][j].get('weight', 1), links_index) for \
                    ((i, j), links_index) in zip(new_edges, \
                                        links_indices, parent=self.networkCurve)]
        elif self.graph.is_directed():
            edges = [EdgeItem(nodes[i], nodes[j], self.graph[i][j].get('weight', 1), \
                    arrows=EdgeItem.ArrowV, parent=self.networkCurve) for (i, j) in new_edges]
        else:
            edges = [EdgeItem(nodes[i], nodes[j], self.graph[i][j].get('weight', 1), \
                    parent=self.networkCurve) for (i, j) in new_edges]
            
        self.networkCurve.add_edges(edges)
        return True
        
    def set_graph(self, graph, curve=None, items=None, links=None):
        self.clear()
        
        if graph is None:
            self.graph = None
            self.networkCurve = None
            self.items = None
            self.links = None
            xMin = -1.0
            xMax = 1.0
            yMin = -1.0
            yMax = 1.0
            self.addMarker("no network", (xMax - xMin) / 2, (yMax - yMin) / 2, alignment=Qt.AlignCenter, size=self.fontSize)
            self.replot()
            return
        
        self.graph = graph
        self.networkCurve = NetworkCurve() if curve is None else curve()
        self.add_custom_curve(self.networkCurve)
        
        self.items = items if items is not None else self.graph.items()
        self.links = links if links is not None else self.graph.links()
        
        #add nodes
        #self.vertices_old = [(None, []) for v in self.graph]
        vertices = dict((v, self.NodeItem(v, parent=self.networkCurve)) for v in self.graph)
        self.networkCurve.set_nodes(vertices)
                
        #build edge to row index
        self.edge_to_row = {}
        if self.links is not None and len(self.links) > 0:
            for i, r in enumerate(self.links):
                u = int(r['u'].value)
                v = int(r['v'].value)
                if u - 1 in self.graph and v - 1 in self.graph:
                    u_dict = self.edge_to_row.get(u, {})
                    v_dict = self.edge_to_row.get(v, {})
                    u_dict[v] = i
                    v_dict[u] = i
                    self.edge_to_row[u] = u_dict
                    self.edge_to_row[v] = v_dict
                else:
                    print 'could not find edge', u, v
              
        #add edges
        if self.links is not None and len(self.links) > 0:
            links = self.links
            links_indices = (self.edge_to_row[i + 1][j + 1] for (i, j) in self.graph.edges())
            
            if self.graph.is_directed():
                edges = [EdgeItem(vertices[i], vertices[j],
                    graph[i][j].get('weight', 1), links_index, arrows=EdgeItem.ArrowV, \
                    parent=self.networkCurve) for ((i, j), links_index) in \
                         zip(self.graph.edges(), links_indices)]
            else:
                edges = [EdgeItem(vertices[i], vertices[j],
                    graph[i][j].get('weight', 1), links_index, \
                    parent=self.networkCurve) for ((i, j), links_index) in \
                         zip(self.graph.edges(), links_indices)]
                
        elif self.graph.is_directed():
            edges = [EdgeItem(vertices[i], vertices[j],
                                      graph[i][j].get('weight', 1), arrows=EdgeItem.ArrowV, parent=self.networkCurve) for (i, j) in self.graph.edges()]
        else:
            edges = [EdgeItem(vertices[i], vertices[j],
                                      graph[i][j].get('weight', 1), parent=self.networkCurve) for (i, j) in self.graph.edges()]
            
        self.networkCurve.set_edges(edges)
        self.networkCurve.update_properties()
        self.replot()  
    
    def update_animations(self, use_animations=None):
        OWPlot.update_animations(self, use_animations)
        self.networkCurve.set_use_animations(self.use_animations)

    def set_labels_on_marked_only(self, labelsOnMarkedOnly):
        self.networkCurve.set_labels_on_marked_only(labelsOnMarkedOnly)
        self.replot()
    
    def set_show_component_distances(self):
        self.networkCurve.set_show_component_distances(self.show_component_distances)
        self.replot()
        
    def replot(self):
        
                #, alignment = -1, bold = 0, color = None, brushColor = None, size=None, antiAlias = None, x_axis_key = xBottom, y_axis_key = yLeft):
        self.set_dirty()
        OWPlot.replot(self)
        if hasattr(self, 'networkCurve') and self.networkCurve is not None:
            self.networkCurve.update()
            
