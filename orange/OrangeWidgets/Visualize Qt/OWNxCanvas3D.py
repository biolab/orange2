from plot.owplot3d import OWPlot3D, GL_FLOAT, GL_LINES, GL_POINTS, glEnable, GL_PROGRAM_POINT_SIZE
from plot.owopenglrenderer import VertexBuffer
from PyQt4.QtCore import Qt
from PyQt4.QtGui import QVBoxLayout
from PyQt4 import QtOpenGL
import orangeqt
import Orange
import orange
import numpy
import math
import os

from orngScaleScatterPlotData import getVariableValueIndices

class Node3D(orangeqt.Node3D):
    # TODO: __slot__
    def __init__(self, index, x=None, y=None, z=None):
        orangeqt.Node3D.__init__(self, index, 0, Qt.blue, 5)
        if x is not None:
            self.set_x(x)
        if y is not None:
            self.set_y(y)
        if z is not None:
            self.set_z(z)

class Edge3D(orangeqt.Edge3D):
    def __init__(self, u=None, v=None, weight=1, links_index=0, arrows=None, label=''):
        orangeqt.Edge3D.__init__(self, u, v)
        self.set_weight(weight)
        self.set_links_index(links_index)
        if arrows is not None:
            self.set_arrows(arrows)
        if label:
            self.set_label(label)

class OWNxCanvas3D(orangeqt.Canvas3D):
    def __init__(self, master, parent=None, name='None'):
        orangeqt.Canvas3D.__init__(self, parent)

        layout = QVBoxLayout()
        self.plot = OWPlot3D(self)
        layout.addWidget(self.plot)
        self.setLayout(layout)
        self.plot.initializeGL()
        self.plot.before_draw_callback = self.draw_callback
        self.plot.replot = self.plot.update
        self.gui = self.plot.gui
        self.saveToFile = self.plot.save_to_file

        # A little workaround, since NetExplorer sometimes calls networkCurve directly
        self.networkCurve = self

        self.Node3D = Node3D
        self.replot = self.update
        self.plot.animate_plot = False
        self.plot.animate_points = False
        self.plot.antialias_plot = False
        self.plot.auto_adjust_performance = False

        self.master = master
        self.parent = parent
        self.graph = None

        self.circles = []
        self.freeze_neighbours = False
        self.labels_on_marked_only = 0

        self.show_indices = False
        self.show_weights = False
        self.trim_label_words = 0
        self.explore_distances = False
        self.show_component_distances = False

        self.show_component_attribute = None
        self.force_vectors = None
        self.font_size = 12

        self.min_component_edge_width = 0
        self.max_component_edge_width = 0
        self.items_matrix = None

        self.items = None
        self.links = None
        self.edge_to_row = None

        self.node_label_attributes = []
        self.edge_label_attributes = []

        self.axis_margin = 0
        self.title_margin = 0
        self.graph_margin = 1

        self._markers = []

    def draw_callback(self):
        if not hasattr(self, '_edge_shader'):
            self._edge_shader = QtOpenGL.QGLShaderProgram()
            self._edge_shader.addShaderFromSourceFile(QtOpenGL.QGLShader.Vertex,
                os.path.join(os.path.dirname(__file__), 'edge.vs'))
            self._edge_shader.addShaderFromSourceFile(QtOpenGL.QGLShader.Fragment,
                os.path.join(os.path.dirname(__file__), 'edge.fs'))

            self._edge_shader.bindAttributeLocation('position', 0)

            if not self._edge_shader.link():
                print('Failed to link edge shader!')

            self._node_shader = QtOpenGL.QGLShaderProgram()
            self._node_shader.addShaderFromSourceFile(QtOpenGL.QGLShader.Vertex,
                os.path.join(os.path.dirname(__file__), 'node.vs'))
            self._node_shader.addShaderFromSourceFile(QtOpenGL.QGLShader.Fragment,
                os.path.join(os.path.dirname(__file__), 'node.fs'))

            self._node_shader.bindAttributeLocation('position', 0)
            self._node_shader.bindAttributeLocation('offset', 1)
            self._node_shader.bindAttributeLocation('color', 2)
            self._node_shader.bindAttributeLocation('selected_marked', 3)

            if not self._node_shader.link():
                print('Failed to link node shader!')

        self._edge_shader.bind()
        self._edge_shader.setUniformValue('projection', self.plot.projection)
        self._edge_shader.setUniformValue('model', self.plot.model)
        self._edge_shader.setUniformValue('view', self.plot.view)
        self._edge_shader.setUniformValue('translation', self.plot.plot_translation)
        self._edge_shader.setUniformValue('scale', self.plot.plot_scale)
        orangeqt.Canvas3D.draw_edges(self)
        self._edge_shader.release()

        self._node_shader.bind()
        self._node_shader.setUniformValue('projection', self.plot.projection)
        self._node_shader.setUniformValue('model', self.plot.model)
        self._node_shader.setUniformValue('view', self.plot.view)
        self._node_shader.setUniformValue('translation', self.plot.plot_translation)
        self._node_shader.setUniformValue('scale', self.plot.plot_scale)
        self._node_shader.setUniformValue('mode', 0.)
        orangeqt.Canvas3D.draw_nodes(self)
        self._node_shader.release()

    def update_canvas(self):
        self.update_component_keywords()
        self.update()

    def hide_selected_nodes(self):
        orangeqt.Canvas3D.hide_selected_nodes(self)
        self.draw_plot_items()

    def hide_unselected_nodes(self):
        orangeqt.Canvas3D.hide_unselected_nodes(self)
        self.draw_plot_items()

    def show_all_vertices(self):
        orangeqt.Canvas3D.show_all_vertices(self)
        self.draw_plot_items()

    def selected_nodes(self):
        return [node.index() for node in self.nodes().itervalues() if node.is_selected()]

    def not_selected_nodes(self):
        return [node.index() for node in self.nodes().itervalues() if not node.is_selected()]

    def marked_nodes(self):
        return [node.index() for node in self.nodes().itervalues() if node.is_marked()]

    def not_marked_nodes(self):
        return [node.index() for node in self.nodes().itervalues() if not node.is_marked()]    

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

        orangeqt.Canvas3D.clear_node_marks(self)
        orangeqt.Canvas3D.set_node_marks(self, dict((i, True) for i in toMark))

    def mark_on_focus_changed(self, node):
        orangeqt.Canvas3D.clear_node_marks(self)
        if node is not None:
            toMark = set(self.get_neighbors_upto(node.index(), self.mark_neighbors))
            orangeqt.Canvas3D.set_node_marks(self, dict((i, True) for i in toMark))

    def update_component_keywords(self):
        if self.show_component_attribute == None or self.graph is None or self.items is None:
            return

        if str(self.show_component_attribute) not in self.items.domain:
            self.show_component_attribute = None
            return

        components = Orange.network.nx.algorithms.components.connected_components(self.graph)
        nodes = self.nodes()

        self._markers = []

        for c in components:
            if len(c) == 0:
                continue

            x1 = sum(nodes[n].x() for n in c) / len(c)
            y1 = sum(nodes[n].y() for n in c) / len(c)
            z1 = sum(nodes[n].z() for n in c) / len(c)
            lbl = str(self.items[c[0]][str(self.show_component_attribute)])

            self._markers.append((lbl, x1, y1, z1))
            #self.add_marker(lbl, x1, y1, alignment=Qt.AlignCenter, size=self.font_size)

    def get_color_indices(self, table, attribute, palette):
        colorIndices = {}
        colorIndex = None
        minValue = None
        maxValue = None

        if attribute[0] != "(" or attribute[-1] != ")":
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

    getColorIndeces = get_color_indices

    def set_node_colors(self, attribute, nodes=None):
        if self.graph is None:
            return

        colorIndices, colorIndex, minValue, maxValue = self.get_color_indices(self.items, attribute, self.discPalette)
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

        orangeqt.Canvas3D.set_node_colors(self, colors)
        self.update()

    def set_node_labels(self, attributes=None):
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
            orangeqt.Canvas3D.set_node_labels(self, dict((node, 
                ', '.join(indices[i] + 
                          [' '.join(str(self.items[node][att]).split(' ')[:min(self.trim_label_words,len(str(self.items[node][att]).split(' ')))])
                for att in label_attributes])) for i, node in enumerate(nodes)))
        else:
            orangeqt.Canvas3D.set_node_labels(self, dict((node, ', '.join(indices[i]+\
                           [str(self.items[node][att]) for att in \
                           label_attributes])) for i, node in enumerate(nodes)))
        self.update()

    def set_edge_colors(self, attribute):
        if self.graph is None:
            return

        colorIndices, colorIndex, minValue, maxValue = self.getColorIndeces(self.links, attribute, self.discPalette)
        colors = []
        
        if colorIndex is not None and self.links.domain[colorIndex].varType == orange.VarTypes.Continuous and minValue == maxValue:
            colors = [self.discEdgePalette[0] for edge in orangeqt.Canvas3D.edge_indices(self)]
        
        elif colorIndex is not None and self.links.domain[colorIndex].varType == orange.VarTypes.Continuous:
            colors = [self.contPalette[(float(self.links[edge.links_index()][colorIndex].value) - minValue) / (maxValue - minValue)]
                          if str(self.links[edge.links_index()][colorIndex].value) != '?' else 
                          self.discPalette[0] for edge in orangeqt.Canvas3D.edges(self)]
            
        elif colorIndex is not None and self.links.domain[colorIndex].varType == orange.VarTypes.Discrete:
            colors = [self.discEdgePalette[colorIndices[self.links[edge.links_index()][colorIndex].value]] for edge in orangeqt.Canvas3D.edges(self)]
            
        else:
            colors = [self.discEdgePalette[0] for edge in orangeqt.Canvas3D.edge_indices(self)]
            
        orangeqt.Canvas3D.set_edge_colors(self, colors)
        self.update()

    def set_edge_labels(self, attributes=None):
        if self.graph is None:
            return 

        edges = self.edge_indices()

        if attributes is not None:
            self.edge_label_attributes = attributes

        label_attributes = []
        if self.links is not None and isinstance(self.links, orange.ExampleTable):
            label_attributes = [self.links.domain[att] for att in \
                self.edge_label_attributes if att in self.links.domain]

        weights = [[] for ex in edges]
        if self.show_weights:
            weights = [["%.2f" % self.graph[u][v].get('weight', 1)] for u,v in edges]

        orangeqt.Canvas3D.set_edge_labels(self,
            [', '.join(weights[i] + [str(self.links[i][att]) for att in label_attributes]) for i,edge in enumerate(edges)])

        self.update()

    def set_tooltip_attributes(self, attributes):
        if self.graph is None or self.items is None or \
           not isinstance(self.items, orange.ExampleTable):
            return

        tooltip_attributes = [self.items.domain[att] for att in attributes if att in self.items.domain]
        orangeqt.Canvas3D.set_node_tooltips(self,
            dict((node, ', '.join(str(self.items[node][att]) for att in tooltip_attributes)) for node in self.graph))

    def change_graph(self, newgraph):
        old_nodes = set(self.graph.nodes_iter())
        new_nodes = set(newgraph.nodes_iter())
        inter_nodes = old_nodes.intersection(new_nodes)
        remove_nodes = list(old_nodes.difference(inter_nodes))
        add_nodes = list(new_nodes.difference(inter_nodes))

        self.graph = newgraph

        if len(remove_nodes) == 0 and len(add_nodes) == 0:
            return False

        current_nodes = self.nodes()

        def closest_nodes_with_pos(nodes):
            neighbors = set()
            for n in nodes:
                neighbors |= set(self.graph.neighbors(n))

            # checked all, none found            
            if len(neighbors-nodes) == 0:
                return []

            inter = old_nodes.intersection(neighbors)
            if len(inter) > 0:
                return inter
            else:
                print "in recursion"
                return closest_nodes_with_pos(neighbors)

        pos = dict((n, [numpy.average(c) for c in zip(*[(current_nodes[u].x(), current_nodes[u].y()) for u in closest_nodes_with_pos(set([n]))])]) for n in add_nodes)

        orangeqt.Canvas3D.remove_nodes(list(remove_nodes))

        nodes = dict((v, self.Node3D(v, x=pos[v][0], y=pos[v][1])) for v in add_nodes)
        self.add_nodes(nodes)
        nodes = self.nodes()

        #add edges
        new_edges = self.graph.edges(add_nodes)

        if self.links is not None and len(self.links) > 0:
            links_indices = (self.edge_to_row[i + 1][j + 1] for (i, j) in new_edges)
            
            if self.graph.is_directed():
                edges = [Edge3D(nodes[i], nodes[j],
                    self.graph[i][j].get('weight', 1), links_index, arrows=Edge3D.ArrowV, \
                    ) for ((i, j), links_index) in \
                         zip(new_edges, links_indices)]
            else:
                edges = [Edge3D(nodes[i], nodes[j],
                    self.graph[i][j].get('weight', 1), links_index) for \
                    ((i, j), links_index) in zip(new_edges, \
                                        links_indices)]
        elif self.graph.is_directed():
            edges = [Edge3D(nodes[i], nodes[j], self.graph[i][j].get('weight', 1), \
                    arrows=Edge3D.ArrowV) for (i, j) in new_edges]
        else:
            edges = [Edge3D(nodes[i], nodes[j], self.graph[i][j].get('weight', 1), \
                    ) for (i, j) in new_edges]

        self.add_edges(edges)
        return True

    def set_graph(self, graph, curve=None, items=None, links=None):
        # TODO: clear previous nodes and edges?

        if graph is None:
            self.graph = None
            self.items = None
            self.links = None
            xMin = -1.0
            xMax = 1.0
            yMin = -1.0
            yMax = 1.0
            zMin = -1.0
            zMax = 1.0
            self._markers.append(('no network', (xMax - xMin) / 2, (yMax - yMin) / 2, (zMax - zMin) / 2))
            self.update()
            return

        self.graph = graph
        self.items = items if items is not None else self.graph.items()
        self.links = links if links is not None else self.graph.links()

        nodes = dict((v, self.Node3D(v)) for v in self.graph)
        orangeqt.Canvas3D.set_nodes(self, nodes)

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
                    print('Could not find edge ' + str(u) + '-' + str(v))

        if self.links is not None and len(self.links) > 0:
            links = self.links
            links_indices = (self.edge_to_row[i + 1][j + 1] for (i, j) in self.graph.edges())

            if self.graph.is_directed():
                edges = [Edge3D(nodes[i], nodes[j],
                    graph[i][j].get('weight', 1), links_index, arrows=Edge3D.ArrowV) for ((i, j), links_index) in zip(self.graph.edges(), links_indices)]
            else:
                edges = [Edge3D(nodes[i], nodes[j],
                    graph[i][j].get('weight', 1), links_index) for ((i, j), links_index) in zip(self.graph.edges(), links_indices)]
        elif self.graph.is_directed():
            edges = [Edge3D(nodes[i], nodes[j],
                                      graph[i][j].get('weight', 1), arrows=Edge3D.ArrowV) for (i, j) in self.graph.edges()]
        else:
            edges = [Edge3D(nodes[i], nodes[j],
                                      graph[i][j].get('weight', 1)) for (i, j) in self.graph.edges()]

        self.set_edges(edges)
        self._nodes = nodes # Store references, so these objects are not destroyed
        self._edges = edges
        self.update()  

    def update_animations(self, use_animations=None):
        orangeqt.Canvas3D.set_use_animations(self, self.use_animations)

    def clear_node_marks(self):
        orangeqt.Canvas3D.clear_node_marks(self)

    def set_node_marks(self, d):
        orangeqt.Canvas3D.set_node_marks(self, d)

    def set_node_coordinates(self, positions):
        orangeqt.Canvas3D.set_node_coordinates(self, positions)

    def random(self):
        orangeqt.Canvas3D.random(self)

    def circular(self, layout):
        orangeqt.Canvas3D.circular(self, layout)

    def set_labels_on_marked_only(self, labels_on_marked_only):
        orangeqt.Canvas3D.set_labels_on_marked_only(self, labels_on_marked_only)
        self.update()

    def set_show_component_distances(self):
        orangeqt.Canvas3D.set_show_component_distances(self.show_component_distances)
        self.update()

    def layout_fr(self, steps, weighted=False, smooth_cooling=False):
        orangeqt.Canvas3D.fr(self, steps, weighted, smooth_cooling)

    def set_node_sizes(self, values={}, min_size=0, max_size=0):
        orangeqt.Canvas3D.set_node_sizes(self, values, min_size, max_size)

    def fragviz_callback(self, a, b, mds, mds_refresh, components, progress_callback):
        """Refresh the UI when running  MDS on network components."""
        # TODO
        if not self.mdsStep % mds_refresh:
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
            
            self.update()
            qApp.processEvents()
            
            if progress_callback is not None:
                progress_callback(a, self.mdsStep) 
            
        self.mdsStep += 1
        return 0 if self.stopMDS else 1

    def layout_fragviz(self, steps, distances, graph, progress_callback=None, opt_from_curr=False):
        """Position the network components according to similarities among them.
        """
        if distances == None or graph == None or distances.dim != graph.number_of_nodes():
            self.information('invalid or no distance matrix')
            return 1

        #edges = self.edges()
        nodes = self.nodes()

        avgLinkage = True
        #rotationOnly = False
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

        self.set_node_coordinates(dict(
           (n, (nodes[n].x()*d_mds/d_fr, nodes[n].y()*d_mds/d_fr)) for n in nodes))

        self.update()
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

        return 0

    def mds_callback(self, a, b, mds, mdsRefresh, progress_callback):
        """Refresh the UI when running  MDS."""

        if not self.mdsStep % mdsRefresh:
            self.set_node_coordinates(dict((u, (mds.points[u][0], \
                                                mds.points[u][1])) for u in \
                                           range(len(mds.points))))
            self.update()
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

        self.set_node_coordinates(dict(
           (n, (nodes[n].x()*d_mds/d_fr, nodes[n].y()*d_mds/d_fr)) for n in nodes))

        self.update()
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

        return 0

    def update(self):
        orangeqt.Canvas3D.update(self)
        self.plot.update()

