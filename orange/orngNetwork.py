import math
import numpy
import orange
import orangeom
import orngMDS
import os.path

class NetworkOptimization(orangeom.NetworkOptimization):
    def __init__(self, network=None, name="None"):
        if network is None:
            network = orangeom.Network(2, 0)
            
        self.setGraph(network)
        self.graph = network
        
        self.maxWidth = 1000
        self.maxHeight = 1000
        
        self.attributeList = {}
        self.attributeValues = {}
        self.vertexDistance = None
        
    def collapse(self):
        if len(self.graph.getNodes(1)) > 0:
            nodes = list(set(range(self.graph.nVertices)) - set(self.graph.getNodes(1)))
                
            if len(nodes) > 0:
                subgraph = orangeom.Network(self.graph.getSubGraph(nodes))
                oldcoors = self.coors
                self.setGraph(subgraph)
                self.graph = subgraph
                    
                for i in range(len(nodes)):
                    self.coors[0][i] = oldcoors[0][nodes[i]]
                    self.coors[1][i] = oldcoors[1][nodes[i]]

        else:
            fullgraphs = self.graph.getLargestFullGraphs()
            subgraph = self.graph
            
            if len(fullgraphs) > 0:
                used = set()
                graphstomerge = list()
                #print fullgraphs
                for fullgraph in fullgraphs:
                    #print fullgraph
                    fullgraph_set = set(fullgraph)
                    if len(used & fullgraph_set) == 0:
                        graphstomerge.append(fullgraph)
                        used |= fullgraph_set
                        
                #print graphstomerge
                #print used
                subgraph = orangeom.Network(subgraph.getSubGraphMergeClusters(graphstomerge))
                                   
                nodescomp = list(set(range(self.graph.nVertices)) - used)
                
                #subgraph.setattr("items", self.graph.items.getitems(nodescomp))
                #subgraph.items.append(self.graph.items[0])
                oldcoors = self.coors
                self.setGraph(subgraph)
                self.graph = subgraph
                for i in range(len(nodescomp)):
                    self.coors[0][i] = oldcoors[0][nodescomp[i]]
                    self.coors[1][i] = oldcoors[1][nodescomp[i]]
                    
                # place meta vertex in center of cluster    
                x, y = 0, 0
                for node in used:
                    x += oldcoors[0][node]
                    y += oldcoors[1][node]
                    
                x = x / len(used)
                y = y / len(used)
                
                self.coors[0][len(nodescomp)] = x
                self.coors[1][len(nodescomp)] = y
            
    def getVars(self):
        vars = []
        if (self.graph != None):
            if hasattr(self.graph, "items"):
                if isinstance(self.graph.items, orange.ExampleTable):
                    vars[:0] = self.graph.items.domain.variables
                
                    metas = self.graph.items.domain.getmetas(0)
                    for i, var in metas.iteritems():
                        vars.append(var)
        return vars
    
    def getEdgeVars(self):
        vars = []
        if (self.graph != None):
            if hasattr(self.graph, "links"):
                if isinstance(self.graph.links, orange.ExampleTable):
                    vars[:0] = self.graph.links.domain.variables
                
                    metas = self.graph.links.domain.getmetas(0)
                    for i, var in metas.iteritems():
                        vars.append(var)
                        
        return [x for x in vars if str(x.name) != 'u' and str(x.name) != 'v']
    
    def getData(self, i, j):
        if self.graph.items is orange.ExampleTable:
            return self.data[i][j]
        elif self.graph.data is type([]):
            return self.data[i][j]
        
    def nVertices(self):
        if self.graph:
            return self.graph.nVertices
        
    def rotateVertices(self, components, M, factor=10):    
        for i in range(len(components)):
            if M[i] == 0:
                continue
            
            component = components[i]
            
            x = self.graph.coors[0][component]
            y = self.graph.coors[1][component]
            
            x_center = x.mean()
            y_center = y.mean()
            
            x = x - x_center
            y = y - y_center
            
            r = numpy.sqrt(x ** 2 + y ** 2)
            fi = numpy.arctan2(y, x)
                
            fi += factor * M[i] * numpy.pi / 180
                
            x = r * numpy.cos(fi)
            y = r * numpy.sin(fi)
            
            self.graph.coors[0][component] = x + x_center
            self.graph.coors[1][component] = y + y_center 
            
    def rotateComponents(self, factor=10, callbackProgress=None, callbackUpdateCanvas=None, maxSteps=100, minMoment=0.01):
        if self.vertexDistance == None:
            return 1
        
        if self.graph == None:
            return 1
        
        if self.vertexDistance.dim != self.graph.nVertices:
            return 1
        
        self.stopRotate = 0
        
        # rotate only components with more than one vertex
        components = [component for component in self.graph.getConnectedComponents() if len(component) > 1]
        vertices = set(range(self.graph.nVertices))
        step = 0
        M = [1]
        while step < maxSteps and max(M) > minMoment and not self.stopRotate:
            M = [0] * len(components) 
            for i in range(len(components)):
                component = components[i]
                
                outer_vertices = vertices - set(component)
                
                x = self.graph.coors[0][component]
                y = self.graph.coors[1][component]
                
                x_center = x.mean()
                y_center = y.mean()
                
                for j in range(len(component)):
                    u = component[j]

                    for v in outer_vertices:
                        d = self.vertexDistance[u, v]
                        u_x = self.graph.coors[0][u]
                        u_y = self.graph.coors[1][u]
                        v_x = self.graph.coors[0][v]
                        v_y = self.graph.coors[1][v]
                        L = [(u_x - v_x), (u_y - v_y)]
                        R = [(u_x - x_center), (u_y - y_center)]
                        e = math.sqrt((v_x - x_center) ** 2 + (v_y - y_center) ** 2)
                        
                        M[i] += (1 - d) / (e ** 2) * numpy.cross(R, L)
                
            self.rotateVertices(components, M, factor)
            if callbackUpdateCanvas: callbackUpdateCanvas()
            if callbackProgress : callbackProgress()
            step += 1
    
    def mdsUpdateData(self, components, mds, mdsFactor, callbackUpdateCanvas):
        component_props = []
        x_mds = []
        y_mds = []
        
        for i in range(len(components)):
            component = components[i]
            
            # if we did average linkage before
            if len(mds.points) == len(components):
                x_avg_mds = mds.points[i][0]
                y_avg_mds = mds.points[i][1]
            else:
                x = [mds.points[u][0] for u in component]
                y = [mds.points[u][1] for u in component]
        
                x_avg_mds = sum(x) / len(x) 
                y_avg_mds = sum(y) / len(y)
            
            x = self.graph.coors[0][component]
            y = self.graph.coors[1][component]
            
            x_avg_graph = sum(x) / len(x)
            y_avg_graph = sum(y) / len(y)
            
            x_mds.append(x_avg_mds) 
            y_mds.append(y_avg_mds)

            component_props.append((x_avg_graph, y_avg_graph, x_avg_mds, y_avg_mds))
        
        diag_mds =  math.sqrt((max(x_mds) - min(x_mds))**2 + (max(y_mds) - min(y_mds))**2)
         

        for i in range(len(components)):
            component = components[i]
            x_avg_graph, y_avg_graph, x_avg_mds, y_avg_mds = component_props[i]
            
            self.graph.coors[0][component] = self.graph.coors[0][component] - x_avg_graph + (x_avg_mds *  mdsFactor * self.diag_coors / diag_mds)
            self.graph.coors[1][component] = self.graph.coors[1][component] - y_avg_graph + (y_avg_mds *  mdsFactor * self.diag_coors / diag_mds)
        
        if callbackUpdateCanvas:
            callbackUpdateCanvas()
    
    def mdsCallback(self, a,b=None):
        if not self.mdsStep % self.mdsRefresh:
            self.mdsUpdateData(self.mdsComponents, self.mds, self.mdsFactor, self.callbackUpdateCanvas)
            
            if self.callbackProgress != None:
                self.callbackProgress(self.mds.avgStress, self.mdsStep)
        
        self.mdsStep += 1

        if self.stopMDS:
            return 0
        else:
            return 1
            
    def mdsComponents(self, mdsSteps, mdsRefresh, mdsFactor, callbackProgress=None, callbackUpdateCanvas=None, torgerson=0, minStressDelta = 0, avgLinkage=False):
        if self.vertexDistance == None:
            self.information('Set distance matrix to input signal')
            return 1
        
        if self.graph == None:
            return 1
        
        if self.vertexDistance.dim != self.graph.nVertices:
            return 1
        
        self.mdsComponents = self.graph.getConnectedComponents()
        self.mdsRefresh = mdsRefresh
        self.mdsStep = 0
        self.stopMDS = 0
        self.vertexDistance.matrixType = orange.SymMatrix.Symmetric
        self.diag_coors = math.sqrt((min(self.graph.coors[0]) - max(self.graph.coors[0]))**2 + (min(self.graph.coors[1]) - max(self.graph.coors[1]))**2)
        
        if avgLinkage:
            matrix = self.vertexDistance.avgLinkage(self.mdsComponents)
        else:
            matrix = self.vertexDistance
            
        self.mds = orngMDS.MDS(matrix)
        # set min stress difference between 0.01 and 0.00001
        self.minStressDelta = minStressDelta
        self.mdsFactor = mdsFactor
        self.callbackUpdateCanvas = callbackUpdateCanvas
        self.callbackProgress = callbackProgress
        
        if torgerson:
            self.mds.Torgerson() 
        
        self.mds.optimize(mdsSteps, orngMDS.SgnRelStress, self.minStressDelta, progressCallback=self.mdsCallback)
        if self.mdsStep % self.mdsRefresh:
            self.mdsUpdateData(self.mdsComponents, self.mds, mdsFactor, callbackUpdateCanvas)
        
            if callbackProgress != None:
                callbackProgress(self.mds.avgStress, self.mdsStep)
        
        del self.diag_coors
        del self.mdsRefresh
        del self.mdsStep
        del self.mds
        del self.mdsComponents
        del self.minStressDelta
        del self.mdsFactor
        del self.callbackUpdateCanvas
        del self.callbackProgress
        return 0

    def mdsComponentsAvgLinkage(self, mdsSteps, mdsRefresh, mdsFactor, callbackProgress=None, callbackUpdateCanvas=None, torgerson=0, minStressDelta = 0):
        return self.mdsComponents(mdsSteps, mdsRefresh, mdsFactor, callbackProgress, callbackUpdateCanvas, torgerson, minStressDelta, True)

    def saveNetwork(self, fn):
        name = ''
        try:
            root, ext = os.path.splitext(fn)
            if ext == '':
                fn = root + '.net'
            
            graphFile = file(fn, 'w+')
        except IOError:
            return 1

        graphFile.write('### This file was generated with Orange Network Visualizer ### \n\n\n')
        if name == '':
            graphFile.write('*Network ' + '"no name" \n\n')
        else:
            graphFile.write('*Network ' + str(name) + ' \n\n')


        #izpis opisov vozlisc
        graphFile.write('*Vertices% 8d\n' % self.graph.nVertices)
        for v in range(self.graph.nVertices):
            graphFile.write('% 8d ' % (v + 1))
#            if verticesParms[v].label!='':
#                self.GraphFile.write(str('"'+ verticesParms[v].label + '"') + ' \t')
#            else:
            try:
                label = self.graph.items[v]['label']
                graphFile.write(str('"' + str(label) + '"') + ' \t')
            except:
                graphFile.write(str('"' + str(v) + '"') + ' \t')
            
            x = self.network.coors[0][v]
            y = self.network.coors[1][v]
            #if x < 0: x = 0
            #if x >= 1: x = 0.9999
            #if y < 0: y = 0
            #if y >= 1: y = 0.9999
            z = 0.5000
            graphFile.write('%.4f    %.4f    %.4f\t' % (x, y, z))
            graphFile.write('\n')

        #izpis opisov povezav
        #najprej neusmerjene
        if self.graph.directed:
            graphFile.write('*Arcs \n')
            for (i, j) in self.graph.getEdges():
                if len(self.graph[i, j]) > 0:
                    graphFile.write('%8d %8d %f' % (i + 1, j + 1, float(str(self.graph[i, j]))))
                    graphFile.write('\n')
        else:
            graphFile.write('*Edges \n')
            for (i, j) in self.graph.getEdges():
                if len(self.graph[i, j]) > 0:
                    graphFile.write('%8d %8d %f' % (i + 1, j + 1, float(str(self.graph[i, j]))))
                    graphFile.write('\n')

        graphFile.write('\n')
        graphFile.close()
        
        if self.graph.items != None and len(self.graph.items) > 0:
            (name, ext) = os.path.splitext(fn)
            self.graph.items.save(name + "_items.tab")
            
        if self.graph.links != None and len(self.graph.links) > 0:
            (name, ext) = os.path.splitext(fn)
            self.graph.links.save(name + "_links.tab")

        return 0
    
    def readNetwork(self, fn, directed=0):
        graph = orangeom.NetworkOptimization.readNetwork(self, fn, directed)
        self.setGraph(graph)
        self.graph = graph
        return graph
    
