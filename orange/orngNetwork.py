import math
import random
import numpy
import orange
import orangeom
import os.path
import orngMDS

class NetworkOptimization(orangeom.NetworkOptimization):
    def __init__(self, graph=None, name="None"):
        if graph is None:
            graph = orangeom.Network(2, 0)
            
        self.setGraph(graph)
        self.graph = graph
        
        self.maxWidth  = 1000
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
        elif self.graph.data is List:
            return self.data[i][j]
        
    def nVertices(self):
        if self.graph:
            return self.graph.nVertices
        
    def mdsComponents(self, mdsSteps, mdsRefresh, mdsFactor, callbackProgress=None, callbackUpdateCanvas=None):
        if self.vertexDistance == None:
            self.information('Set distance matrix to input signal')
            return
        
        if self.graph == None:
            return
        
        
        if self.vertexDistance.dim != self.graph.nVertices:
            return
        
        self.vertexDistance.matrixType = orange.SymMatrix.Symmetric
        mds = orngMDS.MDS(self.vertexDistance)
        mds.Torgerson() 
        mds.getStress(orngMDS.KruskalStress)
        components = self.graph.getConnectedComponents()
        
        stepCount = 0 
        while stepCount < mdsSteps: 
            oldStress = mds.avgStress
            mds.getStress(orngMDS.KruskalStress)
            
            for l in range(mdsRefresh):
                stepCount += 1
                mds.SMACOFstep()
                
                if callbackProgress:
                    callbackProgress(mds.avgStress, stepCount)
                
                if stepCount >= mdsSteps:
                    break;
            
            mds.getStress(orngMDS.KruskalStress)
            component_props = []
            
            for component in components:
                x = [mds.points[u][0] for u in component]
                y = [mds.points[u][1] for u in component]
            
                x_avg_mds = sum(x) / len(x) 
                y_avg_mds = sum(y) / len(y)
                
                x = [self.graph.coors[0][u] for u in component]
                y = [self.graph.coors[1][u] for u in component]
                
                x_avg_graph = sum(x) / len(x)
                y_avg_graph = sum(y) / len(y)
                
                graph_range = max([math.sqrt((x[i]-x_avg_graph)*(x[i]-x_avg_graph) + (y[i]-y_avg_graph)*(y[i]-y_avg_graph)) for i in range(len(x))])
                
                component_props.append((x_avg_graph, y_avg_graph, x_avg_mds, y_avg_mds, graph_range))
            
            
            maxrange = 0
            count = 0
            # find min distance between components
            for i in range(1, len(components)):
                for j in range(i - 1):
                    component_i = components[i]
                    component_j = components[j]
                    
                    x_avg_graph_i, y_avg_graph_i, x_avg_mds_i, y_avg_mds_i, graph_range_i = component_props[i]
                    x_avg_graph_j, y_avg_graph_j, x_avg_mds_j, y_avg_mds_j, graph_range_j = component_props[j]
                    
                    graphsdist = graph_range_i + graph_range_j
                    #graphsdist = 1.1 * graphsdist
                    mdsdist = math.sqrt((x_avg_mds_i-x_avg_mds_j)*(x_avg_mds_i-x_avg_mds_j) + (y_avg_mds_i-y_avg_mds_j)*(y_avg_mds_i-y_avg_mds_j))
                    if mdsdist != 0:
                        component_range = graphsdist / mdsdist                
                        maxrange += component_range
                        count += 1
                    
            maxrange = maxrange / count
            for i in range(len(components)):
                component = components[i]
                x_avg_graph, y_avg_graph, x_avg_mds, y_avg_mds, graph_range = component_props[i]
                
                for u in component:
                    self.graph.coors[0][u] = self.graph.coors[0][u] - x_avg_graph + (x_avg_mds * maxrange * mdsFactor)
                    self.graph.coors[1][u] = self.graph.coors[1][u] - y_avg_graph + (y_avg_mds * maxrange * mdsFactor)
            
            if callbackUpdateCanvas:
                callbackUpdateCanvas()
                    
            if oldStress*1e-3 > math.fabs(oldStress-mds.avgStress): 
                break; 
            
    #procedura za razporejanje nepovezanih vozlisc na kroznico okoli grafa
    def postProcess(self):
        UDist=20
        pos1=where(sum(self.graph,1), 0, 1)
        pos2=where(sum(self.graph,0), 0, 1)
        pos=logical_and(pos1, pos2)  #iscemo SAMO TISTA, ki nimajo ne vhodnih ne izhodnih povezav!

        ncCount=sum(pos)
        if ncCount==0:  #ce je graf povezan
            return

        #else:
        #max in min na povezanem delu grafa
        conCoorsX=compress(logical_not(pos), self.xCoors)
        conCoorsY=compress(logical_not(pos), self.yCoors)

        if len(conCoorsX)==0:  #ce je celoten graf nepovezan
            maxX=self.maxWidth
            maxY=self.maxHeight
            minX=0
            minY=0
        else:
            maxX=max(conCoorsX)
            maxY=max(conCoorsY)
            minX=min(conCoorsX)
            minY=min(conCoorsY)

        cX=(maxX+minX)/2.0  #sredisce
        cY=(maxY+minY)/2.0

        R=max((abs(maxX)-abs(cX)), (abs(maxY)-abs(cY))) * math.sqrt(2) +UDist  #polmer kroga

        angles=arange(0,(2*pi),2*pi/ncCount)  #radiani
        allAngles=zeros(self.nVertices(), 'f')

        #ta zanka ni v Numeric zato, ker je angles[] krajsi od allAngles[] (graf ni povezan)
        count=0
        for i in range(0, self.nVertices()):
            if (pos[i]==1):
                allAngles[i]=angles[count]
                count+=1

        ccX=R*cos(allAngles)+cX  #koordinate na kroznici
        ccY=R*sin(allAngles)+cY
        self.xCoors = where(pos, ccX, self.xCoors)
        self.yCoors = where(pos, ccY, self.yCoors)

    def saveNetwork(self, fn):
        name = ''
        try:
            graphFile = file(fn,'w+')
        except IOError:
            return 1

        graphFile.write('### This file was generated with Orange Network Visualizer ### \n\n\n')
        if name == '':
            graphFile.write('*Network ' + '"no name" \n\n')
        else:
            graphFile.write('*Network ' + str(name) + ' \n\n')


        #izpis opisov vozlisc
        graphFile.write('*Vertices% 8d\n' %self.graph.nVertices)
        for v in range(self.graph.nVertices):
            graphFile.write('% 8d ' % (v + 1))
#            if verticesParms[v].label!='':
#                self.GraphFile.write(str('"'+ verticesParms[v].label + '"') + ' \t')
#            else:
            try:
                label = self.graph.items[v]['label']
                graphFile.write(str('"'+ str(label) + '"') + ' \t')
            except:
                graphFile.write(str('"'+ str(v) + '"') + ' \t')
            
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
            for (i,j) in self.graph.getEdges():
                if len(self.graph[i,j]) > 0:
                    graphFile.write('% 8d % 8d %d' % (i+1, j+1, int(self.graph[i,j][0])))
                    graphFile.write('\n')
        else:
            graphFile.write('*Edges \n')
            for (i,j) in self.graph.getEdges():
                if len(self.graph[i,j]) > 0:
                    graphFile.write('% 8d % 8d %d' % (i+1, j+1, int(self.graph[i,j][0])))
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
    
