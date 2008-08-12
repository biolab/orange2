import math
import random
import numpy
import orange
import orangeom
import os.path

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
    
    def getData(self, i, j):
        if self.graph.items is orange.ExampleTable:
            return self.data[i][j]
        elif self.graph.data is List:
            return self.data[i][j]
        
    def nVertices(self):
        if self.graph:
            return self.graph.nVertices
  
            
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

        R=max((abs(maxX)-abs(cX)), (abs(maxY)-abs(cY))) * sqrt(2) +UDist  #polmer kroga

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
            
            x = self.coors[0][v] / 1000
            y = self.coors[1][v] / 1000
            if x < 0: x = 0
            if x >= 1: x = 0.9999
            if y < 0: y = 0
            if y >= 1: y = 0.9999
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
    
