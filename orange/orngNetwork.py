import math
import random

import numpy

import orange
import orangeom

class NetworkOptimization(orangeom.NetworkOptimization):
    def __init__(self, graph=None, parent=None, name="None"):
        if graph is None:
            graph = orange.GraphAsList(2, 0)
            
        self.setGraph(graph)
        self.graph = graph
        self.parent = parent
        self.maxWidth  = 1000
        self.maxHeight = 1000
        
        self.attributeList = {}
        self.attributeValues = {}
        
    def collapse(self):
        if len(self.graph.getNodes(1)) > 0:
            nodes = list(set(range(self.graph.nVertices)) - set(self.graph.getNodes(1)))
                
            if len(nodes) > 0:
                subgraph = self.graph.getSubGraph(nodes)
                subgraph.setattr("items", self.graph.items.getitems(nodes))
                oldcoors = self.coors
                self.setGraph(subgraph)
                self.graph = subgraph
                    
                for i in range(len(nodes)):
                    self.coors[i][0] = oldcoors[nodes[i]][0]
                    self.coors[i][1] = oldcoors[nodes[i]][1]

        else:
            nodes = self.graph.getLargestFullGraphs()
        
            if len(nodes) > 0:
                nodescomp = list(set(range(self.graph.nVertices)) - set(nodes))
                subgraph = self.graph.getSubGraphMergeCluster(nodes)
                subgraph.setattr("items", self.graph.items.getitems(nodescomp))
                subgraph.items.append(self.graph.items[0])
                oldcoors = self.coors
                self.setGraph(subgraph)
                self.graph = subgraph
                for i in range(len(nodescomp)):
                    self.coors[i][0] = oldcoors[nodescomp[i]][0]
                    self.coors[i][1] = oldcoors[nodescomp[i]][1]
                    
                # place meta vertex in center of cluster    
                x, y = 0, 0
                for node in nodes:
                    x += oldcoors[node][0]
                    y += oldcoors[node][1]
                    
                x = x / len(nodes)
                y = y / len(nodes)
                
                self.coors[len(nodescomp)][0] = x
                self.coors[len(nodescomp)][1] = y
            
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
            graphFile.write('*Network ' + '"Qt network" \n\n')
        else:
            graphFile.write('*Network ' + str(name) + ' \n\n')


        #izpis opisov vozlisc

#        if writeCoordinates:
#            xs=copy.deepcopy(self.GraphBase.Xcoors)
#            ys=copy.deepcopy(self.GraphBase.Ycoors)
#            (xs, ys,)=normalizeCoordinates(MAXX, MAXY, xs, ys)

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
            
            x = self.coors[v][0] / 1000
            y = self.coors[v][1] / 1000
            if x < 0: x = 0
            if x >= 1: x = 0.9999
            if y < 0: y = 0
            if y >= 1: y = 0.9999
            z = 0.5000
            graphFile.write('%.4f    %.4f    %.4f\t' % (x, y, z))
#            if verticesParms[v].inFileDefinedCoors[2]!=None:
#                self.GraphFile.write(str(verticesParms[v].inFileDefinedCoors[2]/MAXZ) + ' \t')
#
#            if verticesParms[v].colorName!=None:
#                self.GraphFile.write(' ic ' + verticesParms[v].colorName + ' \t')
#            if verticesParms[v].borderColorName!=None:
#                self.GraphFile.write(' bc ' + verticesParms[v].borderColorName + ' \t')
#            if verticesParms[v].borderWidthFromFile==True:
#                self.GraphFile.write(' bw ' + str(verticesParms[v].borderWidth) + ' \t')
            graphFile.write('\n')

        #izpis opisov povezav
        #najprej neusmerjene
        graphFile.write('*Edges \n')
        for (i,j) in self.graph.getEdges():
            if len(self.graph[i,j]) > 0:
                graphFile.write('% 8d % 8d %d' % (i+1, j+1, int(self.graph[i,j][0])))
                graphFile.write('\n')

#        for v1 in edgesParms.keys():
#            for v2 in edgesParms[v1].keys():
#                if edgesParms[v1][v2].type==UNDIRECTED:
#                    #osnova
#                    self.GraphFile.write(str(v1+1) + ' ' + str(v2+1) + ' ' + str(edgesParms[v1][v2].weight) + ' \t')
#                    #dodatni parametri
#                    if edgesParms[v1][v2].label != '':
#                        self.GraphFile.write(' l ' + str('"'+edgesParms[v1][v2].label+'"') + ' \t')
#                    if edgesParms[v1][v2].colorName!=None:
#                        self.GraphFile.write(' c ' + edgesParms[v1][v2].colorName + ' \t')
#                    self.GraphFile.write('\n')
#
#        #se usmerjene
#        self.GraphFile.write('*Arcs \n')
#        for v1 in edgesParms.keys():
#            for v2 in edgesParms[v1].keys():
#                if edgesParms[v1][v2].type==DIRECTED:
#                    #osnova
#                    self.GraphFile.write(str(v1+1) + ' ' + str(v2+1) + ' ' + str(edgesParms[v1][v2].weight) + '\t')
#                    #dodatni parametri
#                    if edgesParms[v1][v2].label != '':
#                        self.GraphFile.write(' l ' + str('"'+edgesParms[v1][v2].label+'"') + ' \t')
#                    if edgesParms[v1][v2].colorName!=None:
#                        self.GraphFile.write(' c ' + edgesParms[v1][v2].colorName + ' \t')
#                    self.GraphFile.write('\n')

        graphFile.write('\n')
        graphFile.close()

        return 0
    
    def readNetwork(self, fn):
        graph, table = orangeom.NetworkOptimization.readNetwork(self, fn)
        graph.setattr("items", table)
        self.setGraph(graph)
        self.graph = graph
        return graph, table
    