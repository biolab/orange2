import math
import orange
import orangeom

from random import *
from numpy import *

class NetworkVisualizer(orangeom.NetworkOptimization):
    def __init__(self, graph, parent = None, name = "None"):
        
        self.setGraph(graph)
        self.coors = self.getCoors()
        self.graph = graph

        self.parent = parent
        self.maxWidth  = 1000
        self.maxHeight = 1000
        
        self.attributeList = {}
        self.attributeValues = {}
    
    def getVars(self):
        vars = []
        if (self.graph != None):
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
