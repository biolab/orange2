"""
<name>MDS</name>
<description>Multi dimensional scaling</description>
<icon>MDS.png</icon>
<priority>5000</priority>
"""

import orange
import orngMDS
import OWGUI
import Numeric
import RandomArray
import qt
import sys
import math
import time
import os
import OWGraphTools
import OWToolbars
from random import random
from OWWidget import *
from OWVisGraph import *
from sets import Set

try:
    from OWDataFiles import DataFiles
except:
    class DataFiles:
        pass

class OWMDS(OWWidget):
    settingsList=["graph.ColorAttr", "graph.SizeAttr", "graph.ShapeAttr", "graph.NameAttr", "graph.ShowStress", "graph.NumStressLines", "graph.ShowName",
                  "StressFunc", "graph.LineStyle", "toolbarSelection", "autoSendSelection", "selectionOptions", "computeStress"]
    callbackDeposit=[]
    def __init__(self, parent=None, signalManager=None, name="Multi Dimensional Scaling"):
        OWWidget.__init__(self, parent, signalManager, name)

        self.StressFunc=3
        self.minStressDelta=1e-5
        self.maxIterations=50
        self.maxImprovment=10
        self.autoSendSelection=0
        self.toolbarSelection=0
        self.selectionOptions=0
        self.computeStress=1
        self.ReDraw=1
        self.NumIter=1
        self.RefreshMode=0
        self.inputs=[("Sym Matrix", orange.SymMatrix, self.cmatrix)]
        self.outputs=[("Example Table", ExampleTable), ("Structured Data Files", DataFiles)]

        self.stressFunc=[("Kruskal stress", orngMDS.KruskalStress),
                              ("Sammon stress", orngMDS.SammonStress),
                              ("Signed sammon stress", orngMDS.SgnSammonStress),
                              ("Signed reative stress", orngMDS.SgnRelStress)]

        self.layout=QVBoxLayout(self.mainArea)
        self.graph=MDSGraph(self.mainArea)
        self.layout.addWidget(self.graph)

        tabs=QTabWidget(self.controlArea)
        graph=QVGroupBox(self)
        OWGUI.hSlider(graph, self, "graph.PointSize", box="Point Size", minValue=1, maxValue=20, callback=self.graph.updateData)
        self.colorCombo=OWGUI.comboBox(graph, self, "graph.ColorAttr", box="Color", callback=self.graph.updateData)
        self.sizeCombo=OWGUI.comboBox(graph, self, "graph.SizeAttr", box="Size", callback=self.graph.updateData)
        self.shapeCombo=OWGUI.comboBox(graph, self, "graph.ShapeAttr", box="Shape", callback=self.graph.updateData)
        self.nameCombo=OWGUI.comboBox(graph, self, "graph.NameAttr", box="Label", callback=self.graph.updateData)
        OWGUI.spin(graph, self, "graph.NumStressLines", label="Number of stress lines", min=0, max=1000, callback=self.graph.updateLines)

        self.zoomToolbar=OWToolbars.ZoomSelectToolbar(self, graph, self.graph, self.autoSendSelection)
        self.connect(self.zoomToolbar.buttonSendSelections, SIGNAL("clicked()"), self.sendSelections)
        self.graph.autoSendSelectionCallback = lambda :self.autoSendSelection and self.sendSelections()

        OWGUI.checkBox(graph, self, "autoSendSelection", "Auto send selected")
        OWGUI.radioButtonsInBox(graph, self, "selectionOptions", ["Don't append", "Append coord.", "Append coord. as meta"], box="Append coordinates") 
        
        mds=QVGroupBox(self)
        init=OWGUI.widgetBox(mds, "Initialization")
        OWGUI.button(init, self, "Randomize", self.randomize)
        OWGUI.button(init, self, "Jitter", self.jitter)
        OWGUI.button(init, self, "Torgerson", self.torgerson)
        opt=OWGUI.widgetBox(mds, "Optimization")

        self.startButton=OWGUI.button(opt, self, "Start", self.testStart)
        OWGUI.button(opt, self, "LSMT", self.LSMT)
        OWGUI.button(opt, self, "Step", self.smacofStep)
        #OWGUI.button(opt, self, "Stop", self.stop)
        #OWGUI.button(opt, self, "Redraw graph", callback=self.graph.updateData)
        #OWGUI.checkBox(opt, self, "ReDraw", "Redraw graph after each step")
        #OWGUI.spin(opt, self, "NumIter",box="Num. Iterations per Step",min=1, max=1000)
        OWGUI.radioButtonsInBox(opt, self, "RefreshMode", ["Every step", "Every 10 steps", "Every 100 steps"], "Refresh after optimization") 
        
        
        self.stopping=OWGUI.widgetBox(opt, "Stopping Conditions")
        #OWGUI.checkBox(stopping, self, "computeStress", "Compute stress")
        stress=OWGUI.widgetBox(self.stopping, "Min. Avg. Stress Delta")
        OWGUI.comboBox(self.stopping, self, "StressFunc", box="Stress Function", items=[a[0] for a in self.stressFunc],
                       callback=lambda: not self.mds.getStress(self.stressFunc[self.StressFunc][1]) and self.graph.setLines(True) and self.graph.replot())
        OWGUI.qwtHSlider(stress, self, "minStressDelta", minValue=1e-5, maxValue=1e-2, step=1e-5, precision=6)
        OWGUI.spin(self.stopping, self, "maxIterations", box="Max. Number of Steps", min=1, max=100)
        #OWGUI.spin(stopping, self, "maxImprovment", box="Max. improvment of a run", min=0, max=100, postfix="%")
    
        tabs.addTab(mds, "MDS")
        tabs.addTab(graph, "Graph")
        mds.setSizePolicy(QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum))
        graph.setSizePolicy(QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum))
        self.controlArea.setMinimumWidth(250)
        infoBox=OWGUI.widgetBox(self.controlArea, "Info")
        self.infoA=QLabel("Avg. stress:", infoBox)
        self.infoB=QLabel("Num. steps", infoBox)
        OWGUI.button(self.controlArea, self, "Save", self.graph.saveToFile)
        #self.info
        self.resize(900,700)

        self.done=True

    def cmatrix(self, matrix=None):
        self.origMatrix=matrix
        self.data=data=None
        if matrix:
            self.data=data=getattr(matrix, "items")
        if data and type(data)==orange.ExampleTable:
            self.setExampleTable(data)
        elif type(data)==list:
            self.setList(data)
        self.graph.ColorAttr=0
        self.graph.SizeAttr=0
        self.graph.ShapeAttr=0
        self.graph.NameAttr=0
            
                
        if matrix:
            self.mds=orngMDS.MDS(matrix)
            self.mds.points=RandomArray.random([self.mds.n, self.mds.dim])
            self.mds.getStress()
            self.stress=self.getAvgStress(self.stressFunc[self.StressFunc][1])
            self.graph.setData(self.mds, self.colors, self.sizes, self.shapes, self.names)
        else:
            self.graph.clear()

    def setExampleTable(self, data):
        self.colorCombo.clear()
        self.sizeCombo.clear()
        self.shapeCombo.clear()
        self.nameCombo.clear()
        attributes=[attr for attr in data.domain.variables+data.domain.getmetas().values() or [] ]
        discAttributes=filter(lambda a: a.varType==orange.VarTypes.Discrete, attributes)
        contAttributes=filter(lambda a: a.varType==orange.VarTypes.Continuous, attributes)
        attrName=[attr.name for attr in attributes]
        for name in ["One color"]+attrName:
            self.colorCombo.insertItem(name)
        for name in ["One size"]+map(lambda a:a.name, contAttributes):
            self.sizeCombo.insertItem(name)
        for name in ["One shape"]+map(lambda a: a.name, discAttributes):
            self.shapeCombo.insertItem(name)
        for name in ["No  name"]+attrName:
            self.nameCombo.insertItem(name)
            
        self.attributes=attributes
        self.discAttributes=discAttributes
        self.contAttributes=contAttributes

        self.colors=[[Qt.black]*(len(attributes)+1) for i in range(len(data))]
        self.shapes=[[QwtSymbol.Ellipse]*(len(discAttributes)+1) for i in range(len(data))]
        self.sizes=[[5]*(len(contAttributes)+1) for i in range(len(data))]
        self.names=[[""]*(len(attributes)+1) for i in range(len(data))]
        contI=discI=attrI=1
        for j, attr in enumerate(attributes):
            if attr.varType==orange.VarTypes.Discrete:
                c=OWGraphTools.ColorPaletteHSV(len(attr.values))
                for i in range(len(data)):
                    self.colors[i][attrI]= data[i][attr].isSpecial()  and Qt.black or c[int(data[i][attr])]
                    self.shapes[i][discI]= data[i][attr].isSpecial() and self.graph.shapeList[0] or self.graph.shapeList[int(data[i][attr])%len(self.graph.shapeList)]
                    self.names[i][attrI]=" "+str(data[i][attr])
                    #self.sizes[i][contI]=5
                attrI+=1
                discI+=1
            elif attr.varType==orange.VarTypes.Continuous:
                c=OWGraphTools.ColorPaletteHSV(-1)
                val=[e[j] for e in data]
                minVal=min(val)
                maxVal=max(val)
                for i in range(len(data)):
                    self.colors[i][attrI]=c.getColor((data[i][attr]-minVal)/abs(maxVal-minVal))
                    #self.shapes[i][discI]=self.graph.shapeList[0]
                    self.names[i][attrI]=" "+str(data[i][attr])
                    self.sizes[i][contI]=int(self.data[i][attr]/maxVal*9)+1
                contI+=1
                attrI+=1
            else:
                for i in range(len(data)):
                    self.colors[i][attrI]=Qt.black
                    #self.shapes[i][j+1]=self.graph.shapeList[0]
                    self.names[i][attrI]=" "+str(data[i][attr])
                    #self.sizes[i][j+1]=5
                attrI+=1

    def setList(self, data):
        self.colorCombo.clear()
        self.sizeCombo.clear()
        self.shapeCombo.clear()
        self.nameCombo.clear()
        for name in ["One color", "strain"]:
            self.colorCombo.insertItem(name)
        for name in ["No name", "name", "strain"]:
            self.nameCombo.insertItem(name)
            
        self.colors=[[Qt.black]*3 for i in range(len(data))]
        self.shapes=[[QwtSymbol.Ellipse] for i in range(len(data))]
        self.sizes=[[5] for i in range(len(data))]
        self.names=[[""]*4 for i in range(len(data))]
        try:
            #print dir(data[0][1][0])
            strains=list(Set([d.strain for d in data]))
            c=OWGraphTools.ColorPaletteHSV(len(strains))
            for i, d in enumerate(data):
                self.colors[i][1]=c[strains.index(d.strain)]
                self.names[i][1]=" "+d.name
                self.names[i][2]=" "+d.strain
        except Exception, val:
            print val
        
        
    def smacofStep(self):
        for i in range(self.NumIter):
            self.mds.SMACOFstep()
        if self.computeStress:
            self.mds.getStress(self.stressFunc[self.StressFunc][1])
            self.stress=self.getAvgStress(self.stressFunc[self.StressFunc][1])
        st=time.clock()
        if self.ReDraw:
            self.graph.updateData()
        #print "Update:", time.clock()-st

    def LSMT(self):
        self.mds.LSMT()
        if self.computeStress:
            self.mds.getStress(self.stressFunc[self.StressFunc][1])
            self.stress=self.getAvgStress(self.stressFunc[self.StressFunc][1])
        if self.ReDraw:
            self.graph.updateData()

    def torgerson(self):
        if self.mds:
            self.mds.Torgerson()
            if self.computeStress:
                self.mds.getStress(self.stressFunc[self.StressFunc][1])
                self.stress=self.getAvgStress(self.stressFunc[self.StressFunc][1])
            self.graph.updateData()

    def randomize(self):
        self.mds.points = RandomArray.random(shape=[self.mds.n,2])
        if self.computeStress:
            self.mds.getStress(self.stressFunc[self.StressFunc][1])
            self.stress=self.getAvgStress(self.stressFunc[self.StressFunc][1])
        self.graph.updateData()

    def jitter(self):
        mi = Numeric.argmin(self.mds.points,0)
        ma = Numeric.argmax(self.mds.points,0)
        st = 0.01*(ma-mi)
        for i in range(self.mds.n):
            for j in range(2):
                self.mds.points[i][j] += st[j]*(random()-0.5)
        if self.computeStress:
            self.mds.getStress(self.stressFunc[self.StressFunc][1])
            self.stress=self.getAvgStress(self.stressFunc[self.StressFunc][1])
        self.graph.updateData()

    def start(self):
        if self.done==False:
            self.done=True
            return
        self.done=False
        self.startButton.setText("Stop")
        numIter=0
        self.progressBarInit()
        pcur=0
        startStress=oldStress=stress=self.getAvgStress(self.stressFunc[self.StressFunc][1])
        startTime=time.clock()
        hist=[stress]*3
        while not self.done and numIter<self.maxIterations:
            for i in range(self.NumIter):
                self.mds.SMACOFstep()
                qApp.processEvents()
            if self.computeStress:
                self.mds.getStress(self.stressFunc[self.StressFunc][1])
                self.stress=stress=self.getAvgStress(self.stressFunc[self.StressFunc][1])
                hist.pop(0)
                hist.append(abs(oldStress-stress))
            numIter+=1
            self.infoB.setText("Num. steps: %i" % numIter)
            qApp.processEvents()
            if self.ReDraw:
                self.graph.updateData()
            qApp.processEvents()
            if self.computeStress and abs(sum(hist)/3)<abs(self.minStressDelta*oldStress):
                break
            ## Update progress bar
            p1=abs(self.minStressDelta*oldStress)/max(sum(hist)/3, 1e-6)*100
            if p1>100: p1=0
            pcur=min(max([p1, float(numIter)/self.maxIterations*100, pcur]),99)
            self.progressBarSet(int(pcur))

            oldStress=stress
        self.startButton.setText("Start")
        self.progressBarFinished()
        #if not self.ReDraw:
        self.graph.updateData()
        self.done=True
        #print "time %i " % (time.clock()-startTime)

    def testStart(self):
        if self.done==False:
            self.done=True
            return
        self.done=False
        self.startButton.setText("Stop")
        self.stopping.setDisabled(1)
        self.progressBarInit()
        self.iterNum=0
        self.mds.progressCallback=self.callback
        self.mds.mds.optimize(self.maxIterations, self.stressFunc[self.StressFunc][1], self.minStressDelta)
        if self.iterNum%(math.pow(10,self.RefreshMode)):
            self.graph.updateData()
        self.startButton.setText("Start")
        self.stopping.setDisabled(0)
        self.progressBarFinished()
        self.done=True

    def callback(self, a,b=None):
        if not self.iterNum%(math.pow(10,self.RefreshMode)):
            self.graph.updateData()
        self.iterNum+=1
        self.infoB.setText("Num. steps: %i" % self.iterNum)
        self.infoA.setText("Avg. Stress: %f" % self.mds.avgStress)
        self.progressBarSet(int(a*100))
        qApp.processEvents()
        if self.done:
            return 0
        else:
            return 1
        

    def getAvgStress(self, stressf=orngMDS.SgnRelStress):
        return self.mds.avgStress
        """
        self.mds.getDistance()
        total=0.0
        total=sum([abs(a[0]) for a in self.mds.arr])
        self.infoA.setText("Avg. stress: %.7f" % (total/(self.mds.n*self.mds.n)))
        return total/(self.mds.n*self.mds.n)
        """

    def sendSelections(self):
        selectedInd=[]
        for i,(x,y) in enumerate(self.mds.points):
            if self.graph.isPointSelected(x,y):
                selectedInd+=[i]
        if type(self.data)==orange.ExampleTable:
            self.sendExampleTable(selectedInd)
        elif type(self.data)==list:
            self.sendList(selectedInd)


    def sendExampleTable(self, selectedInd):
        if self.selectionOptions==0:
            self.send("Example Table", orange.ExampleTable(self.data.getitems(selectedInd)))
        else:
            xAttr=orange.FloatVariable("X")
            yAttr=orange.FloatVariable("Y")
            if self.selectionOptions==1:
                domain=orange.Domain([xAttr, yAttr]+[v for v in self.data.domain.variables])
                domain.addmetas(self.data.domain.getmetas())
            else:
                domain=orange.Domain(self.data.domain)
                domain.addmeta(orange.newmetaid(), xAttr)
                domain.addmeta(orange.newmetaid(), yAttr)
            selection=orange.ExampleTable(domain)
            selection.extend(self.data.getitems(selectedInd))
            for i in range(len(selectedInd)):
                selection[i][xAttr]=self.mds.points[selectedInd[i]][0]
                selection[i][yAttr]=self.mds.points[selectedInd[i]][1]
            self.send("Example Table", selection)

    def sendList(self, selectedInd):
        if not selectedInd:
            self.send("Structured Data Files", None)
        else:
            datasets=[self.data[i] for i in selectedInd]
            names=list(Set([d.dirname for d in datasets]))
            data=[(name, [d for d in filter(lambda a:a.strain==name, datasets)]) for name in names]
            self.send("Structured Data Files",data)

class MDSGraph(OWVisGraph):
    def __init__(self, parent=None, name=None):
        OWVisGraph.__init__(self, parent, name)
        self.data=None
        self.mds=None
        self.PointSize=5
        self.ColorAttr=0
        self.SizeAttr=0
        self.ShapeAttr=0
        self.NameAttr=0
        self.ShowStress=True
        self.NumStressLines=10
        self.ShowName=True
        self.curveKeys=[]
        self.pointKeys=[]
        self.points=[]
        self.lines=[]
        self.lineKeys=[]
        self.colors=[]
        self.sizes=[]
        self.shapeList=[QwtSymbol.Ellipse,
                                QwtSymbol.Rect,
                                QwtSymbol.Diamond,
                                QwtSymbol.Triangle,
                                QwtSymbol.DTriangle ,
                                QwtSymbol.UTriangle,
                                QwtSymbol.LTriangle,
                                QwtSymbol.RTriangle,
                                QwtSymbol.Cross, 
                                QwtSymbol.XCross ]

    def setData(self, mds, colors, sizes, shapes, names):
        if mds:
            self.mds=mds
            #self.data=data
            self.colors=colors
            self.sizes=sizes
            self.shapes=shapes
            self.names=names
            
            self.updateData()
                
    def updateData(self):
        if self.mds:
            self.clear()
            self.setPoints()
            self.setLines(True)
        for axis in [QwtPlot.xBottom, QwtPlot.xTop, QwtPlot.yLeft, QwtPlot.yRight]:
            self.setAxisAutoScale(axis)
        self.updateAxes()
        self.repaint()

    def updateLines(self):
        if self.mds:
            self.setLines()
        self.repaint()

    def setPoints(self):
        import sets
        if self.ShapeAttr==0 and self.SizeAttr==0:
            colors=[c[self.ColorAttr] for c in self.colors]
            
            set=[]
            for c in colors:
                if  c not in set:
                    set.append(c)
            #set=reduce(lambda set,a: (not(a in set)) and set.append(a), colors, []) 
            #set=sets.ImmutableSet([c[self.ColorAttr] for c in self.colors])
            
            dict={}
            for i in range(len(self.colors)):
                c=self.colors[i][self.ColorAttr]
                if dict.has_key(c.hsv()):
                    dict[c.hsv()].append(i)
                else:
                    dict[c.hsv()]=[i]
            for color in set:
                print len(dict[color.hsv()]), color.name()
                X=[self.mds.points[i][0] for i in dict[color.hsv()]]
                Y=[self.mds.points[i][1] for i in dict[color.hsv()]]
                self.addCurve("A", color, color, self.PointSize, symbol=QwtSymbol.Ellipse, xData=X, yData=Y)
            return 
        for i in range(len(self.colors)):
            self.addCurve("a", self.colors[i][self.ColorAttr], self.colors[i][self.ColorAttr], self.sizes[i][self.SizeAttr]*1.0/5*self.PointSize,
                          symbol=self.shapes[i][self.ShapeAttr], xData=[self.mds.points[i][0]],yData=[self.mds.points[i][1]])
            if self.NameAttr!=0:
                self.addMarker(self.names[i][self.NameAttr], self.mds.points[i][0], self.mds.points[i][1], Qt.AlignRight)
            
                               
    def setLines(self, reset=False):
        def removeCurve(keys):
            self.removeCurve(keys[0])
            if len(keys)==2:
                self.removeCurve(keys[1])

        if reset:
            for k in self.lineKeys:
                removeCurve(k)
            self.lineKeys=[]
        if self.NumStressLines<len(self.lineKeys):
            for k in self.lineKeys[self.NumStressLines:]:
                removeCurve(k)
            self.lineKeys=self.lineKeys[:self.NumStressLines]
        else:
            #stress=[(abs(s),s,(a,b)) for s,(a,b) in self.mds.arr]
            stress=[(abs(self.mds.stress[a,b]), self.mds.stress[a,b], (a,b))
                    for a in range(self.mds.n) for b in range(a)]
            stress.sort()
            stress.reverse()
            for (as,s,(a,b)) in stress[len(self.lineKeys):min(self.NumStressLines, len(stress))]:
                (xa,ya)=self.mds.points[a]
                (xb,yb)=self.mds.points[b]
                #color=s<0 and Qt.red or Qt.green
                if self.mds.projectedDistances[a,b]-self.mds.distances[a,b]>0:
                    color=Qt.green
                    k1=self.addCurve("A", color, color, 0, QwtCurve.Lines, xData=[xa,xb], yData=[ya,yb], lineWidth=1)
                    r=self.mds.distances[a,b]/max(self.mds.projectedDistances[a,b], 1e-6)
                    xa1=xa+(1-r)/2*(xb-xa)
                    xb1=xb+(1-r)/2*(xa-xb)
                    ya1=ya+(1-r)/2*(yb-ya)
                    yb1=yb+(1-r)/2*(ya-yb)
                    k2=self.addCurve("A", color, color, 0, QwtCurve.Lines, xData=[xa1,xb1], yData=[ya1,yb1], lineWidth=4)    
                    self.lineKeys.append( (k1,k2) )
                else:
                    color=Qt.red
                    r=self.mds.distances[a,b]/max(self.mds.projectedDistances[a,b], 1e-6)
                    xa1=(xa+xb)/2+r/2*(xa-xb)
                    xb1=(xa+xb)/2+r/2*(xb-xa)
                    ya1=(ya+yb)/2+r/2*(ya-yb)
                    yb1=(ya+yb)/2+r/2*(yb-ya)
                    k1=self.addCurve("A", color, color, 0, QwtCurve.Lines, xData=[xa1,xb1], yData=[ya1,yb1], lineWidth=2)
                    self.lineKeys.append( (k1,) )
                    
        
            
if __name__=="__main__":
    app=QApplication(sys.argv)
    w=OWMDS()
    app.setMainWidget(w)
    w.show()
    data=orange.ExampleTable("../../doc/datasets/iris.tab")
    #data=orange.ExampleTable("/home/ales/src/MDSjakulin/eu_nations.txt")
    matrix = orange.SymMatrix(len(data))
    dist = orange.ExamplesDistanceConstructor_Euclidean(data)
    matrix = orange.SymMatrix(len(data))
    matrix.setattr('items', data)
    for i in range(len(data)):
        for j in range(i+1):
            matrix[i, j] = dist(data[i], data[j])

    w.cmatrix(matrix)
    app.exec_loop()

