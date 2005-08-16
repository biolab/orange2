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
import OWGraphTools
import OWToolbars
from random import random
from OWWidget import *


from OWVisGraph import *

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
        self.computeStress=0
        self.ReDraw=1
        self.NumIter=10
        self.inputs=[("Sym Matrix", orange.SymMatrix, self.cmatrix)]
        self.outputs=[("Example Table", ExampleTable)]

        self.stressFunc=[("Kruskal stress", orngMDS.KruskalStress),
                              ("Sammon stress", orngMDS.SammonStress),
                              ("Signed sammon stress", orngMDS.SgnSammonStress),
                              ("Signed reative stress", orngMDS.SgnRelStress)]

        self.layout=QVBoxLayout(self.mainArea)
        self.graph=MDSGraph(self.mainArea)
        self.layout.addWidget(self.graph)

        tabs=QTabWidget(self.controlArea)
        graph=QVGroupBox(self)
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

        OWGUI.button(opt, self, "LSMT", self.LSMT)
        OWGUI.button(opt, self, "Step", self.smacofStep)
        self.startButton=OWGUI.button(opt, self, "Start", self.start)
        #OWGUI.button(opt, self, "Stop", self.stop)
        #OWGUI.button(opt, self, "Redraw graph", callback=self.graph.updateData)
        OWGUI.checkBox(opt, self, "ReDraw", "Redraw graph after each step")
        OWGUI.spin(opt, self, "NumIter",box="Num. Iterations per Step",min=1, max=1000)
        
        
        stopping=OWGUI.widgetBox(opt, "Stopping Conditions")
        OWGUI.checkBox(stopping, self, "computeStress", "Compute stress")
        stress=OWGUI.widgetBox(stopping, "Min. Avg. Stress Delta")
        OWGUI.comboBox(stopping, self, "StressFunc", box="Stress Function", items=[a[0] for a in self.stressFunc],
                       callback=lambda: not self.mds.getStress(self.stressFunc[self.StressFunc][1]) and self.graph.setLines(True) and self.graph.replot())
        OWGUI.qwtHSlider(stress, self, "minStressDelta", minValue=1e-5, maxValue=1e-2, step=1e-5, precision=6)
        OWGUI.spin(stopping, self, "maxIterations", box="Max. Number of Steps", min=1, max=100)
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
            
        if data:
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
                
        if matrix:
            matrix=Numeric.array([m for m in matrix])
            self.mds=orngMDS.MDS(matrix)
            self.mds.getStress()
            self.stress=self.getAvgStress(self.stressFunc[self.StressFunc][1])
            self.graph.setData(self.mds, self.data)
        else:
            self.graph.clear()

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
        self.mds.X = RandomArray.random(shape=[self.mds.n,2])
        if self.computeStress:
            self.mds.getStress(self.stressFunc[self.StressFunc][1])
            self.stress=self.getAvgStress(self.stressFunc[self.StressFunc][1])
        self.graph.updateData()

    def jitter(self):
        mi = Numeric.argmin(self.mds.X,0)
        ma = Numeric.argmax(self.mds.X,0)
        st = 0.01*(ma-mi)
        for i in range(self.mds.n):
            for j in range(2):
                self.mds.X[i][j] += st[j]*(random()-0.5)
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
            print p1, pcur
            self.progressBarSet(int(pcur))

            oldStress=stress
        self.startButton.setText("Start")
        self.progressBarFinished()
        #if not self.ReDraw:
        self.graph.updateData()
        self.done=True
        #print "time %i " % (time.clock()-startTime)

    def getAvgStress(self, stressf=orngMDS.SgnRelStress):
        self.mds.getDistance()
        total=0.0
        total=sum([abs(a[0]) for a in self.mds.arr])
        self.infoA.setText("Avg. stress: %.7f" % (total/(self.mds.n*self.mds.n)))
        return total/(self.mds.n*self.mds.n)

    def sendSelections(self):
        selectedInd=[]
        for i,(x,y) in enumerate(self.mds.X):
            if self.graph.isPointSelected(x,y):
                selectedInd+=[i]
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
                selection[i][xAttr]=self.mds.X[selectedInd[i]][0]
                selection[i][yAttr]=self.mds.X[selectedInd[i]][1]
            self.send("Example Table", selection)

class MDSGraph(OWVisGraph):
    def __init__(self, parent=None, name=None):
        OWVisGraph.__init__(self, parent, name)
        self.data=None
        self.mds=None
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
        self.shapes=[QwtSymbol.Ellipse,
                                QwtSymbol.Rect,
                                QwtSymbol.Diamond,
                                QwtSymbol.Triangle,
                                QwtSymbol.DTriangle ,
                                QwtSymbol.UTriangle,
                                QwtSymbol.LTriangle,
                                QwtSymbol.RTriangle,
                                QwtSymbol.Cross, 
                                QwtSymbol.XCross ]

    def setData(self, mds, data):
        if mds:
            self.mds=mds
            self.data=data
            self.discColors=[]
            self.colors=[]
            self.sizes=[]
            if data:
                self.attributes=[attr for attr in data.domain.variables+data.domain.getmetas().values() or [] ]
                self.discAttributes=filter(lambda a: a.varType==orange.VarTypes.Discrete, self.attributes)
                self.contAttributes=filter(lambda a: a.varType==orange.VarTypes.Continuous, self.attributes)
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
        if self.ColorAttr==0:
            colors=[Qt.black]
        elif self.attributes[self.ColorAttr-1].varType==orange.VarTypes.Discrete:
            colors=OWGraphTools.ColorPaletteHSV(len(self.attributes[self.ColorAttr-1].values))
        elif self.attributes[self.ColorAttr-1].varType==orange.VarTypes.Continuous:
            colors=OWGraphTools.ColorPaletteHSV()
            values=[self.data[i][self.attributes[self.ColorAttr-1]] for i in range(len(self.data))]
            maxVal=max(values)
            minVal=min(values)
            #print minVal, maxVal
        else:
            colors=[Qt.black]
        
        if self.SizeAttr!=0:
            values=[self.data[i][self.contAttributes[self.SizeAttr-1]] for i in range(len(self.data))]
            sizeMinVal=min(values)
            sizeMaxVal=max(values)

        for i,(x,y) in enumerate(self.mds.X[:len(self.data)-1]):
            # ########################
            #Get the color , size and shape
            if self.ColorAttr==0:
                penColor=Qt.black
            elif self.attributes[self.ColorAttr-1].varType==orange.VarTypes.Discrete:
                penColor=colors[int(self.data[i][self.attributes[self.ColorAttr-1]])]
            else:
                #index=int(math.floor(float(self.data[i][self.attributes[self.ColorAttr-1]]-minVal)*100/abs(maxVal-minVal)))
                #penColor=colors[index<100 and index or 99]
                penColor=colors.getColor((self.data[i][self.attributes[self.ColorAttr-1]]-minVal)/abs(maxVal-minVal))

            if self.SizeAttr==0:
                size=5
            else:
                size=int(self.data[i][self.contAttributes[self.SizeAttr-1]]/sizeMaxVal*9)+1

            if self.ShapeAttr==0:
                symbol=self.shapes[0]
            else:
                symbol=self.shapes[int(self.data[i][self.discAttributes[self.ShapeAttr-1]])%len(self.shapes)]
                
            key=self.addCurve(str(i),penColor, penColor, size,symbol=symbol, xData=[x], yData=[y])
            self.curveKeys+=[key]

            if self.NameAttr!=0:
                self.addMarker(str(self.data[i][self.attributes[self.NameAttr-1]]), x,y, Qt.AlignRight)
                
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
            stress=[(abs(s),s,(a,b)) for s,(a,b) in self.mds.arr]
            stress.sort()
            stress.reverse()
            for (as,s,(a,b)) in stress[len(self.lineKeys):min(self.NumStressLines, len(stress))]:
                (xa,ya)=self.mds.X[a]
                (xb,yb)=self.mds.X[b]
                #color=s<0 and Qt.red or Qt.green
                if self.mds.dist[a][b]-self.mds.O[a][b]>0:
                    color=Qt.green
                    k1=self.addCurve("A", color, color, 0, QwtCurve.Lines, xData=[xa,xb], yData=[ya,yb], lineWidth=1)
                    r=self.mds.O[a][b]/max(self.mds.dist[a][b], 1e-6)
                    xa1=xa+(1-r)/2*(xb-xa)
                    xb1=xb+(1-r)/2*(xa-xb)
                    ya1=ya+(1-r)/2*(yb-ya)
                    yb1=yb+(1-r)/2*(ya-yb)
                    k2=self.addCurve("A", color, color, 0, QwtCurve.Lines, xData=[xa1,xb1], yData=[ya1,yb1], lineWidth=4)    
                    self.lineKeys.append( (k1,k2) )
                else:
                    color=Qt.red
                    r=self.mds.O[a][b]/max(self.mds.dist[a][b], 1e-6)
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

