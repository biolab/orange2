"""
<name>MDS</name>
<description>Multi dimensional scaling</description>
<icon>MDS.png</icon>
<contact>Ales Erjavec (ales.erjavec(@at@)fri.uni-lj.si)</contact>
<priority>2500</priority>
"""
from OWWidget import *
import orange
import orngMDS
import OWGUI
import numpy, sys, math, time, os
import OWColorPalette
import OWToolbars
from OWGraph import *
from sets import Set
from PyQt4.Qwt5 import *
from random import random

try:
    from OWDataFiles import DataFiles
except:
    class DataFiles:
        pass

class OWMDS(OWWidget):
    settingsList=["graph.PointSize", "graph.proportionGraphed", "graph.ColorAttr", "graph.SizeAttr", "graph.ShapeAttr", "graph.NameAttr", "graph.ShowStress", "graph.NumStressLines", "graph.ShowName",
                  "StressFunc", "applyLSMT", "toolbarSelection", "autoSendSelection", "selectionOptions", "computeStress",
                  "RefreshMode"]
    contextHandlers={"":DomainContextHandler("", [ContextField("graph.ColorAttr", DomainContextHandler.Optional),
                                                  ContextField("graph.SizeAttr", DomainContextHandler.Optional),
                                                  ContextField("graph.ShapeAttr", DomainContextHandler.Optional),
                                                  ContextField("graph.NameAttr", DomainContextHandler.Optional),
                                                  ContextField("graph.ShowName", DomainContextHandler.Optional)])}
    callbackDeposit=[]
    def __init__(self, parent=None, signalManager=None, name="Multi Dimensional Scaling"):
        OWWidget.__init__(self, parent, signalManager, name)

        self.StressFunc=3
        self.minStressDelta=1e-5
        self.maxIterations=5000
        self.maxImprovment=10
        self.autoSendSelection=0
        self.toolbarSelection=0
        self.selectionOptions=0
        self.computeStress=1
        self.ReDraw=1
        self.NumIter=1
        self.RefreshMode=0
        self.applyLSMT = 0
        self.inputs=[("Distances", orange.SymMatrix, self.cmatrix), ("Example Subset", ExampleTable, self.cselected)]
        self.outputs=[("Example Table", ExampleTable), ("Structured Data Files", DataFiles)]

        self.stressFunc=[("Kruskal stress", orngMDS.KruskalStress),
                              ("Sammon stress", orngMDS.SammonStress),
                              ("Signed Sammon stress", orngMDS.SgnSammonStress),
                              ("Signed relative stress", orngMDS.SgnRelStress)]


        self.graph=MDSGraph(self.mainArea)
        self.mainArea.layout().addWidget(self.graph)

        tabs=OWGUI.tabWidget(self.controlArea)
        
        mds=OWGUI.createTabPage(tabs, "Graph")
        graph=OWGUI.createTabPage(tabs, "MDS")

        ##MDS Tab        
        init=OWGUI.widgetBox(mds, "Initialization")
        OWGUI.button(init, self, "Randomize", self.randomize)
        OWGUI.button(init, self, "Jitter", self.jitter)
        OWGUI.button(init, self, "Torgerson", self.torgerson)
        opt=OWGUI.widgetBox(mds, "Optimization")

        self.startButton=OWGUI.button(opt, self, "Optimize", self.testStart)
        OWGUI.button(opt, self, "Single Step", self.smacofStep)
        box = OWGUI.widgetBox(opt, "Stress Function")
        OWGUI.comboBox(box, self, "StressFunc", items=[a[0] for a in self.stressFunc], callback=self.updateStress)        
        OWGUI.radioButtonsInBox(opt, self, "RefreshMode", ["Every step", "Every 10 steps", "Every 100 steps"], "Refresh During Optimization", callback=lambda :1)
        
        self.stopping=OWGUI.widgetBox(opt, "Stopping Conditions")
        OWGUI.qwtHSlider(self.stopping, self, "minStressDelta", label="Minimal average stress change", minValue=1e-5, maxValue=1e-2, step=1e-5, precision=6)
        OWGUI.qwtHSlider(self.stopping, self, "maxIterations", label="Maximal number of steps", minValue=10, maxValue=5000, step=10, precision=0)

        ##Graph Tab        
        OWGUI.hSlider(graph, self, "graph.PointSize", box="Point Size", minValue=1, maxValue=20, callback=self.graph.updateData)
        self.colorCombo=OWGUI.comboBox(graph, self, "graph.ColorAttr", box="Color", callback=self.graph.updateData)
        self.sizeCombo=OWGUI.comboBox(graph, self, "graph.SizeAttr", box="Size", callback=self.graph.updateData)
        self.shapeCombo=OWGUI.comboBox(graph, self, "graph.ShapeAttr", box="Shape", callback=self.graph.updateData)
        self.nameCombo=OWGUI.comboBox(graph, self, "graph.NameAttr", box="Label", callback=self.graph.updateData)
        box = OWGUI.widgetBox(graph, "Similar pairs")
        cb = OWGUI.checkBox(box, self, "graph.ShowStress", "Show similar pairs", callback = self.graph.updateLinesRepaint)
        b2 = OWGUI.widgetBox(box)
        OWGUI.widgetLabel(b2, "Proportion of connected pairs")
        OWGUI.separator(b2, height=3)
        sl = OWGUI.hSlider(b2, self, "graph.proportionGraphed", minValue=0, maxValue=20, callback=self.graph.updateLinesRepaint)
        cb.disables.append(b2)
        cb.makeConsistent()

        self.zoomToolbar=OWToolbars.ZoomSelectToolbar(self, graph, self.graph, self.autoSendSelection)
        self.connect(self.zoomToolbar.buttonSendSelections, SIGNAL("clicked()"), self.sendSelections)
        self.graph.autoSendSelectionCallback = lambda :self.autoSendSelection and self.sendSelections()

        OWGUI.checkBox(graph, self, "autoSendSelection", "Auto send selected")
        OWGUI.radioButtonsInBox(graph, self, "selectionOptions", ["Don't append", "Append coordinates", "Append coordinates as meta"], box="Append coordinates", callback=self.sendIf)

        mds.setSizePolicy(QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum))
        graph.setSizePolicy(QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum))
        self.controlArea.setMinimumWidth(250)
        OWGUI.separator(self.controlArea)
        infoBox=OWGUI.widgetBox(self.controlArea, "Info")
        self.infoA=OWGUI.widgetLabel(infoBox, "Avg. stress:")
        self.infoB=OWGUI.widgetLabel(infoBox, "Num. steps")
        OWGUI.rubber(self.controlArea)
        OWGUI.button(self.controlArea, self, "Save", self.graph.saveToFile, debuggingEnabled = 0)
        self.resize(900,630)

        self.done=True
        self.data=None
        self.selectedInputExamples=[]
        self.selectedInput=[]

    def cmatrix(self, matrix=None):
        self.closeContext()
        self.origMatrix=matrix
        self.data=data=None
        if matrix:
            self.data=data=getattr(matrix, "items")
        if data and type(data)==orange.ExampleTable:
            self.setExampleTable(data)
        elif type(data)==list:
            self.setList(data)
        elif type(data)==orange.VarList:
            self.setVarList(data)
        self.graph.ColorAttr=0
        self.graph.SizeAttr=0
        self.graph.ShapeAttr=0
        self.graph.NameAttr=0
        self.graph.closestPairs = None
            
                
        if matrix:
            self.mds=orngMDS.MDS(matrix)
            self.mds.points=numpy.random.random(size=[self.mds.n, self.mds.dim])
            self.mds.getStress()
            self.stress=self.getAvgStress(self.stressFunc[self.StressFunc][1])
            if data and type(data) == orange.ExampleTable:
                self.openContext("",self.data)
            self.graph.setData(self.mds, self.colors, self.sizes, self.shapes, self.names, self.selectedInput)
        else:
            self.graph.clear()

    def cselected(self, selected=[]):
        self.selectedInputExamples=selected and selected or[]
        if self.data and type(self.data)==orange.ExampleTable:
            self.setExampleTable(self.data)
            self.graph.setData(self.mds, self.colors, self.sizes, self.shapes, self.names, self.selectedInput)

    def setExampleTable(self, data):
        self.colorCombo.clear()
        self.sizeCombo.clear()
        self.shapeCombo.clear()
        self.nameCombo.clear()
        attributes=[attr for attr in data.domain.variables+data.domain.getmetas().values() or [] ]
        discAttributes=filter(lambda a: a.varType==orange.VarTypes.Discrete, attributes)
        contAttributes=filter(lambda a: a.varType==orange.VarTypes.Continuous, attributes)
        attrName=[attr.name for attr in attributes]
        for name in ["Same color"]+attrName:
            self.colorCombo.addItem(name)
        for name in ["Same size"]+map(lambda a:a.name, contAttributes):
            self.sizeCombo.addItem(name)
        for name in ["Same shape"]+map(lambda a: a.name, discAttributes):
            self.shapeCombo.addItem(name)
        for name in ["No name"]+attrName:
            self.nameCombo.addItem(name)

        self.attributes=attributes
        self.discAttributes=discAttributes
        self.contAttributes=contAttributes

        self.colors=[[Qt.black]*(len(attributes)+1) for i in range(len(data))]
        self.shapes=[[QwtSymbol.Ellipse]*(len(discAttributes)+1) for i in range(len(data))]
        self.sizes=[[5]*(len(contAttributes)+1) for i in range(len(data))]
        self.names=[[""]*(len(attributes)+1) for i in range(len(data))]
        try:
            selectedInput=self.selectedInputExamples.select(data.domain)
        except:
            selectedInput=[]
        self.selectedInput=map(lambda d: selectedInput and (d in selectedInput) or not selectedInput, data)
        contI=discI=attrI=1
        def check(ex, a):
            try:
                ex[a]
            except:
                return False
            return not ex[a].isSpecial()
        
        for j, attr in enumerate(attributes):
            if attr.varType==orange.VarTypes.Discrete:
                c=OWColorPalette.ColorPaletteHSV(len(attr.values))
                for i in range(len(data)):
                    self.colors[i][attrI]= check(data[i],attr)  and c[int(data[i][attr])] or Qt.black
##                    self.shapes[i][discI]= data[i][attr].isSpecial() and self.graph.shapeList[0] or self.graph.shapeList[int(data[i][attr])%len(self.graph.shapeList)]
                    self.shapes[i][discI]= check(data[i],attr) and self.graph.shapeList[int(data[i][attr])%len(self.graph.shapeList)] or self.graph.shapeList[0]
                    self.names[i][attrI]= check(data[i],attr) and " "+str(data[i][attr]) or ""
                    #self.sizes[i][contI]=5
                attrI+=1
                discI+=1
            elif attr.varType==orange.VarTypes.Continuous:
                c=OWColorPalette.ColorPaletteBW(-1)
                #val=[e[attr] for e in data if not e[attr].isSpecial()]
                val=[e[attr] for e in data if check(e, attr)]
                minVal=min(val or [0])
                maxVal=max(val or [1])
                for i in range(len(data)):
                    self.colors[i][attrI]=check(data[i],attr) and c.getColor((data[i][attr]-minVal)/max(maxVal-minVal, 1e-6)) or Qt.black 
                    #self.shapes[i][discI]=self.graph.shapeList[0]
                    self.names[i][attrI]=check(data[i],attr) and " "+str(data[i][attr]) or ""
                    self.sizes[i][contI]=check(data[i],attr) and int(self.data[i][attr]/maxVal*9)+1 or 5
                contI+=1
                attrI+=1
            else:
                for i in range(len(data)):
                    self.colors[i][attrI]=Qt.black
                    #self.shapes[i][j+1]=self.graph.shapeList[0]
                    self.names[i][attrI]= check(data[i],attr) and " "+str(data[i][attr]) or ""
                    #self.sizes[i][j+1]=5
                attrI+=1

    def setList(self, data):
        self.colorCombo.clear()
        self.sizeCombo.clear()
        self.shapeCombo.clear()
        self.nameCombo.clear()
        for name in ["Same color", "strain"]:
            self.colorCombo.addItem(name)
        for name in ["No name", "name", "strain"]:
            self.nameCombo.addItem(name)

        self.colors=[[Qt.black]*3 for i in range(len(data))]
        self.shapes=[[QwtSymbol.Ellipse] for i in range(len(data))]
        self.sizes=[[5] for i in range(len(data))]
        self.selectedInput=[False]*len(data)

        if type(data[0]) in [str, unicode]:
            self.names = [("", di, "", "") for di in data]
        else:
            self.names=[[""]*4 for i in range(len(data))]
            try:
                strains=list(Set([d.strain for d in data]))
                c=OWColorPalette.ColorPaletteHSV(len(strains))
                for i, d in enumerate(data):
                    self.colors[i][1]=c[strains.index(d.strain)]
                    self.names[i][1]=" "+d.name
                    self.names[i][2]=" "+d.strain
            except Exception, val:
                print val

    def setVarList(self, data):
        self.colorCombo.clear()
        self.sizeCombo.clear()
        self.shapeCombo.clear()
        self.nameCombo.clear()
        for name in ["Same color", "Variable"]:
            self.colorCombo.addItem(name)
        for name in ["No name", "Var name"]:
            self.nameCombo.addItem(name)
        self.colors=[[Qt.black]*3 for i in range(len(data))]
        self.shapes=[[QwtSymbol.Ellipse] for i in range(len(data))]
        self.sizes=[[5] for i in range(len(data))]
        self.names=[[""]*4 for i in range(len(data))]
        self.selectedInput=[False]*len(data)
        try:
            c=OWColorPalette.ColorPaletteHSV(len(data))
            for i, d in enumerate(data):
                self.colors[i][1]=c[i]
                self.names[i][1]=" " +str(d.name)
        except Exception, val:
            print val

    def smacofStep(self):
        if not getattr(self, "mds", None):
            return
        for i in range(self.NumIter):
            self.mds.SMACOFstep()
        if self.computeStress:
            self.mds.getStress(self.stressFunc[self.StressFunc][1])
            self.stress=self.getAvgStress(self.stressFunc[self.StressFunc][1])
        #st=time.clock()
        if self.ReDraw:
            self.graph.updateData()
        #print "Update:", time.clock()-st

## I (Janez) disabled LSMT because it is implemented terribly wrong:
#  orngMDS.LSMT transforms the distance matrix itself (indeed there is
#  the original stored, too), and from that point on there is no way the
#  user can "untransform" it, except for resending the signal
#  Since the basic problem is in bad design of orngMDS, I removed the option
#  from the widget. If somebody has time to fix orngMDS first, he's welcome. 
    def LSMT(self):
        if not getattr(self, "mds", None):
            return
        self.mds.LSMT()
        if self.computeStress:
            self.mds.getStress(self.stressFunc[self.StressFunc][1])
            self.stress=self.getAvgStress(self.stressFunc[self.StressFunc][1])
        if self.ReDraw:
            self.graph.updateData()

    def torgerson(self):
        if not getattr(self, "mds", None):
            return
        self.mds.Torgerson()
        if self.computeStress:
            self.mds.getStress(self.stressFunc[self.StressFunc][1])
            self.stress=self.getAvgStress(self.stressFunc[self.StressFunc][1])
        self.graph.updateData()

    def randomize(self):
        if not getattr(self, "mds", None):
            return
        self.mds.points = numpy.random.random(size=[self.mds.n,2])
        if self.computeStress:
            self.mds.getStress(self.stressFunc[self.StressFunc][1])
            self.stress=self.getAvgStress(self.stressFunc[self.StressFunc][1])
        self.graph.updateData()

    def jitter(self):
        if not getattr(self, "mds", None):
            return
        mi = numpy.argmin(self.mds.points,0)
        ma = numpy.argmax(self.mds.points,0)
        st = 0.01*(ma-mi)
        for i in range(self.mds.n):
            for j in range(2):
                self.mds.points[i][j] += st[j]*(random()-0.5)
        if self.computeStress:
            self.mds.getStress(self.stressFunc[self.StressFunc][1])
            self.stress=self.getAvgStress(self.stressFunc[self.StressFunc][1])
        self.graph.updateData()

    def start(self):
        if not getattr(self, "mds", None):
            return
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
        self.startButton.setText("Optimize")
        self.progressBarFinished()
        #if not self.ReDraw:
        self.graph.updateData()
        self.done=True
        #print "time %i " % (time.clock()-startTime)

    def testStart(self):
        if not getattr(self, "mds", None):
            return
        if self.done==False:
            self.done=True
            return
        self.done=False
        self.startButton.setText("Stop Optimization")
        self.stopping.setDisabled(1)
        self.progressBarInit()
        self.iterNum=0
        self.mds.progressCallback=self.callback
        self.mds.mds.optimize(self.maxIterations, self.stressFunc[self.StressFunc][1], self.minStressDelta)
        if self.iterNum%(math.pow(10,self.RefreshMode)):
            self.graph.updateData()
        self.startButton.setText("Optimize")
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

    def sendIf(self, i=-1):
        if self.autoSendSelection:
            self.sendSelections()
        
    def sendSelections(self):
        if not getattr(self, "mds", None):
            return
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
        if self.data and type(self.data[0]) == str:
            xAttr=orange.FloatVariable("X")
            yAttr=orange.FloatVariable("Y")
            nameAttr=  orange.StringVariable("name")
            if self.selectionOptions == 1:
                domain = orange.Domain([xAttr, yAttr, nameAttr])
                selection = orange.ExampleTable(domain)
                for i in range(len(selectedInd)):
                    selection.append(list(self.mds.points[selectedInd[i]]) + [self.data[i]])
            else:
                domain = orange.Domain([nameAttr])
                if self.selectionOptions:
                    domain.addmeta(orange.newmetaid(), xAttr)
                    domain.addmeta(orange.newmetaid(), yAttr)
                selection = orange.ExampleTable(domain)
                for i in range(len(selectedInd)):
                    selection.append([self.data[i]])
                    if self.selectionOptions:
                        selection[i][xAttr]=self.mds.points[selectedInd[i]][0]
                        selection[i][yAttr]=self.mds.points[selectedInd[i]][1]
            self.send("Example Table", selection)
            return
               
        if not selectedInd:
            self.send("Structured Data Files", None)
        else:
            datasets=[self.data[i] for i in selectedInd]
            names=list(Set([d.dirname for d in datasets]))
            data=[(name, [d for d in filter(lambda a:a.strain==name, datasets)]) for name in names]
            self.send("Structured Data Files",data)

    def updateStress(self):
        if not getattr(self, "mds", None):
            return
        self.mds.getStress(self.stressFunc[self.StressFunc][1])
#        self.graph.setLines(True)
        self.graph.replot()

class MDSGraph(OWGraph):
    def __init__(self, parent=None, name=None):
        OWGraph.__init__(self, parent, name)
        self.data=None
        self.mds=None
        self.PointSize=5
        self.ColorAttr=0
        self.SizeAttr=0
        self.ShapeAttr=0
        self.NameAttr=0
        self.ShowStress=False
        self.NumStressLines=10
        self.proportionGraphed = 20
        self.ShowName=True
        #self.curveKeys=[]
        self.pointKeys=[]
        self.points=[]
        self.lines=[]
        self.lineKeys=[]
        self.distanceLineKeys=[]
        self.colors=[]
        self.sizes=[]
        self.closestPairs = None
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

    def setData(self, mds, colors, sizes, shapes, names, showFilled):
        if 1:
#        if mds:
            self.mds=mds
            #self.data=data
            self.colors=colors
            self.sizes=sizes
            self.shapes=shapes
            self.names=names
            self.showFilled=showFilled #map(lambda d: not d, showFilled)
            self.updateData()

    def updateData(self):
#        if self.mds:
        if 1:
            self.clear()
            if self.ShowStress:
                self.updateDistanceLines()
            self.setPoints()
                #self.setLines(True)
##        for axis in [QwtPlot.xBottom, QwtPlot.xTop, QwtPlot.yLeft, QwtPlot.yRight]:
##        for axis in [QwtPlot.xBottom, QwtPlot.yLeft]:
##            self.setAxisAutoScale(axis)
        self.updateAxes()
        self.replot()

    def updateDistanceLines(self):
##        for k in self.distanceLineKeys:
####            self.removeCurve(k)
##            k.detach()

        N = len(self.mds.points)
        np = int(N*(N-1)/2. * self.proportionGraphed/100.)
        needlines = int(math.ceil((1 + math.sqrt(1+8*np)) / 2)) 

        if self.closestPairs is None or len(self.closestPairs) < np:
            matrix = self.mds.originalDistances
            hdist = [(-matrix[i,j], i, j) for i in range(needlines+1) for j in range(i)]
            import heapq
            heapq.heapify(hdist)
            while len(hdist) > np:
                heapq.heappop(hdist)
            for i in range(needlines+1, N):
                for j in range(i):
                    heapq.heappush(hdist, (-matrix[i, j], i, j))
                    heapq.heappop(hdist)
            self.closestPairs = []
            while hdist:
                d, i, j = heapq.heappop(hdist)
                self.closestPairs.append((-d, i, j))
                
        hdist = self.closestPairs[:np]
    
        black = QColor(192,192,192)
        maxdist = hdist[0][0]
        mindist = hdist[-1][0]
        if maxdist != mindist:
            k = 5 / (maxdist - mindist)**2
            for dist, i, j in hdist:
                pti, ptj = self.mds.points[i], self.mds.points[j]
                self.distanceLineKeys.append(self.addCurve("A", black, black, 0, QwtPlotCurve.Lines, xData=[pti[0],ptj[0]], yData=[pti[1],ptj[1]], lineWidth = max(1, (maxdist - dist)**2 * k)))
        else:
            for dist, i, j in hdist:
                pti, ptj = self.mds.points[i], self.mds.points[j]
                self.distanceLineKeys.append(self.addCurve("A", black, black, 0, QwtPlotCurve.Lines, xData=[pti[0],ptj[0]], yData=[pti[1],ptj[1]], lineWidth = 2))
        
                    
    def updateLinesRepaint(self):
        if self.mds:
            self.updateDistanceLines()
            self.repaint()

    def setPoints(self):
        if self.ShapeAttr==0 and self.SizeAttr==0 and self.NameAttr==0:
            colors=[c[self.ColorAttr] for c in self.colors]

            set=[]
            for c in colors:
                if  c not in set:
                    set.append(c)
            #set=reduce(lambda set,a: (not(a in set)) and set.append(a), colors, [])
            #set=sets.ImmutableSet([c[self.ColorAttr] for c in self.colors])

            dict={}
            for i in range(len(self.colors)):
                hsv = QColor(self.colors[i][self.ColorAttr]).getHsv()
                if dict.has_key(hsv):
                    dict[hsv].append(i)
                else:
                    dict[hsv]=[i]
            maxX, maxY = self.mds.points[0] if len(self.mds.points)>0 else (0, 0)
            minX, minY = self.mds.points[0] if len(self.mds.points)>0 else (0, 0)
            for color in set:
                #print len(dict[color.getHsv()]), color.name()
                X=[self.mds.points[i][0] for i in dict[QColor(color).getHsv()] if self.showFilled[i]]
                Y=[self.mds.points[i][1] for i in dict[QColor(color).getHsv()] if self.showFilled[i]]
                self.addCurve("A", color, color, self.PointSize, symbol=QwtSymbol.Ellipse, xData=X, yData=Y)
                
                X=[self.mds.points[i][0] for i in dict[QColor(color).getHsv()] if not self.showFilled[i]]
                Y=[self.mds.points[i][1] for i in dict[QColor(color).getHsv()] if not self.showFilled[i]]
                self.addCurve("A", color, color, self.PointSize, symbol=QwtSymbol.Ellipse, xData=X, yData=Y, showFilledSymbols=False)
        else:
            for i in range(len(self.colors)):
                self.addCurve("a", self.colors[i][self.ColorAttr], self.colors[i][self.ColorAttr], self.sizes[i][self.SizeAttr]*1.0/5*self.PointSize,
                              symbol=self.shapes[i][self.ShapeAttr], xData=[self.mds.points[i][0]],yData=[self.mds.points[i][1]], showFilledSymbols=self.showFilled[i])
                if self.NameAttr!=0:
                    self.addMarker(self.names[i][self.NameAttr], self.mds.points[i][0], self.mds.points[i][1], Qt.AlignRight)
        if len(self.mds.points)>0:
            X = [point[0] for point in self.mds.points]
            Y = [point[1] for point in self.mds.points]
            self.setAxisScale(QwtPlot.xBottom, min(X), max(X))
            self.setAxisScale(QwtPlot.yLeft, min(Y), max(Y))
            
                               
    def setLines(self, reset=False):
        if reset:
            #for k in self.lineKeys:
            #    removeCurve(k)
            self.lineKeys=[]
        if not getattr(self, "mds", None): return
        if self.NumStressLines<len(self.lineKeys):
            for c in self.lineKeys[self.NumStressLines:]:
                c[0].detach()
                if len(c) == 2: c[1].detach()
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
                    k1=self.addCurve("A", color, color, 0, QwtPlotCurve.Lines, xData=[xa,xb], yData=[ya,yb], lineWidth=1)
                    r=self.mds.distances[a,b]/max(self.mds.projectedDistances[a,b], 1e-6)
                    xa1=xa+(1-r)/2*(xb-xa)
                    xb1=xb+(1-r)/2*(xa-xb)
                    ya1=ya+(1-r)/2*(yb-ya)
                    yb1=yb+(1-r)/2*(ya-yb)
                    k2=self.addCurve("A", color, color, 0, QwtPlotCurve.Lines, xData=[xa1,xb1], yData=[ya1,yb1], lineWidth=4)
                    self.lineKeys.append( (k1,k2) )
                else:
                    color=Qt.red
                    r=self.mds.distances[a,b]/max(self.mds.projectedDistances[a,b], 1e-6)
                    xa1=(xa+xb)/2+r/2*(xa-xb)
                    xb1=(xa+xb)/2+r/2*(xb-xa)
                    ya1=(ya+yb)/2+r/2*(ya-yb)
                    yb1=(ya+yb)/2+r/2*(yb-ya)
                    k1=self.addCurve("A", color, color, 0, QwtPlotCurve.Lines, xData=[xa1,xb1], yData=[ya1,yb1], lineWidth=2)
                    self.lineKeys.append( (k1,) )

    def sendData(self, *args):
        pass

if __name__=="__main__":
    app=QApplication(sys.argv)
    w=OWMDS()
    w.show()
    data=orange.ExampleTable("../../doc/datasets/iris.tab")
##    data = orange.ExampleTable(r"E:\Development\Orange Datasets\UCI\iris.tab")
##    data=orange.ExampleTable("/home/ales/src/MDSjakulin/eu_nations.txt")
    matrix = orange.SymMatrix(len(data))
    dist = orange.ExamplesDistanceConstructor_Euclidean(data)
    matrix = orange.SymMatrix(len(data))
    matrix.setattr('items', data)
    for i in range(len(data)):
        for j in range(i+1):
            matrix[i, j] = dist(data[i], data[j])

    w.cmatrix(matrix)
    app.exec_()

