"""
<name>Interactive Discretization</name>
<description>Interactive discretization of continuous attributes</description>
<icon>icons/InteractiveDiscretization.png</icon>
<priority>2105</priority>
<contact>Ales Erjavec (ales.erjavec(@at@)fri.uni-lj.si)</contact>
"""

import orange
from OWWidget import *
from OWGraph import *
from qt import *
from qtcanvas import *
import OWGUI, OWGraphTools

def frange(low, up, steps):
    inc=(up-low)/steps
    return [low+i*inc for i in range(steps)]

class DiscGraph(OWGraph):
    def __init__(self, master, *args):
        OWGraph.__init__(self, *args)
        self.master=master
        self.curCutPoints=[]
        self.cutPoints=[[],[],[]]
        self.rugKeys=[]
        self.cutLineKeys=[]
        self.probCurveKey=self.insertCurve("")
        self.baseCurveKey=self.insertCurve("")
        self.lookaheadCurveKey=self.insertCurve("")

        self.setAxisScale(QwtPlot.yRight, 0.0, 1.0, 0.0)
        self.setYLaxisTitle("Attribute score")
        self.setXaxisTitle("Attribute value")
        self.setYRaxisTitle("Target class probability")
        self.setShowYRaxisTitle(1)
        self.setShowYLaxisTitle(1)
        self.setShowXaxisTitle(1)
        self.enableYRightAxis(1)        

        self.resolution=50
        self.setCursor(Qt.arrowCursor)
        self.canvas().setCursor(Qt.arrowCursor)
        self.curAttribute=0
        self.curCandidate=0
        self.data=None
    
    def snap(self, point):
        if not self.master.snap:
            return point
        varInd=self.master.attribute
        minVal=self.minVal[varInd]
        maxVal=self.maxVal[varInd]
        order=math.ceil(math.log(maxVal-minVal, 10))
        f=point*math.pow(10, -(order-1))
        s="%.1f" % f
        f=float(s)/math.pow(10, -(order-1))
        return f

    def computeBaseScore(self):
        varInd=self.master.attribute
        minVal=self.minVal[varInd]
        maxVal=self.maxVal[varInd]
        candidateSplits=frange(minVal, maxVal, self.resolution)
        score=[]
        for cut in candidateSplits:
            cutPoints=list(self.curCutPoints)
            if cut not in cutPoints:
                cutPoints.append(cut)
                cutPoints.sort()
            idisc=orange.IntervalDiscretizer(points=cutPoints)
            m=self.master.measures[self.master.measure][1](idisc.constructVariable(self.vars[varInd]), self.data)
            score.append(m)
        self.baseCurveX=candidateSplits
        self.baseCurveY=score
            
    
    def computeLookaheadScore(self, split):
        varInd=self.master.attribute
        minVal=self.minVal[varInd]
        maxVal=self.maxVal[varInd]
        candidateSplits=frange(minVal, maxVal, self.resolution)
        score=[]
        for cut in candidateSplits:
            cutPoints=list(self.curCutPoints+[split])
            if cut not in cutPoints:
                cutPoints.append(cut)
                cutPoints.sort()
            idisc=orange.IntervalDiscretizer(points=cutPoints)
            m=self.master.measures[self.master.measure][1](idisc.constructVariable(self.vars[varInd]), self.data)
            score.append(m)
        self.lookaheadCurveX=candidateSplits
        self.lookaheadCurveY=score
    
    def setData(self, data=None):
        self.clear()
        self.data=data
        if not data:
            self.vars=[]
            self.cutLineKeys=[]
            self.rugKeys=[]
            self.curAttribute=0
            self.curCutPoints=[]
            return
        self.vars=self.master.vars
        self.condProb=[orange.ConditionalProbabilityEstimatorConstructor_loess(orange.ContingencyAttrClass(var, self.data)) for var in self.vars]
        self.cutPoints=[[[],[],[]] for i in range(len(self.vars))]
        self.minVal=[min([float(e[var]) for e in self.data if not e[var].isSpecial()]) for var in self.vars]
        self.maxVal=[max([float(e[var]) for e in self.data if not e[var].isSpecial()]) for var in self.vars]
        self.classColors=OWGraphTools.ColorPaletteHSV(len(self.data.domain.classVar.values))

        self.replotAll()

    def plotRug(self):
        if not self.master.showRug:
            return
        if not self.rugKeys:
            self.rugKeys=[self.insertCurve("") for e in self.data]
        color=Qt.blue #self.classColors[self.master.targetClass]
        pen=QPen(color)
        var=self.vars[self.master.attribute]
        targetClass=self.data.domain.classVar.values[self.master.targetClass]
        for e, key in zip(self.data, self.rugKeys):
            if e[var].isSpecial():
                continue
            self.setCurveYAxis(key, QwtPlot.yRight)
            curve=self.curve(key)
            curve.setData([float(e[var]), float(e[var])], e[-1]==targetClass and [1.0, 0.98] or [0.04, 0.025])
            curve.setPen(pen)

    def plotBaseCurve(self):
        if not self.master.showBaseLine:
            return
        curve=self.curve(self.baseCurveKey)
        curve.setData(self.baseCurveX, self.baseCurveY)
        curve.setPen(QPen(Qt.black, 2))

    def plotLookaheadCurve(self):
        if not self.master.showLookaheadLine:
            return
        curve=self.curve(self.lookaheadCurveKey)
        curve.setData(self.lookaheadCurveX, self.lookaheadCurveY)
        #curve.setPen(QPen(Qt.black, 2))

    def plotProbCurve(self):
        if not self.master.showTargetClassProb:
            return
        varInd=self.master.attribute
        targetClass=self.data.domain.classVar.values[self.master.targetClass]
        var=self.vars[varInd]
        X=frange(self.minVal[varInd], self.maxVal[varInd], self.resolution)
        Y=[self.condProb[varInd](var(x))[targetClass] for x in X]
        self.setCurveYAxis(self.probCurveKey, QwtPlot.yRight)
        curve=self.curve(self.probCurveKey)
        curve.setData(X,Y)
        curve.setPen(QPen(Qt.blue))#self.classColors[self.master.targetClass]))

    def plotCutLines(self):
        self.cutLineKeys=[self.insertCurve("") for c in self.curCutPoints]
        for i, key in enumerate(self.cutLineKeys):
            self.setCurveYAxis(key, QwtPlot.yRight)
            cut=self.curCutPoints[i]
            curve=self.curve(key)
            curve.setData([cut, cut], [1.0, 0.015])
            curve.curveInd=i
        if not self.master.showAllSets:
            return
        colors=[Qt.red, Qt.green, Qt.blue]
        for i in range(3):
            symbol=QwtSymbol(QwtSymbol.Triangle, QBrush(colors[i]), QPen(colors[i]), QSize(6,6))
            for cut in self.cutPoints[self.master.attribute][i]:
                key=self.insertCurve("")
                self.setCurveYAxis(key, QwtPlot.yRight)
                curve=self.curve(key)
                curve.setData([cut], [0.021-i*0.013])
                curve.setSymbol(symbol)
            #self.curve(self.insertCurve("")).setData([self.minVal...

    def getCutCurve(self, cut):
        found=False
        ccc=self.transform(QwtPlot.xBottom, cut)
        for i,c in enumerate(self.curCutPoints):
            cc=self.transform(QwtPlot.xBottom, c)
            if abs(cc-ccc)<3:
                curve=self.curve(self.cutLineKeys[i])
                curve.curveInd=i
                return curve
        return None

    def addCutPoint(self, cut):
        self.curCutPoints.append(cut)
        curveKey=self.insertCurve("")
        self.setCurveYAxis(curveKey, QwtPlot.yRight)
        curve=self.curve(curveKey)
        curve.setData([cut, cut], [1.0, 0.015])
        self.cutLineKeys.append(curveKey)
        curve.curveInd=len(self.cutLineKeys)-1
        return curve
        
    def onMousePressed(self, e):
        if not self.data: return
        self.mouseCurrentlyPressed=1
        cut=self.invTransform(QwtPlot.xBottom, e.x())
        curve=self.getCutCurve(cut)
        if not curve:
            cut=self.snap(cut)
            curve=self.getCutCurve(cut)
        if curve:
            if e.button()==Qt.RightButton:
                self.curCutPoints.pop(curve.curveInd)
                self.computeBaseScore()
                self.replotAll()
            else:
                cut=self.curCutPoints.pop(curve.curveInd)
                self.computeBaseScore()
                self.replotAll()
                self.selectedCutPoint=self.addCutPoint(cut)
        else:
            self.selectedCutPoint=self.addCutPoint(cut)
            #self.replotAll()
            self.update()

    def onMouseMoved(self, e):
        if not self.data: return
        if self.mouseCurrentlyPressed:
            if self.selectedCutPoint:
                pos1=self.invTransform(QwtPlot.xBottom, e.x())
                pos=self.snap(pos1)
                if self.curCutPoints[self.selectedCutPoint.curveInd]==pos:
                    return
                if pos1>self.maxVal[self.master.attribute] or pos1<self.minVal[self.master.attribute]:
                    self.curCutPoints.pop(self.selectedCutPoint.curveInd)
                    self.computeBaseScore()
                    self.replotAll()
                    self.mouseCurrentlyPressed=0
                    return
                self.curCutPoints[self.selectedCutPoint.curveInd]=pos
                self.selectedCutPoint.setData([pos, pos], [1.0, 0.015])
                self.computeLookaheadScore(pos)
                self.plotLookaheadCurve()
                self.curve(self.lookaheadCurveKey).setEnabled(1)
                self.update()
        elif self.getCutCurve(self.invTransform(QwtPlot.xBottom, e.x())):
            self.canvas().setCursor(Qt.sizeHorCursor)
        else:
            self.canvas().setCursor(Qt.arrowCursor)
                                  
    def onMouseReleased(self, e):
        if not self.data: return
        self.mouseCurrentlyPressed=0
        self.selectedCutPoint=None
        self.computeBaseScore()
        self.replotAll()
        self.curve(self.lookaheadCurveKey).setEnabled(0)
        self.update()

    def attributeChanged(self):
        self.replotAll()

    def targetClassChanged(self):
        self.plotRug()
        self.plotProbCurve()
        self.update()

    def replotAll(self):
        self.clear()
        if not self.data:
            return 
        self.rugKeys=[]
        self.cutLineKeys=[]
        self.probCurveKey=self.insertCurve("")
        self.baseCurveKey=self.insertCurve("")
        self.lookaheadCurveKey=self.insertCurve("")
        
        self.cutPoints[self.curAttribute][self.curCandidate]=self.curCutPoints
        self.curCutPoints=self.cutPoints[self.master.attribute][self.master.candidate[0]]
        self.curAttribute=self.master.attribute
        self.curCandidate=self.master.candidate[0]

        self.computeBaseScore()
        self.plotRug()
        self.plotBaseCurve()
        self.plotProbCurve()
        self.plotCutLines()
        self.updateLayout()
        self.update()        
        
def pixmap(color):
    pixmap = QPixmap()
    pixmap.resize(13,13)
    painter = QPainter()
    painter.begin(pixmap)
    painter.setBrush(Qt.white)
    painter.setPen(Qt.white)
    painter.drawRect(0,0,13,13)
    painter.setPen( color )
    painter.setBrush( color )
    points=QPointArray(3)
    points.setPoint(0,12,12)
    points.setPoint(1,0,13)
    points.setPoint(2,7,0)
    painter.drawPolygon(points)
    painter.end()
    return pixmap

class OWInteractiveDiscretization(OWWidget):
    settingsList=["attribute", "targetClass", "showBaseLine", "showLookaheadLine", "measure", "targetClass", "showTargetClassProb",
                  "showRug", "snap", "showAllSets", "discretization", "intervals"]
    callbackDeposit=[]
    def __init__(self, parent=None, signalManager=None, name="Interactive Discretization"):
        OWWidget.__init__(self, parent, signalManager, name)
        self.attribute=0
        self.targetClass=0
        self.showBaseLine=1
        self.showLookaheadLine=1
        self.showTargetClassProb=1
        self.showRug=1
        self.snap=1
        self.showAllSets=1
        self.measure=0
        self.targetClass=0
        self.discretization=0
        self.intervals=3
        self.candidate=[0]
        self.sets=[]
        self.loadSettings()
        self.inputs=[("Examples", ExampleTableWithClass, self.cdata)]
        self.outputs=[("Examples", ExampleTableWithClass)]
        self.measures=[("Information gain", orange.MeasureAttribute_info),
                       #("Gain ratio", orange.MeasureAttribute_gainRatio),
                       ("Gini", orange.MeasureAttribute_gini),
                       ("Relevance", orange.MeasureAttribute_relevance)]
                       #("chi-square",)]
        self.discretizationMethods=[("Entropy discretization", orange.EntropyDiscretization),
                                    ("Equal-Frequancy discretizaion", orange.EquiNDiscretization), 
                                    ("Equal-distance discretization", orange.EquiDistDiscretization)]
        self.layout=QVBoxLayout(self.mainArea, QVBoxLayout.TopToBottom,0)
        self.graph=DiscGraph(self, self.mainArea)

        self.layout.addWidget(self.graph)
        box=QVBox(self.controlArea)
        box.setSizePolicy(QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed))
        self.attributeCombo=OWGUI.comboBox(box, self, "attribute", "Attribute", callback=self.graph.replotAll)
        self.targetCombo=OWGUI.comboBox(box, self, "targetClass", "Target Class", callback=self.graph.replotAll)
        self.measureCombo=OWGUI.comboBox(box, self, "measure", "Measure", items=[e[0] for e in self.measures], callback=self.graph.replotAll)

        box=OWGUI.widgetBox(self.controlArea, "Automatic Discretization")
        box.setSizePolicy(QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed))
        OWGUI.comboBox(box, self, "discretization", "Discretization method", items=[e[0] for e in self.discretizationMethods], callback=self.setDiscMethod)
        self.intervalSlider=OWGUI.hSlider(box, self, "intervals", "Intervals", 2, 10)
        OWGUI.button(box, self, "&Apply", callback=self.discretize)
        self.setDiscMethod()
        box=OWGUI.listBox(self.controlArea, self, "candidate", "sets" , "Discretization Sets", callback=self.graph.replotAll)
        
        box=OWGUI.widgetBox(self.controlArea, "Options")
        box.setSizePolicy(QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed))
        OWGUI.checkBox(box, self, "showBaseLine", "Show discretization gain", callback=self.graph.replotAll)
        OWGUI.checkBox(box, self, "showLookaheadLine", "Show lookahead gain", callback=self.graph.replotAll)
        OWGUI.checkBox(box, self, "showTargetClassProb", "Show target class prob", callback=self.graph.replotAll)
        OWGUI.checkBox(box, self, "showRug", "Show rug", callback=self.graph.replotAll)
        OWGUI.checkBox(box, self, "showAllSets", "Show all sets", callback=self.graph.replotAll)
        OWGUI.checkBox(box, self, "snap", "Snap to grid")
        
        QVBox(self.controlArea)
        OWGUI.button(self.controlArea, self,"&Commit", callback=self.commit)
        cc=OWGUI.attributeIconDict
        OWGUI.attributeIconDict={1:pixmap(Qt.red), 2:pixmap(Qt.green), 3:pixmap(Qt.blue), -1:pixmap(Qt.blue)}
        self.sets=[("First set",1), ("Second set", 2), ("Third set", 3)]
        OWGUI.attributeIconDict=cc
        self.candidate=[0]
        self.data=None

    def cdata(self, data=None):
        if data:
            self.data=data
            self.attributeCombo.clear()
            self.vars=[]
            for v in data.domain.attributes:
                if v.varType==orange.VarTypes.Continuous:
                    self.attributeCombo.insertItem(v.name)
                    self.vars.append(v)
            if self.attribute<len(self.vars):
                self.attributeCombo.setCurrentItem(self.attribute)
            else:
                #self.attributeCombo.setCurrentItem(0)
                self.attribute=0
            
            self.targetCombo.clear()
            for v in data.domain.classVar.values:
                self.targetCombo.insertItem(str(v))
            if self.targetClass<len(self.data.domain.classVar.values):
                self.targetCombo.setCurrentItem(self.targetClass)
            else:
                self.targetCombo.setCurrentItem(0)
                self.targetClass=0
            if not self.vars:
                self.graph.setData(None)
                self.send("Examples", self.data)
                return
            self.graph.setData(self.data)
        else:
            self.attributeCombo.clear()
            self.targetCombo.clear()
            self.graph.setData(None)
            self.send("Examples", None)

    def setDiscMethod(self):
        self.intervalSlider.parentWidget().setEnabled(self.discretization!=0)

    def discretize(self):
        if not self.data: return
        if self.discretization==0:
            entro=orange.EntropyDiscretization()
            disc=entro(self.vars[self.attribute], self.data)
            self.graph.curCutPoints=list(disc.getValueFrom.transformer.points)
        elif self.discretization==1:
            inter=orange.EquiNDiscretization(numberOfIntervals = self.intervals)
            disc=inter(self.vars[self.attribute], self.data)
            trans=disc.getValueFrom.transformer
            self.graph.curCutPoints=list(trans.points)
        elif self.discretization==2:
            inter=orange.EquiDistDiscretization(numberOfIntervals = self.intervals)
            disc=inter(self.vars[self.attribute], self.data)
            trans=disc.getValueFrom.transformer
            self.graph.curCutPoints=[trans.firstCut+i*trans.step for i in range(self.intervals-1)]
        self.graph.replotAll()

    def commit(self):
        if self.data:
            newattrs=[]
            i=0
            for attr in self.data.domain.attributes:
                if attr.varType==orange.VarTypes.Continuous:
                    if self.graph.cutPoints[i][self.candidate[0]]:
                        idisc=orange.IntervalDiscretizer(points=self.graph.cutPoints[i][self.candidate[0]])
                        newattrs.append(idisc.constructVariable(attr))
                        i+=1
                    else:
                        newattrs.append(attr)
                        i+=1
                else:
                    newattrs.append(attr)
            newdata=self.data.select(newattrs+[self.data.domain.classVar])
            self.send("Examples", newdata)

import sys
if __name__=="__main__":
    app=QApplication(sys.argv)
    w=OWInteractiveDiscretization()
    app.setMainWidget(w)
    w.show()
    d=orange.ExampleTable("../../doc/datasets/iris.tab")
    w.cdata(d)
    app.exec_loop()
    w.saveSettings()
