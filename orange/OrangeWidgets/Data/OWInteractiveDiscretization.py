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

        self.setAxisScale(QwtPlot.yRight, -0.0, 1.0, 0.0)
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
        self.data=None

    #def computeScore(self, cuts, res=30):
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
        if not data:
            self.vars=[]
            self.cutLineKeys=[]
            self.rugKeys=[]
            self.curAttribute=0            
            return
        self.data=data
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
        color=self.classColors[self.master.targetClass]
        pen=QPen(color)
        var=self.vars[self.master.attribute]
        targetClass=self.data.domain.classVar.values[self.master.targetClass]
        for e, key in zip(self.data, self.rugKeys):
            self.setCurveYAxis(key, QwtPlot.yRight)
            curve=self.curve(key)
            curve.setData([float(e[var]), float(e[var])], e[-1]==targetClass and [1.0, 0.98] or [0.02, 0.0])
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
        curve.setPen(QPen(self.classColors[self.master.targetClass]))
    
    def plotCutLines(self):
        self.cutLineKeys=[self.insertCurve("") for c in self.curCutPoints]
        for i, key in enumerate(self.cutLineKeys):
            self.setCurveYAxis(key, QwtPlot.yRight)
            cut=self.curCutPoints[i]
            curve=self.curve(key)
            curve.setData([cut, cut], [1.0, 0.0])
            curve.curveInd=i
    

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
        curve.setData([cut, cut], [1.0, 0.0])
        self.cutLineKeys.append(curveKey)
        curve.curveInd=len(self.cutLineKeys)-1
        return curve
        
    def onMousePressed(self, e):
        self.mouseCurrentlyPressed=1
        cut=self.invTransform(QwtPlot.xBottom, e.x())
        curve=self.getCutCurve(cut)
        if curve:
            if e.button()==Qt.RightButton:
                self.curCutPoints.pop(curve.curveInd)
                self.computeBaseScore()
                self.replotAll()
            else:
                self.selectedCutPoint=curve
        else:
            self.selectedCutPoint=self.addCutPoint(cut)
            self.update()

    def onMouseMoved(self, e):
        if self.mouseCurrentlyPressed:
            if self.selectedCutPoint:
                pos=self.invTransform(QwtPlot.xBottom, e.x())
                self.curCutPoints[self.selectedCutPoint.curveInd]=pos
                self.selectedCutPoint.setData([pos, pos], [1.0, 0.0])
                self.computeLookaheadScore(pos)
                self.plotLookaheadCurve()
                self.curve(self.lookaheadCurveKey).setEnabled(1)
                self.update()
        elif self.getCutCurve(self.invTransform(QwtPlot.xBottom, e.x())):
            self.canvas().setCursor(Qt.sizeHorCursor)
        else:
            self.canvas().setCursor(Qt.arrowCursor)
                                  
    def onMouseReleased(self, e):
        self.mouseCurrentlyPressed=0
        self.selectedCutPoint=None
        self.computeBaseScore()
        self.plotBaseCurve()
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
        self.rugKeys=[]
        self.cutLineKeys=[]
        self.probCurveKey=self.insertCurve("")
        self.baseCurveKey=self.insertCurve("")
        self.lookaheadCurveKey=self.insertCurve("")
        
        self.cutPoints[self.curAttribute][0]=self.curCutPoints
        self.curCutPoints=self.cutPoints[self.master.attribute][0]
        self.curAttribute=self.master.attribute

        self.computeBaseScore()
        #self.computeLookaheadScore()
        self.plotRug()
        self.plotBaseCurve()
        #self.plotLookaheadCurve()
        self.plotProbCurve()
        self.plotCutLines()
        self.updateLayout()
        self.update()        
        

class OWInteractiveDiscretization(OWWidget):
    settingsList=["attribute", "targetClass", "showBaseLine", "showLookaheadLine", "measure", "targetClass", "showTargetClassProb",
                  "showRug", "snap", "discretization", "intervals"]
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
        self.measure=0
        self.targetClass=0
        self.discretization=0
        self.intervals=3
        self.loadSettings()
        self.inputs=[("Examples", ExampleTableWithClass, self.cdata)]
        self.outputs=[("Examples", ExampleTableWithClass)]
        self.measures=[("Gain ratio", orange.MeasureAttribute_gainRatio),
                       ("Gini", orange.MeasureAttribute_gini),
                       ("Relevance", orange.MeasureAttribute_relevance)]
                       #("chi-square",)]
        self.discretizationMethods=[("Entropy discretization", orange.EntropyDiscretization),
                                    ("Equi-distance discretization", orange.EquiDistDiscretization)]
        self.layout=QVBoxLayout(self.mainArea, QVBoxLayout.TopToBottom,0)
        self.graph=DiscGraph(self, self.mainArea)
        #self.canvasView=DiscCanvasView(self.canvas, self.mainArea )
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
        
        box=OWGUI.widgetBox(self.controlArea, "Options")
        box.setSizePolicy(QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed))
        OWGUI.checkBox(box, self, "showBaseLine", "Show base line", callback=self.graph.replotAll)
        OWGUI.checkBox(box, self, "showLookaheadLine", "Show lookahead line", callback=self.graph.replotAll)
        OWGUI.checkBox(box, self, "showTargetClassProb", "Show target class prob", callback=self.graph.replotAll)
        OWGUI.checkBox(box, self, "showRug", "Show rug", callback=self.graph.replotAll)
        #OWGUI.checkBox(box, self, "snap", "Snap to grid")
        QVBox(self.controlArea)
        OWGUI.button(self.controlArea, self,"&Commit", callback=self.commit)

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
            self.graph.setData(None)
            self.send("Examples", None)

    def setDiscMethod(self):
        self.intervalSlider.parentWidget().setEnabled(self.discretization!=0)

    def discretize(self):
        if self.discretization==0:
            entro=orange.EntropyDiscretization()
            disc=entro(self.vars[self.attribute], self.data)
            self.graph.curCutPoints=list(disc.getValueFrom.transformer.points)
        elif self.discretization==1:
            inter=orange.EquiDistDiscretization(numberOfIntervals = self.intervals)
            disc=inter(self.vars[self.attribute], self.data)
            trans=disc.getValueFrom.transformer
            self.graph.curCutPoints=[trans.firstCut+i*trans.step for i in range(self.intervals-1)]
        self.graph.replotAll()
        
    def commit(self):
        newattrs=[]
        i=0
        for attr in self.data.domain.attributes:
            if attr.varType==orange.VarTypes.Continuous:
                if self.graph.cutPoints[i][0]:
                    idisc=orange.IntervalDiscretizer(points=self.graph.cutPoints[i][0])
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
