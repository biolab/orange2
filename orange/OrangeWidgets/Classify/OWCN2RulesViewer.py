"""
<name>CN2 Rules Viewer</name>
<description>Displays the rules of the CN2 classfier</description>
<icon>CN2RulesViwer.png</icon>
<priority>2120</priority>
"""

import orange
import orngCN2
from OWWidget import *
import OWGUI
import OWGraphTools
import qt
from qtcanvas import *
import sys
import re

class MyCanvasView(QCanvasView):
    def __init__(self, parent, *arg):
        apply(QCanvasView.__init__,(self,)+arg)
        self.parent=parent
        self.buttonPressed=False
        self.brush=QBrush(QColor("yellow"))
        self.lastIndex=-1
        self.flag=False

    def contentsMouseMoveEvent(self,e):
        if self.buttonPressed:
            obj=self.canvas().collisions(e.pos())
            if obj and obj[-1].__class__==QCanvasRectangle and obj[-1].index!=self.lastIndex:
                self.addSelection(obj[-1])
                self.lastIndex=obj[-1].index
    
    def contentsMousePressEvent(self,e):
        self.buttonPressed=True
        self.lastIndex=-1
               
    def contentsMouseReleaseEvent(self,e):
        self.flag=True
        obj=self.canvas().collisions(e.pos())
        if obj and obj[-1].__class__==QCanvasRectangle and obj[-1].index!=self.lastIndex:
            self.addSelection(obj[-1])
        self.flag=False
        self.buttonPressed=False

    def addSelection(self, rect):
        index=rect.index
        if (not self.buttonPressed and self.flag) or not self.parent.ctrlPressed:
            self.parent.selRect=[]
            for r in self.parent.rectObj:
                r.setBrush(QBrush(Qt.NoBrush))
        if rect in self.parent.selRect:
            self.parent.selRect.remove(rect)
            rect.setBrush(QBrush(Qt.NoBrush))
        else:
            self.parent.selRect.append(rect)
            rect.setBrush(self.brush)
        self.parent.canvas.update()
        #print self.parent.selRect
        self.parent.select()
        
            


class OWCN2RulesViewer(OWWidget):
    settingsList=["RuleLen","RuleQ","Coverage","Commit","Rule","Sort"]
    callbackDeposit=[]
    def __init__(self, parent=None, signalManager=None):
        OWWidget.__init__(self, parent, signalManager,"CN2 Rules Viewer")
        
        self.inputs=[("CN2UnorderedClassifier", orngCN2.CN2UnorderedClassifier, self.data)]
        self.outputs=[("ExampleTable", ExampleTable)]
        self.RuleLen=1
        self.RuleQ=1
        self.Coverage=1
        self.Class=1
        self.Dist=1
        self.DistBar=1
        self.Commit=0
        self.Rule=1
        self.Sort=0
        self.loadSettings()
        layout=QVBoxLayout(self.mainArea, QVBoxLayout.TopToBottom,0)
        self.canvas=QCanvas()
        self.canvasView=MyCanvasView(self, self.canvas, self.mainArea)
        self.canvasView.setCanvas(self.canvas)
        self.canvasView.show()
        self.headerCanvas=QCanvas()
        self.headerView=QCanvasView(self.headerCanvas, self.mainArea)
        self.headerView.setMaximumHeight(25)
        self.headerView.setHScrollBarMode(QScrollView.AlwaysOff)
        self.headerView.setVScrollBarMode(QScrollView.AlwaysOff)
        layout.addWidget(self.headerView)
        layout.addWidget(self.canvasView)
        box=OWGUI.widgetBox(self.controlArea,"Show info on")
        OWGUI.checkBox(box,self,"RuleLen","Rule length",callback=self.showRules)
        OWGUI.checkBox(box,self,"RuleQ","Rule quality",callback=self.showRules)
        OWGUI.checkBox(box,self,"Coverage","Coverage",callback=self.showRules)
        OWGUI.checkBox(box,self,"Class","Predicted class", callback=self.showRules)
        OWGUI.checkBox(box,self,"Dist","Distribution", callback=self.showRules)
        OWGUI.checkBox(box,self,"DistBar","Distribution(Bar)",callback=self.showRules)

        self.sortBox=OWGUI.comboBox(self.controlArea, self, "Sort", box="Sorting",
                                    items=["Rule length", "Rule quality", "Coverage", "Predicted class",
                                           "Distribution"]
                                    ,callback=self.drawRules)
        box=OWGUI.widgetBox(self.controlArea,1)
        OWGUI.checkBox(box,self,"Commit", "Commit on change")
        OWGUI.button(box,self,"&Commit",callback=self.commit)
        
        OWGUI.button(self.controlArea,self,"&Save rules to file",callback=self.saveRules)

        self.examples=None
        self.obj=[]
        self.selRect=[]
        self.rectObj=[]
        
        self.ctrlPressed=False
        self.setFocusPolicy(QWidget.StrongFocus)

        self.connect(self.canvasView.horizontalScrollBar(),SIGNAL("valueChanged(int)"),
                self.headerView.horizontalScrollBar().setValue)

    def clear(self):
        for e in self.obj:
            e.setCanvas(None)
        self.obj=[]

    def showRules(self):
        self.clear()
        text=[]
        items=[]
        for i, r in enumerate(self.rules):
            l=[str(r.complexity), "%.3f" % r.quality, "%10s" % str(len(self.classifier.examples.filterref(r.filter))),
                "%10s" % str(r.classifier.defaultValue), str(r.classDistribution), r]
            text.append(l)
        self.text=text
        #self.items=items
        self.drawRules()
        

    def drawRules(self):
        self.sort()
        self.clear()
        for r in self.rectObj:
            r.setCanvas(None)
        self.rectObj=[]
        text=self.text
        filter=[self.RuleLen, self.RuleQ, self.Coverage, self.Class, self.Dist, self.DistBar,self.Rule]
        l=[]
        a=["Length","Quality","Coverage","Class","Distribution", "Distribution(Bar)", "Rule"]
        for i, k in enumerate(a):
            if filter[i]:
                t=QCanvasText(self.headerCanvas)
                t.setText(k)
                l.append(t)
        l.append(QCanvasText(self.canvas))
        items=[]
        items.append(l)
        self.obj.extend(l)
        
        for text in self.text:
            l=[]
            if self.RuleLen:
                t=QCanvasText(self.canvas)
                t.setText("%10s" % text[0])
                l.append(t)
            if self.RuleQ:
                t=QCanvasText(self.canvas)
                t.setText("%10s" % text[1])
                l.append(t)
            if self.Coverage:
                t=QCanvasText(self.canvas)
                t.setText("%15s" % text[2])
                l.append(t)
            if self.Class:
                t=QCanvasText(self.canvas)
                t.setText("%15s" % text[3])
                l.append(t)
            if self.Dist:
                t=QCanvasText(self.canvas)
                t.setText("%15s" % text[4])
                l.append(t)
            if self.DistBar:
                t=DistBar(text[4],text[-1],self.canvas)
                l.append(t)                    
            if self.Rule:
                t=QCanvasText(self.canvas)
                t.setText(self.ruleText(text[-1]))
                l.append(t)
            l.append(QCanvasText(self.canvas))
            self.obj.extend(l)
            items.append(l)
        #print len(items)
                
        textMapV=[10]+map(lambda s:max([t.boundingRect().height()+10 for t in s]), items[1:])
        textMapH=[[s[i].boundingRect().width()+10 for s in items] for i in range(len(items[0]))]
        textMapH=[10]+map(lambda s:max(s), textMapH)

        #print len(textMapV)
        for i in range(1,len(textMapV)):
            textMapV[i]+=textMapV[i-1]

        for i in range(1,len(textMapH)):
            textMapH[i]+=textMapH[i-1]
            
        for i in range(1,len(textMapV)):
            for j in range(len(textMapH)-2):
                items[i][j].move(textMapH[j], textMapV[i-1])
                items[i][j].setZ(0)
                items[i][j].show()
            r=QCanvasRectangle(textMapH[0],textMapV[i], textMapH[-1], textMapV[i-1]-textMapV[i], self.canvas)
            r.setZ(-20)
            r.setPen(QPen(Qt.NoPen))
            r.show()
            r.index=i+1
            r.rule=self.text[i-1][-1]
            self.obj.append(r)
            self.rectObj.append(r)
            
        self.canvas.resize(textMapH[-1], textMapV[-1])
        for i,t in enumerate(items[0][:-1]):
            t.move(textMapH[i],0)
            t.show()
        self.headerCanvas.update()
        self.headerCanvas.resize(textMapH[-1],20)
        self.canvas.update()
        
    def ruleText(self, r):
        str=orngCN2.ruleToString(r)
        list=re.split("([0-9.]*)",str)
        #print list
        for i in range(len(list)):
            try:
                f=float(list[i])
                list[i]="%.3f" % f
                t=int(list[i])
                list[i]=str(t)
            except:
                pass
        #print list
        str="".join(list)
        list=re.split("<[0-9., ]*>$", str)
        #print list
        str=list[0]
        str="AND\n   ".join(str.split("AND"))
        str="\nTHEN".join(str.split("THEN"))
        return str

    def select(self):
        examples=[]
        source=self.classifier.examples
        for r in self.selRect:
            examples.extend(source.filterref(r.rule.filter))
            r.rule.filter.negate=1
            source=source.filterref(r.rule.filter)
            r.rule.filter.negate=0
        if not examples:
            self.examples=None
            self.commit()
            return
        self.examples=examples
        if self.Commit:
            self.commit()
            
    def data(self, classifier):
        #print classifier
        if classifier:
            self.clear()
            self.classifier=classifier
            self.rules=classifier.rules
            self.showRules()
        else:
            self.rules=None
            self.clear()
        self.examples=None
        self.commit()

    def compare(self,a,b):
        if str(a[self.sortBy])<str(b[self.sortBy]):
            return -1
        elif str(a[self.sortBy])>str(b[self.sortBy]):
            return 1
        else:
            return 0
        
    def sort(self):
        text=[]
        if self.Sort>5:
            self.sortBy=self.Sort+1
        else:
            self.sortBy=self.Sort
        self.text.sort(self.compare)
        #print self.text
        if self.Sort>=1:
            self.text.reverse()
        #print self.sortMap
    
        
            

    def commit(self):
        if self.examples:
            self.send("ExampleTable",orange.ExampleTable(self.examples))
        else:
            self.send("ExampleTable",None)

    def saveRules(self):
        fileName=str(QFileDialog.getSaveFileName("Rules.txt",".txt"))
        try:
            f=open(fileName,"w")
        except :
            return
        for r in self.rules:
            f.write(orngCN2.ruleToString(r)+"\n")
            
    def keyPressEvent(self, key):
        if key.key()==Qt.Key_Control:
            self.ctrlPressed=True
        else:
            key.ignore()

    def keyReleaseEvent(self, key):
        if key.key()==Qt.Key_Control:
            self.ctrlPressed=False
        else:
            key.ignore()

barWidth=80
barHeight=20
class DistBar(QCanvasRectangle):
    def __init__(self, dist,rule,canvas, *args):
        apply(QCanvasRectangle.__init__,(self,canvas)+args)
        self.dist=dist
        self.rule=rule
        self.canvas=canvas
        self.rect=[]
        distSum=sum(rule.classDistribution)
        classColor=OWGraphTools.ColorPaletteHSV(len(rule.classDistribution))
        defClass=rule.classifier.defaultValue
        m=max(rule.classDistribution)
        #print len(rule.classDistribution)
        for i in range(len(rule.classDistribution)):
            dist=rule.classDistribution
            r=QCanvasRectangle(self.canvas)
            r.setSize(dist[i]/distSum*barWidth,barHeight)
            r.setPen(QPen(QColor(classColor[i]),2))
            if dist[i]==m:
                r.setBrush(QBrush(classColor[i]))
            self.rect.append(r)
        #print len(self.rect)
        
        
    def move(self, x,y):
        pos=0
        for r in self.rect:
            r.move(x+pos,y+5)
            pos+=r.width()
            
    def show(self):
        for r in self.rect:
            r.show()
    def setCanvas(self, canvas):
        for r in self.rect:
            r.setCanvas(canvas)
    def text(self):
        return str(self.dist)
    
    
if __name__=="__main__":
    ap=QApplication(sys.argv)
    w=OWCN2RulesViewer()
    ap.setMainWidget(w)
    data=orange.ExampleTable("../../doc/datasets/titanic.tab")
    l=orngCN2.CN2UnorderedLearner()
    l.ruleFinder.ruleStoppingValidator=orange.RuleValidator_LRS()
    w.data(l(data))
    w.data(l(data))
    w.show()
    ap.exec_loop()
    
