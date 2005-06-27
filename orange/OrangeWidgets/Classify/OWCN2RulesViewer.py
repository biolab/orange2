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
        if index in self.parent.selRect:
            self.parent.selRect.remove(index)
            rect.setBrush(QBrush(Qt.NoBrush))
        else:
            self.parent.selRect.append(index)
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
        b=OWGUI.widgetBox(self.controlArea, 1)
        box=OWGUI.widgetBox(b,1)
        OWGUI.checkBox(box,self,"RuleLen","Rule length",callback=self.showRules)
        OWGUI.checkBox(box,self,"RuleQ","Rule quality",callback=self.showRules)
        OWGUI.checkBox(box,self,"Coverage","Covearge",callback=self.showRules)

        self.sortBox=OWGUI.comboBox(b, self, "Sort", box="Sorting", label="Sort by", callback=self.drawRules)
        box=OWGUI.widgetBox(b,1)
        OWGUI.checkBox(box,self,"Commit", "Commit on change")
        OWGUI.button(box,self,"&Commit",callback=self.commit)
        
        OWGUI.button(self.controlArea,self,"&Save rules to file",callback=self.saveRules)

        self.examples=None
        self.obj=[]
        self.selRect=[]
        self.rectObj=[]
        
        self.ctrlPressed=False
        self.setFocusPolicy(QWidget.StrongFocus)

    def clear(self):
        for e in self.obj:
            e.setCanvas(None)
        self.obj=[]

    def showRules(self):
        self.clear()
        text=[]
        items=[]
        if self.RuleLen:
            items.append("Lenght")
        if self.RuleQ:
            items.append("Quality")
        if self.Coverage:
            items.append("Coverage")
        if self.Rule:
            items.append("Rule")
        self.sortBox.clear()
        for s in items:
            self.sortBox.insertItem(s)
        self.sortMap=range(len(self.rules))
        for r in self.rules:
            l=[]
            if self.RuleLen:
                t=QCanvasText(self.canvas)
                t.setText("%10s" % str(r.complexity))
                l.append(t)
            if self.RuleQ:
                t=QCanvasText(self.canvas)
                t.setText("%10s" % ("%.3f" % r.quality))
                l.append(t)
            if self.Coverage:
                t=QCanvasText(self.canvas)
                t.setText("%15s" % str(len(self.classifier.examples.filterref(r.filter))))
                l.append(t)
            if self.Rule:
                t=QCanvasText(self.canvas)
                t.setText(self.ruleText(r))
                l.append(t)
            l.append(QCanvasText(self.canvas))
            text.append(l)
            self.obj.extend(l)
        self.text=text
        self.items=items
        self.drawRules()
        

    def drawRules(self):
        self.sort()
        text=self.text
        for r in self.rectObj:
            r.setCanvas(None)
        self.rectObj=[]
        textMapV=[10]+map(lambda s:max([t.boundingRect().height()+10 for t in s]), text)
        textMapH=[[s[i].boundingRect().width()+10 for s in text] for i in range(len(text[0]))]
        textMapH=[10]+map(lambda s:max(s), textMapH)
        
        for i in range(1,len(textMapV)):
            textMapV[i]+=textMapV[i-1]

        for i in range(1,len(textMapH)):
            textMapH[i]+=textMapH[i-1]
            
        for i in range(len(textMapV)-1):
            for j in range(len(textMapH)-2):
                text[i][j].move(textMapH[j], textMapV[i])
                text[i][j].setZ(0)
                text[i][j].show()
            r=QCanvasRectangle(textMapH[0],textMapV[i], textMapH[-1], textMapV[i+1]-textMapV[i], self.canvas)
            r.setZ(-20)
            r.setPen(QPen(Qt.NoPen))
            r.show()
            r.index=i
            self.obj.append(r)
            self.rectObj.append(r)
            
        self.canvas.resize(textMapH[-1], textMapV[-1])
        for i, k in enumerate(self.items):
            t=QCanvasText(k, self.headerCanvas)
            t.move(textMapH[i],0)
            t.show()
            self.obj.append(t)
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
        print list
        str=list[0]
        str="AND\n   ".join(str.split("AND"))
        str="\nTHEN".join(str.split("THEN"))
        return str

    def select(self):
        examples=[]
        source=self.classifier.examples
        for i in self.selRect:
            examples.extend(source.filterref(self.rules[self.sortMap[i]].filter))
            self.rules[self.sortMap[i]].filter.negate=1
            source=source.filterref(self.rules[self.sortMap[i]].filter)
            self.rules[self.sortMap[i]].filter.negate=0
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
        if str(a[0][self.Sort].text())<str(b[0][self.Sort].text()):
            return -1
        elif str(a[0][self.Sort].text())>str(b[0][self.Sort].text()):
            return 1
        else:
            return 0
        
    def sort(self):
        text=[]
        l=[(a,i) for a,i in zip(self.text,self.sortMap)]
        l.sort(self.compare)
        self.text=[a[0]  for a in l]
        self.sortMap=[a[1] for a in l]
        print self.sortMap
    
        
            

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
    
