"""
<name>CN2 Rules Viewer</name>
<description>Viewer of classification rules.</description>
<icon>CN2RulesViewer.png</icon>
<contact>Ales Erjavec (ales.erjavec(@at@)fri.uni-lj.si)</contact> 
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
        self.brush=QBrush(QColor("lightGray").light(112))
        self.lastIndex=-1
        self.flag=False

    def contentsMouseMoveEvent(self,e):
        self.flag=True
        if self.buttonPressed:
            obj=self.canvas().collisions(e.pos())
            if obj and obj[-1].__class__==QCanvasRectangle and obj[-1].index!=self.lastIndex:
                self.addSelection(obj[-1])
                self.parent.select()
                self.lastIndex=obj[-1].index
    
    def contentsMousePressEvent(self,e):
        self.buttonPressed=True
        self.lastIndex=-1
        if not self.parent.ctrlPressed:
            self.parent.selRect=[]
            for r in self.parent.rectObj:
                r.setBrush(QBrush(Qt.NoBrush))
               
    def contentsMouseReleaseEvent(self,e):
        self.flag=False
        obj=self.canvas().collisions(e.pos())
        if obj and obj[-1].__class__==QCanvasRectangle and obj[-1].index!=self.lastIndex:
            self.addSelection(obj[-1])
            self.parent.select()
        self.buttonPressed=False

    def addSelection(self, rect):
        index=rect.index
        if (not self.buttonPressed and not self.flag) and not self.parent.ctrlPressed:
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

class OWCN2RulesViewer(OWWidget):
    settingsList=["RuleLen","RuleQ","Coverage","Commit","Rule","Sort","Dist","DistBar","Class"]
    callbackDeposit=[]
    def __init__(self, parent=None, signalManager=None):
        OWWidget.__init__(self, parent, signalManager,"CN2 Rules Viewer")
        
        self.inputs=[("RuleClassifier", orange.RuleClassifier, self.data)]
        self.inputs=[("Unordered CN2 Classifier", orngCN2.CN2UnorderedClassifier, self.data)]
        self.outputs=[("Examples", ExampleTable), ("Classified Examples", ExampleTableWithClass), ("Attribute List", AttributeList)]
        self.RuleLen=1
        self.RuleQ=1
        self.Coverage=1
        self.Class=1
        self.Dist=1
        self.DistBar=1
        self.Commit=1
        self.SelectedAttrOnly=0
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
        
        box=OWGUI.widgetBox(self.controlArea,"Show info")
        box.setSizePolicy(QSizePolicy(QSizePolicy.Fixed,QSizePolicy.Fixed))

        OWGUI.checkBox(box,self,"RuleLen","Rule length",callback=self.drawRules)
        OWGUI.checkBox(box,self,"RuleQ","Rule quality",callback=self.drawRules)
        OWGUI.checkBox(box,self,"Coverage","Coverage",callback=self.drawRules)
        OWGUI.checkBox(box,self,"Class","Predicted class", callback=self.drawRules)
        OWGUI.checkBox(box,self,"Dist","Distribution", callback=self.drawRules)
        OWGUI.checkBox(box,self,"DistBar","Distribution(Bar)",callback=self.drawRules)

        OWGUI.separator(self.controlArea)
        box=OWGUI.widgetBox(self.controlArea,"Sorting")
        box.setSizePolicy(QSizePolicy(QSizePolicy.Fixed,QSizePolicy.Fixed))
        self.sortBox=OWGUI.comboBox(box, self, "Sort", 
                                    items=["No sorting", "Rule length", "Rule quality", "Coverage", "Predicted class",
                                           "Distribution","Rule"]
                                    ,callback=self.drawRules)
        OWGUI.separator(self.controlArea)
        box=OWGUI.widgetBox(self.controlArea,"Output")
        box.setSizePolicy(QSizePolicy(QSizePolicy.Fixed,QSizePolicy.Fixed))
        OWGUI.checkBox(box,self,"Commit", "Commit on change")
        OWGUI.checkBox(box,self,"SelectedAttrOnly","Selected attributes only")
        OWGUI.button(box,self,"&Commit",callback=self.commit)

        QVBox(self.controlArea)
        
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
        self.selRect=[]

    def showRules(self):
        self.clear()
        text=[]
        items=[]
        for i, r in enumerate(self.rules):
            l=[i,str(int(float(r.complexity))), "%.3f" % r.quality, "%.1f"%r.classDistribution.abs, #str(len(self.classifier.examples.filterref(r.filter))),
                str(r.classifier.defaultValue), self.distText(r), self.ruleText(r),r]
            text.append(l)
            self.distText(r)
        self.text=text
        #self.items=items
        self.drawRules()
        

    def drawRules(self):
        self.oldSelection=[r.rule for r in self.selRect]
        self.sort()
        self.clear()
        for r in self.rectObj:
            r.setCanvas(None)
        self.rectObj=[]
        text=self.text
        filter=[self.RuleLen, self.RuleQ, self.Coverage, self.Class, self.Dist or self.DistBar,self.Rule]
        l=[]
        #a=["Length","Quality","Coverage","Class","Distribution", "Distribution(Bar)", "Rule"]
        a=["Length","Quality","Coverage","Class","Distribution", "Rule"]
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
                t.setText(text[1])
                l.append(t)
            if self.RuleQ:
                t=QCanvasText(self.canvas)
                t.setText(text[2])
                l.append(t)
            if self.Coverage:
                t=QCanvasText(self.canvas)
                t.setText(text[3])
                l.append(t)
            if self.Class:
                t=QCanvasText(self.canvas)
                t.setText(text[4])
                l.append(t)
            if self.Dist and not self.DistBar:
                t=QCanvasText(self.canvas)
                t.setText(text[5])
                l.append(t)
            if self.DistBar:
                t=DistBar(text[5],self.Dist,self.canvas)
                l.append(t)                    
            if self.Rule:
                t=QCanvasText(self.canvas)
                t.setText(text[6])#self.ruleText(text[-1]))
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
            
        self.ctrlPressed=True
        for i in range(1,len(textMapV)):
            for j in range(len(textMapH)-2):
                if items[i][j].__class__==DistBar:
                    items[i][j].fixGeom((textMapH[j+1])-textMapH[j]-10)
                items[i][j].move(textMapH[j], textMapV[i-1])
                items[i][j].setZ(0)
                items[i][j].show()
            r=QCanvasRectangle(0,textMapV[i], textMapH[-1], textMapV[i-1]-textMapV[i], self.canvas)
            r.setZ(-20)
            r.setPen(QPen(Qt.NoPen))
            r.show()
            r.index=i+1
            r.rule=self.text[i-1][-1]
            if r.rule in self.oldSelection:
                self.canvasView.addSelection(r)
            self.obj.append(r)
            self.rectObj.append(r)
        self.ctrlPressed=False
            
        self.canvas.resize(textMapH[-1], textMapV[-1])
        for i,t in enumerate(items[0][:-1]):
            t.move(textMapH[i],0)
            t.show()
        self.headerCanvas.update()
        self.headerCanvas.resize(textMapH[-1]+20,20)
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
        str="".join(list)
        list=re.split("<[0-9., ]*>$", str)
        str=list[0]
        str="AND\n   ".join(str.split("AND"))
        str="\nTHEN".join(str.split("THEN"))
        return str

    def distText(self,r):
        #e=self.classifier.examples.filterref(r.filter)
        d=r.classDistribution#orange.Distribution(r.classifier.classVar,self.classifier.examples.filterref(r.filter))
        s=str(d).strip("<>")
        return "<"+",".join(["%.1f" % float(f) for f in s.split(",")])+">"

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
            return -1    
    def compareDist(self,a,b):
##        if max(a[-1].classDistribution)/a[-1].classDistribution.abs \
##                < max(b[-1].classDistribution)/b[-1].classDistribution.abs:
##            return -1
        if max(a[-1].classDistribution)/a[-1].classDistribution.abs \
               > max(b[-1].classDistribution)/b[-1].classDistribution.abs:
            return 1
        else:
            return -1
        
    def sort(self):
        text=[]
        if self.Sort==5:
            self.text.sort(self.compareDist)
        elif self.Sort==3 or self.Sort==1:
            self.text.sort(lambda a,b:-cmp(float(a[self.Sort]),float(b[self.Sort])))
        elif self.Sort==0:
            self.text.sort(lambda a,b:cmp(a[0],b[0]))
        else:
            if self.Sort>6:
                self.sortBy=self.Sort+1
            else:
                self.sortBy=self.Sort
            self.text.sort(self.compare)
        #print self.text
        if self.Sort>=2 and self.Sort!=4 and self.Sort !=6:
            self.text.reverse()     

    def selectAttributes(self):
        import sets
        selected=[]
        for r in self.selRect:
            string=orngCN2.ruleToString(r.rule)[2:].strip(" ").split("THEN")[0]
            list=string.split("AND")
            for l in list:
                s=re.split("[=<>]", l.strip(" "))
                selected.append(s[0])
        selected=reduce(lambda l,r:(r in l) and l or l+[r], selected, [])
        return selected          

    def commit(self):
        if self.examples:
            selected=self.selectAttributes()
            varList=[self.classifier.examples.domain[s] for s in selected]
            if self.SelectedAttrOnly:
                domain=orange.Domain(varList+[self.classifier.examples.domain.classVar])
                domain.addmetas(self.classifier.examples.domain.getmetas())
                examples=orange.ExampleTable(domain, self.examples)
            else:
                examples = orange.ExampleTable(self.examples)
            self.send("Classified Examples", examples)
            self.send("Examples", examples)
            self.send("Attribute List", orange.VarList(varList))
        else:
            self.send("Classified Examples", None)
            self.send("Examples",None)
            self.send("Attribute List", None)
            

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
barHeight=10
class DistBar(QCanvasRectangle):
    def __init__(self, distText,showText,canvas, *args):
        apply(QCanvasRectangle.__init__,(self,canvas)+args)
        self.distText=distText
        self.showText=showText
        self.canvas=canvas
        self.rect=[]
        self.text=QCanvasText(distText,canvas)
        self.barWidth=max([barWidth, showText and self.text.boundingRect().width()])
        self.setSize(self.barWidth, barHeight)
        
    def fixGeom(self, width):
        distText=self.distText.strip("<>")
        dist=[float(f) for f in distText.split(",")]
        distSum=sum(dist)
        classColor=OWGraphTools.ColorPaletteHSV(len(dist))
        m=max(dist)
        for i in range(len(dist)):
            r=QCanvasRectangle(self.canvas)
            r.setSize(dist[i]/distSum*width,barHeight)
            r.setPen(QPen(QColor(classColor[i]),2))
            if dist[i]==m:
                r.setBrush(QBrush(classColor[i]))
            self.rect.append(r)
            
    def move(self, x,y):
        if self.showText:
            ty, ry=y, y+18
        else:
            ty, ry=0, y+3
        pos=0
        for r in self.rect:
            r.move(x+pos,ry)
            pos+=r.width()
        self.text.move(x,ty)
            
    def show(self):
        for r in self.rect:
            r.show()
        if self.showText:
            self.text.show()
    def setCanvas(self, canvas):
        for r in self.rect:
            r.setCanvas(canvas)
        self.text.setCanvas(canvas)
    
    
if __name__=="__main__":
    ap=QApplication(sys.argv)
    w=OWCN2RulesViewer()
    ap.setMainWidget(w)
    data=orange.ExampleTable("../../doc/datasets/car.tab")
    l=orngCN2.CN2UnorderedLearner()
    l.ruleFinder.ruleStoppingValidator=orange.RuleValidator_LRS()
    w.data(l(data))
    w.data(l(data))
    w.show()
    ap.exec_loop()
    
