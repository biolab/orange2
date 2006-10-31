"""
<name>Classification Tree Graph</name>
<description>Classification tree viewer (graph view).</description>
<icon>icons/ClassificationTreeGraph.png</icon>
<contact>Blaz Zupan (blaz.zupan(@at@)fri.uni-lj.si)</contact> 
<priority>2110</priority>
"""

from OWTreeViewer2D import *
import OWGraphTools

class ClassificationNode(CanvasNode):
    def __init__(self,attrVal,*args):
        CanvasNode.__init__(self,*args)
        self.dist=self.tree.distribution
        self.attrVal=attrVal
        maxInst=max(self.dist)
        #self.majClass=ind=list(self.dist).index(maxProb)
        self.majClass = filter(lambda i, m=maxInst: self.dist[i]==m, range(len(self.dist)))
        ind=self.majClass[0]
        self.majClassName=self.dist.items()[ind][0]
        self.majClassProb=maxInst/self.dist.cases
        self.tarClassProb=self.dist.items()[0][1]/self.dist.cases
        self.numInst=self.dist.cases
        self.title=QCanvasText(attrVal,self.canvas())
        self.texts=[self.majClassName, "%.3f" % self.majClassProb, "%.3f" % self.tarClassProb, "%.1f" % self.numInst]
        self.name = (self.tree.branches and self.tree.branchSelector.classVar.name) or self.majClassName
        self.textAdvance=12
        self.addTextLine(attrVal, fitSquare=False)
        self.addTextLine("", fitSquare=False)
        self.addTextLine("", fitSquare=False)
        self.addTextLine(fitSquare=False)
        self.addTextLine(self.name, fitSquare=False)
            
        self.rule=(isinstance(self.parent, QCanvasRectangle) and self.parent.rule+[(self.parent.tree.branchSelector.classVar, attrVal)]) or []
        
        #print self.rule
        #self.textObj.extend([self.title, self.name]+self.textList)
        self.textInd=[]        
        self.pieObj=[]
        distSum=sum(self.dist)
        color=OWGraphTools.ColorPaletteHSV(len(self.dist))
        startAngle=0
        for i in range(len(self.dist)):
            angle=360/distSum*self.dist[i]*16
            e=QCanvasEllipse(self.height()*0.8,self.height()*0.8,startAngle,angle,self.canvas())
            e.setBrush(QBrush(color[i]))
            e.move(self.height()/2,self.width())
            e.setZ(0)
            startAngle+=angle
            self.pieObj.append(e)
        e=QCanvasEllipse(self.height()*0.8+4,self.height()*0.8+4,0,360*16,self.canvas())
        e.setBrush(QBrush(Qt.black))
        e.move(self.height(), self.width())
        e.setZ(-1)
        self.pieObj.append(e)
        self.canvasObj.extend(self.pieObj)
        self.isPieShown=True            

    def setSize(self,w,h):
        CanvasNode.setSize(self,w,h)
        self.updateText()
        for e in self.pieObj:
            e.setSize(h*0.8,h*0.8)
            e.move(self.x()+self.width(),self.y()+self.height()/2)
        self.pieObj[-1].setSize(h*0.8+2,h*0.8+2)

    def setBrush(self, brush):
        CanvasTextContainer.setBrush(self, brush)
        if self.textObj:
            self.textObj[0].setColor(Qt.black)
            
    def show(self):
        CanvasNode.show(self)
        if not self.isPieShown:
            for e in self.pieObj:
                e.hide()

    def setPieVisible(self, b=True):
        self.isPieShown=b
        if self.isShown and b:
            for e in self.pieObj:
                e.show()
        else:
            for e in self.pieObj:
                e.hide()
                
    def setText(self, textInd=[]):
        self.textInd=textInd
        j=1
        for i in textInd:
            CanvasNode.setText(self, j, self.texts[i], fitSquare=False)
            j+=1
        for i in range(len(textInd),2):
            CanvasNode.setText(self, i+1, "", fitSquare=False)
        
    def updateText(self):
        self.textAdvance=float(self.height())/3
        self.lineSpacing=0
        self.setFont(QFont("",self.textAdvance*0.7), False)
        self.reArangeText(False, -self.textAdvance-self.lineSpacing)


    def reArangeText(self, fitSquare=True, startOffset=0):
        self.textOffset=startOffset
        x,y=self.x(),self.y()
        for i in range(4):
            self.textObj[i].move(x+1, y+(i-1)*self.textAdvance)
        self.spliterObj[0].move(x, y+self.height()-self.textAdvance)

import re
import sets
def parseRules(rules):
    def joinCont(rule1, rule2):
        int1, int2=["(",-1e1000,1e1000,")"], ["(",-1e1000,1e1000,")"]
        rule=[rule1, rule2]
        interval=[int1, int2]
        for i in [0,1]:
            if rule[i][1].startswith("in"):
                r=rule[i][1][2:]
                interval[i]=[r.strip(" ")[0]]+map(lambda a: float(a), r.strip("()[] ").split(","))+[r.strip(" ")[-1]]
            else:
                if "<" in rule[i][1]:
                    interval[i][3]=("=" in rule[i][1] and "]") or ")"
                    interval[i][2]=float(rule[i][1].strip("<>= "))
                else:
                    interval[i][0]=("=" in rule[i][1] and "[") or "("
                    interval[i][1]=float(rule[i][1].strip("<>= "))
               
        inter=[None]*4

        if interval[0][1]<interval[1][1] or (interval[0][1]==interval[1][1] and interval[0][0]=="["):
            interval.reverse()
        inter[:2]=interval[0][:2]

        if interval[0][2]>interval[1][2] or (interval[0][2]==interval[1][2] and interval[0][3]=="]"):
            interval.reverse()
        inter[2:]=interval[0][2:]
            

        if 1e1000 in inter or -1e1000 in inter:
            rule=((-1e1000==inter[1] and "<") or ">")
            rule+=(("[" in inter or "]" in inter) and "=") or ""
            rule+=(-1e1000==inter[1] and str(inter[2])) or str(inter[1])
        else:
            rule="in "+inter[0]+str(inter[1])+","+str(inter[2])+inter[3]
        return (rule1[0], rule)
    
    def joinDisc(rule1, rule2):
        r1,r2=rule1[1],rule2[1]
        r1=re.sub("^in ","",r1)
        r2=re.sub("^in ","",r2)
        r1=r1.strip("[]=")
        r2=r2.strip("[]=")
        s1=sets.Set([s.strip(" ") for s in r1.split(",")])
        s2=sets.Set([s.strip(" ") for s in r2.split(",")])
        s=s1 & s2
        if len(s)==1:
            return (rule1[0], "= "+str(list(s)[0]))
        else:
            return (rule1[0], "in ["+",".join([str(st) for st in s])+"]")

    rules.sort(lambda a,b: (a[0].name<b[0].name and -1) or 1 )
    newRules=[rules[0]]
    for r in rules[1:]:
        if r[0].name==newRules[-1][0].name:
            if re.search("(a-zA-Z\"')+",r[1].lstrip("in")):
                newRules[-1]=joinDisc(r,newRules[-1])
            else:
                newRules[-1]=joinCont(r,newRules[-1])
        else:
            newRules.append(r)
    return newRules      

BodyColor_Default = QColor(255, 225, 10)
BodyCasesColor_Default = QColor(0, 0, 128)

class OWClassificationTreeViewer2D(OWTreeViewer2D):
    def __init__(self, parent=None, signalManager = None, name='ClassificationTreeViewer2D'):
        OWTreeViewer2D.__init__(self, parent, signalManager, name)
        self.settingsList=self.settingsList+["ShowPies","TargetClassIndex"]
        
        self.inputs = [("Classification Tree", orange.TreeClassifier, self.ctree)]
        self.outputs = [("Examples", ExampleTable), ("Classified Examples", ExampleTableWithClass)]
        
        self.ShowPies=1
        self.TargetClassIndex=0
        self.canvas=TreeCanvas(self)
        self.canvasView=TreeCanvasView(self, self.canvas, self.mainArea, "CView")
        layout=QVBoxLayout(self.mainArea)
        layout.addWidget(self.canvasView)
        self.canvas.resize(800,800)
        self.canvasView.bubbleConstructor=self.classificationBubbleConstructor
        self.navWidget=QWidget(None, "Navigator")
        self.navWidget.lay=QVBoxLayout(self.navWidget)
        canvas=TreeCanvas(self.navWidget)
        self.treeNav=TreeNavigator(self.canvasView,self,canvas,self.navWidget, "Nav")
        self.treeNav.setCanvas(canvas)
        self.navWidget.lay.addWidget(self.treeNav)
        self.canvasView.setNavigator(self.treeNav)
        self.navWidget.resize(400,400)
        self.navWidget.setCaption("Navigator")
        # OWGUI.button(self.TreeTab,self,"Navigator",self.toggleNavigator)
        self.setMouseTracking(True)

        nodeInfoBox = QVButtonGroup("Show Info", self.NodeTab)
        nodeInfoButtons = ['Majority class', 'Majority class probability', 'Target class probability', 'Number of instances']
        nodeInfoSettings = ['maj', 'majp', 'tarp', 'inst']
        self.NodeInfoW = []; self.dummy = 0
        for i in range(len(nodeInfoButtons)):
            setattr(self, nodeInfoSettings[i], i in self.NodeInfo)
            w = OWGUI.checkBox(nodeInfoBox, self, nodeInfoSettings[i], \
                               nodeInfoButtons[i], callback=self.setNodeInfo, getwidget=1, id=i)
            self.NodeInfoW.append(w)

        OWGUI.comboBox(self.NodeTab, self, 'NodeColorMethod', items=['Default', 'Instances in node', 'Majority class probability', 'Target class probability', 'Target class distribution'], box='Node Color',                            
                                callback=self.toggleNodeColor)

        OWGUI.checkBox(self.NodeTab, self, 'ShowPies', 'Show pies', box='Pies', tooltip='Show pie graph with class distribution?', callback=self.togglePies)
        self.targetCombo=OWGUI.comboBox(self.NodeTab,self, "TargetClassIndex",items=[],box="Target Class",callback=self.toggleTargetClass)
        OWGUI.button(self.controlArea, self, "Save As", callback=self.saveGraph, debuggingEnabled = 0)
        self.NodeInfoSorted=list(self.NodeInfo)
        self.NodeInfoSorted.sort()
        
    def setNodeInfo(self, widget=None, id=None):
        if widget:
            if widget.isChecked():
                if len(self.NodeInfo) == 2:
                    self.NodeInfoW[self.NodeInfo[0]].setChecked(0)
                self.NodeInfo.append(id)
            else:
                self.NodeInfo.remove(id)
            self.NodeInfoSorted=list(self.NodeInfo)
            self.NodeInfoSorted.sort()
            self.NodeInfoMethod=id
        #print self.NodeInfoSorted
        for n in self.canvas.nodeList:
            n.setText(self.NodeInfoSorted)
        self.canvas.update()

    def activateLoadedSettings(self):
        if not self.tree:
            return 
        OWTreeViewer2D.activateLoadedSettings(self)
        self.setNodeInfo()
        self.toggleNodeColor()
        
    def toggleNodeColor(self):
        for node in self.canvas.nodeList:
            if self.NodeColorMethod == 0:   # default
                node.setBrush(QBrush(BodyColor_Default))
            elif self.NodeColorMethod == 1: # instances in node
                light = 400 - 300*node.tree.distribution.cases/self.tree.distribution.cases
                node.setBrush(QBrush(BodyCasesColor_Default.light(light)))
            elif self.NodeColorMethod == 2: # majority class probability
                light=400- 300*node.majClassProb
                node.setBrush(QBrush(self.ClassColors[node.majClass[0]].light(light)))
            elif self.NodeColorMethod == 3: # target class probability
                light=400-300*node.dist[self.TargetClassIndex]/node.dist.cases
                node.setBrush(QBrush(self.ClassColors[self.TargetClassIndex].light(light)))
            elif self.NodeColorMethod == 4: # target class distribution
                light=200 - 100*node.dist[self.TargetClassIndex]/self.tree.distribution[self.TargetClassIndex]
                node.setBrush(QBrush(self.ClassColors[self.TargetClassIndex].light(light)))
        self.canvas.update()
        self.treeNav.leech()

    def toggleTargetClass(self):
        if self.NodeColorMethod in [3,4]:
            self.toggleNodeColor()
        for n in self.canvas.nodeList:
            n.texts[2]="%.3f" % (n.dist.items()[self.TargetClassIndex][1]/n.dist.cases)
            if 2 in self.NodeInfoSorted:
                n.setText(self.NodeInfoSorted)
        self.canvas.update()
                
    def togglePies(self):
        for n in self.canvas.nodeList:
            n.setPieVisible(self.ShowPies)
        self.canvas.update()

    def ctree(self, tree=None):
        self.targetCombo.clear()
        if tree:
            for name in tree.tree.examples.domain.classVar.values:
                self.targetCombo.insertItem(name)
        if tree and len(tree.tree.distribution)>self.TargetClassIndex:
            self.TargetClassIndex=0
        OWTreeViewer2D.ctree(self, tree)

    def walkcreate(self, tree, parent=None, level=0, attrVal=""):
        node=ClassificationNode(attrVal, tree, parent or self.canvas, self.canvas)
        if tree.branches:
            for i in range(len(tree.branches)):
                if tree.branches[i]:
                    self.walkcreate(tree.branches[i],node,level+1,tree.branchDescriptions[i])
        return node

    def classificationBubbleConstructor(self, node, pos, canvas):
        b=CanvasBubbleInfo(node, pos,canvas)
        rule=list(node.rule)
        #print node.rule, rule
        #rule.sort(lambda:a,b:a[0]<b[0])
        # merge
        if rule:
            try:
                rule=parseRules(list(rule))
            except:
                pass
            text="IF "+" AND\n  ".join([a[0].name+" = "+a[1] for a in rule])+"\nTHEN "+node.majClassName
        else:
            text="THEN "+node.majClassName
        b.addTextLine(text)
        b.addTextLine()
        text="Instances:"+str(node.numInst)+"(%.1f" % (node.numInst/self.tree.distribution.cases*100)+"%)"
        b.addTextLine(text)
        b.addTextLine()
        for i,d in enumerate(node.dist.items()):
            if d[1]!=0:
                b.addTextLine("%s: %i (%.1f" %(d[0],int(d[1]),d[1]/sum(node.dist)*100)+"%)",self.ClassColors[i])
        b.addTextLine()
        b.addTextLine((node.tree.branches and "Partition on: "+node.name) or "(leaf)")
        b.show()
        return b

    def saveGraph(self):
        qfileName = QFileDialog.getSaveFileName("tree.png","Portable Network Graphics (.PNG)\nWindows Bitmap (.BMP)\nGraphics Interchange Format (.GIF)\nDot Tree File(.DOT)", None, "Save to..")
        fileName = str(qfileName)
        if fileName == "": return
        (fil,ext) = os.path.splitext(fileName)
        ext = ext.replace(".","")
        ext = ext.upper()
        if ext=="DOT":
            orngTree.printDot(self.tree, fileName)
            return 
        dSize= self.canvas.size()
        buffer = QPixmap(dSize.width(),dSize.height()) # any size can do, now using the window size     
        painter = QPainter(buffer)
        
        painter.fillRect(buffer.rect(), QBrush(QColor(255, 255, 255))) # make background same color as the widget's background
        self.canvasView.drawContents(painter,0,0,dSize.width(), dSize.height())
        painter.end()
        buffer.save(fileName, ext)
    
if __name__=="__main__":
    a = QApplication(sys.argv)
    ow = OWClassificationTreeViewer2D()
    a.setMainWidget(ow)

    data = orange.ExampleTable('../../doc/datasets/voting.tab')
    tree = orange.TreeLearner(data, storeExamples = 1)
    ow.ctree(tree)

    # here you can test setting some stuff
    ow.show()
    a.exec_loop()
    ow.saveSettings()
