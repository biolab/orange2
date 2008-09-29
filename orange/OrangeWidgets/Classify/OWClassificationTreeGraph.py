"""
<name>Classification Tree Graph</name>
<description>Classification tree viewer (graph view).</description>
<icon>icons/ClassificationTreeGraph.png</icon>
<contact>Blaz Zupan (blaz.zupan(@at@)fri.uni-lj.si)</contact>
<priority>2110</priority>
"""
from OWTreeViewer2D import *
import OWColorPalette

class ClassificationNode(GraphicsNode):
    def __init__(self,attrVal,*args):
        GraphicsNode.__init__(self,*args)
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
##        self.title = QGraphicsTextItem(attrVal, self, self.scene())
        self.texts=[self.majClassName, "%.3f" % self.majClassProb, "%.3f" % self.tarClassProb, "%.1f" % self.numInst]
        self.name = (self.tree.branches and self.tree.branchSelector.classVar.name) or self.majClassName
        self.textAdvance=12
        self.addTextLine(attrVal, fitSquare=False)
        self.addTextLine("", fitSquare=False)
        self.addTextLine("", fitSquare=False)
        self.addTextLine(fitSquare=False)
        self.addTextLine(self.name, fitSquare=False)

        self.rule=(isinstance(self.parent, QGraphicsRectItem) and self.parent.rule+[(self.parent.tree.branchSelector.classVar, attrVal)]) or []

        #print self.rule
        #self.textObj.extend([self.title, self.name]+self.textList)
        self.textInd=[]
        self.pieObj=[]
        distSum=sum(self.dist)
        color=OWColorPalette.ColorPaletteHSV(len(self.dist))
        startAngle=0
        self.pieGroup=QGraphicsItemGroup(self, self.scene())
        self.pieGroup.setPos(self.rect().width(), self.rect().height()/2)
        r=self.rect().height()*0.4
        for i in range(len(self.dist)):
            angle=360/distSum*self.dist[i]*16
##            e=QGraphicsEllipseItem(self.rect().height()/2, self.rect().width(), self.rect().height()*0.8, self.rect().height()*0.8, None, self.scene())
            e=QGraphicsEllipseItem(-r, -r, 2*r, 2*r, self.pieGroup, self.scene())
            e.setStartAngle(startAngle)
            e.setSpanAngle(angle)
            e.setBrush(QBrush(color[i]))
            e.setZValue(0)
            startAngle+=angle
            self.pieObj.append(e)
##        e = QGraphicsEllipseItem(self.rect().height(), self.rect().width(), self.rect().height()*0.8+4, self.rect().height()*0.8+4, None, self.scene())
        e = QGraphicsEllipseItem(-r-2, -r-2, 2*r+2, 2*r+2, self.pieGroup, self.scene())
##        e = QGraphicsEllipseItem(0, 0, r+4, r+4, self.pieGroup, self.scene())
        e.setStartAngle(0)
        e.setSpanAngle(360*16)
        e.setBrush(QBrush(Qt.black))
        e.setZValue(-1)
        self.pieObj.append(e)
        self.sceneObj.extend(self.pieObj)
        self.isPieShown=True

    def setRect(self, x, y, w, h):
        GraphicsNode.setRect(self, x, y, w, h)
        self.updateText()
        self.pieGroup.setPos(x+w, h/2)
        r=h*0.4
        for e in self.pieObj[:-1]:
##            e.setRect(self.x()+self.rect().width(), self.y()+self.rect().height()/2, h*0.8,h*0.8)
            e.setRect(-r, -r, 2*r, 2*r)
##            e.setRect(0, 0, r, r)
        self.pieObj[-1].setRect(-r-2, -r-2, 2*r+2, 2*r+2)
##        self.pieObj[-1].setRect(0, 0, r+4, r+4)

    def setBrush(self, brush):
        GraphicsTextContainer.setBrush(self, brush)
        if self.textObj:
            self.textObj[0].setBrush(QBrush((Qt.black)))

    def show(self):
        GraphicsNode.show(self)
        self.pieGroup.setVisible(self.isPieShown)

    def setPieVisible(self, b=True):
        if self.isVisible():
            self.pieGroup.setVisible(b)
        self.isPieShown=b
##        if self.isShown and b:
##            for e in self.pieObj:
##                e.show()
##        else:
##            for e in self.pieObj:
##                e.hide()

    def setText(self, textInd=[]):
        self.textInd=textInd
        j=1
        for i in textInd:
            GraphicsNode.setText(self, j, self.texts[i], fitSquare=False)
            j+=1
        for i in range(len(textInd),2):
            GraphicsNode.setText(self, i+1, "", fitSquare=False)

    def updateText(self):
        self.textAdvance=float(self.rect().height())/3
        self.lineSpacing=0
        self.setFont(QFont("",self.textAdvance*0.6), False)
        self.reArangeText(False, -self.textAdvance-self.lineSpacing)


    def reArangeText(self, fitSquare=True, startOffset=0):
        self.textOffset=startOffset
        x,y=self.x(),self.y()
        for i in range(4):
##            self.textObj[i].setPos(x+1, y+(i-1)*self.textAdvance)
            self.textObj[i].setPos(1, (i-1)*self.textAdvance)
##        self.spliterObj[0].setPos(x, y+self.rect().height()-self.textAdvance)
        self.spliterObj[0].setPos(0, self.rect().height()-self.textAdvance)

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

class OWClassificationTreeGraph(OWTreeViewer2D):
    settingsList = OWTreeViewer2D.settingsList+['ShowPies']
    contextHandlers = {"": DomainContextHandler("", ["TargetClassIndex"], matchValues=1)}
    def __init__(self, parent=None, signalManager = None, name='ClassificationTreeViewer2D'):
        self.ShowPies=1
        self.TargetClassIndex=0
        
        OWTreeViewer2D.__init__(self, parent, signalManager, name)

        self.inputs = [("Classification Tree", orange.TreeClassifier, self.ctree)]
        self.outputs = [("Examples", ExampleTable)]

        self.scene=TreeGraphicsScene(self)
        self.sceneView=TreeGraphicsView(self, self.scene)
        self.mainArea.layout().addWidget(self.sceneView)
        self.scene.setSceneRect(0,0,800,800)

        self.scene.bubbleConstructor=self.classificationBubbleConstructor

        self.navWidget=QWidget()
        self.navWidget.lay=QVBoxLayout(self.navWidget)

        scene=TreeGraphicsScene(self.navWidget)
        self.treeNav=TreeNavigator(self.sceneView,self,scene,self.navWidget)
        self.treeNav.setScene(scene)
        self.navWidget.lay.addWidget(self.treeNav)
        self.sceneView.setNavigator(self.treeNav)
        self.navWidget.resize(400,400)
        self.navWidget.setWindowTitle("Navigator")
        self.setMouseTracking(True)

        nodeInfoBox = OWGUI.widgetBox(self.NodeTab, "Show Info")
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
        for n in self.scene.nodeList:
            n.setText(self.NodeInfoSorted)
        self.scene.update()

    def activateLoadedSettings(self):
        if not self.tree:
            return
        OWTreeViewer2D.activateLoadedSettings(self)
        self.setNodeInfo()
        self.toggleNodeColor()


    def toggleNodeColor(self):
        for node in self.scene.nodeList:
            if self.NodeColorMethod == 0:   # default
                node.setBrush(QBrush(BodyColor_Default))
            elif self.NodeColorMethod == 1: # instances in node
                div = self.tree.distribution.cases
                if div > 1e-6:
                    light = 400 - 300*node.tree.distribution.cases/div
                else:
                    light = 100
                node.setBrush(QBrush(BodyCasesColor_Default.light(light)))
            elif self.NodeColorMethod == 2: # majority class probability
                light=400- 300*node.majClassProb
                node.setBrush(QBrush(self.ClassColors[node.majClass[0]].light(light)))
            elif self.NodeColorMethod == 3: # target class probability
                div = node.dist.cases
                if div > 1e-6:
                    light=400-300*node.dist[self.TargetClassIndex]/div
                else:
                    light = 100
                node.setBrush(QBrush(self.ClassColors[self.TargetClassIndex].light(light)))
            elif self.NodeColorMethod == 4: # target class distribution
                div = self.tree.distribution[self.TargetClassIndex]
                if div > 1e-6:
                    light=200 - 100*node.dist[self.TargetClassIndex]/div
                else:
                    light = 100
                node.setBrush(QBrush(self.ClassColors[self.TargetClassIndex].light(light)))
        self.scene.update()
        self.treeNav.leech()

    def toggleTargetClass(self):
        if self.NodeColorMethod in [3,4]:
            self.toggleNodeColor()
        for n in self.scene.nodeList:
            n.texts[2]="%.3f" % (n.dist.items()[self.TargetClassIndex][1]/n.dist.cases)
            if 2 in self.NodeInfoSorted:
                n.setText(self.NodeInfoSorted)
        self.scene.update()

    def togglePies(self):
        for n in self.scene.nodeList:
            n.setPieVisible(self.ShowPies)
        self.scene.update()

    def ctree(self, tree=None):
        self.send("Examples", None)
        self.closeContext()
        self.targetCombo.clear()
        if tree:
            for name in tree.tree.examples.domain.classVar.values:
                self.targetCombo.addItem(name)
#        if tree and len(tree.tree.distribution)>self.TargetClassIndex:
            self.TargetClassIndex=0
            self.openContext("", tree.domain)
        else:
            self.openContext("", None)
        OWTreeViewer2D.ctree(self, tree)
        self.togglePies()

    def walkcreate(self, tree, parent=None, level=0, attrVal=""):
        node=ClassificationNode(attrVal, tree, parent, self.scene)
        if tree.branches:
            for i in range(len(tree.branches)):
                if tree.branches[i]:
                    self.walkcreate(tree.branches[i],node,level+1,tree.branchDescriptions[i])
        return node

    def classificationBubbleConstructor(self, node, pos, scene):
        b=GraphicsBubbleInfo(node, pos, scene)
        rule=list(node.rule)
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
        text="Instances: %(ninst)s (%(prop).1f%%)" % {"ninst": str(node.numInst), "prop": node.numInst/self.tree.distribution.cases*100}
        b.addTextLine(text)
        b.addTextLine()
        for i,d in enumerate(node.dist.items()):
            if d[1]!=0:
                b.addTextLine("%s: %i (%.1f" %(d[0],int(d[1]),d[1]/sum(node.dist)*100)+"%)",self.ClassColors[i])
        b.addTextLine()
        b.addTextLine((node.tree.branches and "Partition on: %(nodename)s" % {"nodename": node.name}) or "(leaf)")
        b.show()
        return b


#    def sendReport(self):
#        directory = self.startReport("Classification Tree", True)
#        self.saveGraph(directory + "\\tree.png")
#        self.reportImage(directory + "\\tree.png")
#        self.finishReport()

if __name__=="__main__":
    a = QApplication(sys.argv)
    ow = OWClassificationTreeGraph()
##    a.setMainWidget(ow)

    #data = orange.ExampleTable('../../doc/datasets/voting.tab')
    data = orange.ExampleTable(r"../../doc/datasets/zoo.tab")
    tree = orange.TreeLearner(data, storeExamples = 1)
    ow.ctree(tree)

    # here you can test setting some stuff
    ow.show()
    a.exec_()
    ow.saveSettings()
