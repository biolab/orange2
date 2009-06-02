"""<name>Classification Tree Graph</name>
<description>Classification tree viewer (graph view).</description>
<icon>icons/ClassificationTreeGraph.png</icon>
<contact>Blaz Zupan (blaz.zupan(@at@)fri.uni-lj.si)</contact>
<priority>2110</priority>
"""
from OWTreeViewer2D import *
import OWColorPalette

class PieChart(QGraphicsRectItem):
    def __init__(self, dist, r, parent, scene):
        QGraphicsRectItem.__init__(self, parent, scene)
        self.dist = dist
        self.r = r
        
    def setR(self, r):
        self.r = r
        
    def boundingRect(self):
        return QRectF(-self.r, -self.r, 2*self.r, 2*self.r)
        
    def paint(self, painter, option, widget = None):
        distSum = sum(self.dist)
        startAngle = 0
        colors = self.scene().colorPalette
        for i in range(len(self.dist)):
            angle = self.dist[i]*16 * 360./distSum
            if angle == 0: continue
            painter.setBrush(QBrush(colors[i]))
            painter.setPen(QPen(colors[i]))
            painter.drawPie(-self.r, -self.r, 2*self.r, 2*self.r, int(startAngle), int(angle))
            startAngle += angle
        painter.setPen(QPen(Qt.black))
        painter.setBrush(QBrush())
        painter.drawEllipse(-self.r, -self.r, 2*self.r, 2*self.r)
        
            

class ClassificationNode(GraphicsNode):
    def __init__(self,attrVal, tree, parent, scene):
        GraphicsNode.__init__(self, tree, parent, scene)
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

        self.textInd=[]
        self.pieObj=[]
        distSum=sum(self.dist)
        startAngle=0
        self.pieObj = PieChart(self.dist, self.rect().height()*0.4, self, scene)
        self.pieObj.setZValue(0)
        self.pieObj.setPos(self.rect().width(), self.rect().height()/2)
        self.isPieShown=True

    def setRect(self, x, y, w, h):
        GraphicsNode.setRect(self, x, y, w, h)
        self.updateText()
        self.pieObj.setPos(x+w, h/2)
        self.pieObj.setR(h*0.4)

    def setBrush(self, brush):
        GraphicsTextContainer.setBrush(self, brush)
        if self.textObj:
            self.textObj[0].setBrush(QBrush((Qt.black)))

    def show(self):
        GraphicsNode.show(self)
        self.pieObj.setVisible(self.isPieShown)

    def setPieVisible(self, b=True):
        if self.isVisible():
            self.pieObj.setVisible(b)
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
    settingsList = OWTreeViewer2D.settingsList+['ShowPies', "colorSettings", "selectedColorSettingsIndex"]
    contextHandlers = {"": DomainContextHandler("", ["TargetClassIndex"], matchValues=1)}
    
    nodeColorOpts = ['Default', 'Instances in node', 'Majority class probability', 'Target class probability', 'Target class distribution']
    nodeInfoButtons = ['Majority class', 'Majority class probability', 'Target class probability', 'Number of instances']
    
    def __init__(self, parent=None, signalManager = None, name='ClassificationTreeViewer2D'):
        self.ShowPies=1
        self.TargetClassIndex=0
        self.colorSettings = None
        self.selectedColorSettingsIndex = 0
        
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
        
        OWGUI.button(self.NodeTab, self, "Set Colors", callback=self.setColors, debuggingEnabled = 0)

        nodeInfoBox = OWGUI.widgetBox(self.NodeTab, "Show Info")
        nodeInfoSettings = ['maj', 'majp', 'tarp', 'inst']
        self.NodeInfoW = []; self.dummy = 0
        for i in range(len(self.nodeInfoButtons)):
            setattr(self, nodeInfoSettings[i], i in self.NodeInfo)
            w = OWGUI.checkBox(nodeInfoBox, self, nodeInfoSettings[i], \
                               self.nodeInfoButtons[i], callback=self.setNodeInfo, getwidget=1, id=i)
            self.NodeInfoW.append(w)

        OWGUI.comboBox(self.NodeTab, self, 'NodeColorMethod', items=self.nodeColorOpts, box='Node Color',
                                callback=self.toggleNodeColor)

        OWGUI.checkBox(self.NodeTab, self, 'ShowPies', 'Show pies', box='Pies', tooltip='Show pie graph with class distribution?', callback=self.togglePies)
        self.targetCombo=OWGUI.comboBox(self.NodeTab,self, "TargetClassIndex",items=[],box="Target Class",callback=self.toggleTargetClass)
        OWGUI.button(self.controlArea, self, "Save As", callback=self.saveGraph, debuggingEnabled = 0)
        self.NodeInfoSorted=list(self.NodeInfo)
        self.NodeInfoSorted.sort()
        
        dlg = self.createColorDialog()
        self.scene.colorPalette = dlg.getDiscretePalette("colorPalette")


    def sendReport(self):
        self.reportSettings("Information",
                            [("Node color", self.nodeColorOpts[self.NodeColorMethod]),
                             ("Target class", self.tree.examples.domain.classVar.values[self.TargetClassIndex]),
                             ("Data in nodes", ", ".join(s for i, s in enumerate(self.nodeInfoButtons) if self.NodeInfoW[i].isChecked())),
                             ("Line widths", ["Constant", "Proportion of all instances", "Proportion of parent's instances"][self.LineWidthMethod]),
                             ("Tree size", "%i nodes, %i leaves" % (orngTree.countNodes(self.tree), orngTree.countLeaves(self.tree)))])
        OWTreeViewer2D.sendReport(self)


    def setColors(self):
        dlg = self.createColorDialog()
        if dlg.exec_():
            self.colorSettings = dlg.getColorSchemas()
            self.selectedColorSettingsIndex = dlg.selectedSchemaIndex
            self.scene.colorPalette = dlg.getDiscretePalette("colorPalette")
            self.scene.update()

    def createColorDialog(self):
        c = OWColorPalette.ColorPaletteDlg(self, "Color Palette")
        c.createDiscretePalette("colorPalette", "Discrete Palette")
        c.setColorSchemas(self.colorSettings, self.selectedColorSettingsIndex)
        return c


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
                node.setBrush(QBrush(self.scene.colorPalette[node.majClass[0]].light(light)))
            elif self.NodeColorMethod == 3: # target class probability
                div = node.dist.cases
                if div > 1e-6:
                    light=400-300*node.dist[self.TargetClassIndex]/div
                else:
                    light = 100
                node.setBrush(QBrush(self.scene.colorPalette[self.TargetClassIndex].light(light)))
            elif self.NodeColorMethod == 4: # target class distribution
                div = self.tree.distribution[self.TargetClassIndex]
                if div > 1e-6:
                    light=200 - 100*node.dist[self.TargetClassIndex]/div
                else:
                    light = 100
                node.setBrush(QBrush(self.scene.colorPalette[self.TargetClassIndex].light(light)))
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
                b.addTextLine("%s: %i (%.1f" %(d[0],int(d[1]),d[1]/sum(node.dist)*100)+"%)", self.scene.colorPalette[i])
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
