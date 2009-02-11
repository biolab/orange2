"""
<name> Regression Tree Graph</name>
<description>Regression tree viewer (graph view).</description>
<icon>icons/RegressionTreeGraph.png</icon>
<contact>Ales Erjavec (ales.erjavec(@at@)fri.uni-lj.si)</contact>
<priority>2110</priority>
"""
from OWTreeViewer2D import *
import re
import sets

class RegressionNode(GraphicsNode):
    def __init__(self, attrVal, *args):
        GraphicsNode.__init__(self, *args)
        self.attrVal=attrVal
        self.dist=self.tree.distribution
        self.numInst=self.dist.cases
        self.defVal=self.tree.nodeClassifier.defaultValue
        self.var=self.dist.var()
        self.dev=self.dist.dev()
        self.error=self.dist.error()
        self.texts=["%.4f" % self.defVal,"%.3f" %self.var,"%.3f" %self.dev,"%.3f" % self.error,"%.i" % self.numInst]
        self.name = (self.tree.branches and self.tree.branchSelector.classVar.name) or str(self.defVal)
        self.addTextLine(self.attrVal, None, False)
        self.addTextLine("", None, False)
        self.addTextLine("", None, False)
        self.addTextLine(None, None, None)
        self.addTextLine(self.name, None, False)
        self.textind=[]
        self.rule=(isinstance(self.parent, QGraphicsRectItem) and \
                   self.parent.rule+[(self.parent.tree.branchSelector.classVar, attrVal)]) or []
        self.textAdvance=15

    def setRect(self,x,y,w,h):
        GraphicsNode.setRect(self,x,y,w,h)
        self.updateText()

##    def setBrush(self, brush):
##        GraphicsTextContainer.setBrush(self, brush)
##        if self.textObj:
##            self.textObj[0].setColor(Qt.black)

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
        self.setFont(QFont("",self.textAdvance*0.7), False)
        self.reArangeText(False, -self.textAdvance-self.lineSpacing)


##    def reArangeText(self, fitSquare=True, startOffset=0):
##        self.textOffset=startOffset
##        x,y=self.x(),self.y()
##        for i in range(4):
##            self.textObj[i].move(x+1, y+(i-1)*self.textAdvance)
##        self.spliterObj[0].move(x, y+self.height()-self.textAdvance)

    def reArangeText(self, fitSquare=True, startOffset=0):
        self.textOffset=startOffset
        x,y=self.x(),self.y()
        for i in range(4):
##            self.textObj[i].setPos(x+1, y+(i-1)*self.textAdvance)
            self.textObj[i].setPos(1, (i-1)*self.textAdvance)
##        self.spliterObj[0].setPos(x, y+self.rect().height()-self.textAdvance)
        self.spliterObj[0].setPos(0, self.rect().height()-self.textAdvance)        


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

class OWRegressionTreeViewer2D(OWTreeViewer2D):
    nodeColorOpts = ['Default', 'Instances in node', 'Variance', 'Deviation', 'Error']
    nodeInfoButtons = ['Predicted value', 'Variance', 'Deviation', 'Error', 'Number of instances']

    def __init__(self, parent=None, signalManager = None, name='RegressionTreeViewer2D'):
        OWTreeViewer2D.__init__(self, parent, signalManager, name)

        self.inputs = [("Classification Tree", orange.TreeClassifier, self.ctree)]
        self.outputs = [("Examples", ExampleTable)]
        
        self.scene = TreeGraphicsScene(self)
        self.sceneView = TreeGraphicsView(self, self.scene)
        self.mainArea.layout().addWidget(self.sceneView)
        self.scene.setSceneRect(0,0,800,800)

        self.scene.bubbleConstructor=self.regressionBubbleConstructor

        self.navWidget = QWidget(None)
        self.navWidget.lay=QVBoxLayout(self.navWidget)
##        self.navWidget.setLayout(QVBoxLayout())

        scene = TreeGraphicsScene(self.navWidget)
        self.treeNav=TreeNavigator(self.sceneView,self,scene,self.navWidget)
        self.treeNav.setScene(scene)
        self.navWidget.layout().addWidget(self.treeNav)
        self.sceneView.setNavigator(self.treeNav)
        self.navWidget.resize(400,400)
        self.navWidget.setWindowTitle("Navigator")
        self.setMouseTracking(True)

        nodeInfoBox = OWGUI.widgetBox(self.NodeTab, "Show Info On")
        nodeInfoSettings = ['maj', 'majp', 'tarp', 'error', 'inst']
        self.NodeInfoW = []; self.dummy = 0
        for i in range(len(self.nodeInfoButtons)):
            setattr(self, nodeInfoSettings[i], i in self.NodeInfo)
            w = OWGUI.checkBox(nodeInfoBox, self, nodeInfoSettings[i], \
                               self.nodeInfoButtons[i], callback=self.setNodeInfo, getwidget=1, id=i)
            self.NodeInfoW.append(w)

        OWGUI.comboBox(self.NodeTab, self, 'NodeColorMethod', items=self.nodeColorOpts, box='Node Color',
                                callback=self.toggleNodeColor)
        
        OWGUI.button(self.controlArea, self, "Save As", callback=self.saveGraph)
        self.NodeInfoSorted=list(self.NodeInfo)
        self.NodeInfoSorted.sort()

    def sendReport(self):
        self.reportSettings("Information",
                            [("Node color", self.nodeColorOpts[self.NodeColorMethod]),
                             ("Data in nodes", ", ".join(s for i, s in enumerate(self.nodeInfoButtons) if self.NodeInfoW[i].isChecked())),
                             ("Line widths", ["Constant", "Proportion of all instances", "Proportion of parent's instances"][self.LineWidthMethod]),
                             ("Tree size", "%i nodes, %i leaves" % (orngTree.countNodes(self.tree), orngTree.countLeaves(self.tree)))])
        OWTreeViewer2D.sendReport(self)

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
        numInst=self.tree.distribution.cases
        for node in self.scene.nodeList:
            if self.NodeColorMethod == 0:   # default
                node.setBrush(QBrush(BodyColor_Default))
            elif self.NodeColorMethod == 1: # instances in node
                light = 400 - 300*node.tree.distribution.cases/numInst
                node.setBrush(QBrush(BodyCasesColor_Default.light(light)))
            elif self.NodeColorMethod == 2:
                light = 300-min([node.var,100])
                node.setBrush(QBrush(BodyCasesColor_Default.light(light)))
            elif self.NodeColorMethod == 3:
                light = 300 - min([node.dev,100])
                node.setBrush(QBrush(BodyCasesColor_Default.light(light)))
            elif self.NodeColorMethod == 4:
                light = 400 - 300*node.error
                node.setBrush(QBrush(BodyCasesColor_Default.light(light)))
        self.scene.update()
        self.treeNav.leech()

    def ctree(self, tree=None):
        self.send("Examples", None)
        OWTreeViewer2D.ctree(self, tree)

    def walkcreate(self, tree, parent=None, level=0, attrVal=""):
        node=RegressionNode(attrVal, tree, parent, self.scene)
        if tree.branches:
            for i in range(len(tree.branches)):
                if tree.branches[i]:
                    self.walkcreate(tree.branches[i],node,level+1,tree.branchDescriptions[i])
        return node

    def regressionBubbleConstructor(self, node, pos, scene):
        b=GraphicsBubbleInfo(node, pos, scene)
        rule=list(node.rule)
        #print node.rule, rule
        #rule.sort(lambda:a,b:a[0]<b[0])
        # merge
        if rule:
            try:
                rule=parseRules(list(rule))
            except:
                pass
            text="IF "+" AND\n  ".join([a[0].name+" "+a[1] for a in rule])+"\nTHEN "+str(node.defVal)
        else:
            text="THEN "+str(node.defVal)
        b.addTextLine(text)
        b.addTextLine()
        text="#instances:"+str(node.numInst)+"(%.1f" % (node.numInst/self.tree.distribution.cases*100)+"%)"
        b.addTextLine(text)
        b.addTextLine()
        b.addTextLine((node.tree.branches and "Partition on %s" % node.name) or "(leaf)")
        b.addTextLine()
        b.addTextLine(node.tree.nodeClassifier.classVar.name+" = "+str(node.defVal))
        b.show()
        return b


if __name__=="__main__":
    a = QApplication(sys.argv)
    ow = OWRegressionTreeViewer2D()

    data = orange.ExampleTable('../../doc/datasets/housing.tab')
    tree = orange.TreeLearner(data, storeExamples = 1)
    ow.ctree(tree)

    # here you can test setting some stuff
    ow.show()
    a.exec_()
    ow.saveSettings()
