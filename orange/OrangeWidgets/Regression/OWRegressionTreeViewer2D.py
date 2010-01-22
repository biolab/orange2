"""
<name> Regression Tree Graph</name>
<description>Regression tree viewer (graph view).</description>
<icon>icons/RegressionTreeGraph.png</icon>
<contact>Ales Erjavec (ales.erjavec(@at@)fri.uni-lj.si)</contact>
<priority>2110</priority>
"""
from OWTreeViewer2D import *
import re

        
class RegressionTreeNode(GraphicsNode):
    def __init__(self, attr, tree, parent=None, *args):
        GraphicsNode.__init__(self, tree, parent, *args)
        self.attr = attr
        fm = QFontMetrics(self.document().defaultFont())
        self.attr_text_w = fm.width(str(self.attr if self.attr else ""))
        self.attr_text_h = fm.lineSpacing()
        self.line_descent = fm.descent()
        
    def rule(self):
        return self.parent.rule() + [(self.parent.tree.branchSelector.classVar, self.attr)] if self.parent else []
    
    def rect(self):
        rect = GraphicsNode.rect(self)
        rect.setRight(max(rect.right(), getattr(self, "attr_text_w", 0)))
        return rect
    
    def boundingRect(self):
        if hasattr(self, "attr"):
            attr_rect = QRectF(QPointF(0, -self.attr_text_h), QSizeF(self.attr_text_w, self.attr_text_h))
        else:
            attr_rect = QRectF(0, 0, 1, 1)
        rect = self.rect().adjusted(-5, -5, 5, 5)
        return rect | GraphicsNode.boundingRect(self) | attr_rect
    
    def paint(self, painter, option, widget=None):
        if self.isSelected():
            option.state = option.state.__xor__(QStyle.State_Selected)
        if self.isSelected():
            painter.save()
            painter.setBrush(QBrush(QColor(125, 162, 206, 192)))
            painter.drawRoundedRect(self.boundingRect().adjusted(1, 1, -1, -1), self.borderRadius, self.borderRadius)
            painter.restore()
        painter.setFont(self.document().defaultFont())
        painter.drawText(QPointF(0, -self.line_descent), str(self.attr) if self.attr else "")
        painter.save()
        painter.setBrush(self.backgroundBrush)
        rect = self.rect()
        painter.drawRoundedRect(rect, self.borderRadius, self.borderRadius)
        painter.restore()
        painter.setClipRect(rect | QRectF(QPointF(0, 0), self.document().size()))
        return QGraphicsTextItem.paint(self, painter, option, widget)
        
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
        s1=set([s.strip(" ") for s in r1.split(",")])
        s2=set([s.strip(" ") for s in r2.split(",")])
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
#BodyColor_Default = QColor(Qt.gray)
BodyCasesColor_Default = QColor(0, 0, 128)

class OWRegressionTreeViewer2D(OWTreeViewer2D):
    nodeColorOpts = ['Default', 'Instances in node', 'Variance', 'Deviation', 'Error']
    nodeInfoButtons = ['Predicted value', 'Variance', 'Deviation', 'Error', 'Number of instances']

    def __init__(self, parent=None, signalManager = None, name='RegressionTreeViewer2D'):
        OWTreeViewer2D.__init__(self, parent, signalManager, name)

        self.inputs = [("Classification Tree", orange.TreeClassifier, self.ctree)]
        self.outputs = [("Examples", ExampleTable)]
        
        self.showNodeInfoText = False
        
        self.scene = TreeGraphicsScene(self)
        self.sceneView = TreeGraphicsView(self, self.scene)
        self.mainArea.layout().addWidget(self.sceneView)
        self.toggleZoomSlider()
        
        self.connect(self.scene, SIGNAL("selectionChanged()"), self.updateSelection)

        self.navWidget = OWBaseWidget(self) 
        self.navWidget.lay=QVBoxLayout(self.navWidget)
        
#        scene = TreeGraphicsScene(self.navWidget)
        self.treeNav = TreeNavigator(self.sceneView) #,self,scene,self.navWidget)
#        self.treeNav.setScene(scene)
        self.navWidget.layout().addWidget(self.treeNav)
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
        
        OWGUI.rubber(self.NodeTab)
        
        OWGUI.button(self.controlArea, self, "Save As", callback=self.saveGraph, debuggingEnabled = 0)
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
        flags = sum(2**i for i, name in enumerate(['maj', 'majp', 'tarp', 'error', 'inst']) if getattr(self, name)) 
        for n in self.scene.nodes():
            if hasattr(n, "_rect"):
                delattr(n, "_rect")
            self.updateNodeInfo(n, flags)
        if True:
            w = max([n.rect().width() for n in self.scene.nodes()] + [0])
            for n in self.scene.nodes():
                n.setRect(n.rect() | QRectF(0, 0, w, 1))
        self.scene.fixPos(self.rootNode, 10, 10)
        self.scene.update()
        
    def updateNodeInfo(self, node, flags=63):
        fix = lambda str: str.replace(">", "&gt;").replace("<", "&lt;")
        text = ""
#        if node.attr:
#            text += "%s<hr width=20000>" % fix(node.attr)
        lines = []
        if flags & 1:
            start = "Predicted value: " if self.showNodeInfoText else ""
            lines += [start + fix(str(node.tree.nodeClassifier.defaultValue))]
        if flags & 2:
            start = "Variance: " if self.showNodeInfoText else ""
            lines += [start + "%.1f" % node.tree.distribution.var()]
        if flags & 4:
            start = "Deviance: " if self.showNodeInfoText else ""
            lines += [start + "%.1f" % node.tree.distribution.dev()]
        if flags & 8:
            start = "Error: " if self.showNodeInfoText else ""
            lines += [start + "%.1f" % node.tree.distribution.error()]
        if flags & 16:
            start = "Number of instances: " if self.showNodeInfoText else ""
            lines += [start + "%i" % node.tree.distribution.cases]
        text += "<br>".join(lines)
        if node.tree.branchSelector:
            text += "<hr>%s" % (fix(node.tree.branchSelector.classVar.name))
        else:
            text += "<hr>%s" % (fix(str(node.tree.nodeClassifier.defaultValue)))
                               
        node.setHtml(text) 

    def activateLoadedSettings(self):
        if not self.tree:
            return
        OWTreeViewer2D.activateLoadedSettings(self)
        self.setNodeInfo()
        self.toggleNodeColor()

    def toggleNodeColor(self):
        numInst=self.tree.distribution.cases
        for node in self.scene.nodes():
            if self.NodeColorMethod == 0:   # default
                color = BodyColor_Default
            elif self.NodeColorMethod == 1: # instances in node
                light = 400 - 300*node.tree.distribution.cases/numInst
                color = BodyCasesColor_Default.light(light)
            elif self.NodeColorMethod == 2:
                light = 300-min([node.tree.distribution.var(),100])
                color = BodyCasesColor_Default.light(light)
            elif self.NodeColorMethod == 3:
                light = 300 - min([node.tree.distribution.dev(),100])
                color = BodyCasesColor_Default.light(light)
            elif self.NodeColorMethod == 4:
                light = 400 - 300*node.tree.distribution.error()
                color = BodyCasesColor_Default.light(light)
#            gradient = QLinearGradient(0, 0, 0, 100)
#            gradient.setStops([(0, color.lighter(120)), (1, color.lighter())])
#            node.backgroundBrush = QBrush(gradient)
            node.backgroundBrush = QBrush(color)

        self.scene.update()
#        self.treeNav.leech()

    def ctree(self, tree=None):
        self.send("Examples", None)
        OWTreeViewer2D.ctree(self, tree)

    def walkcreate(self, tree, parent=None, level=0, attrVal=""):
        node=RegressionTreeNode(attrVal, tree, parent, None, self.scene)
        if parent:
            parent.graph_add_edge(GraphicsEdge(None, self.scene, node1=parent, node2=node))
        if tree.branches:
            for i in range(len(tree.branches)):
                if tree.branches[i]:
                    self.walkcreate(tree.branches[i],node,level+1,tree.branchDescriptions[i])
        return node
    
    def nodeToolTip(self, node):
        rule=list(node.rule())
        fix = lambda str: str.replace(">", "&gt;").replace("<", "&lt;")
        if rule:
            try:
                rule=parseRules(list(rule))
            except:
                pass
            text="<b>IF</b> "+" <b>AND</b><br>\n  ".join([fix(a[0].name+" "+a[1]) for a in rule])+"\n<br><b>THEN</b> "+fix(str(node.tree.nodeClassifier.defaultValue))
        else:
            text="<b>THEN</b> "+fix(str(node.tree.nodeClassifier.defaultValue))
        text += "<hr>Instances: %i (%.1f%%)" % (node.tree.distribution.cases, node.tree.distribution.cases/self.tree.distribution.cases*100)
        text += "<hr>Partition on %s<hr>" % node.tree.branchSelector.classVar.name if node.tree.branchSelector else "<hr>"
        text += fix(node.tree.nodeClassifier.classVar.name + " = " + str(node.tree.nodeClassifier.defaultValue))
        return text

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
