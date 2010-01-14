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

class ClassificationTreeNode(GraphicsNode):
    def __init__(self, attr, tree, parent=None, parentItem=None, scene=None):
        GraphicsNode.__init__(self, tree, parent, parentItem, scene)
        self.attr = attr
        self.pie = PieChart(self.tree.distribution, 20, self, scene)
        self.majorityClass, self.majorityCount = max(self.tree.distribution.items(), key=lambda (key, val): val)
        self.updateHtml()
        
    def updateHtml(self):
        text = ""
        if self.attr:
            text += "%s<hr width=20000>" % self.attr
        text += "Majority class: <font color=%s>%s</font><br>" % (self.scene().colorPalette[self.tree.examples.domain.classVar.values.index(self.majorityClass)].name(), self.majorityClass)
        text += "Majority class probability: %.1f<br>" % (100.0 * float(self.majorityCount) / self.tree.distribution.abs)
        text += "Instances: %i" % self.tree.distribution.abs
        text += "<hr width=2000>" + ("Split on: %s" % self.tree.branchSelector.classVar.name if self.tree.branchSelector else "Leaf node")
        self.setHtml(text)
    
    def updateContents(self):
        GraphicsNode.updateContents(self)
        self.pie.setPos(self.rect().right(), self.rect().center().y())
        
    def rect(self):
        rect = GraphicsNode.rect(self)
        return rect.adjusted(0, 0, self.pie.boundingRect().width() / 2 if hasattr(self, "pie") else 0, 0)
    
    def boundingRect(self):
        return self.rect()
        
    def rule(self):
        return self.parent.rule() + [(self.parent.tree.branchSelector.classVar, self.attr)] if self.parent else []

import re
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

BodyColor_Default = QColor(Qt.gray) #QColor(255, 225, 10)
BodyCasesColor_Default = QColor(Qt.blue) #QColor(0, 0, 128)

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
        self.toggleZoomSlider()
#        self.scene.setSceneRect(0,0,800,800)

        self.connect(self.scene, SIGNAL("selectionChanged()"), self.updateSelection)

        self.navWidget= OWBaseWidget(self)
        self.navWidget.lay=QVBoxLayout(self.navWidget)

        scene=TreeGraphicsScene(self.navWidget)
        self.treeNav = TreeNavigator(self.sceneView)
        self.navWidget.lay.addWidget(self.treeNav)
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
        
        OWGUI.rubber(self.NodeTab)
        
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
        flags = sum(2**i for i, name in enumerate(['maj',
                        'majp', 'tarp', 'inst']) if getattr(self, name))
            
        for n in self.scene.nodes():
            self.updateNodeInfo(n, flags)
    
        self.scene.fixPos(self.rootNode, 10, 10)
        
    def updateNodeInfo(self, node, flags=31):
        fix = lambda str: str.replace(">", "&gt;").replace("<", "&lt;")
        text = ""
        if node.attr:
            text += "%s<hr width=20000>" % fix(node.attr)
        lines = []
        if flags & 1:
            lines += ["Majority class: <font color=%s>%s</font>" % (self.scene.colorPalette[node.tree.examples.domain.classVar.values.index(node.majorityClass)].name(), fix(node.majorityClass))]
        if flags & 2:
            lines += ["Majority class probability: %.1f" % (100.0 * float(node.majorityCount) / node.tree.distribution.abs)]
        if flags & 4:
            lines += ["Target class probability: %.1f" % (100.0 * float(node.tree.distribution[self.TargetClassIndex]) / node.tree.distribution.abs)]
        if flags & 8:
            lines += ["Instances: %i" % node.tree.distribution.cases]
        text += "<br>".join(lines) + "<hr width=2000>" + ("Split on: %s" % fix(node.tree.branchSelector.classVar.name) if node.tree.branchSelector else "Leaf node")
        node.setHtml(text)

    def activateLoadedSettings(self):
        if not self.tree:
            return
        OWTreeViewer2D.activateLoadedSettings(self)
        self.setNodeInfo()
        self.toggleNodeColor()

    def toggleNodeColor(self):
        for node in self.scene.nodes():
            if self.NodeColorMethod == 0:   # default
                color = BodyColor_Default
            elif self.NodeColorMethod == 1: # instances in node
                div = self.tree.distribution.cases
                if div > 1e-6:
                    light = 400 - 300*node.tree.distribution.cases/div
                else:
                    light = 100
                color = BodyCasesColor_Default.light(light)
            elif self.NodeColorMethod == 2: # majority class probability
                light=400- 300*float(node.majorityCount) / node.tree.distribution.abs
                color = self.scene.colorPalette[node.tree.examples.domain.classVar.values.index(node.majorityClass)].light(light)
            elif self.NodeColorMethod == 3: # target class probability
                div = node.tree.distribution.cases
                if div > 1e-6:
                    light=400-300*node.tree.distribution[self.TargetClassIndex]/div
                else:
                    light = 100
                color = self.scene.colorPalette[self.TargetClassIndex].light(light)
            elif self.NodeColorMethod == 4: # target class distribution
                div = self.tree.distribution[self.TargetClassIndex]
                if div > 1e-6:
                    light=200 - 100*node.tree.distribution[self.TargetClassIndex]/div
                else:
                    light = 100
                color = self.scene.colorPalette[self.TargetClassIndex].light(light)
            gradient = QLinearGradient(0, 0, 0, 100)
#                gradient.setStops([(0, QColor(Qt.gray).lighter(120)), (1, QColor(Qt.lightGray).lighter())])
            gradient.setStops([(0, color), (1, color.lighter())])
            node.setBackgroundBrush(QBrush(gradient))
        self.scene.update()

    def toggleTargetClass(self):
        if self.NodeColorMethod in [3,4]:
            self.toggleNodeColor()
        self.scene.update()

    def togglePies(self):
        for n in self.scene.nodes():
            n.pie.setVisible(self.ShowPies and n.isVisible())
        self.scene.update()

    def ctree(self, tree=None):
        self.send("Examples", None)
        self.closeContext()
        self.targetCombo.clear()
        if tree:
            for name in tree.tree.examples.domain.classVar.values:
                self.targetCombo.addItem(name)
            self.TargetClassIndex=0
            self.openContext("", tree.domain)
        else:
            self.openContext("", None)
        OWTreeViewer2D.ctree(self, tree)
        self.togglePies()

    def walkcreate(self, tree, parent=None, level=0, attrVal=""):
        node=ClassificationTreeNode(attrVal, tree, parent, None, self.scene)
        if parent:
            parent.graph_add_edge(GraphicsEdge(None, self.scene, node1=parent, node2=node))
        if tree.branches:
            for i in range(len(tree.branches)):
                if tree.branches[i]:
                    self.walkcreate(tree.branches[i],node,level+1,tree.branchDescriptions[i])
        return node
    
    def nodeToolTip(self, node):
        rule = list(node.rule())
        fix = lambda str: str.replace(">", "&gt;").replace("<", "&lt;")
        if rule:
            try:
                rule=parseRules(list(rule))
            except:
                pass
            text="<b>IF</b> "+" <b>AND</b><br>  ".join([fix(a[0].name+" = "+a[1]) for a in rule])+"\n<br><b>THEN</b> "+fix(node.majorityClass) + "<hr>"
        else:
            text="<b>THEN</b> "+fix(node.majorityClass) + "<hr>"
        text += "Instances: %(ninst)i (%(prop).1f%%)<hr>" % {"ninst": node.tree.distribution.cases, "prop": float(node.tree.distribution.cases)/self.tree.distribution.cases*100}
        
        text += "<br>".join(["<font color=%(color)s>%(name)s: %(num)i (%(ratio).1f% %)</font>" % \
                             {"name":fix(d[0]), "num":int(d[1]), "ratio":d[1]/sum(node.tree.distribution)*100, "color":self.scene.colorPalette[i].name()}\
                             for i,d in enumerate(node.tree.distribution.items()) if d[1]!=0])
        text += "<hr>Partition on: %(nodename)s" % {"nodename": node.tree.branchSelector.classVar.name} if node.tree.branches else ""
        return text

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
