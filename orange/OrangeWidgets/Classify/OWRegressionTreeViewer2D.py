"""
<name> Regression Tree Viewer 2D</name>
<description>Regression tree viewer (graph view).</description>
<icon>icons/RegressionTreeViewer2D.png</icon>
<priority>2120</priority>
"""
from OWTreeViewer2D import *

class RegressionNode(CanvasNode):
    def __init__(self, attrVal, *args):
        CanvasNode.__init__(self, *args)
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
        self.rule=(isinstance(self.parent, QCanvasRectangle) and self.parent.rule+[(self.parent.tree.branchSelector.classVar, attrVal)]) or []
        self.textAdvance=15

    def setSize(self,w,h):
        CanvasNode.setSize(self,w,h)
        self.updateText()

    def setBrush(self, brush):
        CanvasTextContainer.setBrush(self, brush)
        if self.textObj:
            self.textObj[0].setColor(Qt.black)
            
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

class OWRegressionTreeViewer2D(OWTreeViewer2D):
    def __init__(self, parent=None, signalManager = None, name='RegressionTreeViewer2D'):
        OWTreeViewer2D.__init__(self, parent, signalManager, name)
        
        self.inputs = [("Classification Tree", orange.TreeClassifier, self.ctree)]
        self.outputs = [("Classified Examples", ExampleTableWithClass), ("Examples", ExampleTable)]
        
        self.canvas=TreeCanvas(self)
        self.canvasView=TreeCanvasView(self, self.canvas, self.mainArea, "CView")
        layout=QVBoxLayout(self.mainArea)
        layout.addWidget(self.canvasView)
        self.canvas.resize(800,800)
        self.canvasView.bubbleConstructor=self.regressionBubbleConstructor
        self.navWidget=QWidget(None, "Navigator")
        self.navWidget.lay=QVBoxLayout(self.navWidget)
        canvas=TreeCanvas(self.navWidget)
        self.treeNav=TreeNavigator(self.canvasView,self,canvas,self.navWidget, "Nav")
        self.treeNav.setCanvas(canvas)
        self.navWidget.lay.addWidget(self.treeNav)
        self.canvasView.setNavigator(self.treeNav)
        self.navWidget.resize(400,400)
        self.navWidget.setCaption("Qt Navigator")
        OWGUI.button(self.TreeTab,self,"Navigator",self.toggleNavigator)
        self.setMouseTracking(True)

        nodeInfoBox = QVButtonGroup("Show Info On", self.NodeTab)
        nodeInfoButtons = ['Predicted Value', 'Variance', 'Deviance', ' Error', 'Number of Instances']
        nodeInfoSettings = ['maj', 'majp', 'tarp', 'error', 'inst']
        self.NodeInfoW = []; self.dummy = 0
        for i in range(len(nodeInfoButtons)):
            setattr(self, nodeInfoSettings[i], i in self.NodeInfo)
            w = OWGUI.checkBox(nodeInfoBox, self, nodeInfoSettings[i], \
                               nodeInfoButtons[i], callback=self.setNodeInfo, getwidget=1, id=i)
            self.NodeInfoW.append(w)

        OWGUI.comboBox(self.NodeTab, self, 'NodeColorMethod', items=['Default', 'Instances in Node', 'Variance', 'Deviance', 'Error'], box='Node Color',                            
                                callback=self.toggleNodeColor)
        
        OWGUI.button(self.controlArea, self, "Save As", callback=self.saveGraph)
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
        numInst=self.tree.distribution.cases
        for node in self.canvas.nodeList:
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
        self.canvas.update()
        self.treeNav.leech()

    def walkcreate(self, tree, parent=None, level=0, attrVal=""):
        node=RegressionNode(attrVal, tree, parent or self.canvas, self.canvas)
        if tree.branches:
            for i in range(len(tree.branches)):
                if tree.branches[i]:
                    self.walkcreate(tree.branches[i],node,level+1,tree.branchDescriptions[i])
        return node

    def regressionBubbleConstructor(self, node, pos, canvas):
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
            text="IF "+" AND\n  ".join([a[0].name+" "+a[1] for a in rule])+"\nTHEN "+str(node.defVal)
        else:
            text="THEN "+str(node.defVal)
        b.addTextLine(text)
        b.addTextLine()
        text="Instances:"+str(node.numInst)+"(%.1f" % (node.numInst/self.tree.distribution.cases*100)+"%)"
        b.addTextLine(text)
        b.addTextLine()
        b.addTextLine((node.tree.branches and "Partition on %s" % node.name) or "(leaf)")
        b.addTextLine()
        b.addTextLine(node.tree.nodeClassifier.classVar.name+" = "+str(node.defVal))
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
    ow = OWRegressionTreeViewer2D()
    a.setMainWidget(ow)

    data = orange.ExampleTable('../../doc/datasets/housing.tab')
    tree = orange.TreeLearner(data, storeExamples = 1)
    ow.ctree(tree)

    # here you can test setting some stuff
    ow.show()
    a.exec_loop()
    ow.saveSettings()
