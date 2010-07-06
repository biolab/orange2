"""
<name>Hierarchical Clustering</name>
<description>Hierarchical clustering based on distance matrix, and a dendrogram viewer.</description>
<icon>icons/HierarchicalClustering.png</icon>
<contact>Ales Erjavec (ales.erjavec(@at@)fri.uni-lj.si)</contact> 
<priority>2100</priority>
"""
from __future__ import with_statement

from OWWidget import *
from OWQCanvasFuncts import *
import OWGUI
import OWColorPalette
import math
import os

from OWDlgs import OWChooseImageSizeDlg

from PyQt4.QtCore import *
from PyQt4.QtGui import *

try:
    from OWDataFiles import DataFiles
except:
    class DataFiles(object):
        pass

class recursion_limit(object):
    def __init__(self, limit=1000):
        self.limit = limit
        
    def __enter__(self):
        self.old_limit = sys.getrecursionlimit()
        sys.setrecursionlimit(self.limit)
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.setrecursionlimit(self.old_limit)

class OWHierarchicalClustering(OWWidget):
    settingsList=["Linkage", "OverwriteMatrix", "Annotation", "Brightness", "PrintDepthCheck",
                "PrintDepth", "HDSize", "VDSize", "ManualHorSize","AutoResize",
                "TextSize", "LineSpacing", "ZeroOffset", "SelectionMode", "DisableHighlights",
                "DisableBubble", "ClassifySelected", "CommitOnChange", "ClassifyName", "addIdAs"]
    
    contextHandlers={"":DomainContextHandler("", [ContextField("Annotation", DomainContextHandler.Required)])}
    
    def __init__(self, parent=None, signalManager=None):
        #OWWidget.__init__(self, parent, 'Hierarchical Clustering')
        OWWidget.__init__(self, parent, signalManager, 'Hierarchical Clustering', wantGraph=True)
        self.inputs=[("Distance matrix", orange.SymMatrix, self.dataset)]
        self.outputs=[("Selected Examples", ExampleTable), ("Unselected Examples", ExampleTable), ("Structured Data Files", DataFiles)]
        self.linkage=[("Single linkage", orange.HierarchicalClustering.Single),
                        ("Average linkage", orange.HierarchicalClustering.Average),
                        ("Ward's linkage", orange.HierarchicalClustering.Ward),
                        ("Complete linkage", orange.HierarchicalClustering.Complete),
                     ]
        self.Linkage=3
        self.OverwriteMatrix=0
        self.Annotation=0
        self.Brightness=5
        self.PrintDepthCheck=0
        self.PrintDepth=100
        self.HDSize=500         #initial horizontal and vertical dendrogram size
        self.VDSize=800
        self.ManualHorSize=0
        self.AutoResize=0
        self.TextSize=8
        self.LineSpacing=4
        self.SelectionMode=0
        self.ZeroOffset=1
        self.DisableHighlights=0
        self.DisableBubble=0
        self.ClassifySelected=0
        self.CommitOnChange=0
        self.ClassifyName="HC_class"
        self.loadSettings()
        self.AutoResize=False
        self.inputMatrix=None
        self.matrixSource="Unknown"
        self.rootCluster=None
        self.selectedExamples=None
        self.ctrlPressed=FALSE
        self.addIdAs = 0
        self.settingsChanged = False

        self.linkageMethods=[a[0] for a in self.linkage]

        #################################
        ##GUI
        #################################

        #Tabs
##        self.tabs = OWGUI.tabWidget(self.controlArea)
##        self.settingsTab = OWGUI.createTabPage(self.tabs, "Settings")
##        self.selectionTab= OWGUI.createTabPage(self.tabs, "Selection")

        #HC Settings
        OWGUI.comboBox(self.controlArea, self, "Linkage", box="Linkage",
                items=self.linkageMethods, tooltip="Choose linkage method",
                callback=self.constructTree, addSpace = 16)
        #Label
        box = OWGUI.widgetBox(self.controlArea, "Annotation", addSpace = 16)
        self.labelCombo=OWGUI.comboBox(box, self, "Annotation",
                items=["None"],tooltip="Choose label attribute",
                callback=self.updateLabel)

        OWGUI.spin(box, self, "TextSize", label="Text size",
                        min=5, max=15, step=1, callback=self.applySettings, controlWidth=40)
        OWGUI.spin(box,self, "LineSpacing", label="Line spacing",
                        min=2,max=8,step=1, callback=self.applySettings, controlWidth=40)
        
#        OWGUI.checkBox(box, self, "DisableBubble", "Disable bubble info")


        #Dendrogram graphics settings
        dendrogramBox=OWGUI.widgetBox(self.controlArea, "Limits", addSpace=16)
        #OWGUI.spin(dendrogramBox, self, "Brightness", label="Brigthtness",min=1,max=9,step=1)
        
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignLeft)
        sw = OWGUI.widgetBox(dendrogramBox, orientation="horizontal", addToLayout=False) #QWidget(dendrogramBox)
        cw = OWGUI.widgetBox(dendrogramBox, orientation="horizontal", addToLayout=False) #QWidget(dendrogramBox)
        
        sllp = OWGUI.hSlider(sw, self, "PrintDepth", minValue=1, maxValue=50, callback=self.applySettings)
        cblp = OWGUI.checkBox(cw, self, "PrintDepthCheck", "Show to depth", callback = self.applySettings, disables=[sw])
        form.addRow(cw, sw)
#        dendrogramBox.layout().addLayout(form)
#        box = OWGUI.widgetBox(dendrogramBox, orientation=form)
        
        option = QStyleOptionButton()
        cblp.initStyleOption(option)
        checkWidth = cblp.style().subElementRect(QStyle.SE_CheckBoxContents, option, cblp).x()
        
#        ib = OWGUI.indentedBox(dendrogramBox, orientation = 0)
#        ib = OWGUI.widgetBox(dendrogramBox, margin=0)
#        ib.layout().setContentsMargins(checkWidth, 5, 5, 5)
#        OWGUI.widgetLabel(ib, "Depth"+ "  ")
#        slpd = OWGUI.hSlider(ib, self, "PrintDepth", minValue=1, maxValue=50, label="Depth  ", callback=self.applySettings)
#        cblp.disables.append(ib)
#        cblp.makeConsistent()
        
#        OWGUI.separator(dendrogramBox)
        #OWGUI.spin(dendrogramBox, self, "VDSize", label="Vertical size", min=100,
        #        max=10000, step=10)
        
#        form = QFormLayout()
        sw = OWGUI.widgetBox(dendrogramBox, orientation="horizontal", addToLayout=False) #QWidget(dendrogramBox)
        cw = OWGUI.widgetBox(dendrogramBox, orientation="horizontal", addToLayout=False) #QWidget(dendrogramBox)
        
        hsb = OWGUI.spin(sw, self, "HDSize", min=200, max=10000, step=10, callback=self.applySettings, callbackOnReturn = False)
        cbhs = OWGUI.checkBox(cw, self, "ManualHorSize", "Horizontal size", callback=self.applySettings, disables=[sw])
        
#        ib = OWGUI.widgetBox(dendrogramBox, margin=0)
#        self.hSizeBox=OWGUI.spin(ib, self, "HDSize", label="Size"+"  ", min=200,
#                max=10000, step=10, callback=self.applySettings, callbackOnReturn = True, controlWidth=45)
#        self.hSizeBox.layout().setContentsMargins(checkWidth, 5, 5, 5)
        self.hSizeBox = hsb
        form.addRow(cw, sw)
        dendrogramBox.layout().addLayout(form)
        
        
#        cbhs.disables.append(self.hSizeBox)
#        cbhs.makeConsistent()
        
        #OWGUI.checkBox(dendrogramBox, self, "ManualHorSize", "Fit horizontal size")
        #OWGUI.checkBox(dendrogramBox, self, "AutoResize", "Auto resize")

        box = OWGUI.widgetBox(self.controlArea, "Selection")
        OWGUI.checkBox(box, self, "SelectionMode", "Show cutoff line", callback=self.updateCutOffLine)
        cb = OWGUI.checkBox(box, self, "ClassifySelected", "Append cluster IDs", callback=self.commitDataIf)
#        self.classificationBox = ib = OWGUI.indentedBox(box, sep=0)
        self.classificationBox = ib = OWGUI.widgetBox(box, margin=0)
        
        form = QWidget()
        le = OWGUI.lineEdit(form, self, "ClassifyName", None, callback=None, orientation="horizontal")
        self.connect(le, SIGNAL("editingFinished()"), self.commitDataIf)
#        le_layout.addWidget(le.placeHolder)
#        OWGUI.separator(ib, height = 4)

        aa = OWGUI.comboBox(form, self, "addIdAs", label = None, orientation = "horizontal", items = ["Class attribute", "Attribute", "Meta attribute"], callback=self.commitDataIf)
        
        layout = QFormLayout()
        layout.setSpacing(8)
        layout.setContentsMargins(0, 5, 0, 5)
        layout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        layout.setLabelAlignment(Qt.AlignLeft)# | Qt.AlignJustify)
        layout.addRow("Name  ", le)
        layout.addRow("Place  ", aa)
        
        form.setLayout(layout)
        
        ib.layout().addWidget(form)
        ib.layout().setContentsMargins(checkWidth, 5, 5, 5)
        
        cb.disables.append(ib)
        cb.makeConsistent()
        
        OWGUI.separator(box)
        cbAuto = OWGUI.checkBox(box, self, "CommitOnChange", "Commit on change")
        btCommit = OWGUI.button(box, self, "&Commit", self.commitData)
        OWGUI.setStopper(self, btCommit, cbAuto, "settingsChanged", self.commitData)
        
        
        OWGUI.rubber(self.controlArea)
#        OWGUI.button(self.controlArea, self, "&Save Graph", self.saveGraph, debuggingEnabled = 0)
        self.connect(self.graphButton, SIGNAL("clicked()"), self.saveGraph)

        scale=QGraphicsScene(self)
        self.headerView=ScaleView(self, scale, self.mainArea)
        self.footerView=ScaleView(self, scale, self.mainArea)
        self.dendrogram = Dendrogram(self)
        self.dendrogramView = DendrogramView(self.dendrogram, self.mainArea)
        
        self.connect(self.dendrogram, SIGNAL("sceneRectChanged(QRectF)"), self.footerView.sceneRectUpdated)
    
        self.mainArea.layout().addWidget(self.headerView)
        self.mainArea.layout().addWidget(self.dendrogramView)
        self.mainArea.layout().addWidget(self.footerView)
        
        self.dendrogram.header=self.headerView
        self.dendrogram.footer=self.footerView

        self.connect(self.dendrogramView.horizontalScrollBar(),SIGNAL("valueChanged(int)"),
                self.footerView.horizontalScrollBar().setValue)
        self.connect(self.dendrogramView.horizontalScrollBar(),SIGNAL("valueChanged(int)"),
                self.headerView.horizontalScrollBar().setValue)
        self.dendrogram.setSceneRect(0, 0, self.HDSize,self.VDSize)
        self.dendrogram.update()
        self.resize(800, 500)
        
        self.matrix = None
        self.selectionList = []

    def sendReport(self):
        self.reportSettings("Settings",
                            [("Linkage", self.linkageMethods[self.Linkage]),
                             ("Annotation", self.labelCombo.currentText()),
                             self.PrintDepthCheck and ("Shown depth limited to", self.PrintDepth),
                             self.SelectionMode and hasattr(self.dendrogram, "cutOffHeight") and ("Cutoff line at", self.dendrogram.cutOffHeight)])
        
        self.reportSection("Dendrogram")
        canvases = header, graph, footer = self.headerView.scene(), self.dendrogramView.scene(), self.footerView.scene()
        buffer = QPixmap(max(c.width() for c in canvases), sum(c.height() for c in canvases))
        painter = QPainter(buffer)
        painter.fillRect(buffer.rect(), QBrush(QColor(255, 255, 255)))
        header.render(painter, QRectF(0, 0, header.width(), header.height()), QRectF(0, 0, header.width(), header.height()))
        graph.render(painter, QRectF(0, header.height(), graph.width(), graph.height()), QRectF(0, 0, graph.width(), graph.height()))
        footer.render(painter, QRectF(0, header.height()+graph.height(), footer.width(), footer.height()), QRectF(0, 0, footer.width(), footer.height()))
        painter.end()
        self.reportImage(lambda filename: buffer.save(filename, os.path.splitext(filename)[1][1:]))
        
    def dataset(self, data):
        self.matrix=data
        self.closeContext()
        if not self.matrix:
            self.rootCluster=None
            self.selectedExamples=None
            self.dendrogram.clear()
            self.footerView.clear()
            self.labelCombo.clear()
            self.send("Selected Examples", None)
            self.send("Unselected Examples", None)
            self.classificationBox.setDisabled(True)
            return

        self.matrixSource="Unknown"
        items=getattr(self.matrix, "items")
        if isinstance(items, orange.ExampleTable): #Example Table from Example Distance

            self.labels=["None", "Default"]+ \
                         [a.name for a in items.domain.attributes]
            if items.domain.classVar:
                self.labels.append(items.domain.classVar.name)

            self.labelInd=range(len(self.labels)-2)
            self.labels.extend([m.name for m in items.domain.getmetas().values()])
            self.labelInd.extend(items.domain.getmetas().keys())
            self.numMeta=len(items.domain.getmetas())
            self.metaLabels=items.domain.getmetas().values()
            self.matrixSource="Example Distance"
        elif isinstance(items, list):   #Structured data files from Data Distance
            self.labels=["None", "Default", "Name", "Strain"]
            self.Annotation=0
            self.matrixSource="Data Distance"
        else:   #From Attribute Distance
            self.labels=["None", "Attribute Name"]
            self.Annotation=1
            self.matrixSource="Attribute Distance"
        self.labelCombo.clear()
        for a in self.labels:
            self.labelCombo.addItem(a)
        if self.labelCombo.count()<self.Annotation-1:
                self.Annotation=0
        self.labelCombo.setCurrentIndex(self.Annotation)
        if self.matrixSource=="Example Distance":
            self.classificationBox.setDisabled(False)
        else:
            self.classificationBox.setDisabled(True)
        if self.matrixSource=="Example Distance":
            self.openContext("", items)
        self.error(0)
        try:
            self.constructTree()
        except orange.KernelException, ex:
            self.error(0, "Could not cluster data! %s" % ex.message)
            self.dataset(None)
            

    def updateLabel(self):
#        self.rootCluster.mapping.setattr("objects", self.matrix.items)
#        self.dendrogram.updateLabel()
#        return
    
        if self.matrix is None:
            return
        items=self.matrix.items
        if self.Annotation==0:
            self.rootCluster.mapping.setattr("objects",
                [" " for i in range(len(items))])

        elif self.Annotation==1:
            if self.matrixSource=="Example Distance" or self.matrixSource=="Data Distance":
                self.rootCluster.mapping.setattr("objects", range(len(items)))
            elif self.matrixSource=="Attribute Distance":
                self.rootCluster.mapping.setattr("objects", [a.name for a in items])
        elif self.matrixSource=="Example Distance":
            try:
                self.rootCluster.mapping.setattr("objects",
                                [str(e[self.labelInd[self.Annotation-2]]) for e in items])
            except IndexError:
                self.Annotation=0
                self.rootCluster.mapping.setattr("objects", [str(e[0]) for e in items])
        elif self.matrixSource=="Data Distance":
            if self.Annotation==2:
                self.rootCluster.mapping.setattr("objects", [getattr(a, "name", "") for a in items])
            else:
                self.rootCluster.mapping.setattr("objects", [getattr(a, "strain", "") for a in items])
        self.dendrogram.updateLabel()

    def constructTree(self):
        if self.matrix:
            self.progressBarInit()
            self.rootCluster=orange.HierarchicalClustering(self.matrix,
                linkage=self.linkage[self.Linkage][1],
                overwriteMatrix=self.OverwriteMatrix,
                progressCallback=self.progressBarSet)
            self.progressBarFinished()
            self.dendrogram.displayTree(self.rootCluster)
            self.updateLabel()

    def applySettings(self):
        self.dendrogram.setSceneRect(0, 0, self.HDSize, self.VDSize)
        self.dendrogram.displayTree(self.rootCluster)

    def progressBarSet(self, value, a):
        OWWidget.progressBarSet(self, value*100)

    def keyPressEvent(self, key):
        if key.key()==Qt.Key_Control:
            self.ctrlPressed=TRUE
        else:
            OWWidget.keyPressEvent(self, key)

    def keyReleaseEvent(self, key):
        if key.key()==Qt.Key_Control:
            self.ctrlPressed=FALSE
        else:
            OWWidget.keyReleaseEvent(self, key)

    def updateCutOffLine(self):
        try:
            if self.SelectionMode:
                self.dendrogram.cutOffLine.show()
                self.footerView.scene().marker.show()
            else:
                self.dendrogram.cutOffLine.hide()
                self.footerView.scene().marker.hide()
        except RuntimeError: # underlying C/C++ object has been deleted
            pass
        self.dendrogram.update()
        self.footerView.scene().update()

    def updateSelection(self, selection):
        if self.matrixSource=="Attribute Distance":
            return
        self.selectionList=selection
        if self.dendrogram.cutOffLineDragged==False:
            self.commitDataIf()

    def commitDataIf(self):
        if self.CommitOnChange:
            self.commitData()
        else:
            self.settingsChanged = True

    def selectionChange(self):
        if self.CommitOnChange:
            self.commitData()

    def commitData(self):
        self.settingsChanged = False
        self.selection=[]
        selection=self.selectionList
        maps=[self.rootCluster.mapping[c.first:c.last] for c in [e.rootCluster for e in selection]]
        self.selection=[self.matrix.items[k] for k in [j for i in range(len(maps)) for j in maps[i]]]
        
        if not self.selection:
            self.send("Selected Examples",None)
            self.send("Unselected Examples",None)
            self.send("Structured Data Files", None)
            return
        items = getattr(self.matrix, "items")
        if self.matrixSource == "Example Distance":
            unselected = [item for item in items if item not in self.selection]
            if self.ClassifySelected:
                clustVar=orange.EnumVariable(str(self.ClassifyName) ,
                            values=["Cluster " + str(i) for i in range(len(maps))] + ["Other"])
                c=[i for i in range(len(maps)) for j in maps[i]]
                origDomain = items.domain
                if self.addIdAs == 0:
                    domain=orange.Domain(origDomain.attributes,clustVar)
                    if origDomain.classVar:
                        domain.addmeta(orange.newmetaid(), origDomain.classVar)
                    aid = -1
                elif self.addIdAs == 1:
                    domain=orange.Domain(origDomain.attributes+[clustVar], origDomain.classVar)
                    aid = len(origDomain.attributes)
                else:
                    domain=orange.Domain(origDomain.attributes, origDomain.classVar)
                    aid=orange.newmetaid()
                    domain.addmeta(aid, clustVar)

                domain.addmetas(origDomain.getmetas())
                table1=orange.ExampleTable(domain) #orange.Domain(self.matrix.items.domain, classVar))
                table1.extend(orange.ExampleTable(self.selection) if self.selection else [])
                for i in range(len(self.selection)):
                    table1[i][aid] = clustVar("Cluster " + str(c[i]))

                table2 = orange.ExampleTable(domain)
                table2.extend(orange.ExampleTable(unselected) if unselected else [])
                for ex in table2:
                    ex[aid] = clustVar("Other")

                self.selectedExamples=table1
                self.unselectedExamples=table2
            else:
                table1=orange.ExampleTable(self.selection)
                self.selectedExamples=table1
                self.unselectedExamples = orange.ExampleTable(unselected) if unselected else None
            self.send("Selected Examples",self.selectedExamples)
            self.send("Unselected Examples", self.unselectedExamples)

        elif self.matrixSource=="Data Distance":
            names=list(set([d.strain for d in self.selection]))
            data=[(name, [d for d in filter(lambda a:a.strain==name, self.selection)]) for name in names]
            self.send("Structured Data Files",data)
            
    def saveGraph(self):
       sizeDlg = OWChooseImageSizeDlg(self.dendrogram)
       sizeDlg.exec_()

leftMargin=10
rightMargin=10
topMargin=10
bottomMargin=10
polyOffset=5
scaleHeight=20

class DendrogramView(QGraphicsView):
    def __init__(self,*args):
        apply(QGraphicsView.__init__, (self,)+args)
        self.viewport().setMouseTracking(True)
        self.setAlignment(Qt.AlignLeft | Qt.AlignTop)

    def resizeEvent(self, e):
        QGraphicsView.resizeEvent(self,e)
        if not self.scene().parent.ManualHorSize:
            self.scene().displayTree(self.scene().rootCluster)
        #self.updateContents()

class Dendrogram(QGraphicsScene):
    def __init__(self, *args):
        apply(QGraphicsScene.__init__, (self,)+args)
        self.parent=args[0]
        self.rootCluster=None
        self.rootTree=None
        self.highlighted=None #MyCanvasRect(None)
        self.header=None
        self.footer=None
        self.cutOffLineDragged=False
        self.selectionList=[]
        self.pen=QPen(QColor("blue"))
        self.selectedPen=QPen(QColor("red"))
        self.highlightPen=QPen(QColor("blue"),2)
        self.brush=QBrush(QColor("white"))
        self.font=QFont()
        self.rectObj=[]
        self.textObj=[]
        self.otherObj=[]
        self.cutOffLine=QGraphicsLineItem(None, self)
        self.cutOffLine.setPen( QPen(QColor("black"),2))
        self.bubbleRect=BubbleRect(None)
        #self.setDoubleBuffering(True)
        self.holdoff=False
        self.treeAreaWidth = 0
        self.selectionList = []
        #self.setMouseTrackingEnabled(True)

    def displayTree(self, root):
        self.clear()
        self.rootCluster=root
        if not self.rootCluster:
            return
        if not self.parent.ManualHorSize:
            width=self.parent.dendrogramView.size().width()
        else:
            width=self.parent.HDSize
        self.setSceneRect(0, 0, width, self.height())
        self.textAreaWidth=100
##        else:
##            self.textSize=self.parent.TextSize
##            self.textAreaWidth=100 #self.textSize*10
        self.textSize=self.parent.TextSize
        self.gTextPosInc=self.textSize+self.parent.LineSpacing
        self.gTextPos=topMargin
        self.gZPos=-1
        self.treeHeight=root.height
        self.treeAreaWidth=self.width()-leftMargin-self.textAreaWidth
        self.font.setPointSize(self.textSize)
        self.header.scene().setSceneRect(0, 0, self.width()+20, scaleHeight)
        with recursion_limit(sys.getrecursionlimit() + len(self.rootCluster) + 1):
            (self.rootGraphics,a)=self.drawTree(self.rootCluster,0)
        self.updateLabel()
        for old in self.oldSelection:
            for new in self.rectObj:
                if new.cluster==old:
                    self.addSelection(new)

        self.header.drawScale(self.treeAreaWidth, root.height)
        
        self.bubbleRect=BubbleRect(None)
        self.addItem(self.bubbleRect)
        self.otherObj.append(self.bubbleRect)
#        fix=max([a.boundingRect().width() for a in self.textObj])
#        self.setSceneRect(0, 0, leftMargin+self.treeAreaWidth+fix+rightMargin, 2*topMargin+self.gTextPos)
        self.setSceneRect(self.itemsBoundingRect().adjusted(-5, -5, 5, 5))
        self.cutOffLine.setLine(0,0,0,self.height())
        self.header.moveMarker(0)
        self.update()

    def drawTree(self, cluster, l):
        level=l+1
        if cluster.branches and (level<=self.parent.PrintDepth or \
                not self.parent.PrintDepthCheck):
            (leftR, hi)=self.drawTree(cluster.left, level)
            (rightR,low)=self.drawTree(cluster.right, level)
            top=leftMargin+self.treeAreaWidth- \
                self.treeAreaWidth*cluster.height/(self.treeHeight or 1)
            rectW=self.width()-self.textAreaWidth-top
            rectH=low-hi
            rect=MyCanvasRect(top, hi, rectW+2, rectH)
            self.addItem(rect)
            rect.left=leftR
            rect.right=rightR
            rect.cluster=cluster
            rect.setBrush(self.brush)
            rect.setPen(self.pen)
            rect.setZValue(self.gZPos)
            self.gZPos-=1
            rect.show()
            self.rectObj.append(rect)
            return (rect, (hi+low)/2)
        else:
            text=MyCanvasText(self, font=self.font, alignment=Qt.AlignLeft)
            text.setPlainText(" ")
            text.cluster=cluster
            text.setPos(leftMargin+self.treeAreaWidth+5,math.ceil(self.gTextPos))
            text.setZValue(1)
            self.textObj.append(text)
            self.gTextPos+=self.gTextPosInc
            return (None, self.gTextPos-self.gTextPosInc/2)

    def removeItem(self, item):
        if item.scene() is self:
            QGraphicsScene.removeItem(self, item)

    def clear(self):
        for a in self.rectObj:
            self.removeItem(a)
        for a in self.textObj:
            self.removeItem(a)
        for a in self.otherObj:
            self.removeItem(a)
        self.rectObj=[]
        self.textObj=[]
        self.otherObj=[]
        self.rootGraphics=None
        self.rootCluster=None
        self.cutOffLine.hide()
        self.cutOffLine.setPos(0,0)
        self.oldSelection=[a.rootCluster for a in self.selectionList]
        self.clearSelection()
        self.update()

    def updateLabel(self):
        if not self.rootCluster:
            return
        for a in self.textObj:
            if len(a.cluster)>1 and not self.parent.Annotation==0:
                a.setPlainText("(%i items)" % len(a.cluster))
            else:
                a.setPlainText(str(a.cluster[0]))
#        self.setSceneRect(0, 0, leftMargin+self.treeAreaWidth+max([a.boundingRect().width() \
#                for a in self.textObj])+rightMargin, self.height())
        self.setSceneRect(self.itemsBoundingRect().adjusted(-5, -5, 5, 5))
        self.update()

    def highlight(self, objList):
        if not self.rootCluster:
            return
        if self.parent.DisableHighlights:
            if self.highlighted:
                self.highlighted.highlight(self.pen)
                self.highlighted=None
                self.update()
            #return
        if not objList or objList[0].__class__!=MyCanvasRect:
            if self.highlighted:
                self.highlighted.highlight(self.pen)
                self.highlighted=None
                self.update()
            return
        i=0
        if objList[i].__class__==SelectionPoly and len(objList)>1:
            i+=1
        if objList[i].__class__==MyCanvasRect and objList[i]!=self.highlighted:
            if self.highlighted:
                self.highlighted.highlight(self.pen)
            if not self.parent.DisableHighlights:
                objList[i].highlight(self.highlightPen)
            self.highlighted=objList[i]
            self.update()

    def clearSelection(self):
        for a in self.selectionList:
            a.clearGraphics()
        self.selectionList=[]
        self.parent.updateSelection(self.selectionList)

    def addSelection(self, obj):
        vertList=[]
        ptr=obj
        while ptr:     #construct upper part of the polygon
            rect=ptr.rect()
            ptr=ptr.left
            vertList.append(QPointF(rect.left()-polyOffset,rect.top()-polyOffset))
            if ptr:
                vertList.append(QPointF(ptr.rect().left()-polyOffset, rect.top()-polyOffset))
            else:
                vertList.append(QPointF(rect.right()+3, rect.top()-polyOffset))

        tmpList=[]
        ptr=obj
        while ptr:        #construct lower part of the polygon
            rect=ptr.rect()
            ptr=ptr.right
            tmpList.append(QPointF(rect.left()-polyOffset,rect.bottom()+polyOffset))
            if ptr:
                tmpList.append(QPointF(ptr.rect().left()-polyOffset, rect.bottom()+polyOffset))
            else:
                tmpList.append(QPointF(rect.right()+3, rect.bottom()+polyOffset))
        tmpList.reverse()
        vertList.extend(tmpList)
        new=SelectionPoly(QPolygonF(vertList))
        self.addItem(new)
        new.rootCluster=obj.cluster
        new.rootGraphics=obj
        self.selectionList.append(new)
        c=float(self.parent.Brightness)/10;
        colorPalette=OWColorPalette.ColorPaletteHSV(len(self.selectionList))
        #color=[(a.red()+(255-a.red())*c, a.green()+(255-a.green())*c,
        #                 a.blue()+(255-a.blue())*c) for a in colorPalette]
        #colorPalette=[QColor(a[0],a[1],a[2]) for a in color]
        colorPalette=[colorPalette.getColor(i, 150) for i in range(len(self.selectionList))]
        for el, col in zip(self.selectionList, colorPalette):
            brush=QBrush(col,Qt.SolidPattern)
            el.setBrush(brush)
        new.setZValue(self.gZPos-2)
        #new.setZValue(2)
        ##
        new.show()
        #self.parent.updateSelection(self.selectionList)
        self.update()

    def removeSelectionItem(self, obj):
        i=self.selectionList.index(obj)
        self.selectionList[i].clearGraphics()
        self.selectionList.pop(i)

    def mousePressEvent(self, e):
        if not self.rootCluster or e.button()!=Qt.LeftButton:
            return
        pos=e.scenePos()
        if self.parent.SelectionMode:
            self.cutOffLineDragged=True
            self.setCutOffLine(pos.x())
            return
        objList=self.items(pos.x(), pos.y(), 1, 1)
        if len(objList)==0 and not self.parent.ctrlPressed:
            self.clearSelection()
            self.update()
            return
        for e in objList:
            if e.__class__==SelectionPoly:
                self.removeSelectionItem(e)
                self.parent.updateSelection(self.selectionList)
                self.update()
                return
        if objList[0].__class__==MyCanvasRect:
            inValid=[]
            for el in self.selectionList:
                if el.rootCluster.first>=objList[0].cluster.first and \
                        el.rootCluster.last<=objList[0].cluster.last:
                    inValid.append(el)
            for el in inValid:
                self.removeSelectionItem(el)
            if not self.parent.ctrlPressed:
                self.clearSelection()
            self.addSelection(objList[0])
            self.parent.updateSelection(self.selectionList)
            self.update()

    def mouseReleaseEvent(self, e):
        self.holdoff=False
        if not self.rootCluster:
            return
        if self.parent.SelectionMode and self.cutOffLineDragged:
            self.cutOffLineDragged=False
            self.bubbleRect.hide()
            self.setCutOffLine(e.scenePos().x())

    def mouseMoveEvent(self, e):
##        print "mouse move"
        if not self.rootCluster:
            return
        if self.parent.SelectionMode==1 and self.cutOffLineDragged:
            self.setCutOffLine(e.scenePos().x())
            if not self.parent.DisableBubble:
                self.bubbleRect.setText("Cut off height: \n %f" % self.cutOffHeight)
                self.bubbleRect.setPos(e.scenePos().x(), e.scenePos().y())
                self.bubbleRect.show()
            self.update()
            return
        objList=self.items(e.scenePos())
        self.highlight(objList)
        if not self.parent.DisableBubble and self.highlighted:
            cluster=self.highlighted.cluster
            text= "Items: %i \nCluster height: %f" % (len(cluster), cluster.height)
            self.bubbleRect.setText(text)
            self.bubbleRect.setPos(e.scenePos().x(),e.scenePos().y())
            self.bubbleRect.show()
            self.update()
        else:
            self.bubbleRect.hide()
            self.update()
##        print objList
        if objList and objList[0].__class__==MyCanvasText and not self.parent.DisableBubble:
            head="Items: %i" %len(objList[0].cluster)
            body=""
            if self.parent.Annotation!=0:
                bodyItems=[str(a) for a in objList[0].cluster]
                if len(bodyItems)>20:
                    bodyItems=bodyItems[:20]+["..."]
                body="\n"+"\n".join(bodyItems)
            self.bubbleRect.setText(head+body)
            self.bubbleRect.setPos(e.scenePos().x(),e.scenePos().y())
##            print head+body
            if body!="":
                self.bubbleRect.show()
            self.update()

    def cutOffSelection(self, node, height):
        if not node:
            return
        if node.cluster.height<height:
            self.addSelection(node)
        else:
            self.cutOffSelection(node.left, height)
            self.cutOffSelection(node.right, height)

    def setCutOffLine(self, x):
        if self.parent.SelectionMode==1:
            self.cutOffLinePos=x
            self.cutOffHeight=self.treeHeight- \
                self.treeHeight/self.treeAreaWidth*(x-leftMargin)
            self.cutOffLine.setPos(x,0)
            self.footer.moveMarker(x)
            self.cutOffLine.show()
            self.update()
            self.clearSelection()
            self.cutOffSelection(self.rootGraphics,self.cutOffHeight)
            self.parent.updateSelection(self.selectionList)

        else:
            self.cutOffLine.hide()
            self.cutOffLine.setPos(x,0)
            self.update()
        
class ScaleView(QGraphicsView):
    def __init__(self, parent, *args):
        apply(QGraphicsView.__init__, (self,)+args)
        self.parent=parent
        self.setFixedHeight(20)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.scene().obj=[]
        self.scene().marker=QGraphicsRectItem(None, self.scene())
        self.scene().treeAreaW = 0
        self.markerDragged=False
        self.markerPos=0

    def clear(self):
        for a in self.scene().obj:
            self.scene().removeItem(a)
        if self.scene().marker.scene() is self.scene():
            self.scene().removeItem(self.scene().marker)
        self.scene().obj = []

    def drawScale(self, treeAreaW, height):
        self.clear()
        self.scene().treeAreaW=treeAreaW
        self.scene().treeHeight=height
        xPos=leftMargin+treeAreaW
        dist=0
        distInc=math.floor(height/5)/2
        if distInc==0:
            distInc=0.25
        while xPos>=leftMargin:
            text=OWCanvasText(self.scene(), str(dist), xPos, 9, Qt.AlignCenter)
            text.setZValue(0)
            line1=OWCanvasLine(self.scene(), xPos, 0, xPos, 2)
            line2=OWCanvasLine(self.scene(), xPos, 16, xPos, 20)
            line1.setZValue(0)
            line2.setZValue(0)
            self.scene().obj.append(text)
            self.scene().obj.append(line1)
            self.scene().obj.append(line2)
            xPos-=(distInc/(height or 1))*treeAreaW
            dist+=distInc

#        self.marker=OWCanvasRectangle(self.scene(),self.markerPos - 1, 0, 1, 20, brushColor=QColor("blue"))
        self.marker = QGraphicsRectItem(QRectF(-1.0, 0.0, 1.0, 20.0), None, self.scene())
        self.marker.setPos(self.markerPos, 0)
        self.marker.setZValue(1)
#        self.scene().obj.append(self.marker)
        self.scene().marker=self.marker
#        self.scene().setSceneRect(QRectF(0.0, 0.0, leftMargin + treeAreaW, 20))

    def mousePressEvent(self, e):
        pos = self.mapToScene(e.pos())
        if pos.x()<0 and pos.x()>leftMargin+self.scene().treeAreaW:
            pos = QPointF(max(min(pos.x(), leftMargin+self.scene().treeAreaW), 0), pos.y())
        self.commitStatus=self.parent.CommitOnChange
        self.parent.CommitOnChange=False
        self.marker=self.scene().marker
        self.markerPos=pos.x()
        self.marker.setPos(self.markerPos, 0)
        self.markerDragged=True
        self.parent.dendrogram.setCutOffLine(pos.x())

    def mouseReleaseEvent(self, e):
        pos = self.mapToScene(e.pos())
        self.markerDragged=False
        self.parent.CommitOnChange=self.commitStatus
        self.parent.dendrogram.setCutOffLine(pos.x())

    def mouseMoveEvent(self, e):
        pos = self.mapToScene(e.pos())
        if pos.x()<0 or pos.x()>leftMargin+self.scene().treeAreaW:
            pos = QPointF(max(min(pos.x(), leftMargin+self.scene().treeAreaW), 0), pos.y())
        if self.markerDragged and pos:
           self.markerPos=pos.x()
           self.marker.setPos(self.markerPos,0)
           self.parent.dendrogram.setCutOffLine(pos.x())

    def moveMarker(self, x):
        self.scene().marker.setPos(x,0)
        self.markerPos = x
        
    def sceneRectUpdated(self, rect):
        rect = QRectF(rect.x(), 0, rect.width(), self.scene().sceneRect().height())
        self.scene().setSceneRect(rect)

class MyCanvasRect(QGraphicsRectItem):
    def __init__(self, *args):
        QGraphicsRectItem.__init__(self, *args)
        self.left = None
        self.right = None
        self.cluster = None
    def highlight(self, pen):
        if self.pen()==pen:
            return
        if self.left:
            self.left.highlight(pen)
        if self.right:
            self.right.highlight(pen)
        self.setPen(pen)


    def paint(self, painter, option, widget=None):
        rect=self.rect()
        painter.setPen(self.pen())
        painter.drawLine(rect.x(),rect.y(),rect.x(),rect.y()+rect.height())   
        if self.left:
            rectL=self.left.rect()
            painter.drawLine(rect.x(),rect.y(),rectL.x(),rect.y())
        else:
            painter.drawLine(rect.x(),rect.y(),rect.x()+rect.width(),rect.y())
        if self.right:
            rectR=self.right.rect()
            painter.drawLine(rect.x(),rect.y()+rect.height(),rectR.x(),
                rect.y()+rect.height())
        else:
            painter.drawLine(rect.x(),rect.y()+rect.height(),rect.x()+rect.width(),
                rect.y()+rect.height())

class MyCanvasText(OWCanvasText):
    def __init__(self, *args, **kw):
        OWCanvasText.__init__(self, *args, **kw)
        self.cluster = None

class SelectionPoly(QGraphicsPolygonItem):
    def __init__(self, *args):
        QGraphicsPolygonItem.__init__(self, *args)
        self.rootCluster=None
        self.rootGraphics=None
        self.setZValue(20)

    def clearGraphics(self):
        self.scene().removeItem(self)

class BubbleRect(QGraphicsRectItem):
    def __init__(self, *args):
        QGraphicsRectItem.__init__(self, *args)
        self.setBrush(QBrush(Qt.white))
        self.text=QGraphicsTextItem(self)
        self.text.setPos(5, 5)
        self.setZValue(30)

    def setText(self, text):
        self.text.setPlainText(text)
        self.setRect(0, 0, self.text.boundingRect().width()+6,self.text.boundingRect().height()+6)

    def setPos(self, x, y):
        if self.scene().sceneRect().contains(x+self.rect().width(),y):
            QGraphicsRectItem.setPos(self, x+5, y+5)
        else:
            QGraphicsRectItem.setPos(self, x-self.rect().width()-5, y+5)


if __name__=="__main__":
    app=QApplication(sys.argv)
    w=OWHierarchicalClustering()
    w.show()
    data=orange.ExampleTable("../../doc/datasets/iris.tab")
    id=orange.newmetaid()
    data.domain.addmeta(id, orange.FloatVariable("a"))
    data.addMetaAttribute(id)
    matrix = orange.SymMatrix(len(data))
    dist = orange.ExamplesDistanceConstructor_Euclidean(data)
    matrix = orange.SymMatrix(len(data))
    matrix.setattr('items', data)
    for i in range(len(data)):
        for j in range(i+1):
            matrix[i, j] = dist(data[i], data[j])

    w.dataset(matrix)
    app.exec_()
