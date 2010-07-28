"""<name>Compare Examples</name>
<description>Compares examples, considering the attributes as distributions</description>
<icon>icons/CompareExamples.png</icon>
<priority>30</priority>
<contact>Janez Demsar (janez.demsar@fri.uni-lj.si)</contact>"""

from OWWidget import *
from OWGUI import *
from OWDlgs import OWChooseImageSizeDlg
from PyQt4.QtGui import QGraphicsEllipseItem
import OWQCanvasFuncts, OWColorPalette, math
#import MapLayer

class OWCompare(OWWidget):
    # We cannot put attribute selection to context settings; see comment at func settingsFromWidget
    contextHandlers = {"": PerfectDomainContextHandler("", [
                                                       ("attrLabel", DomainContextHandler.Optional + DomainContextHandler.IncludeMetaAttributes),
                                                       ("sortingOrder", DomainContextHandler.Optional + DomainContextHandler.IncludeMetaAttributes)])}
    settingsList = ["normalize", "colorSettings", "selectedSchemaIndex"]

    def __init__(self,parent=None, signalManager = None):
        OWWidget.__init__(self, parent, signalManager, "Compare Examples")
        self.inputs = [("Examples", ExampleTable, self.setData, Default)]
        self.outputs = [("Bars", list, Default)]
        self.icons = self.createAttributeIconDict()

        self.examples = None
        self.attributes = []
        self.selectedAttributes = []
        self.normalize = 0
        self.colorSettings = None
        self.selectedSchemaIndex = 0
        self.attrLabel = "(No labels)"
        self.sortingOrder = "(Original order)"
        self.barHeight, self.barWidth = 300, 42
        self.pieHeight, self.pieWidth = 150, 150
        self.resize(900, 550)
        self.loadSettings()

        dlg = self.createColorDialog()
        self.discPalette = dlg.getDiscretePalette("discPalette")

        OWGUI.listBox(self.controlArea, self, "selectedAttributes", labels="attributes", box="Attributes", selectionMode=QListWidget.ExtendedSelection, callback=self.updateDisplay)
        self.attrLabelCombo = OWGUI.comboBox(self.controlArea, self, "attrLabel", box="Label", callback=self.updateDisplay, sendSelectedValue=1, valueType=str, emptyString="(No labels)")
        self.sortingCombo = OWGUI.comboBox(self.controlArea, self, "sortingOrder", box="Sorting", callback=self.updateDisplay, sendSelectedValue=1, valueType=str, emptyString="(Original order)")
        OWGUI.checkBox(self.controlArea, self, "normalize", "Normalize to equal height", box="Settings", callback=self.updateDisplay)
        b1 = OWGUI.widgetBox(self.controlArea, "Colors", orientation = "horizontal")
        OWGUI.button(b1, self, "Set Colors", self.setColors, debuggingEnabled = 0)
        
#        self.btnSendbars = OWGUI.button(self.controlArea, self, "Send Bars", self.sendBars)
#        self.btnSendpie = OWGUI.button(self.controlArea, self, "Send Pie Charts", self.sendPies)
        
        self.canvas = QGraphicsScene(self)
        self.canvasview = QGraphicsView(self.canvas, self.mainArea)
        self.canvasview.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.mainArea.layout().addWidget( self.canvasview )

    def createColorDialog(self):
        dlg = OWColorPalette.ColorPaletteDlg(self, "Color Palette")
        dlg.createDiscretePalette("discPalette", "Discrete Palette")
        dlg.setColorSchemas(self.colorSettings, self.selectedSchemaIndex)
        return dlg
    
    def setColors(self):
        dlg = self.createColorDialog()
        if dlg.exec_():
            self.colorSettings = dlg.getColorSchemas()
            self.selectedSchemaIndex = dlg.selectedSchemaIndex
            self.discPalette = dlg.getDiscretePalette("discPalette")
            self.updateDisplay()

    def constructBar(self, distrib, numbs):
        bar = QGraphicsItemGroup()
        totHeight = 0
        for j, num, dist in reversed(zip(range(len(numbs)), numbs, distrib)):
            rect = QGraphicsRectItem(-self.barWidth/2, totHeight-dist, self.barWidth, dist)
            rect.setBrush(QBrush(self.discPalette[j]))
            rect.setPen(QPen(QColor(255, 255, 255), 1, Qt.SolidLine))
            rect.setToolTip("%s: %.3f" % (self.attributes[self.selectedAttributes[j]][0], num) + (" (%2.1f%%)" % (dist/self.barHeight*100) if self.normalize else ""))
            bar.addToGroup(rect)
            totHeight -= dist
        return bar

    def constructPieChart(self, distrib, numbs):
        pie = QGraphicsItemGroup()
        totArc = 0
        for j, num, dist in reversed(zip(range(len(numbs)), numbs, distrib)):
            arc = QGraphicsEllipseItem(-self.pieWidth/2, -self.pieWidth/2, self.pieWidth, self.pieHeight)
            arc.setPen(QPen(QColor(0, 0, 0), 1, Qt.SolidLine))
            arc.setBrush(QBrush(self.discPalette[j]))
            arc.setToolTip("%s: %.3f" % (self.attributes[self.selectedAttributes[j]][0], num) + (" (%2.1f%%)" % (dist/self.barHeight*100) if self.normalize else ""))
            arc.setStartAngle(totArc)
            arc.setSpanAngle(dist * 2880 / self.barHeight)
            pie.addToGroup(arc)
            totArc += dist * 2880 / self.barHeight
        return pie

    def getDistsNums(self, forceNormal=False):
            nums = [[ex[i] for i in self.selectedAttributes] for ex in self.examples]
            sums = [sum(n) for n in nums]
            norms = [i or 1 for i in sums] if forceNormal or self.normalize else [max(sums) or 1]*len(sums) 
            dists = [[self.barHeight*x/norm for x in num] for num, norm in zip(nums, norms)]
            return dists, nums, sums
        
    def updateDisplay(self):
        self.canvas.clear()
        if self.examples and len(self.selectedAttributes):
            width, height = self.barWidth, self.barHeight
            dists, nums, sums = self.getDistsNums() 

            order = range(len(self.examples))
            sortingOrder = self.sortingOrder
            if sortingOrder and sortingOrder != "(Original order)":
                if sortingOrder=="Total":
                    order.sort(lambda i, j: -cmp(sums[i], sums[j]))
                else:
                    order.sort(lambda i, j: -cmp(self.origExamples[i][sortingOrder], self.origExamples[j][sortingOrder]))
           
            for xpos, col in enumerate(order):
                bar = self.constructBar(dists[col], nums[col])
                self.canvas.addItem(bar)
                bar.setPos(self.barWidth*(xpos*2+1.5), 50)
                if self.attrLabel != "(No labels)":
                    OWQCanvasFuncts.OWCanvasText(self.canvas, str(self.origExamples[col][self.attrLabel]).decode("utf-8"), width*(xpos*2+1.5), 60, alignment=Qt.AlignTop|Qt.AlignHCenter)
            l = OWQCanvasFuncts.OWCanvasLine(self.canvas, width*.8, 51, width*(len(nums)*2+0.2), 50)
            l.setZValue(1)
            if self.normalize:
                l = OWQCanvasFuncts.OWCanvasLine(self.canvas, width*.8, 49 - height, width*(len(nums)*2+0.2), 49 - height)
                l.setZValue(1)
    
            for i, attr in enumerate(self.selectedAttributes):
                OWQCanvasFuncts.OWCanvasRectangle(self.canvas, width*(xpos*2+3.5), 50+i*20 - height, 10, 10, pen=QPen(QColor(255, 255, 255), 1, Qt.SolidLine), brushColor=self.discPalette[i])
                OWQCanvasFuncts.OWCanvasText(self.canvas, self.attributes[attr][0].decode("utf-8"), width*(xpos*2+3.5)+16, 46+i*20 - height)

#    def sendCharts(self, constructor, forceNormal=False):
#        l = []
#        if self.examples and len(self.selectedAttributes):
#            dists, nums, sums = self.getDistsNums(forceNormal)
#            addLabel = self.attrLabel != _("(No labels)")
#            l = [(constructor(dist, num),
#                  str(ex[self.attrLabel]).decode("utf-8") if addLabel else "",
#                  float(ex["latitude"]), float(ex["longitude"])
#                 ) for dist, num, ex in zip(dists, nums, self.origExamples) 
#                   if not (ex["latitude"].isSpecial() or ex["longitude"].isSpecial())]
#            
#            legend = []
#            for j in range(len(self.selectedAttributes)):
#                legend.append((self.discPalette[j], self.attributes[self.selectedAttributes[j]][0]))
#            self.send("Bars", MapLayer.MapLayer(l, legend, sums))
#
#    def sendBars(self):
#        self.sendCharts(self.constructBar)
#    
#    def sendPies(self):
#        self.sendCharts(self.constructPieChart, True)
#    
    # We cannot rely on domain-based context settings since not all attributes from the domain are available
    # (the widget omits those with unknown values)
    def settingsFromWidgetCallback(self, handler, context):
        context.selectedAttributes = [self.attributes[i] for i in self.selectedAttributes]

    def settingsToWidgetCallback(self, handler, context):
        if hasattr(context, "selectedAttributes"):
            self.selectedAttributes = [i for i in range(len(self.attributes)) if self.attributes[i] in context.selectedAttributes]

    def findAttributeSubset(self):
        ex = self.examples[0]
        for i in range(len(ex)):
            s = 0
            for j, e in enumerate(ex[i:]):
                s += e
                if s > 1.01:
                    break
                if s >= 0.99:
                    for ex2 in self.examples:
                        if not 0.99 < sum(ex2[i:j+i+1]) < 1.01:
                            break
                    else:
                        self.selectedAttributes = range(i, j+i+1)
                        return
        
    def setData(self, data):
        self.closeContext()
        self.attrLabelCombo.clear()
        self.sortingCombo.clear()
        if data==None:
            self.attributes = []
            self.examples = None
            self.origExamples = None
#            self.btnSendbars.setDisabled(True)
#            self.btnSendpie.setDisabled(True)
        else:
            contIndices = [i for i, attr in enumerate(data.domain.attributes) if attr.varType == orange.Variable.Continuous]
            usedIndices = [i for i in contIndices if not any(d[i].isSpecial() for d in data)]
            
            self.origExamples = data
            self.examples = [[ex[i] for i in usedIndices] for ex in self.origExamples]
            self.attributes = [(data.domain.attributes[i].name, orange.Variable.Continuous) for i in usedIndices]
            
            self.attrLabelCombo.addItem("(No labels)")
            self.sortingCombo.addItem("(Original order)")
            self.sortingCombo.addItem("Total")
            hasName = None
            for metavar in [data.domain.getmeta(mykey) for mykey in data.domain.getmetas().keys()]:
                self.attrLabelCombo.addItem(self.icons[metavar.varType], metavar.name.decode("utf-8"))
                self.sortingCombo.addItem(self.icons[metavar.varType], metavar.name.decode("utf-8"))
                if metavar.name.lower() == "name":
                    hasName = metavar.name 
            for attr in data.domain:
                self.attrLabelCombo.addItem(self.icons[attr.varType], attr.name.decode("utf-8"))
                self.sortingCombo.addItem(self.icons[attr.varType], attr.name.decode("utf-8"))
                if attr.name.lower() == "name":
                    hasName = attr.name
            if hasName:
                self.attrLabel = hasName

#            try:
#                data[0]["latitude"], data[0]["longitude"]
#                self.btnSendbars.setDisabled(False)
#                self.btnSendpie.setDisabled(False)
#            except:
#                self.btnSendbars.setDisabled(True)
#                self.btnSendpie.setDisabled(True)
                 
        self.openContext("", data)
        if self.attributes and len(self.examples) and not self.selectedAttributes:
            self.findAttributeSubset()
            
        self.updateDisplay()

    def sendReport(self):
        self.startReport("%s" % (self.windowTitle()))
        self.reportImage(lambda *x: OWChooseImageSizeDlg(self.canvas).saveImage(*x))
