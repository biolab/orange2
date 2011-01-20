"""<name>Correspondence Analysis</name>
<description>Takes a ExampleTable and runs correspondence analysis</description>
<icon>icons/CorrespondenceAnalysis.png</icon>
<priority>3300</priority>
"""

from OWWidget import *
from OWGraph import *
from OWToolbars import ZoomSelectToolbar
from OWColorPalette import ColorPaletteHSV

import OWGUI
import orngCA

import math

class OWCorrespondenceAnalysis(OWWidget):
    contextHandlers = {"": DomainContextHandler("", ["colAttr", "rowAttr", "xPricipalAxis", "yPrincipalAxis"])}
    settingsList = ["pointSize", "alpha", "jitter", "showGridlines"]
    
    def __init__(self, parent=None, signalManager=None, name="Correspondence Analysis"):
        OWWidget.__init__(self, parent, signalManager, name, wantGraph=True)
        
        self.inputs = [("Examples", ExampleTable, self.setData)]
        self.outputs = [("Selected Examples", ExampleTable), ("Remaining Examples", ExampleTable)]
        
        self.colAttr = 0
        self.rowAttr = 1
        self.xPrincipalAxis = 0
        self.yPrincipalAxis = 1
        self.pointSize = 5
        self.alpha = 240
        self.jitter = 0
        self.showGridlines = 0
        self.percCol = 100
        self.percRow = 100
        self.autoSend = 0
        
        # GUI
        self.graph = OWGraph(self)
        self.graph.sendData = self.sendData
        self.mainArea.layout().addWidget(self.graph)
        
        
        self.controlAreaTab = OWGUI.tabWidget(self.controlArea)
        # Graph tab
        graphTab = OWGUI.createTabPage(self.controlAreaTab, "Graph")
        self.colAttrCB = OWGUI.comboBox(graphTab, self, "colAttr", "Column Attribute", 
                                        tooltip="Column attribute",
                                        callback=self.runCA)
        
        self.rowAttrCB = OWGUI.comboBox(graphTab, self, "rowAttr", "Row Attribute", 
                                        tooltip="Row attribute",
                                        callback=self.runCA)
        
        self.xAxisCB = OWGUI.comboBox(graphTab, self, "xPrincipalAxis", "Principal Axis X",
                                      tooltip="Principal axis X",
                                      callback=self.updateGraph)
        
        self.yAxisCB = OWGUI.comboBox(graphTab, self, "yPrincipalAxis", "Principal Axis Y",
                                      tooltip="Principal axis Y",
                                      callback=self.updateGraph)
        
        box = OWGUI.widgetBox(graphTab, "Contribution to Inertia")
        self.contributionInfo = OWGUI.widgetLabel(box, "NA\nNA")
        
        OWGUI.hSlider(graphTab, self, "percCol", "Percent of Column Points", 1, 100, 1,
                      callback=self.updateGraph,
                      tooltip="The percent of column points with the largest contribution to inertia")
        
        OWGUI.hSlider(graphTab, self, "percRow", "Percent of Row Points", 1, 100, 1,
                      callback=self.updateGraph,
                      tooltip="The percent of row points with the largest contribution to inertia")
        
        self.zoomSelect = ZoomSelectToolbar(self, graphTab, self.graph, self.autoSend)
        OWGUI.rubber(graphTab)
        
        # Settings tab
        settingsTab = OWGUI.createTabPage(self.controlAreaTab, "Settings")
        OWGUI.hSlider(settingsTab, self, "pointSize", "Point Size", 3, 20, step=1,
                      callback=self.setPointSize)
        
        OWGUI.hSlider(settingsTab, self, "alpha", "Transparancy", 1, 255, step=1,
                      callback=self.updateAlpha)
        
        OWGUI.hSlider(settingsTab, self, "jitter", "Jitter Points", 0, 20, step=1,
                      callback=self.updateGraph)
        
        box = OWGUI.widgetBox(settingsTab, "General Settings")
        OWGUI.checkBox(box, self, "showGridlines", "Show gridlines",
                       tooltip="Show gridlines in the plot.",
                       callback=self.updateGridlines)
        OWGUI.rubber(settingsTab)
        
        self.connect(self.graphButton, SIGNAL("clicked()"), self.graph.saveToFile)
        
        self.contingency = None
        self.contColAttr = None
        self.contRowAttr = None
        
        self.resize(800, 600)
        
    def setData(self, data=None):
        self.closeContext("")
        self.clear()
        self.data = data
        self.warning([0])
        if data is not None:
            attrs = data.domain.variables + data.domain.getmetas().values()
            attrs = [attr for attr in attrs if isinstance(attr, orange.EnumVariable)]
            if not attrs:
                self.warning(0, "Data has no discrete variables!")
                self.clear()
                return
            self.allAttrs = attrs
            self.colAttrCB.clear()
            self.rowAttrCB.clear()
            icons = OWGUI.getAttributeIcons()
            for attr in attrs:
                self.colAttrCB.addItem(QIcon(icons[attr.varType]), attr.name)
                self.rowAttrCB.addItem(QIcon(icons[attr.varType]), attr.name)
                
            self.colAttr = max(min(len(attrs) - 1, self.colAttr), 0)
            self.rowAttr = max(min(len(attrs) - 1, self.rowAttr), min(1, len(attrs) - 1))
            
            self.openContext("", data)
            self.runCA()
            
    def clear(self):
        self.data = None
        self.colAttrCB.clear()
        self.rowAttrCB.clear()
        self.xAxisCB.clear()
        self.yAxisCB.clear()
        self.contributionInfo.setText("NA\nNA")
        self.graph.removeDrawingCurves(True, True, True)
        self.send("Selected Examples", None)
        self.send("Remaining Examples", None)
        self.allAttrs = []
        
    def runCA(self):
        self.contColAttr = colAttr = self.allAttrs[self.colAttr]
        self.contRowAttr = rowAttr = self.allAttrs[self.rowAttr]
        self.contingency = orange.ContingencyAttrAttr(rowAttr, colAttr, self.data)
        self.error(0)
        try:
            self.CA = orngCA.CA([[c for c in row] for row in self.contingency])
        except numpy.linalg.LinAlgError:
            self.error(0, "Could not compute.")
            self.graph.removeDrawingCurves(True, True, True)
            raise
            
        self.rowItems = [s for s, v in self.contingency.outerDistribution.items()]
        self.colItems = [s for s, v in self.contingency.innerDistribution.items()]
        
        self.xAxisCB.clear()
        self.yAxisCB.clear()
        
        self.axisCount = min(self.CA.D.shape)
        self.xAxisCB.addItems([str(i + 1) for i in range(self.axisCount)])
        self.yAxisCB.addItems([str(i + 1) for i in range(self.axisCount)])
        
        self.xPrincipalAxis = min(self.xPrincipalAxis, self.axisCount - 1)
        self.yPrincipalAxis = min(self.yPrincipalAxis, self.axisCount - 1)
        
        self.updateGraph()
        
    def updateGraph(self): 
        self.graph.removeAllSelections()
        self.graph.removeDrawingCurves(True, True, True)
        
        colors = ColorPaletteHSV(2)
        
        rowcor = self.CA.getPrincipalRowProfilesCoordinates((self.xPrincipalAxis, self.yPrincipalAxis))
        numCor = int(math.ceil(len(rowcor) * float(self.percRow) / 100.0))
        indices = self.CA.PointsWithMostInertia(rowColumn=0, axis=(self.xPrincipalAxis, self.yPrincipalAxis))[:numCor]
        rowpoints = numpy.array([rowcor[i] for i in indices])
        rowlabels = [self.rowItems[i] for i in indices]
            
        
        colcor = self.CA.getPrincipalColProfilesCoordinates((self.xPrincipalAxis, self.yPrincipalAxis))
        numRow = int(math.ceil(len(colcor) * float(self.percCol) / 100.0))
        indices = self.CA.PointsWithMostInertia(rowColumn=1, axis=(self.xPrincipalAxis, self.yPrincipalAxis))[:numRow]
        colpoints = numpy.array([colcor[i] for i in indices])
        collabels = [self.colItems[i] for i in indices]
        
        vstack = ((rowpoints,) if rowpoints.size else ()) + \
                 ((colpoints,) if colpoints.size else ())
        allpoints = numpy.vstack(vstack)
        maxx, maxy = numpy.max(allpoints, axis=0)
        minx, miny = numpy.min(allpoints, axis=0)
        spanx = maxx - minx
        spany = maxy - miny
        
        random = numpy.random.mtrand.RandomState(0)
         
        if self.jitter > 0:
            rowpoints[:,0] += random.normal(0, spanx * self.jitter / 100.0, (len(rowpoints),))
            rowpoints[:,1] += random.normal(0, spany * self.jitter / 100.0, (len(rowpoints),))
            
            colpoints[:,0] += random.normal(0, spanx * self.jitter / 100.0, (len(colpoints),))
            colpoints[:,1] += random.normal(0, spany * self.jitter / 100.0, (len(colpoints),))
            
        # Plot the points
        self.graph.addCurve("Row points", brushColor=colors[0],
                            penColor=colors[0], size=self.pointSize,
                            enableLegend=True, xData=rowpoints[:, 0], yData=rowpoints[:, 1],
                            autoScale=True, brushAlpha=self.alpha)
        
        for label, point in zip(rowlabels, rowpoints):
            self.graph.addMarker(label, point[0], point[1], alignment=Qt.AlignCenter | Qt.AlignBottom)
            
        self.graph.addCurve("Column points", brushColor=colors[1],
                            penColor=colors[1], size=self.pointSize,
                            enableLegend=True, xData=colpoints[:, 0], yData=colpoints[:, 1],
                            autoScale=True, brushAlpha=self.alpha)
        
        for label, point in zip(collabels, colpoints):
            self.graph.addMarker(label, point[0], point[1], alignment=Qt.AlignCenter | Qt.AlignBottom)
            
        if self.jitter > 0:
        # Update min, max, span values again due to jittering
            vstack = ((rowpoints,) if rowpoints.size else ()) + \
                     ((colpoints,) if colpoints.size else ())
            allpoints = numpy.vstack(vstack)
            maxx, maxy = numpy.max(allpoints, axis=0)
            minx, miny = numpy.min(allpoints, axis=0)
        
        self.graph.setAxisScale(QwtPlot.xBottom, minx - spanx * 0.05, maxx + spanx * 0.05)
        self.graph.setAxisScale(QwtPlot.yLeft, miny - spany * 0.05, maxy + spany * 0.05)
        
        self.graph.setAxisTitle(QwtPlot.xBottom, "Axis %i" % (self.xPrincipalAxis + 1))
        self.graph.setAxisTitle(QwtPlot.yLeft, "Axis %i" % (self.yPrincipalAxis + 1))
        
        #  Store labeled points for selection 
        self.colPointsLabeled = zip(colpoints, collabels)
        self.rowPointsLabeled = zip(rowpoints, rowlabels) 
        
        inertia = self.CA.InertiaOfAxis(1)
        fmt = """<table><tr><td>Axis %i:</td><td>%.3f%%</td></tr>
        <tr><td>Axis %i:</td><td>%.3f%%</td></tr></table>
        """
        self.contributionInfo.setText(fmt % (self.xPrincipalAxis + 1, inertia[self.xPrincipalAxis],
                                             self.yPrincipalAxis + 1, inertia[self.yPrincipalAxis]))
        self.graph.replot()
        
    def setPointSize(self):
        for curve in self.graph.itemList():
            if isinstance(curve, QwtPlotCurve):
                symbol = curve.symbol()
                symbol.setSize(self.pointSize)
                if QWT_VERSION_STR >= "5.2":
                    curve.setSymbol(symbol)
        self.graph.replot()
    
    def updateAlpha(self):
        for curve in self.graph.itemList():
            if isinstance(curve, QwtPlotCurve):
                brushColor = curve.symbol().brush().color()
                penColor = curve.symbol().pen().color()
                brushColor.setAlpha(self.alpha)
                brush = QBrush(curve.symbol().brush())
                brush.setColor(brushColor)
                penColor.setAlpha(self.alpha)
                symbol = curve.symbol()
                symbol.setBrush(brush)
                symbol.setPen(QPen(penColor))
                if QWT_VERSION_STR >= "5.2":
                    curve.setSymbol(symbol)
        self.graph.replot()
        
    def updateGridlines(self):
        self.graph.enableGridXB(self.showGridlines)
        self.graph.enableGridYL(self.showGridlines)
        
    def sendData(self, *args):
        def selectedLabels(points_labels):
            return [label for (x, y), label in points_labels if self.graph.isPointSelected(x, y)]
        
        if self.contingency and self.data:
            colLabels = set(selectedLabels(self.colPointsLabeled))
            rowLabels = set(selectedLabels(self.rowPointsLabeled))
            colAttr = self.allAttrs[self.colAttr]
            rowAttr = self.allAttrs[self.rowAttr]
            selected = []
            remaining = []
            
            if colLabels and rowLabels:
                def test(ex):
                    return str(ex[colAttr]) in colLabels and str(ex[rowAttr]) in rowLabels
            elif colLabels or rowLabels:
                def test(ex):
                    return str(ex[colAttr]) in colLabels or str(ex[rowAttr]) in rowLabels
            else:
                def test(ex):
                    return False
                
            for ex in self.data:
                if test(ex):
                    selected.append(ex)
                else:
                    remaining.append(ex)
                 
            selected = orange.ExampleTable(self.data.domain, selected) if selected else \
                            orange.ExampleTable(self.data.domain)
            
            remaining = orange.ExampleTable(self.data.domain, remaining) if remaining else \
                            orange.ExampleTable(self.data.domain)
                        
            self.send("Selected Examples", selected)
            self.send("Remaining Examples", remaining)
        else:
            self.send("Selected Examples", None)
            self.send("Remaining Examples", None)

    
if __name__ == "__main__":
    app = QApplication([])
    w = OWCorrespondenceAnalysis()
    data = orange.ExampleTable("../doc/datasets/adult-sample.tab")
    w.setData(data)
    w.show()
    app.exec_()
    w.saveSettings()
        
    
        
        