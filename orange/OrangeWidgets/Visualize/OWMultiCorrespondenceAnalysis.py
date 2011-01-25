"""<name>Multi Correspondence Analysis</name>
<description>Takes a ExampleTable and runs multi correspondence analysis</description>
<icon>icons/CorrespondenceAnalysis.png</icon>
<priority>3350</priority>
<contact>Ales Erjavec (ales.erjavec(@ at @)fri.uni-lj.si</contact>
"""

import orngEnviron

from OWCorrespondenceAnalysis import *

import OWGUI
import itertools

def burtTable(data, attributes):
    """ Construct a Burt table (all values cross-tabulation) from data for attributes
    Return and ordered list of (attribute, value) pairs and a numpy.ndarray with the tabulations
    """
    values = [(attr, value) for attr in attributes for value in attr.values]
    table = numpy.zeros((len(values), len(values)))
    counts = [len(attr.values) for attr in attributes]
    offsets = [sum(counts[: i]) for i in range(len(attributes))]
    for i in range(len(attributes)):
        for j in range(i + 1):
            attr1 = attributes[i]
            attr2 = attributes[j]
            
            cm = orange.ContingencyAttrAttr(attr1, attr2, data)
            cm = numpy.array([list(row) for row in cm])
            
            range1 = range(offsets[i], offsets[i] + counts[i])
            range2 = range(offsets[j], offsets[j] + counts[j])
            start1, end1 = offsets[i], offsets[i] + counts[i]
            start2, end2 = offsets[j], offsets[j] + counts[j]
            
            table[start1: end1, start2: end2] += cm
            if i != j: #also fill the upper part
                table[start2: end2, start1: end1] += cm.T
                
    return values, table


class OWMultiCorrespondenceAnalysis(OWCorrespondenceAnalysis):
    contextHandlers = {"": DomainContextHandler("", ["xPricipalAxis", "yPrincipalAxis",
                                                     ContextField("allAttributes", DomainContextHandler.RequiredList, selected="selectedAttrs")])}
    
    settingsList = OWCorrespondenceAnalysis.settingsList + []
    def __init__(self, parent=None, signalManager=None, title="Multiple Correspondence Analysis"):
        OWCorrespondenceAnalysis.__init__(self, parent, signalManager, title)
        
        self.inputs = [("Examples", ExampleTable, self.setData)]
        self.outputs = [("Selected Examples", ExampleTable), ("Remaining Examples", ExampleTable)]
        
#        self.allAttrs = []
        self.allAttributes = []
        self.selectedAttrs = []
        
        #  GUI
        
        #  Hide the row and column attributes combo boxes
        self.colAttrCB.box.hide()
        self.rowAttrCB.box.hide()
        
        box = OWGUI.widgetBox(self.graphTab, "Attributes", addToLayout=False)
        self.graphTab.layout().insertWidget(0, box)
        self.graphTab.layout().insertSpacing(1, 4)
        
        self.attrsListBox = OWGUI.listBox(box, self, "selectedAttrs", "allAttributes", 
                                          tooltip="Attributes to include in the analysis",
                                          callback=self.runCA,
                                          selectionMode=QListWidget.ExtendedSelection,
                                          )
        
        # Find and hide the "Percent of Column Points" box
        boxes = self.graphTab.findChildren(QGroupBox)
        boxes = [box for box in boxes if str(box.title()).strip() == "Percent of Column Points"]
        if boxes:
            boxes[0].hide()
        
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
            self.allAttributes = [(attr.name, attr.varType) for attr in attrs]
            self.selectedAttrs = [0, 1, 2][:len(attrs)]
            
            self.openContext("", data)
            self.runCA()
            
    def clear(self):
        self.data = None
        self.contingency = None
        self.allAttributes = []
        self.xAxisCB.clear()
        self.yAxisCB.clear()
        self.contributionInfo.setText("NA\nNA")
        self.graph.removeDrawingCurves(True, True, True)
        self.send("Selected Examples", None)
        self.send("Remaining Examples", None)
        self.allAttrs = []
                
    def runCA(self):
        attrs = [self.allAttrs[i] for i in self.selectedAttrs]
        if not attrs:
            return
        
        self.labels, self.contingency = burtTable(self.data, attrs)
        self.error(0)
        try:
            self.CA = orngCA.CA(self.contingency)
        except numpy.linalg.LinAlgError, ex:
            self.error(0, "Could not compute the mapping! " + str(ex))
            self.graph.removeDrawingCurves(True, True, True)
            raise
        
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
        
        attrs = [self.allAttrs[i] for i in self.selectedAttrs]
        colors = dict(zip(attrs, ColorPaletteHSV(len(attrs))))
        
        rowcor = self.CA.getPrincipalRowProfilesCoordinates((self.xPrincipalAxis, self.yPrincipalAxis))
        numCor = max(int(math.ceil(len(rowcor) * float(self.percRow) / 100.0)), 2)
        indices = self.CA.PointsWithMostInertia(rowColumn=0, axis=(self.xPrincipalAxis, self.yPrincipalAxis))[:numCor]
        indices = sorted(indices)
        rowpoints = numpy.array([rowcor[i] for i in indices])
        rowlabels = [self.labels[i] for i in indices]
        
        maxx, maxy = numpy.max(rowpoints, axis=0)
        minx, miny = numpy.min(rowpoints, axis=0)
        spanx = maxx - minx or 1.0
        spany = maxy - miny or 1.0
        
        random = numpy.random.mtrand.RandomState(0)
         
        if self.jitter > 0:
            rowpoints[:,0] += random.normal(0, spanx * self.jitter / 100.0, (len(rowpoints),))
            rowpoints[:,1] += random.normal(0, spany * self.jitter / 100.0, (len(rowpoints),))
            
        # Plot the points
        groups = itertools.groupby(rowlabels, key=lambda label: label[0])
        counts = [len(attr.values) for attr in attrs]
        count = 0
        for attr, labels in groups:
            labels = list(labels)
            advance = len(labels) # TODO add shape for each attribute and colors for each value
            self.graph.addCurve(attr.name, brushColor=colors[attr],
                                penColor=colors[attr], size=self.pointSize,
                                xData=list(rowpoints[count: count+advance, 0]),
                                yData=list(rowpoints[count: count+advance, 1]),
                                autoScale=True, brushAlpha=self.alpha, enableLegend=True)
            count += advance
            
        for label, point in zip(rowlabels, rowpoints):
            self.graph.addMarker("%s: %s" % (label[0].name, label[1]), point[0], point[1], alignment=Qt.AlignCenter | Qt.AlignBottom)
            
            
            
        if self.jitter > 0:
        # Update min, max, span values again due to jittering
            maxx, maxy = numpy.max(rowpoints, axis=0)
            minx, miny = numpy.min(rowpoints, axis=0)
            spanx = maxx - minx or 1.0
            spany = maxy - miny or 1.0
        
        self.graph.setAxisScale(QwtPlot.xBottom, minx - spanx * 0.1, maxx + spanx * 0.1)
        self.graph.setAxisScale(QwtPlot.yLeft, miny - spany * 0.1, maxy + spany * 0.1)
        
        self.graph.setAxisTitle(QwtPlot.xBottom, "Axis %i" % (self.xPrincipalAxis + 1))
        self.graph.setAxisTitle(QwtPlot.yLeft, "Axis %i" % (self.yPrincipalAxis + 1))
        
        #  Store labeled points for selection
        self.rowPointsLabeled = zip(rowpoints, rowlabels) 
        
        inertia = self.CA.InertiaOfAxis(1)
        fmt = """<table><tr><td>Axis %i:</td><td>%.3f%%</td></tr>
        <tr><td>Axis %i:</td><td>%.3f%%</td></tr></table>
        """
        self.contributionInfo.setText(fmt % (self.xPrincipalAxis + 1, inertia[self.xPrincipalAxis],
                                             self.yPrincipalAxis + 1, inertia[self.yPrincipalAxis]))
        self.graph.replot()
        
    def sendData(self, *args):
        def selectedLabels(points_labels):
            return [label for (x, y), label in points_labels if self.graph.isPointSelected(x, y)]
        
        if self.contingency is not None and self.data:
            rowLabels = set(selectedLabels(self.rowPointsLabeled))
            selected = []
            remaining = []
            
            groups = itertools.groupby(sorted(rowLabels), key=lambda label: label[0])
            groups = [(attr, [value for _, value in labels]) for attr, labels in groups]
            
            def testAttr(ex, attr, values):
                if values:
                    return str(ex[attr]) in values
                else:
                    return True
                
            def testAll(ex):
                return reduce(bool.__and__, [testAttr(ex, attr, values) for attr, values in groups], bool(groups))
                
            for ex in self.data:
                if testAll(ex):
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