from OWGraph import *
from orngScaleData import *

DONT_SHOW_TOOLTIPS = 0
VISIBLE_ATTRIBUTES = 1
ALL_ATTRIBUTES = 2


class OWSurveyPlotGraph(OWGraph, orngScaleData):
    def __init__(self, parent = None, name = None):
        "Constructs the graph"
        OWGraph.__init__(self, parent, name)
        orngScaleData.__init__(self)
        self.selectedRectangle = None
        self.exampleTracking = 1
        self.length = 0 # number of shown attributes - we need it also in mouse movement
        self.enabledLegend = 0
        self.tooltipKind = 1
        self.attrLabels = []

    def setData(self, data, **args):
        OWGraph.setData(self, data)
        orngScaleData.setData(self, data, **args)

    #
    # update shown data. Set labels, coloring by className ....
    def updateData(self, labels):
        self.clear()
        self.selectedRectangle = None
        self.tips.removeAll()

        self.attrLabels = labels
        self.length = len(labels)
        indices = [self.attributeNameIndex[label] for label in labels]

        if self.noJitteringScaledData == None or len(self.noJitteringScaledData) == 0 or len(labels) == 0:
            self.setAxisScaleDraw(QwtPlot.xBottom, DiscreteAxisScaleDraw(labels))
            self.setAxisScale(QwtPlot.yLeft, 0, 1, 1)
            return

        validData = self.getValidList(indices)
        totalValid = sum(validData)

        self.setAxisScale(QwtPlot.yLeft, 0, totalValid, totalValid)
        self.setAxisScale(QwtPlot.xBottom, -0.5, len(labels)-0.5, 1)
        self.setAxisScaleDraw(QwtPlot.xBottom, DiscreteAxisScaleDraw(labels))
        self.axisScaleDraw(QwtPlot.xBottom).enableComponent(QwtScaleDraw.Backbone, 0)
        self.axisScaleDraw(QwtPlot.xBottom).enableComponent(QwtScaleDraw.Ticks, 0)

        # draw vertical lines that represent attributes
        for i in range(len(labels)):
            self.addCurve("", style = QwtPlotCurve.Lines, symbol = QwtSymbol.NoSymbol, xData = [i,i], yData = [0, 1])

        xRectsToAdd = {}
        yRectsToAdd = {}
        classNameIndex = -1
        if self.rawData.domain.classVar:
            classNameIndex = self.attributeNameIndex[self.rawData.domain.classVar.name]
            if self.rawData.domain.classVar.varType == orange.VarTypes.Discrete:
                classValDict = getVariableValueIndices(self.dataDomain.classVar)
                #self.discPalette.setNumberOfColors(len(classValDict.keys()))

        y = 0
        for i in range(len(self.rawData)):
            if validData[i] == 0: continue
            if classNameIndex == -1: newColor = (0,0,0)
            elif self.rawData.domain.classVar.varType == orange.VarTypes.Discrete: newColor = self.discPalette.getRGB(classValDict[self.rawData[i].getclass().value])
            else: newColor = self.contPalette.getRGB(self.noJitteringScaledData[classNameIndex][i])

            for j in range(self.length):
                width = self.noJitteringScaledData[indices[j]][i] * 0.45
                if not xRectsToAdd.has_key(newColor):
                    xRectsToAdd[newColor] = []
                    yRectsToAdd[newColor] = []
                xRectsToAdd[newColor].extend([j-width, j+width, j+width, j-width])
                yRectsToAdd[newColor].extend([y, y, y+1, y+1])
            y += 1

        for key in xRectsToAdd.keys():
            RectangleCurve(QPen(QColor(*key)), QBrush(QColor(*key)), xRectsToAdd[key], yRectsToAdd[key]).attach(self)

        if self.enabledLegend and self.rawData.domain.classVar and self.rawData.domain.classVar.varType == orange.VarTypes.Discrete:
            classValues = getVariableValuesSorted(self.dataDomain.classVar)
            self.addCurve("<b>" + self.rawData.domain.classVar.name + ":</b>", QColor(0,0,0), QColor(0,0,0), 0, symbol = QwtSymbol.NoSymbol, enableLegend = 1)
            for ind in range(len(classValues)):
                self.addCurve(classValues[ind], self.discPalette[ind], self.discPalette[ind], 15, symbol = QwtSymbol.Rect, enableLegend = 1)
        self.replot()


    # show rectangle with example shown under mouse cursor
    def mouseMoveEvent(self, e):
        if self.selectedRectangle:
            self.selectedRectangle.detach()
            self.selectedRectangle = None

        if self.mouseCurrentlyPressed:
            OWGraph.mouseMoveEvent(self, e)
        elif not self.rawData:
            return
        else:
            canvasPos = self.canvas().mapFrom(self, e.pos())
            yFloat = math.floor(self.invTransform(QwtPlot.yLeft, canvasPos.y()))
            if self.exampleTracking:
                width = 0.49
                xData = [-width, self.length+width-1, self.length+width-1, -width, -width]
                yData = [yFloat, yFloat, yFloat+1, yFloat+1, yFloat]
                self.selectedRectangle = self.addCurve("", style=QwtPlotCurve.Lines, symbol=QwtSymbol.NoSymbol, xData=xData, yData=yData)
                self.replot()
            else:
                OWGraph.mouseMoveEvent(self, e)

            if (self.tooltipKind == VISIBLE_ATTRIBUTES and self.attrLabels != []) or self.tooltipKind == ALL_ATTRIBUTES:
                if int(yFloat) >= len(self.rawData): return
                if self.tooltipKind == VISIBLE_ATTRIBUTES:      text = self.getExampleTooltipText(self.rawData[int(yFloat)], self.attrLabels)
                else:                                           text = self.getExampleTooltipText(self.rawData[int(yFloat)], [])
                OWGraph.mouseMoveEvent(self, e)
                self.showTip(e.x(), e.y(), text)

if __name__== "__main__":
    #Draw a simple graph
    a = QApplication(sys.argv)
    c = OWSurveyPlotGraph()

    a.setMainWidget(c)
    c.show()
    a.exec_()
