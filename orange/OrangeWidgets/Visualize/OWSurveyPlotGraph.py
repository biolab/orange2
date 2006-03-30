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
        self.selectedRectangle = 0
        self.exampleTracking = 1
        self.length = 0 # number of shown attributes - we need it also in mouse movement
        self.enabledLegend = 0
        self.tooltipKind = 1
        self.yDataIndices = []  # array of indices that show the index in self.rawdata - if there are no missing values then array[i] = i
        self.attrLabels = []

    def setData(self, data):
        OWGraph.setData(self, data)
        orngScaleData.setData(self, data)
        
    #
    # update shown data. Set labels, coloring by className ....
    #
    def updateData(self, labels):
        self.removeCurves()
        self.tips.removeAll()

        self.attrLabels = labels        
        self.length = len(labels)
        indices = [self.attributeNameIndex[label] for label in labels]
        #if self.tooltipKind == DONT_SHOW_TOOLTIPS: MyQToolTip.tip(self.tooltip, QRect(0,0,0,0), "")


        if not self.noJitteringScaledData or len(self.noJitteringScaledData) == 0 or len(labels) == 0:
            self.setAxisScaleDraw(QwtPlot.xBottom, DiscreteAxisScaleDraw(labels))
            self.setAxisScale(QwtPlot.yLeft, 0, 1, 1)
            return

        validData = self.getValidList(indices)
        totalValid = sum(validData)

        self.setAxisScale(QwtPlot.yLeft, 0, totalValid, totalValid)
        self.setAxisScale(QwtPlot.xBottom, -0.5, len(labels)-0.5, 1)
        #self.setAxisMaxMajor(QwtPlot.xBottom, len(labels)-1.0)        
        #self.setAxisMaxMinor(QwtPlot.xBottom, 0)
        self.setAxisScaleDraw(QwtPlot.xBottom, DiscreteAxisScaleDraw(labels))
        self.axisScaleDraw(QwtPlot.xBottom).setTickLength(0, 0, 0)  # hide ticks
        self.axisScaleDraw(QwtPlot.xBottom).setOptions(0)           # hide horizontal line representing x axis
        #self.setAxisScale(QwtPlot.yLeft, 0, 1, 1)
        
        # draw vertical lines that represent attributes
        for i in range(len(labels)):
            newCurveKey = self.insertCurve(labels[i])
            self.setCurveData(newCurveKey, [i,i], [0,1])

        self.repaint()  # we have to repaint to update scale to get right coordinates for tooltip rectangles
        self.updateLayout()

        classNameIndex = -1
        if self.rawdata.domain.classVar:
            classNameIndex = self.attributeNameIndex[self.rawdata.domain.classVar.name]
            if self.rawdata.domain.classVar.varType == orange.VarTypes.Discrete:
                classValDict = getVariableValueIndices(self.rawdata, self.rawdata.domain.classVar)
                colors = ColorPaletteBrewer(len(classValDict))
            else:
                colors = ColorPaletteHSV()
        
        y = 0
        self.yDataIndices = []
        
        for i in range(len(self.rawdata)):
            if validData[i] == 0: continue
            
            curve = subBarQwtPlotCurve(self)
            
            if classNameIndex == -1: newColor = QColor(0,0,0)
            elif self.rawdata.domain.classVar.varType == orange.VarTypes.Discrete: newColor = colors[classValDict[self.rawdata[i].getclass().value]]
            else: newColor = colors[self.noJitteringScaledData[classNameIndex][i]]
                
            curve.color = newColor
            curve.penColor = newColor
            xData = []; yData = []
            for j in range(self.length):
                width = self.noJitteringScaledData[indices[j]][i] * 0.45
                xData += [j-width, j+width]
                yData += [y, y+1]

            ##########
            y += 1
            self.yDataIndices.append(y)

            ckey = self.insertCurve(curve)
            self.setCurveStyle(ckey, QwtCurve.UserCurve)
            self.setCurveData(ckey, xData, yData)

        if self.enabledLegend and self.rawdata.domain.classVar and self.rawdata.domain.classVar.varType == orange.VarTypes.Discrete:
            classValues = getVariableValuesSorted(self.rawdata, self.rawdata.domain.classVar.name)
            self.addCurve("<b>" + self.rawdata.domain.classVar.name + ":</b>", QColor(0,0,0), QColor(0,0,0), 0, symbol = QwtSymbol.None, enableLegend = 1)
            for ind in range(len(classValues)):
                self.addCurve(classValues[ind], colors[ind], colors[ind], 15, symbol = QwtSymbol.Rect, enableLegend = 1)
           

    # show rectangle with example shown under mouse cursor
    def onMouseMoved(self, e):
        self.hideSelectedRectangle()
        if self.mouseCurrentlyPressed:
            OWGraph.onMouseMoved(self, e)
        elif not self.rawdata:
            return
        else:
            yFloat = math.floor(self.invTransform(QwtPlot.yLeft, e.y()))
            if self.exampleTracking:
                width = 0.49
                xData = [-width, self.length+width-1, self.length+width-1, -width, -width]
                yData = [yFloat, yFloat, yFloat+1, yFloat+1, yFloat]
                self.selectedRectangle = self.insertCurve("test")
                self.setCurveData(self.selectedRectangle, xData, yData)
                self.setCurveStyle(self.selectedRectangle, QwtCurve.Lines)
                self.replot()
            else:
                OWGraph.onMouseMoved(self, e)

            if (self.tooltipKind == VISIBLE_ATTRIBUTES and self.attrLabels != []) or self.tooltipKind == ALL_ATTRIBUTES:
                if int(yFloat) >= len(self.rawdata): return
                if self.tooltipKind == VISIBLE_ATTRIBUTES:      text = self.getExampleTooltipText(self.rawdata, self.rawdata[int(yFloat)], self.attrLabels)
                else:                                           text = self.getExampleTooltipText(self.rawdata, self.rawdata[int(yFloat)], [])
                y1Int = self.transform(QwtPlot.yLeft, yFloat)
                y2Int = self.transform(QwtPlot.yLeft, yFloat+1.0)
                MyQToolTip.tip(self.tooltip, QRect(e.x()+self.canvas().frameGeometry().x()-10, y2Int+self.canvas().frameGeometry().y(), 20, y1Int-y2Int), text)
                OWGraph.onMouseMoved(self, e)


    def hideSelectedRectangle(self):
        if self.selectedRectangle != 0:
            self.removeCurve(self.selectedRectangle)
            self.selectedRectangle = 0


if __name__== "__main__":
    #Draw a simple graph
    a = QApplication(sys.argv)        
    c = OWSurveyPlotGraph()
            
    a.setMainWidget(c)
    c.show()
    a.exec_loop()
