#
# OWParallelGraph.py
#
# the base for all parallel graphs

from OWVisGraph import *

DONT_SHOW_TOOLTIPS = 0
VISIBLE_ATTRIBUTES = 1
ALL_ATTRIBUTES = 2


class OWSurveyPlotGraph(OWVisGraph):
    def __init__(self, parent = None, name = None):
        "Constructs the graph"
        OWVisGraph.__init__(self, parent, name)
        self.jitteringType = "none"
        self.selectedRectangle = 0
        self.exampleTracking = 1
        self.length = 0 # number of shown attributes - we need it also in mouse movement
        self.enabledLegend = 0
        self.tooltipKind = 1
        self.yDataIndices = []  # array of indices that show the index in self.rawdata - if there are no missing values then array[i] = i
        self.attrLabels = []
        
    #
    # update shown data. Set labels, coloring by className ....
    #
    def updateData(self, labels):
        self.removeCurves()
        self.tips.removeAll()

        self.attrLabels = labels        
        self.length = len(labels)
        indices = []
        xs = []
        #if self.tooltipKind == DONT_SHOW_TOOLTIPS: MyQToolTip.tip(self.tooltip, QRect(0,0,0,0), "")


        if len(self.scaledData) == 0 or len(labels) == 0:
            self.setAxisScaleDraw(QwtPlot.xBottom, DiscreteAxisScaleDraw(labels))
            self.setAxisScale(QwtPlot.yLeft, 0, 1, 1)
            return

        # create a table of indices that stores the sequence of variable indices
        for label in labels:
            index = self.attributeNames.index(label)
            indices.append(index)

        validData = [1] * len(self.rawdata)
        for i in range(len(self.rawdata)):
            for j in range(self.length):
                if self.scaledData[indices[j]][i] == "?": validData[i] = 0
        totalValid = 0
        for val in validData: totalValid += val

        self.setAxisScale(QwtPlot.yLeft, 0, totalValid, totalValid)
        self.setAxisScale(QwtPlot.xBottom, -0.5, len(labels)-0.5, 1)
        self.setAxisMaxMajor(QwtPlot.xBottom, len(labels)-1.0)        
        self.setAxisMaxMinor(QwtPlot.xBottom, 0)
        self.setAxisScaleDraw(QwtPlot.xBottom, DiscreteAxisScaleDraw(labels))
        #self.setAxisScale(QwtPlot.yLeft, 0, 1, 1)
        
        # draw vertical lines that represent attributes
        for i in range(len(labels)):
            newCurveKey = self.insertCurve(labels[i])
            self.setCurveData(newCurveKey, [i,i], [0,1])

        self.repaint()  # we have to repaint to update scale to get right coordinates for tooltip rectangles
        self.updateLayout()

        classNameIndex = -1
        if self.rawdata.domain.classVar: classNameIndex = self.attributeNames.index(self.rawdata.domain.classVar.name)
        
        xs = range(self.length)
        count = len(self.rawdata)
        pos = 0
        self.yDataIndices = []
        
        for i in range(count):
            if validData[i] == 0: continue
            
            curve = subBarQwtPlotCurve(self)
            newColor = QColor(0,0,0)
            if classNameIndex >= 0: newColor.setHsv(self.coloringScaledData[classNameIndex][i], 255, 255)
                
            curve.color = newColor
            curve.penColor = newColor
            xData = []; yData = []
            for j in range(self.length):
                width = self.scaledData[indices[j]][i] * 0.45
                xData += [j-width, j+width]
                yData += [pos, pos+1]

            ##########
            pos += 1
            self.yDataIndices.append(pos)

            ckey = self.insertCurve(curve)
            self.setCurveStyle(ckey, QwtCurve.UserCurve)
            self.setCurveData(ckey, xData, yData)

        if self.enabledLegend and self.rawdata.domain.classVar and self.rawdata.domain.classVar.varType == orange.VarTypes.Discrete:
            varValues = getVariableValuesSorted(self.rawdata, self.rawdata.domain.classVar.name)
            colors = ColorPaletteHSV(len(varValues))
            for ind in range(len(varValues)):
                self.addCurve(self.rawdata.domain.classVar.name + "=" + varValues[ind], colors.getColor(ind), colors.getColor(ind), self.pointWidth, enableLegend = 1)

           

    # show rectangle with example shown under mouse cursor
    def onMouseMoved(self, e):
        self.hideSelectedRectangle()
        if self.mouseCurrentlyPressed:
            OWVisGraph.onMouseMoved(self, e)
        elif not self.rawdata:
            return
        else:
            yFloat = floor(self.invTransform(QwtPlot.yLeft, e.y()))
            if self.exampleTracking:
                width = 0.49
                xData = [-width, self.length+width-1, self.length+width-1, -width, -width]
                yData = [yFloat, yFloat, yFloat+1, yFloat+1, yFloat]
                self.selectedRectangle = self.insertCurve("test")
                self.setCurveData(self.selectedRectangle, xData, yData)
                self.setCurveStyle(self.selectedRectangle, QwtCurve.Lines)
                self.replot()
            else:
                OWVisGraph.onMouseMoved(self, e)

            if (self.tooltipKind == VISIBLE_ATTRIBUTES and self.attrLabels != []) or self.tooltipKind == ALL_ATTRIBUTES:
                if int(yFloat) >= len(self.rawdata): return
                if self.tooltipKind == VISIBLE_ATTRIBUTES:
                    text = self.getShortExampleText(self.rawdata, self.rawdata[int(yFloat)], self.attrLabels)
                else:
                    text = self.getShortExampleText(self.rawdata, self.rawdata[int(yFloat)], [attr.name for attr in self.rawdata.domain])
                y1Int = self.transform(QwtPlot.yLeft, yFloat)
                y2Int = self.transform(QwtPlot.yLeft, yFloat+1.0)
                MyQToolTip.tip(self.tooltip, QRect(e.x()+self.canvas().frameGeometry().x()-10, y2Int+self.canvas().frameGeometry().y(), 20, y1Int-y2Int), text[:-2].replace("; ", "\n"))
                OWVisGraph.onMouseMoved(self, e)

            
            

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
