#
# OWParallelGraph.py
#
# the base for all parallel graphs

from OWVisGraph import *

class OWSurveyPlotGraph(OWVisGraph):
    def __init__(self, parent = None, name = None):
        "Constructs the graph"
        OWVisGraph.__init__(self, parent, name)
        self.jitteringType = "none"
        self.selectedRectangle = 0
        self.exampleTracking = 1
        self.length = 0 # number of shown attributes - we need it also in mouse movement
        
    #
    # update shown data. Set labels, coloring by className ....
    #
    def updateData(self, labels, className, statusBar = None):
        self.removeCurves()
        self.statusBar = statusBar
        self.tips.removeAll()
        if len(self.scaledData) == 0 or len(labels) == 0: self.updateLayout(); return

        if className != "(One color)": classNameIndex = self.attributeNames.index(className)
        else:                          classNameIndex = -1
        self.length = len(labels)
        indices = []
        xs = []

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
        
        self.setAxisScaleDraw(QwtPlot.xBottom, DiscreteAxisScaleDraw(labels))
        self.setAxisScale(QwtPlot.yLeft, 0, totalValid, totalValid)
        self.setAxisScale(QwtPlot.xBottom, -0.5, len(labels)-0.5, 1)
        self.setAxisMaxMajor(QwtPlot.xBottom, len(labels)-1.0)        
        self.setAxisMaxMinor(QwtPlot.xBottom, 0)

        
        # draw vertical lines that represent attributes
        for i in range(len(labels)):
            newCurveKey = self.insertCurve(labels[i])
            self.setCurveData(newCurveKey, [i,i], [0,1])

        self.repaint()  # we have to repaint to update scale to get right coordinates for tooltip rectangles
        self.updateLayout()
            
        xs = range(self.length)
        count = len(self.rawdata)
        pos = 0
        for i in range(count):
            if validData[i] == 0: continue
            
            curve = subBarQwtPlotCurve(self)
            newColor = QColor(0,0,0)
            if classNameIndex != -1: newColor.setHsv(self.coloringScaledData[classNameIndex][i]*360, 255, 255)
                
            curve.color = newColor
            curve.penColor = newColor
            xData = []; yData = []
            for j in range(self.length):
                width = self.scaledData[indices[j]][i] * 0.45
                xData += [j-width, j+width]
                yData += [pos, pos+1]

            ##########
            # we add a tooltip for this point
            r = QRectFloat(-0.5, pos, self.length, 1)
            text = self.getExampleText(self.rawdata, self.rawdata[i])
            self.tips.addToolTip(r, text)
            pos += 1
            ##########

            ckey = self.insertCurve(curve)
            self.setCurveStyle(ckey, QwtCurve.UserCurve)
            self.setCurveData(ckey, xData, yData)

    # show rectangle with example shown under mouse cursor
    def onMouseMoved(self, e):
        if self.mouseCurrentlyPressed: return
        else:
            self.hideSelectedRectangle()
            if not self.exampleTracking:
                self.replot()
                return
            width = 0.49
            y = floor(self.invTransform(QwtPlot.yLeft, e.y()))
            xData = [-width, self.length+width-1, self.length+width-1, -width, -width]
            yData = [y, y, y+1, y+1, y]
            self.selectedRectangle = self.insertCurve("test")
            self.setCurveData(self.selectedRectangle, xData, yData)
            self.setCurveStyle(self.selectedRectangle, QwtCurve.Lines)
            OWVisGraph.onMouseMoved(self, e)
            self.replot()

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
