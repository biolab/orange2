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
        
    #
    # update shown data. Set labels, coloring by className ....
    #
    def updateData(self, labels, className, statusBar = None):
        self.removeCurves()
        self.statusBar = statusBar
        self.tips.removeAll()
        if len(self.scaledData) == 0 or len(labels) == 0: self.updateLayout(); return
        
        self.setAxisScaleDraw(QwtPlot.xBottom, DiscreteAxisScaleDraw(labels))
        self.setAxisScale(QwtPlot.yLeft, 0, len(self.rawdata), len(self.rawdata))
        self.setAxisScale(QwtPlot.xBottom, -0.5, len(labels)-0.5, 1)
        self.setAxisMaxMajor(QwtPlot.xBottom, len(labels)-1.0)        
        self.setAxisMaxMinor(QwtPlot.xBottom, 0)

        length = len(labels)
        indices = []
        xs = []

        # create a table of indices that stores the sequence of variable indices
        for label in labels:
            index = self.scaledDataAttributes.index(label)
            indices.append(index)

        # create a table of class values that will be used for coloring the lines
        scaledClassData = []
        classValueIndices = {}
        classValueCount = 0
        if className != "(One color)" and className != '' and self.rawdata.domain[className].varType != orange.VarTypes.Discrete:
            scaledClassData = self.scaleData(self.rawdata, className)
        else:
            classValueCount = len(self.rawdata.domain[className].values)
            classValueIndices = self.getVariableValueIndices(self.rawdata, className)

        # draw vertical lines that represent attributes
        for i in range(len(labels)):
            newCurveKey = self.insertCurve(labels[i])
            self.setCurveData(newCurveKey, [i,i], [0,1])

        self.repaint()  # we have to repaint to update scale to get right coordinates for tooltip rectangles
        self.updateLayout()
            
        xs = range(length)
        count = len(self.rawdata)
        for i in range(count):
            curve = subBarQwtPlotCurve(self)
            newColor = QColor(0,0,0)
            if scaledClassData != []:
                newColor.setHsv(scaledClassData[i]*360, 255, 255)
            elif classValueIndices != {}:
                val = self.rawdata[i][className].value
                newColor.setHsv(float(classValueIndices[val]*360)/float(classValueCount), 255, 255)
                
            curve.color = newColor
            curve.penColor = newColor
            xData = []; yData = []
            for j in range(length):
                width = self.scaledData[indices[j]][i] * 0.45
                xData.append(j-width)
                xData.append(j+width)
                yData.append(i)
                yData.append(i+1)

            ##########
            # we add a tooltip for this point
            r = QRectFloat(-0.5, i, length, 1)
            text = self.getExampleText(self.rawdata, self.rawdata[i])
            self.tips.addToolTip(r, text)
            ##########

            ckey = self.insertCurve(curve)
            self.setCurveStyle(ckey, QwtCurve.UserCurve)
            self.setCurveData(ckey, xData, yData)
            


if __name__== "__main__":
    #Draw a simple graph
    a = QApplication(sys.argv)        
    c = OWSurveyPlotGraph()
            
    a.setMainWidget(c)
    c.show()
    a.exec_loop()
