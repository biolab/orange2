#
# OWParallelGraph.py
#
# the base for all parallel graphs

import sys
import math
import orange
import os.path
from qt import *
from OWTools import *
from qwt import *
from Numeric import *

class subBarQwtPlotCurve(QwtPlotCurve):
    def __init__(self, parent = None, text = None):
        QwtPlotCurve.__init__(self, parent, text)
        self.color = Qt.black

    def draw(self, p, xMap, yMap, f, t):
        p.setBackgroundMode(Qt.OpaqueMode)
        p.setBackgroundColor(self.color)
        p.setBrush(self.color)
        p.setPen(Qt.black)
        if t < 0: t = self.dataSize() - 1
        if divmod(f, 2)[1] != 0: f -= 1
        if divmod(t, 2)[1] == 0:  t += 1
        for i in range(f, t+1, 2):
            px1 = xMap.transform(self.x(i))
            py1 = yMap.transform(self.y(i))
            px2 = xMap.transform(self.x(i+1))
            py2 = yMap.transform(self.y(i+1))
            p.drawRect(px1, py1, (px2 - px1), (py2 - py1))

class DiscreteAxisScaleDraw(QwtScaleDraw):
    def __init__(self, labels):
        apply(QwtScaleDraw.__init__, (self,))
        self.labels = labels

    def label(self, value):
        index = int(round(value))
        if (index >= len(self.labels)):
            return ''
        if (index < 0):
            return ''
        return QString(str(self.labels[index]))

class OWParallelGraph(QwtPlot):
    def __init__(self, parent = None, name = None):
        "Constructs the graph"
        QwtPlot.__init__(self, 10007, parent, name)
        self.setWFlags(Qt.WResizeNoErase) #this works like magic.. no flicker during repaint!

        self.scaledData = []
        self.scaledDataAttributes = []
        self.jitteringType = 'none'
        self.showDistributions = 0
        self.GraphCanvasColor = str(Qt.white.name())

        self.enableAxis(QwtPlot.yLeft, 1)
        self.enableAxis(QwtPlot.xBottom, 1)
        self.setAutoReplot(FALSE)
        self.setAutoLegend(FALSE)
        self.setAxisAutoScale(QwtPlot.xBottom)
        self.setAxisAutoScale(QwtPlot.xTop)
        self.setAxisAutoScale(QwtPlot.yLeft)
        self.setAxisAutoScale(QwtPlot.yRight)
        self.setCanvasColor(QColor(Qt.white))
        self.repaint()

        newFont = QFont('Helvetica', 10, QFont.Bold)
        self.setTitleFont(newFont)
        self.enableGridX(FALSE)
        self.enableGridY(FALSE)
        self.setAxisTitleFont(QwtPlot.xBottom, newFont)
        self.setAxisTitleFont(QwtPlot.xTop, newFont)
        self.setAxisTitleFont(QwtPlot.yLeft, newFont)
        self.setAxisTitleFont(QwtPlot.yRight, newFont)
        #self.setAxisScale(QwtPlot.yLeft, 0, 1, 1)
        #self.setAxisScale(QwtPlot.yRight, 0, 1, 1)

        newFont = QFont('Helvetica', 9)
        self.setAxisFont(QwtPlot.xBottom, newFont)
        self.setAxisFont(QwtPlot.xTop, newFont)
        self.setAxisFont(QwtPlot.yLeft, newFont)
        self.setAxisFont(QwtPlot.yRight, newFont)
        self.setLegendFont(newFont)

        self.tipLeft = None
        self.tipRight = None
        self.tipBottom = None
        self.dynamicToolTip = DynamicToolTip(self)

        self.showMainTitle = FALSE
        self.mainTitle = None
        self.showXaxisTitle = FALSE
        self.XaxisTitle = None
        self.showYLaxisTitle = FALSE
        self.YLaxisTitle = None
        self.showYRaxisTitle = FALSE
        self.YRaxisTitle = None
        
        self.noneSymbol = QwtSymbol()
        self.noneSymbol.setStyle(QwtSymbol.None)        
        self.curveIndex = 0

    def setCanvasColor(self, c):
        self.GraphCanvasColor = c
        self.setCanvasBackground(c)
        self.repaint()

    def setShowDistributions(self, showDistributions):
        self.showDistributions = showDistributions


    def setJitteringOption(self, jitteringType):
        self.jitteringType = jitteringType


    def saveToFile(self):
        qfileName = QFileDialog.getSaveFileName("graph.png","Portable Network Graphics (.PNG)\nWindows Bitmap (.BMP)\nGraphics Interchange Format (.GIF)", None, "Save to..")
        fileName = str(qfileName)
        (fil,ext) = os.path.splitext(fileName)
        ext = ext.replace(".","")
        ext = ext.upper()

        buffer = QPixmap(self.size()) # any size can do, now using the window size
        painter = QPainter(buffer)
        painter.fillRect(buffer.rect(), QBrush(self.palette().active().background())) # make background same color as the widget's background
        self.printPlot(painter, buffer.rect())
        painter.end()
        buffer.save(fileName, ext)
    
    def setXlabels(self, labels):
        self.setAxisScaleDraw(QwtPlot.xBottom, DiscreteAxisScaleDraw(labels))
        self.setAxisScale(QwtPlot.xBottom, 0, len(labels) - 1, 1)
        self.setAxisMaxMinor(QwtPlot.xBottom, 0)
        self.setAxisMaxMajor(QwtPlot.xBottom, len(labels))
        #self.updateToolTips()


    def resizeEvent(self, event):
        "Makes sure that the plot resizes"
        #self.updateToolTips()
        self.updateLayout()

    def paintEvent(self, qpe):
        QwtPlot.paintEvent(self, qpe) #let the ancestor do its job
        self.replot()
 
    #
    # scale data at index index to the interval 0 - 1
    #
    def scaleData(self, data, index):
        attr = data.domain[index]
        temp = [];
        # is the attribute discrete
        if attr.varType == orange.VarTypes.Discrete:
            # we create a hash table of variable values and their indices
            variableValueIndices = {}
            for i in range(len(attr.values)):
                variableValueIndices[attr.values[i]] = i

            count = float(len(attr.values))
            if len(attr.values) > 1: num = float(len(attr.values)-1)
            else: num = float(1)

            for i in range(len(data)):
                if data[i][index].isSpecial(): temp.append(1)
                else:
                    val = (1.0 + 2.0*float(variableValueIndices[data[i][index].value])) / float(2*count) + 0.2 * self.rndCorrection(1.0/count)
                    temp.append(val)
                    
        # is the attribute continuous
        else:
            # first find min and max value
            min = data[0][attr].value
            max = data[0][attr].value
            for item in data:
                if item[attr].value < min:
                    min = item[attr].value
                elif item[attr].value > max:
                    max = item[attr].value

            diff = max - min
            # create new list with values scaled from 0 to 1
            for i in range(len(data)):
                temp.append((data[i][attr].value - min) / diff)
        return temp

    #
    # set new data and scale its values
    #
    def setData(self, data):
        self.rawdata = data
        self.scaledData = []
        self.scaledDataAttributes = []
        
        if data == None: return

        self.distributions = []; self.totals = []
        for index in range(len(data.data.domain)):
            attr = data.data.domain[index]
            self.scaledDataAttributes.append(attr.name)
            scaled = self.scaleData(data.data, index)
            self.scaledData.append(scaled)

    #
    # update shown data. Set labels, coloring by className ....
    #
    def updateData(self, labels, className):
        self.removeCurves()
        self.axesKeys = []
        self.curveKeys = []
        
        self.setAxisScaleDraw(QwtPlot.xBottom, DiscreteAxisScaleDraw(labels))
        if self.showDistributions == 1:
            self.setAxisScale(QwtPlot.xBottom, 0, len(labels)-0.5, 1)
        else:
            self.setAxisScale(QwtPlot.xBottom, 0, len(labels), 1)
        self.setAxisMaxMinor(QwtPlot.xBottom, 0)
        self.setAxisMaxMajor(QwtPlot.xBottom, len(labels))
        self.setAxisMaxMinor(QwtPlot.yLeft, 0)
        self.setAxisMaxMajor(QwtPlot.yLeft, 1)
        

        if len(self.scaledData) == 0 or len(labels) == 0: self.updateLayout(); return

        length = len(labels)
        indices = []
        xs = []

        # create a table of indices that stores the sequence of variable indices
        for label in labels:
            index = self.scaledDataAttributes.index(label)
            indices.append(index)

        # create a table of class values that will be used for coloring the lines
        scaledClassData = []
        if className != "(One color)" and className != '':
            ex_jitter = self.jitteringType
            self.setJitteringOption('none')
            scaledClassData = self.scaleData(self.rawdata.data, className)
            self.setJitteringOption(ex_jitter)

        xs = range(length)
        dataSize = len(self.scaledData[0])        
        for i in range(dataSize):
            newCurveKey = self.insertCurve(str(i))
            self.curveKeys.append(newCurveKey)
            newColor = QColor()
            if scaledClassData != []:
                newColor.setHsv(scaledClassData[i]*255, 255, 255)
            self.setCurvePen(newCurveKey, QPen(newColor))
            ys = []
            for index in indices:
                ys.append(self.scaledData[index][i])
            self.setCurveData(newCurveKey, xs, ys)

        # draw vertical lines that represent attributes
        for i in range(len(labels)):
            newCurveKey = self.insertCurve(labels[i])
            self.axesKeys.append(newCurveKey)
            self.setCurveData(newCurveKey, [i,i], [0,1])

        # do we want to show distributions with discrete attributes
        if self.showDistributions and className != "(One color)" and className != "" and self.rawdata.data.domain[className].varType == orange.VarTypes.Discrete:
            self.showDistributionValues(className, self.rawdata.data, indices)
            
        

    def showDistributionValues(self, className, data, indices):
        # get index of class         
        classNameIndex = 0
        for i in range(len(data.domain)):
            if data.domain[i].name == className: classNameIndex = i

        # create color table            
        count = float(len(data.domain[className].values))
        if count < 1:
            count = 1.0

        colors = []
        for i in range(count): colors.append(float(1+2*i)/float(2*count))

        classValueIndices = {}
        # we create a hash table of possible class values (IF we have a discrete class)
        for i in range(count):
            classValueIndices[list(data.domain[className].values)[i]] = i

        for graphAttrIndex in range(len(indices)):
            index = indices[graphAttrIndex]
            if data.domain[index].varType == orange.VarTypes.Discrete:
                attr = data.domain[index]
                attrLen = len(attr.values)
                
                values = []
                totals = [0] * attrLen

                # we create a hash table of variable values and their indices
                variableValueIndices = {}
                for i in range(attrLen):
                    variableValueIndices[attr.values[i]] = i
                
                for i in range(count):
                    values.append([0] * attrLen)

                for i in range(len(data)):
                    if not data[i][index].isSpecial():
                        # processing for distributions
                        attrIndex = variableValueIndices[data[i][index].value]
                        classIndex = classValueIndices[data[i][classNameIndex].value]
                        totals[attrIndex] += 1
                        values[classIndex][attrIndex] = values[classIndex][attrIndex] + 1

                maximum = 0
                for i in range(len(values)):
                    for j in range(len(values[i])):
                        if values[i][j] > maximum: maximum = values[i][j]
                        
                # create bar curve
                for i in range(count):
                    curve = subBarQwtPlotCurve(self)
                    newColor = QColor()
                    newColor.setHsv(colors[i]*255, 255, 255)
                    curve.color = newColor
                    xData = []
                    yData = []
                    for j in range(attrLen):
                        #width = float(values[i][j]*0.5) / float(totals[j])
                        width = float(values[i][j]*0.5) / float(maximum)
                        interval = 1.0/float(2*attrLen)
                        yOff = float(1.0 + 2.0*j)/float(2*attrLen)
                        height = 0.7/float(count*attrLen)

                        yLowBott = yOff - float(count*height)/2.0 + i*height
                        xData.append(graphAttrIndex)
                        xData.append(graphAttrIndex + width)
                        yData.append(yLowBott)
                        yData.append(yLowBott + height)

                    ckey = self.insertCurve(curve)
                    self.setCurveStyle(ckey, QwtCurve.UserCurve)
                    self.setCurveData(ckey, xData, yData)


    def rndCorrection(self, max):
        """
        returns a number from -max to max, self.jitteringType defines which distribution is to be used.
        function is used to plot data points for categorical variables
        """    
        if self.jitteringType == 'none': 
            return 0.0
        elif self.jitteringType  == 'uniform': 
            return (random() - 0.5)*2*max
        elif self.jitteringType  == 'triangle': 
            b = (1 - betavariate(1,1)) ; return choice((-b,b))*max
        elif self.jitteringType  == 'beta': 
            b = (1 - betavariate(1,2)) ; return choice((-b,b))*max
             
    
if __name__== "__main__":
    #Draw a simple graph
    a = QApplication(sys.argv)        
    c = OWParallelGraph()
    c.setCoordinateAxes(['red','green','blue','light blue', 'dark blue', 'yellow', 'orange', 'magenta'])
    #c.setMainTitle("Graph Title")
    #c.setShowMainTitle(1)
        
    a.setMainWidget(c)
    c.show()
    a.exec_loop()
