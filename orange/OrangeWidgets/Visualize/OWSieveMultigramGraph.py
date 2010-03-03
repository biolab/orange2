from OWGraph import *
from orngScaleData import *
from math import sqrt

class OWSieveMultigramGraph(OWGraph, orngScaleData):
    def __init__(self, parent = None, name = None):
        "Constructs the graph"
        OWGraph.__init__(self, parent, name)
        orngScaleData.__init__(self)
        self.lineWidth = 5
        self.minPearson = 2
        self.maxPearson = 10
        self.enableXaxis(0)
        self.enableYLaxis(0)
        self.setAxisScale(QwtPlot.xBottom, -1.25, 1.25, 1)
        self.setAxisScale(QwtPlot.yLeft, -1.25, 1.25, 1)

    def setData(self, data):
        OWGraph.setData(self, data)
        orngScaleData.setData(self, data)

    def updateData(self, data, labels, probabilities):
        self.clear()
        self.tips.removeAll()

        # we must have at least 3 attributes to be able to show anything
        if len(labels) < 3: return

        length = len(labels)
        indices = []
        xs = []

        attrNameList = []
        for attr in data.domain: attrNameList.append(attr.name)

        ###########
        # create a table of indices that stores the sequence of variable indices
        for label in labels: indices.append(attrNameList.index(label))

        ###########
        # create anchor for two edges of every attribute
        anchors = [[],[]]
        for i in range(length):
            x = math.cos(2*math.pi * float(i) / float(length)); strX = "%.4f" % (x)
            y = math.sin(2*math.pi * float(i) / float(length)); strY = "%.4f" % (y)
            anchors[0].append(float(strX))  # this might look stupid, but this way we get rid of rounding errors
            anchors[1].append(float(strY))

        ###########
        # draw polygon
        xData = []; yData = []
        for i in range(len(labels)+1):
            x = math.cos(2*math.pi * float(i) / float(len(labels)))
            y = math.sin(2*math.pi * float(i) / float(len(labels)))
            xData.append(x)
            yData.append(y)
        newColor = QColor()
        newColor.setRgb(0, 0, 0)
        self.addCurve("", newColor, newColor, 2, QwtPlotCurve.Lines, xData = xData, yData = yData)
        
        ###########
        # draw text at lines
        for i in range(length):
            # print attribute name
            self.addMarker(labels[i], 0.6*(anchors[0][i]+anchors[0][(i+1)%length]), 0.6*(anchors[1][i]+anchors[1][(i+1)%length]), Qt.AlignHCenter | Qt.AlignVCenter, bold = 1)
            
            if data.domain[labels[i]].varType == orange.VarTypes.Discrete:
                # print all possible attribute values
                values = data.domain[labels[i]].values
                count = len(values)
                k = 1.08
                for j in range(count):
                    pos = (1.0 + 2.0*float(j)) / float(2*count)
                    self.addMarker(values[j], k*(1-pos)*anchors[0][i]+k*pos*anchors[0][(i+1)%length], k*(1-pos)*anchors[1][i]+k*pos*anchors[1][(i+1)%length], Qt.AlignHCenter | Qt.AlignVCenter)

        # -----------------------------------------------------------
        #  create data lines
        # -----------------------------------------------------------
        for attrXindex in range(len(labels)):
            attrXName = labels[attrXindex]

            for attrYindex in range(attrXindex+1, len(labels)):
                attrYName = labels[attrYindex]

                for valXindex in range(len(data.domain[attrXName].values)):
                    valX = data.domain[attrXName].values[valXindex]

                    for valYindex in range(len(data.domain[attrYName].values)):
                        valY = data.domain[attrYName].values[valYindex]

                        ((nameX, countX),(nameY, countY), actual, sum) = probabilities['%s+%s:%s+%s' %(attrXName, valX, attrYName, valY)]

                        # calculate starting and ending coordinates for lines
                        val = (1.0 + 2.0*float(valXindex)) / float(2*len(data.domain[attrXName].values))
                        attrXDataAnchorX = anchors[0][attrXindex]*(1-val) + anchors[0][(attrXindex+1)%length]*val
                        attrXDataAnchorY = anchors[1][attrXindex]*(1-val) + anchors[1][(attrXindex+1)%length]*val

                        val = (1.0 + 2.0*float(valYindex)) / float(2*len(data.domain[attrYName].values))
                        attrYDataAnchorX = anchors[0][attrYindex]*(1-val) + anchors[0][(attrYindex+1)%length]*val
                        attrYDataAnchorY = anchors[1][attrYindex]*(1-val) + anchors[1][(attrYindex+1)%length]*val

                        self.addLinePearson([attrXDataAnchorX, attrYDataAnchorX], [attrXDataAnchorY, attrYDataAnchorY], countX, countY, actual, sum)
        self.replot()


    def addLinePearson(self, xDataList, yDataList, countX, countY, actual, sum):
        expected = float(countX*countY)/float(sum)
        if actual == expected == 0: return
        elif expected == 0:     # if expected == 0 we have to solve division by zero. In reverse example (when actual == 0) pearson = -expected/sqrt(expected)
            pearson = actual/sqrt(actual)
        else:
            pearson = (actual - expected) / sqrt(expected)

        if abs(pearson) < self.minPearson: return       # we don't want to draw white lines

        if pearson > 0:     # if there are more examples that we would expect under the null hypothesis
            intPearson = min(math.floor(pearson), self.maxPearson)
            b = 255
            r = g = 255 - intPearson*200.0/float(self.maxPearson)
            r = g = max(r, 55)  #
            penWidth = int(float(intPearson*self.lineWidth)/float(self.maxPearson))
        elif pearson < 0:
            intPearson = max(math.ceil(pearson), -self.maxPearson)
            r = 255
            b = g = 255 + intPearson*200.0/float(self.maxPearson)
            b = g = max(b, 55)
            penWidth = int(float(intPearson*self.lineWidth)/float(-self.maxPearson))
        color = QColor(r,g,b)
        key = self.addCurve('', color, color, 0, QwtPlotCurve.Lines, symbol = QwtSymbol.NoSymbol, xData = xDataList, yData = yDataList, lineWidth = penWidth)



if __name__== "__main__":
    #Draw a simple graph
    a = QApplication(sys.argv)
    c = OWSieveMultigramGraph()

    c.show()
    a.exec_()
