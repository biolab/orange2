from qt import *
from qwt import *
#from Numeric import *
#from OWGraphTools import *
from math import sqrt

colorHueValues = [240, 0, 120, 30, 60, 300, 180, 150, 270, 90, 210, 330, 15, 135, 255, 45, 165, 285, 105, 225, 345]

colorHSVValues = [(240, 255, 255), (0, 255, 255), (120, 255, 255), (30, 255, 255), (60, 255, 255),
                  (300, 255, 255), (180, 255, 255), (270, 255, 255), (210, 255, 255), (45, 127, 255),
                  (45, 127, 127), (30, 255, 92), (120, 255, 84), (60, 255, 192), (180, 255, 127),
                  (0, 255, 128), (300, 255, 127)]

#ColorBrewer color set - bad when there are small points
colorRGBValues = [(0, 140, 255), (228, 26, 28), (77, 175, 74), (152, 78, 163), (255, 127, 0), (255, 255, 51), (166, 86, 40), (247, 129, 191), (153, 153, 153)]

class ColorPaletteHSV:
    maxHueVal = 260
    
    def __init__(self, numberOfColors = -1, brightness = 255):
        self.brightness = brightness
        self.numberOfColors = numberOfColors
        
        self.rebuildColors()

    def rebuildColors(self):
        self.colors = []
        self.hueValues = []
        if self.numberOfColors == -1: return  # used for coloring continuous variables
        #elif self.numberOfColors <= len(colorRGBValues): 
        elif self.numberOfColors <= len(colorHSVValues):
            self.hueValues = colorHSVValues[:self.numberOfColors]
            for i in self.hueValues:
                c = QColor()
                c.setHsv(*i)
                self.colors.append(c)
        else:   
            self.hueValues = [int(float(x*self.maxHueVal)/float(self.numberOfColors)) for x in range(self.numberOfColors)]
            for hue in self.hueValues:
                col = QColor()
                col.setHsv(hue, self.brightness, 255)
                self.colors.append(col)
        

    def __getitem__(self, index, brightness = None):
        # is this color for continuous attribute?
        if self.numberOfColors == -1:                
            col = QColor()
            col.setHsv(index*self.maxHueVal, self.brightness, 255)     # index must be between 0 and 1
            return col
        # if we want a standard color, just with a specific brightness value
        elif brightness != None:    
            col = QColor()
            col.setHsv(colorHueValues[index], brightness, 255)
            return col
        # return a color from the built table
        else:                                   # get color for discrete attribute
            return self.colors[index]           # index must be between 0 and self.numberofColors

    # get only hue value for given index
    def getHue(self, index):
        if self.numberOfColors == -1:
            return index * self.maxHueVal
        else:
            return self.hueValues[index]

    def getBrightness(self):
        return self.brightness

    def setBrightness(self, brightness):
        self.brightness = brightness
        self.rebuildColors()

    # get QColor instance for given index
    def getColor(self, index, brightness = None):
        return self.__getitem__(index, brightness)


class ColorPaletteBrewer:
    maxHueVal = 260
    
    def __init__(self, numberOfColors = -1, brightness = 255):
        self.brightness = brightness
        self.numberOfColors = numberOfColors
        
        self.rebuildColors()

    def rebuildColors(self):
        self.colors = []
        self.hueValues = []
        if self.numberOfColors == -1: return  # used for coloring continuous variables
        elif self.numberOfColors <= len(colorRGBValues): 
            for i in range(self.numberOfColors):
                self.colors.append(QColor(colorRGBValues[i][0], colorRGBValues[i][1], colorRGBValues[i][2]))
            self.hueValues = [c.hsv()[0] for c in self.colors]
        else:   
            self.hueValues = [int(float(x*self.maxHueVal)/float(self.numberOfColors)) for x in range(self.numberOfColors)]
            for hue in self.hueValues:
                col = QColor()
                col.setHsv(hue, self.brightness, 255)
                self.colors.append(col)
        

    def __getitem__(self, index, brightness = None):
        # is this color for continuous attribute?
        if self.numberOfColors == -1:                
            col = QColor()
            col.setHsv(index*self.maxHueVal, self.brightness, 255)     # index must be between 0 and 1
            return col
        # if we want a standard color, just with a specific brightness value
        elif brightness != None:    
            col = QColor()
            col.setHsv(colorHueValues[index], brightness, 255)
            return col
        # return a color from the built table
        else:                                   # get color for discrete attribute
            return self.colors[index]           # index must be between 0 and self.numberofColors

    # get only hue value for given index
    def getHue(self, index):
        if self.numberOfColors == -1:
            return index * self.maxHueVal
        else:
            return self.hueValues[index]

    def getBrightness(self):
        return self.brightness

    def setBrightness(self, brightness):
        self.brightness = brightness
        self.rebuildColors()

    # get QColor instance for given index
    def getColor(self, index, brightness = None):
        return self.__getitem__(index, brightness)
            

# black and white color palette
class ColorPaletteBW:
    def __init__(self, numberOfColors = -1, brightest = 50, darkest = 255):
        self.colors = []
        self.numberOfColors = numberOfColors
        self.brightest = brightest
        self.darkest = darkest
        
        if numberOfColors == -1: return  # used for coloring continuous variables
        else:   
            for val in [int(brightest + (darkest-brightest)*x/float(numberOfColors-1)) for x in range(numberOfColors)]:
                self.colors.append(QColor(val, val, val))

    def __getitem__(self, index):
        if self.numberOfColors == -1:                # is this color for continuous attribute?
            val = int(self.brightest + (self.darkest-self.brightest)*index)
            return QColor(val, val, val)
        else:                                   # get color for discrete attribute
            return self.colors[index]           # index must be between 0 and self.numberofColors
                
    # get QColor instance for given index
    def getColor(self, index):
        return self.__getitem__(index)

        

# ####################################################################
# calculate Euclidean distance between two points
def EuclDist(v1, v2):
    val = 0
    for i in range(len(v1)):
        val += (v1[i]-v2[i])**2
    return sqrt(val)
        

# ####################################################################
# add val to sorted list list. if len > maxLen delete last element
def addToList(list, val, ind, maxLen):
    i = 0
    for i in range(len(list)):
        (val2, ind2) = list[i]
        if val < val2:
            list.insert(i, (val, ind))
            if len(list) > maxLen:
                list.remove(list[maxLen])
            return
    if len(list) < maxLen:
        list.insert(len(list), (val, ind))

# ####################################################################
# used in widgets that enable to draw a rectangle or a polygon to select a subset of data points
class SelectionCurve(QwtPlotCurve):
    def __init__(self, parent, name = "", pen = Qt.SolidLine ):
        QwtPlotCurve.__init__(self, parent, name)
        self.pointArrayValid = 0
        self.setStyle(QwtCurve.Lines)
        self.setPen(QPen(QColor(128,128,128), 1, pen))
        
    def addPoint(self, xPoint, yPoint):
        xVals = []
        yVals = []
        for i in range(self.dataSize()):
            xVals.append(self.x(i))
            yVals.append(self.y(i))
        xVals.append(xPoint)
        yVals.append(yPoint)
        self.setData(xVals, yVals)
        self.pointArrayValid = 0        # invalidate the point array

    def removeLastPoint(self):
        xVals = []
        yVals = []
        for i in range(self.dataSize()-1):
            xVals.append(self.x(i))
            yVals.append(self.y(i))
        self.setData(xVals, yVals)
        self.pointArrayValid = 0        # invalidate the point array

    def replaceLastPoint(self, xPoint, yPoint):
        xVals = []
        yVals = []
        for i in range(self.dataSize()-1):
            xVals.append(self.x(i))
            yVals.append(self.y(i))
        xVals.append(xPoint)
        yVals.append(yPoint)
        self.setData(xVals, yVals)
        self.pointArrayValid = 0        # invalidate the point array

    # is point defined at x,y inside a rectangle defined with this curve
    def isInside(self, x, y):       
        xMap = self.parentPlot().canvasMap(self.xAxis());
        yMap = self.parentPlot().canvasMap(self.yAxis());

        if not self.pointArrayValid:
            self.pointArray = QPointArray(self.dataSize() + 1)
            for i in range(self.dataSize()):
                self.pointArray.setPoint(i, xMap.transform(self.x(i)), yMap.transform(self.y(i)))
            self.pointArray.setPoint(self.dataSize(), xMap.transform(self.x(0)), yMap.transform(self.y(0)))
            self.pointArrayValid = 1

        return QRegion(self.pointArray).contains(QPoint(xMap.transform(x), yMap.transform(y)))

    # test if the line going from before last and last point intersect any lines before
    # if yes, then add the intersection point and remove the outer points
    def closed(self):
        if self.dataSize() < 5: return 0
        x1 = self.x(self.dataSize()-3)
        x2 = self.x(self.dataSize()-2)
        y1 = self.y(self.dataSize()-3)
        y2 = self.y(self.dataSize()-2)
        for i in range(self.dataSize()-5, -1, -1):
            X1 = self.x(i)
            X2 = self.x(i+1)
            Y1 = self.y(i)
            Y2 = self.y(i+1)
            (intersect, xi, yi) = self.lineIntersection(x1, y1, x2, y2, X1, Y1, X2, Y2)
            if intersect:
                xData = [xi]; yData = [yi]
                for j in range(i+1, self.dataSize()-2): xData.append(self.x(j)); yData.append(self.y(j))
                xData.append(xi); yData.append(yi)
                self.setData(xData, yData)
                return 1
        return 0

    def lineIntersection(self, x1, y1, x2, y2, X1, Y1, X2, Y2):
        if min(x1,x2) > max(X1, X2) or max(x1,x2) < min(X1,X2): return (0, 0, 0)
        if min(y1,y2) > max(Y1, Y2) or max(y1,y2) < min(Y1,Y2): return (0, 0, 0)
        
        if x2-x1 != 0: k1 = (y2-y1)/(x2-x1)
        else:          k1 = 1e+12
        
        if X2-X1 != 0: k2 = (Y2-Y1)/(X2-X1)
        else:          k2 = 1e+12

        c1 = (y1-k1*x1)
        c2 = (Y1-k2*X1)

        if k1 == 1e+12:
            yTest = k2*x1 + c2
            if yTest > min(y1,y2) and yTest  < max(y1,y2): return (1, x1, yTest)
            else: return (0,0,0)

        if k2 == 1e+12:
            yTest = k1*X1 + c1
            if yTest > min(Y1,Y2) and yTest < max(Y1,Y2): return (1, X1, yTest)
            else: return (0,0,0)

        det_inv = 1/(k2 - k1)

        xi=((c1 - c2)*det_inv)
        yi=((k2*c1 - k1*c2)*det_inv)

        if xi >= min(x1, x2) and xi <= max(x1,x2) and xi >= min(X1, X2) and xi <= max(X1, X2) and yi >= min(y1,y2) and yi <= max(y1, y2) and yi >= min(Y1, Y2) and yi <= max(Y1, Y2):
            return (1, xi, yi)
        else:
            return (0, xi, yi)

# ####################################################################
# draw a rectangle
class subBarQwtPlotCurve(QwtPlotCurve):
    def __init__(self, parent = None, text = None):
        QwtPlotCurve.__init__(self, parent, text)
        self.color = Qt.black
        self.penColor = Qt.black

    def draw(self, p, xMap, yMap, f, t):
        # save ex settings
        back = p.backgroundMode()
        pen = p.pen()
        brush = p.brush()
        
        p.setBackgroundMode(Qt.OpaqueMode)
        p.setBackgroundColor(self.color)
        p.setBrush(self.color)
        p.setPen(self.penColor)
        
        if t < 0: t = self.dataSize() - 1
        if divmod(f, 2)[1] != 0: f -= 1
        if divmod(t, 2)[1] == 0:  t += 1
        for i in range(f, t+1, 2):
            px1 = xMap.transform(self.x(i))
            py1 = yMap.transform(self.y(i))
            px2 = xMap.transform(self.x(i+1))
            py2 = yMap.transform(self.y(i+1))
            p.drawRect(px1, py1, (px2 - px1), (py2 - py1))

        # restore ex settings
        p.setBackgroundMode(back)
        p.setPen(pen)
        p.setBrush(brush)


# ####################################################################
# create a marker in QwtPlot, that doesn't have a transparent background. Currently used in parallel coordinates widget.
class nonTransparentMarker(QwtPlotMarker):
    def __init__(self, backColor, *args):
        QwtPlotMarker.__init__(self, *args)
        self.backColor = backColor

    def draw(self, p, x, y, rect):
        p.setPen(self.labelPen())
        p.setFont(self.font())

        th = p.fontMetrics().height();
        tw = p.fontMetrics().width(self.label()); 
        r = QRect(x + 4, y - th/2 - 2, tw + 4, th + 4)
        p.fillRect(r, QBrush(self.backColor))
        p.drawText(r, Qt.AlignHCenter + Qt.AlignVCenter, self.label());
        

# ####################################################################
# 
class errorBarQwtPlotCurve(QwtPlotCurve):
    def __init__(self, parent = None, text = None, connectPoints = 0, tickXw = 0.1, tickYw = 0.1, showVerticalErrorBar = 1, showHorizontalErrorBar = 0):
        QwtPlotCurve.__init__(self, parent, text)
        self.connectPoints = connectPoints
        self.tickXw = tickXw
        self.tickYw = tickYw
        self.showVerticalErrorBar = showVerticalErrorBar
        self.showHorizontalErrorBar = showHorizontalErrorBar

    def draw(self, p, xMap, yMap, f, t):
        # save ex settings
        pen = p.pen()
        
        self.setPen( self.symbol().pen() )
        p.setPen( self.symbol().pen() )
        if self.style() == QwtCurve.UserCurve:
            back = p.backgroundMode()
            
            p.setBackgroundMode(Qt.OpaqueMode)
            if t < 0: t = self.dataSize() - 1

            if divmod(f, 3)[1] != 0: f -= f % 3
            if divmod(t, 3)[1] == 0:  t += 1
            first = 1
            for i in range(f, t+1, 3):
                px = xMap.transform(self.x(i))
                py = yMap.transform(self.y(i))

                if self.showVerticalErrorBar:
                    vbxl = xMap.transform(self.x(i) - self.tickXw/2.0)
                    vbxr = xMap.transform(self.x(i) + self.tickXw/2.0)

                    vbyt = yMap.transform(self.y(i + 1))
                    vbyb = yMap.transform(self.y(i + 2))

                if self.showHorizontalErrorBar:
                    hbxl = xMap.transform(self.x(i + 1))
                    hbxr = xMap.transform(self.x(i + 2))

                    hbyt = yMap.transform(self.y(i) + self.tickYw/2.0)
                    hbyb = yMap.transform(self.y(i) - self.tickYw/2.0)

                if self.connectPoints:
                    if first:
                        first = 0
                    else:
                        p.drawLine(ppx, ppy, px, py)
                    ppx = px
                    ppy = py

                if self.showVerticalErrorBar:
                    p.drawLine(px,   vbyt, px,   vbyb)   ## |
                    p.drawLine(vbxl, vbyt, vbxr, vbyt) ## T
                    p.drawLine(vbxl, vbyb, vbxr, vbyb) ## _

                if self.showHorizontalErrorBar:
                    p.drawLine(hbxl, py,   hbxr, py)   ## -
                    p.drawLine(hbxl, hbyt, hbxl, hbyb) ## |-
                    p.drawLine(hbxr, hbyt, hbxr, hbyb) ## -|

                self.symbol().draw(p, px, py)

            p.setBackgroundMode(back)
        else:
            QwtPlotCurve.draw(self, p, xMap, yMap, f, t)

        # restore ex settings
        p.setPen(pen)
        
# ####################################################################
# draw labels for discrete attributes
class DiscreteAxisScaleDraw(QwtScaleDraw):
    def __init__(self, labels):
        apply(QwtScaleDraw.__init__, (self,))
        self.labels = labels

    def label(self, value):
        index = int(round(value))
        if index != value: return ""    # if value not an integer value return ""
        if index >= len(self.labels) or index < 0: return ''
        return QString(str(self.labels[index]))

# ####################################################################
# use this class if you want to hide labels on the axis
class HiddenScaleDraw(QwtScaleDraw):
    def __init__(self, *args):
        QwtScaleDraw.__init__(self, *args)
        
    def label(self, value):
        return QString.null
