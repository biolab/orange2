# Nomogram visualization widget. It is used together with OWNomogram

from OWWidget import *
#from Numeric import *
import Numeric
import math
from qtcanvas import *
import time, statc

# constants
SE_Z = -100
HISTOGRAM_Z = -200
aproxZero = 0.0001

def norm_factor(p):
    max = 10.
    min = -10.
    z = 0.
    eps = 0.001
    while ((max-min)>eps):
        pval = statc.zprob(z)
        if pval>p:
            max = z
        if pval<p:
            min = z
        z = (max + min)/2.
    return z

def unique(lst):
    d = {}
    for item in lst:
        d[item] = None
    return d.keys()

# returns difference between continuous label values
def getDiff(d):
    if d < 1 and d>0:
        mnum = d/pow(10, math.floor(math.log10(d)))
    else:
        mnum = d
        
    if d==0:
        return 0
    if str(mnum)[0]>'4':
        return math.pow(10,math.floor(math.log10(d))+1)
    elif str(mnum)[0]<'2':
        return 2*math.pow(10,math.floor(math.log10(d)))
    else:
        return 5*math.pow(10,math.floor(math.log10(d)))


# Detailed description of selected value in attribute:
#     - shows its value on the upper scale (points, log odds, ...)
#     - its real value (example: age = 45) or in case of discrete attribute:
#     - shows proportional value between two (real,possible) values of attribute
class Descriptor(QCanvasRectangle):
    def __init__(self, canvas, attribute, z=60):
        apply(QCanvasRectangle.__init__,(self, canvas))
        self.canvas = canvas
        self.attribute = attribute
        self.setPen(QPen(Qt.black, 2))
        self.setBrush(QBrush(QColor(135,206,250)))
        
        self.splitLine = QCanvasLine(canvas)
        self.header = QCanvasText(canvas)
        self.header.setTextFlags(Qt.AlignLeft+Qt.AlignTop)
        self.headerValue = QCanvasText(canvas)
        self.headerValue.setTextFlags(Qt.AlignRight + Qt.AlignTop)
        self.valName = QCanvasText(canvas)
        self.valName.setTextFlags(Qt.AlignLeft+Qt.AlignTop)
        self.value = QCanvasText(canvas)
        self.value.setTextFlags(Qt.AlignRight + Qt.AlignTop)
        self.supportingValName = QCanvasText(canvas)
        self.supportingValName.setTextFlags(Qt.AlignLeft+Qt.AlignTop)
        self.supportingValue = QCanvasText(canvas)
        self.supportingValue.setTextFlags(Qt.AlignRight + Qt.AlignTop)
        self.setZAll(z)
       

    def drawAll(self, x, y):
        def getNearestAtt(selectedBeta):
            if isinstance(self.attribute, AttrLineCont):
                for i in range(len(self.attribute.attValues)):
                    if self.attribute.attValues[i].betaValue==selectedBeta:
                        nearestRight = self.attribute.attValues[i]
                        nearestLeft = self.attribute.attValues[i]
                        break
                    elif i>0 and self.attribute.attValues[i].betaValue>selectedBeta and selectedBeta>self.attribute.attValues[i-1].betaValue:
                        nearestRight = self.attribute.attValues[i]
                        nearestLeft = self.attribute.attValues[i-1]
                        break
                    elif i>0 and self.attribute.attValues[i].betaValue<selectedBeta and selectedBeta<self.attribute.attValues[i-1].betaValue:
                        nearestRight = self.attribute.attValues[i-1]
                        nearestLeft = self.attribute.attValues[i]
                        break
                    elif i == len(self.attribute.attValues)-1:
                        nearestRight = self.attribute.attValues[i]
                        nearestLeft = self.attribute.attValues[i-1]
                        break
                        
            else:
                nearestLeft = filter(lambda x: x.betaValue == max([at.betaValue for at in filter(lambda x: x.betaValue <= selectedBeta, self.attribute.attValues)]) ,self.attribute.attValues)[0]
                nearestRight = filter(lambda x: x.betaValue == min([at.betaValue for at in filter(lambda x: x.betaValue >= selectedBeta, self.attribute.attValues)]) ,self.attribute.attValues)[0]
            return (nearestLeft, nearestRight)
        
        # I need mapper to calculate various quantities (said in chemistry way) from the attribute and its selected value
        # happens at the time of drawing header and footer canvases
        # x and y should be on canvas!
        
        if ((not isinstance(self.canvas, BasicNomogram) or not self.canvas.onCanvas(x,y)) and
            (not isinstance(self.canvas, BasicNomogramFooter) or not self.canvas.onCanvas(x,y))):
            return True

        if isinstance(self.canvas, BasicNomogramFooter) and self.canvas.onCanvas(x,y):
            self.header.setText(self.attribute.name)
            self.headerValue.setText("")
            self.valName.setText("Value:")
            if self.attribute.selectedValue:
                self.value.setText(str(round(self.attribute.selectedValue[2],2)))
            else:
                self.value.setText("None")
            self.supportingValName.setText("")
            self.supportingValue.setText("")
            points = 1
        else:
            # get points
            selectedBeta = self.attribute.selectedValue[2]
            proportionalBeta = self.canvas.mapper.propBeta(selectedBeta, self.attribute)
            maxValue = self.canvas.mapper.getMaxMapperValue()
            minValue = self.canvas.mapper.getMinMapperValue()
            points = minValue+(maxValue-minValue)*proportionalBeta

            self.header.setText(self.canvas.parent.pointsName[self.canvas.parent.yAxis]+":")
            self.headerValue.setText(str(round(points,2)))
            

            # continuous? --> get attribute value
            if isinstance(self.attribute, AttrLineCont):
                self.valName.setText("Value:")
                if len(self.attribute.selectedValue)==4:
                    self.value.setText(str(round(self.attribute.selectedValue[3],2)))
                else:
                    (nleft, nright) = getNearestAtt(selectedBeta)
                    if nright.betaValue>nleft.betaValue:
                        prop = (selectedBeta-nleft.betaValue)/(nright.betaValue-nleft.betaValue)
                    else:
                        prop = 0
                    if prop == 0:
                        avgValue = (float(nleft.name)+float(nright.name))/2.
                    else:
                        avgValue = float(nleft.name)+prop*(float(nright.name)-float(nleft.name))
                    self.value.setText(str(round(avgValue,2)))
                self.supportingValName.setText("")
                self.supportingValue.setText("")            
            # discrete? --> get left and right value, proportional select values
            else:
                (nleft, nright) = getNearestAtt(selectedBeta)
                if nright.betaValue>nleft.betaValue:
                    prop = (selectedBeta-nleft.betaValue)/(nright.betaValue-nleft.betaValue)
                else:
                    prop = 0
                if prop == 0 or prop == 1:
                    self.valName.setText("Value:")
                    self.supportingValName.setText("")
                    self.supportingValue.setText("")            
                    if prop == 0:
                        self.value.setText(nleft.name)
                    else:
                        self.value.setText(nright.name)
                else:
                    self.valName.setText(nleft.name + ":")
                    self.supportingValName.setText(nright.name + ":")
                    self.value.setText(str(round(1-prop,4)*100)+"%")
                    self.supportingValue.setText(str(round(prop,4)*100)+"%")

        # set height
        height = 15+ self.valName.boundingRect().height() + self.header.boundingRect().height()
        if self.supportingValName.text() != "":
            height+= self.supportingValName.boundingRect().height()

        # set width
        width = 20+max([self.header.boundingRect().width()+2+self.headerValue.boundingRect().width(),
                  self.valName.boundingRect().width()+2+self.value.boundingRect().width(), 
                  self.supportingValName.boundingRect().width()+2+self.supportingValue.boundingRect().width()])

        # if bubble wants to jump of the canvas, better catch it !
        selOffset = 20
        xTemp, yTemp = x+selOffset, y-selOffset-height
        while not self.canvas.onCanvas(xTemp,yTemp) or not self.canvas.onCanvas(xTemp,yTemp+height) or not self.canvas.onCanvas(xTemp+width,yTemp) or not self.canvas.onCanvas(xTemp+width,yTemp+height):
            if yTemp == y-selOffset-height and not xTemp <= x-selOffset-width:
                xTemp-=1
            elif xTemp <= x-selOffset-width and not yTemp >= y+selOffset:
                yTemp+=1
            elif yTemp >= y+selOffset and not xTemp >= x+selOffset:
                xTemp+=1
            elif xTemp>= x+selOffset and not yTemp<y-selOffset-height+2:
                yTemp-=1
            else:
                break

        x,y = xTemp, yTemp
        
        # set coordinates
        self.setX(x)
        self.setY(y)
        self.setSize(width+2, height+2)
        
        # header
        self.header.setX(x+2)
        self.header.setY(y+2)
        self.headerValue.setX(x+width-4)
        self.headerValue.setY(y+2)

        #line
        self.splitLine.setPoints(x, y+4+self.header.boundingRect().height(), x+width, y+4+self.header.boundingRect().height())

        # values        
        self.valName.setX(x+3)
        self.valName.setY(y+7+self.header.boundingRect().height())
        self.value.setX(x+width-4)
        self.value.setY(y+7+self.header.boundingRect().height())
        self.supportingValName.setX(x+3)
        self.supportingValName.setY(y+10+self.header.boundingRect().height()+self.valName.boundingRect().height())
        self.supportingValue.setX(x+width-4)
        self.supportingValue.setY(y+10+self.header.boundingRect().height()+self.valName.boundingRect().height())

        #return false if position is at zero and alignment is centered
        if round(points,3) == 0.0 and self.canvas.parent.alignType == 1:
            return False
        return True
        
                

    def setZAll(self, z):
        self.setZ(z)
        self.header.setZ(z+1)
        self.splitLine.setZ(z+1)
        self.headerValue.setZ(z+1)
        self.valName.setZ(z+1)
        self.value.setZ(z+1)
        self.supportingValName.setZ(z+1)
        self.supportingValue.setZ(z+1)
        
    def showAll(self):
        self.show()
        self.splitLine.show()
        self.header.show()
        self.headerValue.show()
        self.valName.show()
        self.value.show()
        if self.supportingValName.text != "":
            self.supportingValName.show()
            self.supportingValue.show()
        
    def hideAll(self):
        self.hide()
        self.header.hide()
        self.splitLine.hide()
        self.headerValue.hide()
        self.valName.hide()
        self.value.hide()
        self.supportingValName.hide()
        self.supportingValue.hide()


# Attribute value selector -- a small circle
class AttValueMarker(QCanvasEllipse):
    def __init__(self, attribute, canvas, z=50):
        apply(QCanvasEllipse.__init__,(self,10,10,canvas))
        self.attribute = attribute
        #self.canvas = canvas
        self.setZ(z)
        self.setBrush(QBrush(Qt.blue))
        self.name=""
        #self.borderCircle = QCanvasEllipse(15,15,canvas)
        #self.borderCircle.setBrush(QBrush(Qt.red))
        #self.borderCircle.setZ(z-1)
        self.descriptor = Descriptor(canvas, attribute, z+1)

    def setPos(self, x, y):
        self.setX(x)
        self.setY(y)
        #self.borderCircle.setX(x)
        #self.borderCircle.setY(y)
        if not self.descriptor.drawAll(x,y):
            brush = QBrush(self.brush().color(), Qt.Dense4Pattern)
            #brush.setStyle()
            self.setBrush(brush)
        else:
            self.setBrush(QBrush(self.brush().color()))

    def showSelected(self):
        #self.borderCircle.show()
        self.setBrush(QBrush(QColor(253,151,51), self.brush().style()))
        if self.canvas().parent.bubble:
            self.descriptor.showAll()
        
    def hideSelected(self):
        #self.borderCircle.hide()
        self.setBrush(QBrush(Qt.blue, self.brush().style()))
        self.descriptor.hideAll()



# #################################################################### 
# Single Attribute Value
# ####################################################################
class AttValue:
    def __init__(self, name, betaValue, error=0, showErr=False, over=True, lineWidth = 0, markerWidth = 2, enable = True):
        self.name = name
        self.betaValue = betaValue
        self.error = error
        self.showErr = showErr
        self.enable = enable
        self.hideAtValue = False
        self.over = over
        self.lineWidth = lineWidth
        self.markerWidth = markerWidth
        self.attCreation = True # flag shows that vanvas object have to be created first

    def destroy(self):
        if not self.attCreation:
            self.hide()

    def setCreation(self, canvas):
        self.text = QCanvasText(self.name, canvas)
        self.text.setTextFlags(Qt.AlignCenter)
        self.labelMarker = QCanvasLine(canvas)
        self.labelMarker.setPen(QPen(Qt.black, self.markerWidth))
        self.histogram = QCanvasLine(canvas)
        self.histogram.setZ(HISTOGRAM_Z)
        self.histogram.setPen(QPen(QColor(140,140,140),7))
        self.errorLine = QCanvasLine(canvas)
        self.errorLine.setPen(QPen(QColor(25,25,255),1))
        self.errorLine.setZ(SE_Z)
        self.attCreation = False

    def hide(self):
        self.text.hide()
        self.labelMarker.hide()
        self.errorLine.hide()
            
    def paint(self, canvas, rect, mapper):
        def errorCollision(line,z=SE_Z):
            col = filter(lambda x:x.z()==z,line.collisions(True))
            if len(col)>0:
                return True
            return False

        if self.attCreation:
            self.setCreation(canvas)
        self.text.setX(self.x)
        if self.enable:
            lineLength = canvas.fontSize/2
            canvasLength = 0
            if canvas.parent.histogram and isinstance(canvas, BasicNomogram):
                canvasLength = 2+self.lineWidth*canvas.parent.histogram_size
            if self.over:
                self.text.setY(rect.bottom()-4*canvas.fontSize/3)
                self.labelMarker.setPoints(self.x, rect.bottom(), self.x, rect.bottom()+lineLength)
                self.histogram.setPoints(self.x, rect.bottom(), self.x, rect.bottom()+canvasLength)
            else:
                self.text.setY(rect.bottom()+4*canvas.fontSize/3)
                self.labelMarker.setPoints(self.x, rect.bottom(), self.x, rect.bottom()-lineLength)
                self.histogram.setPoints(self.x, rect.bottom(), self.x, rect.bottom()-canvasLength)
            if not self.hideAtValue:
                self.text.show()
            else:
                self.text.hide()
            if canvas.parent.histogram:
                self.histogram.show()
#            else:
#                self.histogram.hide()
        # if value is disabled, draw just a symbolic line        
        else:
            self.labelMarker.setPoints(self.x, rect.bottom(), self.x, rect.bottom()+canvas.fontSize/4)
            self.text.hide()

        # show confidence interval
        if self.showErr:
            self.low_errorX = max(self.low_errorX, 0)
            self.high_errorX = min(self.high_errorX, canvas.size().width())
            if self.low_errorX == 0 and self.high_errorX == canvas.size().width():
                self.errorLine.setPen(QPen(self.errorLine.pen().color(),self.errorLine.pen().width(),Qt.DotLine))
            else:
                self.errorLine.setPen(QPen(self.errorLine.pen().color(), self.errorLine.pen().width()))
            
            if self.over:
                add = 2
                n=0
                self.errorLine.setPoints(self.low_errorX, rect.bottom()+add, self.high_errorX , rect.bottom()+add)
                while errorCollision(self.errorLine):
                    n=n+1
                    if add>0:
                        add = -add
                    else:
                        add =  -add + 2
                    self.errorLine.setPoints(self.low_errorX, rect.bottom()+add, self.high_errorX , rect.bottom()+add)
            else:
                add = -2
                self.errorLine.setPoints(self.low_errorX, rect.bottom()+add, self.high_errorX , rect.bottom()+add)
                while errorCollision(self.errorLine):
                    if add<0:
                        add = -add
                    else:
                        add = -add - 2
                    self.errorLine.setPoints(self.low_errorX, rect.bottom()+add, self.high_errorX , rect.bottom()+add)
            self.errorLine.show()
        self.labelMarker.show()        

    def toString(self):
        return self.name, "beta =", self.betaValue


# #################################################################### 
# Normal attribute - 1d
# ####################################################################
# This is a base class for representing all different possible attributes in nomogram.
# Use it only for discrete/non-ordered values
class AttrLine:
    def __init__(self, name, canvas):
        self.name = name
        self.attValues = []
        self.minValue = self.maxValue = 0
        self.selectedValue = None
        self.initialize(canvas)

    def addAttValue(self, attValue):
        if len(self.attValues)==0:
            self.minValue = attValue.betaValue
            self.maxValue = attValue.betaValue
        else:
            self.minValue = min(self.minValue, attValue.betaValue)
            self.maxValue = max(self.maxValue, attValue.betaValue)
        self.attValues.append(attValue)

    def getHeight(self, canvas):      
        return canvas.parent.verticalSpacing

    # Find the closest (selectable) point to mouse-clicked one.
    def updateValueXY(self, x, y):
        oldSelect = self.selectedValue
        minXDiff = 50
        minYDiff = 50
        minAbs = 100
        for xyCanvas in self.selectValues:
            if (abs(x-xyCanvas[0]) + abs(y-xyCanvas[1]))<minAbs:
                self.selectedValue = xyCanvas
                minYDiff = abs(y-xyCanvas[1])
                minXDiff = abs(x-xyCanvas[0])
                minAbs = minYDiff + minXDiff
        if oldSelect == self.selectedValue:
            return False
        else:
            self.marker.setPos(self.selectedValue[0], self.selectedValue[1])
            return True    

    # Update position of the marker!
    # This is usualy necessary after changing types of nomogram, for example left-aligned to center-aligned.
    # In this situations selected beta should stay the same, but x an y of the marker must change!
    def updateValue(self):
        if not self.selectedValue:
            return
        beta = self.selectedValue[2]
        minBetaDiff = 1
        for xyCanvas in self.selectValues:
            if abs(beta-xyCanvas[2])<minBetaDiff:
                self.selectedValue = xyCanvas
                minBetaDiff = abs(beta-xyCanvas[2])
        self.marker.setPos(self.selectedValue[0], self.selectedValue[1])

    def initialize(self, canvas):
        self.label = QCanvasText(canvas)
        self.label.setText(self.name)
        font = QFont(self.label.font())
        font.setBold(True)
        self.label.setFont(font)         # draw label in bold
        self.line = QCanvasLine(canvas)

        # create blue probability marker
        self.marker = AttValueMarker(self, canvas, 50)

    def drawAttributeLine(self, canvas, rect, mapper):
        atValues_mapped, atErrors_mapped, min_mapped, max_mapped = mapper(self, error_factor = norm_factor(1-((1-float(canvas.parent.confidence_percent)/100.)/2.))) # return mapped values, errors, min, max --> mapper(self)
        self.label.setX(1)
        self.label.setY(rect.bottom()-canvas.fontSize)

        # draw attribute line
        self.line.setPoints(min_mapped, rect.bottom(), max_mapped, rect.bottom())
        zero = 0
        if len([at.betaValue for at in self.attValues]) == 0:
            return
        if min([at.betaValue for at in self.attValues])>0:
            zero = min([at.betaValue for at in self.attValues])
        if max([at.betaValue for at in self.attValues])<0:
            zero = max([at.betaValue for at in self.attValues])
        self.selectValues = [[mapper.mapBeta(zero, self), rect.bottom(), zero]]
        if not self.selectedValue:
            self.selectedValue = self.selectValues[0]
        
    def paint(self, canvas, rect, mapper):
        self.label.setText(self.name)
        atValues_mapped, atErrors_mapped, min_mapped, max_mapped = mapper(self, error_factor = norm_factor(1-((1-float(canvas.parent.confidence_percent)/100.)/2.))) # return mapped values, errors, min, max --> mapper(self)

        self.drawAttributeLine(canvas, rect, mapper)        
        # draw attributes
        val = self.attValues

        # draw values
        for i in range(len(val)):
            # check attribute name that will not cover another name
            val[i].x = atValues_mapped[i]
            val[i].high_errorX = atErrors_mapped[i][1]
            val[i].low_errorX = atErrors_mapped[i][0]
            a = time.time()
            if canvas.parent.confidence_check and val[i].error>0:
                val[i].showErr = True
            else:
                val[i].showErr = False

            val[i].hideAtValue = False                
            val[i].over = True
            val[i].paint(canvas, rect, mapper)

            #find suitable value position
            for j in range(i):
                #if val[j].over and val[j].enable and abs(atValues_mapped[j]-atValues_mapped[i])<(len(val[j].name)*canvas.fontSize/4+len(val[i].name)*canvas.fontSize/4):
                if val[j].over and val[j].enable and not val[j].hideAtValue and val[j].text.collidesWith(val[i].text):
                    val[i].over = False
            if not val[i].over:
                val[i].paint(canvas, rect, mapper)
                for j in range(i):
                    if not val[j].over and val[j].enable and not val[j].hideAtValue and val[j].text.collidesWith(val[i].text):
                        val[i].hideAtValue = True
                if val[i].hideAtValue:
                    val[i].paint(canvas, rect, mapper)
            self.selectValues.append([atValues_mapped[i], rect.bottom(), val[i].betaValue])
            
        atLine = AttrLine("marker", canvas)
        d = 5*(self.maxValue-self.minValue)/max((max_mapped-min_mapped),aproxZero)
        for xc in Numeric.arange(self.minValue, self.maxValue+d, d):
            atLine.addAttValue(AttValue("", xc))
        
        markers_mapped, mark_errors_mapped, markMin_mapped, markMax_mapped = mapper(atLine)
        for mar in range(len(markers_mapped)):
            xVal = markers_mapped[mar]
            if filter(lambda x: abs(x[0]-xVal)<4, self.selectValues) == [] and xVal<max_mapped:
                self.selectValues.append([xVal, rect.bottom(), atLine.attValues[mar].betaValue])

        self.updateValue()
        if max_mapped - min_mapped > 5.0:
            self.line.show()
        self.label.show()

    # some supplementary methods for 2d presentation
    # draw bounding box around cont. attribute
    def drawBox(self, min_mapped, max_mapped, rect):
        # draw box
        self.box.setX(min_mapped)
        self.box.setY(rect.top()+rect.height()/8)
        self.box.setSize(max_mapped-min_mapped, rect.height()*7/8)

        # show att. name
        self.label.setText(self.name)
        self.label.setX(min_mapped)
        self.label.setY(rect.top()+rect.height()/8)

    # draws a vertical legend on the left side of the bounding box
    def drawVerticalLabel(self, attLineLabel, min_mapped, mapped_labels, canvas):
        for at in range(len(attLineLabel.attValues)):
            # draw value
            a = self.contLabel[at]
            a.setX(min_mapped-5)
            a.setY(mapped_labels[at]-canvas.fontSize/2)
            if attLineLabel.attValues[at].enable:
                a.marker.setPoints(min_mapped-2, mapped_labels[at], min_mapped+2, mapped_labels[at])
                a.show()
            # if value is disabled, draw just a symbolic line        
            else:
                a.marker.setPoints(min_mapped-1, mapped_labels[at], min_mapped+1, mapped_labels[at])                
            a.marker.show()

    #finds proportionally where zero value is
    def findZeroValue(self):
        maxPos,zero = 1,0
        while self.attValues[maxPos].betaValue!=0 and self.attValues[maxPos-1].betaValue!=0 and self.attValues[maxPos].betaValue/abs(self.attValues[maxPos].betaValue) == self.attValues[maxPos-1].betaValue/abs(self.attValues[maxPos-1].betaValue):
            maxPos+=1
            if maxPos == len(self.attValues):
                maxPos-=1
                zero = self.attValues[maxPos].betaValue
                break
        if not self.attValues[maxPos].betaValue == self.attValues[maxPos-1].betaValue:
            return ((zero-self.attValues[maxPos-1].betaValue)/(self.attValues[maxPos].betaValue - self.attValues[maxPos-1].betaValue),maxPos,zero)
        else:
            return (zero,maxPos,zero)


    # string representation of attribute
    def toString(self):
        return self.name + str([at.toString() for at in self.attValues])        
        


# #################################################################### 
# Continuous attribute in 2d
# ####################################################################
class AttrLineCont(AttrLine):
    def __init__(self, name, canvas):
        AttrLine.__init__(self, name, canvas)

        # continuous attributes        
        self.cAtt = None
        self.box = QCanvasRectangle(canvas)
        self.box.setPen(QPen(Qt.DotLine))
        self.contValues = []
        self.contLabel = []
    
    def getHeight(self, canvas):
        if canvas.parent.contType == 1:
            return canvas.parent.verticalSpacingContinuous
        return AttrLine.getHeight(self, canvas)

    # initialization before 2d paint
    def initializeBeforePaint(self, canvas):
        self.atNames = AttrLine(self.name, canvas)
        for at in self.attValues:
            self.atNames.addAttValue(AttValue(at.name, float(at.name)))
        verticalRect = QRect(0, 0, self.getHeight(canvas), self.getHeight(canvas))
        verticalMapper = Mapper_Linear_Fixed(self.atNames.minValue, self.atNames.maxValue, verticalRect.left()+verticalRect.width()/4, verticalRect.right(), maxLinearValue = self.atNames.maxValue, minLinearValue = self.atNames.minValue)
        label = verticalMapper.getHeaderLine(canvas, QRect(0,0,self.getHeight(canvas), self.getHeight(canvas))) 
        [l.setCanvas(None) for l in self.contLabel]
        self.contLabel=[]
        for val in label.attValues:
            # draw value
            a = QCanvasText(val.name, canvas)
            a.setTextFlags(Qt.AlignRight)
            a.marker = QCanvasLine(canvas)
            a.marker.setZ(5)
            self.contLabel.append(a)

        #line objects
        if len(self.contValues) == 0:
            for at in self.attValues:
                a = QCanvasLine(canvas)
                a.upperSE = QCanvasLine(canvas)
                a.lowerSE = QCanvasLine(canvas)
                a.setPen(QPen(Qt.black, at.lineWidth))
                a.upperSE.setPen(QPen(Qt.blue, 0))
                a.lowerSE.setPen(QPen(Qt.blue, 0))
                self.contValues.append(a)

                
                # for 1d cont space
                at.setCreation(canvas)

        
    def paint(self, canvas, rect, mapper):
        self.label.setText(self.name)
        atValues_mapped, atErrors_mapped, min_mapped, max_mapped = mapper(self, error_factor = norm_factor(1-((1-float(canvas.parent.confidence_percent)/100.)/2.))) # return mapped values, errors, min, max --> mapper(self)

        self.drawAttributeLine(canvas, rect, mapper)  
        
        # continuous attributes are handled differently
        self.cAtt = self.shrinkSize(canvas, rect, mapper)
        atValues_mapped, atErrors_mapped, min_mapped, max_mapped = mapper(self.cAtt) # return mapped values, errors, min, max --> mapper(self)
        val = self.cAtt.attValues
        for i in range(len(val)):
            # check attribute name that will not cover another name
            val[i].x = atValues_mapped[i]
            val[i].paint(canvas, rect, mapper)
            for j in range(i):
                if val[j].over==val[i].over and val[j].enable and val[i].text.collidesWith(val[j].text):
                    val[i].enable = False
            if not val[i].enable:
                val[i].paint(canvas, rect, mapper)
            self.selectValues.append([atValues_mapped[i], rect.bottom(), val[i].betaValue])


        atLine = AttrLine("marker", canvas)
        d = 5*(self.cAtt.maxValue-self.cAtt.minValue)/max(max_mapped-min_mapped,aproxZero)
        for xc in Numeric.arange(self.cAtt.minValue, self.cAtt.maxValue+d, d):
            atLine.addAttValue(AttValue("", xc))
        
        markers_mapped, mark_errors_mapped, markMin_mapped, markMax_mapped = mapper(atLine)
        for mar in range(len(markers_mapped)):
            xVal = markers_mapped[mar]
            if filter(lambda x: abs(x[0]-xVal)<4, self.selectValues) == [] and xVal<max_mapped:
                self.selectValues.append([xVal, rect.bottom(), atLine.attValues[mar].betaValue])

        self.updateValue()
        self.line.show()
        self.label.show()


    # create an AttrLine object from a continuous variable (to many values for a efficient presentation)
    def shrinkSize(self, canvas, rect, mapper):
        # get monotone subset of this continuous variable
        monotone_subsets, curr_subset = [],[]
        sign=1
        for at_i in range(len(self.attValues)):
            if at_i<len(self.attValues)-1 and sign*(self.attValues[at_i].betaValue-self.attValues[at_i+1].betaValue)>0:
                curr_subset.append(self.attValues[at_i])
                monotone_subsets.append(curr_subset)
                curr_subset = [self.attValues[at_i]]
                sign=-sign
            else:
                curr_subset.append(self.attValues[at_i])
        monotone_subsets.append(curr_subset)                
            
        # create retAttr --> values in between can be easily calculated from first left and first right
        retAttr = AttrLine(self.name, canvas)
        for at in self.attValues:
            if at.betaValue == self.minValue or at.betaValue == self.maxValue:
                at.enable = True
                retAttr.addAttValue(at)
        curr_over = False

        # convert monotone subsets to nice step-presentation        
        for m in monotone_subsets:
            if len(m)<2:
                continue
            curr_over = not curr_over
            maxValue = max(float(m[0].name), float(m[len(m)-1].name))
            minValue = min(float(m[0].name), float(m[len(m)-1].name))
            width = mapper.mapBeta(max(m[0].betaValue, m[len(m)-1].betaValue),self) - mapper.mapBeta(min(m[0].betaValue, m[len(m)-1].betaValue),self)
            curr_rect = QRect(rect.left(), rect.top(), width, rect.height())
            mapperCurr = Mapper_Linear_Fixed(minValue, maxValue, curr_rect.left(), curr_rect.right(), maxLinearValue = maxValue, minLinearValue = minValue)
            label = mapperCurr.getHeaderLine(canvas, curr_rect)
            for at in label.attValues:
                if at.betaValue>=minValue and at.betaValue<=maxValue:
                    for i in range(len(m)):
                        if i<(len(m)-1):
                            if float(m[i].name)<=at.betaValue and float(m[i+1].name)>=at.betaValue:
                                coeff = (at.betaValue-float(m[i].name))/max(float(m[i+1].name)-float(m[i].name),aproxZero)
                                retAttr.addAttValue(AttValue(str(at.betaValue),m[i].betaValue+coeff*(m[i+1].betaValue-m[i].betaValue)))
                                retAttr.attValues[len(retAttr.attValues)-1].over = curr_over

        return retAttr

    def paint2d(self, canvas, rect, mapper):
        self.initializeBeforePaint(canvas)
        self.label.setText(self.name)

        # get all values tranfsormed with current mapper 
        atValues_mapped, atErrors_mapped, min_mapped, max_mapped = mapper(self, error_factor = norm_factor(1-((1-float(canvas.parent.confidence_percent)/100.)/2.))) # return mapped values, errors, min, max --> mapper(self)

        # draw a bounding box
        self.drawBox(min_mapped, max_mapped, rect)

        #draw legend from real values
        verticalRect = QRect(rect.top(), rect.left(), rect.height(), rect.width())
        verticalMapper = Mapper_Linear_Fixed(self.atNames.minValue, self.atNames.maxValue, verticalRect.left()+verticalRect.width()/4, verticalRect.right(), maxLinearValue = self.atNames.maxValue, minLinearValue = self.atNames.minValue, inverse=True)
        label = verticalMapper.getHeaderLine(canvas, verticalRect) 
        mapped_labels, error, min_lab, max_lab = verticalMapper(label) # return mapped values, errors, min, max --> mapper(self)        
        self.drawVerticalLabel(label, min_mapped, mapped_labels, canvas)            
            
        #create a vertical mapper
        atValues_mapped_vertical, atErrors_mapped_vertical, min_mapped_vertical, max_mapped_vertical = verticalMapper(self.atNames) # return mapped values, errors, min, max --> mapper(self)
            

        #find and select zero value (beta = 0)
        (propBeta,maxPos,zero) = self.findZeroValue()
        zeroValue = float(self.attValues[maxPos-1].name) + propBeta*(float(self.attValues[maxPos].name) - float(self.attValues[maxPos-1].name))
        self.selectValues = [[mapper.mapBeta(zero, self),verticalMapper.mapBeta(zeroValue, self.atNames), zero, zeroValue]]
            
        if not self.selectedValue:
            self.selectedValue = self.selectValues[0]            


        # draw lines
        for i in range(len(atValues_mapped)-1):
            a = self.contValues[i]
            if canvas.parent.histogram:
                a.setPen(QPen(Qt.black, 1+self.attValues[i].lineWidth*canvas.parent.histogram_size))
            else:
                a.setPen(QPen(Qt.black, 2))
            #if self.attValues[i].lineWidth>0:
            a.setPoints(atValues_mapped[i], atValues_mapped_vertical[i], atValues_mapped[i+1], atValues_mapped_vertical[i+1])
            a.upperSE.setPoints(atErrors_mapped[i][0], atValues_mapped_vertical[i], atErrors_mapped[i+1][0], atValues_mapped_vertical[i+1])
            a.lowerSE.setPoints(atErrors_mapped[i][1], atValues_mapped_vertical[i], atErrors_mapped[i+1][1], atValues_mapped_vertical[i+1])
            self.selectValues.append([atValues_mapped[i],atValues_mapped_vertical[i], self.attValues[i].betaValue, self.atNames.attValues[i].betaValue])


            # if distance between i and i+1 is large, add some select values.
            n = int(math.sqrt(math.pow(atValues_mapped[i+1]-atValues_mapped[i],2)+math.pow(atValues_mapped_vertical[i+1]-atValues_mapped_vertical[i],2)))/5-1
            self.selectValues = self.selectValues + [[atValues_mapped[i]+(float(j+1)/float(n+1))*(atValues_mapped[i+1]-atValues_mapped[i]),
                                          atValues_mapped_vertical[i]+(float(j+1)/float(n+1))*(atValues_mapped_vertical[i+1]-atValues_mapped_vertical[i]),
                                          self.attValues[i].betaValue+(float(j+1)/float(n+1))*(self.attValues[i+1].betaValue-self.attValues[i].betaValue),
                                          self.atNames.attValues[i].betaValue+(float(j+1)/float(n+1))*(self.atNames.attValues[i+1].betaValue-self.atNames.attValues[i].betaValue)] for j in range(n)]
            a.show()

            if canvas.parent.confidence_check:            
                a.upperSE.show()
                a.lowerSE.show()
            else:
                a.upperSE.hide()
                a.lowerSE.hide()
                
        self.updateValue()
        self.box.show()
        self.label.show()



# #################################################################### 
# Ordered attribute in 2d
# ####################################################################
class AttrLineOrdered(AttrLine):
    def __init__(self, name, canvas):
        AttrLine.__init__(self, name, canvas)

        # continuous attributes        
        self.box = QCanvasRectangle(canvas)
        self.box.setPen(QPen(Qt.DotLine))
        self.contValues = []
        self.contLabel = []
    
    def getHeight(self, canvas):
        if canvas.parent.contType == 1:
            return len(self.attValues)*canvas.parent.diff_between_ordinal+canvas.parent.diff_between_ordinal
        return AttrLine.getHeight(self, canvas)


    # initialization before 2d paint
    def initializeBeforePaint(self, canvas):
        [l.setCanvas(None) for l in self.contLabel]
        self.contLabel=[]
        for val in self.attValues:
            # draw value
            a = QCanvasText(val.name, canvas)
            a.setTextFlags(Qt.AlignRight)
            a.marker = QCanvasLine(canvas)
            a.marker.setZ(5)
            self.contLabel.append(a)

        #line objects
        if len(self.contValues) == 0:
            for at in self.attValues:
                a = QCanvasLine(canvas)
                a.setPen(QPen(Qt.black, at.lineWidth))
                self.contValues.append(a)

                # for 1d cont space
                at.setCreation(canvas)


    def getVerticalCoordinates(self, rect, val):
        return rect.bottom() - val.verticalDistance

    def paint2d_fixedDistance(self, canvas, rect, mapper):
        d = canvas.parent.diff_between_ordinal/2
        for at in self.attValues:
            at.verticalDistance = d
            d += canvas.parent.diff_between_ordinal

        # get all values tranfsormed with current mapper 
        atValues_mapped, atErrors_mapped, min_mapped, max_mapped = mapper(self) # return mapped values, errors, min, max --> mapper(self)
            
        mapped_labels = [self.getVerticalCoordinates(rect,v)-canvas.fontSize/2 for v in self.attValues]
        self.drawVerticalLabel(self, min_mapped, mapped_labels, canvas)
            
        #find and select zero value (beta = 0)
        (propBeta,maxPos,zero) = self.findZeroValue()
        self.selectValues = [[mapper.mapBeta(zero, self),self.getVerticalCoordinates(rect, self.attValues[maxPos-1]), zero]]                
            
        if not self.selectedValue:
            self.selectedValue = self.selectValues[0]            

        # draw lines
        for i in range(len(atValues_mapped)):
            a = self.contValues[i]
            if canvas.parent.histogram:
                a.setPen(QPen(Qt.black, 1+self.attValues[i].lineWidth*canvas.parent.histogram_size))
            else:
                a.setPen(QPen(Qt.black, 2))
            a.setPoints(atValues_mapped[i], self.getVerticalCoordinates(rect, self.attValues[i])-canvas.parent.diff_between_ordinal/2, atValues_mapped[i], self.getVerticalCoordinates(rect, self.attValues[i])+canvas.parent.diff_between_ordinal/2)
            self.selectValues.append([atValues_mapped[i],self.getVerticalCoordinates(rect, self.attValues[i]), self.attValues[i].betaValue])
            if i < len(atValues_mapped)-1:
                a.connection = QCanvasLine(canvas)
                a.connection.setPen(QPen(Qt.black, 1))
                a.connection.setPoints(atValues_mapped[i],
                                       self.getVerticalCoordinates(rect, self.attValues[i])-canvas.parent.diff_between_ordinal/2,
                                       atValues_mapped[i+1],
                                       self.getVerticalCoordinates(rect, self.attValues[i])-canvas.parent.diff_between_ordinal/2)
                a.connection.setPen(QPen(Qt.DotLine))
                a.connection.show()
            # if distance between i and i+1 is large, add some select values.
            x1 = atValues_mapped[i]
            y1 = self.getVerticalCoordinates(rect, self.attValues[i])-canvas.parent.diff_between_ordinal/2
            x2 = atValues_mapped[i]
            y2 = self.getVerticalCoordinates(rect, self.attValues[i])+canvas.parent.diff_between_ordinal/2
            
            n = int(y2-y1)/5-1
            self.selectValues = self.selectValues + [[x1, y1+(float(j+1)/float(n+1))*(y2-y1), self.attValues[i].betaValue] for j in range(n)]
            a.show()
        
    
    def paint2d(self, canvas, rect, mapper):
        self.initializeBeforePaint(canvas)

        # get all values tranfsormed with current mapper 
        atValues_mapped, atErrors_mapped, min_mapped, max_mapped = mapper(self) # return mapped values, errors, min, max --> mapper(self)

        # draw a bounding box
        self.drawBox(min_mapped, max_mapped+1, rect)

        # if fixedDistance:
        self.paint2d_fixedDistance(canvas, rect, mapper)
        

        self.updateValue()
        self.box.show()
        self.label.show()


# #################################################################### 
# Header CANVAS
# ####################################################################
class BasicNomogramHeader(QCanvas):
    def __init__(self, nomogram, parent):
        apply(QCanvas.__init__,(self, parent, ""))
        self.fontSize = parent.fontSize
        self.headerAttrLine = None
        self.nomogram = nomogram
        self.parent = parent
       
    def paintHeader(self, rect, mapper):
        #if self.headerAttrLine:
        #    self.headerAttrLine.destroy()
        [item.setCanvas(None) for item in self.allItems()]

        self.headerAttrLine = mapper.getHeaderLine(self, rect)
        self.headerAttrLine.name = self.nomogram.parent.pointsName[self.nomogram.parent.yAxis]
        self.headerAttrLine.paint(self, rect, mapper)
        self.resize(self.nomogram.pright, rect.height()+16)
        self.update()


# #################################################################### 
# FOOTER CANVAS, sum and probability 
# ####################################################################
class BasicNomogramFooter(QCanvas):
    def __init__(self, nomogram, parent):
        apply(QCanvas.__init__,(self, parent, ""))
        self.fontSize = parent.fontSize
        self.headerAttrLine = None
        self.nomogram = nomogram
        self.footer = None
        self.footerPercent = None
        self.parent = parent
        if self.parent.cl:
            self.footerPercentName = "P(%s=\"%s\")" % (self.parent.cl.domain.classVar.name,self.parent.cl.domain.classVar.values[self.parent.TargetClassIndex])
        else:
            self.footerPercentName = ""
        self.connectedLine = QCanvasLine(self)
        self.connectedLine.setPen(QPen(Qt.blue))
        self.errorLine = QCanvasLine(self)
        
        self.errorPercentLine = QCanvasLine(self)
        self.leftArc = QCanvasPolygon(self)
        self.rightArc = QCanvasPolygon(self)
        self.leftPercentArc = QCanvasPolygon(self)
        self.rightPercentArc = QCanvasPolygon(self)
        self.cilist = [self.errorLine, self.errorPercentLine, self.leftArc, self.rightArc, self.leftPercentArc, self.rightPercentArc]
        for obj in self.cilist:
            obj.setPen(QPen(Qt.blue, 3))
            obj.setZ(100)

        self.linkFunc = self.logit
        self.invLinkFunc = self.invLogit

    def logit(self, val):
        return math.exp(val)/(1+math.exp(val))
    
    def invLogit(self, p):    
        return math.log(p/max(1-p,aproxZero))
    
    def convertToPercent(self, atLine):
        minPercent = self.linkFunc(atLine.minValue)#math.exp(atLine.minValue)/(1+math.exp(atLine.minValue))
        maxPercent = self.linkFunc(atLine.maxValue)#math.exp(atLine.maxValue)/(1+math.exp(atLine.maxValue))

        percentLine = AttrLine(atLine.name, self)
        percentList = filter(lambda x:x>minPercent and x<maxPercent,Numeric.arange(0, maxPercent+0.1, 0.05))
        for p in percentList:
            if int(10*p) != round(10*p,1) and not p == percentList[0] and not p==percentList[len(percentList)-1]:
                percentLine.addAttValue(AttValue(" "+str(p)+" ", self.invLinkFunc(p), markerWidth = 1, enable = False))
            else:
                percentLine.addAttValue(AttValue(" "+str(p)+" ", self.invLinkFunc(p), markerWidth = 1))
        return percentLine  
        
       
    def paintFooter(self, rect, alignType, yAxis, mapper):
        # set height for each scale        
        height = rect.height()/3
        
        # get min and maximum sum, min and maximum beta
        # min beta <--> min sum! , same for maximum
        maxSum = minSum = maxSumBeta = minSumBeta = 0
        for at in self.nomogram.attributes:
            maxSum += mapper.getMaxValue(at)
            minSum += mapper.getMinValue(at)
            maxSumBeta += at.maxValue
            minSumBeta += at.minValue

        # add constant to betas!
        maxSumBeta += self.nomogram.constant.betaValue
        minSumBeta += self.nomogram.constant.betaValue

        # show only reasonable values
        k = (maxSum-minSum)/max((maxSumBeta-minSumBeta),aproxZero)
        if maxSumBeta>4:
            maxSum = (4 - minSumBeta)*k + minSum
            maxSumBeta = 4
        if minSumBeta>3:
            minSum = (3 - minSumBeta)*k + minSum
            minSumBeta = 3
        if minSumBeta<-4:
            minSum = (-4 - minSumBeta)*k + minSum
            minSumBeta = -4
        if maxSumBeta<-3:
            maxSum = (-3 - minSumBeta)*k + minSum
            maxSumBeta = -3

        # draw continous line with values from min and max sum (still have values!)
        self.m = Mapper_Linear_Fixed(minSumBeta, maxSumBeta, rect.left(), rect.right(), maxLinearValue = maxSum, minLinearValue = minSum)
        if self.footer:
            [item.setCanvas(None) for item in self.allItems()]
            #self.footer.destroy()
        self.footer = self.m.getHeaderLine(self, QRect(rect.left(), rect.top(), rect.width(), height))
        self.footer.name = self.nomogram.parent.totalPointsName[self.nomogram.parent.yAxis]

        self.footer.paint(self, QRect(rect.left(), rect.top(), rect.width(), height), self.m)

        # continous line convert to percent and draw accordingly (minbeta = minsum)
        #if self.footerPercent:
        #    self.footerPercent.destroy()

        self.footerPercent = self.convertToPercent(self.footer)

        # create a mapper for footer, BZ CHANGE TO CONSIDER THE TARGET
        self.footerPercent.name = self.footerPercentName
        self.footerPercent.paint(self, QRect(rect.left(), rect.top()+height, rect.width(), 2*height), self.m)                         

        self.resize(self.nomogram.pright, rect.height()+30)
        self.update()
        
    def updateMarkers(self):
        # finds neares beta; use only discrete data
        def getNearestAtt(selectedBeta, at):
            nearestLeft = filter(lambda x: x.betaValue == max([v.betaValue for v in filter(lambda x: x.betaValue <= selectedBeta, at.attValues)]) ,at.attValues)[0]
            nearestRight = filter(lambda x: x.betaValue == min([v.betaValue for v in filter(lambda x: x.betaValue >= selectedBeta, at.attValues)]) ,at.attValues)[0]
            return (nearestLeft, nearestRight)
        
        sum = self.nomogram.constant.betaValue
        for at in self.nomogram.attributes:
            sum += at.selectedValue[2]

        variance = math.pow(self.nomogram.constant.error,2)
        for at in self.nomogram.attributes:
#            if not isinstance(at, AttrLineCont):
            if at.selectedValue[2] == 0.0 and self.parent.alignType == 1:
                continue
            (nleft, nright) = getNearestAtt(at.selectedValue[2], at)
            if nright.betaValue>nleft.betaValue:
                prop = (at.selectedValue[2]-nleft.betaValue)/(nright.betaValue-nleft.betaValue)
            else:
                prop = 0
            if prop == 0:
                variance += math.pow(nleft.error, 2)
            elif prop == 1:
                variance += math.pow(nright.error, 2)
            else:
                variance += math.pow(nleft.error, 2)*(1-prop)
                variance += math.pow(nright.error, 2)*prop

        standard_error = math.sqrt(variance)

        ax=self.m.mapBeta(sum, self.footer)
        # get CI
        ax_maxError = self.m.mapBeta(sum+standard_error*norm_factor(1-((1-float(self.parent.confidence_percent)/100.)/2.)), self.footer)
        ax_minError = self.m.mapBeta(sum-standard_error*norm_factor(1-((1-float(self.parent.confidence_percent)/100.)/2.)), self.footer)
        a = QPointArray()
        a.makeArc(ax_minError, self.footer.marker.y()+10, 10, 10, 0, 180*16)
        self.leftArc.setPoints(a)
        #self.leftArc.setBrush(QBrush(Qt.blue))
    
        a = QPointArray()
        a.makeArc(ax_maxError-10, self.footer.marker.y()-5, 10, 10, 90*16, -90*16)
        self.rightArc.setPoints(a)
        a.makeArc(ax_minError, self.footerPercent.marker.y()-5, 10, 10, 90*16, 180*16)
        self.leftPercentArc.setPoints(a)
        a.makeArc(ax_maxError-10, self.footerPercent.marker.y()-5, 10, 10, 90*16, -90*16)
        self.rightPercentArc.setPoints(a)

        
        axPercentMin=self.m.mapBeta(self.footerPercent.minValue, self.footer)
        axPercentMax=self.m.mapBeta(self.footerPercent.maxValue, self.footer)
        axMin=self.m.mapBeta(self.footer.minValue, self.footer)
        axMax=self.m.mapBeta(self.footer.maxValue, self.footer)

        ax = max(ax, axMin)
        ax = min(ax, axMax)
        self.errorLine.setPoints(ax_minError, self.footer.marker.y(), ax_maxError, self.footer.marker.y())
        self.errorLine.setCanvas(self)
        ax_minError = min(ax_minError, axPercentMax)
        ax_minError = max(ax_minError, axPercentMin)        
        ax_maxError = min(ax_maxError, axPercentMax)
        ax_maxError = max(ax_maxError, axPercentMin)        
        
        self.errorPercentLine.setCanvas(self)
        self.errorPercentLine.setPoints(ax_minError, self.footerPercent.marker.y(), ax_maxError, self.footerPercent.marker.y())
        
        self.footer.selectedValue = [ax,self.footer.marker.y(),self.m.mapBetaToLinear(sum, self.footer)]
        self.footer.marker.setPos(ax, self.footer.marker.y())

        if ax>axPercentMax:
            ax=axPercentMax
        if ax<axPercentMin:
            ax=axPercentMin
        self.footerPercent.selectedValue = [ax,self.footer.marker.y(),1/(1+math.exp(-sum))]
        self.footerPercent.marker.setPos(ax, self.footerPercent.marker.y())
        
        if self.parent.probability:
            self.footer.marker.show()
            self.footerPercent.marker.show()
            if self.footer.marker.x() == self.footerPercent.marker.x():
                self.connectedLine.setPoints(self.footer.marker.x(), self.footer.marker.y(), self.footerPercent.marker.x(), self.footerPercent.marker.y())
                self.connectedLine.setCanvas(self)
                self.connectedLine.show()
            else:
                self.connectedLine.hide()
            if self.parent.confidence_check:
                self.showCI()
            else:
                self.hideCI()
        self.update()
        
    def showCI(self):
        self.errorLine.show()
        self.errorPercentLine.show()

    def hideCI(self):
        self.errorLine.hide()
        self.errorPercentLine.hide()
        self.leftArc.hide()
        self.rightArc.hide()
        self.leftPercentArc.hide()
        self.rightPercentArc.hide()


# #################################################################### 
# Main CANVAS
# ####################################################################
class BasicNomogram(QCanvas):
    def __init__(self, parent, constant, *args):
        apply(QCanvas.__init__,(self, parent, ""))
        
        self.parent=parent
        self.items = []
        
        self.attributes = []
        self.constant = constant
        self.minBeta = 0
        self.maxBeta = 0
        self.max_difference = 0
        
        self.fontSize = parent.fontSize

        self.zeroLine = QCanvasLine(self)
        self.zeroLine.setPen(QPen(Qt.DotLine))
        self.zeroLine.setZ(-10)

        self.header = BasicNomogramHeader(self, parent)
        self.footerCanvas = BasicNomogramFooter(self, parent)
        self.parent.header.setCanvas(self.header)
        self.parent.footer.setCanvas(self.footerCanvas)
        
    def addAttribute(self, attr):
        self.attributes.append(attr)
        if attr.minValue < self.minBeta:
            self.minBeta = attr.minValue
        if attr.maxValue > self.maxBeta:
            self.maxBeta = attr.maxValue
        if attr.maxValue-attr.minValue > self.max_difference:
            self.max_difference = attr.maxValue-attr.minValue

    def hideAllMarkers(self):
        for at in self.attributes:
            at.marker.hide()
        self.footerCanvas.footer.marker.hide()
        self.footerCanvas.footerPercent.marker.hide()
        self.footerCanvas.connectedLine.hide()
        self.footerCanvas.hideCI()
        self.update()
        self.footerCanvas.update()

    def showAllMarkers(self):
        for at in self.attributes:
            at.marker.show()
        self.footerCanvas.footer.marker.show()
        self.footerCanvas.footerPercent.marker.show()
        if self.parent.confidence_check:
            self.footerCanvas.showCI()
        if self.footerCanvas.footer.marker.x() == self.footerCanvas.footerPercent.marker.x():
            self.footerCanvas.connectedLine.setPoints(self.footerCanvas.footer.marker.x(), self.footerCanvas.footer.marker.y(), self.footerCanvas.footerPercent.marker.x(), self.footerCanvas.footerPercent.marker.y())
            self.footerCanvas.connectedLine.setCanvas(self.footerCanvas)
            self.footerCanvas.connectedLine.show()
        self.update()
        self.footerCanvas.update()

    def paint(self, rect, mapper):
        self.zeroLine.setPoints(mapper.mapBeta(0, self.header.headerAttrLine), rect.top(), mapper.mapBeta(0, self.header.headerAttrLine), rect.bottom()-self.parent.verticalSpacing/2 + 25)
        if self.parent.showBaseLine:
            self.zeroLine.show()
        else:
            self.zeroLine.hide()
        curr_rect = QRect(rect.left(), rect.top(), rect.width(), 0)
        disc = False
 
        for at in self.attributes:
            if (isinstance(at, AttrLineCont) or isinstance(at, AttrLineOrdered)) and self.parent.contType == 1:   
                if disc:
                    curr_rect = QRect(rect.left(), curr_rect.bottom()+20, rect.width(), at.getHeight(self))
                    disc=False
                else:
                    curr_rect = QRect(rect.left(), curr_rect.bottom(), rect.width(), at.getHeight(self))
                at.paint2d(self, curr_rect, mapper)
            else:
                disc = True
                curr_rect = QRect(rect.left(), curr_rect.bottom(), rect.width(), at.getHeight(self))
                at.paint(self, curr_rect, mapper)
                # if histograms are used, a larger rect is required
                if self.parent.histogram:
                    curr_rect.setHeight(at.getHeight(self)+self.parent.histogram_size)
            

    def setSizes(self, parent):
        def getBottom():
            bottom, lastAt = 0, None
            for at in self.attributes:
                if lastAt and self.parent.contType == 1 and isinstance(at, AttrLineCont) and not isinstance(lastAt, AttrLineCont):
                    bottom += 20
                if (isinstance(at, AttrLineCont) or isinstance(at, AttrLineOrdered)) and self.parent.contType == 1:   
                    bottom += at.getHeight(self)
                else:
                    bottom += at.getHeight(self)
                    if self.parent.histogram:
                        bottom += self.parent.histogram_size                
                lastAt = at
            return bottom;
        
        self.pleft, self.pright, self.ptop, self.pbottom = 0, parent.graph.width() - 20, 0, self.parent.verticalSpacing
        self.pbottom += getBottom()

        #graph sizes
        self.gleft = 0
        for at in self.attributes:
            if not (self.parent.contType == 1 and isinstance(at, AttrLineCont)) and at.label.boundingRect().width()>self.gleft:
                self.gleft = at.label.boundingRect().width()
        if QCanvasText(self.footerCanvas.footerPercentName, self.footerCanvas).boundingRect().width()>self.gleft:
            self.gleft = QCanvasText(self.footerCanvas.footerPercentName, self.footerCanvas).boundingRect().width()
            
        #self.gleft = max(self.gleft, 100) # should really test footer width, and with of other lables
        self.gleft = max(self.gleft, 80)
        self.gleft +=20
        self.gright=self.pright-(self.pright-self.pleft)/10
        self.gtop = self.ptop + 10
        self.gbottom = self.pbottom - 10

        self.gwidth = self.gright-self.gleft
        self.gheight = self.gbottom - self.gtop

        if self.pbottom < parent.graph.height() - 30:
            self.pbottom = parent.graph.height() - 30


    def show(self):
        # set sizes
        [item.hide() for item in self.allItems()]
        
        self.setSizes(self.parent)
        self.setBackgroundColor(Qt.white)
        self.resize(self.pright, self.pbottom)

        curr_point = self.parent.verticalSpacing
        if self.parent.alignType == 0:
            if self.parent.yAxis == 0:
                self.mapper = Mapper_Linear_Left(self.max_difference,  self.gleft, self.gright)
            else:
                self.mapper = Mapper_Linear_Left(self.max_difference,  self.gleft, self.gright, maxLinearValue = self.max_difference)
        else:
            if self.parent.yAxis == 0:
                self.mapper = Mapper_Linear_Center(self.minBeta, self.maxBeta, self.gleft, self.gright)
            else:
                self.mapper = Mapper_Linear_Center(self.minBeta, self.maxBeta, self.gleft, self.gright, maxLinearValue = self.maxBeta, minLinearValue = self.minBeta)
        # draw HEADER and vertical line
        topRect=QRect(self.gleft, self.gtop, self.gwidth, self.parent.verticalSpacing-20)
        self.header.paintHeader(topRect, self.mapper) 
        # draw nomogram
        middleRect=QRect(self.gleft, self.ptop, self.gwidth, self.gheight)
        self.paint(middleRect, self.mapper)
        # draw final line
        bottomRect=QRect(self.gleft, self.gtop, self.gwidth, 2*self.parent.verticalSpacing-30)
        self.footerCanvas.paintFooter(bottomRect, self.parent.alignType, self.parent.yAxis, self.mapper)        
        self.footerCanvas.updateMarkers()
        if self.parent.probability:
            self.showAllMarkers()
        self.update()

    def showBaseLine(self, show):
        if show:
            self.zeroLine.show()
        else:
            self.zeroLine.hide()
        self.update()

    def printOUT(self):
        print "constant:", str(self.constant.betaValue)
        for a in self.attributes:
            print a.toString()

    def findAttribute(self, y):
        for at in self.attributes:
            if y>at.minCanvasY and y<at.maxCanvasY:
                return at
        return None
        
    def updateValues(self, x, y, obj):
        if obj.attribute.updateValueXY(x, y):
            self.footerCanvas.updateMarkers()
            self.update()
            self.header.headerAttrLine.marker.setX(obj.attribute.marker.x())
            self.header.headerAttrLine.marker.show()
            self.header.update()

    def stopDragging(self):
        self.header.headerAttrLine.marker.hide()
        self.header.update()



# #################################################################### 
# CANVAS VIEWERS
# ####################################################################
class OWNomogramHeader(QCanvasView):
    def __init__(self, canvas, mainArea):
        apply(QCanvasView.__init__,(self,)+(canvas,mainArea))
        self.setMouseTracking(True)
        self.viewport().setMouseTracking(True)
        self.mouseOverObject = None        

    # ###################################################################
    # mouse is running around, perhaps Jerry is nearby ##################
    # or technically: user moved mouse ################################## 
    def contentsMouseMoveEvent(self, ev):
        if self.canvas():
            items = filter(lambda ci: ci.z()==50, self.canvas().collisions(ev.pos()))
            if len(items)>0:
                if self.mouseOverObject:
                    self.mouseOverObject.hideSelected()
                self.mouseOverObject = items[0]
                self.mouseOverObject.showSelected()
                self.canvas().update()
            elif self.mouseOverObject:
                self.mouseOverObject.hideSelected()
                self.mouseOverObject = None
                self.canvas().update()
                

class OWNomogramGraph(QCanvasView):
    def __init__(self, canvas, mainArea):
        apply(QCanvasView.__init__,(self,)+(canvas,mainArea))
        self.selectedObject = None
        self.mouseOverObject = None        
        self.setMouseTracking(True)
        self.viewport().setMouseTracking(True)
        self.bDragging = False
        self.resizing = False

    def resizeEvent(self, event):
        apply(QCanvasView.resizeEvent, (self,event))
        if self.canvas():
            self.resizing = True
            self.canvas().show()

    # ###################################################################
    # mouse button was pressed #########################################
    def contentsMousePressEvent(self, ev):
        if self.canvas() and ev.button() == QMouseEvent.LeftButton:
            items = filter(lambda ci: ci.z()==50, self.canvas().collisions(ev.pos()))
            if len(items)>0:
                self.selectedObject = items[0]
                #self.canvas().updateValues(ev.x(), ev.y(), self.selectedObject)
                self.bDragging = True

    # ###################################################################
    # mouse button was released #########################################
    def contentsMouseReleaseEvent(self, ev):
 #       if self.resizing:
 #           self.resizing = False
 #           self.cavnas().show()
        if self.bDragging:
            self.bDragging = False
            self.canvas().stopDragging()

    # ###################################################################
    # mouse is running around, perhaps Jerry is nearby ##################
    # or technically: user moved mouse ################################## 
    def contentsMouseMoveEvent(self, ev):
        if self.bDragging:
            self.canvas().updateValues(ev.x(), ev.y(), self.selectedObject)
        elif self.canvas():
            items = filter(lambda ci: ci.z()==50, self.canvas().collisions(ev.pos()))
            if len(items)>0:
                if self.mouseOverObject:
                    self.mouseOverObject.hideSelected()
                self.mouseOverObject = items[0]
                self.mouseOverObject.showSelected()
                self.canvas().update()
            elif self.mouseOverObject:
                self.mouseOverObject.hideSelected()
                self.mouseOverObject = None
                self.canvas().update()


# ###################################################################################################################
# ------------------------------------------------------------------------------------------------------------------
# MAPPERS            
# ------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------
def createSetOfVisibleValues(min, max, dif):
    if dif == 0.0 or max == min:
        return [0.0]
    upper = max-min+1.8*dif
#    add = round((min-dif)/dif)*dif
    add = math.ceil((min-0.9*dif)/dif)*dif
        
    dSum = Numeric.arange(0, upper, dif)
    dSum = map(lambda x:x+add, dSum)
    return dSum    


class Mapper_Linear_Fixed:
    def __init__(self, minBeta, maxBeta, left, right, maxLinearValue = 100, minLinearValue = -100, inverse = False):
        self.inverse = inverse
        self.minBeta = minBeta
        self.maxBeta = maxBeta
        self.left = left
        self.right = right
        self.maxLinearValue = maxLinearValue
        self.minLinearValue = minLinearValue

        # find largest absolute beta and set it to maxLinearValue        
        self.maxValue = self.maxLinearValue
        self.minValue = self.minLinearValue

        # set actual values on graph (with roundings, but not yet, later)
        self.minGraphValue = self.minValue
        self.maxGraphValue = self.maxValue
        self.minGraphBeta = self.minBeta
        self.maxGraphBeta = self.maxBeta

    def __call__(self, attrLine, error_factor = 1.96):
        beta = []
        b_error = []
        max_mapped = self.left
        min_mapped = self.right
        for at in attrLine.attValues:
            k = self.propBeta(at.betaValue, attrLine)
            beta.append(self.left+k*(self.right-self.left))
            if self.left+k*(self.right-self.left)>max_mapped:
                max_mapped = self.left+k*(self.right-self.left)
            if self.left+k*(self.right-self.left)<min_mapped:
                min_mapped = self.left+k*(self.right-self.left)
            k1 = self.propBeta(at.betaValue-error_factor*at.error, attrLine)
            k2 = self.propBeta(at.betaValue+error_factor*at.error, attrLine)
            b_error.append([self.left+k1*(self.right-self.left), self.left+k2*(self.right-self.left)])
        if max_mapped<min_mapped+5:
            max_mapped=min_mapped+5
        return (beta, b_error, min_mapped, max_mapped)

    def mapBeta(self, betaVal, attrLine):
        k = self.propBeta(betaVal, attrLine)
        return self.left+k*(self.right-self.left)        

    def mapBetaToLinear(self, betaVal, attrLine):
        k = self.propBeta(betaVal, attrLine)
        return self.minGraphValue+k*(self.maxGraphValue-self.minGraphValue)  

    def getLeftMost(self):
        return self(self.minGraphBeta)

    def getRightMost(self):
        return self(self.maxGraphBeta)

    def getMinValue(self):
        return self.minValue
    def getMaxValue(self):
        return self.maxValue
    
    # return proportional beta
    def propBeta(self, betaVal, attrLine):
        if self.inverse:
            return (self.maxGraphBeta-betaVal)/max((self.maxGraphBeta-self.minGraphBeta), aproxZero)
        else:
            return (betaVal-self.minGraphBeta)/max((self.maxGraphBeta-self.minGraphBeta), aproxZero)

    # delay / offset that a mapper produces
    # in this case no aligning is uses, that is why delay is always 0
    def getDelay(self, nomogram):
        return 0
    
    def getHeaderLine(self, canvas, rect):
        maxnum = rect.width()/(3*canvas.fontSize)
        if maxnum<1:
            maxnum=1
        d = (self.maxValue - self.minValue)/maxnum
        dif = getDiff(d)

        dSum = createSetOfVisibleValues(self.minValue, self.maxValue, dif)
        if round(dSum[0],0) == dSum[0] and round(dSum[len(dSum)-1],0) == dSum[len(dSum)-1] and round(dif,0) == dif:
            conv = int
        else:
            conv = lambda x:x
        
        # set new graph values
        k = (self.maxGraphBeta - self.minGraphBeta)/max((self.maxGraphValue - self.minGraphValue), aproxZero)
        self.maxGraphBeta = (dSum[len(dSum)-1]- self.minGraphValue)*k + self.minGraphBeta                  
        self.minGraphBeta = (dSum[0]- self.minGraphValue)*k + self.minGraphBeta                  
        self.minGraphValue = dSum[0]
        self.maxGraphValue = dSum[len(dSum)-1]
       
        k = (self.maxGraphBeta-self.minGraphBeta)/max((self.maxGraphValue-self.minGraphValue), aproxZero)
        dSumValues = [(d,self.minGraphBeta + (d-self.minGraphValue)*k) for d in dSum]
        headerLine = AttrLine("Points", canvas)
        for d_i,d in enumerate(dSumValues):
            headerLine.addAttValue(AttValue(" "+str(conv(d[0]))+" ", d[1], markerWidth = 1))
            if d != dSumValues[-1]:
                val = AttValue(" "+str((d[0]+dSumValues[d_i+1][0])/2)+ " ", (d[1]+dSumValues[d_i+1][1])/2, markerWidth = 1)
                val.enable = False
                headerLine.addAttValue(val)
        return headerLine


class Mapper_Linear_Center:
    def __init__(self, minBeta, maxBeta, left, right, maxLinearValue = 100, minLinearValue = -100):
        if minBeta == 0:
            self.minBeta = aproxZero
        else:
            self.minBeta = minBeta
        if maxBeta == 0:
            self.maxBeta = aproxZero
        else:
            self.maxBeta = maxBeta
        self.left = left
        self.right = right
        self.maxLinearValue = maxLinearValue
        self.minLinearValue = minLinearValue

        # find largest absolute beta and set it to maxLinearValue        
        if abs(self.maxBeta) > abs(self.minBeta):
            self.maxValue = self.maxLinearValue
            self.minValue = self.maxLinearValue*self.minBeta/self.maxBeta
            if self.minValue < self.minLinearValue:
                self.minValue = self.minLinearValue
        else:
            self.minValue = self.minLinearValue
            self.maxValue = self.minLinearValue*self.maxBeta/self.minBeta
            if self.maxValue > self.maxLinearValue:
                self.maxValue = self.maxLinearValue

        # set actual values on graph (with roundings, but not yet, later)
        self.minGraphValue = self.minValue
        self.maxGraphValue = self.maxValue
        self.minGraphBeta = self.minBeta
        self.maxGraphBeta = self.maxBeta

    def __call__(self, attrLine, error_factor = 1.96):
        beta = []
        b_error = []
        max_mapped = self.left
        min_mapped = self.right
        for at in attrLine.attValues:
            k = self.propBeta(at.betaValue, attrLine)
            beta.append(self.left+k*(self.right-self.left))
            if self.left+k*(self.right-self.left)>max_mapped:
                max_mapped = self.left+k*(self.right-self.left)
            if self.left+k*(self.right-self.left)<min_mapped:
                min_mapped = self.left+k*(self.right-self.left)
            k1 = self.propBeta(at.betaValue-error_factor*at.error, attrLine)
            k2 = self.propBeta(at.betaValue+error_factor*at.error, attrLine)
            b_error.append([self.left+k1*(self.right-self.left), self.left+k2*(self.right-self.left)])

        if max_mapped<min_mapped+5:
            max_mapped=min_mapped+5
        return (beta, b_error, min_mapped, max_mapped)

    def mapBeta(self, betaVal, attrLine):
        k = self.propBeta(betaVal, attrLine)
        return self.left+k*(self.right-self.left)        

    def getLeftMost(self):
        return self(self.minGraphBeta)

    def getRightMost(self):
        return self(self.maxGraphBeta)

    def getMaxMapperValue(self):
        return self.maxGraphValue        
    def getMinMapperValue(self):
        return self.minGraphValue
    
    def getMaxValue(self, attr):
        if self.maxGraphBeta==0:
            return self.maxGraphValue*attr.maxValue/aproxZero             
        return self.maxGraphValue*attr.maxValue/self.maxGraphBeta

    def getMinValue(self, attr):
        if self.minGraphValue == 0:
            return self.minGraphValue*attr.minValue/aproxZero
        return self.minGraphValue*attr.minValue/self.minGraphBeta

        
    
    # return proportional beta
    def propBeta(self, betaVal, attrLine):        
        return (betaVal-self.minGraphBeta)/max((self.maxGraphBeta-self.minGraphBeta), aproxZero)

    # delay / offset that a mapper produces
    # in this case no aligning is uses, that is why delay is always 0
    def getBetaDelay(self, nomogram):
        return 0
    
    def getHeaderLine(self, canvas, rect):
        maxnum = rect.width()/(3*canvas.fontSize)
        if maxnum<1:
            maxnum=1
        d = (self.maxValue - self.minValue)/maxnum
        dif = getDiff(d)
        dSum = createSetOfVisibleValues(self.minValue, self.maxValue, dif);

        if round(dSum[0],0) == dSum[0] and round(dSum[len(dSum)-1],0) == dSum[len(dSum)-1] and round(dif,0) == dif:
            conv = int
        else:
            conv = lambda x:x

            
        # set new graph values
        if self.minGraphValue == 0:
            self.minGraphBeta = self.minBeta
        else:
            self.minGraphBeta = self.minBeta*dSum[0]/self.minGraphValue
        if self.maxGraphValue == 0:
            self.maxGraphBeta = self.maxBeta
        else:
            self.maxGraphBeta = self.maxBeta*dSum[len(dSum)-1]/self.maxGraphValue
        self.minGraphValue = dSum[0]
        self.maxGraphValue = dSum[len(dSum)-1]

        # coefficient to convert values into betas
        k = (self.maxGraphBeta-self.minGraphBeta)/max((self.maxGraphValue-self.minGraphValue), aproxZero)

        headerLine = AttrLine("Points", canvas)
        for at in range(len(dSum)):
            headerLine.addAttValue(AttValue(" "+str(conv(dSum[at]))+" ", self.minGraphBeta + (dSum[at]-self.minGraphValue)*k, markerWidth = 1))
            if at != len(dSum)-1:
                val = AttValue(" "+str((dSum[at]+dSum[at+1])/2)+" ", self.minGraphBeta + ((dSum[at]+dSum[at+1])/2-self.minGraphValue)*k, markerWidth = 1)
                val.enable = False
                headerLine.addAttValue(val)
                
        return headerLine


# it is very similar to Mapper_Linear_Center. It has the same methods, implementation is slightly different
class Mapper_Linear_Left:
    def __init__(self, max_difference, left, right, maxLinearValue = 100):
        self.max_difference = max_difference
        self.left = left
        self.right = right
        self.maxLinearValue = maxLinearValue
    
    def __call__(self, attrLine, error_factor = 1.96):
        beta = []
        b_error = []
        minb = self.right
        maxb = self.left
        for at in attrLine.attValues:
            k = (at.betaValue-attrLine.minValue)/max(self.max_difference, aproxZero)
            beta.append(self.left + k*(self.right-self.left))
            if self.left + k*(self.right-self.left)>maxb:
                maxb = self.left + k*(self.right-self.left)
            if self.left + k*(self.right-self.left)<minb:
                minb = self.left + k*(self.right-self.left)
            k1 = (at.betaValue-error_factor*at.error-attrLine.minValue)/max(self.max_difference, aproxZero)
            k2 = (at.betaValue+error_factor*at.error-attrLine.minValue)/max(self.max_difference, aproxZero)
            b_error.append([self.left + k1*(self.right-self.left), self.left + k2*(self.right-self.left)])

        if maxb<minb+5:
            maxb=minb+5
        return (beta, b_error, minb, maxb)
    
    def mapBeta(self, betaVal, attrLine):
        k = (betaVal-attrLine.minValue)/max(self.max_difference, aproxZero)
        return self.left+k*(self.right-self.left)        

    def propBeta(self, betaVal, attrLine):        
        return (betaVal-attrLine.minValue)/max(self.max_difference, aproxZero)

    def getMaxMapperValue(self):
        return self.maxLinearValue        
    def getMinMapperValue(self):
        return 0
    def getMaxValue(self, attr):
        return self.maxLinearValue*(attr.maxValue-attr.minValue)/max(self.max_difference, aproxZero)
    def getMinValue(self, attr):
        return 0

    def getBetaDelay(self, nomogram):
        delay = 0
        for at in nomogram.attributes:
            delay += at.minValue
        return delay
        
    def getHeaderLine(self, canvas, rect):
        maxnum = rect.width()/(3*canvas.fontSize)
        if maxnum<1:
            maxnum=1
        d = self.maxLinearValue/maxnum
        dif = max(getDiff(d),10e-6)
        dSum = []
        dSum = Numeric.arange(0, self.maxLinearValue+dif, dif)
        if len(dSum)<1:
            dSum = Numeric.arange(0, self.maxLinearValue+dif, dif/2.)
        dSum = map(lambda x:x, dSum)
        if round(dSum[0],0) == dSum[0] and round(dSum[len(dSum)-1],0) == dSum[len(dSum)-1] and round(dif,0) == dif:
            conv = int
        else:
            conv = lambda x:x

        k = self.max_difference/self.maxLinearValue        

        headerLine = AttrLine("", canvas)
        for at in range(len(dSum)):
            headerLine.addAttValue(AttValue(" "+str(conv(dSum[at]))+" ", dSum[at]*k, markerWidth = 1))
            # in the middle add disable values, just to see cross lines
            if at != len(dSum)-1:
                val = AttValue(" "+str((dSum[at]+dSum[at+1])/2)+ " ", (dSum[at]+dSum[at+1])*k/2, markerWidth = 1)
                val.enable = False
                headerLine.addAttValue(val)
                
        return headerLine

# ---------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------

