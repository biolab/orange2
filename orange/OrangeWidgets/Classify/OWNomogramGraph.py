# Nomogram visualization widget. It is used together with OWNomogram

from OWWidget import *
from Numeric import *
from qtcanvas import *
import time, statc

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
    if d==0:
        return 0

    if str(d)[0]>'4':
        return math.pow(10,math.floor(math.log10(d))+1)
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
            if self.attribute.continuous:
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
        if not isinstance(self.canvas, BasicNomogram) or not self.canvas.onCanvas(x,y):
            return True

        # get points
        selectedBeta = self.attribute.selectedValue[2]
        proportionalBeta = self.canvas.mapper.propBeta(selectedBeta, self.attribute)
        maxValue = self.canvas.mapper.getMaxMapperValue()
        minValue = self.canvas.mapper.getMinMapperValue()
        points = minValue+(maxValue-minValue)*proportionalBeta

        self.header.setText(self.canvas.parent.pointsName[self.canvas.parent.yAxis]+":")
        self.headerValue.setText(str(round(points,2)))
        

        # continuous? --> get attribute value
        if self.attribute.continuous:
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
                self.value.setText(str(round(1-prop,2))+"%")
                self.supportingValue.setText(str(round(prop,2))+"%")

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
        
        
        while not self.canvas.onCanvas(xTemp,yTemp) or not self.canvas.onCanvas(xTemp,yTemp+height) or not self.canvas.onCanvas(xTemp+width,yTemp) or not self.canvas.onCanvas(xTemp+width,yTemp+width):
            if yTemp == y-selOffset-height and not xTemp <= x-selOffset-width:
                xTemp-=1
            elif xTemp <= x-selOffset-width and not yTemp == y+selOffset:
                yTemp = y+selOffset
            elif yTemp == y+selOffset and not xTemp >= x+selOffset:
                xTemp+=1
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
        if points == 0 and self.canvas.parent.alignType == 1:
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
        self.borderCircle = QCanvasEllipse(15,15,canvas)
        self.borderCircle.setBrush(QBrush(Qt.red))
        self.borderCircle.setZ(z-1)
        self.descriptor = Descriptor(canvas, attribute, z+1)

    def setPos(self, x, y):
        self.setX(x)
        self.setY(y)
        self.borderCircle.setX(x)
        self.borderCircle.setY(y)
        if not self.descriptor.drawAll(x,y):
            brush = QBrush(Qt.blue)
            brush.setStyle(Qt.Dense4Pattern)
            self.setBrush(brush)
        else:
            self.setBrush(QBrush(Qt.blue))

    def showSelected(self):
        self.borderCircle.show()
        if self.canvas().parent.bubble:
            self.descriptor.showAll()
        
    def hideSelected(self):
        self.borderCircle.hide()
        self.descriptor.hideAll()



    
class AttValue:
    def __init__(self, name, betaValue, error=0, showErr=False, over=True, lineWidth = 0):
        self.name = name
        self.betaValue = betaValue
        self.error = error
        self.showErr = showErr
        self.enable = True
        self.over = over
        self.lineWidth = lineWidth
        self.attCreation = True # flag shows that vanvas object have to be created first

    def destroy(self):
        if not self.attCreation:
            self.hide()

    def setCreation(self, canvas):
        self.text = QCanvasText(self.name, canvas)
        self.text.setTextFlags(Qt.AlignCenter)
        self.labelMarker = QCanvasLine(canvas)
        self.errorLine = QCanvasLine(canvas)
        self.errorLine.setPen(QPen(Qt.blue))
        self.errorLine.setZ(10)
        self.attCreation = False

    def hide(self):
        self.text.hide()
        self.labelMarker.hide()
        self.errorLine.hide()
            
    def paint(self, canvas, rect, mapper):
        def errorCollision(line,z=10):
            col = line.collisions(True)
            for obj in col:
                if obj.z() == z:
                    return True
            return False
        
        if self.attCreation:
            self.setCreation(canvas)
        self.text.setX(self.x)
        if self.enable:
            lineLength = canvas.fontSize/3
            if canvas.parent.histogram and isinstance(canvas, BasicNomogram):
                lineLength = 2+self.lineWidth*canvas.parent.histogram_size
            if self.over:
                self.text.setY(rect.bottom()-4*canvas.fontSize/3)
                self.labelMarker.setPoints(self.x, rect.bottom(), self.x, rect.bottom()+lineLength)
            else:
                self.text.setY(rect.bottom()+4*canvas.fontSize/3)
                self.labelMarker.setPoints(self.x, rect.bottom(), self.x, rect.bottom()-lineLength)
            if self.showErr:
                if self.over:
                    add = 2
                    self.errorLine.setPoints(self.low_errorX, rect.bottom()+add, self.high_errorX , rect.bottom()+add)
                    while errorCollision(self.errorLine):
                        add +=2
                        self.errorLine.setPoints(self.low_errorX, rect.bottom()+add, self.high_errorX , rect.bottom()+add)
                else:
                    add = -2
                    self.errorLine.setPoints(self.low_errorX, rect.bottom()+add, self.high_errorX , rect.bottom()+add)
                    while errorCollision(self.errorLine):
                        add +=2
                        self.errorLine.setPoints(self.low_errorX, rect.bottom()+add, self.high_errorX , rect.bottom()+add)
                self.errorLine.show()
            else:
                self.errorLine.hide()
            self.text.show()
        # if value is disabled, draw just a symbolic line        
        else:
            self.labelMarker.setPoints(self.x, rect.bottom(), self.x, rect.bottom()+canvas.fontSize/6)
            self.text.hide()
        if canvas.parent.histogram and isinstance(canvas, BasicNomogram):
            self.labelMarker.setPen(QPen(Qt.black, 4))
        else:
            self.labelMarker.setPen(QPen(Qt.black, 1))
        self.labelMarker.show()        

    def toString(self):
        return self.name, "beta =", self.betaValue

# this class represent attribute in nomogram
# Attributes:
#   * name - name of attribute
#   * TDposition - line number
#   * attValues - set of values
#   * minValue, maxValue - minValues and maxValues of attribute
class AttrLine:
    def __init__(self, name, canvas, continuous = False):
        self.name = name
        self.attValues = []
        self.minValue = self.maxValue = 0
        self.continuous = continuous
        self.selectedValue = None
        self.initialize(canvas)
        self.initialized = False # complet initialization atm is not possible
        self.cAtt = None

    def destroy(self):
        self.label.hide()
        self.line.hide()
        self.box.hide()
        for obj in self.contValues:
            obj.hide()
        for obj in self.contLabel:
            obj.hide()
        self.marker.hide()

        for at in self.attValues:
            at.destroy()

    def addAttValue(self, attValue):
        if self.attValues == []:
            self.minValue = attValue.betaValue
            self.maxValue = attValue.betaValue
        self.attValues.append(attValue)
        if attValue.betaValue>self.maxValue:
            self.maxValue = attValue.betaValue
        if attValue.betaValue<self.minValue:
            self.minValue = attValue.betaValue
            
    def removeAttValue(self, attValue):
        self.attValues.remove(attValue)
        # change max and min value TODO
        
    def convertToPercent(self, canvas):
        minPercent = exp(self.minValue)/(1+exp(self.minValue))
        maxPercent = exp(self.maxValue)/(1+exp(self.maxValue))

        percentLine = AttrLine(self.name, canvas)
        percentList = filter(lambda x:x>minPercent and x<1,arange(0, maxPercent+0.1, 0.1))
        for p in percentList:
            percentLine.addAttValue(AttValue(str(p), log(p/(1-p))))
        return percentLine

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

        # continuous attributes        
        self.box = QCanvasRectangle(canvas)
        self.box.setPen(QPen(Qt.DotLine))
        self.contValues = []
        self.contLabel = []



    def initializeBeforePaint(self, canvas):
        if self.continuous:
            self.atNames = AttrLine(self.name, canvas)
            for at in self.attValues:
                self.atNames.addAttValue(AttValue(at.name, float(at.name)))
            verticalRect = QRect(0, 0, canvas.parent.verticalSpacingContinuous, canvas.parent.verticalSpacingContinuous)
            verticalMapper = Mapper_Linear_Fixed(self.atNames.minValue, self.atNames.maxValue, verticalRect.left()+verticalRect.width()/4, verticalRect.right(), maxLinearValue = self.atNames.maxValue, minLinearValue = self.atNames.minValue)
            label = verticalMapper.getHeaderLine(canvas, QRect(0,0,canvas.parent.verticalSpacingContinuous, canvas.parent.verticalSpacingContinuous)) 
            for val in label.attValues:
                # draw value
                a = QCanvasText(val.name, canvas)
                a.setTextFlags(Qt.AlignRight)
                a.marker = QCanvasLine(canvas)
                a.marker.setZ(5)
                self.contLabel.append(a)

            #line objects
            for at in self.attValues:
                a = QCanvasLine(canvas)
                a.setPen(QPen(Qt.black, at.lineWidth))
                self.contValues.append(a)

                # for 1d cont space
                at.setCreation(canvas)
            
        self.initialized = True    
                
    
        
    def paint(self, canvas, rect, mapper):
        if not self.initialized:
            self.initializeBeforePaint(canvas)
        self.label.setText(self.name)
        atValues_mapped, atErrors_mapped, min_mapped, max_mapped = mapper(self, error_factor = norm_factor(1-((1-float(canvas.parent.confidence_percent)/100.)/2.))) # return mapped values, errors, min, max --> mapper(self)
        self.label.setX(1)
        self.label.setY(rect.bottom()-canvas.fontSize)

        # draw attribute line
        self.line.setPoints(min_mapped, rect.bottom(), max_mapped, rect.bottom())
        zero = 0
        if min([at.betaValue for at in self.attValues])>0:
            zero = min([at.betaValue for at in self.attValues])
        if max([at.betaValue for at in self.attValues])<0:
            zero = max([at.betaValue for at in self.attValues])
        self.selectValues = [[mapper.mapBeta(zero, self), rect.bottom(), zero]]
        if not self.selectedValue:
            self.selectedValue = self.selectValues[0]
        
        # continuous attributes are handled differently
        if self.continuous:
            #disable all enabled values
            for at in self.attValues:
                at.hide()
            self.cAtt = self.shrinkSize(canvas, max_mapped - min_mapped)
            atValues_mapped, atErrors_mapped, min_mapped, max_mapped = mapper(self.cAtt) # return mapped values, errors, min, max --> mapper(self)
            val = self.cAtt.attValues
            for i in range(len(val)):
                # check attribute name that will not cover another name
                for j in range(i):
                    if val[j].over==val[i].over and val[j].enable and abs(atValues_mapped[j]-atValues_mapped[i])<(len(val[j].name)*canvas.fontSize/4+len(val[i].name)*canvas.fontSize/4):
                        val[i].enable = False
                val[i].x = atValues_mapped[i]
                val[i].paint(canvas, rect, mapper)
                self.selectValues.append([atValues_mapped[i], rect.bottom(), val[i].betaValue])
            
        else:
            # draw attributes
            val = self.attValues
            for i in range(len(val)):
                # check attribute name that will not cover another name
                val[i].x = atValues_mapped[i]
                val[i].high_errorX = atErrors_mapped[i][0]
                val[i].low_errorX = atErrors_mapped[i][1]
                if canvas.parent.confidence_check:
                    val[i].showErr = True
                else:
                    val[i].showErr = False
                val[i].paint(canvas, rect, mapper)
                
                for j in range(i):
                    #if val[j].over and val[j].enable and abs(atValues_mapped[j]-atValues_mapped[i])<(len(val[j].name)*canvas.fontSize/4+len(val[i].name)*canvas.fontSize/4):
                    if val[j].over and val[j].enable and val[j].text.collidesWith(val[i].text):
                        val[i].over = False
                if not val[i].over:
                    val[i].paint(canvas, rect, mapper)
                    
                self.selectValues.append([atValues_mapped[i], rect.bottom(), val[i].betaValue])
                
            for i in range(len(val)):
                val[i].over = True

        atLine = AttrLine("marker", canvas)
        if self.continuous:
            d = 5*(self.cAtt.maxValue-self.cAtt.minValue)/(max_mapped-min_mapped)
            for xc in arange(self.cAtt.minValue, self.cAtt.maxValue+d, d):
                atLine.addAttValue(AttValue("", xc))
        else:
            d = 5*(self.maxValue-self.minValue)/(max_mapped-min_mapped)
            for xc in arange(self.minValue, self.maxValue+d, d):
                atLine.addAttValue(AttValue("", xc))
            
        
        markers_mapped, mark_errors_mapped, markMin_mapped, markMax_mapped = mapper(atLine)
        for mar in range(len(markers_mapped)):
            xVal = markers_mapped[mar]
            if filter(lambda x: abs(x[0]-xVal)<4, self.selectValues) == [] and xVal<max_mapped:
                self.selectValues.append([xVal, rect.bottom(), atLine.attValues[mar].betaValue])

        self.updateValue()
                
        self.box.hide()
        for p in self.contValues:
            p.hide()
        for l in self.contLabel:
            l.hide()
            l.marker.hide()
        self.line.show()
        self.label.show()

# in this method is implemented a 2-dimensional continuous attribute representation. It is useful, when att. 
# distribution is not monotone
    def paintContinuous(self, canvas, rect, mapper):
        if not self.initialized:
            self.initializeBeforePaint(canvas)
        # rect and att. values initialization
        verticalRect = QRect(rect.top(), rect.left(), rect.height(), rect.width())                        
        verticalMapper = Mapper_Linear_Fixed(self.atNames.minValue, self.atNames.maxValue, verticalRect.left()+verticalRect.width()/4, verticalRect.right(), maxLinearValue = self.atNames.maxValue, minLinearValue = self.atNames.minValue)

        atValues_mapped, atErrors_mapped, min_mapped, max_mapped = mapper(self) # return mapped values, errors, min, max --> mapper(self)
        sortVal = atValues_mapped[:]
        sortVal.sort()

        # draw box
        self.box.setX(sortVal[0])
        self.box.setY(rect.top()+rect.height()/8)
        self.box.setSize(sortVal[len(sortVal)-1]-sortVal[0], rect.height()*7/8)

        # show att. name
        self.label.setText(self.name)
        self.label.setX(sortVal[0])
        self.label.setY(rect.top()+rect.height()/8)
        
        # put labels on it
        label = verticalMapper.getHeaderLine(canvas, verticalRect) 
        atValues_mapped_vertical, atErrors_mapped_vertical, min_mapped_vertical, max_mapped_vertical = verticalMapper(self.atNames) # return mapped values, errors, min, max --> mapper(self)
        mapped_labels, error, min_lab, max_lab = verticalMapper(label) # return mapped values, errors, min, max --> mapper(self)        

        maxPos,zero = 1,0
        while self.attValues[maxPos].betaValue!=0 and self.attValues[maxPos-1].betaValue!=0 and self.attValues[maxPos].betaValue/abs(self.attValues[maxPos].betaValue) == self.attValues[maxPos-1].betaValue/abs(self.attValues[maxPos-1].betaValue):
            maxPos+=1
            if maxPos == len(self.attValues):
                maxPos-=1
                zero = self.attValues[maxPos].betaValue
                break
        #minPos = reduce(lambda x,y: x.betaValue<y.betaValue and x or y, filter(lambda x: x.betaValue>=0, self.attValues))
        #maxNeg = reduce(lambda x,y: x.betaValue>y.betaValue and x or y, filter(lambda x: x.betaValue<=0, self.attValues))
        propBeta = (zero-self.attValues[maxPos-1].betaValue)/(self.attValues[maxPos].betaValue - self.attValues[maxPos-1].betaValue)
        zeroValue = float(self.attValues[maxPos-1].name) + propBeta*(float(self.attValues[maxPos].name) - float(self.attValues[maxPos-1].name))
        self.selectValues = [[mapper.mapBeta(zero, self),verticalMapper.mapBeta(zeroValue, self.atNames), zero, zeroValue]]
        if not self.selectedValue:
            self.selectedValue = self.selectValues[0]
            
        for at in range(len(label.attValues)):
            # draw value
            a = self.contLabel[at]
            a.setX(min_mapped-5)
            a.setY(mapped_labels[at]-canvas.fontSize/2)
            if label.attValues[at].enable:
                a.marker.setPoints(sortVal[0]-2, mapped_labels[at], sortVal[0]+2, mapped_labels[at])
                a.show()
            # if value is disabled, draw just a symbolic line        
            else:
                a.marker.setPoints(sortVal[0]-1, mapped_labels[at], sortVal[0]+1, mapped_labels[at])                
            a.marker.show()
        
        # draw lines
        for i in range(len(atValues_mapped)-1):
            a = self.contValues[i]
            if canvas.parent.histogram:
                a.setPen(QPen(Qt.black, 1+self.attValues[i].lineWidth*canvas.parent.histogram_size))
            else:
                a.setPen(QPen(Qt.black, 1))
            #if self.attValues[i].lineWidth>0:
            a.setPoints(atValues_mapped[i], atValues_mapped_vertical[i], atValues_mapped[i+1], atValues_mapped_vertical[i+1])
            self.selectValues.append([atValues_mapped[i],atValues_mapped_vertical[i], self.attValues[i].betaValue, self.atNames.attValues[i].betaValue])
            # if distance between i and i+1 is large, add some select values.
            n = int(math.sqrt(math.pow(atValues_mapped[i+1]-atValues_mapped[i],2)+math.pow(atValues_mapped_vertical[i+1]-atValues_mapped_vertical[i],2)))/5-1
            self.selectValues = self.selectValues + [[atValues_mapped[i]+(float(j+1)/float(n+1))*(atValues_mapped[i+1]-atValues_mapped[i]),
                                          atValues_mapped_vertical[i]+(float(j+1)/float(n+1))*(atValues_mapped_vertical[i+1]-atValues_mapped_vertical[i]),
                                          self.attValues[i].betaValue+(float(j+1)/float(n+1))*(self.attValues[i+1].betaValue-self.attValues[i].betaValue),
                                          self.atNames.attValues[i].betaValue+(float(j+1)/float(n+1))*(self.atNames.attValues[i+1].betaValue-self.atNames.attValues[i].betaValue)] for j in range(n)]
            a.show()
        self.updateValue()
        self.box.show()
        self.label.show()

        self.line.hide()
        for val in self.attValues:
            if not val.attCreation:
                val.text.hide()
                val.labelMarker.hide()
                val.errorLine.hide()

    # create an AttrLine object from a continuous variable (to many values for a efficient presentation)
    def shrinkSize(self, canvas, width):
        def sign(val1, val2):
            if val1>val2:
                return True
            else:
                return False
            
        maxnum = width/(3*canvas.fontSize)
        if maxnum<2:
            maxnum=2

        step = len(self.attValues)/maxnum
        step = math.floor(step)
        if step<=1:
            return self

        curr_over = True        
        retAttr = AttrLine(self.name, canvas)
        for at in range(len(self.attValues)):
            if len(retAttr.attValues)>1:
                sign_before = sign(retAttr.attValues[len(retAttr.attValues)-1].betaValue, retAttr.attValues[len(retAttr.attValues)-2].betaValue)
            if at%step == 0:
                retAttr.addAttValue(self.attValues[at])
            if len(retAttr.attValues)>2:
                sign_after = sign(retAttr.attValues[len(retAttr.attValues)-1].betaValue, retAttr.attValues[len(retAttr.attValues)-2].betaValue)
                if sign_after != sign_before:
                    retAttr.attValues[len(retAttr.attValues)-1].over = not curr_over
                    curr_over = not curr_over
            
        return retAttr

    # string representation of attribute
    def toString(self):
        return self.name + str([at.toString() for at in self.attValues])

class BasicNomogramHeader(QCanvas):
    def __init__(self, nomogram, parent):
        apply(QCanvas.__init__,(self, parent, ""))
        self.fontSize = parent.fontSize
        self.headerAttrLine = None
        self.nomogram = nomogram
        self.parent = parent
       
    def paintHeader(self, rect, mapper):
        if self.headerAttrLine:
            self.headerAttrLine.destroy()
        self.headerAttrLine = mapper.getHeaderLine(self, rect)
        self.headerAttrLine.name = self.nomogram.parent.pointsName[self.nomogram.parent.yAxis]
        self.headerAttrLine.paint(self, rect, mapper)
        self.resize(self.nomogram.pright, rect.height()+16)
        self.update()

class BasicNomogramFooter(QCanvas):
    def __init__(self, nomogram, parent):
        apply(QCanvas.__init__,(self, parent, ""))
        self.fontSize = parent.fontSize
        self.headerAttrLine = None
        self.nomogram = nomogram
        self.footer = None
        self.footerPercent = None
        self.parent = parent
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
        k = (maxSum-minSum)/(maxSumBeta-minSumBeta)
        if maxSumBeta>3:
            maxSum = (3 - minSumBeta)*k + minSum
            maxSumBeta = 3
        if minSumBeta<-3:
            minSum = (-3 - minSumBeta)*k + minSum
            minSumBeta = -3

        # draw continous line with values from min and max sum (still have values!)
        self.m = Mapper_Linear_Fixed(minSumBeta, maxSumBeta, rect.left(), rect.right(), maxLinearValue = maxSum, minLinearValue = minSum)
        if self.footer:
            self.footer.destroy()
        self.footer = self.m.getHeaderLine(self, QRect(rect.left(), rect.top(), rect.width(), height))
        self.footer.name = self.nomogram.parent.totalPointsName[self.nomogram.parent.yAxis]

        self.footer.paint(self, QRect(rect.left(), rect.top(), rect.width(), height), self.m)

        # continous line convert to percent and draw accordingly (minbeta = minsum)
        if self.footerPercent:
            self.footerPercent.destroy()
        self.footerPercent = self.footer.convertToPercent(self)

        # create a mapper for footer, BZ CHANGE TO CONSIDER THE TARGET
        self.footerPercent.name = "P(%s)" % self.parent.cl.domain.classVar.values[0]
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
            if at.continuous == False:
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
        ax_minError = min(ax_minError, axPercentMax)
        ax_minError = max(ax_minError, axPercentMin)        
        ax_maxError = min(ax_maxError, axPercentMax)
        ax_maxError = max(ax_maxError, axPercentMin)        
        
        self.errorPercentLine.setPoints(ax_minError, self.footerPercent.marker.y(), ax_maxError, self.footerPercent.marker.y())
        
        self.footer.marker.setX(ax)

        if ax>axPercentMax:
            ax=axPercentMax
        if ax<axPercentMin:
            ax=axPercentMin
        self.footerPercent.marker.setX(ax)

        
        if self.parent.probability:
            self.footer.marker.show()
            self.footerPercent.marker.show()
            if self.footer.marker.x() == self.footerPercent.marker.x():
                self.connectedLine.setPoints(self.footer.marker.x(), self.footer.marker.y(), self.footerPercent.marker.x(), self.footerPercent.marker.y())
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
        #self.leftArc.show()
        #self.rightArc.show()
        #self.leftPercentArc.show()
        #self.rightPercentArc.show()

    def hideCI(self):
        self.errorLine.hide()
        self.errorPercentLine.hide()
        self.leftArc.hide()
        self.rightArc.hide()
        self.leftPercentArc.hide()
        self.rightPercentArc.hide()


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
            self.footerCanvas.connectedLine.show()
        self.update()
        self.footerCanvas.update()

    def paint(self, rect, mapper):
        self.zeroLine.setPoints(mapper.mapBeta(0, self.header.headerAttrLine), rect.top(), mapper.mapBeta(0, self.header.headerAttrLine), rect.bottom()+10)
        self.zeroLine.show()
        curr_rect = QRect(rect.left(), rect.top(), rect.width(), 0)
        disc = False
        for at in self.attributes:
            if at.continuous and self.parent.contType == 1:
                if disc:
                    curr_rect = QRect(rect.left(), curr_rect.bottom()+20, rect.width(), self.parent.verticalSpacingContinuous)
                    disc=False
                else:
                    curr_rect = QRect(rect.left(), curr_rect.bottom(), rect.width(), self.parent.verticalSpacingContinuous)
                at.paintContinuous(self, curr_rect, mapper)
            else:
                disc = True
                curr_rect = QRect(rect.left(), curr_rect.bottom(), rect.width(), self.parent.verticalSpacing)
                at.paint(self, curr_rect, mapper)
                # if histograms are used, a larger rect is required
                if self.parent.histogram:
                    curr_rect.setHeight(self.parent.verticalSpacing+self.parent.histogram_size)
            

    def setSizes(self, parent):
        self.pleft, self.pright, self.ptop, self.pbottom = 0, parent.graph.width() - 20, 0, self.parent.verticalSpacing
        disc = False
        for at in self.attributes:
            if self.parent.contType == 1 and at.continuous:
                if disc:
                    self.pbottom+=20
                    disc = False
                self.pbottom += self.parent.verticalSpacingContinuous
            elif self.parent.histogram:
                self.pbottom += self.parent.histogram_size+self.parent.verticalSpacing
                disc = True
            else:
                self.pbottom += self.parent.verticalSpacing
                disc=True

        #graph sizes
        self.gleft = 0
        for at in self.attributes:
            if at.label.boundingRect().width()>self.gleft:
                self.gleft = at.label.boundingRect().width()
        #self.gleft = max(self.gleft, 100) # should really test footer width, and with of other lables
        self.gleft = max(self.gleft, 80)
        self.gleft +=20
        self.gright=self.pright-(self.pright-self.pleft)/10
        self.gtop = self.ptop + 10
        self.gbottom = self.pbottom - 10
        
        if self.parent.table:
            self.parent.verticalTableSpace = 0
            self.gright = self.gright - self.parent.verticalTableSpace

        self.gwidth = self.gright-self.gleft
        self.gheight = self.gbottom - self.gtop

        if self.pbottom < parent.graph.height() - 30:
            self.pbottom = parent.graph.height() - 30
           
    def show(self):
        # set sizes
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


    def getNumOfAtt(self):
        return len(self.attributes)

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




# CANVAS VIEWERS
class OWNomogramHeader(QCanvasView):
    def __init__(self, canvas, mainArea):
        apply(QCanvasView.__init__,(self,)+(canvas,mainArea))

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
#        print "mouseRelease"
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


# ------------------------------------------------------------------------------------------------------------------
# MAPPERS            
# ------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------
class Mapper_Linear_Fixed:
    def __init__(self, minBeta, maxBeta, left, right, maxLinearValue = 100, minLinearValue = -100):
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
        return (betaVal-self.minGraphBeta)/(self.maxGraphBeta-self.minGraphBeta)

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
        dUpper = []
        dLower = []
        if self.maxValue>0 and self.minValue>0:
            dLower = arange(0, self.minValue+dif, dif)
            dLower = map(lambda x:-x, dLower)
            low = dLower[len(dLower)-1]
            dUpper = arange(low, self.maxValue+dif, dif)
            dUpper = map(lambda x:x, dUpper)
            dLower = []
        else:
            if self.maxValue>0:
                dUpper = arange(0, self.maxValue+dif, dif)
                dUpper = map(lambda x:x, dUpper)
            if self.minValue<0:
                dLower = arange(0, -self.minValue+dif, dif)
                dLower = map(lambda x:-x, dLower)
        dSum = unique(dLower+dUpper)
        dSum.sort()
        dSum = filter(lambda x:x>self.minValue-dif, dSum)

        # set new graph values

        k = (self.maxGraphBeta - self.minGraphBeta)/(self.maxGraphValue - self.minGraphValue)

        self.maxGraphBeta = (dSum[len(dSum)-1]- self.minGraphValue)*k + self.minGraphBeta                  
        self.minGraphBeta = (dSum[0]- self.minGraphValue)*k + self.minGraphBeta                  

        self.minGraphValue = dSum[0]
        self.maxGraphValue = dSum[len(dSum)-1]
        
        k = (self.maxGraphBeta-self.minGraphBeta)/(self.maxGraphValue-self.minGraphValue)

        headerLine = AttrLine("Points", canvas)
        for at in range(len(dSum)):
            headerLine.addAttValue(AttValue(str(dSum[at]), self.minGraphBeta + (dSum[at]-self.minGraphValue)*k))
            if at != len(dSum)-1:
                val = AttValue(str((dSum[at]+dSum[at+1])/2), self.minGraphBeta + ((dSum[at]+dSum[at+1])/2-self.minGraphValue)*k)
                val.enable = False
                headerLine.addAttValue(val)
                
        return headerLine


class Mapper_Linear_Center:
    def __init__(self, minBeta, maxBeta, left, right, maxLinearValue = 100, minLinearValue = -100):
        if minBeta == 0:
            self.minBeta = 0.00000001
        else:
            self.minBeta = minBeta
        if maxBeta == 0:
            self.maxBeta = 0.00000001
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
#            print k1, k2
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
            return self.maxGraphValue*attr.maxValue/0.00000001             
        return self.maxGraphValue*attr.maxValue/self.maxGraphBeta

    def getMinValue(self, attr):
        if self.minGraphValue == 0:
            return self.minGraphValue*attr.minValue/0.00000001
        return self.minGraphValue*attr.minValue/self.minGraphBeta

        
    
    # return proportional beta
    def propBeta(self, betaVal, attrLine):        
        return (betaVal-self.minGraphBeta)/(self.maxGraphBeta-self.minGraphBeta)

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
        dUpper = []
        dLower = []
        if self.maxValue>0:
            dUpper = arange(0, self.maxValue+dif, dif)
            dUpper = map(lambda x:x, dUpper)
        if self.minValue<0:
            dLower = arange(0, -self.minValue+dif, dif)
            dLower = map(lambda x:-x, dLower)
        dSum = unique(dLower+dUpper)
        dSum.sort()
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
        k = (self.maxGraphBeta-self.minGraphBeta)/(self.maxGraphValue-self.minGraphValue)

        headerLine = AttrLine("Points", canvas)
        for at in range(len(dSum)):
            headerLine.addAttValue(AttValue(str(dSum[at]), self.minGraphBeta + (dSum[at]-self.minGraphValue)*k))
            if at != len(dSum)-1:
                val = AttValue(str((dSum[at]+dSum[at+1])/2), self.minGraphBeta + ((dSum[at]+dSum[at+1])/2-self.minGraphValue)*k)
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
            k = (at.betaValue-attrLine.minValue)/self.max_difference
            beta.append(self.left + k*(self.right-self.left))
            if self.left + k*(self.right-self.left)>maxb:
                maxb = self.left + k*(self.right-self.left)
            if self.left + k*(self.right-self.left)<minb:
                minb = self.left + k*(self.right-self.left)
            k1 = (at.betaValue-error_factor*at.error-attrLine.minValue)/self.max_difference
            k2 = (at.betaValue+error_factor*at.error-attrLine.minValue)/self.max_difference
            b_error.append([self.left + k1*(self.right-self.left), self.left + k2*(self.right-self.left)])

        if maxb<minb+5:
            maxb=minb+5
        return (beta, b_error, minb, maxb)
    
    def mapBeta(self, betaVal, attrLine):
        k = (betaVal-attrLine.minValue)/self.max_difference
        return self.left+k*(self.right-self.left)        

    def propBeta(self, betaVal, attrLine):        
        return (betaVal-attrLine.minValue)/self.max_difference

    def getMaxMapperValue(self):
        return self.maxLinearValue        
    def getMinMapperValue(self):
        return 0
    def getMaxValue(self, attr):
        return self.maxLinearValue*(attr.maxValue-attr.minValue)/self.max_difference
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
        dif = getDiff(d)
        dSum = []
        dSum = arange(0, self.maxLinearValue+dif, dif)
        dSum = map(lambda x:x, dSum)

        k = self.max_difference/self.maxLinearValue        

        headerLine = AttrLine("", canvas)
        for at in range(len(dSum)):
            headerLine.addAttValue(AttValue(str(dSum[at]), dSum[at]*k))
            # in the middle add disable values, just to see cross lines
            if at != len(dSum)-1:
                val = AttValue(str((dSum[at]+dSum[at+1])/2), (dSum[at]+dSum[at+1])*k/2)
                val.enable = False
                headerLine.addAttValue(val)
                
        return headerLine

# ---------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------

