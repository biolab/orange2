#
# OW_NomogramGraph_Martin --> this name is temporarily, change it as
# soon as possible !!!
# draw a nomogram (left scaled or 0-point scaled)
#

from OWWidget import *
from Nomogram_Mappers import *
from Numeric import *

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
        

#-----------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------
    
# this class represents single value in nomogram
# Attributes:
#  * name - used as label in nomogram
#  * betaValue - beta value in model
#  * error - error value -> beta = beta +-error 
#  * enable - paint value?
#  * coordinatesSet - are actual (pixel) coordinates for beta and beta error already set or not
#  * x - x coordinate (actual pixel) in nomogram
#  * errorX - absolute pixel value of coordinates
#  * showErr - show errors or not?

class AttValue:
    def __init__(self, name, betaValue, error=0, showErr=False, over=True):
        self.name = name
        self.betaValue = betaValue
        self.error = error
        self.enable = True
        self.coordinatesSet = False # True if coordinates for this value are set already --> uses in paint function
        self.showErr = showErr
        self.over = over
    def __item__(self, i):
        return self.betaValue

    def disable(self):
        self.enable = False
    def enable(self):
        self.enable = True
    def enabled(self):
        return self.enable
    def showErrors(self, showErr):
        self.showErr = showErr

    # set x coordinates
    # Coordinates should be set before the use of paint function, because
    # it is not always possible to estimate value's position. Sometimes value's position
    # depends on other values!
    # after each use of paint function, coordinates have to be set again!    
    def setCoordinates(self, x, errorX):
        self.x = x
        self.errorX = errorX
        self.coordinatesSet = True
    def resetCoordinates(self):
        self.coordinatesSet = False
    
    def paint(self, painter, rect, mapper):
        if not self.coordinatesSet:
            return
        
        # draw value
        if self.enabled():
            if self.over:
                painter.drawText(self.x-len(self.name)*painter.font().pixelSize()/2, rect.bottom()-4*painter.font().pixelSize()/3, len(self.name)*painter.font().pixelSize(), painter.font().pixelSize(), Qt.AlignCenter, self.name)
                painter.drawLine(self.x, rect.bottom(), self.x, rect.bottom()+painter.font().pixelSize()/3)
            else:
                painter.drawText(self.x-len(self.name)*painter.font().pixelSize()/2, rect.bottom()+painter.font().pixelSize()/5, len(self.name)*painter.font().pixelSize(), painter.font().pixelSize(), Qt.AlignCenter, self.name)
                painter.drawLine(self.x, rect.bottom(), self.x, rect.bottom()-painter.font().pixelSize()/3)
            if self.showErr:
                painter.drawLine(self.x-self.errorX, rect.bottom()+1, self.x+self.errorX, rect.bottom()+1)
        # if value is disabled, draw just a simbolic line        
        else:
            painter.drawLine(self.x, rect.bottom(), self.x, rect.bottom()+painter.font().pixelSize()/6)
        # after each use of paint function, coordinates have to be set again!    
        self.resetCoordinates()

    # string representation of attribute value        
    def toString(self):
        return self.name + ":" + str(self.betaValue)


# this class represent attribute in nomogram
# Attributes:
#   * name - name of attribute
#   * TDposition - line number
#   * attValues - set of values
#   * minValue, maxValue - minValues and maxValues of attribute
class AttrLine:
    def __init__(self, name, TDposition, continuous = False):
        self.name = name
        self.TDposition = TDposition
        self.attValues = []
        self.minValue = self.maxValue = 0
        self.continuous = continuous

    def addAttValue(self, attValue):
        self.attValues.append(attValue)
        if attValue.betaValue>self.maxValue:
            self.maxValue = attValue.betaValue
        if attValue.betaValue<self.minValue:
            self.minValue = attValue.betaValue
    def removeAttValue(self, attValue):
        self.attValues.remove(attValue)
        # change max and min value TODO
        
    def getAttValue(self, name):
        for at in self.attValues:
            if at.getName() == name:
                return at

        return None
    def setName(self, name):
        self.name = name
    
    def getTDPosition(self):
        return self.TDposition
    def getAttValues(self):
        return self.attValues

    def getMinValue(self):
        return self.minValue
    def getMaxValue(self):
        return self.maxValue

    def convertToPercent(self, constant, painter, rect):
        minPercent = exp(self.minValue)/(1+exp(self.minValue))
        maxPercent = exp(self.maxValue)/(1+exp(self.maxValue))

        percentList = filter(lambda x:x>minPercent and x<1,arange(0, maxPercent+0.1, 0.1))
        self.attValues = []
        for p in percentList:
            self.attValues.append(AttValue(str(p), log(p/(1-p))))
        return self

    def paint(self, painter, rect, mapper):
        # draw label in bold
        painter.save()
        font = QFont("Arial", painter.font().pointSize())
        font.setBold(True)
        painter.setFont(font)        
        atValues_mapped, atErrors_mapped, min_mapped, max_mapped = mapper(self) # return mapped values, errors, min, max --> mapper(self)
        painter.drawText(0, rect.bottom()-painter.font().pixelSize()/2, min_mapped-2, painter.font().pixelSize(), Qt.AlignRight, self.name)
        painter.restore()

        # draw attribute line
        painter.drawLine(min_mapped, rect.bottom(), max_mapped, rect.bottom())

        # continuous attributes are handled differently
        if self.continuous:
            cAtt = self.shrinkSize(painter.font().pixelSize(), max_mapped - min_mapped)
            atValues_mapped, atErrors_mapped, min_mapped, max_mapped = mapper(cAtt) # return mapped values, errors, min, max --> mapper(self)
            val = cAtt.attValues
            for i in range(len(val)):
                # check attribute name that will not cover another name
                for j in range(i):
                    if val[j].over==val[i].over and val[j].enabled() and abs(atValues_mapped[j]-atValues_mapped[i])<(len(val[j].name)*painter.font().pixelSize()/4+len(val[i].name)*painter.font().pixelSize()/4):
                        val[i].disable()
                        
                val[i].setCoordinates(atValues_mapped[i], atErrors_mapped[i])
                val[i].paint(painter, rect, mapper)
            
        else:
            # draw attributes
            val = self.attValues
            for i in range(len(val)):
                # check attribute name that will not cover another name
                for j in range(i):
                    if val[j].over and val[j].enabled() and abs(atValues_mapped[j]-atValues_mapped[i])<(len(val[j].name)*painter.font().pixelSize()/4+len(val[i].name)*painter.font().pixelSize()/4):
                        #print "m", atValues_mapped[j], atValues_mapped[i], val[i].name, val[j].name
                        val[i].over = False
                        
                val[i].setCoordinates(atValues_mapped[i], atErrors_mapped[i])
                val[i].paint(painter, rect, mapper)
            for i in range(len(val)):
                val[i].over = True


# in this method is implemented a 2-dimensional continuous attribute representation. It is useful, when att. 
# distribution is not monotone
    def paintContinuous(self, painter, rect, mapper):
        # rect and att. values initialization
        verticalRect = QRect(rect.top(), rect.left(), rect.height(), rect.width())                        

        atNames = AttrLine(self.name, 0)
        for at in self.attValues:
            atNames.addAttValue(AttValue(at.name, float(at.name)))

        verticalMapper = Mapper_Linear_Fixed(atNames.getMinValue(), atNames.getMaxValue(), verticalRect.left()+verticalRect.width()/4, verticalRect.right(), maxLinearValue = atNames.getMaxValue(), minLinearValue = atNames.getMinValue())
        atValues_mapped_vertical, atErrors_mapped_vertical, min_mapped_vertical, max_mapped_vertical = verticalMapper(atNames) # return mapped values, errors, min, max --> mapper(self)

        atValues_mapped, atErrors_mapped, min_mapped, max_mapped = mapper(self) # return mapped values, errors, min, max --> mapper(self)
        # draw box
        sortVal = atValues_mapped[:]
        sortVal.sort()
        painter.setPen(Qt.DotLine)
        painter.drawRect(sortVal[0], rect.top()+rect.height()/8, sortVal[len(sortVal)-1]-sortVal[0], rect.height()*7/8)
        painter.setPen(Qt.SolidLine)

        # show att. name
        painter.save()
        font = QFont("Arial", painter.font().pointSize())
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(sortVal[0], rect.top()+rect.height()/8, sortVal[len(sortVal)-1]-sortVal[0], painter.font().pixelSize()*2, Qt.AlignCenter, self.name)
        painter.restore()
        
        # put labels on it
        label = verticalMapper.getHeaderLine(painter, verticalRect) 
        mapped_labels, error, min_lab, max_lab = verticalMapper(label) # return mapped values, errors, min, max --> mapper(self)        
        for at in range(len(label.attValues)):
            # draw value
            if label.attValues[at].enabled():
                painter.drawText(0, mapped_labels[at]-painter.font().pixelSize()/2, sortVal[0]-3, painter.font().pixelSize(), Qt.AlignRight, label.attValues[at].name)
                painter.drawLine(sortVal[0]-2, mapped_labels[at], sortVal[0]+2, mapped_labels[at])
            # if value is disabled, draw just a simbolic line        
            else:
                painter.drawLine(sortVal[0]-1, mapped_labels[at], sortVal[0]+1, mapped_labels[at])

        # draw lines
        for i in range(len(atValues_mapped)-1):
            painter.drawLine(atValues_mapped[i], atValues_mapped_vertical[i], atValues_mapped[i+1], atValues_mapped_vertical[i+1])

    # create an AttrLine object from a continuous variable (to many values for a efficient presentation)
    def shrinkSize(self, fontSize, width):
        def sign(val1, val2):
            if val1>val2:
                return True
            else:
                return False
            
        maxnum = width/(3*fontSize)
        if maxnum<2:
            maxnum=2

        step = len(self.attValues)/maxnum
        step = math.floor(step)
        if step<=1:
            return self

        curr_over = True        
        retAttr = AttrLine(self.name, self.TDposition)
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
        

class BasicNomogram:
    def __init__(self, constant):
        self.attributes = []
        self.constant = constant
        self.minBeta = 0
        self.maxBeta = 0
        self.max_difference = 0

    def addAttribute(self, attr):
        self.attributes.append(attr)
        if attr.getMinValue() < self.minBeta:
            self.minBeta = attr.getMinValue()
        if attr.getMaxValue() > self.maxBeta:
            self.maxBeta = attr.getMaxValue()
        if attr.getMaxValue()-attr.getMinValue() > self.max_difference:
            self.max_difference = attr.getMaxValue()-attr.getMinValue()

    def paintHeader(self, painter, rect, mapper):
        headerAttrLine = mapper.getHeaderLine(painter, rect)
        headerAttrLine.setName("")
        headerAttrLine.paint(painter, rect, mapper)

    def paint(self, parent, painter, rect, mapper):
        painter.setPen(Qt.DotLine)
        painter.drawLine(mapper.mapBeta(0), rect.top(), mapper.mapBeta(0), rect.bottom()+10)
        painter.setPen(Qt.SolidLine)
        curr_rect = QRect(rect.left(), rect.top(), rect.width(), 0)
        for at in self.attributes:
            if at.continuous and parent.continuous == 1:
                curr_rect = QRect(rect.left(), curr_rect.bottom(), rect.width(), parent.verticalSpacingContinuous)
                at.paintContinuous(painter, curr_rect, mapper)
            else:
                curr_rect = QRect(rect.left(), curr_rect.bottom(), rect.width(), parent.verticalSpacing)
                at.paint(painter, curr_rect, mapper)
            
    def paintFooter(self, painter, rect, alignType, yAxis, mapper):
        # set height for each scale        
        height = rect.height()/2
        
        # get min and maximum sum, min and maximum beta
        # min beta <--> min sum! , same for maximum
        maxSum = minSum = maxSumBeta = minSumBeta = 0
        for at in self.attributes:
            maxSum += mapper.getMaxValue(at)
            minSum += mapper.getMinValue(at)
            maxSumBeta += at.getMaxValue()
            minSumBeta += at.getMinValue()

        # add constant to betas!
        maxSumBeta += self.constant.betaValue
        minSumBeta += self.constant.betaValue
        
        # show only reasonable values
        k = (maxSum-minSum)/(maxSumBeta-minSumBeta)
        if maxSumBeta>3:
            maxSum = (3 - minSumBeta)*k + minSum
            maxSumBeta = 3
        if minSumBeta<-3:
            minSum = (-3 - minSumBeta)*k + minSum
            minSumBeta = -3

        # draw continous line with values from min and max sum (still have values!)
        m = Mapper_Linear_Fixed(minSumBeta, maxSumBeta, rect.left(), rect.right(), maxLinearValue = maxSum, minLinearValue = minSum)
        footer = m.getHeaderLine(painter, QRect(rect.left(), rect.top(), rect.width(), height))
        #print footer.toString()
        footer.setName("sum")
        footer.paint(painter, QRect(rect.left(), rect.top(), rect.width(), height), m)

        # continous line convert to percent and draw accordingly (minbeta = minsum)        
        footer = footer.convertToPercent(mapper.getBetaDelay(self), painter, rect)


        # create a mapper for footer
        footer.setName("P")
        footer.paint(painter, QRect(rect.left(), rect.top()+height, rect.width(), height), m)                         
        
    def getNumOfAtt(self):
        return len(self.attributes)

    def printOUT(self):
        print "constant:", str(self.constant.betaValue)
        for a in self.attributes:
            print a.toString()


# here should be a class for a table (perhaps I will use bnomogram or not)

class OWNomogramGraph(QScrollView):
    bnomogram = None
    def __init__(self, parent=None, name=None):
        self.parentWidget = parent
        self.graphManager = None
        QScrollView.__init__(self,parent,name,Qt.WResizeNoErase | Qt.WRepaintNoErase)                
        self.setWFlags(Qt.WResizeNoErase | Qt.WRepaintNoErase) #this works like magic.. no flicker during repaint!
        self.setBackgroundMode(QWidget.NoBackground)
        self.viewport().setBackgroundMode(QWidget.NoBackground) # This does the trick. Above just refuses to work
        self.buffer=QPixmap() #off-screen buffer

        self.alignType = 1 # use 0 for left alignment and 1 for 0-point alignment, where 0-point is where place where attribute has no influence on class 
        self.yAxis = 0 # 0 - normalize to 0-100, 1 - beta coeff, 2 - odds ration
        self.showErrors = 0
        self.verticalSpacing = 40
        self.verticalSpacingContinuous = 100
        self.fontSize = 9
        self.lineWidth = 1
        self.showPercentage = 1
        self.showTable = 0
        self.continuous = 1 # 0 - 1d presentation, 1 - 2d presentation

        self.wl = 0 #width of space for the left y axis labels
        psize=self.size()
        self.setSizes(self.rect())
        self.buffer.resize(self.size()) #make same size as widget
        self.repaintGraph()

    def resizeBuffer(self, newSize):
        bufSize = self.buffer.size()
        if (newSize.width() != bufSize.width()) or (newSize.height() != bufSize.height()):
            self.buffer.resize(newSize)
            
    def resizeEvent(self,qre):
        psize=qre.size()
        #TODO: change it so that will not bne possible to resize
        if psize.width()<250:
            return
        self.setSizes(psize)
        if self.bnomogram!=None:
            hgt =  5*self.verticalSpacing
            for at in self.bnomogram.attributes:
                if self.continuous == 1 and at.continuous:
                    hgt += self.verticalSpacingContinuous
                else:
                    hgt += self.verticalSpacing
            self.resizeContents(psize.width()-30, hgt)            

    def setSizes(self,psize):
        "Sets internal variables for window borders and graph sizes used in drawing"
        #window borders
        self.pleft=0
        self.pright=psize.width()
        self.ptop = 0
        if self.bnomogram == None:
            self.pbottom = 100
        else:
            self.pbottom =  5*self.verticalSpacing
            for at in self.bnomogram.attributes:
                if self.continuous == 1 and at.continuous:
                    self.pbottom += self.verticalSpacingContinuous
                else:
                    self.pbottom += self.verticalSpacing

        #graph sizes
        self.gleft=self.pleft+(self.pright-self.pleft)/10
        self.gright=self.pright-(self.pright-self.pleft)/10
        self.gtop = self.ptop + 10
        self.gbottom = self.pbottom - 10
        
        if self.showTable:
            self.verticalTableSpace = 0
            self.gright = self.gright - self.verticalTableSpace

        self.gwidth = self.gright-self.gleft
        self.gheight = self.gbottom - self.gtop
        return (self.pright, self.pbottom)
            

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def drawContents(self, painter, clipx=0, clipy=0, clipw=0, cliph=0):
        """
        Paints the graph (graph area + calls paintGraph to paint the graph).
        Called whenever repaint is needed by the system
        or user explicitly calls repaint()
        """
        # drawContents gets called with 2 parameters only ocassionaly. This is
        # to prevent error messages from appearing
        if clipw == 0 and cliph == 0:
            return
        psize=self.size()
        self.resizeBuffer(QSize(clipw, cliph))
        offScreenPainter=QPainter(self.buffer) #create a painter in the off-screen buffer and paint onto it
        prect=QRect(clipx, clipy, clipw, cliph)

        offScreenPainter.setWindow(clipx, clipy, clipw, cliph)
        offScreenPainter.setViewport(0, 0, clipw, cliph)
        self.gx = 0
        self.gy = 0


        #fill the background with white color
        offScreenPainter.fillRect(prect, QBrush(QColor(255,255,255)))

        pen=QPen(Qt.black,1,Qt.SolidLine)
        offScreenPainter.setPen(pen)

        # Draw the actual graph
        self.paintGraph(offScreenPainter, QRect(clipx, clipy, clipw, cliph),
                                          QRect(self.contentsX(), self.contentsY(), self.contentsWidth(), self.contentsHeight()))

        #copy from off-screen buffer
        bitBlt(painter.device(), QPoint(clipx-self.contentsX(), clipy-self.contentsY()),
               self.buffer,      QRect(0, 0, clipw, cliph),
               Qt.CopyROP)

        

    # actually draw nomogram    
    def paintGraph(self,painter,rect,visibleRect):
        """
        Draws the actual Nomogram

        """
        if self.bnomogram == None:
            return
        # returns beta values from self.items
        
        blackPen=QColor()
        blackPen.setHsv(0, 0, 0)

        # Define font sizes for Y axis labels
        fontsize = self.fontSize
        font = QFont("Arial", fontsize)
        painter.setFont(font)
        painter.save()
        
        
        curr_point = self.verticalSpacing
        if self.alignType == 0:
            self.mapper = Mapper_Linear_Left(self.bnomogram.max_difference,  self.gleft, self.gright)
        else:
            self.mapper = Mapper_Linear_Center(self.bnomogram.minBeta, self.bnomogram.maxBeta, self.gleft, self.gright)
        
        # draw HEADER and vertical line
        topRect=QRect(self.gleft, self.gtop, self.gwidth, self.verticalSpacing)
        self.bnomogram.paintHeader(painter, topRect, self.mapper) 

        # draw nomogram
        middleRect=QRect(self.gleft, self.gtop+self.verticalSpacing, self.gwidth, self.gheight-4*self.verticalSpacing)
        self.bnomogram.paint(self, painter, middleRect, self.mapper)
        #self.bnomogram.paint(painter, rect)

        # draw final line
        bottomRect=QRect(self.gleft, self.gbottom-2*self.verticalSpacing, self.gwidth, 2*self.verticalSpacing)
        self.bnomogram.paintFooter(painter, bottomRect, self.alignType, self.yAxis, self.mapper)

        
        
    def setNomogramData(self, bnomogram):
        self.bnomogram = bnomogram
        
        w,h=self.updateAll()
        self.resizeContents(w-30, h)        

    def setAlignType(self, alignType):
        print "grem v setAlignType"
        self.alignType = alignType
        self.updateAll()

    def setContType(self, contType):
        self.continuous = contType
        w,h=self.updateAll()
        self.resizeContents(w-30, h)        

    def updateAll(self):    
        psize=self.size()
        w,h=self.setSizes(psize)
        self.buffer.resize(self.size()) #make same size as widget
        self.repaintGraph()
        return (w,h)
            
        
    def repaintGraph(self):
        """Repaints the visible viewport region. Plain repaint() does not work."""
        self.repaintContents(self.contentsX(), self.contentsY(), self.contentsWidth(), self.contentsHeight(), 1)



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

    def __call__(self, attrLine):
        beta = []
        b_error = []
        max_mapped = self.left
        min_mapped = self.right
        for at in attrLine.attValues:
            k = self.propBeta(at.betaValue)
            beta.append(self.left+k*(self.right-self.left))
            if self.left+k*(self.right-self.left)>max_mapped:
                max_mapped = self.left+k*(self.right-self.left)
            if self.left+k*(self.right-self.left)<min_mapped:
                min_mapped = self.left+k*(self.right-self.left)
            k1 = self.propBeta(at.betaValue+at.error)
            b_error.append((k1-k)*(self.right-self.left))
        return (beta, b_error, min_mapped, max_mapped)

    def mapBeta(self, betaVal):
        k = self.propBeta(betaVal)
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
    def propBeta(self, betaVal):        
        return (betaVal-self.minGraphBeta)/(self.maxGraphBeta-self.minGraphBeta)

    # delay / offset that a mapper produces
    # in this case no aligning is uses, that is why delay is always 0
    def getDelay(self, nomogram):
        return 0
    
    def getHeaderLine(self, painter, rect):
        maxnum = rect.width()/(3*painter.font().pixelSize())
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
        dSum = filter(lambda x:x>self.minValue-dif, dSum)

        # set new graph values
        k = (self.maxGraphBeta - self.minGraphBeta)/(self.maxGraphValue - self.minGraphValue)
    

        self.maxGraphBeta = (dSum[len(dSum)-1]- self.minGraphValue)*k + self.minGraphBeta                  
        self.minGraphBeta = (dSum[0]- self.minGraphValue)*k + self.minGraphBeta                  
        
        self.minGraphValue = dSum[0]
        self.maxGraphValue = dSum[len(dSum)-1]

        k = (self.maxGraphBeta-self.minGraphBeta)/(self.maxGraphValue-self.minGraphValue)

        headerLine = AttrLine("header", 0)
        for at in range(len(dSum)):
            headerLine.addAttValue(AttValue(str(dSum[at]), self.minGraphBeta + (dSum[at]-self.minGraphValue)*k))
            if at != len(dSum)-1:
                val = AttValue(str((dSum[at]+dSum[at+1])/2), self.minGraphBeta + ((dSum[at]+dSum[at+1])/2-self.minGraphValue)*k)
                val.disable()
                headerLine.addAttValue(val)
                
        return headerLine


class Mapper_Linear_Center:
    def __init__(self, minBeta, maxBeta, left, right, maxLinearValue = 100, minLinearValue = -100):
        self.minBeta = minBeta
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

    def __call__(self, attrLine):
        beta = []
        b_error = []
        max_mapped = self.left
        min_mapped = self.right
        for at in attrLine.attValues:
            k = self.propBeta(at.betaValue)
            beta.append(self.left+k*(self.right-self.left))
            if self.left+k*(self.right-self.left)>max_mapped:
                max_mapped = self.left+k*(self.right-self.left)
            if self.left+k*(self.right-self.left)<min_mapped:
                min_mapped = self.left+k*(self.right-self.left)
            k1 = self.propBeta(at.betaValue+at.error)
            b_error.append((k1-k)*(self.right-self.left))
        return (beta, b_error, min_mapped, max_mapped)

    def mapBeta(self, betaVal):
        k = self.propBeta(betaVal)
        return self.left+k*(self.right-self.left)        

    def getLeftMost(self):
        return self(self.minGraphBeta)

    def getRightMost(self):
        return self(self.maxGraphBeta)

    def getMaxValue(self, attr):
        return self.maxGraphValue*attr.maxValue/self.maxGraphBeta

    def getMinValue(self, attr):
        return self.minGraphValue*attr.minValue/self.minGraphBeta

        
    
    # return proportional beta
    def propBeta(self, betaVal):        
        return (betaVal-self.minGraphBeta)/(self.maxGraphBeta-self.minGraphBeta)

    # delay / offset that a mapper produces
    # in this case no aligning is uses, that is why delay is always 0
    def getBetaDelay(self, nomogram):
        return 0
    
    def getHeaderLine(self, painter, rect):
        maxnum = rect.width()/(3*painter.font().pixelSize())
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

        headerLine = AttrLine("header", 0)
        for at in range(len(dSum)):
            headerLine.addAttValue(AttValue(str(dSum[at]), self.minGraphBeta + (dSum[at]-self.minGraphValue)*k))
            if at != len(dSum)-1:
                val = AttValue(str((dSum[at]+dSum[at+1])/2), self.minGraphBeta + ((dSum[at]+dSum[at+1])/2-self.minGraphValue)*k)
                val.disable()
                headerLine.addAttValue(val)
                
        return headerLine


# it is very similar to Mapper_Linear_Center. It has the same methods, implementation is slightly different
class Mapper_Linear_Left:
    def __init__(self, max_difference, left, right, maxLinearValue = 100):
        self.max_difference = max_difference
        self.left = left
        self.right = right
        self.maxLinearValue = maxLinearValue
    
    def __call__(self, attrLine):
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
            k1 = at.error/self.max_difference
            b_error.append(k1*(self.right-self.left))
        return (beta, b_error, minb, maxb)
    
    def mapBeta(self, betaVal):
        k = self.propBeta(betaVal)
        return self.left+k*(self.right-self.left)        

    # return proportional beta --> to ni pravilno TODO!
    def propBeta(self, betaVal):        
        return betaVal/self.max_difference

    def getMaxValue(self):
        return self.maxLinearValue        
    def getMinValue(self):
        return 0
    def getMaxValue(self, attr):
        return self.maxLinearValue*(attr.maxValue-attr.minValue)/self.max_difference
    def getMinValue(self, attr):
        return 0

    def getBetaDelay(self, nomogram):
        delay = 0
        for at in nomogram.attributes:
            delay += at.getMinValue()
        return delay
        
    def getHeaderLine(self, painter, rect):
        maxnum = rect.width()/(3*painter.font().pixelSize())
        if maxnum<1:
            maxnum=1
        d = self.maxLinearValue/maxnum
        dif = getDiff(d)
        dSum = []
        dSum = arange(0, self.maxLinearValue+dif, dif)
        dSum = map(lambda x:x, dSum)

        k = self.max_difference/self.maxLinearValue        

        headerLine = AttrLine("", 0)
        for at in range(len(dSum)):
            headerLine.addAttValue(AttValue(str(dSum[at]), dSum[at]*k))
            # in the middle add disable values, just to see cross lines
            if at != len(dSum)-1:
                val = AttValue(str((dSum[at]+dSum[at+1])/2), (dSum[at]+dSum[at+1])*k/2)
                val.disable()
                headerLine.addAttValue(val)
                
        return headerLine

# ---------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------
