#
# OWScatterPlot3DGraph.py
#
# the base for scatterplot

from OWVisGraph import *
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from qtgl import *


###########################################################################################
##### CLASS : OWSCATTERPLOTGRAPH
###########################################################################################
class OWScatterPlot3DGraph(QGLWidget):
    def __init__(self, parent = None, name = None):
        "Constructs the graph"
        QGLWidget.__init__(self, parent, name)

        self.jitterContinuous = 0
        self.enabledLegend = 0
        self.showFilledSymbols = 1
        self.showAttributeValues = 1
        
        self.rawdata = []
        self.pointWidth = 5
        self.scaledData = []
        self.attributeNames = []
        self.jitteringType = 'none'
        self.jitterSize = 10
        self.globalValueScaling = 0

        self.blankClick = 0
        self.statusBar = None
        self.rotateZ = 60.0
        self.rotateXY = 60.0

        self.discreteX = 0
        self.discreteY = 0
        self.discreteZ = 0
        self.dataIsPrepared = 0

    def setJitteringOption(self, jitteringType):
        self.jitteringType = jitteringType

    def setJitterSize(self, size):
        self.jitterSize = size

    def setPointWidth(self, pointWidth):
        self.pointWidth = pointWidth

    def setGlobalValueScaling(self, globalScale):
        self.globalValueScaling = globalScale
    
    def enableGraphLegend(self, enable):
        self.enabledLegend = enable

    def setJitterContinuous(self, enable):
        self.jitterContinuous = enable

    def setShowFilledSymbols(self, filled):
        self.showFilledSymbols = filled

    def setShowAttributeValues(self, show):
        self.showAttributeValues = show

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
    
        # return a list of sorted values for attribute at index index
    def getVariableValuesSorted(self, data, index):
        if data.domain[index].varType == orange.VarTypes.Continuous:
            print "Invalid index for getVariableValuesSorted"
            return []
        
        values = list(data.domain[index].values)
        intValues = []
        i = 0
        # do all attribute values containt integers?
        try:
            while i < len(values):
                temp = int(values[i])
                intValues.append(temp)
                i += 1
        except: pass

        # if all values were intergers, we first sort them ascendently
        if i == len(values):
            intValues.sort()
            values = intValues
        out = []
        for i in range(len(values)):
            out.append(str(values[i]))

        return out

    # create a dictionary with variable at index index. Keys are variable values, key values are indices (transform from string to int)
    # in case all values are integers, we also sort them
    def getVariableValueIndices(self, data, index):
        if data.domain[index].varType == orange.VarTypes.Continuous:
            print "Invalid index for getVariableValueIndices"
            return {}

        values = self.getVariableValuesSorted(data, index)

        dict = {}
        for i in range(len(values)):
            dict[values[i]] = i
        return dict

    #
    # get min and max value of data attribute at index index
    #
    def getMinMaxVal(self, data, index):
        attr = data.domain[index]

        # is the attribute discrete
        if attr.varType == orange.VarTypes.Discrete:
            count = float(len(attr.values))
            return (0, count-1)
                    
        # is the attribute continuous
        else:
            # first find min and max value
            i = 0
            while data[i][attr].isSpecial() == 1: i+=1
            min = data[i][attr].value
            max = data[i][attr].value
            for item in data:
                if item[attr].isSpecial() == 1: continue
                if item[attr].value < min:
                    min = item[attr].value
                elif item[attr].value > max:
                    max = item[attr].value
            return (min, max)
        print "incorrect attribute type for scaling"
        return (0, 1)

    def rescaleAttributesGlobaly(self, data, attrList):
        min = -1; max = -1; first = TRUE
        for attr in attrList:
            if data.domain[attr].varType == orange.VarTypes.Discrete: continue
            index = self.attributeNames.index(attr)
            (minVal, maxVal) = self.getMinMaxVal(data, index)
            if first == TRUE:
                min = minVal; max = maxVal
                first = FALSE
            else:
                if minVal < min: min = minVal
                if maxVal > max: max = maxVal

        for attr in attrList:
            index = self.attributeNames.index(attr)
            scaled, values = self.scaleData(data, index, min, max)
            self.scaledData[index] = scaled
            self.attrValues[attr] = values

    #
    # scale data at index index to the interval 0 - 1
    #
    def scaleData(self, data, index):
        attr = data.domain[index]
        temp = [];
        # is the attribute discrete
        if attr.varType == orange.VarTypes.Discrete:
            # we create a hash table of variable values and their indices
            variableValueIndices = self.getVariableValueIndices(data, index)

            count = float(len(attr.values))
            for i in range(len(data)):
                #val = (1.0 + 2.0*float(variableValueIndices[data[i][index].value])) / float(2*count)
                val = float(variableValueIndices[data[i][index].value]) / float(count)
                temp.append(val)
            return (temp, (0, count-1))

                    
        # is the attribute continuous
        else:
            # first find min and max value
            i = 0
            while data[i][attr].isSpecial() == 1: i+=1
            min = data[i][attr].value
            max = data[i][attr].value
            for item in data:
                if item[attr].isSpecial() == 1: continue
                if item[attr].value < min:
                    min = item[attr].value
                elif item[attr].value > max:
                    max = item[attr].value

            diff = max - min
            # create new list with values scaled from 0 to 1
            for i in range(len(data)):
                temp.append((data[i][attr].value - min) / diff)

            return (temp, (min, max))

    #
    # set new data and scale its values
    #
    def setData(self, data):

        self.rawdata = data
        self.scaledData = []
        self.attributeNames = []
        
        if data == None: return

        self.attrVariance = []
        for index in range(len(data.domain)):
            attr = data.domain[index]
            self.attributeNames.append(attr.name)
            (scaled, variance)= self.scaleData(data, index)
            self.scaledData.append(scaled)
            self.attrVariance.append(variance)

    def initializeGL(self):
        glDisable(GL_LIGHTING)
        #glClearColor(1, 1, 1, 0)
        glClearColor(0, 0, 0, 0)
        #glMatrixMode(GL_PROJECTION)
        #glLoadIdentity()
        #gluLookAt(0,0,0,3,3,3,0,0,1)
        #glMatrixMode(GL_MODELVIEW)
        #gluPerspective(100.0, 1.0, 0, 100)
        glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST)

    def resizeGL(self, width, height):
        glViewport(0,0,width, height)

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glViewport (0, 0, self.width(), self.height())
        
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()

        print self.rotateXY, self.rotateZ        
        gluLookAt(0,0,0,-self.rotateZ,30,self.rotateXY,0,0,1)
        #glRotate(self.rotate, 0,0,1)
        #glRotatef(self.rotateZ, 0.0, 0.0, 1.0)
        #glRotatef(self.rotateXY, 1.0, 0.0, 0.0)
        

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        # draw coordinate axes
        glBegin(GL_LINES)
        glColor3f(0,0,1)
        glVertex4f(0,0,0, 1)
        glVertex4f(2,0,0, 1)
        
        glVertex4f(0,0,0, 1)
        glVertex4f(0,2,0, 1)

        glVertex4f(0,0,0, 1)
        glVertex4f(0,0,2, 1)
        glEnd()

        # show data
        if len(self.scaledData) == 0: return

        xIndex = self.attributeNames.index(self.xAttr)
        yIndex = self.attributeNames.index(self.yAttr)
        zIndex = self.attributeNames.index(self.zAttr)

        for i in range(len(self.rawdata)):
            x = self.scaledData[xIndex][i]*0.5
            y = self.scaledData[yIndex][i]*0.5
            z = self.scaledData[zIndex][i]*0.5

            newColor = QColor(0,0,0)
            
            if self.colorIndex != -1:
                newColor.setHsv(self.scaledData[self.colorIndex][i]*self.MAX_HUE_VAL, 255, 255)
            (r,g,b) = (float(newColor.red())/255.0, float(newColor.green())/255.0, float(newColor.blue())/255.0)

            size = self.pointWidth
            if self.sizeShapeIndex != -1:
                size = self.MIN_SHAPE_SIZE + round(self.scaledData[self.sizeShapeIndex][i] * self.MAX_SHAPE_DIFF)
            size = float(size)/2.0

            glPointSize(size)
            glBegin(GL_POINTS)
            glColor3f(r,g,b)
            glVertex4f(x,y,z,0.5)
            glEnd()
        glFlush()

    #
    # update shown data. Set labels, coloring by className ....
    #
    def updateData(self, xAttr, yAttr, zAttr, colorAttr, shapeAttr = "", sizeShapeAttr = "", showColorLegend = 0, statusBar = None):
        (xVarMin, xVarMax) = self.attrVariance[self.attributeNames.index(xAttr)]
        (yVarMin, yVarMax) = self.attrVariance[self.attributeNames.index(yAttr)]
        (zVarMin, zVarMax) = self.attrVariance[self.attributeNames.index(zAttr)]
        self.xVar = xVarMax - xVarMin
        self.yVar = yVarMax - yVarMin
        self.zVar = zVarMax - zVarMin

        self.xAttr = xAttr
        self.yAttr = yAttr
        self.zAttr = zAttr
        self.colorAttr = colorAttr
        self.shapeAttr = shapeAttr
        self.sizeShapeAttr = sizeShapeAttr
        self.showColorLegend = showColorLegend
        self.statusBar = statusBar

        self.MIN_SHAPE_SIZE = 10
        self.MAX_SHAPE_DIFF = 10
        self.MAX_HUE_VALUE = 300
        self.colorIndex = -1
        if colorAttr != "" and colorAttr != "(One color)":
            self.colorIndex = self.attributeNames.index(colorAttr)
            if self.rawdata.domain[colorAttr].varType == orange.VarTypes.Discrete: self.MAX_HUE_VAL = 360

        self.sizeShapeIndex = -1
        if sizeShapeAttr != "" and sizeShapeAttr != "(One size)":
            self.sizeShapeIndex = self.attributeNames.index(sizeShapeAttr)

        # create hash tables in case of discrete X axis attribute
        self.attrXIndices = {}
        self.discreteX = 0
        if self.rawdata.domain[xAttr].varType == orange.VarTypes.Discrete:
            self.discreteX = 1
            self.attrXIndices = self.getVariableValueIndices(self.rawdata, xAttr)

        # create hash tables in case of discrete Y axis attribute
        self.attrYIndices = {}
        self.discreteY = 0
        if self.rawdata.domain[yAttr].varType == orange.VarTypes.Discrete:
            self.discreteY = 1
            self.attrYIndices = self.getVariableValueIndices(self.rawdata, yAttr)

        # create hash tables in case of discrete Z axis attribute
        self.attrZIndices = {}
        self.discreteZ = 0
        if self.rawdata.domain[zAttr].varType == orange.VarTypes.Discrete:
            self.discreteZ = 1
            self.attrZIndices = self.getVariableValueIndices(self.rawdata, zAttr)

        self.dataIsPrepared = 1
        self.paintGL()

    def mousePressEvent(self, ev):
        self.exMousePos = QPoint(ev.pos().x(), ev.pos().y())

    def mouseMoveEvent(self, ev):
        x  = ev.x() - self.exMousePos.x()
        y  = ev.y() - self.exMousePos.y()
        self.rotateZ += x
        self.rotateXY += y
        self.exMousePos = QPoint(ev.pos().x(), ev.pos().y())
        self.updateGL ()

    
if __name__== "__main__":
    #Draw a simple graph
    a = QApplication(sys.argv)        
    c = OWScatterPlot3DGraph()
        
    a.setMainWidget(c)
    c.show()
    a.exec_loop()
