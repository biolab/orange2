"""
<name>Attribute Statistics</name>
<description>Basic attribute statistics.</description>
<contact>Jure Zabkar (jure.zabkar@fri.uni-lj.si)</contact>
<icon>icons/AttributeStatistics.png</icon>
<priority>200</priority>
"""
#
# OWAttributeStatistics.py
#

#import orange
from OWWidget import *
from OWGUI import *
from OWDlgs import OWChooseImageSizeDlg
import OWQCanvasFuncts
from orngDataCaching import *
import random

class OWAttributeStatistics(OWWidget):
    contextHandlers = {"": DomainContextHandler("", ["HighlightedAttribute"])}
    settingsList = ["sorting"]

    def __init__(self,parent=None, signalManager = None):
        OWWidget.__init__(self, parent, signalManager, "AttributeStatistics", TRUE)

#        self.callbackDeposit = []

        #set default settings
        self.cwbias = 250 # canvas_width = widget_width - 300 pixels
        self.chbias = 30
        self.sorting = 0

        self.cw = self.width()-self.cwbias
        self.ch = self.height()-self.chbias

        #load settings
        self.loadSettings()

        self.dataset = None
        self.canvas = None
        self.HighlightedAttribute = None
        #list inputs and outputs
        self.inputs = [("Data", ExampleTable, self.setData, Default)]
        self.outputs = [("Feature Statistics", ExampleTable)]

        #GUI

        AttsBox = OWGUI.widgetBox(self.controlArea, 'Attributes', addSpace=True)
        self.attributes = OWGUI.listBox(AttsBox, self, selectionMode = QListWidget.SingleSelection, callback = self.attributeHighlighted)
        
        OWGUI.comboBox(self.controlArea, self, "sorting", "Value sorting", items = ["No sorting", "Descending", "Ascending"], callback = self.attributeHighlighted, sendSelectedValue = 0, tooltip = "Should the list of attribute values for discrete attributes be sorted?")
        #self.attributes.setMinimumSize(150, 200)
        #connect controls to appropriate functions

        #OWGUI.separator(self.controlArea, 0,16)

        #give mainArea a layout
        self.canvas = DisplayStatistics(self)
        self.canvas.canvasW, self.canvas.canvasH = self.cw, self.ch
        self.canvasview = QGraphicsView(self.canvas, self.mainArea)
        self.canvasview.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.canvasview.setFocusPolicy(Qt.WheelFocus)
        self.mainArea.layout().addWidget( self.canvasview )

        self.icons = self.createAttributeIconDict()
        self.connect(self.graphButton, SIGNAL("clicked()"), self.saveToFileCanvas)
        self.resize(700, 600)


    def resizeEvent(self, event):
        if self.canvas and self.dataset and self.HighlightedAttribute>=0 and len(self.dataset.domain) < self.HighlightedAttribute:
            # canvas height should be a bit less than the height of the widget frame
            self.ch = self.height()-20
            self.canvas.canvasW, self.canvas.canvasH = self.cw, self.ch
            #self.canvas = DisplayStatistics (self.cw, self.ch)
            # the height of the bar should be 150 pixels smaller than the height of the canvas
            self.canvas.bar_height_pixels = self.height()-150
            #self.canvas.bar_height_pixels = 50
            self.canvas.displayStat(self.dataset, self.HighlightedAttribute, self.dist)
            #self.canvasview.setCanvas(self.canvas)
            #self.canvas.update()

    def attributeHighlighted(self):
        if self.attributes.selectedItems() == []: return
        self.HighlightedAttribute = self.attributes.row(self.attributes.selectedItems()[0])
        self.ch = self.height()-self.chbias
        #self.canvas = DisplayStatistics (self.cw, self.ch)
        self.canvas.canvasW, self.canvas.canvasH = self.cw, self.ch
        self.canvas.bar_height_pixels = self.height()-160
        self.canvas.displayStat(self.dataset, self.HighlightedAttribute, self.dist)
        #self.canvasview.setCanvas(self.canvas)
        #self.canvas.update()


    def setData(self, data):
        self.closeContext()

        self.attributes.clear()
        if data==None:
            self.dataset = self.dist = self.stat = None
            self.canvasview.hide()
            self.send("Feature Statistics", None)
        else:
            self.canvasview.show()

            self.dataset = data
            self.dist = getCached(self.dataset, orange.DomainDistributions, (self.dataset,))

            for a in self.dataset.domain:
                self.attributes.addItem(QListWidgetItem(self.icons[a.varType], a.name))

            self.stat = orange.DomainDistributions(data)
            dt = orange.Domain(data.domain.variables)
            id=orange.newmetaid()
            dt.addmeta(id, orange.StringVariable("statistics"))
            ndata = orange.ExampleTable(dt)
            ndata.append([a.average() if a and a.variable.varType == orange.Variable.Continuous else "" for a in self.stat])
            ndata.append([a.dev() if a and a.variable.varType == orange.Variable.Continuous else "" for a in self.stat])
            ndata.append([a.modus() if a and a.variable.varType == orange.Variable.Discrete else "" for a in self.stat])
            ndata[0][id] = "average"
            ndata[1][id] = "variance"
            ndata[2][id] = "modus"
            self.send("Feature Statistics", ndata)

        self.HighlightedAttribute = 0
        self.openContext("", data)
        self.attributes.setCurrentItem(self.attributes.item(self.HighlightedAttribute))


    def saveToFileCanvas(self):
        sizeDlg = OWChooseImageSizeDlg(self.canvas, parent=self)
        sizeDlg.exec_()

    def sendReport(self):
        if self.dataset:
            self.startReport("%s [%s]" % (self.windowTitle(), self.canvas.attr.name))
            self.reportImage(lambda *x: OWChooseImageSizeDlg(self.canvas).saveImage(*x))
        else:
            self.startReport(self.windowTitle())
"""
class DisplayStatistics
constructs a canvas to display some statistics
"""
class DisplayStatistics (QGraphicsScene):
    def __init__(self, parent = None):
        QGraphicsScene.__init__(self, parent)
        self.bar_height_pixels=None
        self.bar_width_pixels=None
        self.vbias, self.hbias = 60, 200
        self.parent = parent

    def displayStat(self, data, ind, dist):
        if not data:
            return
        self.vbias, self.hbias = 60, 200
        for item in self.items():
            self.removeItem(item)

        self.attr = attr = data.domain[ind]
        attr_name = OWQCanvasFuncts.OWCanvasText(self, attr.name, 10, 10)
        if not dist[ind] or not dist[ind].items():
            if not dist[ind]:
                attr_name.setPlainText("The widget cannot show distributions for attributes of this type.")
            else:
                attr_name.setPlainText("The attribute has no defined values.")
            return
        
        title_str = "Category"
        if attr.varType == orange.VarTypes.Continuous:
            title_str = "Values"
        category = OWQCanvasFuncts.OWCanvasText(self, title_str, self.hbias-20, 30, Qt.AlignRight)

        if attr.varType == orange.VarTypes.Discrete:
            totalvalues = OWQCanvasFuncts.OWCanvasText(self, "Total Values", self.hbias+30, 30)
            rect_len = 100
            rect_width = 20
            attrDist = dist[ind]
            if len(attrDist) > 0 and max(attrDist) > 0:
                if self.parent.sorting == 0:
                    keys = attrDist.keys()
                else:
                    d = [(val, key) for (key, val) in attrDist.items()]
                    d.sort()
                    keys = [item[1] for item in d]
                    if self.parent.sorting == 1:
                        keys.reverse()
                              
                f = rect_len/max(attrDist)
                for key in keys:
                    t = OWQCanvasFuncts.OWCanvasText(self, key, self.hbias-10, self.vbias, Qt.AlignRight)
                    bar_len = attrDist[key]*f
                    if int(bar_len)==0 and bar_len!=0:
                        bar_len=1
                    r = OWQCanvasFuncts.OWCanvasRectangle(self, self.hbias, self.vbias, bar_len, rect_width-2, pen = QPen(Qt.NoPen), brushColor = QColor(0,0,254))

                    t1 = OWQCanvasFuncts.OWCanvasText(self, "%i   (%2.1f %%)" % (attrDist[key], 100*attrDist[key]/(len(data) or 1)), self.hbias+attrDist[key]*rect_len/max(attrDist)+10, self.vbias, Qt.AlignLeft)
                    self.vbias+=rect_width
                if self.vbias > self.canvasH:
                    self.canvasH = self.vbias+50
        if attr.varType == orange.VarTypes.Continuous:
            quartiles_list = reduce(lambda x, y: x+y, [[x[0]]*int(x[1]) for x in dist[ind].items()])
            qlen = len(quartiles_list)
            if qlen%2 == 0:
                self.median = (quartiles_list[qlen/2] + quartiles_list[qlen/2 -1])/2.0
            else:
                self.median = quartiles_list[qlen/2]
            if qlen%4 == 0:
                self.q1 = (quartiles_list[qlen/4] + quartiles_list[qlen/4 -1])/2.0
                self.q3 = (quartiles_list[3*qlen/4] + quartiles_list[3*qlen/4 -1])/2.0
            else:
                self.q1 = quartiles_list[qlen/4]
                self.q3 = quartiles_list[3*qlen/4]
            if self.bar_height_pixels==None: self.bar_height_pixels = 300
            if self.bar_width_pixels==None: self.bar_width_pixels = 40
            self.mini = quartiles_list[0]
            self.maxi = quartiles_list[-1]
            self.total_values = len(quartiles_list)
            self.distinct_values = len(dist[ind])
            self.mean = dist[ind].average()
            self.stddev = dist[ind].dev()
            self.drawCStat(dist[ind])

    def drawCStat(self, dist):
        # draw the main rectangle
        bar_height = self.maxi-self.mini
        #all = QCanvasRectangle (self.hbias, self.vbias, self.bar_width_pixels, self.bar_height_pixels, self)
        #all.show()
        textoffset = 15
        # draw a max line and text
        maxi_txt = OWQCanvasFuncts.OWCanvasText(self, "max")
        # assume equal fonts for all the text
        self.textHeight = maxi_txt.boundingRect().height()
        maxvTextPos = self.vbias - self.textHeight*0.5
        maxi_txt.setPos (self.hbias+self.bar_width_pixels+15, maxvTextPos)

        maxi_txtL = OWQCanvasFuncts.OWCanvasText(self, "%5.2f" % self.maxi, self.hbias-textoffset, maxvTextPos, Qt.AlignRight)

        max_line = OWQCanvasFuncts.OWCanvasLine(self, self.hbias-5, self.vbias, self.hbias+self.bar_width_pixels+5, self.vbias, z = 1)

        # draw a min line and text
        mini_txt = OWQCanvasFuncts.OWCanvasText(self, "min")
        minvTextPos = self.bar_height_pixels+self.vbias - self.textHeight*0.5
        mini_txt.setPos(self.hbias+self.bar_width_pixels+textoffset, minvTextPos)

        mini_txtL = OWQCanvasFuncts.OWCanvasText(self, "%5.2f" % self.mini, self.hbias-textoffset, minvTextPos, Qt.AlignRight)

        min_line = OWQCanvasFuncts.OWCanvasLine(self, self.hbias-5, self.vbias+self.bar_height_pixels, self.hbias+self.bar_width_pixels+5, self.vbias+self.bar_height_pixels, z = 1)

        # draw a rectangle from the 3rd quartile to max; add line and text
        quartile3 =  int(self.bar_height_pixels*(self.maxi-self.q3)/(bar_height or 1))
        crq3 = OWQCanvasFuncts.OWCanvasRectangle(self, self.hbias, self.vbias, self.bar_width_pixels, quartile3, pen = QPen(Qt.NoPen), brushColor = QColor(207, 255, 207))

        q3line = self.vbias + quartile3
        line2 = OWQCanvasFuncts.OWCanvasLine(self, self.hbias-5, q3line, self.hbias+self.bar_width_pixels+5, q3line, z = 1)

        q3vTextPos = q3line - self.textHeight*0.5
        crq3tR = OWQCanvasFuncts.OWCanvasText(self, "75%", self.hbias+self.bar_width_pixels+textoffset, q3vTextPos)

        crq3tL = OWQCanvasFuncts.OWCanvasText(self, "%5.2f" % self.q3, self.hbias-textoffset, q3vTextPos, Qt.AlignRight)

        # draw a rectangle from the median to the 3rd quartile; add line and text
        med = int(self.bar_height_pixels*(self.maxi-self.median)/(bar_height or 1))
        crm = OWQCanvasFuncts.OWCanvasRectangle(self, self.hbias, self.vbias+quartile3, self.bar_width_pixels, med-quartile3, pen = QPen(Qt.NoPen), brushColor = QColor(164, 239, 164))

        mline = self.vbias + med
        line3 = OWQCanvasFuncts.OWCanvasLine(self, self.hbias-5, mline, self.hbias+self.bar_width_pixels+5, mline, z = 1)

        medvTextPos = mline - self.textHeight*0.5
        crmtR = OWQCanvasFuncts.OWCanvasText(self, "median", self.hbias+self.bar_width_pixels+textoffset, medvTextPos)

        crmtL = OWQCanvasFuncts.OWCanvasText(self, "%5.2f" % self.median, self.hbias-textoffset, medvTextPos, Qt.AlignRight)

        # draw a rectangle from the 1st quartile to the median; add line and text
        quartile1 = int(self.bar_height_pixels*(self.maxi-self.q1)/(bar_height or 1))
        crq1 = OWQCanvasFuncts.OWCanvasRectangle(self, self.hbias, self.vbias+med, self.bar_width_pixels, quartile1-med, pen = QPen(Qt.NoPen), brushColor = QColor(126,233,126))
        q1line = self.vbias + quartile1
        line4 = OWQCanvasFuncts.OWCanvasLine(self, self.hbias-5, q1line, self.hbias+self.bar_width_pixels+5, q1line, z = 1)

        q1vTextPos = q1line - self.textHeight*0.5
        crq1tR = OWQCanvasFuncts.OWCanvasText(self, "25%", self.hbias+self.bar_width_pixels+textoffset, q1vTextPos)

        crq1tL = OWQCanvasFuncts.OWCanvasText(self, "%5.2f" % self.q1, self.hbias-textoffset, q1vTextPos, Qt.AlignRight)

        # draw a rectangle from min to the 1st quartile
        cr = OWQCanvasFuncts.OWCanvasRectangle(self, self.hbias, self.vbias+quartile1, self.bar_width_pixels, self.bar_height_pixels-quartile1, pen = QPen(Qt.NoPen), brushColor = QColor(91,207,91))

        # draw a horizontal mean line; add text
        self.meanpos = int(self.bar_height_pixels*(self.maxi-self.mean)/(bar_height or 1))
        self.stddev1 = int(self.bar_height_pixels*self.stddev/(bar_height or 1))
        #print "stddev ",self.stddev1, self.bar_height_pixels, bar_height
        mvbias = self.meanpos+self.vbias
        line = OWQCanvasFuncts.OWCanvasLine(self, self.hbias+self.bar_width_pixels, mvbias, self.hbias+self.bar_width_pixels +70, mvbias, penColor = QColor(255, 0, 0), z = 1)

        meanvTextPos = mvbias - self.textHeight*0.5
        t = OWQCanvasFuncts.OWCanvasText(self, "mean", self.hbias+self.bar_width_pixels+110, meanvTextPos, Qt.AlignRight)
        t.setDefaultTextColor(QColor(255, 0, 0))

        t3 = OWQCanvasFuncts.OWCanvasText(self, "%5.2f +- %5.2f" % (self.mean, self.stddev), self.hbias-textoffset, meanvTextPos, Qt.AlignRight)
        t3.setDefaultTextColor(QColor(255, 0, 0))


        # draw the short bold mean line in the bar
        bline = OWQCanvasFuncts.OWCanvasLine(self, self.hbias+self.bar_width_pixels*0.25, mvbias, self.hbias+self.bar_width_pixels*0.75, mvbias, penColor = QColor(255, 0, 0), penWidth = 3, z = 1)

        # draw the std dev. line
        vert = OWQCanvasFuncts.OWCanvasLine(self, self.hbias+self.bar_width_pixels*0.5, mvbias-self.stddev1, self.hbias+self.bar_width_pixels*0.5, mvbias+self.stddev1, penColor = QColor(255, 0, 0), z = 1)

        # display the numbers of total and distinct values
        t1 = OWQCanvasFuncts.OWCanvasText(self, "%d total values" % self.total_values, 10,self.vbias+self.bar_height_pixels+20)

        t2 = OWQCanvasFuncts.OWCanvasText(self, "%d distinct values" % self.distinct_values, 10,self.vbias+self.bar_height_pixels+40)

        vspace = self.textHeight  # +4 space for text plus 2 pixels above and below
        #pos =['max':maxvTextPos, 'q3':q3vTextPos, 'mean':meanvTextPos, 'med':medvTextPos, 'q1':q1vTextPos, 'min':minvTextPos]
        #positions = [maxvTextPos, q3vTextPos, meanvTextPos, medvTextPos, q1vTextPos, minvTextPos]
        if meanvTextPos < medvTextPos:
            positions = [(maxvTextPos,'max'), (q3vTextPos,'q3'), (meanvTextPos,'mean'), (medvTextPos,'med'), (q1vTextPos,'q1'), (minvTextPos,'min')]
        elif meanvTextPos > medvTextPos:
            positions = [(maxvTextPos,'max'), (q3vTextPos,'q3'), (medvTextPos,'med'), (meanvTextPos,'mean'), (q1vTextPos,'q1'), (minvTextPos,'min')]
        else: # mean == median; put median below or above mean (where there's more space)
            if meanvTextPos-maxvTextPos >= minvTextPos-meanvTextPos:
                positions = [(maxvTextPos,'max'), (q3vTextPos,'q3'), (medvTextPos,'med'), (meanvTextPos,'mean'), (q1vTextPos,'q1'), (minvTextPos,'min')]
            else:
                positions = [(maxvTextPos,'max'), (q3vTextPos,'q3'), (meanvTextPos,'mean'), (medvTextPos,'med'), (q1vTextPos,'q1'), (minvTextPos,'min')]
        lp = len(positions)
        mean_index = -1
        for i in range(len(positions)):
            if positions[i][1]=='mean':
                mean_index = i
                break
        if mean_index==-1:
            print "ERROR in OWAttributeStatistics"
        #above = [positions[i] for i in range(mean_index,-1,-1)]
        #below = [positions[i] for i in range(mean_index, lp)]
        above = [i for i in positions if i[0]<=meanvTextPos]
        below = [i for i in positions if i[0]>=meanvTextPos]
        above.sort()
        above.reverse()
        below.sort()
        above_space = above[0][0] - above[-1][0] - (len(above)-2)*vspace
        below_space = below[-1][0] - below[0][0] - (len(below)-2)*vspace
        for i in range(1,len(above)):
            dif = above[i-1][0] - above[i][0]
            if dif < vspace:
                #if i==len(above)-1:
                #    above[i-1] = (above[i-1][0] + vspace - dif, above[i-1][1])
                #    print "ABOVE 1", i
                #    print
                #else:
                above[i] = (above[i][0] - vspace + dif, above[i][1])
        for i in range(1,len(below)):
            dif = below[i][0] - below[i-1][0]
            if dif < vspace:
                below[i] = (below[i][0] + vspace - dif, below[i][1])
        # move the text to the new coordinates
        for i in range(1,len(above)):
            val, lab = above[i][0], above[i][1]
            if lab == 'max':
                if val != maxvTextPos:
                    maxi_txt.setPos(self.hbias+self.bar_width_pixels+textoffset, val)
                    maxi_txtL.setPos(self.hbias-textoffset, val)
                    OWQCanvasFuncts.OWCanvasLine(self, self.hbias+self.bar_width_pixels+5, self.vbias, self.hbias+self.bar_width_pixels+10, val+self.textHeight*0.5)
                    OWQCanvasFuncts.OWCanvasLine(self, self.hbias+self.bar_width_pixels+10, val+self.textHeight*0.5, self.hbias+self.bar_width_pixels+12, val+self.textHeight*0.5)
                    OWQCanvasFuncts.OWCanvasLine(self, self.hbias-5, self.vbias, self.hbias-10, val+self.textHeight*0.5)
                    OWQCanvasFuncts.OWCanvasLine(self, self.hbias-10, val+self.textHeight*0.5, self.hbias-12, val+self.textHeight*0.5)
            elif lab == 'q3':
                if val != q3vTextPos:
                    crq3tR.setPos(self.hbias+self.bar_width_pixels+textoffset, val)
                    crq3tL.setPos(self.hbias-textoffset, val)
                    OWQCanvasFuncts.OWCanvasLine(self, self.hbias+self.bar_width_pixels+5, q3line, self.hbias+self.bar_width_pixels+10, val+self.textHeight*0.5)
                    OWQCanvasFuncts.OWCanvasLine(self, self.hbias+self.bar_width_pixels+10, val+self.textHeight*0.5, self.hbias+self.bar_width_pixels+12, val+self.textHeight*0.5)
                    OWQCanvasFuncts.OWCanvasLine(self, self.hbias-5, q3line, self.hbias-10, val+self.textHeight*0.5)
                    OWQCanvasFuncts.OWCanvasLine(self, self.hbias-10, val+self.textHeight*0.5, self.hbias-12, val+self.textHeight*0.5)
            elif lab == 'med':
                if val != medvTextPos:
                    crmtR.setPos (self.hbias+self.bar_width_pixels+15, val)
                    crmtL.setPos (self.hbias-15, val)
                    OWQCanvasFuncts.OWCanvasLine(self, self.hbias+self.bar_width_pixels+5, mline, self.hbias+self.bar_width_pixels+10, val+self.textHeight*0.5)
                    OWQCanvasFuncts.OWCanvasLine(self, self.hbias+self.bar_width_pixels+10, val+self.textHeight*0.5, self.hbias+self.bar_width_pixels+12, val+self.textHeight*0.5)
                    OWQCanvasFuncts.OWCanvasLine(self, self.hbias-5, mline, self.hbias-10, val+self.textHeight*0.5)
                    OWQCanvasFuncts.OWCanvasLine(self, self.hbias-10, val+self.textHeight*0.5, self.hbias-12, val+self.textHeight*0.5)
            elif lab == 'q1':
                if val != q1vTextPos:
                    crq1tR.setPos (self.hbias+self.bar_width_pixels+textoffset, val)
                    crq1tL.setPos (self.hbias-textoffset, val)
                    OWQCanvasFuncts.OWCanvasLine(self, self.hbias+self.bar_width_pixels+5, q1line, self.hbias+self.bar_width_pixels+10, val+self.textHeight*0.5)
                    OWQCanvasFuncts.OWCanvasLine(self, self.hbias+self.bar_width_pixels+10, val+self.textHeight*0.5, self.hbias+self.bar_width_pixels+12, val+self.textHeight*0.5)
                    OWQCanvasFuncts.OWCanvasLine(self, self.hbias-5, q1line, self.hbias-10, val+self.textHeight*0.5)
                    OWQCanvasFuncts.OWCanvasLine(self, self.hbias-10, val+self.textHeight*0.5, self.hbias-12, val+self.textHeight*0.5)
        for i in range(1,len(below)):
            val, lab = below[i][0], below[i][1]
            if lab == 'min':
                if val != minvTextPos:
                    mini_txt.setPos (self.hbias+self.bar_width_pixels+textoffset, val)
                    mini_txtL.setPos (self.hbias-textoffset, val)
                    OWQCanvasFuncts.OWCanvasLine(self, self.hbias+self.bar_width_pixels+5, self.bar_height_pixels+self.vbias, self.hbias+self.bar_width_pixels+10, val+self.textHeight*0.5)
                    OWQCanvasFuncts.OWCanvasLine(self, self.hbias+self.bar_width_pixels+10, val+self.textHeight*0.5, self.hbias+self.bar_width_pixels+12, val+self.textHeight*0.5)
                    OWQCanvasFuncts.OWCanvasLine(self, self.hbias-5, self.bar_height_pixels+self.vbias, self.hbias-10, val+self.textHeight*0.5)
                    OWQCanvasFuncts.OWCanvasLine(self, self.hbias-10, val+self.textHeight*0.5, self.hbias-12, val+self.textHeight*0.5)
            elif lab == 'q1':
                if val != q1vTextPos:
                    crq1tR.setPos (self.hbias+self.bar_width_pixels+textoffset, val)
                    crq1tL.setPos (self.hbias-textoffset, val)
                    OWQCanvasFuncts.OWCanvasLine(self, self.hbias+self.bar_width_pixels+5, q1line, self.hbias+self.bar_width_pixels+10, val+self.textHeight*0.5)
                    OWQCanvasFuncts.OWCanvasLine(self, self.hbias+self.bar_width_pixels+10, val+self.textHeight*0.5, self.hbias+self.bar_width_pixels+12, val+self.textHeight*0.5)
                    OWQCanvasFuncts.OWCanvasLine(self, self.hbias-5, q1line, self.hbias-10, val+self.textHeight*0.5)
                    OWQCanvasFuncts.OWCanvasLine(self, self.hbias-10, val+self.textHeight*0.5, self.hbias-12, val+self.textHeight*0.5)
            elif lab == 'med':
                if val != medvTextPos:
                    crmtR.setPos (self.hbias+self.bar_width_pixels+textoffset, val)
                    crmtL.setPos (self.hbias-textoffset, val)
                    OWQCanvasFuncts.OWCanvasLine(self, self.hbias+self.bar_width_pixels+5, mline, self.hbias+self.bar_width_pixels+10, val+self.textHeight*0.5)
                    OWQCanvasFuncts.OWCanvasLine(self, self.hbias+self.bar_width_pixels+10, val+self.textHeight*0.5, self.hbias+self.bar_width_pixels+12, val+self.textHeight*0.5)
                    OWQCanvasFuncts.OWCanvasLine(self, self.hbias-5, mline, self.hbias-10, val+self.textHeight*0.5)
                    OWQCanvasFuncts.OWCanvasLine(self, self.hbias-10, val+self.textHeight*0.5, self.hbias-12, val+self.textHeight*0.5)
            elif lab == 'q3':
                if val != q3vTextPos:
                    crq3tR.setPos (self.hbias+self.bar_width_pixels+textoffset, val)
                    crq3tL.setPos (self.hbias-textoffset, val)
                    OWQCanvasFuncts.OWCanvasLine(self, self.hbias+self.bar_width_pixels+5, q3line, self.hbias+self.bar_width_pixels+10, val+self.textHeight*0.5)
                    OWQCanvasFuncts.OWCanvasLine(self, self.hbias+self.bar_width_pixels+10, val+self.textHeight*0.5, self.hbias+self.bar_width_pixels+12, val+self.textHeight*0.5)
                    OWQCanvasFuncts.OWCanvasLine(self, self.hbias-5, q3line, self.hbias-10, val+self.textHeight*0.5)
                    OWQCanvasFuncts.OWCanvasLine(self, self.hbias-10, val+self.textHeight*0.5, self.hbias-12, val+self.textHeight*0.5)
            if dist.cases <= 1000:
                pen = QPen(Qt.black)
                pen.setWidth(2)
                pen.setStyle(Qt.SolidLine)
                maxs = max(dist.values())
                cent = self.hbias + self.bar_width_pixels/2.
                fact = self.bar_width_pixels * 0.8 / maxs
                for val, occ in dist.items():
                    val = self.vbias+int(self.bar_height_pixels*(self.maxi-val)/(bar_height or 1))
                    if occ > self.bar_width_pixels:
                        r = QGraphicsLineItem(cent-fact*occ, val, cent+fact*occ, val, None, self)
                        r.setPen(pen)
                        r.setZValue(1)
                    else:
                        hocc = (occ-1)/2.
                        for e in range(int(occ)):
                            x = cent + (e - hocc) * fact
                            r = QGraphicsLineItem(x, val, x, val, None, self)
                            r.setPen(pen)
                            r.setZValue(1)
        #print


#test widget appearance
if __name__=="__main__":
    a=QApplication(sys.argv)
    ow=OWAttributeStatistics()
    #data = orange.ExampleTable('adult_sample')
    #data = orange.ExampleTable('adult_sample_noclass')
    data = orange.ExampleTable(r"E:\Development\Orange Datasets\UCI\iris.tab")
    ow.data(data)
    ow.show()
    a.exec_()
    ow.saveSettings()
