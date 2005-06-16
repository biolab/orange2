"""
<name>Attribute Statistics</name>
<description>Show basic statistics about attributes.</description>
<icon>icons/AttributeStatistics.png</icon>
<priority>200</priority>
"""
#
# OWAttributeStatistics.py
#

#import orange
from qtcanvas import *
from OWWidget import *
from OWGUI import *

class OWAttributeStatistics(OWWidget):
    settingsList=["LastAttributeSelected"]

    def __init__(self,parent=None, signalManager = None):
        OWWidget.__init__(self, parent, signalManager, "AttributeStatistics", TRUE)

        self.callbackDeposit = []

        #set default settings
        self.cwbias = 300 # canvas_width = widget_width - 300 pixels
        self.chbias = 30

        self.cw = self.width()-self.cwbias
        self.ch = self.height()-self.chbias

        #load settings
        self.LastAttributeSelected = None
        self.loadSettings()

        self.dataset = None
        self.canvas = None
        self.HighlightedAttribute = None
        #list inputs and outputs
        self.inputs = [("Examples", ExampleTable, self.data, 1)]

        #GUI

        AttsBox = QVGroupBox('Attributes',self.controlArea)
        self.attributes = QListBox(AttsBox)
        self.attributes.setSelectionMode(QListBox.Single)
        self.attributes.setMinimumSize(150, 200)
        #connect controls to appropriate functions
        self.connect(self.attributes, SIGNAL("highlighted(int)"), self.attributeHighlighted)

        QWidget(self.controlArea).setFixedSize(0, 16)

        #give mainArea a layout
        self.layout=QVBoxLayout (self.mainArea)
        self.canvas = DisplayStatistics (self.cw, self.ch)
        self.canvasview = QCanvasView (self.canvas, self.mainArea)
        self.layout.addWidget ( self.canvasview )
        self.canvasview.show()

    def resizeEvent(self, event):
        if self.canvas and self.HighlightedAttribute>=0:
            # canvas height should be a bit less than the height of the widget frame
            self.ch = self.height()-20
            self.canvas = DisplayStatistics (self.cw, self.ch)
            # the height of the bar should be 150 pixels smaller than the height of the canvas
            self.canvas.bar_height_pixels = self.height()-150
            #self.canvas.bar_height_pixels = 50
            self.canvas.displayStat(self.dataset, self.HighlightedAttribute, self.dist)
            self.canvasview.setCanvas(self.canvas)
            self.canvas.update()

    def attributeHighlighted(self, ind):
        self.HighlightedAttribute = ind
        self.ch = self.height()-self.chbias
        self.canvas = DisplayStatistics (self.cw, self.ch)
        self.canvas.bar_height_pixels = self.height()-160
        self.canvas.displayStat(self.dataset, ind, self.dist)
        self.canvasview.setCanvas(self.canvas)
        self.canvas.update()
        self.LastAttributeSelected = self.dataset.domain.attributes[ind].name

    def data(self,data):
        if data==None:
            self.dataset = None
            self.canvasview.hide()
        else:
            self.attributes.clear()
            self.canvasview.show()
            # we do a trick here, and make a new domain that includes the class var, if one exists
            # this for a reason to change any of the code, plus to be able to use such functions
            # as DomainDistributions
            newdomain = orange.Domain(data.domain.attributes+[data.domain.classVar],0)
            self.dataset = orange.ExampleTable(newdomain, data)
            self.dist = orange.DomainDistributions(self.dataset)
            for a in self.dataset.domain.attributes:
                self.attributes.insertItem(a.name)
            atts = [x.name for x in self.dataset.domain.attributes]
            if self.LastAttributeSelected in atts:
                ind = atts.index(self.LastAttributeSelected)
            else:
                ind = 0
            self.attributes.setCurrentItem(ind)
            self.attributeHighlighted(ind)

"""
class DisplayStatistics
constructs a canvas to display some statistics
"""
class DisplayStatistics (QCanvas):
	def __init__(self,*args):
		apply(QCanvas.__init__, (self,)+args)
		self.bar_height_pixels=None
		self.bar_width_pixels=None
		self.canvasW, self.canvasH = args[0], args[1]
		self.vbias, self.hbias = 60, 200

	def displayStat(self, data, ind, dist):
		attr = data.domain.attributes[ind]
		attr_name = QCanvasText (attr.name, self)
		attr_name.move(10, 10)
		attr_name.show()
		title_str = "Category"
		if attr.varType == orange.VarTypes.Continuous:
			title_str = "Values"
		category = QCanvasText (title_str, self)
		category.move(self.hbias-20, 30)
		category.setTextFlags(Qt.AlignRight)
		category.show()
		if attr.varType == orange.VarTypes.Discrete:
			totalvalues = QCanvasText ("Total Values", self)
			totalvalues.move(self.hbias+30, 30)
			totalvalues.show()
			rect_len = 100
			rect_width = 20
			f = rect_len/max(dist[ind])
			for v in range(len(attr.values)):
				t = QCanvasText (attr.values[v],self)
				t.move(self.hbias-10,self.vbias)
				t.setTextFlags(Qt.AlignRight)
				t.show()
				bar_len = dist[ind][v]*f
				if int(bar_len)==0 and bar_len!=0:
					bar_len=1
				r = QCanvasRectangle(self.hbias, self.vbias, bar_len, rect_width-2, self)
				r.setPen (QPen(Qt.NoPen))
				r.setBrush (QBrush(QColor(0,0,254)))
				r.show()
				t1 = QCanvasText (str(dist[ind][v]), self)
				t1.move(self.hbias+dist[ind][v]*rect_len/max(dist[ind])+10, self.vbias)
				t1.show()
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
			if self.bar_width_pixels==None: self.bar_width_pixels = 20
			self.mini = quartiles_list[0]
			self.maxi = quartiles_list[-1]
			self.total_values = len(quartiles_list)
			self.distinct_values = len(dist[ind])
			self.mean = dist[ind].average()
			self.stddev = dist[ind].dev()
			self.drawCStat()
		self.resize(self.canvasW+10, self.canvasH)

	def drawCStat(self):
		# draw the main rectangle 
		bar_height = self.maxi-self.mini
		#all = QCanvasRectangle (self.hbias, self.vbias, self.bar_width_pixels, self.bar_height_pixels, self)
		#all.show()
		textoffset = 15
		# draw a max line and text
		maxi_txt = QCanvasText ("max", self)
		# assume equal fonts for all the text
		self.textHeight = maxi_txt.boundingRect().height()
		maxvTextPos = self.vbias - self.textHeight*0.5
		maxi_txt.move (self.hbias+self.bar_width_pixels+15, maxvTextPos)
		maxi_txt.show()
		maxi_txtL = QCanvasText ("%5.2f" % self.maxi, self)
		maxi_txtL.move (self.hbias-textoffset, maxvTextPos)
		maxi_txtL.setTextFlags(Qt.AlignRight)
		maxi_txtL.show()
		max_line = QCanvasLine(self)
		max_line.setPoints (self.hbias-5, self.vbias, self.hbias+self.bar_width_pixels+5, self.vbias)
		max_line.show()
		max_line.setZ(1.0)

		# draw a min line and text
		mini_txt = QCanvasText ("min", self)
		minvTextPos = self.bar_height_pixels+self.vbias - self.textHeight*0.5
		mini_txt.move (self.hbias+self.bar_width_pixels+textoffset, minvTextPos)
		mini_txt.show()
		mini_txtL = QCanvasText ("%5.2f" % self.mini, self)
		mini_txtL.move (self.hbias-textoffset, minvTextPos)
		mini_txtL.setTextFlags(Qt.AlignRight)
		mini_txtL.show()
		min_line = QCanvasLine(self)
		min_line.setPoints (self.hbias-5, self.vbias+self.bar_height_pixels, self.hbias+self.bar_width_pixels+5, self.vbias+self.bar_height_pixels)
		min_line.show()
		min_line.setZ(1.0)

		# draw a rectangle from the 3rd quartile to max; add line and text
		quartile3 =  int(self.bar_height_pixels*(self.maxi-self.q3)/bar_height)
		crq3 = QCanvasRectangle (self.hbias, self.vbias, self.bar_width_pixels, quartile3, self)
		crq3.setPen (QPen(Qt.NoPen))
		crq3.setBrush (QBrush(QColor(0,175,0)))
		crq3.show()
		q3line = self.vbias + quartile3
		line2 = QCanvasLine(self)
		line2.setPoints (self.hbias-5, q3line, self.hbias+self.bar_width_pixels+5, q3line)
		line2.show()
		line2.setZ(1.0)
		crq3tR = QCanvasText ("75%", self)
		q3vTextPos = q3line - self.textHeight*0.5
		crq3tR.move(self.hbias+self.bar_width_pixels+textoffset, q3vTextPos)
		crq3tR.show()
		crq3tL = QCanvasText ("%5.2f" % self.q3, self)
		crq3tL.move(self.hbias-textoffset, q3vTextPos)
		crq3tL.setTextFlags(Qt.AlignRight)
		crq3tL.show()

		# draw a rectangle from the median to the 3rd quartile; add line and text
		med = int(self.bar_height_pixels*(self.maxi-self.median)/bar_height)
		crm = QCanvasRectangle (self.hbias, self.vbias+quartile3, self.bar_width_pixels, med-quartile3, self)
		crm.setPen (QPen(Qt.NoPen))
		crm.setBrush (QBrush(QColor(0,134,0)))
		crm.show()
		mline = self.vbias + med
		line3 = QCanvasLine(self)
		line3.setPoints (self.hbias-5, mline, self.hbias+self.bar_width_pixels+5, mline)
		line3.show()
		line3.setZ(1.0)
		crmtR = QCanvasText ("median", self)
		medvTextPos = mline - self.textHeight*0.5
		crmtR.move(self.hbias+self.bar_width_pixels+textoffset, medvTextPos)
		crmtR.show()
		crmtL = QCanvasText ("%5.2f" % self.median, self)
		crmtL.move(self.hbias-textoffset, medvTextPos)
		crmtL.setTextFlags(Qt.AlignRight)
		crmtL.show()

		# draw a rectangle from the 1st quartile to the median; add line and text
		quartile1 = int(self.bar_height_pixels*(self.maxi-self.q1)/bar_height)
		crq1 = QCanvasRectangle (self.hbias, self.vbias+med, self.bar_width_pixels, quartile1-med, self)
		crq1.setPen (QPen(Qt.NoPen))
		crq1.setBrush (QBrush(QColor(0,92,0)))
		crq1.show()
		q1line = self.vbias + quartile1
		line4 = QCanvasLine(self)
		line4.setPoints (self.hbias-5, q1line, self.hbias+self.bar_width_pixels+5, q1line)
		line4.show()
		line4.setZ(1.0)
		crq1tR = QCanvasText ("25%", self)
		q1vTextPos = q1line - self.textHeight*0.5
		crq1tR.move(self.hbias+self.bar_width_pixels+textoffset, q1vTextPos)
		crq1tR.show()
		crq1tL = QCanvasText ("%5.2f" % self.q1, self)
		crq1tL.move(self.hbias-textoffset, q1vTextPos)
		crq1tL.setTextFlags(Qt.AlignRight)
		crq1tL.show()

		# draw a rectangle from min to the 1st quartile
		cr = QCanvasRectangle (self.hbias, self.vbias+quartile1, self.bar_width_pixels, self.bar_height_pixels-quartile1, self)
		cr.setPen (QPen(Qt.NoPen))
		cr.setBrush (QBrush(QColor(0,51,0)))
		cr.show()

		# draw a horizontal mean line; add text
		self.meanpos = int(self.bar_height_pixels*(self.maxi-self.mean)/bar_height)
		self.stddev1 = int(self.bar_height_pixels*self.stddev/bar_height)
		#print "stddev ",self.stddev1, self.bar_height_pixels, bar_height
		mvbias = self.meanpos+self.vbias
		line = QCanvasLine(self)
		line.setPoints (self.hbias+self.bar_width_pixels, mvbias, self.hbias+self.bar_width_pixels +70, mvbias)
		line.setPen (QPen(QColor(255, 0, 0), 1, Qt.SolidLine))
		line.show()
		line.setZ(1.0)
		t = QCanvasText ("mean", self)
		meanvTextPos = mvbias - self.textHeight*0.5
		t.setColor (QColor(255, 0, 0))
		t.move(self.hbias+self.bar_width_pixels+110, meanvTextPos)
		t.setTextFlags(Qt.AlignRight)
		t.show()
		t3 = QCanvasText ("%5.2f +- %5.2f" % (self.mean, self.stddev), self)
		t3.setColor (QColor(255, 0, 0))
		t3.move(self.hbias-textoffset, meanvTextPos)
		t3.setTextFlags(Qt.AlignRight)
		t3.show()

		# draw the short bold mean line in the bar
		bline = QCanvasLine(self)
		bline.setPoints (self.hbias+self.bar_width_pixels*0.25, mvbias, self.hbias+self.bar_width_pixels*0.75, mvbias)
		bline.setPen (QPen(QColor(255, 0, 0), 3, Qt.SolidLine))
		bline.show()
		bline.setZ(1.0)

		# draw the std dev. line
		vert = QCanvasLine(self)
		vert.setPoints (self.hbias+self.bar_width_pixels*0.5, mvbias-self.stddev1, self.hbias+self.bar_width_pixels*0.5, mvbias+self.stddev1)
		vert.setPen (QPen(QColor(255, 0, 0), 1, Qt.SolidLine))
		vert.show()
		vert.setZ(1.0)

		# display the numbers of total and distinct values
		t1 = QCanvasText ("%d total values" % self.total_values, self)
		t1.move(10,self.vbias+self.bar_height_pixels+20)
		t1.show()
		t2 = QCanvasText ("%d distinct values" % self.distinct_values, self)
		t2.move(10,self.vbias+self.bar_height_pixels+40)
		t2.show()

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
		#print above
		#print below
		#print positions
		above_space = above[0][0] - above[-1][0] - (len(above)-2)*vspace
		below_space = below[-1][0] - below[0][0] - (len(below)-2)*vspace
		#print above_space, below_space
		for i in range(1,len(above)):
			dif = above[i-1][0] - above[i][0]
			if dif < vspace:
				#if i==len(above)-1:
				#	above[i-1] = (above[i-1][0] + vspace - dif, above[i-1][1])
				#	print "ABOVE 1", i
				#	print
				#else:
				above[i] = (above[i][0] - vspace + dif, above[i][1])
		#print above
		for i in range(1,len(below)):
			dif = below[i][0] - below[i-1][0]
			if dif < vspace:
				#if i==len(below)-1:
				#	below[i-1] = (below[i-1][0] - vspace +dif, below[i-1][1])
				#	print "BELOW 1", i
				#	print "dif ", dif
				#else:
				below[i] = (below[i][0] + vspace - dif, below[i][1])
		#print below
		# move the text to the new coordinates
		for i in range(1,len(above)):
			val, lab = above[i][0], above[i][1]
			if lab == 'max':
				if val != maxvTextPos:
					maxi_txt.move (self.hbias+self.bar_width_pixels+textoffset, val)
					maxi_txtL.move (self.hbias-textoffset, val)
					l = QCanvasLine(self)
					print max_line
					l.setPoints (self.hbias+self.bar_width_pixels+5, self.vbias, self.hbias+self.bar_width_pixels+10, val+self.textHeight*0.5)
					l.show()
					l = QCanvasLine(self)
					l.setPoints (self.hbias+self.bar_width_pixels+10, val+self.textHeight*0.5, self.hbias+self.bar_width_pixels+12, val+self.textHeight*0.5)
					l.show()
					l = QCanvasLine(self)
					l.setPoints (self.hbias-5, self.vbias, self.hbias-10, val+self.textHeight*0.5)
					l.show()
					l = QCanvasLine(self)
					l.setPoints (self.hbias-10, val+self.textHeight*0.5, self.hbias-12, val+self.textHeight*0.5)
					l.show()
			elif lab == 'q3':
				if val != q3vTextPos:
					crq3tR.move (self.hbias+self.bar_width_pixels+textoffset, val)
					crq3tL.move (self.hbias-textoffset, val)
					l = QCanvasLine(self)
					l.setPoints (self.hbias+self.bar_width_pixels+5, q3line, self.hbias+self.bar_width_pixels+10, val+self.textHeight*0.5)
					l.show()
					l = QCanvasLine(self)
					l.setPoints (self.hbias+self.bar_width_pixels+10, val+self.textHeight*0.5, self.hbias+self.bar_width_pixels+12, val+self.textHeight*0.5)
					l.show()
					l = QCanvasLine(self)
					l.setPoints (self.hbias-5, q3line, self.hbias-10, val+self.textHeight*0.5)
					l.show()
					l = QCanvasLine(self)
					l.setPoints (self.hbias-10, val+self.textHeight*0.5, self.hbias-12, val+self.textHeight*0.5)
					l.show()
			elif lab == 'med':
				if val != medvTextPos:
					crmtR.move (self.hbias+self.bar_width_pixels+15, val)
					crmtL.move (self.hbias-15, val)
					l = QCanvasLine(self)
					l.setPoints (self.hbias+self.bar_width_pixels+5, mline, self.hbias+self.bar_width_pixels+10, val+self.textHeight*0.5)
					l.show()
					l = QCanvasLine(self)
					l.setPoints (self.hbias+self.bar_width_pixels+10, val+self.textHeight*0.5, self.hbias+self.bar_width_pixels+12, val+self.textHeight*0.5)
					l.show()
					l = QCanvasLine(self)
					l.setPoints (self.hbias-5, mline, self.hbias-10, val+self.textHeight*0.5)
					l.show()
					l = QCanvasLine(self)
					l.setPoints (self.hbias-10, val+self.textHeight*0.5, self.hbias-12, val+self.textHeight*0.5)
					l.show()
			elif lab == 'q1':
				if val != q1vTextPos:
					crq1tR.move (self.hbias+self.bar_width_pixels+textoffset, val)
					crq1tL.move (self.hbias-textoffset, val)
					l = QCanvasLine(self)
					l.setPoints (self.hbias+self.bar_width_pixels+5, q1line, self.hbias+self.bar_width_pixels+10, val+self.textHeight*0.5)
					l.show()
					l = QCanvasLine(self)
					l.setPoints (self.hbias+self.bar_width_pixels+10, val+self.textHeight*0.5, self.hbias+self.bar_width_pixels+12, val+self.textHeight*0.5)
					l.show()
					l = QCanvasLine(self)
					l.setPoints (self.hbias-5, q1line, self.hbias-10, val+self.textHeight*0.5)
					l.show()
					l = QCanvasLine(self)
					l.setPoints (self.hbias-10, val+self.textHeight*0.5, self.hbias-12, val+self.textHeight*0.5)
					l.show()
		for i in range(1,len(below)):
			val, lab = below[i][0], below[i][1]
			if lab == 'min':
				if val != minvTextPos:
					mini_txt.move (self.hbias+self.bar_width_pixels+textoffset, val)
					mini_txtL.move (self.hbias-textoffset, val)
					l = QCanvasLine(self)
					l.setPoints (self.hbias+self.bar_width_pixels+5, self.bar_height_pixels+self.vbias, self.hbias+self.bar_width_pixels+10, val+self.textHeight*0.5)
					l.show()
					l = QCanvasLine(self)
					l.setPoints (self.hbias+self.bar_width_pixels+10, val+self.textHeight*0.5, self.hbias+self.bar_width_pixels+12, val+self.textHeight*0.5)
					l.show()
					l = QCanvasLine(self)
					l.setPoints (self.hbias-5, self.bar_height_pixels+self.vbias, self.hbias-10, val+self.textHeight*0.5)
					l.show()
					l = QCanvasLine(self)
					l.setPoints (self.hbias-10, val+self.textHeight*0.5, self.hbias-12, val+self.textHeight*0.5)
					l.show()
			elif lab == 'q1':
				if val != q1vTextPos:
					crq1tR.move (self.hbias+self.bar_width_pixels+textoffset, val)
					crq1tL.move (self.hbias-textoffset, val)
					l = QCanvasLine(self)
					l.setPoints (self.hbias+self.bar_width_pixels+5, q1line, self.hbias+self.bar_width_pixels+10, val+self.textHeight*0.5)
					l.show()
					l = QCanvasLine(self)
					l.setPoints (self.hbias+self.bar_width_pixels+10, val+self.textHeight*0.5, self.hbias+self.bar_width_pixels+12, val+self.textHeight*0.5)
					l.show()
					l = QCanvasLine(self)
					l.setPoints (self.hbias-5, q1line, self.hbias-10, val+self.textHeight*0.5)
					l.show()
					l = QCanvasLine(self)
					l.setPoints (self.hbias-10, val+self.textHeight*0.5, self.hbias-12, val+self.textHeight*0.5)
					l.show()
			elif lab == 'med':
				if val != medvTextPos:
					crmtR.move (self.hbias+self.bar_width_pixels+textoffset, val)
					crmtL.move (self.hbias-textoffset, val)
					l = QCanvasLine(self)
					l.setPoints (self.hbias+self.bar_width_pixels+5, mline, self.hbias+self.bar_width_pixels+10, val+self.textHeight*0.5)
					l.show()
					l = QCanvasLine(self)
					l.setPoints (self.hbias+self.bar_width_pixels+10, val+self.textHeight*0.5, self.hbias+self.bar_width_pixels+12, val+self.textHeight*0.5)
					l.show()
					l = QCanvasLine(self)
					l.setPoints (self.hbias-5, mline, self.hbias-10, val+self.textHeight*0.5)
					l.show()
					l = QCanvasLine(self)
					l.setPoints (self.hbias-10, val+self.textHeight*0.5, self.hbias-12, val+self.textHeight*0.5)
					l.show()
			elif lab == 'q3':
				if val != q3vTextPos:
					crq3tR.move (self.hbias+self.bar_width_pixels+textoffset, val)
					crq3tL.move (self.hbias-textoffset, val)
					l = QCanvasLine(self)
					l.setPoints (self.hbias+self.bar_width_pixels+5, q3line, self.hbias+self.bar_width_pixels+10, val+self.textHeight*0.5)
					l.show()
					l = QCanvasLine(self)
					l.setPoints (self.hbias+self.bar_width_pixels+10, val+self.textHeight*0.5, self.hbias+self.bar_width_pixels+12, val+self.textHeight*0.5)
					l.show()
					l = QCanvasLine(self)
					l.setPoints (self.hbias-5, q3line, self.hbias-10, val+self.textHeight*0.5)
					l.show()
					l = QCanvasLine(self)
					l.setPoints (self.hbias-10, val+self.textHeight*0.5, self.hbias-12, val+self.textHeight*0.5)
					l.show()
		#print

#test widget appearance
if __name__=="__main__":
	a=QApplication(sys.argv)
	ow=OWAttributeStatistics()
	a.setMainWidget(ow)
	data = orange.ExampleTable('adult_sample')
	ow.data(data)
	ow.show()
	a.exec_loop()
	ow.saveSettings()
