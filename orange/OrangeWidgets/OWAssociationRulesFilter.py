
import orange
import orngAssoc
from OData import *
from OWWidget import *

import sys
import string

from qt import *
from qtcanvas import *

class AssociationRulesFilterCanvas(QCanvas):
	def __init__(self, rules, numcols, numrows, support_min, support_max, confidence_min, confidence_max, cell_width, cell_height, statusBar, *args):
		apply(QCanvas.__init__, (self, ) + args)
		self.rules = rules
		self.statusBar = statusBar
		self.draw(numcols, numrows, support_min, support_max, confidence_min, confidence_max, cell_width, cell_height)
		
	def draw(self, numcols, numrows, support_min, support_max, confidence_min, confidence_max, cell_width, cell_height):
		# Najprej skrij vse celice nato jih izbriši
		for a in self.allItems():
			a.hide()
		self.cells = []		
		
		# nastavi se na ustrezno velikost
		self.resize(numcols * cell_width +1, numrows * cell_height +1)
		
		# preštej pravila po celicah, ugotovi skupno število pravil
		rules_count = 0
		cell_rule_counts = []
		for x in range(numcols):
			for y in range(numrows):
				cell_rule_counts.append(0)

		for rule in self.rules:
			# Ali je pravilo v zaželenem okvirju
			if rule.support > support_min and rule.support <= support_max and rule.confidence > confidence_min and rule.confidence <= confidence_max:
				rules_count += 1

				x = int((rule.support - support_min) * numcols / (support_max - support_min))

				# upoštevaj tudi možnost, da je rezultat na robu
				if x == (rule.support - support_min) * numcols / (support_max - support_min):
					x -= 1
				y = numrows - 1 - int((rule.confidence - confidence_min) * numrows / (confidence_max - confidence_min))

				# upoštevaj tudi možnost, da je rezultat na robu
				if y == numrows - 1 - ((rule.confidence - confidence_min) * numrows / (confidence_max - confidence_min)):
					y += 1

				cell_rule_counts[x * numrows + y] += 1
			

		# poiši maksimalno število pravil v celici							
		max_cell_rule_count = 1
		for x in cell_rule_counts:
			if x > max_cell_rule_count:
				max_cell_rule_count = x

		# èrna naj pomeni vsaj 10 pravil
		if max_cell_rule_count < 10:
			max_cell_rule_count = 10
			
		# nariši mrežo
		for x in range(numcols):
			for y in range(numrows):
				# število pravil v trenutni celici
				cell_rule_count = cell_rule_counts[x*numrows + y]
				# nastavi velikost in položaj posamezne celice
				cell = QCanvasRectangle(x * cell_width, y * cell_height, cell_width+1, cell_height+1, self)
				# nastavi barvo celice
				if cell_rule_count == 0:
					cell.setBrush(QBrush(QColor(255, 255, 255)))
				else:
					color = 255 - (cell_rule_count * 235 / max_cell_rule_count)
					cell.setBrush(QBrush(QColor(color-20, color-20, color)))
				# nastavi barvo in tip roba celice
				cell.setPen(QPen(QColor(200,200,200), 1));
				# pokaži celico
				cell.show()
				# shrani celico, da je garbage collector ne izbriše
				self.cells.append(cell)
		
		self.update()
		self.statusBar.message('Support('+ str(support_min) +':'+ str(support_max) +')     Confidence('+ str(confidence_min) +':'+ str(confidence_max) +')     Rules('+ str(rules_count) + ')')

				
class AssociationRulesFilterBrowser(QCanvasView):
	def __init__(self, *args):
		apply(QCanvasView.__init__,(self, ) + args)
		self.rectangleDrawn = False
		# nastavljen na fiksno velikost
		self.setFixedSize(365, 365)
		self.update()

	# miškin klik, zapomni si koordinato	
	def contentsMousePressEvent(self, ev):
		self.startX = ev.pos().x()
		self.startY = ev.pos().y()
		self.endX = self.startX+1
		self.endY = self.startY+1
		if (self.rectangleDrawn):
			self.rectangle.hide()
		else:
			self.rectangleDrawn = True
		self.rectangle = QCanvasRectangle(self.startX, self.startY, 1, 1, self.canvas())
		# risi cez podlago
		self.rectangle.setZ(1)
		self.rectangle.setPen(QPen(QColor(Qt.gray), 2, Qt.DashLine))
		self.rectangle.show()
		self.canvas().update()

	# miška premaknjena, tipka še ne spušèena (javljaj dogodek za izris legende)										  		
	def contentsMouseMoveEvent(self, ev):
		self.endX = ev.pos().x()
		self.endY = ev.pos().y()
		self.emit(PYSIGNAL("sigNewAreaSelecting"), (self.startX, self.startY, self.endX, self.endY, ))
		self.rectangle.setSize(self.endX - self.startX, self.endY - self.startY)
		self.canvas().update()

	# tipka na miški spušèena, poglej za katero tipko je šlo in sproži ustrezni dogodek			
	def contentsMouseReleaseEvent(self, ev):
		if (self.startX > self.endX):
			t = self.startX
			self.startX = self.endX
			self.endX = t
		if (self.startY > self.endY):
			t = self.startY
			self.startY = self.endY
			self.endY = t
			
		# popravi selekcijo na mrežo
		#if (ev.button() & QMouseEvent.LeftButton):
		if True:
			if (self.startX < 0):
				self.startX = 0
			if (self.startY < 0):
				self.startY = 0
			#TODO: ne pride do collision
			if (self.endX > self.width()):
				self.endX = self.width()-10
			if (self.endY > self.height()):
				self.endY = self.height()-10
							
			try:
				items = self.canvas().collisions(QPoint(self.startX, self.startY))
				self.startX = int(items[len(items)-1].x())
				self.startY = int(items[len(items)-1].y())
			except:
				pass
			
			try:	
				items = self.canvas().collisions(QPoint(self.endX, self.endY))
				self.endX = int(items[len(items)-1].x() + items[len(items)-1].width())
				self.endY = int(items[len(items)-1].y() + items[len(items)-1].height()) 
			except:
				pass
				
			self.rectangle.hide()
			self.rectangle = QCanvasRectangle(self.startX +1,
											  self.startY +1,
											  self.endX - self.startX -1,
											  self.endY - self.startY -1,
											  self.canvas())
			self.rectangle.setZ(1)
			self.rectangle.setPen(QPen(QColor(Qt.gray), 2, Qt.DashLine))
			self.rectangle.show()

		# nariši spremembe
		self.canvas().update()
		
		# ali je bil pritisnjen levi ali desni miškin gumb		
		if (ev.button() & QMouseEvent.LeftButton):	
			self.emit(PYSIGNAL("sigNewRulesAreaSelected"),(self.startX, self.startY, self.endX, self.endY, ))
		else:
			self.emit(PYSIGNAL("sigNewGridAreaSelected"),(self.startX, self.startY, self.endX, self.endY, ))					 								 																						
		
class OWAssociationRulesFilter(OWWidget):
	def __init__(self, parent=None):
		OWWidget.__init__(self,
            parent,
            "AssociationRulesFilter",
            """OWAssociationRulesFilter is orange widget for\nadvanced selection of Association rules.\n\n        Authors: Jure Germovsek, Petra Kralj, Matjaz Jursic        \nMay 25, 2003
            """,
            FALSE,
            FALSE,
            "OrangeWidgetsIcon.png",
            "OrangeWidgetsLogo.png")

		self.addInput("arules")

		# zapomni si glavne karakteristike
		self.support_max = 1.0
		self.support_min = 0.0
		self.confidence_max = 1.0
		self.confidence_min = 0.0
		self.numcols = 18
		self.numrows = 18
		self.cellwidth = 20
		self.cellheight = 20

		# ne rabi settings list ker nastavitve prilagaja podatkom.
		#self.settingsList = []   
        #self.loadSettings()                            

		# attributi za gradnjo asoc. pravil
		self.rules = []
		self.allrules = []
		self.dataset = []
		
		# poskrbi za avtomatski layout
		self.hbox = QHBoxLayout(self.mainArea)
		self.vbox = QVBoxLayout(self.hbox)
		self.hbox1 = QHBoxLayout(self.vbox)
		self.vbox1 = QVBoxLayout(self.hbox1)
		self.vbox2 = QVBoxLayout(self.hbox1)
		self.hbox3 = QHBoxLayout(self.vbox2)
		
		# ustvari elemente
		self.statusBar = QStatusBar(self.mainArea)
		self.gridGB = QGroupBox(1, QGroupBox.Vertical , 'Support Horizontal('+ str(self.support_min) +':'+ str(self.support_max) +')     Confidence Vertical('+ str(self.confidence_min) +':'+ str(self.confidence_max) +')', self.mainArea)
		self.AssociationRulesFilterCanvas = AssociationRulesFilterCanvas(self.allrules, self.numrows, self.numcols, self.support_min, self.support_max, self.confidence_min, self.confidence_max, self.cellwidth, self.cellheight, self.statusBar, self.mainArea)
		self.AssociationRulesFilterBrowser = AssociationRulesFilterBrowser(self.AssociationRulesFilterCanvas, self.gridGB)
		self.edtRulesGB = QGroupBox(2, QGroupBox.Vertical , 'Rules View', self.mainArea)
		self.edtRules = QMultiLineEdit(self.edtRulesGB)
		self.edtRules.setReadOnly(True)
				
		self.support = QGroupBox(2, QGroupBox.Vertical , 'Support View', self.controlArea)
		self.supportMinValueLabel = QLabel('MinValue:', self.support)
		self.supportMaxValueLabel = QLabel('MaxValue:', self.support)
		self.supportMinValueSpinBox = QSpinBox(self.support)
		self.supportMinValueSpinBox.setRange(0, 99)
		self.supportMinValueSpinBox.setSuffix(' %')
		self.supportMaxValueSpinBox = QSpinBox(self.support)
		self.supportMaxValueSpinBox.setRange(1,100)
		self.supportMaxValueSpinBox.setSuffix(' %')
		
		self.confidence = QGroupBox(2, QGroupBox.Vertical , 'Confidence View', self.controlArea)
		self.confidenceMinValueLabel = QLabel('MinValue:', self.confidence)
		self.confidenceMaxValueLabel = QLabel('MaxValue:', self.confidence)
		self.confidenceMinValueSpinBox = QSpinBox(self.confidence)
		self.confidenceMinValueSpinBox.setRange(0, 99)
		self.confidenceMinValueSpinBox.setSuffix(' %')
		self.confidenceMaxValueSpinBox = QSpinBox(self.confidence)
		self.confidenceMaxValueSpinBox.setRange(1,100)
		self.confidenceMaxValueSpinBox.setSuffix(' %')
		
		#self.gridsettings = QGroupBox(5, QGroupBox.Vertical , 'Grid View Settings', self)
		#self.gridsettingsCellWidthLabel = QLabel('Cell Width:', self.gridsettings)
		#self.gridsettingsCellHeightLabel = QLabel('Cell Height:', self.gridsettings)
		#self.gridsettingsCellNumLabel = QLabel('Num of cells in', self.gridsettings)
		#self.gridsettingsCellRowNumLabel = QLabel('Row:', self.gridsettings)
		#self.gridsettingsCellColNumLabel = QLabel('Column:', self.gridsettings)
		#self.gridsettingsCellWidthSpinBox = QSpinBox(self.gridsettings)
		#self.gridsettingsCellWidthSpinBox.setSuffix(' px')
		#self.gridsettingsCellHeightSpinBox = QSpinBox(self.gridsettings)
		#self.gridsettingsCellHeightSpinBox.setSuffix(' px')
		#self.gridsettingsCellNumLabel2 = QLabel(self.gridsettings)
		#self.gridsettingsCellRowNumSpinBox = QSpinBox(self.gridsettings)
		#self.gridsettingsCellColNumSpinBox = QSpinBox(self.gridsettings)
		
		self.buttonApply = QPushButton('Zoom In', self.controlArea)
		self.buttonReset = QPushButton('Default Zoom', self.controlArea)
		self.buttonShowEntire = QPushButton('No Zoom', self.controlArea)
		
		self.writeValuesToUser(self.support_min, self.support_max, self.confidence_min, self.confidence_max)
		
		# dodaj elemente na layout
		self.vbox1.addWidget(self.gridGB)
		self.vbox.addWidget(self.statusBar)
		self.hbox3.addWidget(self.edtRulesGB)		

		# poberi odveèen prostor pod mrežo
		self.vbox1spacer = QSpacerItem(0, 0, QSizePolicy.Fixed , QSizePolicy.Expanding)
		self.vbox1.addItem(self.vbox1spacer)

		# naj se razteguje prostor za pravila
		self.vboxspacer = QSpacerItem(0, 0, QSizePolicy.Fixed , QSizePolicy.Fixed)
		self.vbox.addItem(self.vboxspacer)		
		
		self.connect(self.buttonApply,SIGNAL("clicked()"),self.applyButtonClicked)
		self.connect(self.buttonReset,SIGNAL("clicked()"),self.resetButtonClicked)
		self.connect(self.buttonShowEntire,SIGNAL("clicked()"),self.showEntireButtonClicked)
		self.connect(self.AssociationRulesFilterBrowser, PYSIGNAL("sigNewAreaSelecting"), self.slotNewAreaSelecting)
		self.connect(self.AssociationRulesFilterBrowser, PYSIGNAL("sigNewGridAreaSelected"), self.slotNewGridAreaSelected)
		self.connect(self.AssociationRulesFilterBrowser, PYSIGNAL("sigNewRulesAreaSelected"), self.slotNewGridAreaSelected)
		self.connect(self,PYSIGNAL("rulesChanged"),self.displayRules)
	
	# ponovno nariši grid po podatkih, ki jih hraniš
	def redrawGrid(self):
		if self.support_min == self.support_max:
			self.support_max = self.support_max + 0.01
		if self.confidence_max == self.confidence_min:
			self.confidence_max = self.confidence_max + 0.01

		self.gridGB.setTitle('Support Horizontal('+ str(self.support_min) +':'+ str(self.support_max) +')     Confidence Vertical('+ str(self.confidence_min) +':'+ str(self.confidence_max) +')')		
		self.AssociationRulesFilterCanvas.draw(self.numcols, self.numrows, self.support_min, self.support_max, self.confidence_min, self.confidence_max, self.cellwidth, self.cellheight)
		
	# podatke, ki jih hraniš, izpiši v okenca	
	def writeValuesToUser(self, support_min, support_max, confidence_min, confidence_max):
		self.supportMinValueSpinBox.setValue(support_min * 100)

		# upoštevaj tudi tisoèice pri support in confidence max
		if int(support_max * 100) < 100 * support_max:
			self.supportMaxValueSpinBox.setValue(support_max * 100 + 1)
		else:
			self.supportMaxValueSpinBox.setValue(support_max * 100)
		
		self.confidenceMinValueSpinBox.setValue(confidence_min * 100)
		if int(confidence_max * 100) < 100 * confidence_max:
			self.confidenceMaxValueSpinBox.setValue(confidence_max * 100 + 1)
		else:
			self.confidenceMaxValueSpinBox.setValue(confidence_max * 100)
		
		#self.gridsettingsCellWidthSpinBox.setValue(self.cellwidth)
		#self.gridsettingsCellHeightSpinBox.setValue(self.cellheight)
		#self.gridsettingsCellColNumSpinBox.setValue(self.numcols)
		#self.gridsettingsCellRowNumSpinBox.setValue(self.numrows)
	
	# podatke iz okenc preberi v podatke, ki jih hraniš
	def readValuesFromUser(self):
		#self.numcols = self.gridsettingsCellColNumSpinBox.value()
		#self.numrows = self.gridsettingsCellRowNumSpinBox.value()
		#self.cellwidth = self.gridsettingsCellWidthSpinBox.value()
		#self.cellheight = self.gridsettingsCellHeightSpinBox.value()
		
		self.confidence_min = float(self.confidenceMinValueSpinBox.value()) / 100
		self.confidence_max = float(self.confidenceMaxValueSpinBox.value()) / 100
		
		self.support_min = float(self.supportMinValueSpinBox.value()) / 100
		self.support_max = float(self.supportMaxValueSpinBox.value()) / 100
		
	def applyButtonClicked(self):
		self.statusBar.clear()
		self.readValuesFromUser()
		self.redrawGrid()

	def resetButtonClicked(self):
		self.statusBar.clear()
		del(self.rules[:])
		self.rules.extend(self.allrules)

		# nastavi priporoèeno obmoèje
		if len(self.allrules) > 0:
			rule = self.allrules[0]
			self.support_max = rule.support
			self.support_min = rule.support
			self.confidence_max = rule.confidence
			self.confidence_min = rule.confidence
			for rule in self.allrules:
				if rule.confidence > self.confidence_max:
					self.confidence_max = rule.confidence
				if rule.confidence < self.confidence_min:
					self.confidence_min = rule.confidence
				if rule.support > self.support_max:
					self.support_max = rule.support
				if rule.support < self.support_min:
					self.support_min = rule.support

			# zaradi kasnejsih preverjanj pogojev rule.support < support_min in rule.confidence < confidence_min
			if self.support_min == round(self.support_min, 2):
				self.support_min -= 0.01
			if self.confidence_min == round(self.confidence_min, 2):
				self.confidence_min -= 0.01
		else:
			self.support_max = 1.0
			self.support_min = 0.0
			self.confidence_max = 1.0
			self.confidence_min = 0.0
			
		self.writeValuesToUser(self.support_min, self.support_max, self.confidence_min, self.confidence_max)
		self.readValuesFromUser()
		self.redrawGrid()
		self.emit(PYSIGNAL("rulesChanged"), (self.rules,))  #send a signal that new rules are to be vizualized									
		

	def showEntireButtonClicked(self):
		self.statusBar.clear()
		self.support_max = 1.0
		self.support_min = 0.0
		self.confidence_max = 1.0
		self.confidence_min = 0.0
		del(self.rules[:])
		self.rules.extend(self.allrules)
		self.writeValuesToUser(self.support_min, self.support_max, self.confidence_min, self.confidence_max)
		self.redrawGrid()
		self.emit(PYSIGNAL("rulesChanged"), (self.rules,))  #send a signal that new rules are to be vizualized									
		
	
	def slotNewAreaSelecting(self, x1, y1, x2, y2):
		if x1 > x2:
			t = x2
			x2 = x1
			x1 = t
		if y1 > y2:
			t = y2
			y2 = y1
			y1 = t
		
		if x2 - x1 > 10:
			if y2 - y1 > 10:
				support_min = self.support_min
				support_max = self.support_max
				confidence_min = self.confidence_min
				confidence_max = self.confidence_max 

				support_min += x1 * (self.support_max - self.support_min) / self.AssociationRulesFilterCanvas.width()
				support_max -= (self.AssociationRulesFilterCanvas.width() - x2) * (self.support_max - self.support_min) / self.AssociationRulesFilterCanvas.width()
				confidence_min += (self.AssociationRulesFilterCanvas.height() - y2) * (self.confidence_max - self.confidence_min) / self.AssociationRulesFilterCanvas.height()
				confidence_max -= y1 * (self.confidence_max - self.confidence_min) / self.AssociationRulesFilterCanvas.height()
												
				if not support_min > 0:
					support_min = 0
				
				if not support_max < 1:
					support_max = 1
				
				if not confidence_min > 0:
					confidence_min = 0
				
				if not confidence_max < 1:
					confidence_max = 1

				# normaliziraj na normalno število digitalk
				confidence_max = round(confidence_max, 3)
				confidence_min = round(confidence_min, 3)
				support_min = round(support_min, 3)
				support_max = round(support_max, 3)
				
				self.statusBar.message('Support('+ str(support_min) +':'+ str(support_max) +')     Confidence('+ str(confidence_min) +':'+ str(confidence_max) +')', 30000)
					 
	def slotNewGridAreaSelected(self, x1, y1, x2, y2):
		if x1 > x2:
			t = x2
			x2 = x1
			x1 = t
		if y1 > y2:
			t = y2
			y2 = y1
			y1 = t
			
		if x2 - x1 > 10:
			if y2 - y1 > 10:
				support_min = self.support_min
				support_max = self.support_max
				confidence_min = self.confidence_min
				confidence_max = self.confidence_max 

				x11 = (x1+1) * self.numcols / self.AssociationRulesFilterCanvas.width() 
				x12 = (x2) * self.numcols / self.AssociationRulesFilterCanvas.width()
				y11 = (y2) * self.numrows / self.AssociationRulesFilterCanvas.height() 
				y12 = (y1+1) * self.numrows / self.AssociationRulesFilterCanvas.height()

				support_min = (self.support_max - self.support_min) / (self.numcols) * x11 + self.support_min
				support_max = (self.support_max - self.support_min) / (self.numcols) * x12 + self.support_min
				confidence_min = ((self.numrows) - y11) * (self.confidence_max - self.confidence_min) / (self.numrows) + self.confidence_min
				confidence_max = ((self.numrows) - y12) * (self.confidence_max - self.confidence_min) / (self.numrows) + self.confidence_min
				
				if support_min < 0.0:
					support_min = 0.0
				
				if support_max > 1.0:
					support_max = 1.0
				
				if confidence_min < 0.0:
					confidence_min = 0.0
					
				if confidence_max > 1.0:
					confidence_max = 1.0

				# normaliziraj na normalno število digitalk
				confidence_max = round(confidence_max, 3)
				confidence_min = round(confidence_min, 3)
				support_min = round(support_min, 3)
				support_max = round(support_max, 3)

				rules_count = 0
				del (self.rules[:])
				for rule in self.allrules:
					if round(rule.support, 3) > support_min:
						if round(rule.support, 3) <= support_max:
							if round(rule.confidence, 3) > confidence_min:
								if round(rule.confidence, 3) <= confidence_max:
									rules_count = rules_count + 1
									self.rules.append(rule)
			
				self.writeValuesToUser( support_min, support_max, confidence_min, confidence_max )
				self.statusBar.message('Support('+ str(support_min) +':'+ str(support_max) +')     Confidence('+ str(confidence_min) +':'+ str(confidence_max) +')     Rules('+ str(rules_count) + ')')

				self.emit(PYSIGNAL("rulesChanged"), (self.rules,))  #send a signal that new rules are to be vizualized									
			
				#self.redrawGrid()
				
	
	def displayRules(self):
		self.edtRules.clear()
		for rule in self.rules:
			self.edtRules.append(orngAssoc.printRule(rule))
		self.send("arules", self.rules)
	def arules(self,rules):                # channel po katerem dobi podatke
		del(self.allrules[:])
		self.allrules.extend(rules)
		
		self.resetButtonClicked()
        
if __name__=="__main__":
    a=QApplication(sys.argv)
    ow=OWAssociationRulesFilter()
    a.setMainWidget(ow)
    ow.resize(750, 380)

    dataset = orange.ExampleTable('lenses.tab')
    od = OrangeData(dataset)

    rules = orngAssoc.build(od.table, 0.1)
    ow.arules(rules)

    ow.show()
    a.exec_loop()
    ow.saveSettings()
			