# Author: Gregor Leban (gregor.leban@fri.uni-lj.si)
# Description:
#	tab for showing widgets and widget button class
#
from qt import *
import sys
import os
import string
from xml.dom.minidom import Document, parse

TRUE  = 1
FALSE = 0 

class WidgetButton(QToolButton):
	def __init__(self, *args):
		apply(QToolButton.__init__,(self,)+ args)

	def setValue(self, name, fileName, inList, outList, icon, description, priority, canvasDlg):
		self.name = name
		self.fileName = fileName
		self.iconName = icon
		self.priority = priority
		self.description = description
	
		inList = inList.replace(",", " ")
		outList = outList.replace(",", " ")
		self.inList = inList.split()
		self.outList = outList.split()

		formatedInList = ""
		for item in self.inList:
			formatedInList = formatedInList + canvasDlg.getChannelName(item) + ", "

		if len(formatedInList) > 1 and formatedInList[-2] == ",":
			formatedInList = formatedInList[:-2]
		else:
			formatedInList = formatedInList + "None"
			
		formatedOutList = ""
		for item in self.outList:
			formatedOutList = formatedOutList + canvasDlg.getChannelName(item) + ", "

		if len(formatedOutList) > 1 and formatedOutList[-2] == ",":
			formatedOutList = formatedOutList[:-2]
		else:
			formatedOutList = formatedOutList + "None"
		
		#tooltipText = name + "\nClass name: " + fileName + "\nin: " + formatedInList + "\nout: " + formatedOutList + "\ndescription: " + description
		tooltipText = "<b>%s</b><br>Class name: %s<br>in: %s<br>out: %s<br>description: %s" % (name, fileName, formatedInList, formatedOutList, description)
		QToolTip.add( self, tooltipText)


		self.canvasDlg = canvasDlg
		
		self.setPixmap(QPixmap(self.iconName))
		self.setMaximumSize(48, 48)
		self.setMinimumSize(48, 48)

	def clicked(self):
		win = self.canvasDlg.workspace.activeWindow()
		if (win != None):
			win.addWidget(self)

class WidgetTab(QWidget):
	def __init__(self, *args):
		apply(QWidget.__init__,(self,)+ args)
		self.HItemBox = QHBoxLayout(self)
		#self.setMaximumHeight(60)
		self.widgets = []

	def addWidget(self, widget):
		self.HItemBox.addWidget(widget)
		self.widgets.append(widget)

	def finishedAdding(self):
		self.HItemBox.addStretch(10)

class WidgetTabs(QTabWidget):
	def __init__(self, *args):
		apply(QTabWidget.__init__,(self,) + args)
		self.tabs = []
		self.canvasDlg = None
		self.allWidgets = []

		tab = WidgetTab(self, "Data")
		self.tabs.append(tab)
		self.insertTab(tab, "Data")
		
		tab = WidgetTab(self, "Classification")
		self.tabs.append(tab)
		self.insertTab(tab, "Classification")
		
	def setCanvasDlg(self, canvasDlg):
		self.canvasDlg = canvasDlg

	# read the xml registry and show all installed widgets
	def readInstalledWidgets(self, registryFileName, widgetDir, picsDir, defaultPic):
		self.widgetDir = widgetDir
		self.picsDir = picsDir
		self.defaultPic = defaultPic
		doc = parse(registryFileName)
		orangeCanvas = doc.firstChild
		categories = orangeCanvas.getElementsByTagName("widget-categories")[0]
		if (categories == None):
			return

		categoryList = categories.getElementsByTagName("category")
		for category in categoryList:
			self.addWidgetCategory(category)

	# add all widgets inside the category to the tab
	def addWidgetCategory(self, category):
		tab = None
		for madeTab in self.tabs:
			if madeTab.name() == str(category.getAttribute("name")):
				tab = madeTab
		if tab == None:
			tab = WidgetTab(self, str(category.getAttribute("name")))
			#tab.setMinimumHeight(60)
			self.tabs.append(tab)
			self.insertTab(tab, str(category.getAttribute("name")))

		priorityList = []
		inListList = []
		outListList = []
		nameList = []
		iconNameList = []
		descriptionList = []
		fileNameList = []
		
		widgetList = category.getElementsByTagName("widget")
		for widget in widgetList:
			name = widget.getAttribute("name")
			fileName = widget.getAttribute("file")
			inList = widget.getAttribute("in")
			outList = widget.getAttribute("out")
			priority = int(widget.getAttribute("priority"))

			icon = widget.getAttribute("icon")
			iconName = icon
			if (icon != ""):
				if os.path.isfile(self.picsDir + icon):
					iconName = self.picsDir + icon			
				elif os.path.isfile(self.widgetDir + icon):
					iconName = self.widgetDir + icon
				else:
					iconName = self.defaultPic
			
			# it's a complicated way to get to the widget description
			descNode= widget.getElementsByTagName("description")[0]
			description = ""
			for node in widget.childNodes:
				if node.nodeType == node.TEXT_NODE:
					description = description + node.nodeValue
				else:
					for n2 in node.childNodes:
						if n2.nodeType == node.TEXT_NODE:
							description = description + n2.nodeValue

			description = string.strip(description)
			i = 0
			while i < len(priorityList) and priority > priorityList[i]:
				i = i + 1
			priorityList.insert(i, priority)
			nameList.insert(i, name)
			fileNameList.insert(i, fileName)
			inListList.insert(i, inList)
			outListList.insert(i, outList)
			iconNameList.insert(i, iconName)
			descriptionList.insert(i, description)

		exIndex = 0
		for i in range(len(priorityList)):			
			button = WidgetButton(tab)
			button.setValue(nameList[i], fileNameList[i], inListList[i], outListList[i], iconNameList[i], descriptionList[i], priorityList[i], self.canvasDlg)
			self.connect( button, SIGNAL( 'clicked()' ), button.clicked)
			if exIndex != priorityList[i] / 1000:
				frame = QFrame(tab)
				frame.setMinimumWidth(20)
				frame.setMaximumWidth(20)
				tab.addWidget(frame)
				exIndex = priorityList[i] / 1000
			tab.addWidget(button)
			self.allWidgets.append(button)

		tab.finishedAdding()