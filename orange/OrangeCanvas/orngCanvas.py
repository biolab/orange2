# Author: Gregor Leban (gregor.leban@fri.uni-lj.si)
# Description:
#	main file, that creates the MDI environment

from qt import *
import sys, os, cPickle
import orngTabs, orngDoc, orngDlgs, orngOutput, orngResources, xmlParse


TRUE  = 1
FALSE = 0

class OrangeCanvasDlg(QMainWindow):
	def __init__(self,*args):
		apply(QMainWindow.__init__,(self,) + args)
		self.ctrlPressed = 0	# we have to save keystate, so that orngView can access information about keystate
		self.debugMode = 1		# print extra output for debuging
		self.resize(900,800)
		self.setCaption("Qt Orange Canvas")
		self.windows = []	# list of id for windows in Window menu

		# if widget path not registered -> register
		self.widgetDir = os.path.realpath("../OrangeWidgets") + "/"
		if not os.path.exists(self.widgetDir): print "Error. Directory %s not found. Unable to locate widgets." % (self.widgetDir)

		self.picsDir = os.path.realpath("../OrangeWidgets/icons") + "/"
		if not os.path.exists(self.picsDir): print "Error. Directory %s not found. Unable to locate widget icons." % (self.picsDir)
		
		self.canvasDir = os.path.realpath("./") + "/"

		self.defaultPic = self.picsDir + "Unknown.png"
		self.registryFileName = self.canvasDir + "widgetregistry.xml"
		
		if sys.path.count(self.widgetDir) == 0:
			sys.path.append(self.widgetDir)

		self.workspace = WidgetWorkspace(self)
		self.workspace.setBackgroundColor(QColor(255,255,255))
		self.setCentralWidget(self.workspace)
		self.statusBar = QStatusBar(self)
		self.connect(self.workspace, SIGNAL("windowActivated(QWidget*)"), self.focusDocument)
		
		self.settings = {}
		self.loadSettings()
		self.rebuildSignals()	# coloring of signals - unused!
		self.useLargeIcons = FALSE
		self.snapToGrid = TRUE
		if self.settings.has_key("useLargeIcons"): self.useLargeIcons = self.settings["useLargeIcons"]
		if self.settings.has_key("snapToGrid"): self.snapToGrid = self.settings["snapToGrid"]
		if not self.settings.has_key("printOutputInStatusBar"): self.settings["printOutputInStatusBar"] = 0
		if not self.settings.has_key("printExceptionInStatusBar") : self.settings["printExceptionInStatusBar"] = 1
		
		self.toolbar = QToolBar(self, 'toolbar')
		self.widgetsToolBar = QToolBar( self, 'Widgets')

		self.initMenu()
		
		self.toolNew  = QToolButton(QPixmap(orngResources.file_new), "New schema" , QString.null, self.menuItemNewSchema, self.toolbar, 'new schema')
		#self.toolNew.setUsesTextLabel (TRUE)
		self.toolOpen = QToolButton(QPixmap(orngResources.file_open), "Open schema" , QString.null, self.menuItemOpen , self.toolbar, 'open schema') 
		self.toolSave = QToolButton(QPixmap(orngResources.file_save), "Save schema" ,QString.null, self.menuItemSave, self.toolbar, 'save schema')
		self.toolbar.addSeparator()
		toolPrint = QToolButton(QPixmap(orngResources.file_print), "Print" ,QString.null, self.menuItemPrinter, self.toolbar, 'print')
		self.addToolBar(self.toolbar, "Toolbar", QMainWindow.Top, TRUE)

		# read widgets	
		self.widgetsToolBar.setHorizontalStretchable(TRUE)		
		self.createWidgetsToolbar(not os.path.exists(self.registryFileName))

		# read recent files
		self.recentDocs = []
		self.readRecentFiles()
		
		# center window in the desktop
		deskH = app.desktop().height()
		deskW = app.desktop().width()
		h = deskH/2 - self.height()/2
		w = deskW/2 - self.width()/2
		if h<0 or w<0:	# if the window is too small, resize the window to desktop size
			self.resize(app.desktop().height(), app.desktop().width())
			self.move(0,0)
		else:			
			self.move(w,h)
		
		# apply output settings
		self.output = orngOutput.OutputWindow(self, self.workspace)
		self.output.show()
		self.output.catchException(self.settings["catchException"])
		self.output.catchOutput(self.settings["catchOutput"])
		self.output.setFocusOnException(self.settings["focusOnCatchException"])
		self.output.setFocusOnOutput(self.settings["focusOnCatchOutput"])
		self.output.printExceptionInStatusBar(self.settings["printExceptionInStatusBar"])
		self.output.printOutputInStatusBar(self.settings["printOutputInStatusBar"])

		self.show()
		
		# create new schema
		win = self.menuItemNewSchema()
		self.workspace.cascade()

	def createWidgetsToolbar(self, rebuildRegistry):
		self.widgetsToolBar.clear()
		self.tabs = orngTabs.WidgetTabs(self.widgetsToolBar, 'tabs')
		self.addToolBar(self.widgetsToolBar, "Widgets", QMainWindow.Top, TRUE)
		
		self.tabs.setCanvasDlg(self)
		
		# if registry doesn't exist yet, we create it
		if rebuildRegistry == 1:
			parse = xmlParse.WidgetsToXML()
			parse.ParseWidgetRoot(self.widgetDir, self.canvasDir)
			
		# if registry still doesn't exist then something is very wrong...
		if not os.path.exists(self.registryFileName):
			QMessageBox.critical( None, "Orange Canvas", "Unable to locate widget registry. Exiting...", QMessageBox.Ok, QMessageBox.Cancel)
			self.quit()

		if self.settings.has_key("WidgetTabs"):
			widgetTabList = self.settings["WidgetTabs"]
		else:
			widgetTabList = ["Data", "Classify", "Evaluate", "Visualize", "Associate", "Genomics", "Miscelaneous"]
			
		for tab in widgetTabList: self.tabs.insertWidgetTab(tab)
				
		# read widget registry file and create tab with buttons
		self.tabs.readInstalledWidgets(self.registryFileName, self.widgetDir, self.picsDir, self.defaultPic, self.useLargeIcons)

		# store order to settings list
		widgetTabList = []
		for tab in self.tabs.tabs:
			widgetTabList.append(str(self.tabs.tabLabel(tab)))
		self.settings["WidgetTabs"] = widgetTabList
			
		
	def initMenu(self):
		# ###################
		# menu items
		# ###################
		self.menuRecent = QPopupMenu(self)
		
		self.menuFile = QPopupMenu( self )
		self.menuFile.insertItem(QIconSet(QPixmap(orngResources.file_new)), "&New",  self.menuItemNewSchema, Qt.CTRL+Qt.Key_N )
		self.menuFile.insertItem( "New from template",  self.menuItemNewFromTemplate)
		self.menuFile.insertItem( "New from wizard",  self.menuItemNewWizard)
		self.menuFile.insertItem(QIconSet(QPixmap(orngResources.file_open)), "&Open", self.menuItemOpen, Qt.CTRL+Qt.Key_O )
		self.menuFile.insertItem( "&Close", self.menuItemClose )
		self.menuFile.insertSeparator()
		self.menuSaveID = self.menuFile.insertItem(QIconSet(QPixmap(orngResources.file_save)), "&Save", self.menuItemSave, Qt.CTRL+Qt.Key_S )
		self.menuSaveAsID = self.menuFile.insertItem( "&Save As..", self.menuItemSaveAs)
		self.menuFile.insertItem( "&Save As Application (Tabs)", self.menuItemSaveAsAppTabs)
		self.menuFile.insertItem( "&Save As Application (Buttons)", self.menuItemSaveAsAppButtons)
		self.menuFile.insertSeparator()
		self.menuFile.insertItem(QIconSet(QPixmap(orngResources.file_print)), "&Print Schema / Save image", self.menuItemPrinter, Qt.CTRL+Qt.Key_P )
		self.menuFile.insertSeparator()
		self.menuFile.insertItem( "Recent Files", self.menuRecent)
		self.menuFile.insertSeparator()
		#self.menuFile.insertItem( "E&xit",  qApp, SLOT( "quit()" ), Qt.CTRL+Qt.Key_Q )
		self.menuFile.insertItem( "E&xit",  self.close, Qt.CTRL+Qt.Key_Q )

		self.menuEdit = QPopupMenu( self )
		self.menuEdit.insertItem( "Cut",  self.menuItemCut, Qt.CTRL+Qt.Key_X )
		self.menuEdit.insertItem( "Copy",  self.menuItemCopy, Qt.CTRL+Qt.Key_C )
		self.menuEdit.insertItem( "Paste",  self.menuItemPaste, Qt.CTRL+Qt.Key_V )
		self.menuFile.insertSeparator()
		self.menuEdit.insertItem( "Select All",  self.menuItemSelectAll, Qt.CTRL+Qt.Key_A )

		self.menuOptions = QPopupMenu( self )
		#self.menuOptions.insertItem( "Grid",  self.menuItemGrid )
		#self.menuOptions.insertSeparator()
		#self.menuOptions.insertItem( "Show Grid",  self.menuItemShowGrid)

		self.menuOptions.insertItem( "Enable All Links",  self.menuItemEnableAll, Qt.CTRL+Qt.Key_E)
		self.menuOptions.insertItem( "Disable All Links",  self.menuItemDisableAll, Qt.CTRL+Qt.Key_D)
		self.menuOptions.insertItem( "Clear Scheme",  self.menuItemClearWidgets)
		self.menuOptions.insertSeparator()
		#self.menuOptions.insertItem( "Channel preferences",  self.menuItemPreferences)
		#self.menuOptions.insertSeparator()
		self.menuOptions.insertItem( "Rebuild widget registry",  self.menuItemRebuildWidgetRegistry)
		self.menuOptions.insertSeparator()
		self.menuOptions.insertItem( "Canvas options",  self.menuItemCanvasOptions)
		
		self.menuWindow = QPopupMenu( self )		
		self.menuWindow.insertItem("Cascade", self.workspace.cascade)
		self.menuWindow.insertItem("Tile", self.workspace.tile)
		self.menuWindow.insertSeparator()
		
		self.connect(self.menuWindow, SIGNAL("aboutToShow()"), self.showWindows)

		self.menupopupShowToolbarID = self.menuWindow.insertItem( "Toolbar",  self.menuItemShowToolbar )
		if self.settings.has_key("showToolbar"): self.showToolbar = self.settings["showToolbar"]
		else:									self.showToolbar = TRUE
		if not self.showToolbar: self.toolbar.hide()
		self.menuWindow.setItemChecked(self.menupopupShowToolbarID, self.showToolbar)

		self.menupopupShowWidgetToolbarID = self.menuWindow.insertItem( "Widget toolbar",  self.menuItemShowWidgetToolbar)
		if self.settings.has_key("showWidgetToolbar"): self.showWidgetToolbar = self.settings["showWidgetToolbar"]
		else:									self.showWidgetToolbar = TRUE
		if not self.showWidgetToolbar: self.widgetsToolBar.hide()
		self.menuWindow.setItemChecked(self.menupopupShowWidgetToolbarID, self.showWidgetToolbar)
		
		self.menuWindow.insertSeparator()
		self.menuOutput = QPopupMenu(self)
		self.menuWindow.insertItem( "Output Window", self.menuOutput)
		self.menuOutput.insertItem("Show Output Window", self.menuItemShowOutputWindow)
		self.menuOutput.insertItem("Clear Output Window", self.menuItemClearOutputWindow)
		self.menuOutput.insertSeparator()
		self.menuOutput.insertItem("Save Output Text", self.menuItemSaveOutputWindow)
		self.menuWindow.insertSeparator()

		self.menuWindow.insertItem("Minimize All", self.menuMinimizeAll)
		self.menuWindow.insertItem("Restore All", self.menuRestoreAll)
		self.menuWindow.insertItem("Close All", self.menuCloseAll)
		self.menuWindow.insertSeparator()

		self.menuBar = QMenuBar( self ) 
		self.menuBar.insertItem( "&File", self.menuFile )
		#self.menuBar.insertItem( "&Edit", self.menuEdit )
		self.menuBar.insertItem( "&Options", self.menuOptions )
		self.menuBar.insertItem("&Window", self.menuWindow)

		self.printer = QPrinter() 


	def showWindows(self):
		for id in self.windows:
			self.menuWindow.removeItem(id)
		self.windows = []
		wins = self.workspace.windowList()
		for win in wins:
			#id = self.menuWindow.insertItem(str(win.caption()), SLOT("windowsMenuActivated( int )"), self.activateWindow)
			id = self.menuWindow.insertItem(str(win.caption()), self.activateWindow)
			self.windows.append(id)

	def activateWindow(self, id):
		caption = self.menuWindow.text(id)
		winList = self.workspace.windowList()
		for win in winList:
			if str(win.caption()) == caption:
				win.setFocus()

	def menuItemNewSchema(self):
		win = orngDoc.SchemaDoc(self, self.workspace)
		self.workspace.setDefaultDocPosition(win)
		return win

	def menuItemNewFromTemplate(self):
		return
	
	def menuItemNewWizard(self):
		return

	def menuItemOpen(self):
		name = QFileDialog.getOpenFileName( os.getcwd(), "Orange Widget Scripts (*.ows)", self, "", "Open File")
		if name.isEmpty():
			return
		win = self.menuItemNewSchema()
		win.loadDocument(str(name))
		self.addToRecentMenu(str(name))

	def menuItemClose(self):
		win = self.workspace.activeWindow()
		win.close()
		
	def menuItemSave(self):
		win = self.workspace.activeWindow()
		if isinstance(win, orngDoc.SchemaDoc):
			win.saveDocument()
		elif isinstance(win, orngOutput.OutputWindow):
			self.menuItemSaveOutputWindow()

	def menuItemSaveAs(self):
		win = self.workspace.activeWindow()
		if isinstance(win, orngDoc.SchemaDoc):
			win.saveDocumentAs()

	def menuItemSaveAsAppButtons(self):
		win = self.workspace.activeWindow()
		if isinstance(win, orngDoc.SchemaDoc):
			win.saveDocumentAsApp(asTabs = 0)

	def menuItemSaveAsAppTabs(self):
		win = self.workspace.activeWindow()
		if isinstance(win, orngDoc.SchemaDoc):
			win.saveDocumentAsApp(asTabs = 1)	

	def menuItemPrinter(self):
		try:
			import OWDlgs
		except:
			print "Missing file OWDlgs.py. This file should be in widget directory. Unable to print/save image."
			return
		win = self.workspace.activeWindow()
		if not isinstance(win, orngDoc.SchemaDoc):
			return
		sizeDlg = OWDlgs.OWChooseImageSizeDlg(win.canvas)
		sizeDlg.exec_loop()
		

	def readRecentFiles(self):
		if not self.settings.has_key("RecentFiles"): return
		recentDocs = self.settings["RecentFiles"]
		self.menuRecent.clear()

		for i in range(len(recentDocs)):
			name = recentDocs[i]
			shortName = os.path.basename(name)
			self.menuRecent.insertItem(shortName, eval("self.menuItemRecent"+str(i+1)))

	def openRecentFile(self, index):
		if len(self.settings["RecentFiles"]) >= index:
			win = self.menuItemNewSchema()
			win.loadDocument(self.settings["RecentFiles"][index-1])
			self.addToRecentMenu(self.settings["RecentFiles"][index-1])

	def addToRecentMenu(self, name):
		recentDocs = []
		if self.settings.has_key("RecentFiles"):
			recentDocs = self.settings["RecentFiles"]
		
		if name in recentDocs:
			recentDocs.remove(name)
		recentDocs.insert(0, name)
	
		if len(recentDocs)> 5:
			recentDocs.remove(recentDocs[5])
		self.settings["RecentFiles"] = recentDocs
		self.readRecentFiles()

	def menuItemRecent1(self):
		self.openRecentFile(1)

	def menuItemRecent2(self):
		self.openRecentFile(2)

	def menuItemRecent3(self):
		self.openRecentFile(3)

	def menuItemRecent4(self):
		self.openRecentFile(4)

	def menuItemRecent5(self):
		self.openRecentFile(5)

	def menuItemCut(self):
		return

	def menuItemCopy(self):
		return

	def menuItemPaste(self):
		return

	def menuItemSelectAll(self):
		return

	def menuItemGrid(self):
		return
		
	def menuItemShowGrid(self):
		return
		
	def updateSnapToGrid(self):
		if self.snapToGrid == TRUE:
			for win in self.workspace.windowList():
				if not isinstance(win, orngDoc.SchemaDoc): continue
				for widget in win.widgets:
					widget.setCoords(widget.x(), widget.y())
					widget.moveToGrid()
					widget.repaintAllLines()
				win.canvas.update()

	def updateUseLargeIcons(self):
		self.tabs.hide()
		self.tabs = orngTabs.WidgetTabs(self.widgetsToolBar, 'tabs')
		self.tabs.setCanvasDlg(self)
		self.tabs.readInstalledWidgets(self.registryFileName, self.widgetDir, self.picsDir, self.defaultPic, self.useLargeIcons)
		
	def menuItemEnableAll(self):
		win = self.workspace.activeWindow()
		if isinstance(win, orngDoc.SchemaDoc):
			win.enableAllLines()
		return
		
	def menuItemDisableAll(self):
		win = self.workspace.activeWindow()
		if isinstance(win, orngDoc.SchemaDoc):
			win.disableAllLines()
		return

	def menuItemClearWidgets(self):
		win = self.workspace.activeWindow()
		if win != None:
			win.clear()

	def menuItemShowOutputWindow(self):
		self.output.show()
		self.output.setFocus()

	def menuItemClearOutputWindow(self):
		self.output.textOutput.setText("")
		self.statusBar.message("")

	def menuItemSaveOutputWindow(self):
		qname = QFileDialog.getSaveFileName( self.canvasDir + "/Output.htm", "HTML Document (*.htm)", self, "", "Save Output To File")
		if qname.isEmpty(): return
		name = str(qname)

		text = str(self.output.textOutput.text())
		text = text.replace("</nobr>", "</nobr><br>")

		file = open(name, "wt")
		file.write(text)
		file.close()


	def menuItemShowToolbar(self):
		self.showToolbar = not self.showToolbar
		self.settings["showToolbar"] = self.showToolbar
		self.menuWindow.setItemChecked(self.menupopupShowToolbarID, self.showToolbar)
		if self.showToolbar: self.toolbar.show()
		else: self.toolbar.hide()

	def menuItemShowWidgetToolbar(self):
		self.showWidgetToolbar = not self.showWidgetToolbar
		self.settings["showWidgetToolbar"] = self.showWidgetToolbar
		self.menuWindow.setItemChecked(self.menupopupShowWidgetToolbarID, self.showWidgetToolbar)
		if self.showWidgetToolbar: self.widgetsToolBar.show()
		else: self.widgetsToolBar.hide()

	def menuItemPreferences(self):
		dlg = orngDlgs.PreferencesDlg(self, None, "", TRUE)
		dlg.exec_loop()
		if dlg.result() == QDialog.Accepted:
			self.rebuildSignals()

	def menuItemRebuildWidgetRegistry(self):
		self.createWidgetsToolbar(TRUE)
		
	def menuCloseAll(self):
		wins = self.workspace.windowList()
		for win in wins:
			win.close()
			
	def menuMinimizeAll(self):
		win.showMinimized()
			
	def menuRestoreAll(self):
		win.showNormal()

	def rebuildSignals(self):
		self.channels = {}
		if self.settings.has_key("Channels"):
			channels = self.settings["Channels"]
			for (key, value) in channels.items():
				items = value.split("::")
				self.channels[key] = items

	def getChannelName(self, symbName):
		if self.channels.has_key(symbName):
			return (self.channels[symbName])[0]
		return symbName
	
	def getChannelInfo(self, symbName):
		if self.channels.has_key(symbName):
			return self.channels[symbName]
		else:
			return [symbName, str(1), "green"]

	def focusDocument(self, w):	
		if w != None:
			w.setFocus()


	def menuItemCanvasOptions(self):
		dlg = orngDlgs.CanvasOptionsDlg(self, None, "", TRUE)

		# set general options settings
		dlg.snapToGridCB.setChecked(self.snapToGrid)
		dlg.useLargeIconsCB.setChecked(self.useLargeIcons)

		# set current exception settings
		dlg.catchExceptionCB.setChecked(self.settings["catchException"])
		dlg.focusOnCatchExceptionCB.setChecked(self.settings["focusOnCatchException"])
		dlg.printExceptionInStatusBarCB.setChecked(self.settings["printExceptionInStatusBar"])
		dlg.catchOutputCB.setChecked(self.settings["catchOutput"])
		dlg.focusOnCatchOutputCB.setChecked(self.settings["focusOnCatchOutput"])
		dlg.printOutputInStatusBarCB.setChecked(self.settings["printOutputInStatusBar"])

		# fill categories tab list
		oldTabList = []
		for tab in self.tabs.tabs:
			dlg.tabOrderList.insertItem(self.tabs.tabLabel(tab))
			oldTabList.append(str(self.tabs.tabLabel(tab)))

		dlg.exec_loop()
		if dlg.result() == QDialog.Accepted:
			# save general settings
			if self.snapToGrid != dlg.snapToGridCB.isChecked():
				self.snapToGrid = dlg.snapToGridCB.isChecked()
				self.settings["snapToGrid"] = self.snapToGrid
				self.updateSnapToGrid()

			if self.useLargeIcons != dlg.useLargeIconsCB.isChecked():
				self.useLargeIcons = dlg.useLargeIconsCB.isChecked()
				self.settings["useLargeIcons"] = self.useLargeIcons
				self.updateUseLargeIcons()
					
			# save exceptions settings
			self.settings["catchException"] = dlg.catchExceptionCB.isChecked()
			self.settings["catchOutput"] = dlg.catchOutputCB.isChecked()
			self.settings["printExceptionInStatusBar"] = dlg.printExceptionInStatusBarCB.isChecked()
			self.settings["focusOnCatchException"] = dlg.focusOnCatchExceptionCB.isChecked()
			self.settings["focusOnCatchOutput"] = dlg.focusOnCatchOutputCB.isChecked()
			self.settings["printOutputInStatusBar"] = dlg.printOutputInStatusBarCB.isChecked()
			self.output.catchException(self.settings["catchException"])
			self.output.catchOutput(self.settings["catchOutput"])
			self.output.printExceptionInStatusBar(self.settings["printExceptionInStatusBar"])
			self.output.printOutputInStatusBar(self.settings["printOutputInStatusBar"])
			self.output.setFocusOnException(self.settings["focusOnCatchException"])
			self.output.setFocusOnOutput(self.settings["focusOnCatchOutput"])

			# save tab order settings
			newTabList = []
			for i in range(dlg.tabOrderList.count()):
				newTabList.append(str(dlg.tabOrderList.text(i)))
			if newTabList != oldTabList:
				self.settings["WidgetTabs"] = newTabList
				self.createWidgetsToolbar(0)


	def setStatusBarEvent(self, text):
		if text == "":
			self.statusBar.message("")
			return
		elif text == "\n": return
		self.statusBar.message(QString("Last event: " + text))
		
	#######################
	# SETTINGS
	#######################

	# Loads settings from the widget's .ini file	
	def loadSettings(self):
		filename = self.canvasDir + "orngCanvas.ini"
		if os.path.exists(filename):
			file = open(filename)
			self.settings = cPickle.load(file)
			file.close()
		else:
			self.settings = {}

		if not self.settings.has_key("catchException"): self.settings["catchException"] = 1
		if not self.settings.has_key("catchOutput"): self.settings["catchOutput"] = 1
		if not self.settings.has_key("focusOnCatchException"): self.settings["focusOnCatchException"] = 1
		if not self.settings.has_key("focusOnCatchOutput"): self.settings["focusOnCatchOutput"] = 0
				

	# Saves settings to this widget's .ini file
	def saveSettings(self):
		filename = self.canvasDir + "orngCanvas.ini"
		file=open(filename, "w")
		cPickle.dump(self.settings, file)
		file.close()


	#######################
	# EVENTS
	#######################

	def closeEvent(self, ce):
		for win in self.workspace.windowList():
			win.close()

		self.saveSettings()
		if len(self.workspace.windowList()) == 0:
			ce.accept()
		else:
			ce.ignore()

	def keyPressEvent(self, e):
		if e.state() & Qt.ControlButton != 0:
			self.ctrlPressed = 1

	def keyReleaseEvent(self, e):
		self.ctrlPressed = 0

	def enableSave(self, enable):
		self.toolSave.setEnabled(enable)
		self.menuFile.setItemEnabled(self.menuSaveID, enable)
		self.menuFile.setItemEnabled(self.menuSaveAsID, enable)

class WidgetWorkspace(QWorkspace):
	def __init__(self,*args):
		apply(QWorkspace.__init__,(self,) + args)
		self.off = 30

	# ###########
	# override the default cascade function
	def cascade(self):
		list = self.windowList()
		outputWin = None
		for item in list:
			if isinstance(item, orngOutput.OutputWindow):
				outputWin = item
		if outputWin:
			list.remove(outputWin)

		# move schemas
		pos = 0
		for item in list:
			item.parentWidget().move(pos,pos)
			item.parentWidget().resize(self.width()-pos, self.height()-pos)
			pos += self.off

		# move output win
		if outputWin:
			outputWin.parentWidget().move(pos,pos)
			outputWin.parentWidget().resize(self.width()-pos, self.height()-pos)

	# #################
	# position new window down and right to the last window. move output window down and right to the new window
	def setDefaultDocPosition(self, win):
		k = len(self.windowList())-2
		win.parentWidget().move(k*self.off,k*self.off)
		win.parentWidget().resize(self.width()-k*self.off, self.height()-k*self.off)

		list = self.windowList()
		for item in list:
			if isinstance(item, orngOutput.OutputWindow):
				item.parentWidget().move((k+1)*self.off,(k+1)*self.off)
				item.parentWidget().resize(self.width()-(k+1)*self.off, self.height()-(k+1)*self.off)


app = QApplication(sys.argv) 
dlg = OrangeCanvasDlg()
app.setMainWidget(dlg)
dlg.show()
app.exec_loop() 