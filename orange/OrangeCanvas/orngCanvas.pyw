# Author: Gregor Leban (gregor.leban@fri.uni-lj.si)
# Description:
#    main file, that creates the MDI environment
from PyQt4.QtCore import *
from PyQt4.QtGui import *
import sys, os, cPickle, orngGui
import orngTabs, orngDoc, orngDlgs, orngOutput
import orange, user, orngMisc, orngRegistry, orngEnviron

class OrangeCanvasDlg(QMainWindow):
    def __init__(self, app, parent = None, flags = 0):
        QMainWindow.__init__(self, parent)
        self.debugMode = 1        # print extra output for debuging
        self.setWindowTitle("Orange Canvas")
        self.windows = []    # list of id for windows in Window menu
        self.windowsDict = {}    # dict. with id:menuitem for windows in Window menu
        self.recentDocs = []
        self.iDocIndex = 1
        self.iconSizeList = ["16 x 16", "32 x 32", "40 x 40", "48 x 48", "60 x 60"]
        self.iconSizeDict = dict((val, int(val[:2])) for val in self.iconSizeList)
        self.originalPalette = QApplication.palette()

        self.__dict__.update(orngEnviron.directoryNames)

        self.defaultPic = os.path.join(self.picsDir, "Unknown.png")
        canvasPicsDir  = os.path.join(self.canvasDir, "icons")
        self.file_new  = os.path.join(canvasPicsDir, "doc.png")
        self.outputPix = os.path.join(canvasPicsDir, "output.png")
        self.file_open = os.path.join(canvasPicsDir, "open.png")
        self.file_save = os.path.join(canvasPicsDir, "save.png")
        self.text_icon = os.path.join(canvasPicsDir, "text.png")
        self.file_print= os.path.join(canvasPicsDir, "print.png")
        self.file_exit = os.path.join(canvasPicsDir, "exit.png")
        canvasIconName = os.path.join(canvasPicsDir, "CanvasIcon.png")
        if os.path.exists(canvasIconName):
            self.setWindowIcon(QIcon(canvasIconName))

        # create error and warning icons
        informationIconName = os.path.join(canvasPicsDir, "information.png")
        warningIconName = os.path.join(canvasPicsDir, "warning.png")
        errorIconName = os.path.join(canvasPicsDir, "error.png")
        if os.path.exists(errorIconName) and os.path.exists(warningIconName) and os.path.exists(informationIconName):
            self.errorIcon = QPixmap(errorIconName)
            self.warningIcon = QPixmap(warningIconName)
            self.informationIcon = QPixmap(informationIconName)
            self.widgetIcons = {"Info": self.informationIcon, "Warning": self.warningIcon, "Error": self.errorIcon}
        else:
            self.errorIcon = None
            self.warningIcon = None
            self.informationIcon = None
            self.widgetIcons = None
            print "Unable to load all necessary icons. Please reinstall Orange."

        self.workspace = WidgetMdiArea(self)
        #self.workspace.setBackgroundColor(QColor(255,255,255))
        self.setCentralWidget(self.workspace)
        self.statusBar = MyStatusBar(self)
        self.setStatusBar(self.statusBar)
        #self.connect(self.workspace, SIGNAL("subWindowActivated(QMdiSubWindow*)"), self.focusDocument)

        self.settings = {}
        self.widgetInfo = {} # this is a dictionary with items: category-widget_name : {info about widget (inList, outList, description...}

        self.rebuildSignals()    # coloring of signals - unused!
        self.menuSaveSettingsID = -1
        self.menuSaveSettings = 1

        self.loadSettings()

        self.updateStyle()

        self.widgetSelectedColor = QColor(self.settings["widgetSelectedColor"][0], self.settings["widgetSelectedColor"][1], self.settings["widgetSelectedColor"][2])
        self.widgetActiveColor   = QColor(self.settings["widgetActiveColor"][0], self.settings["widgetActiveColor"][1], self.settings["widgetActiveColor"][2])
        self.lineColor           = QColor(self.settings["lineColor"][0], self.settings["lineColor"][1], self.settings["lineColor"][2])

        # create toolbar
        self.toolbar = self.addToolBar("Manage Schemas")
        self.toolbar.setOrientation(Qt.Horizontal)
        self.widgetListTypeToolbar = self.addToolBar("Widget List Type")
        self.widgetListTypeToolbar.setOrientation(Qt.Horizontal)


        # create menu
        self.initMenu()

        self.toolbar.addAction(QIcon(self.file_new), "New schema" , self.menuItemNewSchema)
        self.toolbar.addAction(QIcon(self.file_open), "Open schema", self.menuItemOpen)
        self.toolSave = self.toolbar.addAction(QIcon(self.file_save), "Save schema", self.menuItemSave)
        self.toolbar.addSeparator()
        self.toolbar.addAction(QIcon(self.file_print), "Print", self.menuItemPrinter)

        w = QWidget()
        w.setLayout(QHBoxLayout())
        self.widgetListTypeToolbar.addWidget(w)
        self.widgetOrganizationCombo = orngGui.comboBox(w, label = "Style:", orientation = "horizontal", items = ["Tool box", "Tree view", "Tabs without labels", "Tabs with labels"])
        
        self.connect(self.widgetOrganizationCombo, SIGNAL('activated(int)'), self.widgetListTypeChanged)
        try:        # maybe we will someday remove some of the options and the index will be too big
            self.widgetOrganizationCombo.setCurrentIndex(self.settings["widgetListType"])
        except:
            pass
        
#        self.widgetListTypeGroup = QActionGroup(self)
#        self.widgetListTypeGroup.addAction(self.widgetListTypeToolbar.addAction(QIcon(self.text_icon), "Show list of widgets in a toolbox", self.widgetListTypeChanged))
#        self.widgetListTypeGroup.addAction(self.widgetListTypeToolbar.addAction(QIcon(self.text_icon), "Show list of widgets in a tab bar", self.widgetListTypeChanged))
#        self.widgetListTypeGroup.addAction(self.widgetListTypeToolbar.addAction(QIcon(self.text_icon), "Show list of widgets and their names in a tab bar ", self.widgetListTypeChanged))
#        for action in self.widgetListTypeGroup.actions():
#            action.setCheckable(1)
#        self.widgetListTypeGroup.actions()[self.settings["widgetListType"]].setChecked(1)


        self.widgetListTypeToolbar.addSeparator()

        self.widgetListTypeToolbar.addWidget(QLabel("Icon size: ", self))
        self.iconSizeCombo = QComboBox(self)
        self.iconSizeCombo.addItems(self.iconSizeList)
        self.widgetListTypeToolbar.addWidget(self.iconSizeCombo)
        self.connect(self.iconSizeCombo, SIGNAL("activated(int)"), self.iconSizeChanged)
        try:
            self.iconSizeCombo.setCurrentIndex(self.iconSizeList.index(self.settings["iconSize"]))
        except:
            self.iconSizeCombo.setCurrentIndex(self.iconSizeList.index("48 x 48"))

        self.addToolBarBreak()

        self.createWidgetsToolbar()

        self.readShortcuts()

        # read recent files
        self.readRecentFiles()

        width  = self.settings.get("canvasWidth", 700)
        height = self.settings.get("canvasHeight", 600)
        self.resize(width, height)

        # center window in the desktop
        # in newer versions of Qt we can also find the center of a primary screen
        # on multiheaded desktops
        desktop = app.desktop()
        deskH = desktop.screenGeometry(desktop.primaryScreen()).height()
        deskW = desktop.screenGeometry(desktop.primaryScreen()).width()
        h = max(0, deskH/2 - height/2)  # if the window is too small, resize the window to desktop size
        w = max(0, deskW/2 - width/2)
        self.move(w,h)


        # apply output settings
        self.output = orngOutput.OutputWindow(self, self.workspace)
        self.output.show()
        #self.output.catchException(self.settings["catchException"])
        #self.output.catchOutput(self.settings["catchOutput"])
        self.output.catchException(1)
        self.output.catchOutput(1)
        self.output.setFocusOnException(self.settings["focusOnCatchException"])
        self.output.setFocusOnOutput(self.settings["focusOnCatchOutput"])
        self.output.printExceptionInStatusBar(self.settings["printExceptionInStatusBar"])
        self.output.printOutputInStatusBar(self.settings["printOutputInStatusBar"])
        self.output.setWriteLogFile(self.settings["writeLogFile"])
        self.output.setVerbosity(self.settings["outputVerbosity"])

        self.show()

        # did Orange crash the last time we used it? If yes, you will find a TempSchemaX.ows file
        tempSchemaNames = []
        loadedTempSchemas = False
        for fname in os.listdir(self.canvasSettingsDir):
            if "TempSchema " in fname:
                tempSchemaNames.append(os.path.join(self.canvasSettingsDir, fname))
        mb = QMessageBox('Orange Canvas', "Your previous Orange Canvas session was not closed successfully.\nYou can choose to reload your unsaved work or start a new session.\n\nIf you choose 'Reload', the links will be disabled to prevent reoccurence of the crash.\nYou can enable them by clicking Options/Enable all links.", QMessageBox.Information, QMessageBox.Ok | QMessageBox.Default, QMessageBox.Cancel | QMessageBox.Escape, QMessageBox.NoButton)
        mb.setButtonText(QMessageBox.Ok, "Reload")
        mb.setButtonText(QMessageBox.Cancel, "New schema")
        if tempSchemaNames != [] and mb.exec_() == QMessageBox.Ok:
            loadedTempSchemas = True
            for fname in tempSchemaNames:
                win = self.menuItemNewSchema()
                win.loadDocument(fname, freeze = 1, isTempSchema = 1)
        else:
            # first remove old temp schemas if they exist
            for fname in tempSchemaNames:
                os.remove(fname)

            # if not temp schemas were loaded create a new schema
            # in case we also received a schema's name as the argument, we load it
            win = self.menuItemNewSchema()
            if len(sys.argv) > 1 and os.path.splitext(sys.argv[-1])[1].lower() == ".ows":
                win.loadDocument(sys.argv[-1])

        self.workspace.cascadeSubWindows()

        # show message box if no numpy
        qApp.processEvents()
        try:
            import numpy
        except:
            if QMessageBox.warning(self,'Orange Canvas','Several widgets now use numpy module, \nthat is not yet installed on this computer. \nDo you wish to download it?',QMessageBox.Ok | QMessageBox.Default, QMessageBox.Cancel | QMessageBox.Escape) == QMessageBox.Ok:
                import webbrowser
                webbrowser.open("http://sourceforge.net/projects/numpy/")


    def createWidgetsToolbar(self):
        if hasattr(self, "widgetsToolBar"):
            if isinstance(self.widgetsToolBar, QToolBar):
                self.removeToolBar(self.widgetsToolBar)
            elif isinstance(self.widgetsToolBar, orngTabs.WidgetToolBox):
                self.settings["toolboxWidth"] = self.widgetsToolBar.toolbox.width()
                self.removeDockWidget(self.widgetsToolBar)
            elif isinstance(self.widgetsToolBar, orngTabs.WidgetTree):
                self.settings["toolboxWidth"] = self.widgetsToolBar.treeWidget.width()
                self.removeDockWidget(self.widgetsToolBar)

        if self.settings["widgetListType"] == 0:
            self.tabs = self.widgetsToolBar = orngTabs.WidgetToolBox(self, self.widgetInfo)
            self.addDockWidget(Qt.LeftDockWidgetArea, self.widgetsToolBar)
        elif self.settings["widgetListType"] == 1:
            self.tabs = self.widgetsToolBar = orngTabs.WidgetTree(self, self.widgetInfo)
            self.addDockWidget(Qt.LeftDockWidgetArea, self.widgetsToolBar)
        else:
            self.widgetsToolBar = self.addToolBar("Widgets")
            self.insertToolBarBreak(self.widgetsToolBar)
            self.tabs = orngTabs.WidgetTabs(self, self.widgetInfo, self.widgetsToolBar)
            self.widgetsToolBar.addWidget(self.tabs)

        if self.settings.has_key("WidgetTabs") and self.settings["WidgetTabs"] != []:
            widgetTabList = self.settings["WidgetTabs"]
        else:
            widgetTabList = [(name, Qt.Checked) for name in ["Data", "Visualize", "Classify", "Regression", "Evaluate", "Unsupervised", "Associate", "Text", "Genomics", "Prototypes"]]

        # find widgets and create tab with buttons
        self.tabs.readInstalledWidgets(widgetTabList, self.widgetDir, self.picsDir, self.defaultPic)
        self.settings["WidgetTabs"] = [(name, show) for (name, show, w) in self.tabs.tabs]

        #qApp.processEvents()
        self.workspace.cascadeSubWindows()



    def readShortcuts(self):
        self.widgetShortcuts = {}
        shfn = os.path.join(self.canvasSettingsDir, "shortcuts.txt")
        if os.path.exists(shfn):
            for t in file(shfn).readlines():
                key, button = [x.strip() for x in t.split(":")]
                widget = self.tabs.widgetInfo.get(button)
                if widget:
                    self.widgetShortcuts[key] = widget["button"]
                else:
                    print "Warning: invalid shortcut %s (%s)" % (key, button)

    def initMenu(self):
        # ###################
        # menu items
        # ###################
        self.menuRecent = QMenu("Recent Schemas", self)

        self.menuFile = QMenu("&File", self)
        self.menuFile.addAction(QIcon(self.file_new), "&New",  self.menuItemNewSchema, Qt.CTRL+Qt.Key_N )
        #self.menuFile.addAction( "New from template",  self.menuItemNewFromTemplate)
        #self.menuFile.addAction( "New from wizard",  self.menuItemNewWizard)
        self.menuFile.addAction(QIcon(self.file_open), "&Open...", self.menuItemOpen, Qt.CTRL+Qt.Key_O )
        self.menuFile.addAction(QIcon(self.file_open), "&Open and Freeze...", self.menuItemOpenFreeze)
        if os.path.exists(os.path.join(self.canvasSettingsDir, "_lastSchema.ows")):
            self.menuFile.addAction("Reload Last Schema", self.menuItemOpenLastSchema, Qt.CTRL+Qt.Key_R)
        self.menuFile.addAction( "&Close", self.menuItemClose, Qt.CTRL+Qt.Key_W)
        self.menuFile.addSeparator()
        self.menuSaveID = self.menuFile.addAction(QIcon(self.file_save), "&Save", self.menuItemSave, Qt.CTRL+Qt.Key_S )
        self.menuSaveAsID = self.menuFile.addAction( "Save &As...", self.menuItemSaveAs)
        self.menuFile.addAction( "&Save as Application (Tabs)...", self.menuItemSaveAsAppTabs)
        self.menuFile.addAction( "&Save as Application (Buttons)...", self.menuItemSaveAsAppButtons)
        self.menuFile.addSeparator()
        self.menuFile.addAction(QIcon(self.file_print), "&Print Schema / Save image", self.menuItemPrinter, Qt.CTRL+Qt.Key_P )
        self.menuFile.addSeparator()
        self.menuFile.addMenu(self.menuRecent)
        self.menuFile.addSeparator()
        #self.menuFile.addAction( "E&xit",  qApp, SLOT( "quit()" ), Qt.CTRL+Qt.Key_Q )
        self.menuFile.addAction( "E&xit",  self.close, Qt.CTRL+Qt.Key_Q )

        self.menuEdit = QMenu("&Edit", self)
        self.menuEdit.addAction( "Cu&t",  self.menuItemCut, Qt.CTRL+Qt.Key_X )
        self.menuEdit.addAction( "&Copy",  self.menuItemCopy, Qt.CTRL+Qt.Key_C )
        self.menuEdit.addAction( "&Paste",  self.menuItemPaste, Qt.CTRL+Qt.Key_V )
        self.menuFile.addSeparator()
        self.menuEdit.addAction( "Select &All",  self.menuItemSelectAll, Qt.CTRL+Qt.Key_A )

        self.menuOptions = QMenu("&Options", self)
        #self.menuOptions.addAction( "Grid",  self.menuItemGrid )
        #self.menuOptions.addSeparator()
        #self.menuOptions.addAction( "Show Grid",  self.menuItemShowGrid)

        self.menuOptions.addAction( "Enable All Links",  self.menuItemEnableAll, Qt.CTRL+Qt.Key_E)
        self.menuOptions.addAction( "Disable All Links",  self.menuItemDisableAll, Qt.CTRL+Qt.Key_D)
        self.menuOptions.addAction( "Clear Scheme",  self.menuItemClearWidgets)

        # uncomment this only for debugging
        #self.menuOptions.addAction("Dump widget variables", self.dumpVariables)

        self.menuOptions.addSeparator()
        #self.menuOptions.addAction( "Channel preferences",  self.menuItemPreferences)
        #self.menuOptions.addSeparator()
        self.menuOptions.addAction( "&Customize Shortcuts",  self.menuItemEditWidgetShortcuts)
        self.menuOptions.addAction( "&Delete Widget Settings",  self.menuItemDeleteWidgetSettings)
        self.menuOptions.addSeparator()
        self.menuOptions.addAction( "Canvas &Options...",  self.menuItemCanvasOptions)

        self.menuWindow = QMenu("&Window", self)
        self.menuWindow.addAction("&Cascade", self.workspace.cascadeSubWindows)
        self.menuWindow.addAction("&Tile", self.workspace.tileSubWindows)
        self.menuWindow.addSeparator()

        self.connect(self.menuWindow, SIGNAL("aboutToShow()"), self.showWindows)

        self.menupopupShowToolbarID = self.menuWindow.addAction( "Toolbar",  self.menuItemShowToolbar )
        self.menupopupShowToolbarID.setCheckable(True)
        if self.settings.has_key("showToolbar"): self.showToolbar = self.settings["showToolbar"]
        else:                                    self.showToolbar = True
        if not self.showToolbar: self.toolbar.hide()
        self.menupopupShowToolbarID.setChecked(self.showToolbar)

        self.menupopupShowWidgetToolbarID = self.menuWindow.addAction( "Widget Toolbar",  self.menuItemShowWidgetToolbar)
        self.menupopupShowWidgetToolbarID.setCheckable(True)
        if self.settings.has_key("showWidgetToolbar"): self.showWidgetToolbar = self.settings["showWidgetToolbar"]
        else:                                          self.showWidgetToolbar = True
        if not self.showWidgetToolbar: self.widgetsToolBar.hide()
        self.menupopupShowWidgetToolbarID.setChecked(self.showWidgetToolbar)

        self.menuWindow.addSeparator()
        self.menuOutput = QMenu("Output", self)
        self.menuWindow.addMenu(self.menuOutput)
        self.menuOutput.addAction("Show Output Window", self.menuItemShowOutputWindow)
        self.menuOutput.addAction("Clear Output Window", self.menuItemClearOutputWindow)
        self.menuOutput.addSeparator()
        self.menuOutput.addAction("Save Output Text...", self.menuItemSaveOutputWindow)
        self.menuWindow.addSeparator()

        self.menuWindow.addAction("&Minimize All", self.menuMinimizeAll)
        self.menuWindow.addAction("Restore All", self.menuRestoreAll)
        self.menuWindow.addAction("Close All", self.menuCloseAll)
        self.menuWindow.addSeparator()

        localHelp = 0
        self.menuHelp = QMenu("&Help", self)
        if os.path.exists(os.path.join(self.orangeDir, r"doc/reference/default.htm")) or os.path.exists(os.path.join(self.orangeDir, r"doc/canvas/default.htm")):
            if os.path.exists(os.path.join(self.orangeDir, r"doc/reference/default.htm")): self.menuHelp.addAction("Orange Help", self.menuOpenLocalOrangeHelp)
            if os.path.exists(os.path.join(self.orangeDir, r"doc/canvas/default.htm")): self.menuHelp.addAction("Orange Canvas Help", self.menuOpenLocalCanvasHelp)

        self.menuHelp.addAction("Orange Online Help", self.menuOpenOnlineOrangeHelp)
        #self.menuHelp.addAction("Orange Canvas Online Help", self.menuOpenOnlineCanvasHelp)

        if os.path.exists(os.path.join(self.orangeDir, r"updateOrange.py")):
            self.menuHelp.addSeparator()
            self.menuHelp.addAction("Check for updates", self.menuCheckForUpdates)

        #self.menuHelp.addSeparator()
        #self.menuHelp.addAction("About Orange Canvas", self.menuHelpAbout)

        # widget popup menu
        self.widgetPopup = QMenu("Widget", self)
        self.widgetPopup.addAction( "Open",  self.openActiveWidget)
        self.widgetPopup.addSeparator()
        rename = self.widgetPopup.addAction( "&Rename", self.renameActiveWidget, Qt.Key_F2)
        delete = self.widgetPopup.addAction("Remove", self.removeActiveWidget, Qt.Key_Delete)

        self.menuBar = QMenuBar(self)
        self.menuBar.addMenu(self.menuFile)
        #self.menuBar.addMenu(self.menuEdit)
        self.menuBar.addMenu(self.menuOptions)
        self.menuBar.addMenu(self.menuWindow)
        self.menuBar.addMenu(self.widgetPopup)
        self.menuBar.addMenu(self.menuHelp)
        self.setMenuBar(self.menuBar)

        self.printer = QPrinter()


    def showWindows(self):
        for id in self.windowsDict.keys():
            self.menuWindow.removeAction(id)
        self.windowsDict = {}
        wins = [w for w in self.workspace.subWindowList() if w.isVisible()]
        for i in range(len(wins)):
            txt = str(i+1) + ' ' + str(wins[i].windowTitle())
            if i<10: txt = "&" + txt
            id = self.menuWindow.addAction(txt, wins[i].setFocus)
            self.windowsDict[id] = wins[i]

##    def actionEvent(self, event):
##        if event.type() == QEvent.ActionChanged and event.action() in self.windowsDict.keys():
##            self.windowsDict[event.action()].parentWidget.setFocus()


    def removeActiveWidget(self):
        win = self.workspace.activeSubWindow()
        if isinstance(win, orngDoc.SchemaDoc):
            win.canvasView.removeActiveWidget()

    def renameActiveWidget(self):
        win = self.workspace.activeSubWindow()
        if isinstance(win, orngDoc.SchemaDoc):
            win.canvasView.renameActiveWidget()

    def openActiveWidget(self):
        win = self.workspace.activeSubWindow()
        if isinstance(win, orngDoc.SchemaDoc):
            win.canvasView.openActiveWidget()


    def menuItemNewSchema(self, forceNew = 1):
        if not forceNew:
            for doc in self.workspace.getDocumentList():
                if doc.widgets == []: return doc
        win = orngDoc.SchemaDoc(self, self.workspace)
        #self.workspace.addSubWindow(win)
        win.show()
        return win

    def menuItemNewFromTemplate(self):
        return

    def menuItemNewWizard(self):
        return

    def menuItemOpen(self):
        name = QFileDialog.getOpenFileName(self, "Open File", self.settings["saveSchemaDir"], "Orange Widget Scripts (*.ows)")
        if name.isEmpty():
            return
        win = self.menuItemNewSchema(0)
        win.loadDocument(str(name), freeze = 0)
        self.addToRecentMenu(str(name))

    def menuItemOpenFreeze(self):
        name = QFileDialog.getOpenFileName(self, "Open File", self.settings["saveSchemaDir"], "Orange Widget Scripts (*.ows)")
        if name.isEmpty():
            return
        win = self.menuItemNewSchema(0)
        win.loadDocument(str(name), freeze = 1)
        self.addToRecentMenu(str(name))

    def menuItemOpenLastSchema(self):
        if os.path.exists(os.path.join(self.canvasSettingsDir, "_lastSchema.ows")):
            win = self.menuItemNewSchema(0)
            win.loadDocument(os.path.join(self.canvasSettingsDir, "_lastSchema.ows"), str(win.windowTitle()))

    def menuItemClose(self):
        win = self.workspace.activeSubWindow().close()


    def menuItemSave(self):
        win = self.workspace.activeSubWindow()
        if isinstance(win, orngDoc.SchemaDoc):
            win.saveDocument()
        elif isinstance(win, orngOutput.OutputWindow):
            self.menuItemSaveOutputWindow()

    def menuItemSaveAs(self):
        win = self.workspace.activeSubWindow()
        if isinstance(win, orngDoc.SchemaDoc):
            win.saveDocumentAs()

    def menuItemSaveAsAppButtons(self):
        win = self.workspace.activeSubWindow()
        if isinstance(win, orngDoc.SchemaDoc):
            win.saveDocumentAsApp(asTabs = 0)

    def menuItemSaveAsAppTabs(self):
        win = self.workspace.activeSubWindow()
        if isinstance(win, orngDoc.SchemaDoc):
            win.saveDocumentAsApp(asTabs = 1)

    def menuItemPrinter(self):
        try:
            import OWDlgs
        except:
            print "Missing file 'OWDlgs.py'. This file should be in OrangeWidgets folder. Unable to print/save image."
            return
        win = self.workspace.activeSubWindow()
        if not isinstance(win, orngDoc.SchemaDoc):
            return
        sizeDlg = OWDlgs.OWChooseImageSizeDlg(win.canvas)
        sizeDlg.exec_()


    def readRecentFiles(self):
        self.menuRecent.clear()
        if not self.settings.has_key("RecentFiles"): return
        recentDocs = self.settings["RecentFiles"]

        # remove missing recent files
        for i in range(len(recentDocs)-1,-1,-1):
            if not os.path.exists(recentDocs[i]):
                recentDocs.remove(recentDocs[i])

        recentDocs = recentDocs[:9]
        self.settings["RecentFiles"] = recentDocs

        for i in range(len(recentDocs)):
            shortName = "&" + str(i+1) + " " + os.path.basename(recentDocs[i])
            self.menuRecent.addAction(shortName, eval("self.menuItemRecent"+str(i+1)))

    def openRecentFile(self, index):
        if len(self.settings["RecentFiles"]) >= index:
            win = self.menuItemNewSchema(0)
            win.loadDocument(self.settings["RecentFiles"][index-1])
            self.addToRecentMenu(self.settings["RecentFiles"][index-1])

    def addToRecentMenu(self, name):
        recentDocs = []
        if self.settings.has_key("RecentFiles"):
            recentDocs = self.settings["RecentFiles"]

        # convert to a valid file name
        name = os.path.realpath(name)

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

    def menuItemRecent6(self):
        self.openRecentFile(6)

    def menuItemRecent7(self):
        self.openRecentFile(7)

    def menuItemRecent8(self):
        self.openRecentFile(8)

    def menuItemRecent9(self):
        self.openRecentFile(9)

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
        if self.settings["snapToGrid"]:
            for win in self.workspace.windowList():
                if not isinstance(win, orngDoc.SchemaDoc): continue
                for widget in win.widgets:
                    widget.setCoords(widget.x(), widget.y())
                    widget.repaintAllLines()
                win.canvas.update()

    def widgetListTypeChanged(self, ind):
        self.settings["widgetListType"] = ind
        self.createWidgetsToolbar()

    def iconSizeChanged(self, ind):
        self.settings["iconSize"] = self.iconSizeList[ind]
        self.createWidgetsToolbar()

    def menuItemEnableAll(self):
        win = self.workspace.activeSubWindow()
        if isinstance(win, orngDoc.SchemaDoc):
            win.enableAllLines()

    def menuItemDisableAll(self):
        win = self.workspace.activeSubWindow()
        if isinstance(win, orngDoc.SchemaDoc):
            win.disableAllLines()

    def menuItemSaveSettings(self):
        self.menuSaveSettings = not self.menuSaveSettings
        self.menuOptions.setItemChecked(self.menuSaveSettingsID, self.menuSaveSettings)

    def menuItemClearWidgets(self):
        win = self.workspace.activeSubWindow()
        if win != None:
            win.clear()

    def dumpVariables(self):
        win = self.workspace.activeSubWindow()
        if isinstance(win, orngDoc.SchemaDoc):
            win.dumpWidgetVariables()

    def menuItemShowOutputWindow(self):
        self.output.show()
        self.output.setFocus()

    def menuItemClearOutputWindow(self):
        self.output.textOutput.clear()
        self.statusBar.showMessage("")

    def menuItemSaveOutputWindow(self):
        qname = QFileDialog.getSaveFileName(self, "Save Output To File", self.canvasSettingsDir + "/Output.htm", "HTML Document (*.htm)")
        if qname.isEmpty(): return
        name = str(qname)

        text = str(self.output.textOutput.toHtml())
        #text = text.replace("</nobr>", "</nobr><br>")

        file = open(name, "wt")
        file.write(text)
        file.close()


    def menuItemShowToolbar(self):
        self.showToolbar = not self.showToolbar
        self.settings["showToolbar"] = self.showToolbar
        self.menupopupShowToolbarID.setChecked(self.showToolbar)
        if self.showToolbar: self.toolbar.show()
        else: self.toolbar.hide()

    def menuItemShowWidgetToolbar(self):
        self.showWidgetToolbar = not self.showWidgetToolbar
        self.settings["showWidgetToolbar"] = self.showWidgetToolbar
        self.menupopupShowWidgetToolbarID.setChecked(self.showWidgetToolbar)
        if self.showWidgetToolbar: self.widgetsToolBar.show()
        else: self.widgetsToolBar.hide()

    """
    def menuItemPreferences(self):
        dlg = orngDlgs.PreferencesDlg(self, None, "", True)
        dlg.exec_()
        if dlg.result() == QDialog.Accepted:
            self.rebuildSignals()
    """

    def menuItemEditWidgetShortcuts(self):
        dlg = orngDlgs.WidgetShortcutDlg(self)
        if dlg.exec_() == QDialog.Accepted:
            self.widgetShortcuts = dict([(y, x) for x, y in dlg.invDict.items()])
            shf = file(os.path.join(self.canvasSettingsDir, "shortcuts.txt"), "wt")
            for k, v in self.widgetShortcuts.items():
                shf.write("%s: %s\n" % (k, v.nameKey))

    def menuItemDeleteWidgetSettings(self):
        if QMessageBox.warning(self,'Orange Canvas','If you want to delete widget settings press Ok, otherwise press Cancel.\nFor the deletion to be complete there cannot be any widgets on your schemas.\nIf there are, close schemas first.',QMessageBox.Ok | QMessageBox.Default, QMessageBox.Cancel | QMessageBox.Escape) == QMessageBox.Ok:
            if os.path.exists(self.widgetSettingsDir):
                for f in os.listdir(self.widgetSettingsDir):
                    if os.path.splitext(f)[1].lower() == ".ini":
                        os.remove(os.path.join(self.widgetSettingsDir, f))

    def menuCloseAll(self):
        wins = self.workspace.getDocumentList()
        for win in wins:
            win.close()
        wins = self.workspace.getDocumentList()

    def menuMinimizeAll(self):
        wins = self.workspace.windowList()
        for win in wins:
            win.showMinimized()

    def menuRestoreAll(self):
        wins = self.workspace.windowList()
        for win in wins:
            win.showNormal()

    def menuOpenLocalOrangeHelp(self):
        import webbrowser
        webbrowser.open("file:///" + os.path.join(self.orangeDir, "doc/reference/default.htm"))

    def menuOpenLocalCanvasHelp(self):
        import webbrowser
        webbrowser.open(os.path.join(self.orangeDir, "doc/canvas/default.htm"))

    def menuOpenOnlineOrangeHelp(self):
        import webbrowser
        webbrowser.open("http://www.ailab.si/orange")

    def menuOpenOnlineCanvasHelp(self):
        import webbrowser
        #webbrowser.open("http://www.ailab.si/orange/orangeCanvas") # to be added on the web
        webbrowser.open("http://www.ailab.si/orange")

    def menuCheckForUpdates(self):
        import updateOrange
        self.updateDlg = updateOrange.updateOrangeDlg(None, "", Qt.WDestructiveClose)

    def menuHelpAbout(self):
        pass

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
            return [symbName, str(1), "blue"]

#    def focusDocument(self, w):
#        pass
#        #if w: w.setFocus()


## to see the signals you have to call: self.output.catchException(0); self.output.catchOutput(0)
## and run orngCanvas.pyw from command line using "python.exe orngCanvas.pyw" 
#    def event(self, e):
#        eventDict = dict([(0, "None"), (130, "AccessibilityDescription"), (119, "AccessibilityHelp"), (86, "AccessibilityPrepare"), (114, "ActionAdded"), (113, "ActionChanged"), (115, "ActionRemoved"), (99, "ActivationChange"), (121, "ApplicationActivated"), (122, "ApplicationDeactivated"), (36, "ApplicationFontChange"), (37, "ApplicationLayoutDirectionChange"), (38, "ApplicationPaletteChange"), (35, "ApplicationWindowIconChange"), (68, "ChildAdded"), (69, "ChildPolished"), (71, "ChildRemoved"), (40, "Clipboard"), (19, "Close"), (82, "ContextMenu"), (52, "DeferredDelete"), (60, "DragEnter"), (62, "DragLeave"), (61, "DragMove"), (63, "Drop"), (98, "EnabledChange"), (10, "Enter"), (150, "EnterEditFocus"), (124, "EnterWhatsThisMode"), (116, "FileOpen"), (8, "FocusIn"), (9, "FocusOut"), (97, "FontChange"), (159, "GraphicsSceneContextMenu"), (164, "GraphicsSceneDragEnter"), (166, "GraphicsSceneDragLeave"), (165, "GraphicsSceneDragMove"), (167, "GraphicsSceneDrop"), (163, "GraphicsSceneHelp"), (160, "GraphicsSceneHoverEnter"), (162, "GraphicsSceneHoverLeave"), (161, "GraphicsSceneHoverMove"), (158, "GraphicsSceneMouseDoubleClick"), (155, "GraphicsSceneMouseMove"), (156, "GraphicsSceneMousePress"), (157, "GraphicsSceneMouseRelease"), (168, "GraphicsSceneWheel"), (18, "Hide"), (27, "HideToParent"), (127, "HoverEnter"), (128, "HoverLeave"), (129, "HoverMove"), (96, "IconDrag"), (101, "IconTextChange"), (83, "InputMethod"), (6, "KeyPress"), (7, "KeyRelease"), (89, "LanguageChange"), (90, "LayoutDirectionChange"), (76, "LayoutRequest"), (11, "Leave"), (151, "LeaveEditFocus"), (125, "LeaveWhatsThisMode"), (88, "LocaleChange"), (153, "MenubarUpdated"), (43, "MetaCall"), (102, "ModifiedChange"), (4, "MouseButtonDblClick"), (2, "MouseButtonPress"), (3, "MouseButtonRelease"), (5, "MouseMove"), (109, "MouseTrackingChange"), (13, "Move"), (12, "Paint"), (39, "PaletteChange"), (131, "ParentAboutToChange"), (21, "ParentChange"), (75, "Polish"), (74, "PolishRequest"), (123, "QueryWhatsThis"), (14, "Resize"), (117, "Shortcut"), (51, "ShortcutOverride"), (17, "Show"), (26, "ShowToParent"), (50, "SockAct"), (112, "StatusTip"), (100, "StyleChange"), (87, "TabletMove"), (92, "TabletPress"), (93, "TabletRelease"), (171, "TabletEnterProximity"), (172, "TabletLeaveProximity"), (1, "Timer"), (120, "ToolBarChange"), (110, "ToolTip"), (78, "UpdateLater"), (77, "UpdateRequest"), (111, "WhatsThis"), (118, "WhatsThisClicked"), (31, "Wheel"), (132, "WinEventAct"), (24, "WindowActivate"), (103, "WindowBlocked"), (25, "WindowDeactivate"), (34, "WindowIconChange"), (105, "WindowStateChange"), (33, "WindowTitleChange"), (104, "WindowUnblocked"), (126, "ZOrderChange"), (169, "KeyboardLayoutChange"), (170, "DynamicPropertyChange")])
#        if eventDict.has_key(e.type()):
#            print str(self.windowTitle()), eventDict[e.type()]
#        return QMainWindow.event(self, e)


    def menuItemCanvasOptions(self):
        dlg = orngDlgs.CanvasOptionsDlg(self, None)

        # set general options settings
        dlg.snapToGridCB.setChecked(self.settings["snapToGrid"])
        #dlg.useLargeIconsCB.setChecked(self.useLargeIcons)
        dlg.writeLogFileCB.setChecked(self.settings["writeLogFile"])
        dlg.dontAskBeforeCloseCB.setChecked(self.settings["dontAskBeforeClose"])
        #dlg.autoSaveSchemasOnCloseCB.setChecked(self.settings["autoSaveSchemasOnClose"])
        dlg.saveWidgetsPositionCB.setChecked(self.settings["saveWidgetsPosition"])
        dlg.useContextsCB.setChecked(self.settings["useContexts"])
##        dlg.autoLoadSchemasOnStartCB.setChecked(self.settings["autoLoadSchemasOnStart"])
        dlg.showSignalNamesCB.setChecked(self.settings["showSignalNames"])

        dlg.heightEdit.setText(str(self.settings.get("canvasHeight", 600)))
        dlg.widthEdit.setText(str(self.settings.get("canvasWidth", 700)))

        items = [str(n) for n in QStyleFactory.keys()]
        ind = items.index(self.settings.get("style", "WindowsXP"))
        dlg.stylesCombo.setCurrentIndex(ind)
        dlg.stylesPalette.setChecked(self.settings["useDefaultPalette"])

        # set current exception settings
        dlg.focusOnCatchExceptionCB.setChecked(self.settings["focusOnCatchException"])
        dlg.printExceptionInStatusBarCB.setChecked(self.settings["printExceptionInStatusBar"])
        dlg.focusOnCatchOutputCB.setChecked(self.settings["focusOnCatchOutput"])
        dlg.printOutputInStatusBarCB.setChecked(self.settings["printOutputInStatusBar"])
        dlg.verbosityCombo.setCurrentIndex(self.settings["outputVerbosity"])
        dlg.ocShow.setChecked(self.settings["ocShow"])
        dlg.ocInfo.setChecked(self.settings["ocInfo"])
        dlg.ocWarning.setChecked(self.settings["ocWarning"])
        dlg.ocError.setChecked(self.settings["ocError"])
        dlg.owShow.setChecked(self.settings["owShow"])
        dlg.owInfo.setChecked(self.settings["owInfo"])
        dlg.owWarning.setChecked(self.settings["owWarning"])
        dlg.owError.setChecked(self.settings["owError"])

        # for backward compatibility we have to transform a list of strings into tuples
        oldTabList = [(name, show) for (name, show, inst) in self.tabs.tabs]

        # fill categories tab list
        for i in range(len(oldTabList)):
            dlg.tabOrderList.addItem(oldTabList[i][0])
            dlg.tabOrderList.item(i).setCheckState(oldTabList[i][1] and Qt.Checked or Qt.Unchecked)

        if dlg.exec_() == QDialog.Accepted:
            h = int(str(dlg.heightEdit.text()));
            w = int(str(dlg.widthEdit.text()))
            self.settings["style"] = str(dlg.stylesCombo.currentText())
            self.settings["useDefaultPalette"] = dlg.stylesPalette.isChecked()
            self.updateStyle()

            # save general settings
            if self.settings["snapToGrid"] != dlg.snapToGridCB.isChecked():
                self.settings["snapToGrid"] = dlg.snapToGridCB.isChecked()
                self.updateSnapToGrid()

            # save exceptions settings
            #self.settings["catchException"] = dlg.catchExceptionCB.isChecked()
            #self.settings["catchOutput"] = dlg.catchOutputCB.isChecked()
            self.settings["printExceptionInStatusBar"] = dlg.printExceptionInStatusBarCB.isChecked()
            self.settings["focusOnCatchException"] = dlg.focusOnCatchExceptionCB.isChecked()
            self.settings["focusOnCatchOutput"] = dlg.focusOnCatchOutputCB.isChecked()
            self.settings["printOutputInStatusBar"] = dlg.printOutputInStatusBarCB.isChecked()
            self.settings["outputVerbosity"] = dlg.verbosityCombo.currentIndex()
            self.settings["ocShow"] = dlg.ocShow.isChecked()
            self.settings["ocInfo"] = dlg.ocInfo.isChecked()
            self.settings["ocWarning"] = dlg.ocWarning.isChecked()
            self.settings["ocError"] = dlg.ocError.isChecked()
            self.settings["owShow"] = dlg.owShow.isChecked()
            self.settings["owInfo"] = dlg.owInfo.isChecked()
            self.settings["owWarning"] = dlg.owWarning.isChecked()
            self.settings["owError"] = dlg.owError.isChecked()

            self.settings["writeLogFile"] = dlg.writeLogFileCB.isChecked()
            self.settings["canvasHeight"] = int(str(dlg.heightEdit.text()))
            self.settings["canvasWidth"] =  int(str(dlg.widthEdit.text()))
            self.settings["showSignalNames"] = dlg.showSignalNamesCB.isChecked()
            self.settings["dontAskBeforeClose"] = dlg.dontAskBeforeCloseCB.isChecked()
            #self.settings["autoSaveSchemasOnClose"] = dlg.autoSaveSchemasOnCloseCB.isChecked()
            self.settings["saveWidgetsPosition"] = dlg.saveWidgetsPositionCB.isChecked()
            self.settings["useContexts"] = dlg.useContextsCB.isChecked()
##            self.settings["autoLoadSchemasOnStart"] = dlg.autoLoadSchemasOnStartCB.isChecked()

            self.settings["widgetSelectedColor"] = (dlg.selectedWidgetIcon.color.red(), dlg.selectedWidgetIcon.color.green(), dlg.selectedWidgetIcon.color.blue())
            self.settings["widgetActiveColor"]   = (dlg.activeWidgetIcon.color.red(), dlg.activeWidgetIcon.color.green(), dlg.activeWidgetIcon.color.blue())
            self.settings["lineColor"]           = (dlg.lineIcon.color.red(), dlg.lineIcon.color.green(), dlg.lineIcon.color.blue())

            self.widgetSelectedColor = dlg.selectedWidgetIcon.color
            self.widgetActiveColor   = dlg.activeWidgetIcon.color
            self.lineColor           = dlg.lineIcon.color

            # update settings in widgets in current documents
            for win in self.workspace.getDocumentList():
                for widget in win.widgets:
                    widget.instance._useContexts = self.settings["useContexts"]
                    widget.instance._owInfo      = self.settings["owInfo"]
                    widget.instance._owWarning   = self.settings["owWarning"]
                    widget.instance._owError     = self.settings["owError"]
                    widget.instance._owShowStatus= self.settings["owShow"]
                    widget.instance.updateStatusBarState()
                    widget.updateWidgetState()

            for win in self.workspace.getDocumentList():
                win.canvasView.repaint()

            # update tooltips for lines in all documents
            show = dlg.showSignalNamesCB.isChecked()
            for doc in self.workspace.getDocumentList():
                for line in doc.lines:
                    line.showSignalNames = show
                    line.updateTooltip()

            #self.output.catchException(self.settings["catchException"])
            #self.output.catchOutput(self.settings["catchOutput"])
            self.output.printExceptionInStatusBar(self.settings["printExceptionInStatusBar"])
            self.output.printOutputInStatusBar(self.settings["printOutputInStatusBar"])
            self.output.setFocusOnException(self.settings["focusOnCatchException"])
            self.output.setFocusOnOutput(self.settings["focusOnCatchOutput"])
            self.output.setWriteLogFile(self.settings["writeLogFile"])
            self.output.setVerbosity(self.settings["outputVerbosity"])

            for toRemove in dlg.removeTabs:
                orngRegistry.addWidgetCategory(toRemove, "", False)

            # save tab order settings
            newTabList = [(str(dlg.tabOrderList.item(i).text()), dlg.tabOrderList.item(i).checkState()) for i in range(dlg.tabOrderList.count())]
            if newTabList != oldTabList:
                self.settings["WidgetTabs"] = newTabList
                self.createWidgetsToolbar()

    def updateStyle(self):
        QApplication.setStyle(QStyleFactory.create(self.settings["style"]))
        qApp.setStyleSheet(" QDialogButtonBox { button-layout: 0; }")       # we want buttons to go in the "windows" direction (Yes, No, Cancel)
        if self.settings["useDefaultPalette"]:
            QApplication.setPalette(qApp.style().standardPalette())
        else:
            QApplication.setPalette(self.originalPalette)


    def setStatusBarEvent(self, text):
        if text == "" or text == None:
            self.statusBar.showMessage("")
            return
        elif text == "\n": return
        text = str(text)
        text = text.replace("<nobr>", ""); text = text.replace("</nobr>", "")
        text = text.replace("<b>", ""); text = text.replace("</b>", "")
        text = text.replace("<i>", ""); text = text.replace("</i>", "")
        text = text.replace("<br>", ""); text = text.replace("&nbsp", "")
        self.statusBar.showMessage("Last event: " + str(text), 5000)

    #######################
    # SETTINGS
    #######################

    # Loads settings from the widget's .ini file
    def loadSettings(self):
        filename = os.path.join(self.canvasSettingsDir, "orngCanvas.ini")
        if os.path.exists(filename):
            file = open(filename, "rb")
            self.settings = cPickle.load(file)
            file.close()
        else:
            self.settings = {}

        self.settings.setdefault("widgetListType", 2)
        self.settings.setdefault("iconSize", "40 x 40")
        self.settings.setdefault("toolboxWidth", 200)
        self.settings.setdefault("snapToGrid", 1)
        self.settings.setdefault("writeLogFile", 1)
        self.settings.setdefault("dontAskBeforeClose", 0)
        #self.settings.setdefault("autoSaveSchemasOnClose", 0)
        self.settings.setdefault("saveWidgetsPosition", 0)
##        self.settings.setdefault("autoLoadSchemasOnStart", 0)

        self.settings.setdefault("widgetSelectedColor", (0, 255, 0))
        self.settings.setdefault("widgetActiveColor", (0,0,255))
        self.settings.setdefault("lineColor", (0,255,0))

        #if not self.settings.has_key("catchException"): self.settings["catchException"] = 1
        #if not self.settings.has_key("catchOutput"): self.settings["catchOutput"] = 1

        self.settings.setdefault("saveSchemaDir", self.canvasSettingsDir)
        self.settings.setdefault("saveApplicationDir", self.canvasSettingsDir)
        self.settings.setdefault("showSignalNames", 1)
        self.settings.setdefault("useContexts", 1)

        self.settings.setdefault("canvasWidth", 700)
        self.settings.setdefault("canvasHeight", 600)

        if not self.settings.has_key("style"):
            items = [str(n) for n in QStyleFactory.keys()]
            lowerItems = [str(n).lower() for n in QStyleFactory.keys()]
            currStyle = str(qApp.style().objectName()).lower()
            self.settings.setdefault("style", items[lowerItems.index(currStyle)])
        self.settings.setdefault("useDefaultPalette", 1)


        self.settings.setdefault("focusOnCatchException", 1)
        self.settings.setdefault("focusOnCatchOutput" , 0)
        self.settings.setdefault("printOutputInStatusBar", 1)
        self.settings.setdefault("printExceptionInStatusBar", 1)
        self.settings.setdefault("outputVerbosity", 0)
        self.settings.setdefault("ocShow", 1)
        self.settings.setdefault("owShow", 0)
        self.settings.setdefault("ocInfo", 1)
        self.settings.setdefault("owInfo", 1)
        self.settings.setdefault("ocWarning", 1)
        self.settings.setdefault("owWarning", 1)
        self.settings.setdefault("ocError", 1)
        self.settings.setdefault("owError", 1)


    # Saves settings to this widget's .ini file
    def saveSettings(self):
        filename = os.path.join(self.canvasSettingsDir, "orngCanvas.ini")
        file=open(filename, "wb")
        cPickle.dump(self.settings, file)
        file.close()

    #######################
    # EVENTS
    #######################
    def closeEvent(self, ce):
        totalDocs = len(self.workspace.getDocumentList())
        closedDocs = 0
        for win in self.workspace.getDocumentList():
            closed = win.close()
            if closed: closedDocs += 1

        # save the current width of the toolbox, if we are using it
        if isinstance(self.widgetsToolBar, orngTabs.WidgetToolBox):
            self.settings["toolboxWidth"] = self.widgetsToolBar.toolbox.width()

        self.saveSettings()
        if closedDocs == totalDocs:
            self.canvasIsClosing = 1        # output window (and possibly report window also) will check this variable before it will close the window
            self.output.logFile.close()
            ce.accept()
        else:
            ce.ignore()

    def enableSave(self, enable):
        self.toolSave.setEnabled(bool(enable))
        self.menuSaveID.setEnabled(bool(enable))
        self.menuSaveAsID.setEnabled(bool(enable))

#    def resizeEvent(self, e):
#        self.tabs.widget(self.tabs.currentIndex()).resizeEvent(e)       # show/hide left and right buttons
#        #self.tabs.widget(self.tabs.currentIndex()).repaint()



class MyStatusBar(QStatusBar):
    def __init__(self, parent):
        QStatusBar.__init__(self, parent)
        self.parentWidget = parent

    def mouseDoubleClickEvent(self, ev):
        self.parentWidget.menuItemShowOutputWindow()

class WidgetMdiArea(QMdiArea):
    def __init__(self, parent):
        self.off = 30
        QMdiArea.__init__(self, parent)

    def getDocumentList(self):
        list = self.subWindowList()
        for item in list:
            if isinstance(item, orngOutput.OutputWindow):
                list.remove(item)
        return list

    def cascadeSubWindows(self):
        list = self.subWindowList()
        outputWin = None
        for item in list:
            if isinstance(item, orngOutput.OutputWindow):
                outputWin = item
        if outputWin:
            list.remove(outputWin)

        # move schemas
        pos = 0
        for item in list:
            item.move(pos,pos)
            item.resize(self.width()-pos, self.height()-pos)
            pos += self.off

        # move output win
        if outputWin:
            outputWin.move(pos,pos)
            outputWin.resize(self.width()-pos, self.height()-pos)

        if len(list) > 0:
            self.setActiveSubWindow(list[0])


def main(argv = None):
    if argv == None:
        argv = sys.argv

    app = QApplication(sys.argv)
    dlg = OrangeCanvasDlg(app)
    dlg.show()
    for arg in sys.argv[1:]:
        if arg == "-reload":
            dlg.menuItemOpenLastSchema()
    app.exec_()
    app.closeAllWindows()

if __name__ == "__main__":
    sys.exit(main())
