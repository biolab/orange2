# Author: Gregor Leban (gregor.leban@fri.uni-lj.si)
# Description:
#    main file, that creates the MDI environment
from PyQt4.QtCore import *
from PyQt4.QtGui import *
import sys, os, cPickle, orngRegistry, orngEnviron, orngGui
import orngTabs, orngDoc, orngDlgs, orngOutput, orngHelp, OWReport
import orange, user, orngMisc

class OrangeCanvasDlg(QMainWindow):
    def __init__(self, app, parent = None, flags = 0):
        QMainWindow.__init__(self, parent)
        self.debugMode = 1        # print extra output for debuging
        self.setWindowTitle("Orange Canvas")
        self.windows = []    # list of id for windows in Window menu
        self.recentDocs = []
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
            
        self.settings = {}
        self.menuSaveSettingsID = -1
        self.menuSaveSettings = 1

        self.loadSettings()
        self.widgetSelectedColor = QColor(*self.settings["widgetSelectedColor"])
        self.widgetActiveColor   = QColor(*self.settings["widgetActiveColor"])
        self.lineColor           = QColor(*self.settings["lineColor"])

        if not self.settings.has_key("WidgetTabs") or self.settings["WidgetTabs"] == []:
            self.settings["WidgetTabs"] = [(name, Qt.Checked) for name in ["Data", "Visualize", "Classify", "Regression", "Evaluate", "Unsupervised", "Associate", "Text", "Genomics", "Prototypes"]]
        
        # output window
        self.output = orngOutput.OutputWindow(self)
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
        self.output.hide()

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

        self.setStatusBar(MyStatusBar(self))
                
        self.widgetRegistry = orngRegistry.readCategories()
        self.updateStyle()
        
        # create toolbar
        self.toolbar = self.addToolBar("Toolbar")
        self.toolbar.setOrientation(Qt.Horizontal)
        if not self.settings.get("showToolbar", True): self.toolbar.hide()
        
        # create a schema
        self.schema = orngDoc.SchemaDoc(self)
        self.setCentralWidget(self.schema)
        self.schema.setFocus()

        orngTabs.constructCategoriesPopup(self)

        # create menu
        self.initMenu()

        self.toolbar.addAction(QIcon(self.file_open), "Open schema", self.menuItemOpen)
        self.toolSave = self.toolbar.addAction(QIcon(self.file_save), "Save schema", self.menuItemSave)
        self.toolbar.addSeparator()
        self.toolbar.addAction(QIcon(self.file_print), "Print", self.menuItemPrinter)

        self.toolbar.addSeparator()
        w = QWidget()
        w.setLayout(QHBoxLayout())
        self.widgetOrganizationCombo = orngGui.comboBox(w, label = "Style:", orientation = "horizontal", items = ["Tool box", "Tree view", "Tabs without labels", "Tabs with labels"])
        self.toolbar.addWidget(w)
        self.connect(self.widgetOrganizationCombo, SIGNAL('activated(int)'), self.widgetListTypeChanged)
        try:        # maybe we will someday remove some of the options and the index will be too big
            self.widgetOrganizationCombo.setCurrentIndex(self.settings["widgetListType"])
        except:
            pass

        self.toolbar.addSeparator()

        self.toolbar.addWidget(QLabel("Icon size: ", self))
        self.iconSizeCombo = QComboBox(self)
        self.iconSizeCombo.addItems(self.iconSizeList)
        self.toolbar.addWidget(self.iconSizeCombo)
        self.connect(self.iconSizeCombo, SIGNAL("activated(int)"), self.iconSizeChanged)
        try:
            self.iconSizeCombo.setCurrentIndex(self.iconSizeList.index(self.settings["iconSize"]))
        except:
            self.iconSizeCombo.setCurrentIndex(self.iconSizeList.index("48 x 48"))

        self.addToolBarBreak()
        self.createWidgetsToolbar()
        self.readShortcuts()
        self.readRecentFiles()

        width, height = self.settings.get("canvasWidth", 700), self.settings.get("canvasHeight", 600)
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

        self.helpWindow = orngHelp.HelpWindow(self)
        self.reportWindow = OWReport.ReportWindow()
        
        self.show()

        # did Orange crash the last time we used it? If yes, you will find a temSchema.tmp file
        if os.path.exists(os.path.join(self.canvasSettingsDir, "tempSchema.tmp")):
            mb = QMessageBox('Orange Canvas', "Your previous Orange Canvas session was not closed successfully.\nYou can choose to reload your unsaved work or start a new session.\n\nIf you choose 'Reload', the links will be disabled to prevent reoccurence of the crash.\nYou can enable them by clicking Options/Enable all links.", QMessageBox.Information, QMessageBox.Ok | QMessageBox.Default, QMessageBox.Cancel | QMessageBox.Escape, QMessageBox.NoButton)
            mb.setButtonText(QMessageBox.Ok, "Reload")
            mb.setButtonText(QMessageBox.Cancel, "New schema")
            if mb.exec_() == QMessageBox.Ok:
                self.schema.loadDocument(os.path.join(self.canvasSettingsDir, "tempSchema.tmp"), freeze = 1)
        
        
        if self.schema.widgets == [] and len(sys.argv) > 1 and os.path.exists(sys.argv[-1]) and os.path.splitext(sys.argv[-1])[1].lower() == ".ows":
            self.schema.loadDocument(sys.argv[-1])
        
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
            self.tabs = self.widgetsToolBar = orngTabs.WidgetToolBox(self, self.widgetRegistry)
            self.addDockWidget(Qt.LeftDockWidgetArea, self.widgetsToolBar)
        elif self.settings["widgetListType"] == 1:
            self.tabs = self.widgetsToolBar = orngTabs.WidgetTree(self, self.widgetRegistry)
            self.addDockWidget(Qt.LeftDockWidgetArea, self.widgetsToolBar)
        else:
            self.widgetsToolBar = self.addToolBar("Widgets")
            self.insertToolBarBreak(self.widgetsToolBar)
            self.tabs = orngTabs.WidgetTabs(self, self.widgetRegistry, self.widgetsToolBar)
            self.widgetsToolBar.addWidget(self.tabs)

        # find widgets and create tab with buttons
        self.settings["WidgetTabs"] = self.tabs.createWidgetTabs(self.settings["WidgetTabs"], self.widgetRegistry, self.widgetDir, self.picsDir, self.defaultPic)
        if not self.settings.get("showWidgetToolbar", True): 
            self.widgetsToolBar.hide()



    def readShortcuts(self):
        self.widgetShortcuts = {}
        shfn = os.path.join(self.canvasSettingsDir, "shortcuts.txt")
        if os.path.exists(shfn):
            for t in file(shfn).readlines():
                key, info = [x.strip() for x in t.split(":")]
                if len(info)== 0: continue
                if info[0] == "(" and info[-1] == ")":
                    cat, widgetName = eval(info)            # new style of shortcuts are of form F: ("Data", "File")
                else:
                    cat, widgetName = info.split(" - ")   # old style of shortcuts are of form F: Data - File
                if self.widgetRegistry.has_key(cat) and self.widgetRegistry[cat].has_key(widgetName):
                    self.widgetShortcuts[key] = self.widgetRegistry[cat][widgetName]


    def initMenu(self):
        self.menuRecent = QMenu("Recent Schemas", self)

        self.menuFile = QMenu("&File", self)
        self.menuFile.addAction( "New Scheme",  self.menuItemNewScheme)
        self.menuFile.addAction(QIcon(self.file_open), "&Open...", self.menuItemOpen, Qt.CTRL+Qt.Key_O )
        self.menuFile.addAction(QIcon(self.file_open), "&Open and Freeze...", self.menuItemOpenFreeze)
        if os.path.exists(os.path.join(self.canvasSettingsDir, "lastSchema.tmp")):
            self.menuFile.addAction("Reload Last Schema", self.menuItemOpenLastSchema, Qt.CTRL+Qt.Key_R)
        #self.menuFile.addAction( "&Clear", self.menuItemClear)
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
        self.menuFile.addAction( "E&xit",  self.close, Qt.CTRL+Qt.Key_Q )

        self.menuOptions = QMenu("&Options", self)
        self.menuOptions.addAction( "Enable All Links",  self.menuItemEnableAll, Qt.CTRL+Qt.Key_E)
        self.menuOptions.addAction( "Disable All Links",  self.menuItemDisableAll, Qt.CTRL+Qt.Key_D)
        self.menuOptions.addSeparator()
        self.menuOptions.addAction("Show Output Window", self.menuItemShowOutputWindow)
        self.menuOptions.addAction("Clear Output Window", self.menuItemClearOutputWindow)
        self.menuOptions.addAction("Save Output Text...", self.menuItemSaveOutputWindow)

        # uncomment this only for debugging
        #self.menuOptions.addSeparator()
        #self.menuOptions.addAction("Dump widget variables", self.dumpVariables)

        self.menuOptions.addSeparator()
        #self.menuOptions.addAction( "Channel preferences",  self.menuItemPreferences)
        #self.menuOptions.addSeparator()
        self.menuOptions.addAction( "&Customize Shortcuts",  self.menuItemEditWidgetShortcuts)
        self.menuOptions.addAction( "&Delete Widget Settings",  self.menuItemDeleteWidgetSettings)
        self.menuOptions.addSeparator()
        self.menuOptions.addAction( sys.platform == "darwin" and "&Preferences..." or "Canvas &Options...",  self.menuItemCanvasOptions)

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
        self.widgetPopup.addAction( "Open",  self.schema.canvasView.openActiveWidget)
        self.widgetPopup.addSeparator()
        rename = self.widgetPopup.addAction( "&Rename", self.schema.canvasView.renameActiveWidget, Qt.Key_F2)
        delete = self.widgetPopup.addAction("Remove", self.schema.canvasView.removeActiveWidget, Qt.Key_Delete)
        self.widgetPopup.setEnabled(0)

        self.menuBar = QMenuBar(self)
        self.menuBar.addMenu(self.menuFile)
        self.menuBar.addMenu(self.menuOptions)
        self.menuBar.addMenu(self.widgetPopup)
        self.menuBar.addMenu(self.menuHelp)
        self.setMenuBar(self.menuBar)

        self.printer = QPrinter()

    def menuItemOpen(self):
        name = QFileDialog.getOpenFileName(self, "Open File", self.settings["saveSchemaDir"], "Orange Widget Scripts (*.ows)")
        if name.isEmpty():
            return
        self.schema.clear()
        self.schema.loadDocument(str(name), freeze = 0)
        self.addToRecentMenu(str(name))

    def menuItemOpenFreeze(self):
        name = QFileDialog.getOpenFileName(self, "Open File", self.settings["saveSchemaDir"], "Orange Widget Scripts (*.ows)")
        if name.isEmpty():
            return
        self.schema.clear()
        self.schema.loadDocument(str(name), freeze = 1)
        self.addToRecentMenu(str(name))

    def menuItemOpenLastSchema(self):
        fullName = os.path.join(self.canvasSettingsDir, "lastSchema.tmp")
        if os.path.exists(fullName):
            self.schema.loadDocument(fullName)

    def menuItemSave(self):
        self.schema.saveDocument()
        
    def menuItemSaveAs(self):
        self.schema.saveDocumentAs()

    def menuItemSaveAsAppButtons(self):
        self.schema.saveDocumentAsApp(asTabs = 0)

    def menuItemSaveAsAppTabs(self):
        self.schema.saveDocumentAsApp(asTabs = 1)

    def menuItemPrinter(self):
        try:
            import OWDlgs
            sizeDlg = OWDlgs.OWChooseImageSizeDlg(self.schema.canvas)
            sizeDlg.exec_()
        except:
            print "Missing file 'OWDlgs.py'. This file should be in OrangeWidgets folder. Unable to print/save image."
        

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
            self.schema.clear()
            self.schema.loadDocument(self.settings["RecentFiles"][index-1])
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

    def menuItemSelectAll(self):
        return

    def updateSnapToGrid(self):
        if self.settings["snapToGrid"]:
            for widget in self.schema.widgets:
                widget.setCoords(widget.x(), widget.y())
            self.schema.canvas.update()

    def widgetListTypeChanged(self, ind):
        self.settings["widgetListType"] = ind
        self.settings["showWidgetToolbar"] = self.widgetsToolBar.isVisible()        # we have to save these settings before we create a new toolbar
        self.createWidgetsToolbar()

    def iconSizeChanged(self, ind):
        self.settings["iconSize"] = self.iconSizeList[ind]
        self.createWidgetsToolbar()

    def menuItemEnableAll(self):
        self.schema.enableAllLines()

    def menuItemDisableAll(self):
        self.schema.disableAllLines()

    def menuItemSaveSettings(self):
        self.menuSaveSettings = not self.menuSaveSettings
        self.menuOptions.setItemChecked(self.menuSaveSettingsID, self.menuSaveSettings)

    def menuItemNewScheme(self):
        self.schema.clear()

    def dumpVariables(self):
        self.schema.dumpWidgetVariables()

    def menuItemShowOutputWindow(self):
        self.output.hide()
        self.output.show()
        #self.output.setFocus()

    def menuItemClearOutputWindow(self):
        self.output.textOutput.clear()
        self.statusBar().showMessage("")

    def menuItemSaveOutputWindow(self):
        qname = QFileDialog.getSaveFileName(self, "Save Output To File", self.canvasSettingsDir + "/Output.html", "HTML Document (*.html)")
        if qname.isEmpty(): return
        name = str(qname)

        text = str(self.output.textOutput.toHtml())
        #text = text.replace("</nobr>", "</nobr><br>")

        file = open(name, "wt")
        file.write(text)
        file.close()


    def menuItemShowToolbar(self):
        self.settings["showToolbar"] = not self.settings.get("showToolbar", True)
        if self.settings["showToolbar"]: self.toolbar.show()
        else: self.toolbar.hide()

    def menuItemShowWidgetToolbar(self):
        self.settings["showWidgetToolbar"] = not self.settings.get("showWidgetToolbar", True)
        if self.settings["showWidgetToolbar"]: self.widgetsToolBar.show()
        else: self.widgetsToolBar.hide()


    def menuItemEditWidgetShortcuts(self):
        dlg = orngDlgs.WidgetShortcutDlg(self)
        if dlg.exec_() == QDialog.Accepted:
            self.widgetShortcuts = dict([(y, x) for x, y in dlg.invDict.items()])
            shf = file(os.path.join(self.canvasSettingsDir, "shortcuts.txt"), "wt")
            for k, widgetInfo in self.widgetShortcuts.items():
                shf.write("%s: %s\n" % (k, (widgetInfo.category, widgetInfo.name)))

    def menuItemDeleteWidgetSettings(self):
        if QMessageBox.warning(self,'Orange Canvas','If you want to delete widget settings press Ok, otherwise press Cancel.\nFor the deletion to be complete there cannot be any widgets on your schema.\nIf there are, clear the schema first.',QMessageBox.Ok | QMessageBox.Default, QMessageBox.Cancel | QMessageBox.Escape) == QMessageBox.Ok:
            if os.path.exists(self.widgetSettingsDir):
                for f in os.listdir(self.widgetSettingsDir):
                    if os.path.splitext(f)[1].lower() == ".ini":
                        os.remove(os.path.join(self.widgetSettingsDir, f))

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

        # fill categories tab list
        ind = 0
        for (name, show) in self.settings["WidgetTabs"]:
            if self.widgetRegistry.has_key(name):
                dlg.tabOrderList.addItem(name)
                dlg.tabOrderList.item(ind).setCheckState(show and Qt.Checked or Qt.Unchecked)
                ind+=1

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

            self.settings["widgetSelectedColor"] = dlg.selectedWidgetIcon.color.getRgb()
            self.settings["widgetActiveColor"]   = dlg.activeWidgetIcon.color.getRgb()
            self.settings["lineColor"]           = dlg.lineIcon.color.getRgb()

            self.widgetSelectedColor = dlg.selectedWidgetIcon.color
            self.widgetActiveColor   = dlg.activeWidgetIcon.color
            self.lineColor           = dlg.lineIcon.color

            # update settings in widgets in current documents
            for widget in self.schema.widgets:
                widget.instance._useContexts = self.settings["useContexts"]
                widget.instance._owInfo      = self.settings["owInfo"]
                widget.instance._owWarning   = self.settings["owWarning"]
                widget.instance._owError     = self.settings["owError"]
                widget.instance._owShowStatus= self.settings["owShow"]
                widget.instance.updateStatusBarState()
                widget.updateWidgetState()

            self.schema.canvasView.repaint()

            # update tooltips for lines in all documents
            show = dlg.showSignalNamesCB.isChecked()
            for line in self.schema.lines:
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

            import orngEnviron, orngRegistry
            if dlg.toAdd != []:
                for (name, dir) in dlg.toAdd: 
                    orngEnviron.registerAddOn(name, dir)
            
            if dlg.toRemove != []:
                for (name, dir) in dlg.toRemove:
                    orngEnviron.registerAddOn(name, dir, add = False)
            
            if dlg.toAdd != [] or dlg.toRemove != []:
                self.widgetRegistry = orngRegistry.readCategories()

            # save tab order settings
            newTabList = [(str(dlg.tabOrderList.item(i).text()), int(dlg.tabOrderList.item(i).checkState())) for i in range(dlg.tabOrderList.count())]
            if newTabList != self.settings["WidgetTabs"]:
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
            self.statusBar().showMessage("")
            return
        elif text == "\n": return
        text = str(text)
        text = text.replace("<nobr>", ""); text = text.replace("</nobr>", "")
        text = text.replace("<b>", ""); text = text.replace("</b>", "")
        text = text.replace("<i>", ""); text = text.replace("</i>", "")
        text = text.replace("<br>", ""); text = text.replace("&nbsp", "")
        self.statusBar().showMessage("Last event: " + str(text), 5000)

    # Loads settings from the widget's .ini file
    def loadSettings(self):
        filename = os.path.join(self.canvasSettingsDir, "orngCanvas.ini")
        if os.path.exists(filename):
            file = open(filename, "rb")
            self.settings = cPickle.load(file)
            file.close()
        else:
            self.settings = {}

        self.settings.setdefault("widgetListType", 0)
        self.settings.setdefault("iconSize", "40 x 40")
        self.settings.setdefault("toolboxWidth", 200)
        self.settings.setdefault("snapToGrid", 1)
        self.settings.setdefault("writeLogFile", 1)
        self.settings.setdefault("dontAskBeforeClose", 1)
        #self.settings.setdefault("autoSaveSchemasOnClose", 0)
        self.settings.setdefault("saveWidgetsPosition", 1)
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
        
        self.settings.setdefault("synchronizeHelp", 1)


    # Saves settings to this widget's .ini file
    def saveSettings(self):
        filename = os.path.join(self.canvasSettingsDir, "orngCanvas.ini")
        file=open(filename, "wb")
        if self.settings["widgetListType"] == 1:        # tree view
            self.settings["treeItemsOpenness"] = dict([(key, self.tabs.tabDict[key].isExpanded()) for key in self.tabs.tabDict.keys()])
        cPickle.dump(self.settings, file)
        file.close()

    def closeEvent(self, ce):
        # save the current width of the toolbox, if we are using it
        if isinstance(self.widgetsToolBar, orngTabs.WidgetToolBox):
            self.settings["toolboxWidth"] = self.widgetsToolBar.toolbox.width()

        closed = self.schema.close()
        if closed:
            self.canvasIsClosing = 1        # output window (and possibly report window also) will check this variable before it will close the window
            self.output.logFile.close()
            self.output.hide()
            self.settings["showWidgetToolbar"] = self.widgetsToolBar.isVisible()
            self.settings["showToolbar"] = self.toolbar.isVisible()
            ce.accept()
            
            self.helpWindow.close()
            self.reportWindow.close()
        else:
            ce.ignore()
        
        self.reportWindow.removeTemp()
        self.saveSettings()


    def setCaption(self, caption = ""):
        if caption:
            caption = caption.split(".")[0]
            self.setWindowTitle(caption + " - Orange Canvas")
        else:
            self.setWindowTitle("Orange Canvas")
        
    def getFullWidgetIconName(self, widgetInfo):
        name = widgetInfo.icon
        widgetDir = str(self.widgetRegistry[widgetInfo.category].directory)  #os.path.split(self.getFileName())[0]

        for paths in [(self.picsDir, name), 
                      (self.widgetDir, name), 
                      (name,), 
                      (widgetDir, name), 
                      (widgetDir, "icons", name)]:
            fname = os.path.join(*paths)
            if os.path.exists(fname):
                return fname

        return self.defaultPic


class MyStatusBar(QStatusBar):
    def __init__(self, parent):
        QStatusBar.__init__(self, parent)
        self.parentWidget = parent

    def mouseDoubleClickEvent(self, ev):
        self.parentWidget.menuItemShowOutputWindow()
        
        
class OrangeQApplication(QApplication):
    def __init__(self, *args):
        QApplication.__init__(self, *args)
        
#    def notify(self, receiver, event):
#        eventDict = {0: 'None', 1: 'Timer', 2: 'MouseButtonPress', 3: 'MouseButtonRelease', 4: 'MouseButtonDblClick', 5: 'MouseMove', 6: 'KeyPress', 7: 'KeyRelease', 8: 'FocusIn', 9: 'FocusOut', 10: 'Enter', 11: 'Leave', 12: 'Paint', 13: 'Move', 14: 'Resize', 17: 'Show', 18: 'Hide', 19: 'Close', 21: 'ParentChange', 24: 'WindowActivate', 25: 'WindowDeactivate', 26: 'ShowToParent', 27: 'HideToParent', 31: 'Wheel', 33: 'WindowTitleChange', 34: 'WindowIconChange', 35: 'ApplicationWindowIconChange', 36: 'ApplicationFontChange', 37: 'ApplicationLayoutDirectionChange', 38: 'ApplicationPaletteChange', 39: 'PaletteChange', 40: 'Clipboard', 43: 'MetaCall', 50: 'SockAct', 51: 'ShortcutOverride', 52: 'DeferredDelete', 60: 'DragEnter', 61: 'DragMove', 62: 'DragLeave', 63: 'Drop', 68: 'ChildAdded', 69: 'ChildPolished', 70: 'ChildInserted', 71: 'ChildRemoved', 74: 'PolishRequest', 75: 'Polish', 76: 'LayoutRequest', 77: 'UpdateRequest', 78: 'UpdateLater', 82: 'ContextMenu', 83: 'InputMethod', 86: 'AccessibilityPrepare', 87: 'TabletMove', 88: 'LocaleChange', 89: 'LanguageChange', 90: 'LayoutDirectionChange', 92: 'TabletPress', 93: 'TabletRelease', 94: 'OkRequest', 96: 'IconDrag', 97: 'FontChange', 98: 'EnabledChange', 99: 'ActivationChange', 100: 'StyleChange', 101: 'IconTextChange', 102: 'ModifiedChange', 103: 'WindowBlocked', 104: 'WindowUnblocked', 105: 'WindowStateChange', 109: 'MouseTrackingChange', 110: 'ToolTip', 111: 'WhatsThis', 112: 'StatusTip', 113: 'ActionChanged', 114: 'ActionAdded', 115: 'ActionRemoved', 116: 'FileOpen', 117: 'Shortcut', 118: 'WhatsThisClicked', 119: 'AccessibilityHelp', 120: 'ToolBarChange', 121: 'ApplicationActivate', 122: 'ApplicationDeactivate', 123: 'QueryWhatsThis', 124: 'EnterWhatsThisMode', 125: 'LeaveWhatsThisMode', 126: 'ZOrderChange', 127: 'HoverEnter', 128: 'HoverLeave', 129: 'HoverMove', 130: 'AccessibilityDescription', 131: 'ParentAboutToChange', 132: 'WinEventAct', 150: 'EnterEditFocus', 151: 'LeaveEditFocus', 153: 'MenubarUpdated', 155: 'GraphicsSceneMouseMove', 156: 'GraphicsSceneMousePress', 157: 'GraphicsSceneMouseRelease', 158: 'GraphicsSceneMouseDoubleClick', 159: 'GraphicsSceneContextMenu', 160: 'GraphicsSceneHoverEnter', 161: 'GraphicsSceneHoverMove', 162: 'GraphicsSceneHoverLeave', 163: 'GraphicsSceneHelp', 164: 'GraphicsSceneDragEnter', 165: 'GraphicsSceneDragMove', 166: 'GraphicsSceneDragLeave', 167: 'GraphicsSceneDrop', 168: 'GraphicsSceneWheel', 169: 'KeyboardLayoutChange', 170: 'DynamicPropertyChange', 171: 'TabletEnterProximity', 172: 'TabletLeaveProximity', 173: 'NonClientAreaMouseMove', 174: 'NonClientAreaMouseButtonPress', 175: 'NonClientAreaMouseButtonRelease', 176: 'NonClientAreaMouseButtonDblClick', 177: 'MacSizeChange', 178: 'ContentsRectChange', 181: 'GraphicsSceneResize', 182: 'GraphicsSceneMove', 183: 'CursorChange', 184: 'ToolTipChange', 186: 'GrabMouse', 187: 'UngrabMouse', 188: 'GrabKeyboard', 189: 'UngrabKeyboard'}
#        import time
#        try:
#            if str(receiver.windowTitle()) != "":
#                print time.strftime("%H:%M:%S"), "%15s" % str(receiver.windowTitle()) + ": ",      # print only events for QWidget classes and up
#                if eventDict.has_key(event.type()):
#                    print eventDict[event.type()] 
#                else:
#                    print "unknown event name (" + str(event.type()) + ")"
#        except:
#            pass
#            #print str(receiver.objectName()),
#                
#        return QApplication.notify(self, receiver, event)


def main(argv = None):
    if argv == None:
        argv = sys.argv

    app = OrangeQApplication(sys.argv)
    dlg = OrangeCanvasDlg(app)
    qApp.canvasDlg = dlg
    dlg.show()
    for arg in sys.argv[1:]:
        if arg == "-reload":
            dlg.menuItemOpenLastSchema()
    app.exec_()
    app.closeAllWindows()

if __name__ == "__main__":
    sys.exit(main())
