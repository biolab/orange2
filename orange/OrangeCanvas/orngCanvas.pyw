    # Author: Gregor Leban (gregor.leban@fri.uni-lj.si)
# Description:
#    main file, that creates the MDI environment

from qt import *
import sys, os, cPickle, user
import orngTabs, orngDoc, orngDlgs, orngOutput, orngRegistry

class OrangeCanvasDlg(QMainWindow):
    def __init__(self,*args):
        apply(QMainWindow.__init__,(self,) + args)
        self.ctrlPressed = 0    # we have to save keystate, so that orngView can access information about keystate
        self.debugMode = 1        # print extra output for debuging
        if (int(qVersion()[0]) >= 3):
            self.setCaption("Orange Canvas")
        else:
            self.setCaption("Qt Orange Canvas")
        self.windows = []    # list of id for windows in Window menu
        self.windowsDict = {}    # dict. with id:menuitem for windows in Window menu

        self.__dict__.update(orngRegistry.directoryNames)

        #self.setFocusPolicy(QWidget.StrongFocus)

##        import OWReport
##        OWReport.IEFeeder(self.reportsDir)

        self.defaultPic = os.path.join(self.picsDir, "Unknown.png")

        canvasIconName = os.path.join(self.canvasDir, "icons/CanvasIcon.png")
        if os.path.exists(canvasIconName):
            self.setIcon(QPixmap(canvasIconName))

        if not self.orangeDir in sys.path:
            sys.path.append(self.orangeDir)

        if not self.widgetDir in sys.path:
            sys.path.append(self.widgetDir)

        # create error and warning icons
        informationIconName = os.path.join(self.canvasDir, "icons/triangle-blue.png")
        warningIconName = os.path.join(self.canvasDir, "icons/triangle-orange.png")
        errorIconName = os.path.join(self.canvasDir, "icons/triangle-red.png")
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

        self.workspace = WidgetWorkspace(self)
        #self.workspace.setBackgroundColor(QColor(255,255,255))
        self.setCentralWidget(self.workspace)
        self.statusBar = MyStatusBar(self)
        self.connect(self.workspace, SIGNAL("windowActivated(QWidget*)"), self.focusDocument)

        self.settings = {}
        self.widgetInfo = {} # this is a dictionary with items: category-widget_name : {info about widget (inList, outList, description...}

        self.rebuildSignals()    # coloring of signals - unused!
        self.useLargeIcons = False
        self.snapToGrid = True
        self.menuSaveSettingsID = -1
        self.menuSaveSettings = 1

        self.loadSettings()

        self.useLargeIcons = self.settings["useLargeIcons"]
        self.snapToGrid = self.settings["snapToGrid"]

        self.widgetSelectedColor = QColor(self.settings["widgetSelectedColor"][0], self.settings["widgetSelectedColor"][1], self.settings["widgetSelectedColor"][2])
        self.widgetActiveColor   = QColor(self.settings["widgetActiveColor"][0], self.settings["widgetActiveColor"][1], self.settings["widgetActiveColor"][2])
        self.lineColor           = QColor(self.settings["lineColor"][0], self.settings["lineColor"][1], self.settings["lineColor"][2])

        self.toolbar = QToolBar(self, 'Toolbar')
        self.widgetsToolBar = QToolBar( self, 'Widgets')

        self.iDocIndex = 1

        canvasPicsDir  = os.path.join(self.canvasDir, "icons")
        self.file_new  = os.path.join(canvasPicsDir, "doc.png")
        self.outputPix = os.path.join(canvasPicsDir, "output.png")
        self.file_open = os.path.join(canvasPicsDir, "open.png")
        self.file_save = os.path.join(canvasPicsDir, "save.png")
        self.file_print= os.path.join(canvasPicsDir, "print.png")
        self.file_exit = os.path.join(canvasPicsDir, "exit.png")
        self.move_left = os.path.join(canvasPicsDir, "moveleft.png")
        self.move_right= os.path.join(canvasPicsDir, "moveright.png")

        self.initMenu()

        self.toolNew  = QToolButton(QIconSet(QPixmap(self.file_new)), "New schema" , QString.null, self.menuItemNewSchema, self.toolbar, 'new schema')
        #self.toolNew.setUsesTextLabel (True)
        self.toolOpen = QToolButton(QIconSet(QPixmap(self.file_open)), "Open schema" , QString.null, self.menuItemOpen , self.toolbar, 'open schema')
        self.toolSave = QToolButton(QIconSet(QPixmap(self.file_save)), "Save schema" ,QString.null, self.menuItemSave, self.toolbar, 'save schema')
        self.toolbar.addSeparator()
        toolPrint = QToolButton(QIconSet(QPixmap(self.file_print)), "Print" ,QString.null, self.menuItemPrinter, self.toolbar, 'print')

        # qt version compatibility
        if hasattr(self, "addDockWindow"): self.addDockWindow(self.toolbar, "Toolbar", Qt.DockTop, True)
        else:                  self.addToolBar(self.toolbar, "Toolbar", QMainWindow.Top, True)

        # read widgets
        if hasattr(self.widgetsToolBar, "setHorizontalStretchable"): self.widgetsToolBar.setHorizontalStretchable(True)
        else: self.widgetsToolBar.setHorizontallyStretchable(True)
        self.createWidgetsToolbar(not os.path.exists(self.registryFileName))

        orngRegistry.addWidgetDirectories()
        self.readShortcuts()

        # read recent files
        self.recentDocs = []
        self.readRecentFiles()

        width  = self.settings.get("canvasWidth", 700)
        height = self.settings.get("canvasHeight", 600)
        self.resize(width, height)

        # center window in the desktop
        deskH = app.desktop().height()
        deskW = app.desktop().width()
        h = max(0, deskH/2 - height/2)  # if the window is too small, resize the window to desktop size
        w = max(0, deskW/2 - width/2)
        self.move(w,h)

        # apply output settings
        self.output = orngOutput.OutputWindow(self, self.workspace, "", Qt.WDestructiveClose)
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
        for fname in os.listdir(self.outputDir):
            if "TempSchema " in fname:
                tempSchemaNames.append(os.path.join(self.outputDir, fname))
        if tempSchemaNames != [] and QMessageBox.information(self,'Orange Canvas',"Your previous Orange Canvas session was not closed successfully.\nYou can choose to reload your unsaved work or start a new session.\n\nIf you choose 'Reload', the links will be disabled to prevent reoccurence of the crash.\nYou can enable them by clicking Options/Enable all links.", 'Reload','New session', QString.null, 0, 1) == 0:
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

        self.workspace.cascade()

        # show message box if no numpy
        qApp.processEvents()
        try:
            import numpy
        except:
            if QMessageBox.warning(self,'Orange Canvas','Several widgets now use numpy module, \nthat is not yet installed on this computer. \nDo you wish to download it?',QMessageBox.Ok | QMessageBox.Default, QMessageBox.Cancel | QMessageBox.Escape) == QMessageBox.Ok:
                import webbrowser
                webbrowser.open("http://sourceforge.net/project/showfiles.php?group_id=1369&package_id=175103&release_id=468153")


    def createWidgetsToolbar(self, rebuildRegistry):
        self.widgetsToolBar.clear()
        self.tabs = orngTabs.WidgetTabs(self.widgetInfo, self.widgetsToolBar, 'tabs')

        if hasattr(self, "addDockWindow") : self.addDockWindow(self.widgetsToolBar, "Widgets", Qt.DockTop, True)
        else:                               self.addToolBar(self.widgetsToolBar, "Widgets", QMainWindow.Top, True)

        self.tabs.setCanvasDlg(self)

        # if registry doesn't exist yet, we create it
        ## the registry is now build already in the orngRegistry when setting up the directory names
        if rebuildRegistry == 1:
            parse = orngRegistry.WidgetsToXML()
            parse.ParseWidgetRoot(self.widgetDir, self.outputDir)

        # if registry still doesn't exist then something is very wrong...
        if not os.path.exists(self.registryFileName):
            QMessageBox.critical( None, "Orange Canvas", "Unable to locate widget registry. Exiting...", QMessageBox.Ok, QMessageBox.Cancel)
            self.quit()

        if self.settings.has_key("WidgetTabs"):
            widgetTabList = self.settings["WidgetTabs"]
        else:
            widgetTabList = ["Data", "Classify", "Evaluate", "Visualize", "Associate", "Genomics", "Other"]

        for tab in widgetTabList:
            self.tabs.insertWidgetTab(tab)

        # read widget registry file and create tab with buttons
        self.tabs.readInstalledWidgets(self.registryFileName, self.widgetDir, self.picsDir, self.defaultPic, self.useLargeIcons)

        # store order to settings list
        widgetTabList = []
        for tab in self.tabs.tabs:
            widgetTabList.append(str(self.tabs.tabLabel(tab)))
        self.settings["WidgetTabs"] = widgetTabList


    def readShortcuts(self):
        self.widgetShortcuts = {}
        shfn = os.path.join(self.outputDir, "shortcuts.txt")
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
        self.menuRecent = QPopupMenu(self)

        self.menuFile = QPopupMenu( self )
        self.menuFile.insertItem(QIconSet(QPixmap(self.file_new)), "&New",  self.menuItemNewSchema, Qt.CTRL+Qt.Key_N )
        #self.menuFile.insertItem( "New from template",  self.menuItemNewFromTemplate)
        #self.menuFile.insertItem( "New from wizard",  self.menuItemNewWizard)
        self.menuFile.insertItem(QIconSet(QPixmap(self.file_open)), "&Open...", self.menuItemOpen, Qt.CTRL+Qt.Key_O )
        self.menuFile.insertItem(QIconSet(QPixmap(self.file_open)), "&Open and Freeze...", self.menuItemOpenFreeze)
        if os.path.exists(os.path.join(self.outputDir, "_lastSchema.ows")):
            self.menuFile.insertItem("&Reload Last Schema", self.menuItemOpenLastSchema, Qt.CTRL + Qt.Key_R)
        self.menuFile.insertItem( "Close", self.menuItemClose )
        self.menuFile.insertSeparator()
        self.menuSaveID = self.menuFile.insertItem(QIconSet(QPixmap(self.file_save)), "&Save", self.menuItemSave, Qt.CTRL+Qt.Key_S )
        self.menuSaveAsID = self.menuFile.insertItem( "Save As...", self.menuItemSaveAs)
        self.menuFile.insertItem( "Save as Application (Tabs)...", self.menuItemSaveAsAppTabs)
        self.menuFile.insertItem( "Save as Application (Buttons)...", self.menuItemSaveAsAppButtons)
        self.menuFile.insertSeparator()
        self.menuFile.insertItem(QIconSet(QPixmap(self.file_print)), "Print Schema / Save image", self.menuItemPrinter)
        self.menuFile.insertSeparator()
        self.menuFile.insertItem( "&Recent Files", self.menuRecent)
        self.menuFile.insertSeparator()
        #self.menuFile.insertItem( "E&xit",  qApp, SLOT( "quit()" ), Qt.CTRL+Qt.Key_Q )
        self.menuFile.insertItem( "Exit",  self.close, Qt.CTRL+Qt.Key_Q )

##        self.menuEdit = QPopupMenu( self )
##        self.menuEdit.insertItem( "Cu&t",  self.menuItemCut, Qt.CTRL+Qt.Key_X )
##        self.menuEdit.insertItem( "&Copy",  self.menuItemCopy, Qt.CTRL+Qt.Key_C )
##        self.menuEdit.insertItem( "&Paste",  self.menuItemPaste, Qt.CTRL+Qt.Key_V )
##        self.menuFile.insertSeparator()
##        self.menuEdit.insertItem( "Select &All",  self.menuItemSelectAll, Qt.CTRL+Qt.Key_A )

        self.menuOptions = QPopupMenu( self )
        #self.menuOptions.insertItem( "Grid",  self.menuItemGrid )
        #self.menuOptions.insertSeparator()
        #self.menuOptions.insertItem( "Show Grid",  self.menuItemShowGrid)

        self.menuOptions.insertItem( "&Enable All Links",  self.menuItemEnableAll, Qt.CTRL+Qt.Key_E)
        self.menuOptions.insertItem( "&Disable All Links",  self.menuItemDisableAll, Qt.CTRL+Qt.Key_D)
        self.menuOptions.insertItem( "Clear Scheme",  self.menuItemClearWidgets)

        # uncomment this only for debugging
        #self.menuOptions.insertItem("Dump widget variables", self.dumpVariables)

        self.menuOptions.insertSeparator()
        #self.menuOptions.insertItem( "Channel preferences",  self.menuItemPreferences)
        #self.menuOptions.insertSeparator()
        self.menuSaveSettingsID = self.menuOptions.insertItem( "Save Widget Settings On Exit", self.menuItemSaveSettings)
        self.menuOptions.setItemChecked(self.menuSaveSettingsID, 1)
        self.menuOptions.insertItem( "&Rebuild Widget Registry",  self.menuItemRebuildWidgetRegistry)
        self.menuOptions.insertItem( "&Customize Shortcuts",  self.menuItemEditWidgetShortcuts)
        self.menuOptions.insertItem( "&Delete Widget Settings",  self.menuItemDeleteWidgetSettings)
        self.menuOptions.insertSeparator()
        if sys.platform == "darwin":
            self.menuOptions.insertItem("&Preferences...", self.menuItemCanvasOptions)
        else:
            self.menuOptions.insertItem( "Canvas &Options...",  self.menuItemCanvasOptions)

        self.menuWindow = QPopupMenu( self )
        self.menuWindow.insertItem("&Cascade", self.workspace.cascade)
        self.menuWindow.insertItem("&Tile", self.workspace.tile)
        self.menuWindow.insertSeparator()

        self.connect(self.menuWindow, SIGNAL("aboutToShow()"), self.showWindows)

        self.menupopupShowToolbarID = self.menuWindow.insertItem( "Toolbar",  self.menuItemShowToolbar )
        if self.settings.has_key("showToolbar"): self.showToolbar = self.settings["showToolbar"]
        else:                                    self.showToolbar = True
        if not self.showToolbar: self.toolbar.hide()
        self.menuWindow.setItemChecked(self.menupopupShowToolbarID, self.showToolbar)

        self.menupopupShowWidgetToolbarID = self.menuWindow.insertItem( "Widget Toolbar",  self.menuItemShowWidgetToolbar)
        if self.settings.has_key("showWidgetToolbar"): self.showWidgetToolbar = self.settings["showWidgetToolbar"]
        else:                                    self.showWidgetToolbar = True
        if not self.showWidgetToolbar: self.widgetsToolBar.hide()
        self.menuWindow.setItemChecked(self.menupopupShowWidgetToolbarID, self.showWidgetToolbar)

        self.menuWindow.insertSeparator()
        self.menuOutput = QPopupMenu(self)
        self.menuWindow.insertItem( "Output &Window", self.menuOutput)
        self.menuOutput.insertItem("Show Output Window", self.menuItemShowOutputWindow)
        self.menuOutput.insertItem("Clear Output Window", self.menuItemClearOutputWindow)
        self.menuOutput.insertSeparator()
        self.menuOutput.insertItem("Save Output Text...", self.menuItemSaveOutputWindow)
        self.menuWindow.insertSeparator()

        self.menuWindow.insertItem("&Minimize All", self.menuMinimizeAll)
        self.menuWindow.insertItem("Restore All", self.menuRestoreAll)
        self.menuWindow.insertItem("Close All", self.menuCloseAll)
        self.menuWindow.insertSeparator()

        localHelp = 0
        self.menuHelp = QPopupMenu( self )
        if os.path.exists(os.path.join(self.orangeDir, r"doc/reference/default.htm")) or os.path.exists(os.path.join(self.orangeDir, r"doc/canvas/default.htm")):
            if os.path.exists(os.path.join(self.orangeDir, r"doc/reference/default.htm")): self.menuHelp.insertItem("Orange Help", self.menuOpenLocalOrangeHelp)
            if os.path.exists(os.path.join(self.orangeDir, r"doc/canvas/default.htm")): self.menuHelp.insertItem("Orange Canvas Help", self.menuOpenLocalCanvasHelp)

        self.menuHelp.insertItem("Orange Online Help", self.menuOpenOnlineOrangeHelp)
        #self.menuHelp.insertItem("Orange Canvas Online Help", self.menuOpenOnlineCanvasHelp)

        if os.path.exists(os.path.join(self.orangeDir, r"updateOrange.py")):
            self.menuHelp.insertSeparator()
            self.menuHelp.insertItem("Check for Updates...", self.menuCheckForUpdates)

        #self.menuHelp.insertSeparator()
        #self.menuHelp.insertItem("About Orange Canvas", self.menuHelpAbout)

        self.menuBar = QMenuBar( self )
        self.menuBar.insertItem( "&File", self.menuFile )
        #self.menuBar.insertItem( "&Edit", self.menuEdit )
        self.menuBar.insertItem( "&Options", self.menuOptions )
        self.menuBar.insertItem("&Window", self.menuWindow)
        self.menuBar.insertItem("&Help", self.menuHelp)

        self.printer = QPrinter()


    def showWindows(self):
        for id in self.windowsDict.keys():
            self.menuWindow.removeItem(id)
        self.windowsDict = {}
        wins = self.workspace.windowList()
        for i in range(len(wins)):
            txt = str(i+1) + ' ' + str(wins[i].caption())
            if i<10: txt = "&" + txt
            id = self.menuWindow.insertItem(txt, self.activateWindowById)
            self.windowsDict[id]=wins[i]

    def activateWindowById(self, id):
        self.windowsDict[id].setFocus()

    def menuItemNewSchema(self, forceNew = 1):
        if not forceNew:
            for doc in self.workspace.getDocumentList():
                if doc.widgets == []: return doc
        win = orngDoc.SchemaDoc(self, self.workspace, "", Qt.WDestructiveClose)
        self.workspace.setDefaultDocPosition(win)
        return win

    def menuItemNewFromTemplate(self):
        return

    def menuItemNewWizard(self):
        return

    def menuItemOpen(self, freeze = 0):
        name = QFileDialog.getOpenFileName( self.settings["saveSchemaDir"], "Orange Widget Scripts (*.ows)", self, "", "Open File")
        if name.isEmpty():
            return
        win = self.menuItemNewSchema(0)
        win.loadDocument(str(name), freeze = freeze)
        self.addToRecentMenu(str(name))

    def menuItemOpenFreeze(self):
        self.menuItemOpen(freeze = 1)

    def menuItemOpenLastSchema(self):
        if os.path.exists(os.path.join(self.outputDir, "_lastSchema.ows")):
            win = self.menuItemNewSchema(0)
            win.loadDocument(os.path.join(self.outputDir, "_lastSchema.ows"), str(win.caption()))

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
            print "Missing file 'OWDlgs.py'. This file should be in OrangeWidgets folder. Unable to print/save image."
            return
        win = self.workspace.activeWindow()
        if not isinstance(win, orngDoc.SchemaDoc):
            return
        sizeDlg = OWDlgs.OWChooseImageSizeDlg(win.canvas)
        sizeDlg.exec_loop()


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
            self.menuRecent.insertItem(shortName, eval("self.menuItemRecent"+str(i+1)))

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
        if self.snapToGrid == True:
            for win in self.workspace.windowList():
                if not isinstance(win, orngDoc.SchemaDoc): continue
                for widget in win.widgets:
                    widget.setCoords(widget.x(), widget.y())
                    widget.moveToGrid()
                    widget.repaintAllLines()
                win.canvas.update()

    def updateUseLargeIcons(self):
        self.createWidgetsToolbar(0)

    def menuItemEnableAll(self):
        win = self.workspace.activeWindow()
        if isinstance(win, orngDoc.SchemaDoc):
            win.enableAllLines()

    def menuItemDisableAll(self):
        win = self.workspace.activeWindow()
        if isinstance(win, orngDoc.SchemaDoc):
            win.disableAllLines()

    def menuItemSaveSettings(self):
        self.menuSaveSettings = not self.menuSaveSettings
        self.menuOptions.setItemChecked(self.menuSaveSettingsID, self.menuSaveSettings)

    def menuItemClearWidgets(self):
        win = self.workspace.activeWindow()
        if win != None:
            win.clear()

    def dumpVariables(self):
        win = self.workspace.activeWindow()
        if isinstance(win, orngDoc.SchemaDoc):
            win.dumpWidgetVariables()

    def menuItemShowOutputWindow(self):
        self.output.show()
        self.output.setFocus()

    def menuItemClearOutputWindow(self):
        self.output.textOutput.setText("")
        self.statusBar.message("")

    def menuItemSaveOutputWindow(self):
        qname = QFileDialog.getSaveFileName( self.settings["saveSchemaDir"] + "/Output.htm", "HTML Document (*.htm)", self, "", "Save Output To File")
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

    """
    def menuItemPreferences(self):
        dlg = orngDlgs.PreferencesDlg(self, None, "", True)
        dlg.exec_loop()
        if dlg.result() == QDialog.Accepted:
            self.rebuildSignals()
    """

    def menuItemRebuildWidgetRegistry(self):
        self.createWidgetsToolbar(True)

    def menuItemEditWidgetShortcuts(self):
        dlg = orngDlgs.WidgetRegistryDlg(self, None, "", True)
        dlg.exec_loop()
        if dlg.result() == QDialog.Accepted:

            self.widgetShortcuts = dict([(y, x) for x, y in dlg.invDict.items()])
            shf = file(os.path.join(self.outputDir, "shortcuts.txt"), "wt")
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

    def focusDocument(self, w):
        if w: w.setFocus()

    def menuItemCanvasOptions(self):
        dlg = orngDlgs.CanvasOptionsDlg(self, None, "", True)

        # set general options settings
        dlg.snapToGridCB.setChecked(self.snapToGrid)
        dlg.useLargeIconsCB.setChecked(self.useLargeIcons)
        dlg.writeLogFileCB.setChecked(self.settings["writeLogFile"])
        dlg.dontAskBeforeCloseCB.setChecked(self.settings["dontAskBeforeClose"])
        #dlg.autoSaveSchemasOnCloseCB.setChecked(self.settings["autoSaveSchemasOnClose"])
        dlg.saveWidgetsPositionCB.setChecked(self.settings["saveWidgetsPosition"])
        dlg.useContextsCB.setChecked(self.settings["useContexts"])
##        dlg.autoLoadSchemasOnStartCB.setChecked(self.settings["autoLoadSchemasOnStart"])
        dlg.showSignalNamesCB.setChecked(self.settings["showSignalNames"])

        dlg.heightEdit.setText(str(self.settings.get("canvasHeight", 600)))
        dlg.widthEdit.setText(str(self.settings.get("canvasWidth", 700)))

        # exception tab
        dlg.focusOnCatchExceptionCB.setChecked(self.settings["focusOnCatchException"])
        dlg.printExceptionInStatusBarCB.setChecked(self.settings["printExceptionInStatusBar"])
        dlg.focusOnCatchOutputCB.setChecked(self.settings["focusOnCatchOutput"])
        dlg.printOutputInStatusBarCB.setChecked(self.settings["printOutputInStatusBar"])
        dlg.verbosityCombo.setCurrentItem(self.settings["outputVerbosity"])
        dlg.ocShow.setChecked(self.settings["ocShow"])
        dlg.ocInfo.setChecked(self.settings["ocInfo"])
        dlg.ocWarning.setChecked(self.settings["ocWarning"])
        dlg.ocError.setChecked(self.settings["ocError"])
        dlg.owShow.setChecked(self.settings["owShow"])
        dlg.owInfo.setChecked(self.settings["owInfo"])
        dlg.owWarning.setChecked(self.settings["owWarning"])
        dlg.owError.setChecked(self.settings["owError"])

        # fill categories tab list
        oldTabList = []
        for tab in self.tabs.tabs:
            dlg.tabs = self.tabs
            dlg.tabOrderList.insertItem(self.tabs.tabLabel(tab))
            oldTabList.append(str(self.tabs.tabLabel(tab)))
        dlg.removeTabs = []

        dlg.exec_loop()
        if dlg.result() == QDialog.Accepted:
            h = int(str(dlg.heightEdit.text()))
            w = int(str(dlg.widthEdit.text()))

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
            #self.settings["catchException"] = dlg.catchExceptionCB.isChecked()
            #self.settings["catchOutput"] = dlg.catchOutputCB.isChecked()
            self.settings["printExceptionInStatusBar"] = dlg.printExceptionInStatusBarCB.isChecked()
            self.settings["focusOnCatchException"] = dlg.focusOnCatchExceptionCB.isChecked()
            self.settings["focusOnCatchOutput"] = dlg.focusOnCatchOutputCB.isChecked()
            self.settings["printOutputInStatusBar"] = dlg.printOutputInStatusBarCB.isChecked()
            self.settings["outputVerbosity"] = dlg.verbosityCombo.currentItem()
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
                    widget.updateSettings()
                    widget.instance._useContexts = self.settings["useContexts"]
                    widget.instance._owInfo      = self.settings["owInfo"]
                    widget.instance._owWarning   = self.settings["owWarning"]
                    widget.instance._owError     = self.settings["owError"]
                    widget.instance._owShowStatus= self.settings["owShow"]
                    widget.instance.updateStatusBarState()

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
            newTabList = []
            for i in range(dlg.tabOrderList.count()):
                newTabList.append(str(dlg.tabOrderList.text(i)))
            if newTabList != oldTabList:
                self.settings["WidgetTabs"] = newTabList
                self.createWidgetsToolbar(0)


    def setStatusBarEvent(self, text):
        if text == "" or text == None:
            self.statusBar.message("")
            return
        elif text == "\n": return
        text = str(text)
        text = text.replace("<nobr>", ""); text = text.replace("</nobr>", "")
        text = text.replace("<b>", ""); text = text.replace("</b>", "")
        text = text.replace("<i>", ""); text = text.replace("</i>", "")
        text = text.replace("<br>", ""); text = text.replace("&nbsp", "")
        self.statusBar.message(QString("Last event: " + str(text)))

    #######################
    # SETTINGS
    #######################

    # Loads settings from the widget's .ini file
    def loadSettings(self):
        filename = os.path.join(self.outputDir, "orngCanvas.ini")
        if os.path.exists(filename):
            file = open(filename, "rb")
            self.settings = cPickle.load(file)
            file.close()
        else:
            self.settings = {}

        self.settings.setdefault("useLargeIcons", 0)
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

        if sys.platform == "darwin":
            self.settings.setdefault("saveSchemaDir", user.home)
            self.settings.setdefault("saveApplicationDir", user.home)
        else:
            self.settings.setdefault("saveSchemaDir", self.outputDir)
            self.settings.setdefault("saveApplicationDir", self.outputDir)
        self.settings.setdefault("showSignalNames", 1)
        self.settings.setdefault("useContexts", 1)

        self.settings.setdefault("canvasWidth", 700)
        self.settings.setdefault("canvasHeight", 600)

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
        filename = os.path.join(self.outputDir, "orngCanvas.ini")
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

        self.saveSettings()
        if closedDocs == totalDocs:
            self.output.logFile.close()
            ce.accept()
            QMainWindow.closeEvent(self, ce)


    def keyPressEvent(self, e):
        QMainWindow.keyPressEvent(self,e)
        if e.state() & Qt.ControlButton != 0:
            self.ctrlPressed = 1

    def keyReleaseEvent(self, e):
        QMainWindow.keyReleaseEvent(self,e)
        self.ctrlPressed = 0

    def enableSave(self, enable):
        self.toolSave.setEnabled(enable)
        self.menuFile.setItemEnabled(self.menuSaveID, enable)
        self.menuFile.setItemEnabled(self.menuSaveAsID, enable)

class MyStatusBar(QStatusBar):
    def __init__(self, parent, name = ""):
        QStatusBar.__init__(self, parent, name)
        self.parentWidget = parent

    def mouseDoubleClickEvent(self, ev):
        self.parentWidget.menuItemShowOutputWindow()

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

    def getDocumentList(self):
        list = self.windowList()
        for item in list:
            if isinstance(item, orngOutput.OutputWindow):
                list.remove(item)
        return list

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
dlg = OrangeCanvasDlg(None, "", Qt.WDestructiveClose)
app.setMainWidget(dlg)
dlg.show()
for arg in sys.argv[1:]:
    if arg == "-reload":
        dlg.menuItemOpenLastSchema()
app.exec_loop()
