#
# OWWidget.py
# Orange Widget
# A General Orange Widget, from which all the Orange Widgets are derived
#
from OWTools import *
from OWContexts import *
import sys, time, random, user, os, os.path, cPickle, copy, orngMisc
import orange
from string import *
from orngSignalManager import *

def unisetattr(self, name, value, grandparent):
    if "." in name:
        names = name.split(".")
        lastname = names.pop()
        obj = reduce(lambda o, n: getattr(o, n, None),  names, self)
    else:
        lastname, obj = name, self

    if not obj:
        print "unable to set setting ", name, " to value ", value
    else:
        if hasattr(grandparent, "__setattr__") and isinstance(obj, grandparent):
            grandparent.__setattr__(obj, lastname,  value)
        else:
            obj.__dict__[lastname] = value

    controlledAttributes = getattr(self, "controlledAttributes", None)
    controlCallback = controlledAttributes and controlledAttributes.get(name, None)
    if controlCallback:
        for callback in controlCallback:
            callback(value)
#        controlCallback(value)

    # controlled things (checkboxes...) never have __attributeControllers
    else:
        if hasattr(self, "__attributeControllers"):
            for controller, myself in self.__attributeControllers.keys():
                if getattr(controller, myself, None) != self:
                    del self.__attributeControllers[(controller, myself)]
                    continue

                controlledAttributes = getattr(controller, "controlledAttributes", None)
                if controlledAttributes:
                    fullName = myself + "." + name

                    controlCallback = controlledAttributes.get(fullName, None)
                    if controlCallback:
                        for callback in controlCallback:
                            callback(value)

                    else:
                        lname = fullName + "."
                        dlen = len(lname)
                        for controlled in controlledAttributes.keys():
                            if controlled[:dlen] == lname:
                                self.setControllers(value, controlled[dlen:], controller, fullName)
                                # no break -- can have a.b.c.d and a.e.f.g; needs to set controller for all!


    # if there are any context handlers, call the fastsave to write the value into the context
    if hasattr(self, "contextHandlers"):
        for contextName, contextHandler in self.contextHandlers.items():
            contextHandler.fastSave(self.currentContexts.get(contextName), self, name, value)



##################
# this definitions are needed only to define ExampleTable as subclass of ExampleTableWithClass
class ExampleTable(orange.ExampleTable):
    pass

#class ExampleTableWithClass(ExampleTable):
#    pass

class AttributeList(list):
    pass

class ExampleList(list):
    pass

class OWBaseWidget(QDialog):
    def __init__(self, parent = None, signalManager = None, title="Qt Orange BaseWidget", modal=FALSE, savePosition = False):
        # the "currentContexts" MUST be the first thing assigned to a widget
        self.currentContexts = {}
        self._guiElements = []      # used for automatic widget debugging
        self._useContexts = 1       # do you want to use contexts
        self._owInfo = 1
        self._owWarning = 1
        self._owError = 1

        # do we want to save widget position and restore it on next load
        self.savePosition = savePosition
        if savePosition:
            self.settingsList = getattr(self, "settingsList", []) + ["widgetWidth", "widgetHeight", "widgetXPosition", "widgetYPosition", "widgetShown"]

        self.title = title.replace("&","")

        QDialog.__init__(self, parent, self.title, modal, Qt.WStyle_Customize + Qt.WStyle_NormalBorder + Qt.WStyle_Title + Qt.WStyle_SysMenu + Qt.WStyle_Minimize + Qt.WStyle_Maximize)

        # directories are better defined this way, otherwise .ini files get written in many places
        self.widgetDir = os.path.dirname(__file__) + "/"

        # create output directory for widget settings
        if os.name == "nt":
            self.outputDir = os.path.join(os.path.join(user.home, "Application Data"), "Orange")                  # directory for saving settings and stuff
        else:
            self.outputDir = os.path.join(user.home, "Orange")                  # directory for saving settings and stuff
        if not os.path.exists(self.outputDir): os.mkdir(self.outputDir)
        self.outputDir = os.path.join(self.outputDir, "widgetSettings")
        if not os.path.exists(self.outputDir): os.mkdir(self.outputDir)

        self.loadContextSettings()

        self.captionTitle = title.replace("&","")     # used for widget caption

        # if we want the widget to show the title then the title must start with "Qt"
        if self.captionTitle[:2].upper() != "QT":
            self.captionTitle = "Qt " + self.captionTitle

        # number of control signals, that are currently being processed
        # needed by signalWrapper to know when everything was sent
        self.parent = parent
        self.needProcessing = 0     # used by signalManager
        if not signalManager: self.signalManager = globalSignalManager        # use the global instance of signalManager  - not advised
        else:                 self.signalManager = signalManager              # use given instance of signal manager

        self.inputs = []     # signalName:(dataType, handler, onlySingleConnection)
        self.outputs = []    # signalName: dataType
        self.wrappers =[]    # stored wrappers for widget events
        self.linksIn = {}      # signalName : (dirty, widgetFrom, handler, signalData)
        self.linksOut = {}       # signalName: (signalData, id)
        self.connections = {}   # dictionary where keys are (control, signal) and values are wrapper instances. Used in connect/disconnect
        self.controlledAttributes = ControlledAttributesDict(self)
        self.progressBarHandler = None  # handler for progress bar events
        self.processingHandler = None   # handler for processing events
        self.eventHandler = None
        self.callbackDeposit = []
        self.startTime = time.time()    # used in progressbar

        self.widgetStateHandler = None
        self.widgetState = {"Info":{}, "Warning":{}, "Error":{}}

        #the title
        self.setCaption(self.captionTitle)

        if hasattr(self, "contextHandlers"):
            for contextHandler in self.contextHandlers.values():
                contextHandler.initLocalContext(self)

    # uncomment this when you need to see which events occured
    """
    def event(self, e):
        eventDict = dict([(0, 'None'), (1, 'Timer'), (2, 'MouseButtonPress'), (3, 'MouseButtonRelease'), (4, 'MouseButtonDblClick'), (5, 'MouseMove'), (6, 'KeyPress'), (7, 'KeyRelease'), (8, 'FocusIn'), (9, 'FocusOut'), (10, 'Enter'), (11, 'Leave'), (12, 'Paint'), (13, 'Move'), (14, 'Resize'), (15, 'Create'), (16, 'Destroy'), (17, 'Show'), (18, 'Hide'), (19, 'Close'), (20, 'Quit'), (21, 'Reparent'), (22, 'ShowMinimized'), (23, 'ShowNormal'), (24, 'WindowActivate'), (25, 'WindowDeactivate'), (26, 'ShowToParent'), (27, 'HideToParent'), (28, 'ShowMaximized'), (30, 'Accel'), (31, 'Wheel'), (32, 'AccelAvailable'), (33, 'CaptionChange'), (34, 'IconChange'), (35, 'ParentFontChange'), (36, 'ApplicationFontChange'), (37, 'ParentPaletteChange'), (38, 'ApplicationPaletteChange'), (40, 'Clipboard'), (42, 'Speech'), (50, 'SockAct'), (51, 'AccelOverride'), (60, 'DragEnter'), (61, 'DragMove'), (62, 'DragLeave'), (63, 'Drop'), (64, 'DragResponse'), (70, 'ChildInserted'), (71, 'ChildRemoved'), (72, 'LayoutHint'), (73, 'ShowWindowRequest'), (80, 'ActivateControl'), (81, 'DeactivateControl'), (1000, 'User')])
        print self, eventDict[e.type()]
        return QDialog.event(self, e)
    """

    def setWidgetIcon(self, iconName):
        if os.path.exists(iconName):
            QDialog.setIcon(self, QPixmap(iconName))
        elif os.path.exists(self.widgetDir + iconName):
            QDialog.setIcon(self, QPixmap(self.widgetDir + iconName))
        elif os.path.exists(self.widgetDir + "icons/" + iconName):
            QDialog.setIcon(self, QPixmap(self.widgetDir + "icons/" + iconName))
        elif os.path.exists(self.widgetDir + "icons/Unknown.png"):
            QDialog.setIcon(self, QPixmap(self.widgetDir + "icons/Unknown.png"))

    # ##############################################
    def createAttributeIconDict(self):
        import OWGUI
        return OWGUI.getAttributeIcons()

    def isDataWithClass(self, data, wantedVarType = None):
        self.error([1234, 1235])
        if not data:
            return 0
        if not data.domain.classVar:
            self.error(1234, "A data set with a class attribute is required.")
            return 0
        if wantedVarType and data.domain.classVar.varType != wantedVarType:
            self.error(1235, "Unable to handle %s class." % (data.domain.classVar.varType == orange.VarTypes.Discrete and "discrete" or "continuous"))
            return 0
        return 1

    # call processEvents(), but first remember position and size of widget in case one of the events would be move or resize
    # call this function if needed in __init__ of the widget
    def safeProcessEvents(self):
        keys = ["widgetXPosition", "widgetYPosition", "widgetShown", "widgetWidth", "widgetHeight"]
        vals = [(key, getattr(self, key, None)) for key in keys]
        qApp.processEvents()
        for (key, val) in vals:
            if val != None:
                setattr(self, key, val)


    # this function is called at the end of the widget's __init__ when the widgets is saving its position and size parameters
    def restoreWidgetPosition(self):
        if self.savePosition:
            if getattr(self, "widgetXPosition", None) != None and getattr(self, "widgetYPosition", None) != None:
##                print self.title, "restoring position", self.widgetXPosition, self.widgetYPosition
                self.move(self.widgetXPosition, self.widgetYPosition)
            if getattr(self, "widgetWidth", None) != None and getattr(self, "widgetHeight", None) != None:
                self.resize(self.widgetWidth, self.widgetHeight)

    # this is called in canvas when loading a schema. it opens the widgets that were shown when saving the schema
    def restoreWidgetStatus(self):
        if self.savePosition and getattr(self, "widgetShown", None):
            self.show()

    # when widget is resized, save new width and height into widgetWidth and widgetHeight. some widgets can put this two
    # variables into settings and last widget shape is restored after restart
    def resizeEvent(self, ev):
        QDialog.resizeEvent(self, ev)
        if self.savePosition:
            self.widgetWidth = self.width()
            self.widgetHeight = self.height()
##            print self.title, "saving resize", self.widgetWidth, self.widgetHeight


    # when widget is moved, save new x and y position into widgetXPosition and widgetYPosition. some widgets can put this two
    # variables into settings and last widget position is restored after restart
    def moveEvent(self, ev):
        QDialog.moveEvent(self, ev)
        if self.savePosition:
            self.widgetXPosition = ev.pos().x()
            self.widgetYPosition = ev.pos().y()
##            print self.title, "saving position", self.widgetXPosition, self.widgetYPosition

    # set widget state to hidden
    def hideEvent(self, ev):
        QDialog.hideEvent(self, ev)
        if self.savePosition:
            self.widgetShown = 0

    # set widget state to shown
    def showEvent(self, ev):
        QDialog.showEvent(self, ev)
        if self.savePosition:
            self.widgetShown = 1

    def setCaption(self, caption):
        if self.parent != None and isinstance(self.parent, QTabWidget): self.parent.changeTab(self, caption)
        else: QDialog.setCaption(self, caption)

    def setCaptionTitle(self, caption):
        self.captionTitle = caption     # we have to save caption title in case progressbar will change it
        self.setCaption(caption)

    # put this widget on top of all windows
    def reshow(self):
        self.hide()
        self.show()

    def send(self, signalName, value, id = None):
        if not self.hasOutputName(signalName):
            print "Warning! Signal '%s' is not a valid signal name for the '%s' widget. Please fix the signal name." % (signalName, self.title)

        if self.linksOut.has_key(signalName):
            self.linksOut[signalName][id] = value
        else:
            self.linksOut[signalName] = {id:value}

        self.signalManager.send(self, signalName, value, id)


    def getdeepattr(self, attr, **argkw):
        try:
            return reduce(lambda o, n: getattr(o, n, None),  attr.split("."), self)
        except:
            if argkw.has_key("default"):
                return argkw[default]
            else:
                raise AttributeError, "'%s' has no attribute '%s'" % (self, attr)


    # Set all settings
    # settings - the map with the settings
    def setSettings(self,settings):
        for key in settings:
            self.__setattr__(key, settings[key])
        #self.__dict__.update(settings)

    # Get all settings
    # returns map with all settings
    def getSettings(self):
        settings = {}
        if hasattr(self, "settingsList"):
            for name in self.settingsList:
                try:
                    settings[name] =  self.getdeepattr(name)
                except:
                    #print "Attribute %s not found in %s widget. Remove it from the settings list." % (name, self.title)
                    pass
        return settings


    def getSettingsFile(self, file):
        if file==None:
            if os.path.exists(os.path.join(self.outputDir, self.title + ".ini")):
                file = os.path.join(self.outputDir, self.title + ".ini")
            else:
                return
        if type(file) == str:
            if os.path.exists(file):
                return open(file, "r")
        else:
            return file


    # Loads settings from the widget's .ini file
    def loadSettings(self, file = None):
        file = self.getSettingsFile(file)
        if file:
            try:
                settings = cPickle.load(file)
            except:
                settings = None

            # can't close everything into one big try-except since this would mask all errors in the below code
            if settings:
                if hasattr(self, "settingsList"):
                    self.setSettings(settings)

                contextHandlers = getattr(self, "contextHandlers", {})
                for contextHandler in contextHandlers.values():
                    if not getattr(contextHandler, "globalContexts", False): # don't have it or empty
                        contexts = settings.get(contextHandler.localContextName, False)
                        if contexts != False:
                            contextHandler.globalContexts = contexts
                    else:
                        if contextHandler.syncWithGlobal:
                            setattr(self, contextHandler.localContextName, contextHandler.globalContexts)


    def loadContextSettings(self, file = None):
        if not hasattr(self.__class__, "savedContextSettings"):
            file = self.getSettingsFile(file)
            if file:
                try:
                    settings = cPickle.load(file)
                except:
                    settings = None

                # can't close everything into one big try-except since this would mask all errors in the below code
                if settings:
                    if settings.has_key("savedContextSettings"):
                        self.__class__.savedContextSettings = settings["savedContextSettings"]
                        return

            self.__class__.savedContextSettings = {}


    def saveSettings(self, file = None):
        settings = self.getSettings()

        contextHandlers = getattr(self, "contextHandlers", {})
        for contextHandler in contextHandlers.values():
            contextHandler.mergeBack(self)
            settings[contextHandler.localContextName] = contextHandler.globalContexts

        if settings:
            if file==None:
                file = os.path.join(self.outputDir, self.title + ".ini")
            if type(file) == str:
                file = open(file, "w")
            cPickle.dump(settings, file)

    # Loads settings from string str which is compatible with cPickle
    def loadSettingsStr(self, str):
        if str == None or str == "":
            return

        settings = cPickle.loads(str)
        self.setSettings(settings)

        contextHandlers = getattr(self, "contextHandlers", {})
        for contextHandler in contextHandlers.values():
            if settings.has_key(contextHandler.localContextName):
                setattr(self, contextHandler.localContextName, settings[contextHandler.localContextName])

    # return settings in string format compatible with cPickle
    def saveSettingsStr(self):
        str = ""
        settings = self.getSettings()

        contextHandlers = getattr(self, "contextHandlers", {})
        for contextHandler in contextHandlers.values():
            settings[contextHandler.localContextName] = getattr(self, contextHandler.localContextName)

        return cPickle.dumps(settings)

    # this function is only intended for derived classes to send appropriate signals when all settings are loaded
    def activateLoadedSettings(self):
        pass

    # reimplemented in other widgets
    def setOptions(self):
        pass

    # does widget have a signal with name in inputs
    def hasInputName(self, name):
        for input in self.inputs:
            if name == input[0]: return 1
        return 0

    # does widget have a signal with name in outputs
    def hasOutputName(self, name):
        for output in self.outputs:
            if name == output[0]: return 1
        return 0

    def getInputType(self, signalName):
        for input in self.inputs:
            if input[0] == signalName: return input[1]
        return None

    def getOutputType(self, signalName):
        for output in self.outputs:
            if output[0] == signalName: return output[1]
        return None

    # ########################################################################
    def connect(self, control, signal, method):
        wrapper = SignalWrapper(self, method)
        self.connections[(control, signal)] = wrapper   # save for possible disconnect
        self.wrappers.append(wrapper)
        QDialog.connect(control, signal, wrapper)
        #QWidget.connect(control, signal, method)        # ordinary connection useful for dialogs and windows that don't send signals to other widgets


    def disconnect(self, control, signal, method):
        wrapper = self.connections[(control, signal)]
        QDialog.disconnect(control, signal, wrapper)


    def signalIsOnlySingleConnection(self, signalName):
        for i in self.inputs:
            input = InputSignal(*i)
            if input.name == signalName: return input.single

    def addInputConnection(self, widgetFrom, signalName):
        for i in range(len(self.inputs)):
            if self.inputs[i][0] == signalName:
                handler = self.inputs[i][2]
                break

        existing = []
        if self.linksIn.has_key(signalName):
            existing = self.linksIn[signalName]
            for (dirty, widget, handler, data) in existing:
                if widget == widgetFrom: return             # no need to add new tuple, since one from the same widget already exists
        self.linksIn[signalName] = existing + [(0, widgetFrom, handler, [])]    # (dirty, handler, signalData)
        #if not self.linksIn.has_key(signalName): self.linksIn[signalName] = [(0, widgetFrom, handler, [])]    # (dirty, handler, signalData)

    # delete a link from widgetFrom and this widget with name signalName
    def removeInputConnection(self, widgetFrom, signalName):
        if self.linksIn.has_key(signalName):
            links = self.linksIn[signalName]
            for i in range(len(self.linksIn[signalName])):
                if widgetFrom == self.linksIn[signalName][i][1]:
                    self.linksIn[signalName].remove(self.linksIn[signalName][i])
                    if self.linksIn[signalName] == []:  # if key is empty, delete key value
                        del self.linksIn[signalName]
                    return

    # return widget, that is already connected to this singlelink signal. If this widget exists, the connection will be deleted (since this is only single connection link)
    def removeExistingSingleLink(self, signal):
        for i in self.inputs:
            input = InputSignal(*i)
            if input.name == signal and not input.single: return None

        for signalName in self.linksIn.keys():
            if signalName == signal:
                widget = self.linksIn[signalName][0][1]
                del self.linksIn[signalName]
                return widget

        return None

    # signal manager calls this function when all input signals have updated the data
    def processSignals(self):
        if self.processingHandler: self.processingHandler(self, 1)    # focus on active widget

        # we define only a way to handle signals that have defined a handler function
        for signal in self.inputs:        # we go from the first to the last defined input
            key = signal[0]
            if self.linksIn.has_key(key):
                for i in range(len(self.linksIn[key])):
                    (dirty, widgetFrom, handler, signalData) = self.linksIn[key][i]
                    if not (handler and dirty): continue

                    qApp.setOverrideCursor(QWidget.waitCursor)
                    try:
                        for (value, id, nameFrom) in signalData:
                            if self.signalIsOnlySingleConnection(key):
                                self.printEvent("ProcessSignals: Calling %s with %s" % (handler, value), eventVerbosity = 2)
                                handler(value)
                            else:
                                self.printEvent("ProcessSignals: Calling %s with %s (%s, %s)" % (handler, value, nameFrom, id), eventVerbosity = 2)
                                handler(value, (widgetFrom, nameFrom, id))
                    except:
                        type, val, traceback = sys.exc_info()
                        sys.excepthook(type, val, traceback)  # we pretend that we handled the exception, so that we don't crash other widgets

                    qApp.restoreOverrideCursor()
                    self.linksIn[key][i] = (0, widgetFrom, handler, []) # clear the dirty flag

        if hasattr(self, "handleNewSignals"):
            self.handleNewSignals()

        if self.processingHandler:
            self.processingHandler(self, 0)    # remove focus from this widget
        self.needProcessing = 0

    # set new data from widget widgetFrom for a signal with name signalName
    def updateNewSignalData(self, widgetFrom, signalName, value, id, signalNameFrom):
        if not self.linksIn.has_key(signalName): return
        for i in range(len(self.linksIn[signalName])):
            (dirty, widget, handler, signalData) = self.linksIn[signalName][i]
            if widget == widgetFrom:
                if self.linksIn[signalName][i][3] == []:
                    self.linksIn[signalName][i] = (1, widget, handler, [(value, id, signalNameFrom)])
                else:
                    found = 0
                    for j in range(len(self.linksIn[signalName][i][3])):
                        (val, ID, nameFrom) = self.linksIn[signalName][i][3][j]
                        if ID == id and nameFrom == signalNameFrom:
                            self.linksIn[signalName][i][3][j] = (value, id, signalNameFrom)
                            found = 1
                    if not found:
                        self.linksIn[signalName][i] = (1, widget, handler, self.linksIn[signalName][i][3] + [(value, id, signalNameFrom)])
        self.needProcessing = 1


    # ############################################
    # PROGRESS BAR FUNCTIONS
    def progressBarInit(self):
        self.progressBarValue = 0
        self.startTime = time.time()
        self.setCaption(self.captionTitle + " (0% complete)")
        if self.progressBarHandler: self.progressBarHandler(self, -1)

    def progressBarSet(self, value):
        if value > 0:
            self.progressBarValue = value
            diff = time.time() - self.startTime
            total = diff * 100.0/float(value)
            remaining = max(total - diff, 0)
            h = int(remaining/3600)
            min = int((remaining - h*3600)/60)
            sec = int(remaining - h*3600 - min*60)
            if h > 0:
                text = "%(h)d h, %(min)d min, %(sec)d sec" % vars()
            else:
                text = "%(min)d min, %(sec)d sec" % vars()
            self.setCaption(self.captionTitle + " (%(value).2f%% complete, remaining time: %(text)s)" % vars())
        else:
            self.setCaption(self.captionTitle + " (0% complete)" )
        if self.progressBarHandler: self.progressBarHandler(self, value)
        qApp.processEvents()

    def progressBarAdvance(self, value):
        self.progressBarSet(self.progressBarValue+value)

    def progressBarFinished(self):
        self.setCaption(self.captionTitle)
        if self.progressBarHandler: self.progressBarHandler(self, 101)

    # handler must be a function, that receives 2 arguments. First is the widget instance, the second is the value between -1 and 101
    def setProgressBarHandler(self, handler):
        self.progressBarHandler = handler

    def setProcessingHandler(self, handler):
        self.processingHandler = handler

    def setEventHandler(self, handler):
        self.eventHandler = handler

    def setWidgetStateHandler(self, handler):
        self.widgetStateHandler = handler

    # if we are in debug mode print the event into the file
    def printEvent(self, text, eventVerbosity = 1):
        self.signalManager.addEvent(self.captionTitle[3:] + ": " + text, eventVerbosity = eventVerbosity)
        if self.eventHandler:
            self.eventHandler(self.captionTitle[3:] + ": " + text, eventVerbosity)

    def openWidgetHelp(self):
        if orangedir:
            try:
                import win32help
                win32help.HtmlHelp(0, "%s/doc/catalog.chm::/catalog/%s/%s.htm" % (orangedir, self.category, self.__class__.__name__[2:]), win32help.HH_DISPLAY_TOPIC)
                return
            except:
                pass

            try:
                import webbrowser
                webbrowser.open("file://%s/doc/widgets/catalog/%s/%s.htm" % (orangedir, self.category, self.__class__.__name__[2:]), 0, 1)
                return
            except:
                pass

        try:
            import webbrowser
            webbrowser.open("http://www.ailab.si/orange/doc/widgets/catalog/%s/%s.htm" % (self.category, self.__class__.__name__[2:]))
            return
        except:
            pass


    def keyPressEvent(self, e):
        if e.key() == 0x1030:
            self.openWidgetHelp()
#            e.ignore()
        else:
            QDialog.keyPressEvent(self, e)

    def information(self, id = 0, text = None):
        self.setState("Info", id, text)

    def warning(self, id = 0, text = ""):
        self.setState("Warning", id, text)

    def error(self, id = 0, text = ""):
        self.setState("Error", id, text)

    def setState(self, stateType, id, text):
        if type(id) == list:
            for val in id:
                if self.widgetState[stateType].has_key(val):
                    self.widgetState[stateType].pop(val)
        else:
            if type(id) == str:
                text = id; id = 0       # if we call information(), warning(), or error() function with only one parameter - a string - then set id = 0
            if not text:
                if self.widgetState[stateType].has_key(id):
                    self.widgetState[stateType].pop(id)
            else:
                self.widgetState[stateType][id] = text

        if self.widgetStateHandler:
            self.widgetStateHandler()
        elif text and stateType != "Info":
            self.printEvent(stateType + " - " + text)
        qApp.processEvents()

    def synchronizeContexts(self):
        if hasattr(self, "contextHandlers"):
            for contextName, handler in self.contextHandlers.items():
                context = self.currentContexts.get(contextName, None)
                if context:
                    handler.settingsFromWidget(self, context)

    def openContext(self, contextName="", *arg):
        if not self._useContexts:
            return
        handler = self.contextHandlers[contextName]
        context = handler.openContext(self, *arg)
        if context:
            self.currentContexts[contextName] = context


    def closeContext(self, contextName=""):
        if not self._useContexts:
            return
        curcontext = self.currentContexts.get(contextName)
        if curcontext:
            self.contextHandlers[contextName].closeContext(self, curcontext)
            del self.currentContexts[contextName]

    def settingsToWidgetCallback(self, handler, context):
        pass

    def settingsFromWidgetCallback(self, handler, context):
        pass

    def setControllers(self, obj, controlledName, controller, prefix):
        while obj:
            if prefix:
#                print "SET CONTROLLERS: %s %s + %s" % (obj.__class__.__name__, prefix, controlledName)
                if obj.__dict__.has_key("attributeController"):
                    obj.__dict__["__attributeControllers"][(controller, prefix)] = True
                else:
                    obj.__dict__["__attributeControllers"] = {(controller, prefix): True}

            parts = controlledName.split(".", 1)
            if len(parts) < 2:
                break
            obj = getattr(obj, parts[0], None)
            prefix += parts[0]
            controlledName = parts[1]

    def __setattr__(self, name, value):
        return unisetattr(self, name, value, QDialog)

    def randomlyChangeSettings(self, verboseMode = 0):
        if len(self._guiElements) == 0: return

        try:
            index = random.randint(0, len(self._guiElements)-1)
            elementType, widget = self._guiElements[index][0], self._guiElements[index][1]

            if elementType == "qwtPlot":
                widget.randomChange()
                return

            if not widget.isEnabled(): return
            newValue = ""
            callback = None
            if elementType == "checkBox":
                elementType, widget, value, callback = self._guiElements[index]
                newValue = "Changing checkbox %s to %s" % (value, not self.getdeepattr(value))
                setattr(self, value, not self.getdeepattr(value))
            elif elementType == "button":
                elementType, widget, callback = self._guiElements[index]
                if widget.isToggleButton():
                    newValue = "Clicking button %s. State is %d" % (str(widget.text()).strip(), not widget.isOn())
                    widget.setOn(not widget.isOn())
                else:
                    newValue = "Pressed button %s" % (str(self.caption()))
            elif elementType == "listBox":
                elementType, widget, value, callback = self._guiElements[index]
                if widget.count():
                    itemIndex = random.randint(0, widget.count()-1)
                    newValue = "Listbox %s. Changed selection of item %d to %s" % (value, itemIndex, not widget.isSelected(itemIndex))
                    widget.setSelected(itemIndex, not widget.isSelected(itemIndex))
                else:
                    callback = None
            elif elementType == "radioButtonsInBox":
                elementType, widget, value, callback = self._guiElements[index]
                radioIndex = random.randint(0, len(widget.buttons)-1)
                if widget.buttons[radioIndex].isEnabled():
                    newValue = "Set radio button %s to index %d" % (value, radioIndex)
                    setattr(self, value, radioIndex)
                else:
                    callback = None
            elif elementType == "radioButton":
                elementType, widget, value, callback = self._guiElements[index]
                newValue = "Set radio button %s to %d" % (value, not self.getdeepattr(value))
                setattr(self, value, not self.getdeepattr(value))
            elif elementType in ["hSlider", "qwtHSlider", "spin"]:
                elementType, widget, value, min, max, step, callback = self._guiElements[index]
                currentValue = self.getdeepattr(value)
                if currentValue == min:   setattr(self, value, currentValue+step)
                elif currentValue == max: setattr(self, value, currentValue-step)
                else:                     setattr(self, value, currentValue + [-step,step][random.randint(0,1)])
                newValue = "Changed value of %s to %f" % (value, self.getdeepattr(value))
            elif elementType == "comboBox":
                elementType, widget, value, sendSelectedValue, valueType, callback = self._guiElements[index]
                if widget.count():
                    pos = random.randint(0, widget.count()-1)
                    newValue = "Changed value of combo %s to %s" % (value, str(widget.text(pos)))
                    if sendSelectedValue:
                        setattr(self, value, valueType(str(widget.text(pos))))
                    else:
                        setattr(self, value, pos)
                else:
                    callback = None
            if newValue != "":
                self.printEvent("Widget %s. %s" % (str(self.caption()), newValue), eventVerbosity = 1)
            if callback:
                if type(callback) == list:
                    for c in callback:
                        c()
                else:
                    callback()
        except:
            sys.stderr.write("------------------\n")
            if newValue != "":
                sys.stderr.write("Widget %s. %s\n" % (str(self.caption()), newValue))
            eType, val, traceback = sys.exc_info()
            sys.excepthook(eType, val, traceback)  # print the exception
            sys.stderr.write("Widget settings are:\n")
            for i, setting in enumerate(getattr(self, "settingsList", [])):
                sys.stderr.write("%30s: %7s\n" % (setting, str(self.getdeepattr(setting))))


if __name__ == "__main__":
    a=QApplication(sys.argv)
    oww=OWBaseWidget()
    a.setMainWidget(oww)
    oww.show()
    a.exec_loop()
    oww.saveSettings()
