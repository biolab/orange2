#
# OWWidget.py
# Orange Widget
# A General Orange Widget, from which all the Orange Widgets are derived
#
from OWTools import *
import sys, time, random, user, os, os.path, cPickle, copy, orngMisc
import orange
from string import *
from orngSignalManager import *

ERROR = 0
WARNING = 1

def mygetattr(obj, attr, **argkw):
    robj = obj
    try:
        for name in attr.split("."):
            robj = getattr(robj, name)
        return robj
    except:
        if argkw.has_key("default"):
            return argkw[default]
        else:
            raise AttributeError, "'%s' has no attribute '%s'" % (obj, attr)


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
        if hasattr(grandparent, "__setattr__"):
            grandparent.__setattr__(obj, lastname,  value)
        else:
            obj.__dict__[lastname] = value

    controlledAttributes = getattr(self, "controlledAttributes", None)
    controlCallback = controlledAttributes and controlledAttributes.get(name, None)
    if controlCallback:
        controlCallback(value)

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
                        controlCallback(value)

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


class Context:
    def __init__(self, **argkw):
        self.time = time.time()
        self.__dict__.update(argkw)

    def __getstate__(self):
        s = dict(self.__dict__)
        for nc in getattr(self, "noCopy", []):
            if s.has_key(nc):
                del s[nc]
        return s

    
class ContextHandler:
    maxSavedContexts = 50
    
    def __init__(self, contextName = "", cloneIfImperfect = True, loadImperfect = True, findImperfect = True, syncWithGlobal = True, **args):
        self.contextName = contextName
        self.localContextName = "localContexts"+contextName
        self.cloneIfImperfect, self.loadImperfect, self.findImperfect = cloneIfImperfect, loadImperfect, findImperfect
        self.syncWithGlobal = syncWithGlobal
        self.globalContexts = []
        self.__dict__.update(args)

    def newContext(self):
        return Context()

    def openContext(self, widget, *arg, **argkw):
        context, isNew = self.findOrCreateContext(widget, *arg, **argkw)
        if context:
            if isNew:
                self.settingsFromWidget(widget, context)
            else:
                self.settingsToWidget(widget, context)
        return context

    def findOrCreateContext(self, widget, *arg, **argkw):        
        if not hasattr(widget, self.localContextName):
            if self.syncWithGlobal:
                setattr(widget, self.localContextName, self.globalContexts)
            else:
                setattr(widget, self.localContextName, copy.deepcopy(self.globalContexts))

        index, context, score = self.findMatch(widget, self.findImperfect and self.loadImperfect, *arg, **argkw)
        if context:
            if index < 0:
                self.addContext(widget, context)
            else:
                self.moveContextUp(widget, index)
            return context, False
        else:
            context = self.newContext()
            self.addContext(widget, context)
            return context, True

    def closeContext(self, widget, context):
        self.settingsFromWidget(widget, context)

    def fastSave(self, context, widget, name):
        pass

    def settingsToWidget(self, widget, context):
        cb = getattr(widget, "settingsToWidgetCallback" + self.contextName, None)
        return cb and cb(self, context)

    def settingsFromWidget(self, widget, context):
        cb = getattr(widget, "settingsFromWidgetCallback" + self.contextName, None)
        return cb and cb(self, context)

    def findMatch(self, widget, imperfect = True, *arg, **argkw):
        bestI, bestContext, bestScore = -1, None, -1
        for i, c in enumerate(getattr(widget, self.localContextName)):
            score = self.match(c, imperfect, *arg, **argkw)
            if score == 2:
                return i, c, score
            if score and score > bestScore:
                bestI, bestContext, bestScore = i, c, score

        if bestContext and self.cloneIfImperfect:
            if hasattr(self, "cloneContext"):
                bestContext = self.cloneContext(bestContext, *arg, **argkw)
            else:
                import copy
                bestContext = copy.deepcopy(bestContext)
            bestI = -1
                
        return bestI, bestContext, bestScore
            
    def moveContextUp(self, widget, index):
        localContexts = getattr(widget, self.localContextName)
        l = getattr(widget, self.localContextName)
        l.insert(0, l.pop(index))

    def addContext(self, widget, context):
        l = getattr(widget, self.localContextName)
        l.insert(0, context)
        while len(l) > self.maxSavedContexts:
            del l[-1]

    def mergeBack(self, widget):
        if not self.syncWithGlobal:
            self.globalContexts.extend(getattr(widget, self.localContextName))
            self.globalContexts.sort(lambda c1,c2: -cmp(c1.time, c2.time))
            self.globalContexts = self.globalContexts[:self.maxSavedContexts]


class ContextField:
    def __init__(self, name, flags, **argkw):
        self.name = name
        self.flags = flags
        self.__dict__.update(argkw)
    

class ControlledAttributesDict(dict):
    def __init__(self, master):
        self.master = master

    def __setitem__(self, key, value):
        dict.__setitem__(self, key, value)
        self.master.setControllers(self.master, key, self.master, "")
        
        
class DomainContextHandler(ContextHandler):
    Optional, SelectedRequired, Required = range(3)
    RequirementMask = 3
    NotAttribute = 4
    List = 8
    RequiredList = Required + List
    SelectedRequiredList = SelectedRequired + List
    
    def __init__(self, contextName, fields = [],
                 cloneIfImperfect = True, loadImperfect = True, findImperfect = True, syncWithGlobal = True, **args):
        ContextHandler.__init__(self, contextName, cloneIfImperfect, loadImperfect, findImperfect, syncWithGlobal, **args)
        self.fields = []
        for field in fields:
            if isinstance(field, ContextField):
                self.fields.append(field)
            elif type(field)==str:
                self.fields.append(ContextField(field, self.Required))
            # else it's a tuple
            else:
                flags = field[1]
                if type(field[0]) == list:
                    self.fields.extend([ContextField(x, flags) for x in field[0]])
                else:
                    self.fields.append(ContextField(field[0], flags))
        
    def encodeDomain(self, domain):
        d = dict([(attr.name, attr.varType) for attr in domain])
        d.update(dict([(attr.name, attr.varType) for attr in domain.getmetas().values()]))
        return d
    
    def findOrCreateContext(self, widget, domain):
        if not domain:
            return None, False
        if type(domain) != orange.Domain:
            domain = domain.domain
            
        encodedDomain = self.encodeDomain(domain)
        context, isNew = ContextHandler.findOrCreateContext(self, widget, domain, encodedDomain)
        if not context:
            return None, False
        
        context.encodedDomain = encodedDomain

        metaIds = domain.getmetas().keys()
        metaIds.sort()
        context.orderedDomain = [(attr.name, attr.varType) for attr in domain] + [(domain[i].name, domain[i].varType) for i in metaIds]

        if isNew:
            context.values = {}
            context.noCopy = ["orderedDomain"]
        return context, isNew

    def settingsToWidget(self, widget, context):
        ContextHandler.settingsToWidget(self, widget, context)
        excluded = {}
        for field in self.fields:
            name, flags = field.name, field.flags

            excludes = getattr(field, "reservoir", [])
            if excludes:
                if type(excludes) != list:
                    excludes = [excludes]
                for exclude in excludes:
                    excluded.setdefault(exclude, [])

            if not context.values.has_key(name):
                continue
            
            value = context.values[name]

            if not flags & self.List:
                setattr(widget, name, value[0])
                for exclude in excludes:
                    excluded[exclude].append(value)

            else:
                newLabels, newSelected = [], []
                oldSelected = hasattr(field, "selected") and context.values.get(field.selected, []) or []
                for i, saved in enumerate(value):
                    if saved in context.orderedDomain:
                        if i in oldSelected:
                            newSelected.append(len(newLabels))
                        newLabels.append(saved)

                context.values[name] = newLabels
                setattr(widget, name, value)

                if hasattr(field, "selected"):
                    context.values[field.selected] = newSelected
                    setattr(widget, field.selected, context.values[field.selected])

                for exclude in excludes:
                    excluded[exclude].extend(value)

        for name, values in excluded.items():
            ll = filter(lambda a: a not in values, context.orderedDomain)
            setattr(widget, name, ll)
            

    def settingsFromWidget(self, widget, context):
        ContextHandler.settingsFromWidget(self, widget, context)
        context.values = {}
        for field in self.fields:
            if not field.flags & self.List:
                self.saveLow(context, widget, field.name, mygetattr(widget, field.name))
            else:
                context.values[field.name] = mygetattr(widget, field.name)
                if hasattr(field, "selected"):
                    context.values[field.selected] = list(mygetattr(widget, field.selected))

    def fastSave(self, context, widget, name, value):
        if context:
            for field in self.fields:
                if name == field.name:
                    if field.flags & self.List:
                        context.values[field.name] = value
                    else:
                        self.saveLow(context, widget, name, value)
                    return
                if name == getattr(field, "selected", None):
                    context.values[field.selected] = list(value)
                    return

    def saveLow(self, context, widget, field, value):
        if type(value) == str:
            context.values[field] = value, context.encodedDomain.get(value, -1) # -1 means it's not an attribute
        else:
            context.values[field] = value, -2

    def match(self, context, imperfect, domain, encodedDomain):
        if encodedDomain == context.encodedDomain:
            return 2
        if not imperfect:
            return 0

        filled = potentiallyFilled = 0
        for field in self.fields:
            value = context.values.get(field.name, None)
            if value:
                if field.flags & self.List:
                    potentiallyFilled += len(value)
                    if field.flags & self.RequirementMask == self.Required:
                        filled += len(value)
                        for item in value:
                            if encodedDomain.get(item[0], None) != item[1]:
                                return 0
                    else:
                        selectedRequired = field.flags & self.RequirementMask == self.SelectedRequired
                        for i in context.values.get(field.selected, []):
                            if value[i] in encodedDomain:
                                filled += 1
                            else:
                                if selectedRequired:
                                    return 0
                else:
                    potentiallyFilled += 1
                    if value[1] >= 0:
                        if encodedDomain.get(value[0], None) == value[1]:
                            filled += 1
                        else:
                            if field.flags & self.Required:
                                return 0

            if not potentiallyFilled:
                return 1.0
            else:
                return filled / float(potentiallyFilled)

    def cloneContext(self, context, domain, encodedDomain):
        import copy
        context = copy.deepcopy(context)
        
        for field in self.fields:
            value = context.values.get(field.name, None)
            if value:
                if field.flags & self.List:
                    i = j = realI = 0
                    selected = context.values.get(field.selected, [])
                    selected.sort()
                    nextSel = selected and selected[0] or None
                    while i < len(value):
                        if encodedDomain.get(value[i][0], None) != value[i][1]:
                            del value[i]
                            if nextSel == realI:
                                del selected[j]
                                nextSel = j < len(selected) and selected[j] or None
                        else:
                            if nextSel == realI:
                                selected[j] -= realI - i
                                j += 1
                                nextSel = j < len(selected) and selected[j] or None
                            i += 1
                        realI += 1
                    if hasattr(field, "selected"):
                        context.values[field.selected] = selected[:j]
                else:
                    if value[1] >= 0 and encodedDomain.get(value[0], None) != value[1]:
                        del context.values[field.name]
                        
        context.encodedDomain = encodedDomain
        context.orderedDomain = [(attr.name, attr.varType) for attr in domain]
        return context

    
##################
# this definitions are needed only to define ExampleTable as subclass of ExampleTableWithClass
class ExampleTable(orange.ExampleTable):
    pass

class ExampleTableWithClass(ExampleTable):
    pass

class AttributeList(list):
    pass

class ExampleList(list):
    pass

class OWBaseWidget(QDialog):
    def __init__(self, parent = None, signalManager = None, title="Qt Orange BaseWidget", modal=FALSE):
        # the "currentContexts" MUST be the first thing assigned to a widget
        self.currentContexts = {}
        self._guiElements = []      # used for automatic widget debugging
        
        self.title = title.replace("&","")

        QDialog.__init__(self, parent, self.title, modal, Qt.WStyle_Customize + Qt.WStyle_NormalBorder + Qt.WStyle_Title + Qt.WStyle_SysMenu + Qt.WStyle_Minimize + Qt.WStyle_Maximize)
        
        # directories are better defined this way, otherwise .ini files get written in many places
        self.widgetDir = os.path.dirname(__file__) + "/"

        # create output directory for widget settings
        if os.name == "nt": self.outputDir = self.widgetDir
        else:               self.outputDir = os.path.join(user.home, "Orange")
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
        self.widgetState = None
    
        #the title
        self.setCaption(self.captionTitle)

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

    def setCaption(self, caption):
        if self.parent != None and isinstance(self.parent, QTabWidget): self.parent.changeTab(self, caption)
        else: QDialog.setCaption(self, caption)

    def setCaptionTitle(self, caption):
        self.captionTitle = caption     # we have to save caption title in case progressbar will change it
        QDialog.setCaption(self, caption)
        
    # put this widget on top of all windows
    def reshow(self):
        self.hide()
        self.show()

    def send(self, signalName, value, id = None):
        if self.linksOut.has_key(signalName):
            self.linksOut[signalName][id] = value
        else:
            self.linksOut[signalName] = {id:value}
            
        self.signalManager.send(self, signalName, value, id)

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
                    settings[name] =  mygetattr(self, name)
                except:
                    print "Attribute %s not found in %s widget. Remove it from the settings list." % (name, self.title)
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
                        if contexts:
                            contextHandler.globalContexts = contexts
            

        
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
        for key in self.linksIn.keys():
            for i in range(len(self.linksIn[key])):
                (dirty, widgetFrom, handler, signalData) = self.linksIn[key][i]
                if not (handler and dirty): continue
                
                qApp.setOverrideCursor(QWidget.waitCursor)
                try:                    
                    for (value, id, nameFrom) in signalData:
                        if self.signalIsOnlySingleConnection(key):
                            self.printVerbose("ProcessSignals: calling %s with %s" % (handler, value))
                            handler(value)
                        else:
                            self.printVerbose("ProcessSignals: calling %s with %s (%s, %s)" % (handler, value, nameFrom, id))
                            handler(value, (widgetFrom, nameFrom, id))
                except:
                    type, val, traceback = sys.exc_info()
                    sys.excepthook(type, val, traceback)  # we pretend that we handled the exception, so that we don't crash other widgets

                qApp.restoreOverrideCursor()
                self.linksIn[key][i] = (0, widgetFrom, handler, []) # clear the dirty flag

        if self.processingHandler: self.processingHandler(self, 0)    # remove focus from this widget
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
            text = ""
            if h > 0: text += "%d h, " % (h)
            text += "%d min, %d sec" %(min, sec)
            self.setCaption(self.captionTitle + " (%.2f%% complete, remaining time: %s)" % (value, text))
        else:
            self.setCaption(self.captionTitle + " (0% complete)" )
        if self.progressBarHandler: self.progressBarHandler(self, value)

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

    def setStatusBarText(self, text):
        self.statusBar.message(text)

    def printVerbose(self, text):
        orngMisc.printVerbose(text)

    # if we are in debug mode print the event into the file
    def printEvent(self, type, text):
        if text: self.signalManager.addEvent(type + " from " + self.captionTitle[3:] + ": " + text)
        
        if not self.eventHandler: return
        if text == None:
            self.eventHandler("")
        else:                
            self.eventHandler(type + " from " + self.captionTitle[3:] + ": " + text)
        
    def information(self, text = None):
        self.printEvent("Information", text)

    def warning(self, text = "", code = 0):
        self.setState(WARNING, text, code)
        
        if text and type(text) == str and not self.widgetStateHandler:
            self.printEvent("Warning", text)

    def error(self, text = "", code = 0):
        self.setState(ERROR, text, code)

        if text and type(text) == str and not self.widgetStateHandler:
            self.printEvent("Error", text)

    def setState(self, stateType, text, code):
        if not self.widgetState: self.widgetState = [[], []]

        if type(text) == int:
            code = text; text = ""        # when we want to remove an error we can call simply error(int_val). in this case text will actually be an integer

        if not text and code == 0: self.widgetState[stateType] = []
        else:
            for (c, t) in self.widgetState[stateType]:
                if c == code: self.widgetState[stateType].remove((c,t))
            if text:
                self.widgetState[stateType].append((code, text))

        if self.widgetState == [[],[]]: self.widgetState = None

        if self.widgetStateHandler:
            self.widgetStateHandler()

    def synchronizeContexts(self):
        if hasattr(self, "contextHandlers"):
            for contextName, handler in self.contextHandlers.items():
                context = self.currentContexts.get(contextName, None)
                if context:
                    handler.settingsFromWidget(self, context)

    def openContext(self, contextName="", *arg):
        #self.closeContext(contextName)
        handler = self.contextHandlers[contextName]
        context = handler.openContext(self, *arg)
        if context:
            self.currentContexts[contextName] = context


    def closeContext(self, contextName=""):
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

    def randomlyChangeSettings(self):
        if len(self._guiElements) == 0: return
        
        index = random.randint(0, len(self._guiElements)-1)
        type, widget = self._guiElements[index][0], self._guiElements[index][1]
        if not widget.isEnabled(): return
        if type == "checkBox":
            type, widget, value, callback = self._guiElements[index]
            setattr(self, value, not mygetattr(self, value))
            if callback:
                callback()
        elif type == "button":
            type, widget, callback = self._guiElements[index]
            if widget.isToggleButton():
                widget.setOn(not widget.isOn())
            if callback:
                callback()
        elif type == "listBox":
            type, widget, value, callback = self._guiElements[index]
            if widget.count():
                itemIndex = random.randint(0, widget.count()-1)
                widget.setSelected(itemIndex, not widget.isSelected(itemIndex))
                if callback:
                    callback()
        elif type == "radioButtonsInBox":
            type, widget, value, callback = self._guiElements[index]
            radioIndex = random.randint(0, len(widget.buttons)-1)
            if widget.buttons[radioIndex].isEnabled():
                setattr(self, value, radioIndex)
                if callback:
                    callback()
        elif type == "radioButton":
            type, widget, value, callback = self._guiElements[index]
            setattr(self, value, not mygetattr(self, value))
            if callback:
                callback()
        elif type in ["hSlider", "qwtHSlider", "spin"]:
            type, widget, value, min, max, step, callback = self._guiElements[index]
            currentValue = mygetattr(self, value)
            if currentValue == min:   setattr(self, value, currentValue+step)
            elif currentValue == max: setattr(self, value, currentValue-step)
            else:                     setattr(self, value, currentValue + [-step,step][random.randint(0,1)])
            if callback:
                callback()
        elif type == "comboBox":
            type, widget, value, sendSelectedValue, valueType, callback = self._guiElements[index]
            if widget.count():
                pos = random.randint(0, widget.count()-1)
                if sendSelectedValue:
                    setattr(self, value, valueType(str(widget.text(pos))))
                else:
                    setattr(self, value, pos)
                if callback:
                    callback()  

if __name__ == "__main__":  
    a=QApplication(sys.argv)
    oww=OWBaseWidget()
    a.setMainWidget(oww)
    oww.show()
    a.exec_loop()
    oww.saveSettings()
