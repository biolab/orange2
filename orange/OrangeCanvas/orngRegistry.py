# Author: Gregor Leban (gregor.leban@fri.uni-lj.si)
#

import os, sys, re, glob, stat
from orngSignalManager import OutputSignal, InputSignal
from PyQt4.QtCore import *
from PyQt4.QtGui import *
import widgetParser

orangeDir = os.path.split(os.path.split(os.path.abspath(__file__))[0])[0]
if not orangeDir in sys.path:
    sys.path.append(orangeDir)

import orngEnviron, Orange.misc.addons

class WidgetDescription:
    def __init__(self, **attrs):
        self.__dict__.update(attrs)

    def docDir(self):
        if not self.addOn:  # A built-in widget
            dir, widgetDir = os.path.realpath(self.directory), os.path.realpath(orngEnviron.widgetDir)
            subDir = os.path.relpath(dir, widgetDir) if "relpath" in os.path.__dict__ else dir.replace(widgetDir, "")
            return os.path.join(orngEnviron.orangeDocDir, "catalog", subDir)
        else:  # An add-on widget
            addOnDocDir = self.addOn.directoryDocumentation()
            return os.path.join(addOnDocDir, "widgets")


class WidgetCategory(dict):
    def __init__(self, name, widgets=None):
        if widgets:
            self.update(widgets)
        self.name = name
   
def readCategories(silent=False):
    currentCacheVersion = 2
    
    global widgetsWithError, widgetsWithErrorPrototypes
    widgetDirName = os.path.realpath(orngEnviron.directoryNames["widgetDir"])
    canvasSettingsDir = os.path.realpath(orngEnviron.directoryNames["canvasSettingsDir"])
    cacheFilename = os.path.join(canvasSettingsDir, "cachedWidgetDescriptions.pickle")

    try:
        import cPickle
        cacheFile = file(cacheFilename, "rb")
        cats = cPickle.load(cacheFile)
        try:
            version = cPickle.load(cacheFile)
        except EOFError:
            version = 0
        if version == currentCacheVersion:
            cachedWidgetDescriptions = dict([(w.fullName, w) for cat in cats.values() for w in cat.values()])
        else:
            cachedWidgetDescriptions = {}
    except:
        cachedWidgetDescriptions = {} 

    directories = [] # tuples (defaultCategory, dirName, plugin, isPrototype)
    for dirName in os.listdir(widgetDirName):
        directory = os.path.join(widgetDirName, dirName)
        if os.path.isdir(directory):
            directories.append((None, directory, None, "prototypes" in dirName.lower()))
            
    # read list of add-ons
    for addOn in Orange.misc.addons.installed_addons.values() + Orange.misc.addons.registered_addons:
        addOnWidgetsDir = os.path.join(addOn.directory, "widgets")
        if os.path.isdir(addOnWidgetsDir):
            directories.append((addOn.name, addOnWidgetsDir, addOn, False))
        addOnWidgetsPrototypesDir = os.path.join(addOnWidgetsDir, "prototypes")
        if os.path.isdir(addOnWidgetsPrototypesDir):
            directories.append((None, addOnWidgetsPrototypesDir, addOn, True))

    categories = {}     
    for defCat, dirName, addOn, isPrototype in directories:
        widgets = readWidgets(dirName, cachedWidgetDescriptions, isPrototype, silent=silent, addOn=addOn, defaultCategory=defCat)
        for (wName, wInfo) in widgets:
            catName = wInfo.category
            if not catName in categories:
                categories[catName] = WidgetCategory(catName)
            if wName in categories[catName]:
                print "Warning! A widget with duplicated name '%s' in category '%s' has been found! It will _not_ be available in the Canvas." % (wName, catName)
            else:
                categories[catName][wName] = wInfo

    cacheFile = file(cacheFilename, "wb")
    cPickle.dump(categories, cacheFile)
    cPickle.dump(currentCacheVersion, cacheFile)
    if splashWindow:
        splashWindow.hide()

    if not silent:
        if widgetsWithError != []:
            print "The following widgets could not be imported and will not be available: " + ", ".join(set(widgetsWithError)) + "."
        if widgetsWithErrorPrototypes != []:
            print "The following prototype widgets could not be imported and will not be available: " + ", ".join(set(widgetsWithErrorPrototypes)) + "."

    return categories


hasErrors = False
splashWindow = None
widgetsWithError = []
widgetsWithErrorPrototypes = []

def readWidgets(directory, cachedWidgetDescriptions, prototype=False, silent=False, addOn=None, defaultCategory=None):
    import sys, imp
    global hasErrors, splashWindow, widgetsWithError, widgetsWithErrorPrototypes
    
    widgets = []
    
    if not defaultCategory:
        predir, defaultCategory = os.path.split(directory.strip(os.path.sep).strip(os.path.altsep))
        if defaultCategory == "widgets":
            defaultCategory = os.path.split(predir.strip(os.path.sep).strip(os.path.altsep))[1]
    
    if defaultCategory.lower() == "prototypes" or prototype:
        defaultCategory = "Prototypes"
    
    for filename in glob.iglob(os.path.join(directory, "*.py")):
        if os.path.isdir(filename):
            continue
        
        datetime = str(os.stat(filename)[stat.ST_MTIME])
        cachedDescription = cachedWidgetDescriptions.get(filename, None)
        if cachedDescription and cachedDescription.time == datetime and hasattr(cachedDescription, "inputClasses"):
            widgets.append((cachedDescription.name, cachedDescription))
            continue
        
        data = file(filename).read()
        try:
            meta = widgetParser.WidgetMetaData(data, defaultCategory, enforceDefaultCategory=prototype)
        except:   # Probably not an Orange widget module.
            continue

        dirname, fname = os.path.split(filename)
        widgname = os.path.splitext(fname)[0]
        try:
            if not splashWindow:
                import orngEnviron
                logo = QPixmap(os.path.join(orngEnviron.directoryNames["canvasDir"], "icons", "splash.png"))
                splashWindow = QSplashScreen(logo, Qt.WindowStaysOnTopHint)
                splashWindow.setMask(logo.mask())
                splashWindow.show()
                
            splashWindow.showMessage("Registering widget %s" % meta.name, Qt.AlignHCenter + Qt.AlignBottom)
            qApp.processEvents()
            
            # We import modules using imp.load_source to avoid storing them in sys.modules,
            # but we need to append the path to sys.path in case the module would want to load
            # something
            dirnameInPath = dirname in sys.path
            if not dirnameInPath:
                sys.path.append(dirname)
            wmod = imp.load_source(widgname, filename)
            if not dirnameInPath and dirname in sys.path: # I have no idea, why we need this, but it seems to disappear sometimes?!
                sys.path.remove(dirname)
            widgClass = wmod.__dict__[widgname]
            inputClasses = set(eval(x[1], wmod.__dict__).__name__ for x in eval(meta.inputList))
            outputClasses = set(y.__name__ for x in eval(meta.outputList) for y in eval(x[1], wmod.__dict__).mro())
            
            widgetInfo = WidgetDescription(
                             name = meta.name,
                             time = datetime,
                             fileName = widgname,
                             fullName = filename,
                             directory = directory,
                             addOn = addOn,
                             inputList = meta.inputList, outputList = meta.outputList,
                             inputClasses = inputClasses, outputClasses = outputClasses
                             )
    
            for attr in ["contact", "icon", "priority", "description", "category"]:
                setattr(widgetInfo, attr, getattr(meta, attr))
    
            # build the tooltip
            widgetInfo.inputs = [InputSignal(*signal) for signal in eval(widgetInfo.inputList)]
            if len(widgetInfo.inputs) == 0:
                formatedInList = "<b>Inputs:</b><br> &nbsp;&nbsp; None<br>"
            else:
                formatedInList = "<b>Inputs:</b><br>"
                for signal in widgetInfo.inputs:
                    formatedInList += " &nbsp;&nbsp; - " + signal.name + " (" + signal.type + ")<br>"
    
            widgetInfo.outputs = [OutputSignal(*signal) for signal in eval(widgetInfo.outputList)]
            if len(widgetInfo.outputs) == 0:
                formatedOutList = "<b>Outputs:</b><br> &nbsp; &nbsp; None<br>"
            else:
                formatedOutList = "<b>Outputs:</b><br>"
                for signal in widgetInfo.outputs:
                    formatedOutList += " &nbsp; &nbsp; - " + signal.name + " (" + signal.type + ")<br>"

            addOnName = "" if not widgetInfo.addOn else " (from add-on %s)" % widgetInfo.addOn.name
    
            widgetInfo.tooltipText = "<b><b>&nbsp;%s</b></b>%s<hr><b>Description:</b><br>&nbsp;&nbsp;%s<hr>%s<hr>%s" % (meta.name, addOnName, widgetInfo.description, formatedInList[:-4], formatedOutList[:-4]) 
            widgets.append((meta.name, widgetInfo))
        except Exception, msg:
            if not hasErrors and not silent:
                print "There were problems importing the following widgets:"
                hasErrors = True
            if not silent:
                print "   %s: %s" % (widgname, msg)

            if not prototype:
                widgetsWithError.append(widgname)
            else:
                widgetsWithErrorPrototypes.append(widgname)
       
    return widgets
