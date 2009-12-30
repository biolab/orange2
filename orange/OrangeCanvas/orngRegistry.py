# Author: Gregor Leban (gregor.leban@fri.uni-lj.si)
#

import os, sys, re, glob, stat
from orngSignalManager import OutputSignal, InputSignal
from PyQt4.QtCore import *
from PyQt4.QtGui import *

orangeDir = os.path.split(os.path.split(os.path.abspath(__file__))[0])[0]
if not orangeDir in sys.path:
    sys.path.append(orangeDir)

import orngEnviron

class WidgetDescription:
    def __init__(self, **attrs):
        self.__dict__.update(attrs)

class WidgetCategory(dict):
    def __init__(self, name, widgets):
        self.update(widgets)
        self.name = name
    def __init__(self, name):
        self.name = name
   
def readCategories(silent=False):
    currentCacheVersion = 1
    
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

    directories = [] # tuples (catName, dirName, plugin, isPrototype)
    for dirName in os.listdir(widgetDirName):
        directory = os.path.join(widgetDirName, dirName)
        if os.path.isdir(directory):
            directories.append((directory, "", "prototypes" in dirName.lower()))

    # read list of add-ons (in orange/add-ons as well as those additionally registered by the user)
    for (name, dirName) in orngEnviron.addOns:
        addOnWidgetsDir = os.path.join(dirName, "widgets")
        if os.path.isdir(addOnWidgetsDir):
            directories.append((addOnWidgetsDir, addOnWidgetsDir, False))
        addOnWidgetsPrototypesDir = os.path.join(addOnWidgetsDir, "prototypes")
        if os.path.isdir(addOnWidgetsPrototypesDir):
            directories.append((addOnWidgetsPrototypesDir, addOnWidgetsPrototypesDir, True))

    categories = {}     
    for dirName, plugin, isPrototype in directories:
        widgets = readWidgets(dirName, cachedWidgetDescriptions, isPrototype, silent=silent)
        for (wName, wInfo) in widgets:
            catName = wInfo.category
            if not catName in categories:
                categories[catName] = WidgetCategory(catName)
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


re_inputs = re.compile(r'[ \t]+self.inputs\s*=\s*(?P<signals>\[[^]]*\])', re.DOTALL)
re_outputs = re.compile(r'[ \t]+self.outputs\s*=\s*(?P<signals>\[[^]]*\])', re.DOTALL)

hasErrors = False
splashWindow = None
widgetsWithError = []
widgetsWithErrorPrototypes = []

def readWidgets(directory, cachedWidgetDescriptions, prototype=False, silent=False):
    import sys, imp
    global hasErrors, splashWindow, widgetsWithError, widgetsWithErrorPrototypes
    
    widgets = []
    widgetdir = os.path.split(directory.strip(os.path.sep).strip(os.path.altsep))[1]
    for filename in glob.iglob(os.path.join(directory, "*.py")):
        if os.path.isdir(filename) or os.path.islink(filename):
            continue
        
        datetime = str(os.stat(filename)[stat.ST_MTIME])
        cachedDescription = cachedWidgetDescriptions.get(filename, None)
        if cachedDescription and cachedDescription.time == datetime and hasattr(cachedDescription, "inputClasses"):
            widgets.append((cachedDescription.name, cachedDescription))
            continue
        
        data = file(filename).read()
        istart = data.find("<name>")
        iend = data.find("</name>")
        if istart < 0 or iend < 0:
            continue
        name = data[istart+6:iend]
        inputList = getSignalList(re_inputs, data)
        outputList = getSignalList(re_outputs, data)
        
        dirname, fname = os.path.split(filename)
        widgname = os.path.splitext(fname)[0]
        try:
            if not splashWindow:
                import orngEnviron
                logo = QPixmap(os.path.join(orngEnviron.directoryNames["canvasDir"], "icons", "splash.png"))
                splashWindow = QSplashScreen(logo, Qt.WindowStaysOnTopHint)
                splashWindow.setMask(logo.mask())
                splashWindow.show()
                
            splashWindow.showMessage("Registering widget %s" % name, Qt.AlignHCenter + Qt.AlignBottom)
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
            inputClasses = set(eval(x[1], wmod.__dict__).__name__ for x in eval(inputList))
            outputClasses = set(y.__name__ for x in eval(outputList) for y in eval(x[1], wmod.__dict__).mro())
            
            widgetInfo = WidgetDescription(
                             name = data[istart+6:iend],
                             time = datetime,
                             fileName = widgname,
                             fullName = filename,
                             directory = directory,
                             inputList = inputList, outputList = outputList,
                             inputClasses = inputClasses, outputClasses = outputClasses
                             )
    
            for attr, deflt in (("contact>", "") , ("icon>", "icons/Unknown.png"), ("priority>", "5000"), ("description>", ""), ("category>", widgetdir)):
                istart, iend = data.find("<"+attr), data.find("</"+attr)
                setattr(widgetInfo, attr[:-1], istart >= 0 and iend >= 0 and data[istart+1+len(attr):iend].strip() or deflt)
                
    
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
    
            widgetInfo.tooltipText = "<b><b>&nbsp;%s</b></b><hr><b>Description:</b><br>&nbsp;&nbsp;%s<hr>%s<hr>%s" % (name, widgetInfo.description, formatedInList[:-4], formatedOutList[:-4]) 
            widgets.append((name, widgetInfo))
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


re_tuple = re.compile(r"\(([^)]+)\)")

def getSignalList(regex, data):
    inmo = regex.search(data)
    if inmo:
        return str([tuple([y[0] in "'\"" and y[1:-1] or str(y) for y in (x.strip() for x in ttext.group(1).split(","))])
               for ttext in re_tuple.finditer(inmo.group("signals"))])
    else:
        return "[]"