# Author: Gregor Leban (gregor.leban@fri.uni-lj.si)
#

import os, sys, re, glob, stat
from PyQt4.QtCore import *
from PyQt4.QtGui import *

orangeDir = os.path.split(os.path.split(os.path.abspath(__file__))[0])[0]
if not orangeDir in sys.path:
    sys.path.append(orangeDir)

from orngEnviron import *

class WidgetDescription:
    def __init__(self, **attrs):
        self.__dict__.update(attrs)

class WidgetCategory:
    def __init__(self, name, widgets, directory):
        self.name = name
        self.widgets = widgets
        self.directory = directory
   
storedCategories = None
def readCategories():
    global storedCategories
    if storedCategories:
        return storedCategories
    
    
    widgetDirName = os.path.realpath(directoryNames["widgetDir"])
    canvasSettingsDir = os.path.realpath(directoryNames["canvasSettingsDir"])
    cacheFilename = os.path.join(canvasSettingsDir, "cachedWidgetDescriptions.pickle") 

    try:
        import cPickle
        cats = cPickle.load(file(cacheFilename, "rb"))
        cachedWidgetDescriptions = dict([(w.fullname, w) for cat in cats for w in cat.widgets])
    except:
        cachedWidgetDescriptions = {} 

    directories = []
    for dirName in os.listdir(widgetDirName):
        directory = os.path.join(widgetDirName, dirName)
        if os.path.isdir(directory):
            directories.append((dirName, directory, ""))

    # read list of add-ons (in orange/add-ons as well as those additionally registered by the user)
    for (name, dirName) in addOns:
        addOnWidgetsDir = os.path.join(dirName, "widgets")
        if os.path.isdir(addOnWidgetsDir):
            directories.append((name, addOnWidgetsDir, addOnWidgetsDir))
        addOnWidgetsPrototypesDir = os.path.join(addOnWidgetsDir, "prototypes")
        if os.path.isdir(addOnWidgetsDir):
            directories.append(("Prototypes", addOnWidgetsPrototypesDir, addOnWidgetsPrototypesDir))

    categories = []
    for catName, dirName, plugin in directories:
        widgets = readWidgets(dirName, cachedWidgetDescriptions)
        if widgets:
            categories.append(WidgetCategory(catName, widgets, plugin and dirName or ""))

    cPickle.dump(categories, file(cacheFilename, "wb"))
    storedCategories = categories
    if splashWindow:
        splashWindow.hide()
    return categories


re_inputs = re.compile(r'[ \t]+self.inputs\s*=\s*(?P<signals>\[[^]]*\])', re.DOTALL)
re_outputs = re.compile(r'[ \t]+self.outputs\s*=\s*(?P<signals>\[[^]]*\])', re.DOTALL)

hasErrors = False
splashWindow = None

def readWidgets(directory, cachedWidgetDescriptions):
    import sys, imp
    global hasErrors, splashWindow
    
    widgets = []
    for filename in glob.iglob(os.path.join(directory, "*.py")):
        if os.path.isdir(filename) or os.path.islink(filename):
            continue
        
        datetime = str(os.stat(filename)[stat.ST_MTIME])
        cachedDescription = cachedWidgetDescriptions.get(filename, None)
        if cachedDescription and cachedDescription.time == datetime and hasattr(cachedDescription, "inputClasses"):
            widgets.append(cachedDescription)
            continue
        
        data = file(filename).read()
        istart = data.find("<name>")
        if istart < 0:
            continue
        iend = data.find("</name>")
        if iend < 0:
            continue
        name = data[istart+6:iend]
        
        inputList=getSignalList(re_inputs, data)
        outputList=getSignalList(re_outputs, data)
        
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
            sys.path.append(dirname)
            wmod = imp.load_source(widgname, filename)
            try: # I have no idea, why we need this, but it seems to disappear sometimes?!
                sys.path.remove(dirname)
            except:
                pass
            widgClass = wmod.__dict__[widgname]
            inputClasses = set(eval(x[1], wmod.__dict__).__name__ for x in eval(inputList))
            outputClasses = set(y.__name__ for x in eval(outputList) for y in eval(x[1], wmod.__dict__).mro())
            
            widgetDesc = WidgetDescription(
                             name=name,
                             time=datetime,
                             filename=widgname,
                             fullname=filename,
                             inputList=inputList, outputList=outputList,
                             inputClasses=inputClasses, outputClasses=outputClasses
                             )
    
            for attr, deflt in (("contact>", "") , ("icon>", "icons/Unknown.png"), ("priority>", "5000"), ("description>", "")):
                istart, iend = data.find("<"+attr), data.find("</"+attr)
                setattr(widgetDesc, attr[:-1], istart >= 0 and iend >= 0 and data[istart+1+len(attr):iend].strip() or deflt)
        
            widgets.append(widgetDesc)
        except Exception, msg:
            if not hasErrors:
                print "The following widgets could not be scanned and will not be available"
                hasErrors = True 
            print "   %s: %s" % (widgname, msg)
        
    return widgets


re_tuple = re.compile(r"\(([^)]+)\)")

def getSignalList(regex, data):
    inmo = regex.search(data)
    if inmo:
        return str([tuple([y[0] in "'\"" and y[1:-1] or str(y) for y in (x.strip() for x in ttext.group(1).split(","))])
               for ttext in re_tuple.finditer(inmo.group("signals"))])
    else:
        return "[]"
