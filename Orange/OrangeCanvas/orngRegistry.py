# Author: Gregor Leban (gregor.leban@fri.uni-lj.si)
#

import os, sys, re, glob, stat
import pkg_resources
from orngSignalManager import OutputSignal, InputSignal, resolveSignal
from PyQt4.QtCore import *
from PyQt4.QtGui import *
import widgetParser

orangeDir = os.path.split(os.path.split(os.path.abspath(__file__))[0])[0]
if not orangeDir in sys.path:
    sys.path.append(orangeDir)

import orngEnviron, Orange.utils.addons
from Orange.utils import addons

WIDGETS_ENTRY_POINT = 'orange.widgets'

class WidgetDescription(object):
    def __init__(self, **attrs):
        self.__dict__.update(attrs)

    def docDir(self):
        if not self.addOn:  # A built-in widget
            dir, widgetDir = os.path.realpath(self.directory), os.path.realpath(orngEnviron.widgetDir)
            subDir = os.path.relpath(dir, widgetDir) if "relpath" in os.path.__dict__ else dir.replace(widgetDir, "")
            return os.path.join(orngEnviron.orangeDocDir, "widgets", subDir)
        else:  # An add-on widget
            return None  # new style add-ons only have on-line documentation
            #addOnDocDir = self.addOn.directory_documentation()
            #return os.path.join(addOnDocDir, "widgets")


class WidgetCategory(dict):
    def __init__(self, name, widgets=None):
        if widgets:
            self.update(widgets)
        self.name = name

def load_new_addons(directories = []):
    # New-type add-ons
    for entry_point in pkg_resources.iter_entry_points(WIDGETS_ENTRY_POINT):
        try:
            module = entry_point.load()
            if hasattr(module, '__path__'):
                # It is a package
                directories.append((entry_point.name, module.__path__[0], entry_point.name, False, module))
            else:
                # It is a module
                # TODO: Implement loading of widget modules
                # (This should be default way to load widgets, not parsing them as files, or traversing directories, just modules and packages (which load modules))
                pass
        except ImportError, err:
            print "While loading, importing widgets '%s' failed: %s" % (entry_point.name, err)
        except pkg_resources.DistributionNotFound, err:
            print "Loading add-on '%s' failed because of a missing dependency: '%s'" % (entry_point.name, err)
        except Exception, err:
            print "An exception occurred during the loading of '%s':\n%r" %(entry_point.name, err)
    return directories

def readCategories(silent=False):
    try:
        from Orange.version import version as orange_version
    except ImportError:
        # Orange.version module is writen by setup.py, what if orange was build
        # using make
        orange_version = "???"
    # Add orange version to the cache version (because cache contains names
    # of types inside the Orange hierarchy, if that changes the cache should be
    # invalidated)
    currentCacheVersion = (3, orange_version)
    
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
    except Exception:
        cachedWidgetDescriptions = {} 

    directories = [] # tuples (defaultCategory, dirName, plugin, isPrototype)
    for dirName in os.listdir(widgetDirName):
        directory = os.path.join(widgetDirName, dirName)
        if os.path.isdir(directory):
            directories.append((None, directory, None, "prototypes" in dirName.lower(), None))
            
    # read list of add-ons
    #TODO Load registered add-ons!

    load_new_addons(directories)

    categories = {}     
    for defCat, dirName, addOn, isPrototype, module in directories:
        widgets = readWidgets(dirName, cachedWidgetDescriptions, isPrototype, silent=silent, addOn=addOn, defaultCategory=defCat, module=module)
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

def readWidgets(directory, cachedWidgetDescriptions, prototype=False, silent=False, addOn=None, defaultCategory=None, module=None):
    import sys
    global hasErrors, splashWindow, widgetsWithError, widgetsWithErrorPrototypes
    
    widgets = []

    if not defaultCategory:
        predir, defaultCategory = os.path.split(directory.strip(os.path.sep).strip(os.path.altsep))
        if defaultCategory == "widgets":
            defaultCategory = os.path.split(predir.strip(os.path.sep).strip(os.path.altsep))[1]
    
    if defaultCategory.lower() == "prototypes" or prototype:
        defaultCategory = "Prototypes"
   
    if module:
        files = [f for f in pkg_resources.resource_listdir(module.__name__, '') if f.endswith('.py')]
    else:
        files = glob.iglob(os.path.join(directory, "*.py"))

    for filename in files:
        if module:
            if pkg_resources.resource_isdir(module.__name__, filename):
                continue
        else:
            if os.path.isdir(filename):
                continue
        
        if module:
            if getattr(module, '__loader__', None):
                datetime = str(os.stat(module.__loader__.archive)[stat.ST_MTIME])
            else:
                datetime = str(os.stat(pkg_resources.resource_filename(module.__name__, filename))[stat.ST_MTIME])
        else:
            datetime = str(os.stat(filename)[stat.ST_MTIME])
        cachedDescription = cachedWidgetDescriptions.get(filename, None)
        if cachedDescription and cachedDescription.time == datetime and hasattr(cachedDescription, "inputClasses"):
            widgets.append((cachedDescription.name, cachedDescription))
            continue
        if module:
            data = pkg_resources.resource_string(module.__name__, filename)
        else:
            data = file(filename).read()
        try:
            meta = widgetParser.WidgetMetaData(data, defaultCategory, enforceDefaultCategory=prototype)
        except:   # Probably not an Orange widget module.
            continue

        widgetPrototype = meta.prototype == "1" or meta.prototype.lower().strip() == "true" or prototype
        if widgetPrototype:
            meta.category = "Prototypes"

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

            if module:
                import_name = "%s.%s" % (module.__name__, widgname)
            else:
                import_name = widgname
            wmod = __import__(import_name, fromlist=[""])

            wmodFilename = wmod.__file__
            if os.path.splitext(wmodFilename)[1] != "py":
                # Replace .pyc, .pyo with bare .py extension
                # (used as key in cachedWidgetDescription)
                wmodFilename = os.path.splitext(wmodFilename)[0] + ".py"

            # Evaluate the input/output list (all tuple items are strings)
            inputs = eval(meta.inputList)
            outputs = eval(meta.outputList)

            inputs = [InputSignal(*input) for input in inputs]
            outputs = [OutputSignal(*output) for output in outputs]

            # Resolve signal type names into concrete type instances
            inputs = [resolveSignal(input, globals=wmod.__dict__)
                      for input in inputs]
            outputs = [resolveSignal(output, globals=wmod.__dict__)
                      for output in outputs]

            inputClasses = set([s.type.__name__ for s in inputs])
            outputClasses = set([klass.__name__ for s in outputs
                                 for klass in s.type.mro()])

            # Convert all signal types back into qualified names.
            # This is to prevent any possible import problems when cached
            # descriptions are unpickled (the relevant code using this lists
            # should be able to handle missing types better).
            for s in inputs + outputs:
                s.type = "%s.%s" % (s.type.__module__, s.type.__name__)

            widgetInfo = WidgetDescription(
                             name = meta.name,
                             time = datetime,
                             fileName = widgname,
                             module = module.__name__ if module else None,
                             fullName = wmodFilename,
                             directory = directory,
                             addOn = addOn,
                             inputList = meta.inputList, outputList = meta.outputList,
                             inputClasses = inputClasses, outputClasses = outputClasses,
                             tags=meta.tags,
                             inputs=inputs,
                             outputs=outputs,
                             )

            for attr in ["contact", "icon", "priority", "description", "category"]:
                setattr(widgetInfo, attr, getattr(meta, attr))

            # build the tooltip
            if len(widgetInfo.inputs) == 0:
                formatedInList = "<b>Inputs:</b><br> &nbsp;&nbsp; None<br>"
            else:
                formatedInList = "<b>Inputs:</b><br>"
                for signal in widgetInfo.inputs:
                    formatedInList += " &nbsp;&nbsp; - " + signal.name + " (" + signal.type + ")<br>"

            if len(widgetInfo.outputs) == 0:
                formatedOutList = "<b>Outputs:</b><br> &nbsp; &nbsp; None<br>"
            else:
                formatedOutList = "<b>Outputs:</b><br>"
                for signal in widgetInfo.outputs:
                    formatedOutList += " &nbsp; &nbsp; - " + signal.name + " (" + signal.type + ")<br>"

            addOnName = "" if not widgetInfo.addOn else " (from add-on %s)" % widgetInfo.addOn
    
            widgetInfo.tooltipText = "<b><b>&nbsp;%s</b></b>%s<hr><b>Description:</b><br>&nbsp;&nbsp;%s<hr>%s<hr>%s" % (meta.name, addOnName, widgetInfo.description, formatedInList[:-4], formatedOutList[:-4]) 
            widgets.append((meta.name, widgetInfo))
        except Exception, msg:
            if not hasErrors and not silent:
                print "There were problems importing the following widgets:"
                hasErrors = True
            if not silent:
                print "   %s: %s" % (widgname, msg)

            if not widgetPrototype:
                widgetsWithError.append(widgname)
            else:
                widgetsWithErrorPrototypes.append(widgname)
       
    return widgets
