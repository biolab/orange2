# Author: Gregor Leban (gregor.leban@fri.uni-lj.si)
#

import os, sys, re, glob, stat
from orngOrangeFoldersQt4 import *

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

    # there can be additional addons specified in additionalCategories file
    additionalFile = os.path.join(canvasSettingsDir, "additionalCategories")
    if os.path.exists(additionalFile):
        for lne in open(additionalFile, "rt"):
            try:
                catName, dirName = [x.strip() for x in lne.split("\t")]
                directories.append((catName, dirName, dirName))
            except:
                pass
            
    # there can also be addons in the orange/addons folder
    addonsPath = os.path.join(directoryNames["orangeDir"], "addOns")
    if os.path.exists(addonsPath):
        for dir in os.listdir(addonsPath):
            addon = os.path.join(addonsPath, dir)
            addonWidgets = os.path.join(addon, "widgets")
            if os.path.isdir(addon) and os.path.isdir(addonWidgets):
                directories.append((dir, addonWidgets, addonWidgets))
            
    categories = []
    for catName, dirName, plugin in directories:
        widgets = readWidgets(dirName, cachedWidgetDescriptions)
        if widgets:
            categories.append(WidgetCategory(catName, widgets, plugin and dirName or ""))
            if dirName not in sys.path:
                sys.path.insert(0, dirName)

    cPickle.dump(categories, file(cacheFilename, "wb"))
    storedCategories = categories
    return categories


re_inputs = re.compile(r'[ \t]+self.inputs\s*=\s*(?P<signals>\[[^]]*\])', re.DOTALL)
re_outputs = re.compile(r'[ \t]+self.outputs\s*=\s*(?P<signals>\[[^]]*\])', re.DOTALL)

def readWidgets(directory, cachedWidgetDescriptions):
    widgets = []
    for filename in glob.iglob(os.path.join(directory, "*.py")):
        if os.path.isdir(filename) or os.path.islink(filename):
            continue
        
        datetime = str(os.stat(filename)[stat.ST_MTIME])
        cachedDescription = cachedWidgetDescriptions.get(filename, None)
        if cachedDescription and cachedDescription.time == datetime:
            widgets.append(cachedDescription)
            continue
        
        data = file(filename).read()
        istart = data.find("<name>")
        if istart < 0:
            continue
        iend = data.find("</name>")
        if iend < 0:
            continue

        widgetDesc = WidgetDescription(
                         name=data[istart+6:iend],
                         time=datetime,
                         filename=os.path.splitext(os.path.split(filename)[1])[0],
                         fullname = filename,
                         inputList=getSignalList(re_inputs, data),
                         outputList=getSignalList(re_outputs, data)
                         )

        for attr, deflt in (("contact>", "") , ("icon>", "icons/Unknown.png"), ("priority>", "5000"), ("description>", "")):
            istart, iend = data.find("<"+attr), data.find("</"+attr)
            setattr(widgetDesc, attr[:-1], istart >= 0 and iend >= 0 and data[istart+1+len(attr):iend].strip() or deflt)
    
        widgets.append(widgetDesc)
        
    return widgets


re_tuple = re.compile(r"\(([^)]+)\)")

def getSignalList(regex, data):
    inmo = regex.search(data)
    if inmo:
        return str([tuple([y[0] in "'\"" and y[1:-1] or str(y) for y in (x.strip() for x in ttext.group(1).split(","))])
               for ttext in re_tuple.finditer(inmo.group("signals"))])
    else:
        return "[]"


def readAdditionalCategories():
    addCatFile = os.path.join(directoryNames["canvasSettingsDir"], "additionalCategories")
    if os.path.exists(addCatFile):
        return [tuple([x.strip() for x in lne.split("\t")]) for lne in file(addCatFile, "r")]
    else:
        return []

def writeAdditionalCategories(categories):
    file(os.path.join(directoryNames["canvasSettingsDir"], "additionalCategories"), "w").write("\n".join(["\t".join(l) for l in categories]))

def addWidgetCategory(category, directory, add = True):
    if os.path.isfile(directory):
        directory = os.path.dirname(directory)
    writeAdditionalCategories([x for x in readAdditionalCategories() if x[0] != category and x[1] != directory] + (add and [(category, directory)] or []))


if __name__=="__main__":
    readCategories()
