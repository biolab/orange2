# coding=utf-8
import os, sys, user
from zipfile import ZipFile
import urllib

if os.name == "nt":
    paths = os.environ["PATH"].split(";")
    paths.sort(lambda x,y: -1 if "PyQt4" in x else (1 if "miktex" in y and os.path.exists(os.path.join(y, "QtCore4.dll")) else 0))
    os.environ["PATH"] = ";".join(paths)
    
if sys.platform == "darwin" and sys.prefix.startswith("/sw"):
    sys.path.append(os.path.join(sys.prefix, "lib/qt4-mac/lib/python" + sys.version[:3] + "/site-packages")) 

def __getDirectoryNames():
    """Return a dictionary with Orange directories."""
    try:
        orangeDir = os.path.split(os.path.abspath(__file__))[0]
    except:
        import orange
        orangeDir = os.path.split(os.path.abspath(orange.__file__))[0]

    orangeDocDir = os.path.join(orangeDir, "doc")
    #TODO This might be redefined in orngConfiguration.

    try:
        orangeVer = orangeDir.split(os.path.sep)[-1]
    except:
        orangeVer = "orange"

    canvasDir = os.path.join(orangeDir, "OrangeCanvas")
    widgetDir = os.path.join(orangeDir, "OrangeWidgets")
    picsDir = os.path.join(widgetDir, "icons")
    addOnsDirSys = os.path.join(orangeDir, "add-ons")

    if not os.path.isdir(widgetDir) or not os.path.isdir(widgetDir):
        canvasDir = None
        widgetDir = None
    if not os.path.isdir(picsDir):
        picsDir = ""

    home = user.home
    if home[-1] == ":":
        home += "\\"
    if os.name == "nt":
        applicationDir = os.path.join(home, "Application Data")
        if not os.path.isdir(applicationDir):
            try: os.makedirs(applicationDir)
            except: pass
        outputDir = os.path.join(applicationDir, orangeVer)                  # directory for saving settings and stuff
        defaultReportsDir = os.path.join(home, "My Documents", "Orange Reports")
    elif sys.platform == "darwin":
        applicationDir = os.path.join(home, "Library", "Application Support")
        if not os.path.isdir(applicationDir):
            try: os.makedirs(applicationDir)
            except: pass
        outputDir = os.path.join(applicationDir, orangeVer)
        defaultReportsDir = os.path.join(home, "Library/Application Support/orange/Reports")
    else:
        outputDir = os.path.join(home, "."+orangeVer)                  # directory for saving settings and stuff
        defaultReportsDir = os.path.join(home, "orange-reports")

    addOnsDirUser = os.path.join(outputDir, "add-ons")

    orangeSettingsDir = outputDir
    if sys.platform == "darwin":
        bufferDir = os.path.join(home, "Library")
        bufferDir = os.path.join(bufferDir, "Caches")
        bufferDir = os.path.join(bufferDir, orangeVer)
    else:
        bufferDir = os.path.join(outputDir, "buffer")
    canvasSettingsDir = os.path.join(outputDir, "OrangeCanvasQt4") if canvasDir <> None else None
    widgetSettingsDir = os.path.join(outputDir, "widgetSettingsQt4") if widgetDir <> None else None

    for dname in [orangeSettingsDir, bufferDir, widgetSettingsDir, canvasSettingsDir, defaultReportsDir]:
        if dname <> None and not os.path.isdir(dname):
            try: os.makedirs(dname)        # Vista has roaming profiles that will say that this folder does not exist and will then fail to create it, because it exists...
            except: pass

    return dict([(name, vars()[name]) for name in ["orangeDir", "orangeDocDir", "canvasDir", "widgetDir", "picsDir", "addOnsDirSys", "addOnsDirUser", "defaultReportsDir", "orangeSettingsDir", "widgetSettingsDir", "canvasSettingsDir", "bufferDir"]])

def samepath(path1, path2):
    return os.path.normcase(os.path.normpath(path1)) == os.path.normcase(os.path.normpath(path2))

def addOrangeDirectoriesToPath():
    """Add orange directory paths to Python path."""
    pathsToAdd = [orangeDir]

    if canvasDir <> None:
        pathsToAdd.append(canvasDir)

    if widgetDir <> None and os.path.isdir(widgetDir):
        pathsToAdd.append(widgetDir)
        defaultWidgetsDirs = [os.path.join(widgetDir, x) for x in os.listdir(widgetDir) if os.path.isdir(os.path.join(widgetDir, x))]
        pathsToAdd.extend(defaultWidgetsDirs)

    for path in pathsToAdd:
        if os.path.isdir(path) and not any([samepath(path, x) for x in sys.path]):
            sys.path.insert(0, path)

directoryNames = __getDirectoryNames()
globals().update(directoryNames)

addOrangeDirectoriesToPath()