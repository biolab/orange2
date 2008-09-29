import os, sys, user

def __getDirectoryNames():
    try:
        canvasDir = os.path.split(os.path.abspath(__file__))[0]
        orangeDir = canvasDir[:-13]
    except:
        import orange
        orangeDir = os.path.split(os.path.abspath(orange.__file__))[0]
        canvasDir = os.path.join(orangeDir, "OrangeCanvas")

    try:
        orangeVer = orangeDir.split(os.path.sep)[-1]
    except:
        orangeVer = "orange"

    widgetDir = os.path.join(orangeDir, "OrangeWidgets")
    if not os.path.exists(widgetDir):
        print "Error. Directory %s not found. Unable to locate widgets." % widgetDir

    reportsDir = os.path.join(orangeDir, "report")
    if not os.path.exists(reportsDir):
        try: os.mkdir(reportsDir)        # Vista has roaming profiles that will say that this folder does not exist and will then fail to create it, because it exists...
        except: pass

    picsDir = os.path.join(widgetDir, "icons")
    if not os.path.exists(picsDir):
        print "Error. Directory %s not found. Unable to locate widget icons." % picsDir

    home = user.home
    if home[-1] == ":":
        home += "\\"
    if os.name == "nt":
        applicationDir = os.path.join(home, "Application Data")
        if not os.path.exists(applicationDir):
            try: os.mkdir(applicationDir)
            except: pass
        outputDir = os.path.join(applicationDir, orangeVer)                  # directory for saving settings and stuff
    elif sys.platform == "darwin":
        applicationDir = os.path.join(home, "Library")
        applicationDir = os.path.join(applicationDir, "Application Support")
        outputDir = os.path.join(applicationDir, orangeVer)
    else:
        outputDir = os.path.join(home, "."+orangeVer)                  # directory for saving settings and stuff

    if not os.path.exists(outputDir):
        try: os.mkdir(outputDir)        # Vista has roaming profiles that will say that this folder does not exist and will then fail to create it, because it exists...
        except: pass
        
    bufferDir = os.path.join(outputDir, "buffer")
    if not os.path.exists(bufferDir):
        try: os.mkdir(bufferDir)        # Vista has roaming profiles that will say that this folder does not exist and will then fail to create it, because it exists...
        except: pass

    widgetSettingsDir = os.path.join(outputDir, "widgetSettingsQt4")
    if not os.path.exists(widgetSettingsDir):
        try: os.mkdir(widgetSettingsDir)        # Vista has roaming profiles that will say that this folder does not exist and will then fail to create it, because it exists...
        except: pass

    canvasSettingsDir = os.path.join(outputDir, "OrangeCanvasQt4")
    if not os.path.exists(canvasSettingsDir):
        try: os.mkdir(canvasSettingsDir)        # Vista has roaming profiles that will say that this folder does not exist and will then fail to create it, because it exists...
        except: pass


    return dict([(name, vars()[name]) for name in ["canvasDir", "orangeDir", "widgetDir", "reportsDir", "picsDir", "widgetSettingsDir", "canvasSettingsDir", "bufferDir"]])


def addOrangeDirectoriesToPath():
    sys.path = [x for x in sys.path if "orange\\orange" not in x.lower() and "orange/orange" not in x.lower()]
    orangeDir = directoryNames["orangeDir"]
    widgetDir = directoryNames["widgetDir"]
    canvasDir = directoryNames["canvasDir"]
    sys.path.insert(0, canvasDir)
    if orangeDir not in sys.path:
        sys.path.insert(0, orangeDir)
    if widgetDir not in sys.path:
        sys.path.insert(0, widgetDir)

directoryNames = __getDirectoryNames()
#vars().update(directoryNames)
addOrangeDirectoriesToPath()
