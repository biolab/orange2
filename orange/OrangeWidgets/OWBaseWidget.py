#
# OWWidget.py
# Orange Widget
# A General Orange Widget, from which all the Orange Widgets are derived
#

import sys
import ConfigParser, os, os.path
import orange
from string import *
import cPickle
from OWTools import *
from OWAboutX import *
from orngSignalManager import *
import time


class ExampleTable(orange.ExampleTable):
    pass

class ExampleTableWithClass(ExampleTable):
    pass

class SignalWrapper:
    def __init__(self, widget, method):
        self.widget = widget
        self.method = method

    def __call__(self, *k):
        signalManager.signalProcessingInProgress += 1
        apply(self.method, k)
        signalManager.signalProcessingInProgress -= 1
        if not signalManager.signalProcessingInProgress:
            signalManager.processNewSignals(self.widget)


class OWBaseWidget(QDialog):
    def __init__(
    self,
    parent=None,
    title="Qt Orange BaseWidget",
    description="This a base for OWWidget. It encorporates saving, loading settings and signal processing.",
    wantSettings=FALSE,
    wantGraph = FALSE, 
    wantAbout = FALSE,
    icon="OrangeWidgetsIcon.png",
    logo="OrangeWidgetsLogo.png",
    modal=FALSE
    ):
        """
        Initialization
        Parameters: 
            title - The title of the\ widget, including a "&" (for shortcut in about box)
            description - self explanatory
            wantSettings - display settings button or not
            icon - the icon file
            logo - the logo file
        """
        # directories are better defined this way, otherwise .ini files get written in many places
        self.widgetDir = os.path.dirname(__file__) + "/"
        fullIcon = self.widgetDir + "icons/" + icon 
        logo = self.widgetDir + "icons/" + logo

        self.title = title.replace("&","")          # used for ini file
        self.captionTitle=title.replace("&","")     # used for widget caption

        # if we want the widget to show the title then the title must start with "Qt"
        if self.captionTitle[:2].upper != "QT":
            self.captionTitle = "Qt " + self.captionTitle

        apply(QDialog.__init__, (self, parent, title, modal, Qt.WStyle_Customize + Qt.WStyle_NormalBorder + Qt.WStyle_Title + Qt.WStyle_SysMenu + Qt.WStyle_Minimize))

        # number of control signals, that are currently being processed
        # needed by signalWrapper to know when everything was sent
        #self.stackHeight = 0
        self.needProcessing = 0     # used by signalManager

        self.inputs = []     # signalName:(dataType, handler, onlySingleConnection)
        self.outputs = []    # signalName: dataType
        self.wrappers =[]    # stored wrappers for widget events
        self.linksIn = {}      # signalName : (dirty, widgetFrom, handler, signalData)
        self.linksOut = {}       # signalName: (signalData, id)
        self.controledAttributes = []
        self.progressBarHandler = None  # handler for progress bar events
        self.callbackDeposit = []
        self.startTime = time.time()    # used in progressbar

    
        #the map with settings
        if not hasattr(self, 'settingsList'):
            self.__class__.settingsList = []

        #the title
        self.setCaption(self.captionTitle)
        self.setIcon(QPixmap(fullIcon))

        #about box
        self.about=OWAboutX(title,description,fullIcon,logo)
        self.buttonBackground=QVBox(self)
        if wantSettings: self.settingsButton=QPushButton("&Settings",self.buttonBackground)
        if wantGraph:    self.graphButton=QPushButton("&Save Graph",self.buttonBackground)
        if wantAbout:
            self.aboutButton=QPushButton("&About",self.buttonBackground)
            self.connect(self.aboutButton,SIGNAL("clicked()"),self.about.show)


    def setCaptionTitle(self, caption):
        self.captionTitle = caption     # we have to save caption title in case progressbar will change it
        QDialog.setCaption(self, caption)
        
    # put this widget on top of all windows
    def reshow(self):
        self.hide()
        self.show()

    def send(self, signalName, value, id = None):
        self.linksOut[signalName] = (value, id)
        signalManager.send(self, signalName, value, id)

    # Set all settings
    # settings - the map with the settings       
    def setSettings(self,settings):
        for key in settings:
            self.__setattr__(key, settings[key])
        #self.__dict__.update(settings)

    def getSettings(self):
        """
        Get all settings
        returns map with all settings
        """
        return dict([(x, getattr(self, x, None)) for x in settingsList])
   
    def loadSettings(self, file = None):
        """
        Loads settings from the widget's .ini file
        """
        if hasattr(self, "settingsList"):
            if file==None:
                if os.path.exists(self.widgetDir + "widgetSettings/" + self.title + ".ini"):
                    file = self.widgetDir + "widgetSettings/" + self.title + ".ini"
                elif os.path.exists(self.widgetDir + self.title + ".ini"):
                    file = self.widgetDir + self.title + ".ini"
                else:
                    return
            if type(file) == str:
                if os.path.exists(file):
                    file = open(file, "r")
                    settings = cPickle.load(file)
                    file.close()
                else:
                    settings = {}
            else:
                settings = cPickle.load(file)
            
            self.setSettings(settings)

        
    def saveSettings(self, file = None):
        if hasattr(self, "settingsList"):
            settings = dict([(name, getattr(self, name)) for name in self.settingsList])
            if file==None:
                if not os.path.exists(self.widgetDir + "widgetSettings/"):
                    os.mkdir(self.widgetDir + "widgetSettings")
                file = self.widgetDir + "widgetSettings/" + self.title + ".ini"
            if type(file) == str:
                file = open(file, "w")
                cPickle.dump(settings, file)
                file.close()
            else:
                cPickle.dump(settings, file)

    def loadSettingsStr(self, str):
        """
        Loads settings from string str which is compatible with cPickle
        """
        if str == None: return
        if hasattr(self, "settingsList"):
            settings = cPickle.loads(str)
            self.setSettings(settings)

    def saveSettingsStr(self):
        """
        return settings in string format compatible with cPickle
        """
        str = ""
        if hasattr(self, "settingsList"):
            settings = dict([(name, getattr(self, name)) for name in self.settingsList])
            str = cPickle.dumps(settings)
        return str

    # this function is only intended for derived classes to send appropriate signals when all settings are loaded
    def activateLoadedSettings(self):
        pass
        
    def addInput(self,signalName, dataType, handler, onlySingleConnection=TRUE):
        self.inputs.append((signalName, dataType, handler, onlySingleConnection))
            
    def addOutput(self, signalName, dataType):
        self.outputs.append((signalName, dataType))

    def setOptions(self):
        pass

    # ########################################################################
    def connect(self, control, signal, method):
        wrapper = SignalWrapper(self, method)
        self.wrappers.append(wrapper)
        QDialog.connect(control, signal, wrapper)
        #QWidget.connect(control, signal, method)        # ordinary connection useful for dialogs and windows that don't send signals to other widgets

    def findSignalTypeFrom(self, signalName):
        for (signal, dataType) in self.outputs:
            if signal == signalName: return dataType
        return dataType 

    def findSignalTypeTo(self, signalName):
        for (signal, dataType, handler, onlySingleConnection) in self.inputs:
            if signalName == signal: return dataType
        return None

    def signalIsOnlySingleConnection(self, signalName):
        for (signal, dataType, handler, onlySingleConnection) in self.inputs:
            if signal == signalName: return onlySingleConnection

    def addInputConnection(self, widgetFrom, signalName):
        handler = None
        for (signal, dataType, h, onlySingle) in self.inputs:
            if signalName == signal: handler = h
            
        existing = []
        if self.linksIn.has_key(signalName): existing = self.linksIn[signalName]
        self.linksIn[signalName] = existing + [(0, widgetFrom, handler, None, None)]    # (dirty, handler, signalData, idValue)

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
        #(type, handler, single) = self.inputs[signal]
        #if not single: return []
        for (signalName, dataType, handler, onlySingle) in self.inputs:
            if signalName == signal and not onlySingle: return None
            
        for signalName in self.linksIn.keys():
            if signalName == signal:
                widget = self.linksIn[signalName][0][1]
                del self.linksIn[signalName]
                return widget
               
        return None
        
    # signal manager calls this function when all input signals have updated the data
    def processSignals(self):
        #if self.stackHeight > 0: return  # if this widet is already processing something return
        
        # we define only a way to handle signals that have defined a handler function
        for key in self.linksIn.keys():
            for i in range(len(self.linksIn[key])):
                (dirty, widgetFrom, handler, signalData, idValue) = self.linksIn[key][i]
                if not (handler and dirty): continue
                    
                self.linksIn[key][i] = (0, widgetFrom, handler, signalData, idValue) # clear the dirty flag
                if self.signalIsOnlySingleConnection(key):
                    handler(signalData)
                else:
                    # if one widget sends signal using send("signal name", value, id), where id != None.
                    # this is used in cases where one widget sends more signals of same "signal name"
                    if idValue: handler(signalData, (widgetFrom, idValue))
                    else:       handler(signalData, widgetFrom)

        self.needProcessing = 0

    # set new data from widget widgetFrom for a signal with name signalName
    def updateNewSignalData(self, widgetFrom, signalName, value, id):
        if not self.linksIn.has_key(signalName): return
        for i in range(len(self.linksIn[signalName])):
            (dirty, widget, handler, oldValue, idValue) = self.linksIn[signalName][i]
            if widget == widgetFrom and idValue == id:
                self.linksIn[signalName][i] = (1, widget, handler, value, id)
        self.needProcessing = 1


    # ############################################
    # PROGRESS BAR FUNCTIONS
    def progressBarInit(self):
        self.startTime = time.time()
        self.setCaption(self.captionTitle + " (0% complete)")
        if self.progressBarHandler: self.progressBarHandler(self, -1)
        
    def progressBarSet(self, value):
        if value > 0:
            diff = time.time() - self.startTime
            total = diff * 100.0/float(value)
            remaining = max(total - diff, 0)
            h = int(remaining/3600)
            min = int((remaining - h*3600)/60)
            sec = int(remaining - h*3600 - min*60)
            text = ""
            if h > 0: text += "%d h, " % (h)
            text += "%d min, %d sec" %(min, sec)
            self.setCaption(self.captionTitle + " (%.1f%% complete, remaining time: %s)" % (value, text))
        else:
            self.setCaption(self.captionTitle + " (0% complete)" )
        if self.progressBarHandler: self.progressBarHandler(self, value)

    def progressBarFinished(self):
        self.setCaption(self.captionTitle)
        if self.progressBarHandler: self.progressBarHandler(self, 101)

    # handler must be a function, that receives 2 arguments. First is the widget instance, the second is the value between -1 and 101
    def progressBarSetHandler(self, handler):
        self.progressBarHandler = handler

    def __setattr__(self, name, value):
        self.__dict__[name] = value
        if hasattr(self, "controledAttributes"):
            for attrname, func in self.controledAttributes:
                if attrname == name:
                    func(value)
                    return

    
if __name__ == "__main__":  
    a=QApplication(sys.argv)
    oww=OWBaseWidget()
    a.setMainWidget(oww)
    oww.show()
    a.exec_loop()
    oww.saveSettings()
