#
# OWWidget.py
# Orange Widget
# A General Orange Widget, from which all the Orange Widgets are derived
#

import sys
import os, os.path
import orange
from string import *
import cPickle
from OWTools import *
from OWAbout import *
from orngSignalManager import *
import time

class OWBaseWidget(QDialog):
    def __init__(
    self,
    parent = None,
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
        iconName = self.widgetDir + "icons/" + icon
        defaultWidgetIcon = self.widgetDir + "icons/OrangeWidgetsIcon.png"
        logo = self.widgetDir + "icons/OrangeWidgetsLogo.png"

        self.title = title.replace("&","")          # used for ini file
        self.captionTitle=title.replace("&","")     # used for widget caption

        # if we want the widget to show the title then the title must start with "Qt"
        if self.captionTitle[:2].upper() != "QT":
            self.captionTitle = "Qt " + self.captionTitle

        apply(QDialog.__init__, (self, parent, title, modal, Qt.WStyle_Customize + Qt.WStyle_NormalBorder + Qt.WStyle_Title + Qt.WStyle_SysMenu + Qt.WStyle_Minimize))

        # number of control signals, that are currently being processed
        # needed by signalWrapper to know when everything was sent
        #self.stackHeight = 0
        self.needProcessing = 0     # used by signalManager
        self.signalManager = None

        self.inputs = []     # signalName:(dataType, handler, onlySingleConnection)
        self.outputs = []    # signalName: dataType
        self.wrappers =[]    # stored wrappers for widget events
        self.linksIn = {}      # signalName : (dirty, widgetFrom, handler, signalData)
        self.linksOut = {}       # signalName: (signalData, id)
        self.connections = {}   # dictionary where keys are (control, signal) and values are wrapper instances. Used in connect/disconnect
        self.controledAttributes = []
        self.progressBarHandler = None  # handler for progress bar events
        self.callbackDeposit = []
        self.startTime = time.time()    # used in progressbar

    
        #the map with settings
        if not hasattr(self, 'settingsList'):
            self.__class__.settingsList = []

        #the title
        self.setCaption(self.captionTitle)
        
        self.buttonBackground=QVBox(self)
        if wantSettings: self.settingsButton=QPushButton("&Settings",self.buttonBackground)
        if wantGraph:    self.graphButton=QPushButton("&Save Graph",self.buttonBackground)

        #about box
        if wantAbout:
            self.about = OWAboutX(title, description, iconName, defaultWidgetIcon, logo)
            self.aboutButton=QPushButton("&About",self.buttonBackground)
            self.connect(self.aboutButton,SIGNAL("clicked()"),self.about.show)

    def setWidgetIcon(self, iconName):
        if os.path.exists(iconName):
            QDialog.setIcon(self, QPixmap(iconName))
        elif os.path.exists(self.widgetDir + iconName):
            QDialog.setIcon(self, QPixmap(self.widgetDir + iconName))
        elif os.path.exists(self.widgetDir + "icons/" + iconName):
            QDialog.setIcon(self, QPixmap(self.widgetDir + "icons/" + iconName))
        elif os.path.exists(self.widgetDir + "icons/Unknown.png"):
            QDialog.setIcon(self, QPixmap(self.widgetDir + "icons/Unknown.png"))

    def setCaptionTitle(self, caption):
        self.captionTitle = caption     # we have to save caption title in case progressbar will change it
        QDialog.setCaption(self, caption)
        
    # put this widget on top of all windows
    def reshow(self):
        self.hide()
        self.show()

    def send(self, signalName, value, id = None):
        self.linksOut[signalName] = (value, id)
        if self.signalManager:
            self.signalManager.send(self, signalName, value, id)
        else:
            signalManager.send(self, signalName, value, id)

    # Set all settings
    # settings - the map with the settings       
    def setSettings(self,settings):
        for key in settings:
            self.__setattr__(key, settings[key])
        #self.__dict__.update(settings)

    # Get all settings
    # returns map with all settings
    def getSettings(self):
        return dict([(x, getattr(self, x, None)) for x in settingsList])

    # Loads settings from the widget's .ini file 
    def loadSettings(self, file = None):
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

    # Loads settings from string str which is compatible with cPickle
    def loadSettingsStr(self, str):
        if str == None: return
        if hasattr(self, "settingsList"):
            settings = cPickle.loads(str)
            self.setSettings(settings)

    # return settings in string format compatible with cPickle
    def saveSettingsStr(self):
        str = ""
        if hasattr(self, "settingsList"):
            settings = dict([(name, getattr(self, name)) for name in self.settingsList])
            str = cPickle.dumps(settings)
        return str

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
        handler = None
        for i in self.inputs:
            input = InputSignal(*i)
            if signalName == input.name: handler = input.handler
            
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
        if hasattr(QDialog, "__setattr__"): QDialog.__setattr__(self, name, value)  # for linux and mac platforms
        else:                               self.__dict__[name] = value             # for windows platform
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
