#
# OWWidget.py
# Orange Widget
# A General Orange Widget, from which all the Orange Widgets are derived
#

import sys
import ConfigParser,os
from string import *
import cPickle
from OWTools import *
from OWAboutX import *

class OWWidget(QWidget):
    def __init__(
    self,
    parent=None,
    title="O&range Widget",
    description="This a general Orange Widget\n from which all the other Orange Widgets are derived.",
    wantSettings=FALSE,
    wantGraph=FALSE,
    icon="OrangeWidgetsIcon.png",
    logo="OrangeWidgetsLogo.png",
    ):
        """
        Initialization
        Parameters: 
            title - The title of the\ widget, including a "&" (for shortcut in about box)
            description - self explanatory
            wantSettings - display settings button or not
            wantGraph - displays a save graph button or not
            icon - the icon file
            logo - the logo file
            parent - parent of the widget if needed
        """
        icon = sys.prefix + "/lib/site-packages/Orange/OrangeWidgets/icons/" + icon
        logo = sys.prefix + "/lib/site-packages/Orange/OrangeWidgets/icons/" + logo
        self.widgetDir = sys.prefix + "/lib/site-packages/Orange/OrangeWidgets/"
        self.title=title.replace("&","")
        QWidget.__init__(self,parent)
        #list of all active connections to this widget
        self.connections=[]
        #list of inputs - should list all the channels that can be received
        self.inputs=[]
        self.multipleInputs=[]
        #list of outputs - should list all the channels that can be emited
        self.outputs=[]
        #the map with settings
        if not hasattr(self, 'settingsList'):
            type(self).settingsList = []
        #is widget enabled?
        self.enabled=TRUE
        #the title
        self.setCaption(self.title)
        self.setIcon(QPixmap(icon))
        #about box
        self.about=OWAboutX(title,description,icon,logo)
        self.buttonBackground=QVBox(self)
        self.aboutButton=QPushButton("&About",self.buttonBackground)
        if wantSettings:
            self.settingsButton=QPushButton("&Settings",self.buttonBackground)
        if wantGraph:
            self.graphButton=QPushButton("&Save Graph",self.buttonBackground)
        if parent==None:
            self.closeButton=QPushButton("&Close",self.buttonBackground)
        self.connect(self.aboutButton,SIGNAL("clicked()"),self.about.show)
        if parent==None:
            self.connect(self.closeButton,SIGNAL("clicked()"),self.close)
        self.mainArea=QWidget(self)
#        self.mainArea.setBackgroundColor(Qt.white)
        self.controlArea=QVBox(self)
        self.space=QVBox(self)
#        self.controlArea.setBackgroundColor(Qt.white)
        self.grid=QGridLayout(self,3,2,5)
        self.grid.addWidget(self.buttonBackground,2,0)
        self.grid.addWidget(self.space,1,0)
        self.grid.setRowStretch(1,10)
        self.grid.setColStretch(1,10)
        self.grid.addWidget(self.controlArea,0,0)
        self.grid.addMultiCellWidget(self.mainArea,0,3,1,1)
        self.resize(640,480)
#        self.connect(self,SIGNAL("exit"),self.saveSettings)
        self.linkBuffer={}
        
    def link(self,source,channel):
        """
        Link a widget to this widget
        Parameters:
            source - source widget
            cnannel - name of the channel and the function it connects to
        """
        #what to do with those tree:classifier signals?!
        
        #check if this channel exist
        if channel not in self.inputs and channel not in self.multipleInputs:
            return
        if channel not in self.connections:
            self.connections.append([channel,source])
            #there should be a function with the same name as the signal 
            #in the destination class, otherwise this won't work!
            #We could check if the destination functions exists.. to be done.
            if self.enabled:                                
#                print "self."+channel
                if (channel in self.inputs or channel in self.multipleInputs):
                    self.connect(source,PYSIGNAL(channel),self.receive)
                    source.resend(channel)
                else:
                    print "Cannot link: the channel %s does not exist" % channel
        
    def unlink(self,source,channel):
        """
        Destroy link from a widget to this widget
        Parameters:
            source - source widget
            cnannel - name of the channel and the function it connects to
        """
        try:
            self.connections.remove([channel,source])
#            self.disconnect(source,PYSIGNAL(channel),eval("self."+channel))
            self.disconnect(source,PYSIGNAL(channel),self.receive)
        except:
            pass
            
    def send(self,channel,data):
        """
        Send data over a channel
            channel - the channel name
            data - the data to be sent, must be a tuple!
        Saves some typing compared to QObject.emit. 
        Only sends if connected.
        """
#        print "sending", data, "over channel", channel
#        self.emit(PYSIGNAL(channel),(data,))
        #save the data for later
        self.linkBuffer[channel]=data
        self.emit(PYSIGNAL(channel),(data,channel,id(self)))
        
    def resend(self,channel):
        """
        resends the data that was last sent through the channel
        """
        if channel in self.linkBuffer:
            self.send(channel,self.linkBuffer[channel])
    
    def receive(self,zdata,channel,source):
        """
        Receives data over a channel. Passes it to the right function
            data - the data
            channel - the name of the channel
            source - the source of the channel
        """      
        if channel in self.inputs:
            eval("self."+channel)(zdata)
        elif channel in self.multipleInputs:
            eval("self."+channel)(zdata,source)
        else:
            print "Error: this channel does not exist as input!" #impossible
    
    def getConnections(self):
        """
        Gets all connections
        Returns the list with all connections
        """
        return self.connections
        
    def setEnabled(self,enable):
        """
        Set this widget in enabled or disabled state
        enable: TRUE - widget is enabled 
                FALSE - widget is disabled
        """
        #setEnabled(FALSE) removes all links in connections
        #setEnabled(TRUE) restores all this links
        #we'll see if this works
        self.enabled=enable
        for connection in self.connections:
            if enabled:
#                self.connect(connection[0],PYSIGNAL(connection[1]),eval("self."+connection[1]))
                self.connect(connection[0],PYSIGNAL(connection[1]),self.receive)
                #resend in case something went wrong
                self.resend(connection[1],connection[0])
            else:
#                self.disconnect(connection[0],PYSIGNAL(connection[1]),eval("self."+connection[1]))
                self.disconnect(connection[0],PYSIGNAL(connection[1]),self.receive)
        
#    def setSetting(self,name,value):
#        """
#        Set a setting
#        name - the name that uniquely identifies the setting
#        value - the value of the setting
#        """
#        if not getattr(self, "settingsList", None):
#            self.settingsList = [name]
#        if not name in self.settingsList:
#            self.settingsList.append(name)
#        setattr(self, name, value)
        
#    def getSetting(self,name):
#        """
#        Get a setting 
#        name - the name that uniquely identifies the setting
#        returns the setting
#        """
#        return getattr(self, name, None)
    
    def setSettings(self,settings):
        """
        Set all settings
        settings - the map with the settings
        """
        self.__dict__.update(settings)

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
                file = self.widgetDir + self.title + ".ini"
            if type(file) == str:
                if os.path.exists(file):
                    file = open(file, "r")
                    settings = cPickle.load(file)
                    file.close()
                else:
                    settings = {}
            else:
                settings = cPickle.load(file)
            
            self.__dict__.update(settings)

        
    def saveSettings(self, file = None):
        if hasattr(self, "settingsList"):
            settings = dict([(name, getattr(self, name)) for name in self.settingsList])
            if file==None:
                file = self.widgetDir + self.title + ".ini"                
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
        if hasattr(self, "settingsList"):
            settings = cPickle.loads(str)
            self.__dict__.update(settings)

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
        
    def addInput(self,inpu,singleSignal=TRUE):
        """
        Adds an input.
        Should be called for all functions that can be inputs (link destinations).
        """
        if singleSignal:
            self.inputs.append(inpu)
        else:
            self.multipleInputs.append(inpu)
    
    def addOutput(self,output):
        """
        Adds an output.
        Should be called for all the outputs (link sources) that can be generated.
        """
        self.outputs.append(output)
        
    def getInputs(self):
        """
        Get the names of all inputs
        """
        return self.inputs
        
    def getOutputs(self):
        """
        Get the names of all outputs
        """
        return self.outputs

if __name__ == "__main__":  
    a=QApplication(sys.argv)
    oww=OWWidget()
#    oww.setSetting("gk",[43,23])
#    oww.setSettings(eval('{"one":1,"two":2}')) # this is good.. can read a line and set it easily
#    print oww.getSettings()
    a.setMainWidget(oww)
    oww.show()
    a.exec_loop()
    oww.saveSettings()
