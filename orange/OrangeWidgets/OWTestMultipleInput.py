#
# OWTestMultipleInput.py
#
# OW is an Orange Widget
# line 2
#

from OWWidget import *
from OWTestMultipleInputOptions import *
# from OWTestMultipleInputGraph import * #if using a graph

class OWTestMultipleInput(OWWidget):
    def __init__(self,parent=None):
        OWWidget.__init__(self,
        parent,
        "TestMultipleInput",
        """OW is an Orange Widget
line 2""",
        1,
        0)
        
        #set default settings
        #self.setSetting("setting","x")        
        #load settings
        self.loadSettings()
              
        #add a settings dialog (modify the automatically generated OWTestMultipleInputOptions dialog
        #by adding your setting controls, like checkboxes, radiobuttons and sliders
        #and don't forget tooltips)
        self.options=OWTestMultipleInputOptions()
        #set initial values in the options dialog
        #self.setOptions()

        #add your own initialization here
        
        #connect settingsbutton to show options
        self.connect(self.settingsButton,SIGNAL("clicked()"),self.options.show),
        
        #connect GUI controls of options in options dialog to settings
        #self.connect(self.options.control,SIGNAL("done()"),self.setControl) 
        #setControl is a function that calls setSetting() and applies the setting              
        
        #add all inputs
        self.addInput("data",FALSE)
        #and inputs from multiple sources
        #self.addinput("bla",FALSE)
        #add outputs
        #self.addoutput("bla")
        
        #GUI
        self.labelID=QLabel(self.mainArea)
        
        #give mainArea a layout (it hasn't got one so it as flexible as possible)
        #self.layout=QLayout(self.mainArea)
        #add your components here
        #self.x=QX(self.layout)
        
        #add controls to self.controlArea widget 
        #(it's a VBox, so just add them, they will appear in top to bottom order)
        #self.button=QPushButton("test",self.control)
        #connect controls to appropriate functions
        
        self.inputtables=[]
        
    def data(self,zdata,sourceID):
        self.inputtables.append(sourceID)
        self.labelID.setText(str(self.inputtables))
        
#test widget appearance        
if __name__=="__main__":
    a=QApplication(sys.argv)
    ow=OWTestMultipleInput()
    a.setMainWidget(ow)
#here you can test setting some stuff
    ow.show()
    a.exec_loop()
    
    #save settings 
    ow.saveSettings()

