#
# OWTestMultipleInputOptions.py
#
# options dialog for distributions graph
#

from OWOptions import *

class OWTestMultipleInputOptions(OWOptions):
    def __init__(self,parent=None,name=None):
        OWOptions.__init__(self,"TestMultipleInput Options","OrangeWidgetsIcon.png",parent,name)  
        #add your controls here      
               
if __name__=="__main__":
    a=QApplication(sys.argv)
    w=OWTestMultipleInputOptions()
    a.setMainWidget(w)
    w.show()
    a.exec_loop()

