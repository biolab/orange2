"""
<name>Outliers</name>
<description>Indentification of outliers</description>
<icon>icons/Outliers.png</icon>
<contact>Marko Toplak (marko.toplak(@at@)gmail.com)</contact> 
<priority>2150</priority>
"""

from OWWidget import *
import OWGUI, orange
import orngOutlier
from exceptions import Exception

##############################################################################

class OWOutliers(OWWidget):
    settingsList = ["zscore", "metric", "k"]
    
    def __init__(self, parent=None, signalManager = None, name='Outlier'):
        OWWidget.__init__(self, parent, signalManager, name, wantMainArea = 0)

        self.inputs = [("Examples", ExampleTable, self.cdata),("Distance matrix", orange.SymMatrix, self.cdistance)]
        self.outputs = [("Outliers", ExampleTable), ("Inliers", ExampleTable), ("Examples with Z-scores", ExampleTable)]
               
        # Settings
        self.zscore = '4.0'
        self.k = 1
        self.metric = 0
        self.loadSettings()
        
        self.haveInput = 0
        self.data = None                    # input data set
        self.dataInput = None
        self.distanceMatrix = None
        
        kernelSizeValid = QDoubleValidator(self.controlArea)


        self.metrics = [("Euclidean", orange.ExamplesDistanceConstructor_Euclidean),
                        ("Manhattan", orange.ExamplesDistanceConstructor_Manhattan),
                        ("Hamming", orange.ExamplesDistanceConstructor_Hamming),
                        ("Relief", orange.ExamplesDistanceConstructor_Relief)]

        self.ks = [("All",0), ("1", 1), ("2",2), ("3",3), ("5",5), ("10",10), ("15",15)]
        items = [x[0] for x in self.metrics]
        itemsk = [x[0] for x in self.ks]
        
        OWGUI.comboBox(self.controlArea, self, "metric", box="Distance Metrics", items=items,
                       tooltip="Metrics to measure pairwise distance between data instances.",
                       callback=self.dataChange)

        OWGUI.comboBox(self.controlArea, self, "k", box="Nearest Neighbours", items=itemsk,
                       tooltip="Neighbours considered when computing the distance.",
                       callback=self.applySettings)

        OWGUI.separator(self.controlArea)
        box = OWGUI.widgetBox(self.controlArea, "Outliers")
        OWGUI.lineEdit(box, self, 'zscore',
                       label = 'Outlier Z:', labelWidth=80,
                       orientation='horizontal', # box=None,
                       validator = kernelSizeValid,
                       tooltip="Minimum Z-score of an outlier.",
                       callback=self.applySettings)

        OWGUI.separator(self.controlArea)
        
        self.loadSettings()
      
        self.resize(100,100)
        self.applySettings()

    def activateLoadedSettings(self):
        self.applySettings()

    def applySettings(self):              
        """use the setting from the widget, identify the outliers"""
        if self.haveInput == 1:
            outlier = self.outlier
            outlier.setKNN(self.ks[self.k][1])
       
            newdomain = orange.Domain(self.data.domain)
            newdomain.addmeta(orange.newmetaid(), orange.FloatVariable("Z score"))
            
            self.newdata = orange.ExampleTable(newdomain, self.data)

            zv = outlier.zValues()
            for i, el in enumerate(zv):
                self.newdata[i]["Z score"] = el            

            self.send("Examples with Z-scores", self.newdata)
            
            filterout = orange.Filter_values(domain=self.newdata.domain)
            filterout["Z score"] = (orange.Filter_values.Greater, eval(self.zscore))
            outliers = filterout(self.newdata)

            filterin = orange.Filter_values(domain=self.newdata.domain)
            filterin["Z score"] = (orange.Filter_values.LessEqual, eval(self.zscore))
            inliers = filterin(self.newdata)
            
            self.send("Outliers", outliers)
            self.send("Inliers", inliers)
        else:
            self.send("Examples with Z-scores", None)
            self.send("Outliers", None)
            self.send("Inliers", None)
  
    def cdata(self, data):
        """handles examples input signal"""
        self.dataInput = data
        self.dataChange()

    def cdistance(self, distances):
        """handles distance matrix input signal"""
        self.distanceMatrix = distances
        self.dataChange()
        
    def dataChange(self):
        self.haveInput = 0        
        outlier = orngOutlier.OutlierDetection()
        
        if self.distanceMatrix is not None:
            outlier.setDistanceMatrix(self.distanceMatrix) 
            self.data=getattr(self.distanceMatrix, "items")
            self.haveInput = 1     
        elif self.dataInput is not None:
            self.data = self.dataInput
            outlier.setExamples(self.data, self.metrics[self.metric][1](self.data))        
            self.haveInput = 1
        else:
            self.data = None
        self.outlier = outlier
        self.applySettings()

##############################################################################
# Test the widget, run from DOS prompt

if __name__=="__main__":
    a=QApplication(sys.argv)
    ow=OWOutliers()
    a.setMainWidget(ow)

    ow.show()
    a.exec_loop()
    ow.saveSettings()
