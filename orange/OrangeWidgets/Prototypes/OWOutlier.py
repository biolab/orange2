"""
<name>Outlier detection</name>
<description>Outlier detection</description>
<icon>icons/Outlier.png</icon>
<contact>Marko Toplak (marko.toplak(@at@)gmail.com)</contact>
<priority>4010</priority>
"""

from OWWidget import *
import OWGUI, orange
import orngOutlier
from exceptions import Exception

##############################################################################

class OWOutlier(OWWidget):
    settingsList = ["zscore","metric","k"]

    def __init__(self, parent=None, signalManager = None, name='Outlier'):
        OWWidget.__init__(self, parent, signalManager, name)

        self.inputs = [("Examples", ExampleTable, self.cdata),("Distance matrix", orange.SymMatrix, self.cdistance)]
        self.outputs = [("Outliers", ExampleTable),("Examples with Z-scores", ExampleTable)]

        # Settings
        self.name = 'Outlier detection'           # name of the classifier/learner
        self.zscore = '4.0'

        self.haveInput = 0

        self.data = None                    # input data set
        self.dataInput = None
        self.distanceMatrix = None
        self.loadSettings()

        kernelSizeValid = QDoubleValidator(self.controlArea)

        self.metric = 0

        self.metrics = [("Euclidean", orange.ExamplesDistanceConstructor_Euclidean),
                        ("Manhattan", orange.ExamplesDistanceConstructor_Manhattan),
                        ("Hamming", orange.ExamplesDistanceConstructor_Hamming),
                        ("Relief", orange.ExamplesDistanceConstructor_Relief)]


        self.k = 0

        self.ks = [("All",0),("1", 1), ("2",2), ("3",3), ("5",5), ("10",10),("15",15)]

        items = [x[0] for x in self.metrics]

        itemsk = [x[0] for x in self.ks]

        OWGUI.comboBox(self.controlArea, self, "metric", box="Distance Metrics", items=items,
                       tooltip="Choose metrics to measure pairwise distance between examples.",
                       callback=self.dataChange)

        OWGUI.comboBox(self.controlArea, self, "k", box="Nearest Neighbours", items=itemsk,
                       tooltip="Choose how many neighbours are considered when analysing data.",
                       callback=self.applySettings)



        box = QVGroupBox(self.controlArea)
        box.setTitle('Settings')

        OWGUI.lineEdit(box, self, 'zscore', label = 'Outlier Z:', orientation='horizontal', box=None, tooltip='Minimum Z score of an outlier', callback=None, valueType = str, validator = kernelSizeValid)

        OWGUI.separator(self.controlArea)

        self.applyBtn = OWGUI.button(box, self, "&Apply", callback=self.applySettings)

        self.resize(100,100)
        self.applySettings()

    def activateLoadedSettings(self):
        self.applySettings()

    # setup the bayesian learner
    def applySettings(self):

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

            filter = orange.Filter_values(domain=self.newdata.domain)
            filter["Z score"] = (orange.Filter_values.Greater, eval(self.zscore))
            self.outliers = filter(self.newdata)

            self.send("Outliers", self.outliers)
        else:
            self.send("Examples with Z-scores", None)
            self.send("Outliers", None)



    # handles examples input signal
    def cdata(self, data):
        self.dataInput = data
        self.dataChange()

    #handles distance matrix input signal
    def cdistance(self, distances):
        self.distanceMatrix = distances
        self.dataChange()

    def dataChange(self):

        self.haveInput = 0

        outlier = orngOutlier.OutlierDetection()

        if self.distanceMatrix <> None:
            outlier.setDistanceMatrix(self.distanceMatrix)
            self.data=getattr(self.distanceMatrix, "items")
            self.haveInput = 1
        else:
          if self.dataInput <> None:
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
    ow=OWOutlier()

    ow.show()
    a.exec_()
    ow.saveSettings()
