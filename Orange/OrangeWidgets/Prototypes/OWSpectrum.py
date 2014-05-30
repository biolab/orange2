"""<name>Spectrum</name>
<description>Helps remove features with too many missing values</description>
<icon>icons/Spectrum.svg</icon>
<priority>30</priority>
<contact>Janez Demsar (janez.demsar@fri.uni-lj.si)</contact>"""

from OWWidget import *
from OWGUI import *
import numpy as np
import Orange


class OWSpectrum(OWWidget):
    settingsList = ["time_points", "periods", "n_rows", "criteria"]

    def __init__(self,parent=None, signalManager = None):
        OWWidget.__init__(self, parent, signalManager, "Frequencies", noReport=True)
        self.inputs = [("Data", ExampleTable, self.setData, Default)]
        self.outputs = [("Spectra", ExampleTable, Default)]

        self.time_points = "0 1 2 3 4 5 6 7 8 9 10 11 12 14 16 18 20 22 24"
        self.periods = "0.5 1 2 3 4 5 6 7 8 9 11 12 18 24 30 36"
        self.n_rows = 1000
        self.criteria = 0
#        self.loadSettings()

        b = OWGUI.widgetBox(self.controlArea, "Parameters", addSpace=True)
        OWGUI.lineEdit(b, self, "time_points", "Time points", labelWidth=100, controlWidth=300, orientation=0)
        OWGUI.lineEdit(b, self, "periods", "Periods", labelWidth=100, controlWidth=300, orientation=0)

        b = OWGUI.widgetBox(self.controlArea, "Selection", addSpace=True)
        b1 = OWGUI.widgetBox(b, orientation=0)
        OWGUI.lineEdit(b1, self, "n_rows", "Number of rows ", labelWidth=100, controlWidth=50, orientation=0, valueType=float, validator=QDoubleValidator())
        OWGUI.separator(b1)
        OWGUI.rubber(b1)
        b1 = OWGUI.widgetBox(b, orientation=0)
        b2 = OWGUI.widgetBox(b1, orientation=1)
        OWGUI.widgetLabel(b2, "Select by ")
        OWGUI.rubber(b2)
        b2 = OWGUI.widgetBox(b1, orientation=1)
        OWGUI.radioButtonsInBox(b2, self, "criteria", ["Maximal peak", "Sum"], orientation=0)
        OWGUI.rubber(b1)
        OWGUI.separator(self.controlArea)

        OWGUI.button(self.controlArea, self, "Analyze", self.analyze)

    def setData(self, data):
        self.data = data
        self.analyze()

    def analyze(self):
        if not self.data:
            self.send("Spectra", None)
            return

        d, c, metas = self.data.toNumpy("a/cw")
        tpoints = np.array(map(float, self.time_points.split()))
        #ws = np.arange(0.25, 12, 0.25)
        #curves = np.exp(2 * np.pi * 1j * ws[:, np.newaxis] * tpoints / 24)
        ws = np.array(map(float, self.periods.split()))
        curves = np.exp(2 * np.pi * 1j * tpoints / ws[:, np.newaxis])
        freqs = np.dot(d, curves.T)
        freqs = np.abs(freqs)
        freqs = freqs / np.average(freqs, axis=0)

        if self.criteria == 0:
            sel = np.argsort(np.max(freqs, axis=1))[-self.n_rows:]
        else:
            sel = np.argsort(np.sum(freqs, axis=1))[-self.n_rows:]

        freqs = freqs[sel]

        if c is None:
            Xy = freqs
        else:
            Xy = np.hstack((freqs, c[sel].reshape(self.n_rows, 1)))

        domain = Orange.data.Domain(
            [Orange.feature.Continuous("W_%.2f" % w) for w in ws],
            self.data.domain.classVar)
        domain.addmetas(self.data.domain.getmetas())
        data = Orange.data.Table(domain, Xy)
        if data.domain.getmetas() is not None:
            id = data.domain.getmetas().keys()[0]
            for ii, i in enumerate(sel):
                data[ii][id] = self.data[i][id]
        self.send("Spectra", data)
