# -*- coding=utf-8 -*-
"""
<name>SVM Regression</name>
<description>Support Vector Machine Regression.</description>
<icon>icons/BasicSVM.png</icon>
<contact>Ales Erjavec (ales.erjavec(@at@)fri.uni-lj.si)</contact>
<priority>100</priority>
<keywords>Support, Vector, Machine, Regression</keywords>

"""

import orngSVM

from OWSVM import *

class OWSVMRegression(OWSVM):
    def __init__(self, parent=None, signalManager=None, title="SVM Regression"):
        OWSVM.__init__(self, parent, signalManager, title)
        
        self.inputs=[("Example Table", ExampleTable, self.setData), ("Preprocess", PreprocessedLearner, self.setPreprocessor)]
        self.outputs=[("Learner", orange.Learner, Default),("Classifier", orange.Classifier, Default),("Support Vectors", ExampleTable)]
        
        buttons = self.findChildren(QRadioButton)
        b_parent = None
        for b in buttons:
            if "C-SVM" in b.text():
                b.setText(u"ε-SVR")
                b_parent = b.parent()
                b.setToolTip("Epsilon SVR")
            if u"ν-SVM" in b.text():
                b.setText(u"ν-SVR")
                b.setToolTip("Nu SVR")
        
        if b_parent:
            grid = b_parent.layout()
            for i in range(3):
                item = grid.itemAtPosition(1, i)
                widget = item.widget()
                index = grid.indexOf(widget)
                grid.takeAt(index)
                if i == 1:
                    grid.addWidget(widget, 2, i, Qt.AlignRight)
                else:
                    grid.addWidget(widget, 2, i)
            
            grid.addWidget(QLabel(u"Loss Epsilon (ε)", b_parent), 1, 1, Qt.AlignRight)
            epsilon = OWGUI.doubleSpin(b_parent, self, "p", 0.05, 1.0, 0.05,
                        addToLayout=False,
                        callback=lambda *x: self.setType(0),
                        alignment=Qt.AlignRight)
            grid.addWidget(epsilon, 1, 2)
        
        self.probability = False
        self.probBox.hide()
        
if __name__ == "__main__":
    app = QApplication([])
    w = OWSVMRegression()
    w.show()
    app.exec_()