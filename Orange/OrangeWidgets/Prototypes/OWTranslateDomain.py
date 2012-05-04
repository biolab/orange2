"""
<name>Translate Domain</name>
<description>Translate input data into a target domain</description>
<contact>ales.erjavec(@ at @)fri.uni-lj.si</contact>
<icons>icons/TranslateDomain.png</icons>
<tags>translate,transformation</tags>

"""

from OWWidget import *

import OWGUI
import Orange


class OWTranslateDomain(OWWidget):
    def __init__(self, parent=None, signalManager=None,
                 title="Translate Domain"):
        OWWidget.__init__(self, parent, signalManager, title,
                          wantMainArea=False)

        self.inputs = [("Target Domain", Orange.data.Table, self.set_target),
                       ("Input Data", Orange.data.Table, self.set_input)]

        self.outputs = [("Translated Data", Orange.data.Table)]

        box = OWGUI.widgetBox(self.controlArea, "Info")
        self.info = OWGUI.widgetLabel(box, "No target domain\nNo input data")

        OWGUI.rubber(self.controlArea)

        self.target = None
        self.input_data = None

    def set_target(self, target=None):
        self.target = target

    def set_input(self, input_data=None):
        self.input_data = input_data

    def handleNewSignals(self):
        self.update_info()
        self.error(0)
        self.commit()

    def update_info(self):
        target_lines = ["No target domain"]
        input_lines = ["No input data"]
        if self.target is not None:
            class_var = self.target.domain.class_var
            if class_var:
                class_str = type(class_var).__name__.lower()
            else:
                class_str = "no"
            num_features = len(self.target.domain.features)
            target_lines = ["Target domain with %i features and %s class." % \
                            (num_features, class_str)]
        if self.input_data is not None:
            input_lines = ["Input data with %i instances" % \
                           len(self.input_data)]
        self.info.setText("\n".join(target_lines + input_lines))

    def commit(self):
        self.error(0)
        translated = None
        if self.target is not None and self.input_data is not None:
            try:
                translated = self.input_data.translate(self.target.domain)
            except Exception, ex:
                import traceback
                traceback.print_exc(limit=8, file=sys.stdout)
                self.error("Failed to convert the domain (%r)." % ex)
        self.send("Translated Data", translated)
