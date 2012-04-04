"""
<name>Transpose</name>
<description>Transpose a data table</description>
<icon>icons/Transpose.png</icon>

"""

from OWWidget import *
from OWItemModels import VariableListModel
import OWGUI

import Orange
from Orange import feature
import warnings
from operator import add

def float_or_na(val):
    if val.is_special():
        return "NA"
    else:
        return float(val)

class VariableOrNoneListModel(VariableListModel):
    def data(self, index, role=Qt.DisplayRole):
        i = index.row()
        var = self[i]
        if var is None:
            if role == Qt.DisplayRole:
                return QVariant("(None)")
            elif role == Qt.ToolTipRole:
                return QVariant("None - use default naming instead.")
        else:
            return VariableListModel.data(self, index, role)

def transpose(data, feature_names=None, error_non_continuous="error",
              error_meta="ignore"):
    domain = data.domain
    attributes = list(domain.variables)
    metas = domain.get_metas().values()
    if any(not isinstance(f, feature.Continuous) \
                for f in attributes):
        if error_non_continuous == "error":
            raise TypeError()
        elif error_non_continuous == "warn":
            warnings.warn("Non-continuous features in the domain will be ignored!",
                          UserWarning)
        elif error_non_continuous == "ignore":
            pass
        else:
            raise ValueError(error_discrete)
        attributes = [attr for attr in attributes \
                      if isinstance(attr, feature.Continuous)]

    if any(not isinstance(m, feature.String) \
                for m in metas):
        if error_meta == "error":
            raise TypeError("Non string meta features.")
        elif error_meta == "ignore":
            metas = [m for m in metas if isinstance(m, feature.String)]
        elif error_meta == "warn":
            warnings.warn("Non string meta features in the domain will be ignored!",
                          UserWarning)
        else:
            raise ValueError(error_meta)

    if feature_names is not None:
        if isinstance(feature_names, basestring):
            # Name of the feature
            feature_names = domain[feature_names]
        if isinstance(feature_names, feature.String):
            feature_names = [str(inst[feature_names]) for inst in data]
        elif isinstance(feature_names, list):
            # List of names
            pass
        else:
            raise ValueError(feature_names)
    else:
        feature_names = ["F_%s" % (i + 1) for i in range(len(data))]

    new_features = map(feature.Continuous, feature_names)
    labels = [f.attributes.keys() for f in domain.variables]
    labels = sorted(reduce(set.union, labels, set()))

    new_metas = map(feature.String, labels)
    new_metas = dict((feature.Descriptor.new_meta_id(), m) for m in new_metas)

    new_labels = [m.name for m in metas]
    new_domain = Orange.data.Domain(new_features, False)

    new_domain.add_metas(new_metas)

    new_data = Orange.data.Table(new_domain)

    for f in attributes:
        vals = [float_or_na(inst[f]) for inst in data]
        new_ins = Orange.data.Instance(new_domain, vals)

        for key, value in f.attributes.items():
            new_ins[key] = str(value)

        new_data.append(new_ins)

    for new_f, inst in zip(new_features, data):
        for m in metas:
            mval = inst[m]
            if not mval.is_special():
                new_f.attributes[m.name] = str(mval)

    return new_data

def is_cont(f):
    return isinstance(f, feature.Continuous)

def is_disc(f):
    return isinstance(f, feature.Discrete)

def is_string(f):
    return isinstance(f, feature.String)

class OWTranspose(OWWidget):
    contextHandlers = {"": DomainContextHandler("", ["row_name_attr"])}
    def __init__(self, parent=None, signalManager=None, title="Transpose"):
        OWWidget.__init__(self, parent, signalManager, title,
                          wantMainArea=False)

        self.inputs = [("Data", Orange.data.Table, self.set_data)]
        self.outputs = [("Transposed Data", Orange.data.Table)]

        # Settings
        self.row_name_attr = None

        box = OWGUI.widgetBox(self.controlArea, "Info")
        self.info_w = OWGUI.widgetLabel(box, "No data on input.")

        box = OWGUI.widgetBox(self.controlArea, "Row Names")
        self.row_name_combo = QComboBox(self, objectName="row_name_combo",
                                toolTip="Row to use for new feature names.",
                                activated=self.on_row_name_changed)
        self.row_name_model = VariableOrNoneListModel()
        self.row_name_combo.setModel(self.row_name_model)
        box.layout().addWidget(self.row_name_combo)

        OWGUI.rubber(self.controlArea)

    def clear(self):
        """Clear the widget state."""
        self.warning(1)
        self.error(0)
        self.row_name_attr = None
        self.row_name_model[:] = []
        self.data = None

    def set_data(self, data=None):
        self.closeContext("")
        self.clear()
        self.data = data
        if data is not None:
            variables = data.domain.variables
            cont_features = [f for f in variables if is_cont(f)]
            non_cont = [f for f in variables if not is_cont(f)]
            info = "Data with %i continuous feature%s\n" \
                    % (len(cont_features), "s" if len(cont_features) > 1 else "")
            if non_cont:
                ignore_text = "Ignoring %i non-continuous feature%s" \
                                % (len(non_cont), "s" if len(non_cont) > 1 else "")
                info += ignore_text
                self.warning(1, ignore_text)

            self.info_w.setText(info)

            all_vars = data.domain.variables + data.domain.get_metas().values()
            str_features = [f for f in all_vars if is_string(f)]
            str_feature_names = [f.name for f in str_features]

            if len(str_feature_names):
                self.row_name_attr = str_feature_names[0]

            self.openContext("", data)

            self.row_name_model[:] = [None] + str_features
            # If changed after 'openContext'
            if self.row_name_attr in str_feature_names:
                index = str_feature_names.index(self.row_name_attr) + 1
                self.row_name_combo.setCurrentIndex(index)
            elif not str_feature_names:
                self.row_name_attr = None

            self.do_transpose()
        else:
            self.info_w.setText("No data on input")
            self.send("Transposed Data", None)

    def on_row_name_changed(self, index):
        if len(self.row_name_model) and index > 0:
            self.row_name_attr = self.row_name_model[index].name
        else:
            self.row_name_attr = None
        self.do_transpose()

    def do_transpose(self):
        transposed = None
        self.error(0)
        if self.data is not None:
            try:
                transposed = transpose(self.data, self.row_name_attr,
                                       error_non_continuous="ignore",
                                       error_meta="ignore")
            except Exception, ex:
                self.error(0, str(ex))
                raise

        self.send("Transposed Data", transposed)


if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    w = OWTranspose()
#    data = Orange.data.Table("doc:dicty-express.tab")
    data = Orange.data.Table("doc:small-sample.tab")
#    data = Orange.data.Table("iris")
    w.set_data(data)
    w.show()
    sys.exit(app.exec_())
