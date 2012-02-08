import Orange

from multibase import MultiLabelLearner
from multibase import MultiLabelClassifier

from br import BinaryRelevanceLearner
from br import BinaryRelevanceClassifier

from lp import LabelPowersetLearner
from lp import LabelPowersetClassifier

from multiknn import MultikNNLearner
from multiknn import MultikNNClassifier

from mlknn import MLkNNLearner
from mlknn import MLkNNClassifier

from brknn import BRkNNLearner
from brknn import BRkNNClassifier

def is_multilabel(data):
    if not data.domain.class_vars:
        return False
    for c in data.domain.class_vars:
        if type(c) is not Orange.feature.Discrete or sorted(c.values) != ['0', '1']:
            return False
    return True
