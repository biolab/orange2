"""
Feature scoring, selection, discretization, continuzation, imputation,
construction and feature interaction analysis.
"""

import scoring
import selection
import discretization
import continuization
import imputation

from Orange.core import Variable as Descriptor
from Orange.core import EnumVariable as Discrete
from Orange.core import FloatVariable as Continuous
from Orange.core import PythonVariable as Python
from Orange.core import StringVariable as String

from Orange.core import VarTypes as Type

__docformat__ = 'restructuredtext'
