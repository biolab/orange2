"""
Feature scoring, selection, discretization, continuzation, imputation,
construction and feature interaction analysis.
"""

__all__ = ["Descriptor", "Discrete", "Continuous", "Python", "String", "Type"]

from Orange import core

Descriptor = core.Variable
Discrete = core.EnumVariable
Continuous = core.FloatVariable
Python = core.PythonVariable
String = core.StringVariable
Type = core.VarTypes

__docformat__ = 'restructuredtext'
