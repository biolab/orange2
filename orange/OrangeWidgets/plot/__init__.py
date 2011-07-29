"""

*************************
Plot classes and tools for use in Orange widgets
*************************

The main class of this module is :obj:`OrangeWidgets.plot.OWPlot`, from which all plots 
in visualization widgets should inherit. 

This module also contains plot elements, which are normally use by the :obj:`OrangeWidgets.plot.OWPlot`, but can 
be either subclassed or used directly from outside. These elements are:
* :obj: `OrangeWidgets.plot.OWCurve`
* :obj: `OrangeWidgets.plot.OWPoint`
* :obj: `OrangeWidgets.plot.OWAxis`
* :obj: `OrangeWidgets.plot.OWLegend`
"""

from owcurve import *
from owpoint import *
from owlegend import *
from owaxis import *
from owplot import *
from owtools import *
