"""

.. index:: misc

Module Orange.misc contains common functions and classes which are used in other modules.

==================
Counters
==================

.. index:: misc
.. index::
   single: misc; counters

.. automodule:: Orange.misc.counters
  :members:

==================
Render
==================

.. index:: misc
.. index::
   single: misc; render

.. automodule:: Orange.misc.render
  :members:

==================
Selection
==================

.. index:: selection
.. index::
   single: misc; selection

Many machine learning techniques generate a set different solutions or have to
choose, as for instance in classification tree induction, between different
features. The most trivial solution is to iterate through the candidates,
compare them and remember the optimal one. The problem occurs, however, when
there are multiple candidates that are equally good, and the naive approaches
would select the first or the last one, depending upon the formulation of
the if-statement.

:class:`Orange.misc.selection` provides a class that makes a random choice
in such cases. Each new candidate is compared with the currently optimal
one; it replaces the optimal if it is better, while if they are equal,
one is chosen by random. The number of competing optimal candidates is stored,
so in this random choice the probability to select the new candidate (over the
current one) is 1/w, where w is the current number of equal candidates,
including the present one. One can easily verify that this gives equal
chances to all candidates, independent of the order in which they are presented.

.. automodule:: Orange.misc.selection
  :members:

Example
--------

The following snippet loads the data set lymphography and prints out the
feature with the highest information gain.

part of `misc-selection-bestonthefly.py`_ (uses `lymphography.tab`_)

.. literalinclude:: code/misc-selection-bestonthefly.py
  :lines: 7-16

Our candidates are tuples gain ratios and features, so we set
:obj:`callCompareOn1st` to make the compare function compare the first element
(gain ratios). We could achieve the same by initializing the object like this:

part of `misc-selection-bestonthefly.py`_ (uses `lymphography.tab`_)

.. literalinclude:: code/misc-selection-bestonthefly.py
  :lines: 18-18


The other way to do it is through indices.

`misc-selection-bestonthefly.py`_ (uses `lymphography.tab`_)

.. literalinclude:: code/misc-selection-bestonthefly.py
  :lines: 25-

.. _misc-selection-bestonthefly.py: code/misc-selection-bestonthefly.py.py
.. _lymphography.tab: code/lymphography.tab

Here we only give gain ratios to :obj:`bestOnTheFly`, so we don't have to specify a
special compare operator. After checking all features we get the index of the 
optimal one by calling :obj:`winnerIndex`.

==================
Server files
==================

.. index:: server files

.. automodule:: Orange.misc.serverfiles

"""

import counters
import selection
import render
import serverfiles

__all__ = ["counters", "selection", "render", "serverfiles"]

import random, types, sys

def getobjectname(x, default=""):
    if type(x)==types.StringType:
        return x
      
    for i in ["name", "shortDescription", "description", "func_doc", "func_name"]:
        if getattr(x, i, ""):
            return getattr(x, i)

    if hasattr(x, "__class__"):
        r = repr(x.__class__)
        if r[1:5]=="type":
            return str(x.__class__)[7:-2]
        elif r[1:6]=="class":
            return str(x.__class__)[8:-2]
    return default


def demangleExamples(x):
    if type(x)==types.TupleType:
        return x
    else:
        return x, 0


def frange(*argw):
    start, stop, step = 0.0, 1.0, 0.1
    if len(argw)==1:
        start=step=argw[0]
    elif len(argw)==2:
        stop, step = argw
    elif len(argw)==3:
        start, stop, step = argw
    elif len(argw)>3:
        raise AttributeError, "1-3 arguments expected"

    stop+=1e-10
    i=0
    res=[]
    while 1:
        f=start+i*step
        if f>stop:
            break
        res.append(f)
        i+=1
    return res

verbose = 0

def printVerbose(text, *verb):
    if len(verb) and verb[0] or verbose:
        print text

class ConsoleProgressBar(object):
    def __init__(self, title="", charwidth=40, step=1, output=sys.stderr):
        self.title = title + " "
        self.charwidth = charwidth
        self.step = step
        self.currstring = ""
        self.state = 0
        self.output = output

    def clear(self, i=-1):
        try:
            if hasattr(self.output, "isatty") and self.output.isatty():
                self.output.write("\b" * (i if i != -1 else len(self.currstring)))
            else:
                self.output.seek(-i if i != -1 else -len(self.currstring), 2)
        except Exception: ## If for some reason we failed 
            self.output.write("\n")

    def getstring(self):
        progchar = int(round(float(self.state) * (self.charwidth - 5) / 100.0))
        return self.title + "=" * (progchar) + ">" + " " * (self.charwidth\
            - 5 - progchar) + "%3i" % int(round(self.state)) + "%"

    def printline(self, string):
        try:
            self.clear()
            self.output.write(string)
            self.output.flush()
        except Exception:
            pass
        self.currstring = string

    def __call__(self, newstate=None):
        if newstate == None:
            newstate = self.state + self.step
        if int(newstate) != int(self.state):
            self.state = newstate
            self.printline(self.getstring())
        else:
            self.state = newstate

    def finish(self):
        self.__call__(100)
        self.output.write("\n")

def progressBarMilestones(count, iterations=100):
    return set([int(i*count/float(iterations)) for i in range(iterations)])
