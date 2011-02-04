"""

.. index:: misc

Module Orange.misc contains common functions and classes which are used in other modules.


==================
Counters
==================

.. index:: misc
.. index::
   single: misc; counters


==================
Renders
==================

.. index:: misc
.. index::
   single: misc; Renders

==================
Renders
==================

.. index:: selection
.. index::
   single: misc; selection


"""


__all__ = ["counters", "selection", "render"]

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
        return self.title + "=" * (progchar) + ">" + " " * (self.charwidth - 5 - progchar) + "%3i" % int(round(self.state)) + "%"

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