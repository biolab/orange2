### writes a text file with code that defines methods for sequence slots

from pyprops import ClassDefinition
classes = {None: None}

definition ="""
$wrappedlistname$ P$pyname$_FromArguments(PyObject *arg) { return $classname$::P_FromArguments(arg); }
PyObject *$pyname$_FromArguments(PyTypeObject *type, PyObject *arg) { return $classname$::_FromArguments(type, arg); }
PyObject *$pyname$_new(PyTypeObject *type, PyObject *arg, PyObject *kwds) BASED_ON(Orange, "(<list of $pyelement$>)") ALLOWS_EMPTY { return $classname$::_new(type, arg, kwds); }
PyObject *$pyname$_getitem_sq(TPyOrange *self, int index) { return $classname$::_getitem(self, index); }
int       $pyname$_setitem_sq(TPyOrange *self, int index, PyObject *item) { return $classname$::_setitem(self, index, item); }
PyObject *$pyname$_getslice(TPyOrange *self, int start, int stop) { return $classname$::_getslice(self, start, stop); }
int       $pyname$_setslice(TPyOrange *self, int start, int stop, PyObject *item) { return $classname$::_setslice(self, start, stop, item); }
int       $pyname$_len_sq(TPyOrange *self) { return $classname$::_len(self); }
PyObject *$pyname$_richcmp(TPyOrange *self, PyObject *object, int op) { return $classname$::_richcmp(self, object, op); }
PyObject *$pyname$_concat(TPyOrange *self, PyObject *obj) { return $classname$::_concat(self, obj); }
PyObject *$pyname$_repeat(TPyOrange *self, int times) { return $classname$::_repeat(self, times); }
PyObject *$pyname$_str(TPyOrange *self) { return $classname$::_str(self); }
PyObject *$pyname$_repr(TPyOrange *self) { return $classname$::_str(self); }
int       $pyname$_contains(TPyOrange *self, PyObject *obj) { return $classname$::_contains(self, obj); }
PyObject *$pyname$_append(TPyOrange *self, PyObject *item) PYARGS(METH_O, "($pyelement$) -> None") { return $classname$::_append(self, item); }
PyObject *$pyname$_extend(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(sequence) -> None") { return $classname$::_extend(self, obj); }
PyObject *$pyname$_count(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "($pyelement$) -> int") { return $classname$::_count(self, obj); }
PyObject *$pyname$_filter(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "([filter-function]) -> $pyname$") { return $classname$::_filter(self, args); }
PyObject *$pyname$_index(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "($pyelement$) -> int") { return $classname$::_index(self, obj); }
PyObject *$pyname$_insert(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "(index, item) -> None") { return $classname$::_insert(self, args); }
PyObject *$pyname$_native(TPyOrange *self) PYARGS(METH_NOARGS, "() -> list") { return $classname$::_native(self); }
PyObject *$pyname$_pop(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "() -> $pyelement$") { return $classname$::_pop(self, args); }
PyObject *$pyname$_remove(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "($pyelement$) -> None") { return $classname$::_remove(self, obj); }
PyObject *$pyname$_reverse(TPyOrange *self) PYARGS(METH_NOARGS, "() -> None") { return $classname$::_reverse(self); }
PyObject *$pyname$_sort(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "([cmp-func]) -> None") { return $classname$::_sort(self, args); }
PyObject *$pyname$__reduce__(TPyOrange *self, PyObject *) { return $classname$::_reduce(self); }
"""

wdefinition = "\nextern PyTypeObject PyOr$pyelement$_Type_inh;\n" + definition

udefinition = """
bool convertFromPython(PyObject *, $elementname$ &);
PyObject *convertToPython(const $elementname$ &);
""" \
+ definition

# removed from udefinition
# #define $listname$ _TOrangeVector<$elementname$>
# typedef GCPtr< $listname$ > $wrappedlistname$;

outf = open("lib_vectors_auto.txt", "wt")

def normalList(name, goesto):
  return tuple([x % name for x in ("%sList", "%s", "P%sList", "T%sList", "P%s")] + [goesto])


#  list name in Python,    element name in Py, wrapped list name in C, list name in C,         list element name in C, interface file
for (pyname, pyelementname, wrappedlistname, listname, elementname, goesto) in \
  [("ValueList",           "Value",            "PValueList",           "TValueList",           "TValue",               "cls_value.cpp"),
   ("VarList",             "Variable",         "PVarList",             "TVarList",             "PVariable",            "lib_kernel.cpp"),
   ("VarListList",         "VarList",          "PVarListList",         "TVarListList",         "PVarList",             "lib_kernel.cpp"),
   ("DomainDistributions", "Distribution",     "PDomainDistributions", "TDomainDistributions", "PDistribution",        "lib_kernel.cpp"),
   normalList("Distribution", "lib_kernel.cpp"),
   normalList("ExampleGenerator", "lib_kernel.cpp"),
   normalList("Classifier", "lib_kernel.cpp"),
   
   ("DomainBasicAttrStat", "BasicAttrStat",    "PDomainBasicAttrStat", "TDomainBasicAttrStat", "PBasicAttrStat",       "lib_components.cpp"),
   ("DomainContingency",   "Contingency",      "PDomainContingency",   "TDomainContingency",   "PContingencyClass",    "lib_components.cpp"),
   normalList("ValueFilter", "lib_components.cpp"),
   normalList("Filter", "lib_components.cpp"),
   normalList("HierarchicalCluster", "lib_components.cpp"),
   
   ("AssociationRules",    "AssociationRule",  "PAssociationRules",    "TAssociationRules",    "PAssociationRule",     "lib_learner.cpp"),
   normalList("TreeNode", "lib_learner.cpp"),
   normalList("C45TreeNode", "lib_learner.cpp"),
   normalList("Rule", "lib_learner.cpp"),
   normalList("ConditionalProbabilityEstimator", "lib_components.cpp"),
   normalList("ProbabilityEstimator", "lib_components.cpp"),
   normalList("EVCDist", "lib_learner.cpp"),

   normalList("Heatmap", "orangene.cpp"),
   normalList("SOMNode", "som.cpp")
   ]:
  outf.write("**** This goes to '%s' ****\n" % goesto)
  outf.write(wdefinition.replace("$pyname$", pyname)
                        .replace("$classname$", "ListOfWrappedMethods<%s, %s, %s, &PyOr%s_Type>" % (wrappedlistname, listname, elementname, pyelementname))
                       .replace("$pyelement$", pyelementname)
                       .replace("$wrappedlistname$", wrappedlistname)
                       .replace(">>", "> >")
             +"\n\n"
            )

  classes[listname] = ClassDefinition(listname, "TOrange")  



coutf = open("lib_vectors.cpp", "wt")

coutf.write("""/*
    This file is part of Orange.

    Orange is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    Orange is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Orange; if not, write to the Free Software
    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

    Authors: Janez Demsar, Blaz Zupan, 1996--2002
    Contact: janez.demsar@fri.uni-lj.si
*/


#include "orvector.hpp"
#include "cls_orange.hpp"
#include "vectortemplates.hpp"
#include "externs.px"

#include "distance_dtw.hpp"
""")

for (pyname, pyelementname, wrappedlistname, listname, elementname, wrapped) in \
  [("BoolList",         "bool",     "PBoolList",         "TBoolList",         "bool", 0),
   ("IntList",          "int",      "PIntList",          "TIntList",          "int", 0),
   ("FloatList",        "float",    "PFloatList",        "TFloatList",        "float", 0),
   ("FloatListList",    "FloatList","PFloatListList",    "TFloatListList",    "PFloatList", 1),
   ("StringList",       "string",   "PStringList",       "TStringList",       "string", 0),
   ("LongList",         "int",      "PLongList",         "TLongList",         "long", 0),
   ("_Filter_index",     "int",     "PFilter_index",     "TFilter_index",     "FOLDINDEXTYPE", 0),
   ("AlignmentList",    "Alignment", "PAlignmentList",  "TAlignmentList",    "TAlignment", 0),

   ("IntFloatList",     "tuple(int, float)",   "PIntFloatList",    "TIntFloatList",     "pair<int, float>", 0),
   ("FloatFloatList",   "tuple(float, float)", "PFloatFloatList",  "TFloatFloatList",   "pair<float, float>", 0),
   ]:
  if (pyname[0]=="_"):
    pyname = pyname[1:]
    outfile=outf
  else:
    outfile=coutf
  if wrapped:
    classname = "ListOfWrappedMethods<%s, %s, %s, &PyOr%s_Type>" % (wrappedlistname, listname, elementname, pyelementname)
  else:
    classname = "ListOfUnwrappedMethods<%s, %s, %s>" % (wrappedlistname, listname, elementname)

  outfile.write((wrapped and wdefinition or udefinition)
                       .replace("$pyname$", pyname)
                       .replace("$classname$", classname)
                       .replace("$pyelement$", pyelementname)
                       .replace("$wrappedlistname$", wrappedlistname)
                       .replace("$listname$", listname)
                       .replace("$elementname$", elementname)
                       .replace(">>", "> >")
             +"\n\n"
            )

  classes[listname] = ClassDefinition(listname, "TOrange")

coutf.write('#include "lib_vectors.px"\n')

outf.close()
coutf.close()

import pickle

classes["TAttributedFloatList"] = ClassDefinition("TAttributedFloatList", "TFloatList")
classes["TAttributedBoolList"] = ClassDefinition("TAttributedBoolList", "TBoolList")

import os
if not os.path.exists("../orange/ppp"):
  os.mkdir("../orange/ppp")
pickle.dump(classes, file("../orange/ppp/lists", "wt"))