### writes a text file with code that defines methods for sequence slots

definition ="""
P$name$ P$name$_FromArguments(PyObject *arg) { return TMM_$name$::P_FromArguments(arg); }
PyObject *$name$_FromArguments(PyTypeObject *type, PyObject *arg) { return TMM_$name$::_FromArguments(type, arg); }
PyObject *$name$_new(PyTypeObject *type, PyObject *arg, PyObject *kwds) BASED_ON(Orange, "(items)") { return TMM_$name$::_new(type, arg, kwds); }
PyObject *$name$_str(TPyOrange *self) { return TMM_$name$::_str(self); }
PyObject *$name$_repr(TPyOrange *self) { return TMM_$name$::_str(self); }
PyObject *$name$_getitem(TPyOrange *self, PyObject *key) { return TMM_$name$::_getitem(self, key); }
int       $name$_setitem(TPyOrange *self, PyObject *key, PyObject *value) { return TMM_$name$::_setitem(self, key, value); }
int       $name$_len(TPyOrange *self) { return TMM_$name$::_len(self); }
int       $name$_contains(TPyOrange *self, PyObject *key) { return TMM_$name$::_contains(self, key); }

PyObject *$name$_has_key(TPyOrange *self, PyObject *key) PYARGS(METH_O, "(key) -> None") { return TMM_$name$::_has_key(self, key); }
PyObject *$name$_get(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "(key[, default]) -> value") { return TMM_$name$::_get(self, args); }
PyObject *$name$_setdefault(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "(key[, default]) -> value") { return TMM_$name$::_setdefault(self, args); }
PyObject *$name$_clear(TPyOrange *self, PyObject *args) PYARGS(METH_NOARGS, "() -> None") { return TMM_$name$::_clear(self); }
PyObject *$name$_keys(TPyOrange *self, PyObject *args) PYARGS(METH_NOARGS, "() -> keys") { return TMM_$name$::_keys(self); }
PyObject *$name$_values(TPyOrange *self, PyObject *args) PYARGS(METH_NOARGS, "() -> values") { return TMM_$name$::_values(self); }
PyObject *$name$_items(TPyOrange *self, PyObject *args) PYARGS(METH_NOARGS, "() -> items") { return TMM_$name$::_items(self); }
PyObject *$name$_update(TPyOrange *self, PyObject *args) PYARGS(METH_O, "(items) -> None") { return TMM_$name$::_update(self, args); }


"""

outf = open("lib_maps_auto.txt", "wt")

def normalList(name, goesto):
  return tuple([x % name for x in ("%sList", "%s", "P%sList", "T%sList", "P%s")] + [goesto])

#  list name in Python,    element name in Py, wrapped list name in C, list name in C,         list element name in C, interface file
for (name,                 pykeytype,                 pyvaluetype,             c_key,              c_value,           goesto) in \
  [("VariableFloatMap",    "&PyOrVariable_Type",      "",                      "PVariable",        "float",           "lib_preprocess.cpp"),
   ("VariableFilterMap",   "&PyOrVariable_Type",      "&PyOrValueFilter_Type", "PVariable",        "PValueFilter",    "lib_preprocess.cpp")
  ]:

  outf.write("**** This goes to '%s' ****\n" % goesto)
  outf.write("typedef MapMethods<P%s, T%s, %s, %s> TMM_%s;\n" % (name, name, c_key, c_value, name))
  keyfunc = pykeytype and "_orangeValue" or "_nonOrangeValue"
  valuefunc = pyvaluetype and "_orangeValue" or "_nonOrangeValue"
  outf.write("INITIALIZE_MAPMETHODS(TMM_%s, %s, %s, %sFromPython<%s>, %sFromPython<%s>, %sToPython<%s>, %sToPython<%s>)\n"
              % (name, pykeytype or "NULL", pyvaluetype or "NULL", keyfunc, c_key, valuefunc, c_value, keyfunc, c_key, valuefunc, c_value))
  outf.write(definition.replace("$name$", name))
  
outf.close()

##definition = """
##bool convertFromPython(PyObject *, $elementname$ &);
##PyObject *convertToPython(const $elementname$ &);
###define $listname$ _TOrangeVector<$elementname$>
##typedef GCPtr< $listname$ > $wrappedlistname$;
##""" \
##+ definition \
##+ "inline PyObject *$pyname$_repr(TPyOrange *self) { return $classname$::_str(self); }\n"
##
##
##coutf = open("lib_vectors.cpp", "wt")
##
##coutf.write("""\
###include "orvector.hpp"
###include "cls_orange.hpp"
###include "vectortemplates.hpp"
###include "externs.px"
##""")
##
##for (pyname, pyelementname, wrappedlistname, listname, elementname) in \
##  [("IntList",          "int",    "PIntList",          "TIntList",          "int"),
##   ("FloatList",        "float",  "PFloatList",        "TFloatList",        "float"),
##   ("StringList",       "string", "PStringList",       "TStringList",       "string"),
##   ("LongList",         "int",    "PLongList",         "TLongList",         "long"),
##   ("_Filter_index",     "int",    "PFilter_index",     "TFilter_index",     "FOLDINDEXTYPE"),
##   ]:
##  if (pyname[0]=="_"):
##    pyname = pyname[1:]
##    outfile=outf
##  else:
##    outfile=coutf
##  outfile.write(definition.replace("$pyname$", pyname)
##                       .replace("$classname$", "ListOfUnwrappedMethods<%s, %s, %s>" % (wrappedlistname, listname, elementname))
##                       .replace("$pyelement$", pyelementname)
##                       .replace("$wrappedlistname$", wrappedlistname)
##                       .replace("$listname$", listname)
##                       .replace("$elementname$", elementname)
##             +"\n\n"
##            )
##
##coutf.write('#include "lib_vectors.px"\n')
##
##outf.close()
##coutf.close()