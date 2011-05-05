/*
    This file is part of Orange.
    
    Copyright 1996-2010 Faculty of Computer and Information Science, University of Ljubljana
    Contact: janez.demsar@fri.uni-lj.si

    Orange is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Orange is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Orange.  If not, see <http://www.gnu.org/licenses/>.
*/


#ifdef _MSC_VER
  #pragma warning (disable : 4786 4114 4018 4267 4244)
#endif

#include "vars.hpp"
#include "domain.hpp"
#include "examplegen.hpp"
#include "table.hpp"

#include "cls_orange.hpp"
#include "externs.px"

PVarList knownVars(PyObject *keywords); // defined in lib_kernel.cpp
TMetaVector *knownMetas(PyObject *keywords); // ibid
PDomain knownDomain(PyObject *keywords); // ibid

PyObject *encodeStatus(const vector<int> &Status);  // in cls_misc.cpp
PyObject *encodeStatus(const vector<pair<int, int> > &metaStatus);

/* ************ FILE EXAMPLE GENERATORS ************ */

#include "filegen.hpp"
BASED_ON(FileExampleGenerator - Orange.data.io.FileExampleGenerator, ExampleGenerator)

#include "tabdelim.hpp"
#include "c45inter.hpp"
#include "basket.hpp"


bool divDot(const string &name, string &before, string &after)
{ string::const_iterator bi(name.begin()), ei(name.end());
  for(; (ei!=bi) && (*(--ei)!='.'); );
  if (*ei!='.') return false;
  
  before=string(bi, ei); after=string(ei++, name.end());
  return true;
}


NO_PICKLE(BasketExampleGenerator)
NO_PICKLE(C45ExampleGenerator)
NO_PICKLE(FileExampleGenerator)
NO_PICKLE(TabDelimExampleGenerator)
NO_PICKLE(BasketFeeder)

BASED_ON(BasketFeeder - Orange.data.io.BasketFeeder, Orange)




PyObject *TabDelimExampleGenerator_new(PyTypeObject *type, PyObject *args, PyObject *keywords) BASED_ON(FileExampleGenerator - Orange.data.io.TabDelimExampleGenerator, "(examples[, use=domain|varlist])")
{ PyTRY
    char *fileName;
    int createNewOn = TVariable::Incompatible;
    if (!PyArg_ParseTuple(args, "s|i:TabDelimExampleGenerator.__new__", &fileName, &createNewOn))
      return NULL;

    string name(fileName), b, a;
    if (!divDot(name, b, a))
      name+=".tab";
    
    vector<int> status;
    vector<pair<int, int> > metaStatus;
    TExampleGenerator *egen = mlnew TTabDelimExampleGenerator(name, false, false, createNewOn, status, metaStatus);
    return Py_BuildValue("NNN", WrapNewOrange(egen, type), encodeStatus(status), encodeStatus(metaStatus));
  PyCATCH
}


PyObject *BasketExampleGenerator_new(PyTypeObject *type, PyObject *args, PyObject *keywords) BASED_ON(FileExampleGenerator - Orange.data.io.BasketExampleGenerator, "(examples[, use=domain])")
{ PyTRY
    char *fileName;
    int createNewOn = TVariable::Incompatible;
    if (!PyArg_ParseTuple(args, "s|i:BasketExampleGenerator.__new__", &fileName, &createNewOn))
      return NULL;

    string name(fileName), b, a;
    if (!divDot(name, b, a))
      name+=".basket";

    vector<int> status;
    vector<pair<int, int> > metaStatus;
    TExampleGenerator *egen = mlnew TBasketExampleGenerator(name, PDomain(), createNewOn, status, metaStatus);
    return Py_BuildValue("NNN", WrapNewOrange(egen, type), encodeStatus(status), encodeStatus(metaStatus));
  PyCATCH
}


PyObject *BasketFeeder_clearCache(PyObject *, PyObject *) PYARGS(METH_O, "() -> None")
{ PyTRY
    TBasketFeeder::clearCache();
    RETURN_NONE;
  PyCATCH
}



PyObject *C45ExampleGenerator_new(PyTypeObject *type, PyObject *args, PyObject *keywords) BASED_ON(FileExampleGenerator - Orange.data.io.C45ExampleGenerator, "(examples[, use=domain|varlist])")
{ PyTRY
    char *stem;
    int createNewOn = TVariable::Incompatible;
    if (!PyArg_ParseTuple(args, "s|i:C45ExampleGenerator.__new__", &stem, &createNewOn))
      return NULL;

    string domain, data;
    string b, a;
    if (divDot(stem, b, a))
      { data=stem; domain=b+".names"; }
    else
      { data=string(stem)+".data"; domain=string(stem)+".names"; }

    vector<int> status;
    vector<pair<int, int> > metaStatus;
    TExampleGenerator *egen = mlnew TC45ExampleGenerator(data, domain, createNewOn, status, metaStatus);
    return Py_BuildValue("NNO", WrapNewOrange(egen, type), encodeStatus(status), encodeStatus(metaStatus));
  PyCATCH
}




int pt_ExampleGenerator(PyObject *args, void *egen);

void tabDelim_writeDomain(FILE *, PDomain, bool autodetect, char delim = '\t', bool listDiscreteValues = true);
void tabDelim_writeExamples(FILE *, PExampleGenerator, char delim = '\t', const char *DK = NULL, const char *DC = NULL);


FILE *openWReplacedExtension(const char *filename, const char *extension, const char *oldExtension)
{
  const char *newname = replaceExtension(filename, extension, oldExtension);
  FILE *ostr = fopen(newname, "wt");
  if (!ostr)
    PyErr_Format(PyExc_SystemError, "cannot open file '%s'", newname);
  mldelete const_cast<char *>(newname);
  return ostr;
}

    
FILE *openExtended(const char *filename, const char *defaultExtension)
{
  const char *extension = getExtension(filename);
  const char *extended = extension ? filename : replaceExtension(filename, defaultExtension, NULL);
  FILE *ostr = fopen(extended, "wt");
  if (!ostr)
    PyErr_Format(PyExc_SystemError, "cannot open file '%s'", extended);
  if (!extension)
    mldelete const_cast<char *>(extended);
  return ostr;
}


int getStringIfExists(PyObject *keyws, const char *name, char *&res)
{
  PyObject *ldv = PyDict_GetItemString(keyws, name);
  if (ldv) {
    if (!PyString_Check(ldv)) {
      PyErr_Format(PyExc_TypeError, "string value expected for '%s'", name);
      return -1;
    }
   
    res = PyString_AsString(ldv);
    return 0;
  }

  return 1;
}


bool readUndefinedSpecs(PyObject *keyws, char *&DK, char *&DC)
{
  if (keyws) {
    int res;

    char *tmp;
    res = getStringIfExists(keyws, "NA", tmp);
    if (res == -1)
      return false;
    if (!res)
      DK = DC = tmp;

    res = getStringIfExists(keyws, "DC", DC);
    if (res == -1)
      return false;

    res = getStringIfExists(keyws, "DK", DK);
    if (res == -1)
      return false;
  }

  return true;
}


PyObject *tabDelimBasedWrite(PyObject *args, PyObject *keyws, const char *defaultExtension, bool skipAttrTypes, char delim, bool listDiscreteValues = true)
{ PyTRY
    char *filename;
    PExampleGenerator gen;

    if (!PyArg_ParseTuple(args, "sO&", &filename, pt_ExampleGenerator, &gen))
      PYERROR(PyExc_TypeError, "string and example generator expected", PYNULL);

    if (skipAttrTypes && !gen->domain->classVar) {
      PyErr_Format(PyExc_TypeError, "Format .%s cannot save classless data sets", defaultExtension);
      return PYNULL;
    }
    
    char *DK = NULL, *DC = NULL;
    if (!readUndefinedSpecs(keyws, DK, DC))
      return PYNULL;
  
    FILE *ostr = openExtended(filename, defaultExtension);
    if (!ostr)
      return PYNULL;

    tabDelim_writeDomain(ostr, gen->domain, skipAttrTypes, delim, listDiscreteValues);
    tabDelim_writeExamples(ostr, gen, delim, DK, DC);
    fclose(ostr);

    RETURN_NONE
  PyCATCH
}


PyObject *saveTabDelimited(PyObject *, PyObject *args, PyObject *keyws) PYARGS(METH_VARARGS | METH_KEYWORDS, "(filename, examples[, list_discrete_values=1]) -> None")
{
  bool listDiscrete = true;

  if (keyws) {
    PyObject *ldv = PyDict_GetItemString(keyws, "list_discrete_values");
    if (!ldv) {
        ldv = PyDict_GetItemString(keyws, "listDiscreteValues");
    }
    listDiscrete = !ldv || (PyObject_IsTrue(ldv)!=0);
  }

  return tabDelimBasedWrite(args, keyws, "tab", false, '\t', listDiscrete);
}

PyObject *saveTxt(PyObject *, PyObject *args, PyObject *keyws) PYARGS(METH_VARARGS | METH_KEYWORDS, "(filename, examples) -> None")
{
  return tabDelimBasedWrite(args, keyws, "txt", true, '\t');
}


PyObject *saveCsv(PyObject *, PyObject *args, PyObject *keyws) PYARGS(METH_VARARGS | METH_KEYWORDS, "(filename, examples) -> None")
{
  return tabDelimBasedWrite(args, keyws, "csv", true, ',');
}


void c45_writeDomain(FILE *, PDomain);
void c45_writeExamples(FILE *, PExampleGenerator);

PyObject *saveC45(PyObject *, PyObject *args) PYARGS(METH_VARARGS, "(filename, examples) -> None")
{ PyTRY
    char *filename;
    PExampleGenerator gen;

    if (!PyArg_ParseTuple(args, "sO&", &filename, pt_ExampleGenerator, &gen))
      PYERROR(PyExc_TypeError, "string and example generator expected", PYNULL)
  
    if (!gen->domain->classVar)
      PYERROR(PyExc_SystemError, "C4.5 file cannot store classless data sets.", PYNULL);

    if (gen->domain->classVar->varType!=TValue::INTVAR)
      PYERROR(PyExc_SystemError, "Class in C4.5 file must be discrete.", PYNULL);

    const char *oldExtension = getExtension(filename);

    FILE *ostr;
    ostr = openWReplacedExtension(filename, "names", oldExtension);
    if (!ostr)
      return PYNULL;
    c45_writeDomain(ostr, gen->domain);
    fclose(ostr);

    ostr = openWReplacedExtension(filename, "data", oldExtension);
    if (!ostr)
      return PYNULL;
    c45_writeExamples(ostr, gen);
    fclose(ostr);

    RETURN_NONE
  PyCATCH
}



#include "spec_gen.hpp"


void basket_writeExamples(FILE *, PExampleGenerator, set<int> &missing);
void raiseWarning(bool, const char *s);

PyObject *saveBasket(PyObject *, PyObject *args) PYARGS(METH_VARARGS, "(filename, examples) -> None")
{ PyTRY
    char *filename;
    PExampleGenerator gen;

    if (!PyArg_ParseTuple(args, "sO&:saveBasket", &filename, pt_ExampleGenerator, &gen))
      return PYNULL;

    if (gen->domain->variables->size())
      PYERROR(PyExc_TypeError, ".basket format can only store meta-attribute values", PYNULL);

    FILE *ostr = openExtended(filename, "basket");
    if (!ostr)
      return PYNULL;

    set<int> missing;

    try {
      basket_writeExamples(ostr, gen, missing);
    }
    catch (...) {
      fclose(ostr);
      remove(filename);
      throw;
    }

    fclose(ostr);

    if (!missing.empty()) {
      if (missing.size() == 1) {
        char excbuf[512];
        snprintf(excbuf, 512, "saveBasket: attribute with id %i was not found in Domain and has not been stored", *(missing.begin()));
        raiseWarning(false, excbuf);
      }

      else {
        string misss;
        bool comma = false;
        const_ITERATE(set<int>, mi, missing) {
          if (comma)
            misss += ", ";
          else
            comma = true;

          char ns[20];
          sprintf(ns, "%i", (*mi));
          misss += ns;
        }

        char *excbuf = mlnew char[misss.length() + 128];
        sprintf(excbuf, "saveBasket: attributes with ids not found in Domain have not been stored (%s)", misss.c_str());
        try {
          raiseWarning(false, excbuf);
        }
        catch (...) {
          mldelete excbuf;
          throw;
        }

        mldelete excbuf;
      }
    }

    RETURN_NONE
  PyCATCH
}


#include "lib_io.px"
