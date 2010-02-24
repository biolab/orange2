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


#include "stladdon.hpp"
#include "orange.hpp"
#include "cls_misc.hpp"
#include "stringvars.hpp"
#include "domain.hpp"
#include "domaindepot.hpp"
#include "meta.hpp"
#include "externs.px"


DATASTRUCTURE(DomainDepot, TPyDomainDepot, 0)
NO_PICKLE(DomainDepot)

void DomainDepot_dealloc(TPyDomainDepot *self)
{ 
  mldelete self->domainDepot;
  /* Should not call tp_free if it is a reference 
     Destructor is also called by exit proc, not by wrapped */
  if (PyObject_IsPointer(self)) {
    PyObject_GC_UnTrack((PyObject *)self);
    self->ob_type->tp_free((PyObject *)self); 
  }
}


int DomainDepot_traverse(TPyDomainDepot *self, visitproc visit, void *arg)
{ return 0; }


int DomainDepot_clear(TPyDomainDepot *self)
{ return 0; }


PyObject *DomainDepot_new(PyTypeObject *type, PyObject *args, PyObject *) BASED_ON(ROOT, "() -> DomainDepot")
{
  PyTRY
    if (args && PyTuple_Size(args))
      PYERROR(PyExc_TypeError, "no arguments expected", PYNULL);

    TPyDomainDepot *dd = PyObject_GC_New(TPyDomainDepot, type);
    if (!dd)
      return PYNULL;

    dd->domainDepot = mlnew TDomainDepot;

    PyObject_GC_Track(dd);
    return (PyObject *)dd;
  PyCATCH
}


bool convertMetasFromPython(PyObject *dict, TMetaVector &metas);

bool decodeDescriptors(PyObject *pynames,
                       TDomainDepot::TAttributeDescriptions &attributeDescriptions,
                       TDomainDepot::TAttributeDescriptions &metaDescriptions,
                       bool &hasClass)
{
  if (!PyList_Check(pynames))
    PYERROR(PyExc_TypeError, "list of attribute names expected", PYNULL);

  hasClass = false;
  TDomainDepot::TAttributeDescription classDescription("", -1);

  for(Py_ssize_t i = 0, e = PyList_Size(pynames); i<e; i++) {
    PyObject *name = PyList_GetItem(pynames, i);
    if (!PyString_Check(name)) {
      PyErr_Format(PyExc_TypeError, "name at index %i is not a string", i);
      return false;
    }

    char *cname = PyString_AsString(name);

    if (strlen(cname) < 3) {
      PyErr_Format(PyExc_TypeError, "invalid format for attribute %i", i);
      return false;
    }

    int c = 0;
    TDomainDepot::TAttributeDescription *description;
    if (*cname == 'm') {
      metaDescriptions.push_back(TDomainDepot::TAttributeDescription("", -1));
      description = &metaDescriptions.back();
      c++;
    }
    else if (*cname == 'c') {
      if (hasClass)
        PYERROR(PyExc_TypeError, "more than one attribute is marked as class attribute", false);

      description = &classDescription;
      c++;
      hasClass = true;
    }
    else {
      attributeDescriptions.push_back(TDomainDepot::TAttributeDescription("", -1));
      description = &attributeDescriptions.back();
    }

   switch (cname[c]) {
      case 'C': description->varType = TValue::FLOATVAR; break;
      case 'D': description->varType = TValue::INTVAR; break;
      case 'S': description->varType = STRINGVAR; break;
      case '?': description->varType = -1; break;
      default:
        PyErr_Format(PyExc_TypeError, "name at index %i has invalid type characted ('%c')", i, cname[1]);
        return false;
    }

    if (cname[++c] != '#') {
      PyErr_Format(PyExc_TypeError, "invalid format for attribute %i", i);
      return false;
    }

    description->name = cname+c+1;
  }

  if (hasClass)
    attributeDescriptions.push_back(classDescription);

  return true;
}


PyObject *codeMetaIDs(int *&metaIDs, const int &size)
{
  PyObject *pyMetaIDs = PyList_New(size);
  int *mi = metaIDs;
  for(int i = 0; i<size; i++, mi++)
    PyList_SetItem(pyMetaIDs, i, PyInt_FromLong(*mi));

  mldelete metaIDs;
  metaIDs = NULL;

  return pyMetaIDs;
}

PyObject *DomainDepot_checkDomain(PyObject *self, PyObject *args, PyObject *) PYARGS(METH_VARARGS, "(domain, list-of-names)")
{
  PyTRY
    PDomain domain;
    PyObject *pynames;
    TDomainDepot::TAttributeDescriptions attributeDescriptions, metaDescriptions;
    bool hasClass;
    if (   !PyArg_ParseTuple(args, "O&O", &cc_Domain, &domain, &pynames)
        || !decodeDescriptors(pynames, attributeDescriptions, metaDescriptions, hasClass))
      return PYNULL;

    int *metaIDs = mlnew int[metaDescriptions.size()];
    TDomainDepot::TPAttributeDescriptions adescs, mdescs;
    TDomainDepot::pattrFromtAttr(attributeDescriptions, adescs);
    TDomainDepot::pattrFromtAttr(metaDescriptions, mdescs);
    bool domainOK = TDomainDepot::checkDomain(domain.getUnwrappedPtr(), &adescs, hasClass, &mdescs, metaIDs);
    return Py_BuildValue("iN", domainOK ? 1: 0, codeMetaIDs(metaIDs, metaDescriptions.size()));
  PyCATCH
}
    
    
PyObject *encodeStatus(const vector<int> &status)
{
  PyObject *pystatus = PyList_New(status.size());
  int i = 0;
  const_ITERATE(vector<int>, si, status)
    PyList_SetItem(pystatus, i++, PyInt_FromLong(*si));
  return pystatus;
}


PyObject *encodeStatus(const vector<pair<int, int> > &metaStatus)  
{
  PyObject *pymetastatus = PyDict_New();
  for(vector<pair<int, int> >::const_iterator mii(metaStatus.begin()), mie(metaStatus.end()); mii != mie; mii++) {
    PyObject *id = PyInt_FromLong(mii->first);
    PyObject *status = PyInt_FromLong(mii->second);
    PyDict_SetItem(pymetastatus, id, status);
    Py_DECREF(id);
    Py_DECREF(status);
  }
  return pymetastatus;
}

    
PyObject *DomainDepot_prepareDomain(PyObject *self, PyObject *args, PyObject *) PYARGS(METH_VARARGS, "(list-of-names[, knownVars[, knownMetas[, dont-store[, dont-check-stored]]]])")
{
  PyTRY
    PyObject *pynames = NULL;
    int createNewOn = TVariable::Incompatible;

    TDomainDepot::TAttributeDescriptions attributeDescriptions, metaDescriptions;
    bool hasClass;

    if (   !PyArg_ParseTuple(args, "O|i:DomainDepot.prepareDomain", &pynames, &createNewOn)
        || !decodeDescriptors(pynames, attributeDescriptions, metaDescriptions, hasClass))
      return PYNULL;

    vector<int> status;
    vector<pair<int, int> > metaStatus;
    TDomainDepot::TPAttributeDescriptions adescs, mdescs;
    TDomainDepot::pattrFromtAttr(attributeDescriptions, adescs);
    TDomainDepot::pattrFromtAttr(metaDescriptions, mdescs);
    PDomain newDomain = ((TPyDomainDepot *)(self))->domainDepot->prepareDomain(&adescs, hasClass, &mdescs, createNewOn, status, metaStatus);

    return Py_BuildValue("NNN", WrapOrange(newDomain), encodeStatus(status), encodeStatus(metaStatus));
  PyCATCH
}

#include "cls_misc.px"
