/*
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


/********************************

This file includes constructors and specialized methods for ML* object, defined in project Io

*********************************/

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

/* ************ FILE EXAMPLE GENERATORS ************ */

#include "filegen.hpp"
BASED_ON(FileExampleGenerator, ExampleGenerator)

#include "tabdelim.hpp"
#include "c45inter.hpp"
#include "retisinter.hpp"
#include "assistant.hpp"


bool divDot(const string &name, string &before, string &after)
{ string::const_iterator bi(name.begin()), ei(name.end());
  for(; (ei!=bi) && (*(--ei)!='.'); );
  if (*ei!='.') return false;
  
  before=string(bi, ei); after=string(ei++, name.end());
  return true;
}

PyObject *TabDelimExampleGenerator_new(PyTypeObject *type, PyObject *args, PyObject *keywords) BASED_ON(FileExampleGenerator, "(examples[, use=domain|varlist])")
{ PyTRY
    char *fileName;
    if (!PyArg_ParseTuple(args, "s", &fileName))
      PYERROR(PyExc_TypeError, "TabDelimExampleGenerator expects a string argument", PYNULL)

    string name(fileName), b, a;
    if (!divDot(name, b, a))
      name+=".tab";

    return WrapNewOrange(mlnew TTabDelimExampleGenerator(name, false, knownVars(keywords), knownMetas(keywords), knownDomain(keywords), false, false), type);
  PyCATCH
}

PyObject *RetisExampleGenerator_new(PyTypeObject *type, PyObject *args, PyObject *keywords) BASED_ON(FileExampleGenerator, "(examples[, use=domain|varlist])")
{ PyTRY
    char *stem;
    if (!PyArg_ParseTuple(args, "s", &stem))
      PYERROR(PyExc_TypeError, "RetisExampleGenerator expects a string argument", PYNULL)
    
    string domain, data;
    string b, a;
    if (divDot(stem, b, a))
      { data=stem; domain=b+".rdo"; }
    else
      { data=string(stem)+".rda"; domain=string(stem)+".rdo"; }
      
    return WrapNewOrange(mlnew TRetisExampleGenerator(data, domain, knownVars(keywords), knownDomain(keywords), false, false), type);
  PyCATCH
}

PyObject *C45ExampleGenerator_new(PyTypeObject *type, PyObject *args, PyObject *keywords) BASED_ON(FileExampleGenerator, "(examples[, use=domain|varlist])")
{ PyTRY
    char *stem;
    if (!PyArg_ParseTuple(args, "s", &stem))
      PYERROR(PyExc_TypeError, "C45ExampleGenerator expects a string argument", PYNULL)

    string domain, data;
    string b, a;
    if (divDot(stem, b, a))
      { data=stem; domain=b+".names"; }
    else
      { data=string(stem)+".data"; domain=string(stem)+".names"; }

    return WrapNewOrange(mlnew TC45ExampleGenerator(data, domain, knownVars(keywords), knownDomain(keywords), false, false), type);
  PyCATCH
}


PyObject *AssistantExampleGenerator_new(PyTypeObject *type, PyObject *args, PyObject *keywords) BASED_ON(FileExampleGenerator, "(examples[, use=domain|varlist])")
{ PyTRY
    char *stem;
    if (!PyArg_ParseTuple(args, "s", &stem))
      PYERROR(PyExc_TypeError, "AssistantExampleGenerator expects a string argument", PYNULL)

    string domain, data;
    if (strlen(stem)<=4) // we guess this is the xxxx part of ASDAxxxx.DAT
      { domain="ASDO"+string(stem)+".DAT"; data="ASDA"+string(stem)+".DAT"; }
    else if (strncmp(stem, "ASDA", 4)==0)
      { domain="ASDO"+string(stem+4)+".DAT"; data=string(stem); }
    else if (strncmp(stem, "ASDO", 4)==0)
      { domain=string(stem); data="ASDA"+string(stem+4)+".DAT"; }
    else // this is a longer name, but starting with ASDA
      { domain="ASDO"+string(stem+4); data=string(stem); }

    return WrapNewOrange(mlnew TAssistantExampleGenerator(data, domain, knownVars(keywords), knownDomain(keywords), false, false), type);
  PyCATCH
}



int pt_ExampleGenerator(PyObject *args, void *egen);

void tabDelim_writeDomain(FILE *, PDomain, bool autodetect);
void tabDelim_writeExamples(FILE *, PExampleGenerator);

PyObject *saveTabDelimited(PyObject *, PyObject *args) PYARGS(METH_VARARGS, "(filename, examples) -> None")
{ PyTRY
    char *filename;
    PExampleGenerator gen;

    if (!PyArg_ParseTuple(args, "sO&", &filename, pt_ExampleGenerator, &gen))
      PYERROR(PyExc_TypeError, "string and example generator expected", PYNULL)
  
    FILE *ostr = fopen(filename, "wt");
    if (!ostr) {
      PyErr_Format(PyExc_SystemError, "cannot open file '%s'", filename);
      return PYNULL;
    }

    tabDelim_writeDomain(ostr, gen->domain, false);
    tabDelim_writeExamples(ostr, gen);
    fclose(ostr);

    RETURN_NONE
  PyCATCH
}


PyObject *saveTxt(PyObject *, PyObject *args) PYARGS(METH_VARARGS, "(filename, examples) -> None")
{ PyTRY
    char *filename;
    PExampleGenerator gen;

    if (!PyArg_ParseTuple(args, "sO&", &filename, pt_ExampleGenerator, &gen))
      PYERROR(PyExc_TypeError, "string and example generator expected", PYNULL)
  
    FILE *ostr = fopen(filename, "wt");
    if (!ostr) {
      PyErr_Format(PyExc_SystemError, "cannot open file '%s'", filename);
      return PYNULL;
    }

    tabDelim_writeDomain(ostr, gen->domain, true);
    tabDelim_writeExamples(ostr, gen);
    fclose(ostr);

    RETURN_NONE
  PyCATCH
}


void c45_writeDomain(FILE *, PDomain);
void c45_writeExamples(FILE *, PExampleGenerator);

PyObject *saveC45(PyObject *, PyObject *args) PYARGS(METH_VARARGS, "(filename, examples) -> None")
{ PyTRY
    char *filename;
    PExampleGenerator gen;

    if (!PyArg_ParseTuple(args, "sO&", &filename, pt_ExampleGenerator, &gen))
      PYERROR(PyExc_TypeError, "string and example generator expected", PYNULL)
  
    if (gen->domain->classVar->varType!=TValue::INTVAR)
      PYERROR(PyExc_SystemError, "Class in C4.5 must be discrete.", PYNULL);

    FILE *ostr = fopen((string(filename)+".names").c_str(), "wt");
    if (!ostr) {
      PyErr_Format(PyExc_SystemError, "cannot open file '%s.names'", filename);
      return PYNULL;
    }

    c45_writeDomain(ostr, gen->domain);
    fclose(ostr);


    ostr = fopen((string(filename)+".data").c_str(), "wt");
    if (!ostr) {
      PyErr_Format(PyExc_SystemError, "cannot open file '%s.data'", filename);
      return PYNULL;
    }

    c45_writeExamples(ostr, gen);
    fclose(ostr);

    RETURN_NONE
  PyCATCH
}


void assistant_writeDomain(FILE *, PDomain);
void assistant_writeExamples(FILE *, PExampleGenerator);

PyObject *saveAssistant(PyObject *, PyObject *args) PYARGS(METH_VARARGS, "(filename, examples) -> None")
{ PyTRY
    char *filename;
    PExampleGenerator gen;

    if (!PyArg_ParseTuple(args, "sO&", &filename, pt_ExampleGenerator, &gen))
      PYERROR(PyExc_TypeError, "string and example generator expected", PYNULL)
  
    if (gen->domain->classVar->varType!=TValue::INTVAR)
      PYERROR(PyExc_SystemError, "Class in assistant must be discrete.", PYNULL);

    FILE *ostr = fopen(("asdo" + string(filename)+".dat").c_str(), "wt");
    if (!ostr) {
      PyErr_Format(PyExc_SystemError, "cannot open file 'asdo%s.dat'", filename);
      return PYNULL;
    }

    assistant_writeDomain(ostr, gen->domain);
    fclose(ostr);


    ostr = fopen(("asda" + string(filename)+".dat").c_str(), "wt");
    if (!ostr) {
      PyErr_Format(PyExc_SystemError, "cannot open file 'asda%s.dat'", filename);
      return PYNULL;
    }

    assistant_writeExamples(ostr, gen);
    fclose(ostr);

    RETURN_NONE
  PyCATCH
}



void retis_writeDomain(FILE *, PDomain);
void retis_writeExamples(FILE *, PExampleGenerator);

#include "spec_gen.hpp"

PyObject *saveRetis(PyObject *, PyObject *args) PYARGS(METH_VARARGS, "(filename, examples) -> None")
{ PyTRY
    char *filename;
    PExampleGenerator gen;

    if (!PyArg_ParseTuple(args, "sO&", &filename, pt_ExampleGenerator, &gen))
      PYERROR(PyExc_TypeError, "string and example generator expected", PYNULL)
  
    if (gen->domain->classVar->varType!=TValue::FLOATVAR)
      PYERROR(PyExc_SystemError, "Class in Retis domain must be continuous.", PYNULL);

    TFilter_hasSpecial tfhs(true);
    PExampleGenerator filtered=mlnew TFilteredGenerator(PFilter(tfhs), gen);
    PExampleGenerator wnounk=mlnew TExampleTable(filtered);

    FILE *ostr = fopen((string(filename)+".rdo").c_str(), "wt");
    if (!ostr) {
      PyErr_Format(PyExc_SystemError, "cannot open file '%s.rdo'", filename);
      return PYNULL;
    }

    retis_writeDomain(ostr, wnounk->domain);
    fclose(ostr);


    ostr = fopen((string(filename)+".rda").c_str(), "wt");
    if (!ostr) {
      PyErr_Format(PyExc_SystemError, "cannot open file '%s.rda'", filename);
      return PYNULL;
    }

    c45_writeExamples(ostr, wnounk);
    fclose(ostr);

    RETURN_NONE
  PyCATCH
}


#include "lib_io.px"
