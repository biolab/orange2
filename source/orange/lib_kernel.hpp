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


#ifndef __LIB_KERNEL_HPP
#define __LIB_KERNEL_HPP

#include "vars.hpp"
#include "domain.hpp"
#include "examples.hpp"
#include "examplegen.hpp"
#include "table.hpp"

TExampleTable *readListOfExamples(PyObject *args);
TExampleTable *readListOfExamples(PyObject *args, PDomain, bool filterMetas = false);
PExampleGenerator exampleGenFromArgs(PyObject *args, int *weightID = NULL);
PExampleGenerator exampleGenFromParsedArgs(PyObject *args);
bool varListFromDomain(PyObject *boundList, PDomain domain, TVarList &boundSet, bool allowSingle=true, bool checkForIncludance=true);
bool varListFromVarList(PyObject *boundList, PVarList varlist, TVarList &boundSet, bool allowSingle = true, bool checkForIncludance = true);
PVariable varFromArg_byDomain(PyObject *obj, PDomain domain=PDomain(), bool checkForIncludance = false);
PVariable varFromArg_byVarList(PyObject *obj, PVarList varlist, bool checkForIncludance = false);
bool convertFromPythonWithVariable(PyObject *obj, string &str);
bool varNumFromVarDom(PyObject *pyvar, PDomain domain, int &);


bool convertFromPythonWithML(PyObject *obj, string &str, const TOrangeType &base);

inline bool exampleGenFromParsedArgs(PyObject *args, PExampleGenerator &gen)
{ gen = exampleGenFromParsedArgs(args);
  return bool(gen);
}

int pt_ExampleGenerator(PyObject *args, void *egen);

typedef int (*converter)(PyObject *, void *);
converter ptd_ExampleGenerator(PDomain domain);

bool weightFromArg_byDomain(PyObject *pyweight, PDomain domain, int &weightID);
converter pt_weightByGen(PExampleGenerator &peg);

int pt_DomainContingency(PyObject *args, void *egen);

#endif
