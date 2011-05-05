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


#ifndef __LIB_KERNEL_HPP
#define __LIB_KERNEL_HPP

#include "vars.hpp"
#include "domain.hpp"
#include "examples.hpp"
#include "examplegen.hpp"
#include "table.hpp"

class TFiletypeDefinition {
public:
  string name;
  TStringList extensions;
  PyObject *loader;
  PyObject *saver;

  TFiletypeDefinition(const char *, PyObject *, PyObject *);
  TFiletypeDefinition(const TFiletypeDefinition &);
  ~TFiletypeDefinition();
};

ORANGE_API TExampleTable *readListOfExamples(PyObject *args);
ORANGE_API TExampleTable *readListOfExamples(PyObject *args, PDomain, bool filterMetas = false);
ORANGE_API PExampleGenerator exampleGenFromArgs(PyObject *args, int &weightID);
ORANGE_API PExampleGenerator exampleGenFromArgs(PyObject *args);
ORANGE_API PExampleGenerator exampleGenFromParsedArgs(PyObject *args);
ORANGE_API bool varListFromDomain(PyObject *boundList, PDomain domain, TVarList &boundSet, bool allowSingle=true, bool checkForIncludance=true);
ORANGE_API bool varListFromVarList(PyObject *boundList, PVarList varlist, TVarList &boundSet, bool allowSingle = true, bool checkForIncludance = true);
ORANGE_API PVariable varFromArg_byDomain(PyObject *obj, PDomain domain=PDomain(), bool checkForIncludance = false);
ORANGE_API PVariable varFromArg_byVarList(PyObject *obj, PVarList varlist, bool checkForIncludance = false);
ORANGE_API bool convertFromPythonWithVariable(PyObject *obj, string &str);
ORANGE_API bool varNumFromVarDom(PyObject *pyvar, PDomain domain, int &);


ORANGE_API bool convertFromPythonWithML(PyObject *obj, string &str, const TOrangeType &base);

inline bool exampleGenFromParsedArgs(PyObject *args, PExampleGenerator &gen)
{ gen = exampleGenFromParsedArgs(args);
  return bool(gen);
}

ORANGE_API int pt_ExampleGenerator(PyObject *args, void *egen);

typedef int (*converter)(PyObject *, void *);
ORANGE_API converter ptd_ExampleGenerator(PDomain domain);

ORANGE_API bool weightFromArg_byDomain(PyObject *pyweight, PDomain domain, int &weightID);
ORANGE_API converter pt_weightByGen(PExampleGenerator &peg);

ORANGE_API int pt_DomainContingency(PyObject *args, void *egen);
ORANGE_API int ptn_DomainContingency(PyObject *args, void *egen);

ORANGE_API void registerFileType(const char *, const char *[], PyObject *, PyObject *);

#endif
