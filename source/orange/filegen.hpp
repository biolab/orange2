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


#ifndef __FILEGEN_HPP
#define __FILEGEN_HPP

#include <string>
#include <list>

#include "examplegen.hpp"

bool skipNonEmptyLine(FILE *file, const char *filename, const char &commentChar);

class TFileExampleIteratorData {
public:
  FILE *file;
  const string &filename;
  int line;

  TFileExampleIteratorData(const string &name, const int &startDataPos = 0, const int &startDataline = 0);
  TFileExampleIteratorData(FILE *, const string &name, const int &line);
  TFileExampleIteratorData(const TFileExampleIteratorData &);
  ~TFileExampleIteratorData();
};


/*  A generator which retrieves examples from the file. It has an abstract
    method for reading a TExample; by defining it, descendants of
    TFileExampleGenerator can read different file formats. */
class TFileExampleGenerator : public TExampleGenerator {
public:
  __REGISTER_ABSTRACT_CLASS

  string filename; //P filename
  int startDataPos; //P starting position of the data in file
  int startDataLine; //P line in the file where the data starts

  TFileExampleGenerator(const string &, PDomain &);

  virtual TExampleIterator begin();
  virtual TExampleIterator begin(TExampleIterator &);
  virtual bool randomExample(TExample &);

  virtual int numberOfExamples();

protected:
  virtual void     increaseIterator(TExampleIterator &);
  virtual bool     sameIterators(const TExampleIterator &, const TExampleIterator &);
  virtual void     deleteIterator(TExampleIterator &source);
  virtual void     copyIterator(const TExampleIterator &source, TExampleIterator &dest);

  // An abstract method for reading examples. Derived classes must provide this method.
  virtual bool readExample (TFileExampleIteratorData &, TExample &)=0;
};


#define stringVarType 6

class TKnownVariables : public list<TVariable *> {
public:
  void add(PVariable wvar, TVariable::TDestroyNotifier *notifier);
};


/* Creates a variable with given name and type. */
PVariable createVariable(const string &name, const int &varType);

/* Tries to find a variable with given name and type in sourceDomain (variables and metas), sourceVars or storedVars.
   Any of these can be omitted. If variable is not found, createVariable is called to create a new one.
   If storedVars and destroyNotifier are given, the new variable is stored in storedVars */
PVariable makeVariable(const string &name, const int &varType, PVarList sourceVars, PDomain sourceDomain, TKnownVariables *storedVars, TVariable::TDestroyNotifier *);

/* Tries to find a variable the given name and type in knownVars or metaVector.
   Any of these (or both) can be omitted. If the variable is found in metaVector,
   the id is set as well; if not, id is set to 0. If the variable is not found,
   a new one is created unless dontCreateNew is set to false. */
PVariable makeVariable(const string &name, unsigned char varType, int &id, PVarList knownVars, const TMetaVector * = NULL, bool dontCreateNew = false);


/* Tries to find a variable with given name and type in sourceDomain (variables and metas), sourceVars, storedVars or storedDomains (metas only!)
   Any of these can be omitted. If variable is not found, createVariable is called to create a new one, unles dontCreateNew is true.
   If variable is found in metas, id is set. If storedVars and destroyNotifier are given, the new variable is stored in storedVars */
   
class TStringVariable;

template<class T>
PVariable makeVariable(const string &name, const int &varType, PVarList sourceVars, PDomain sourceDomain, TKnownVariables *storedVars, list<T *> *storedDomains, int &id, bool dontCreateNew, TVariable::TDestroyNotifier *notifier)
{ 
  if (sourceDomain) {
    PVariable var = makeVariable(name, varType, id, sourceDomain->variables, &sourceDomain->metas, true);
    if (var)
      return var;
  }

  if (sourceVars) {
    PVariable var = makeVariable(name, varType, id, sourceVars, NULL, true);
    if (var)
      return var;
  }

  id = 0;

  /* In domains, we only check metas.
     Ordinary attributes are retrieved through knownVariables */
  if (storedDomains)
    PITERATE(list<T *>, di, storedDomains)
      const_ITERATE(TMetaVector, mi, (*di)->metas)
        if (   ((*mi).variable->name == name)
            && (    (varType == -1)
                 || (varType==stringVarType) && (*mi).variable.is_derived_from(TStringVariable)
                 || ((*mi).variable->varType==varType))) {
          id = (*mi).id;
          return (*mi).variable;
      }

  if (storedVars)
    PITERATE(TKnownVariables, vi, storedVars)
      if (   ((*vi)->name==name)
          && (    (varType==-1)
               || (varType==stringVarType) && (dynamic_cast<TStringVariable *>(*vi) != NULL)
               || ((*vi)->varType==varType)))
        // The variable is rewrapped here (we have a pure pointer, but it has already been wrapped)
        return *vi;

  if (dontCreateNew)
    return PVariable();

  PVariable var = createVariable(name, varType);

  if (storedVars && notifier)
    storedVars->add(var.AS(TVariable), notifier);

  return var;
}


bool sameDomains(const TDomain *dom1, const TDomain *dom2);

#endif
