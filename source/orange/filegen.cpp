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

#include <stdio.h>
#include "errors.hpp"

#include "vars.hpp"
#include "domain.hpp"
#include "examples.hpp"
#include "stringvars.hpp"
#include "examplegen.hpp"

#include "filegen.ppp"



TFileExampleIteratorData::TFileExampleIteratorData(const string &aname, const int &startDataPos,  const int &startDataLine)
: file(NULL),
  filename(aname),
  line(startDataLine)
{ if (filename.size()) {
    file = fopen(filename.c_str(), "rb");
    if (!file)
      raiseErrorWho("FileExampleGenerator", "cannot open file '%s'", filename.c_str());

    if (startDataPos)
      fseek(file, startDataPos, SEEK_SET);

    if (ferror(file))
      raiseErrorWho("FileExampleGenerator", "error while reading '%s'", filename.c_str());
  }
}


TFileExampleIteratorData::TFileExampleIteratorData(FILE *fle, const string &name, const int &aline)
: file(fle),
  filename(name),
  line(aline)
{}


TFileExampleIteratorData::TFileExampleIteratorData(const TFileExampleIteratorData &old)
: file(NULL),
  filename(old.filename),
  line(old.line)
{
  if (old.file) {
    file = fopen(filename.c_str(), "rb");
    if (!file)
      raiseErrorWho("FileExampleGenerator", "cannot open file '%s'", filename.c_str());
    fseek(file, ftell(old.file), SEEK_SET);
  }
}


TFileExampleIteratorData::~TFileExampleIteratorData()
{ if (file)
    fclose(file);
}


// Skips empty lines and the first non-empty line of the stream
bool skipNonEmptyLine(FILE *file, const char *filename, const char &commentChar)
{ if (feof(file))
    return 0;

  char lne[10240], *curr;
  do {
    if (!fgets(lne, 10240, file)) {
      if (feof(file))
        return false;
      raiseErrorWho("FileExampleGenerator", "error while reading '%s'", filename);
    }
    if (strlen(lne)>=10240-1)
      raiseErrorWho("FileExampleGenerator", "error while reading '%s' (line too long)", filename);

    curr = lne;
    while (*curr && (*curr<=' '))
      curr++;
  } while (   !feof(file)
           && (!*curr || (*curr == commentChar)));
  return !feof(file);
}




TFileExampleGenerator::TFileExampleGenerator(const string &afname, PDomain dom)
: TExampleGenerator(dom),
  filename(afname),
  startDataPos(0),
  startDataLine(0)
{}

  
TExampleIterator TFileExampleGenerator::begin()
{ 
  #ifdef _MSC_VER
  // This is more efficient, but gcc seems to dislike it...
  return begin(TExampleIterator(domain, this, (void *)(mlnew TFileExampleIteratorData(filename, startDataPos, startDataLine))));
  #else
  TExampleIterator it(domain, this, (void *)(mlnew TFileExampleIteratorData(filename, startDataPos, startDataLine)));
  return begin(it);
  #endif
}



/* Reads the first example and initializes example and data fields.
   This method is used by derived objects which must skip first lines of the file (like TTabDelimitedGenerator) */
TExampleIterator TFileExampleGenerator::begin(TExampleIterator &fei)
{
  try {
    if (!readExample(*(TFileExampleIteratorData *)(fei.data), fei.privateExample))
      deleteIterator(fei);
  } catch (exception) {
    deleteIterator(fei);
    throw;
  }

  return fei;
}


bool TFileExampleGenerator::randomExample(TExample &)
{ return 0; }


int TFileExampleGenerator::numberOfExamples()
{ return NOEX_TRACTABLE; }



void TFileExampleGenerator::increaseIterator(TExampleIterator &i)
{ TFileExampleIteratorData *sru = (TFileExampleIteratorData *)(i.data);
  if (feof(sru->file) || !readExample(*sru, i.privateExample))
    deleteIterator(i);
}


/* Returns true if both iterators are at end or if the belong to the
   same generator and point to the same position in the file */
bool TFileExampleGenerator::sameIterators(const TExampleIterator &d1, const TExampleIterator &d2)
{ return d1.example ?    d2.example 
                      && (d1.generator == d2.generator)
                      && (ftell(((TFileExampleIteratorData *)(d1.data))->file) == ftell(((TFileExampleIteratorData *)(d2.data))->file))
                    :   !d2.example;
}


void TFileExampleGenerator::deleteIterator(TExampleIterator &i)
{ mldelete (TFileExampleIteratorData *)(i.data);
  TExampleGenerator::deleteIterator(i);
};


/* Unlike STL stream iterators copying this iterator opens a new file
   and seeks to the position of the the original iterator is. */
void TFileExampleGenerator::copyIterator(const TExampleIterator &source, TExampleIterator &dest)
{ TExampleGenerator::copyIterator(source, dest);
  if (source.data)
    dest.data = (void *)(mlnew TFileExampleIteratorData(*(TFileExampleIteratorData *)(source.data)));
  else
    dest.data = NULL;
}


TFileExampleGenerator::TAttributeDescription::TAttributeDescription(const string &n, const int &vt, bool ord)
: name(n),
  varType(vt),
  ordered(ord)
{}


bool TFileExampleGenerator::checkDomain(const TDomain *domain, 
                                        const TAttributeDescriptions *attributes,
                                        bool hasClass,
                                        const TAttributeDescriptions *metas,
                                        int *metaIDs)
{
  // check the number of attributes and meta attributes, and the presence of class attribute
  if (    (domain->variables->size() != attributes->size())
       || (bool(domain->classVar) != hasClass)
       || (metas ? (metas->size() != domain->metas.size()) : domain->metas.size() )
     )
    return false;

  // check the names and types of attributes
  TVarList::const_iterator vi(domain->variables->begin());
  const_PITERATE(TAttributeDescriptions, ai, attributes)
    if (    ((*ai).name != (*vi)->name)
         || ((*ai).varType<0) && ((*ai).varType != (*vi)->varType))
      return false;
    else
      vi++;

  // check the meta attributes if they exist
  if (metas)
    const_PITERATE(TAttributeDescriptions, mi, metas) {
      PVariable var = domain->getMetaVar((*mi).name, false);
      if (    !var
           || (((*mi).varType > 0) && ((*mi).varType != var->varType))
         )
        return false;
      if (metaIDs)
        *(metaIDs++) = domain->getMetaNum((*mi).name, false);
    }

  return true;
}


PDomain TFileExampleGenerator::prepareDomain(const TAttributeDescriptions *attributes,
                                             bool hasClass,
                                             const TAttributeDescriptions *metas,
                                             bool &domainIsNew,
                                             list<TDomain *> *knownDomains,
                                             PVarList knownVars,
                                             const TMetaVector *knownMetas,
                                             int *metaIDs)
{ if (knownDomains)
    PITERATE(list<TDomain *>, kdi, knownDomains)
      if (checkDomain(*kdi, attributes, hasClass, metas, metaIDs)) {
        domainIsNew = false;
        return *kdi;
      }

  TVarList attrList;
  int foo;
  const_PITERATE(TAttributeDescriptions, ai, attributes)
    attrList.push_back(makeVariable((*ai).name, (*ai).varType, (*ai).values, foo, knownVars, knownMetas, false, false));

  PDomain newDomain;

  PVariable classVar;
  if (hasClass) {
    classVar = attrList.back();
    attrList.erase(attrList.end()-1);
  }
  
  newDomain = mlnew TDomain(classVar, attrList);

  if (metas)
    const_PITERATE(TAttributeDescriptions, mi, metas) {
      int id;
      PVariable var = makeVariable((*mi).name, (*mi).varType, (*mi).values, id, knownVars, knownMetas, false, true);
      if (!id)
        id = getMetaID();
      newDomain->metas.push_back(TMetaDescriptor(id, var));
      if (metaIDs)
        *(metaIDs++) = id;
    }
    
  domainIsNew = true;
  return newDomain;
}


PVariable createVariable(const string &name, const int &varType, PStringList values)
{
  switch (varType) {
    case TValue::INTVAR:  return mlnew TEnumVariable(name, values ? values : PStringList(mlnew TStringList()));
    case TValue::FLOATVAR: return mlnew TFloatVariable(name);
    case stringVarType: return mlnew TStringVariable(name);
  }

  if (varType==-1)
    ::raiseErrorWho("makeVariable", "unknown type for attribute '%s'", name.c_str());

  return (TVariable *)NULL;
}


PVariable makeVariable(const string &name, unsigned char varType, PStringList values, int &id, PVarList knownVars, const TMetaVector *metas, bool dontCreateNew, bool preferMetas)
{ if (!preferMetas && knownVars)
    const_PITERATE(TVarList, vi, knownVars)
      if (   ((*vi)->name==name)
          && (    (varType==-1)
               || (varType==stringVarType) && (*vi).is_derived_from(TStringVariable)
               || ((*vi)->varType==varType))) {
        id = 0;
        return *vi;
      }

  if (metas)
    const_PITERATE(TMetaVector, mi, metas)
      if (   ((*mi).variable->name == name)
          && (    (varType == -1)
               || (varType==stringVarType) && (*mi).variable.is_derived_from(TStringVariable)
               || ((*mi).variable->varType==varType))) {
        id = (*mi).id;
        return (*mi).variable;
      }

  if (preferMetas && knownVars)
    const_PITERATE(TVarList, vi, knownVars)
      if (   ((*vi)->name==name)
          && (    (varType==-1)
               || (varType==stringVarType) && (*vi).is_derived_from(TStringVariable)
               || ((*vi)->varType==varType))) {
        id = 0;
        return *vi;
      }
  
  id = 0;

  return dontCreateNew ? PVariable() : createVariable(name, varType, values);
}
