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


#include <fstream>
#include <strstream>

#include "values.hpp"
#include "vars.hpp"
#include "domain.hpp"
#include "examples.hpp"

#include "assistant.ppp"


TAssistantExampleGenerator::TAssistantExampleGenerator(const string &afname, PDomain dom)
  : TFileExampleGenerator(afname, dom)
  {}
  

// Overloaded to skip the first line of the file (number of examples)
TExampleIterator TAssistantExampleGenerator::begin()
{ 
  #ifdef _MSC_VER
  // this is more efficient, but gcc seems to dislike it...
  return TFileExampleGenerator::begin(TExampleIterator(domain, this, (void *)(mlnew TFileExampleIteratorData(filename, 1))));
  #else
  TExampleIterator it(domain, this, (void *)(mlnew TFileExampleIteratorData(filename, 1)));
  return TFileExampleGenerator::begin(it);
  #endif  
}


// Reads a line of the file. Atoms are converted to example values using str2val_add methods of corresponding variables
bool TAssistantExampleGenerator::readExample(TFileExampleIteratorData &fei, TExample &exam)
{ char line[1024], *curr;

  TAssistantDomain *mydomain = domain.AS(TAssistantDomain);
  if (!mydomain)
    raiseError("'domain' should be derived from AssistantDomain.");

  while(!feof(fei.file)) {
    fei.line++;
    if (!fgets(line, 32768, fei.file)) {
      if (feof(fei.file))
        return false;
      raiseErrorWho("TabDelimExampleGenerator", "error while reading line %i of file '%s'", fei.line, fei.filename.c_str());
    }
    if (strlen(line)>=1024-1)
      raiseError("line %i of file '%s' too long", fei.line, fei.filename.c_str());

    curr = line;
    while ((*curr) && (*curr<=' '))
      curr++;
    if (*curr)
      break;
  }

  if (feof(fei.file))
    return false;

  strstream linestr(line, 1024, ios_base::out);

  int i;
  float f;

  linestr >> i;
  exam = TExample(domain);
  exam.setClass(TValue(i-1));
  if (!linestr.good())
    raiseError("error while reading line %i of file '%s'example", fei.line, fei.filename.c_str());

  TExample::iterator ei = exam.begin();
  vector<vector<float> *>::const_iterator ri(mydomain->intervals.begin());
  for(TVarList::iterator vi(domain->attributes->begin()), ve(domain->attributes->end()); vi!=ve; vi++, ri++, ei++)
    if ((*vi)->varType==TValue::INTVAR)
	    if (*ri) {
  	    linestr >> f;
	      vector<float>::const_iterator rri((*ri)->begin()), eri((*ri)->end());
        for(; (rri!=eri) && (f>*rri); rri++);
		    *ei=TValue(int(rri-(*ri)->begin()));
	    }
	    else {
          linestr >> i;
          if (i>(*vi).AS(TEnumVariable)->noOfValues())
            *ei=(*vi)->DC();
		      else
            *ei=TValue(int(i-1));
	    }
    else {
      linestr >> f;
      *ei=TValue(f);
    }

  if (!linestr.good())
    raiseError("error while reading line %i of file '%s'example", fei.line, fei.filename.c_str());
  return true;
}


// This is defined in retis.cpp
string getLine(istream &str);

TAssistantDomain::TAssistantDomain(const string &stem, PVarList knownVars) : TDomain()
{ ifstream str(stem.c_str(), ios::binary);
  if (!str.is_open())
    raiseError("cannot open file '%s'", stem.c_str());

  classVar = makeVariable(getLine(str), knownVars, TValue::INTVAR);
  variables->push_back(classVar);
  TEnumVariable *evar = classVar.AS(TEnumVariable);
  for(int noval = atoi(getLine(str).c_str()); noval; noval--)
    evar->addValue(getLine(str).c_str());

  int noAttr = atoi(getLine(str).c_str());
  intervals = vector<vector<float> *>(noAttr);
  fill(intervals.begin(), intervals.end(), (vector<float> *)NULL);
  vector<vector<float> *>::iterator ri = intervals.begin();

  while(noAttr--) {
    string name = getLine(str);
	  int values = atoi(getLine(str).c_str());
	  if (values>0) {
      PVariable newvar = makeVariable(name, knownVars, TValue::INTVAR);
      evar = newvar.AS(TEnumVariable);
      while(values--)
        evar->addValue(getLine(str).c_str());
      addVariable(newvar);
    }
	  else if (values<0) {
      PVariable newvar = makeVariable(name, knownVars, TValue::INTVAR);
      evar = newvar.AS(TEnumVariable);
      *ri = mlnew vector<float>();
      char buf[128];
	    while(values++) {
        (*ri)->push_back(atof(getLine(str).c_str()));
        sprintf(buf, "v%i", (*ri)->size());
        evar->addValue(buf);
	    }
      sprintf(buf, "v%i", (*ri)->size()+1);
      evar->addValue(buf);
      addVariable(newvar);
    }
    else
      addVariable(makeVariable(name, knownVars, TValue::FLOATVAR));

    ri++;
  }
}


void assistant_writeEnumAttribute(FILE *file, PVariable enumv)
{
  if (!enumv->name.length())
    raiseErrorWho("assistant_writeDomain", "assistant format does not support anonimous attributes");

  fprintf(file, "%s\n%i\n", enumv->name.c_str(), enumv->noOfValues());

  TValue val;
  string sval;
  if (!enumv->firstValue(val))
    raiseErrorWho("assistant_writeDomain", "attribute '%s' has no values", enumv->name.c_str());
  do {
    enumv->val2str(val, sval); 
    fprintf(file, "%s\n", sval.c_str());
  } while(enumv->nextValue(val));
}


void assistant_writeDomain(FILE *file, PDomain dom)
{
  if (!dom || !dom->classVar || !dom->classVar->varType==TValue::INTVAR)
    raiseErrorWho("assistant_writeDomain", "assistant format assumes discrete class variable");

  assistant_writeEnumAttribute(file, dom->classVar);
    
  fprintf(file, "%i\n", dom->attributes->size());

  const_PITERATE(TVarList, vi, dom->attributes) {
    if ((*vi)->varType == TValue::INTVAR)
      assistant_writeEnumAttribute(file, *vi);
    else if ((*vi)->varType == TValue::FLOATVAR) {
      if (!(*vi)->name.length())
        raiseErrorWho("assistant_writeDomain", "assistant format does not support anonimous attributes");
      fprintf(file, "%s\n0\n", (*vi)->name.c_str());
    }
    else
      raiseErrorWho("assistant_writeDomain", "assistant format supports only discrete and continuous attributes");
  }
}


void assistant_writeExample(FILE *file, const TExample &ex)
{
  fprintf(file, "%i\n", int(ex.getClass())+1);

  TVarList::const_iterator vi(ex.domain->attributes->begin()), ve(ex.domain->attributes->end());
  TExample::const_iterator ri(ex.begin());
  for(; vi!=ve; ri++, vi++) {
    if ((*ri).isSpecial()) 
      fprintf(file, " ?");
    else if ((*vi)->varType==TValue::INTVAR)
      fprintf(file, " %i", int(*ri)+1);
    else
      fprintf(file, " %5.3f", float(*ri));
  }
  fprintf(file, "\n");
}

void assistant_writeExamples(FILE *file, PExampleGenerator rg)
{ 
  int noex = rg->numberOfExamples();
  if (noex<0)
    raiseErrorWho("assistant_writeExamples", "cannot determine the number of examples that the generator will generate");

  fprintf(file, "%i", noex);

  PEITERATE(gi, rg)
    assistant_writeExample(file, *gi);
}
