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

#include "retisinter.ppp"


TDomainDepot TRetisExampleGenerator::domainDepot;

TRetisExampleGenerator::TRetisExampleGenerator(const string &datafile, const string &domainfile, PVarList sourceVars, PDomain sourceDomain, bool dontCheckStored, bool dontStore)
: TFileExampleGenerator(datafile, readDomain(domainfile, sourceVars, sourceDomain, dontCheckStored, dontStore))
{}
  

string getLine(istream &str)
{ char line[1024];
  str.getline(line, 1024);
  if (str.gcount()==1024-1)
    raiseErrorWho("RetisExampleGenerator", "line too long");
  char *dele=line+strlen(line);
  while(*dele<' ')
    *(dele--)=0;
  return line;
}

// Overloaded to skip the first line of the file (number of examples)
TExampleIterator TRetisExampleGenerator::begin()
{ 
  #ifdef _MSC_VER
  // This is more efficient, but gcc doesn't like it...
  return TFileExampleGenerator::begin(TExampleIterator(domain, this, (void *)(mlnew TFileExampleIteratorData(filename, 1))));
  #else
  TExampleIterator it(domain, this, (void *)(mlnew TFileExampleIteratorData(filename, 1)));
  return TFileExampleGenerator::begin(it);
  #endif
}


// Reads one line of the file. Atoms are converted to example values using str2val_add methods of corresponding variables
bool TRetisExampleGenerator::readExample(TFileExampleIteratorData &fei, TExample &exam)
{ 
  char line[1024], *curr;

  while(!feof(fei.file)) {
    fei.line++;
    if (!fgets(line, 1024, fei.file)) {
      if (feof(fei.file))
        return false;
      raiseErrorWho("RetisExampleGenerator", "error while reading line %i of file '%s'", fei.line, fei.filename.c_str());
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

  linestr >> f;
  exam.setClass(TValue(f));
  if (!linestr.good())
    raiseError("error while reading examples from .rda file");

  TExample::iterator ei=exam.begin();
  for(TVarList::const_iterator vi(domain->attributes->begin()), ve(domain->attributes->end()); vi!=ve; vi++)
    if ((*vi)->varType==TValue::INTVAR) {
      linestr >> i;
      *(ei++)=TValue(i-1);
    }
    else {
      linestr >> f;
      *(ei++)=TValue(f);
    }
  if (!linestr.good())
    raiseError("error while reading examples from .rda file.");
  return true;
}


// Reads the .names file. The format allow using different delimiters, not just those specified by the original format
PDomain TRetisExampleGenerator::readDomain(const string &stem, PVarList sourceVars, PDomain sourceDomain, bool dontCheckStored, bool dontStore)
{ ifstream str(stem.c_str(), ios::binary);
  if (!str.is_open())
    ::raiseError("RetisDomain: file '%s' not found", stem.c_str());

  TDomainDepot::TAttributeDescriptions attributeDescriptions;

  string className = getLine(str);
  getLine(str); getLine(str);
  int noAttr = atoi(getLine(str).c_str());

  while(noAttr--) {
    string name = getLine(str);
    string type = getLine(str);
    if (type=="discrete") {
      attributeDescriptions.push_back(TDomainDepot::TAttributeDescription(name, TValue::INTVAR));
      TDomainDepot::TAttributeDescription &desc = attributeDescriptions.back();
      int noVals = atoi(getLine(str).c_str());
      while(noVals--)
        desc.addValue(getLine(str).c_str());
    }
    else if (type=="continuous") {
      attributeDescriptions.push_back(TDomainDepot::TAttributeDescription(name, TValue::FLOATVAR));
      getLine(str); getLine(str);
    }
    else
      ::raiseError("RetisDomain: invalid type ('%s') for attribute '%s'", type.c_str(), name.c_str());
  }

  attributeDescriptions.push_back(TDomainDepot::TAttributeDescription(className, TValue::FLOATVAR));

  if (sourceDomain) {
    if (!domainDepot.checkDomain(sourceDomain.AS(TDomain), &attributeDescriptions, true, NULL))
      raiseError("given domain does not match the file");
    else
      return sourceDomain;
  }

  return domainDepot.prepareDomain(&attributeDescriptions, true, NULL, sourceVars, NULL, dontStore, dontCheckStored);
}



void retis_writeDomain(FILE *file, PDomain dom)
{
  if (!dom || !dom->classVar || (dom->classVar->varType != TValue::FLOATVAR))
    raiseErrorWho("retis_writeDomain", "Retis format assumes continuous class attribute");

  fprintf(file, "%s\n5\n3\n%i\n", dom->classVar->name.c_str(), int(dom->attributes->size()));

  PITERATE(TVarList, vi, dom->attributes) {
    fprintf(file, "%s\n", (*vi)->name.c_str());

    if ((*vi)->varType == TValue::INTVAR) {
      fprintf(file, "discrete\n%i\n", (*vi)->name.c_str(), (*vi)->noOfValues());
      TValue val;
      string sval;
      if ((*vi)->firstValue(val))
        raiseErrorWho("retis_writeDomain", "attribute '%s' has no values", (*vi)->name.c_str());
      do {
        (*vi)->val2str(val, sval); 
        fprintf(file, "%s\n", sval.c_str());
      } while((*vi)->nextValue(val));
    }
    else
      fprintf(file, "continuous\n5\n3\n");
  }
}


void retis_writeExample(FILE *file, const TExample &ex)
{
  fprintf(file, "%5.3", float(ex.getClass()));

  TVarList::const_iterator vi(ex.domain->attributes->begin()), ve(ex.domain->attributes->end());
  TExample::const_iterator ri(ex.begin());
  for(; vi!=ve; ri++, vi++) {
    string st;
    if ((*ri).isSpecial())
      fprintf(file, " ?");
    else 
      if ((*vi)->varType==TValue::INTVAR)
        fprintf(file, " %i", int(*ri)+1);
      else
        fprintf(file, " %5.3f", float(*ri));
  }
  fprintf(file, "\n");
}

void retis_writeExamples(FILE *file, PExampleGenerator rg)
{ 
  int noex = rg->numberOfExamples();
  if (noex<0)
    raiseErrorWho("assistant_writeExamples", "cannot determine the number of examples that the generator will generate");

  fprintf(file, "%i", noex);

  PEITERATE(gi, rg)
    retis_writeExample(file, *gi);
}
