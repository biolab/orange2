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


#include <string>
#include <stdio.h>

#include "values.hpp"
#include "vars.hpp"
#include "domain.hpp"
#include "examples.hpp"
#include "domaindepot.hpp"

#include "c45inter.ppp"

bool readC45Atom(TFileExampleIteratorData &fei, vector<string> &atoms);

TC45ExampleGenerator::TC45ExampleGenerator(const string &datafile, const string &domainfile, const int createNewOn, vector<int> &status, vector<pair<int, int> > &metaStatus)
: TFileExampleGenerator(datafile, PDomain()),
  skip (mlnew TBoolList())
{ // domain needs to be initialized after skip!
  domain = readDomain(domainfile, createNewOn, status, metaStatus);
}


TC45ExampleGenerator::TC45ExampleGenerator(const TC45ExampleGenerator &old)
: TFileExampleGenerator(old),
  skip(CLONE(TBoolList, old.skip))
{}


/* Reads one line of the file. Atoms are converted to example values using str2val_add methods of corresponding
   variables and skipping the attributes with 'skip' flag set ('domain' field is cast to TC45Domain). */
bool TC45ExampleGenerator::readExample(TFileExampleIteratorData &fei, TExample &exam)
{
  vector<string> atoms;
  while(!feof(fei.file) && !readC45Atom(fei, atoms));

  if (!atoms.size())
    return false;

  TExample::iterator ei = exam.begin();
  TVarList::iterator vi(domain->variables->begin()), ve(domain->variables->end());
  vector<string>::iterator ai(atoms.begin()), ae(atoms.end());
  TBoolList::iterator si(skip->begin());
  for (; (vi!=ve) && (ai!=ae); ai++)
    if (!*si++)
      (*(vi++))->str2val_add(*ai, *(ei++));

  if ((vi!=ve) || (ai!=ae))
    raiseError("invalid length of example");

  return true;
}


// Reads the .names file. The format allow using different delimiters, not just those specified by the original format
PDomain TC45ExampleGenerator::readDomain(const string &stem, const int createNewOn, vector<int> &status, vector<pair<int, int> > &metaStatus)
{ TFileExampleIteratorData fei(stem);
  
  vector<string> atoms;
  while(!feof(fei.file) && !readC45Atom(fei, atoms));
  if (!atoms.size())
    ::raiseError("empty or invalid names file");

  TDomainDepot::TAttributeDescriptions attributeDescriptions;
  TDomainDepot::TAttributeDescription classDescription("y", TValue::INTVAR);

  for(vector<string>::iterator ai(atoms.begin()), ei(atoms.end()); ai!=ei; ai++)
    classDescription.addValue(*ai);
  
  do {
    while(!feof(fei.file) && !readC45Atom(fei, atoms));
    if (!atoms.size())
      break;
    if (atoms.size()<2)
      ::raiseError("invalid .names file");

    vector<string>::iterator ai(atoms.begin());
    string name = *(ai++);

    if (*ai=="ignore")
      skip->push_back(true);
    else {
      skip->push_back(false);

      if ((ai==atoms.end()) || (string((*ai).begin(), (*ai).begin()+9)=="discrete "))
        attributeDescriptions.push_back(TDomainDepot::TAttributeDescription(name, TValue::INTVAR));
      else if (*ai=="continuous")
        attributeDescriptions.push_back(TDomainDepot::TAttributeDescription(name, TValue::FLOATVAR));
      else {
        attributeDescriptions.push_back(TDomainDepot::TAttributeDescription(name, TValue::INTVAR));
        TDomainDepot::TAttributeDescription &desc = attributeDescriptions.back();
        while(ai!=atoms.end())
          desc.addValue(*ai++);
      }
    }
  } while (!feof(fei.file));
            
  if (!attributeDescriptions.size())
    ::raiseError("names file contains no variables but class variable");

  attributeDescriptions.push_back(classDescription);
  skip->push_back(false);

  TDomainDepot::TPAttributeDescriptions adescs;
  TDomainDepot::pattrFromtAttr(attributeDescriptions, adescs);
  return domainDepot.prepareDomain(&adescs, true, NULL, createNewOn, status, metaStatus);
}


bool writeValues(FILE *file, PVariable var, bool justDiscrete=false)
{
  TEnumVariable *enumv = var.AS(TEnumVariable);
  if (enumv) {
    if (justDiscrete)
      fprintf(file, "discrete 20.\n");
    else {
      TValue val;
      string sval;
      if (!enumv->firstValue(val))
        fprintf(file, "discrete 20.\n");
      else {
        enumv->val2str(val, sval); 
        fprintf(file, sval.c_str());
        while(enumv->nextValue(val)) {
          enumv->val2str(val, sval);
          fprintf(file, ", %s", sval.c_str());
        }
      }
      fprintf(file, ".\n");
    }

    return true;
  }

  fprintf(file, "continuous.\n");
  return false;
}

void c45_writeDomain(FILE *file, PDomain dom)
{ 
  if (!dom->classVar)
    raiseErrorWho("c45_writeDomain", "C4.5 format cannot store data sets without a class attribute");

  fprintf(file, "| Names file for %s\n", dom->classVar->name.c_str());
  if (!writeValues(file, dom->classVar))
    raiseErrorWho("c45_writeDomain", "C4.5 format cannot store a data set with non-discrete class attribute");
  
  const_PITERATE(TVarList, vi, dom->attributes) {
    fprintf(file, "%s: ", (*vi)->name.c_str());
    writeValues(file, *vi, true);
  }
}


void c45_writeExample(FILE *file, const TExample &ex)
{
  TVarList::const_iterator vi (ex.domain->variables->begin());
  TExample::const_iterator ri (ex.begin()), ei(ex.end());
  string st;
  if ((*ri).isSpecial())
    fprintf(file, "?");
  else {
    (*vi)->val2str(*ri, st);
    fprintf(file, st.c_str());
  }
  
  for(ri++, vi++; ri!=ei; ri++, vi++)
    if ((*ri).isSpecial())
      fprintf(file, ", ?");
    else {
      (*vi)->val2str(*ri, st);
      fprintf(file, ", %s", st.c_str());
    }
  fprintf(file, ".\n");
}


void c45_writeExamples(FILE *file, PExampleGenerator rg)
{ 
  PEITERATE(gi, rg)
    c45_writeExample(file, *gi);
}



/*  Divides the line onto atoms which are separated by commas and colons. Escape sequences are ignored, but \\
    is replaced by \. If | is encountered, the rest of the line is ignored. A space can be a part of the atom,
    multiple spaces are replaced by a single space. A dot can also be a part of the name; if dot is followed only
    by white-space, it is recognized as an end-of-line sign and is not added to the name. */
#define MAX_LINE_LENGTH 10240
bool readC45Atom(TFileExampleIteratorData &fei, vector<string> &atoms)
{
  atoms.clear();

  char line[MAX_LINE_LENGTH], *curr=line;

  fei.line++;
  if (!fgets(line, MAX_LINE_LENGTH, fei.file)) {
    if (feof(fei.file))
      return false;
    raiseErrorWho("C45ExampleGenerator", "error while reading line %i of file '%s'", fei.line, fei.filename.c_str());
  }

  if (strlen(line)>=MAX_LINE_LENGTH-1)
    raiseErrorWho("C45ExampleGenerator", "line %i of file '%s' too long", fei.line, fei.filename.c_str());

  for(;*curr && (*curr<=' ');curr++); // skip whitespace

  string atom;
  while (*curr) {
    switch(*curr) {
      case '|' :
        if (atom.length())
          atoms.push_back(atom);  // end of line
        return atoms.size()>0;

      case '\\':
        if (*++curr)
          atom += curr;
        break;

      case ',' :
      case ':' :
        atoms.push_back(atom);
        atom = string();
        while( *++curr && (*curr<=' '));
        break;

      case 13:
      case 10:
      case '.':
        if (*++curr<=' ') {
          if (atom.length())
            atoms.push_back(atom);
          atom = string();
          return atoms.size()>0;
        }
        else
          atom+='.';
        break;

      case ' ':
        atom+=*curr;
        while(*++curr==' ');
        break;

      default:
        if (*curr>' ')
          atom += *curr;
        curr++;
    };
  }

  if (atom.length())
    atoms.push_back(atom);

  return atoms.size()>0;
}   
