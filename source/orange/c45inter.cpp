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


#include <string>
#include <stdio.h>
#include "errors.hpp"

#include "values.hpp"
#include "vars.hpp"
#include "domain.hpp"
#include "examples.hpp"

#include "c45inter.ppp"


TC45ExampleGenerator::TC45ExampleGenerator(const string &afname, PDomain dom)
  : TFileExampleGenerator(afname, dom)
  {}
  

/* Reads one line of the file. Atoms are converted to example values using str2val_add methods of corresponding
   variables and skipping the attributes with 'skip' flag set ('domain' field is casted to TC45Domain). */
bool TC45ExampleGenerator::readExample(TFileExampleIteratorData &fei, TExample &exam)
{
  TC45Domain *mydomain;
  domain.dynamic_cast_to(mydomain);
  if (!mydomain)
    raiseError("'domain' should be derived from C45Domain");

  TIdList atoms;
  while(!feof(fei.file) && !readC45Atom(fei, atoms));

  if (!atoms.size())
    return false;

  TExample::iterator ei = exam.begin();
  TVarList::iterator vi(mydomain->variables->begin());
  TIdList::iterator ai(atoms.begin());
  vector<bool>::iterator si(mydomain->skip->begin());
  for (; (vi!=domain->variables->end()) && (ai!=atoms.end()); ai++)
    if (!*si++) 
      (*(vi++))->str2val_add(*ai, *(ei++));

  if ((vi!=domain->variables->end()) || (ai!=atoms.end()))
    raiseError("invalid length of example");

  return true;
}


TC45Domain::TC45Domain(const TC45Domain &old)
: TDomain(old),
  skip(CLONE(TBoolList, old.skip))
{}

/* Reads the .names file. The format does not exactly follow Quinlan's specifications (that is, a file, using
   wrong delimiters can be read). However, when writing C4.5 files, they are written correctly.  */
TC45Domain::TC45Domain(const string &stem, PVarList knownVars)
: TDomain(),
  skip(mlnew TBoolList())
{ TFileExampleIteratorData fei(stem);
  
  TIdList atoms;
  while(!feof(fei.file) && !readC45Atom(fei, atoms));
  if (!atoms.size())
    raiseError("empty or invalid names file");

  classVar=makeVariable("y", knownVars, TValue::INTVAR);
  TEnumVariable *evar=classVar.AS(TEnumVariable);
  { for(TIdList::iterator ai(atoms.begin()), ei(atoms.end()); ai!=ei; ) 
      evar->addValue(*(ai++)); 
  }

  do {
    while(!feof(fei.file) && !readC45Atom(fei, atoms));
    if (!atoms.size()) break;
    if (atoms.size()<2)
      raiseError("invalid names file");

    TIdList::iterator ai(atoms.begin());
    string name=*(ai++);

    if (*ai=="ignore")
      skip->push_back(true);
    else {
      skip->push_back(false);

      if ((ai==atoms.end()) || (string((*ai).begin(), (*ai).begin()+9)=="discrete "))
        attributes->push_back(makeVariable(name, knownVars, TValue::INTVAR));
      else if (*ai=="continuous")
        attributes->push_back(makeVariable(name, knownVars, TValue::FLOATVAR));
      else {
        attributes->push_back(makeVariable(name, knownVars, TValue::INTVAR));
        evar = attributes->back().AS(TEnumVariable);
        while(ai!=atoms.end())
          evar->addValue(*(ai++));
      }
    }
  } while (!feof(fei.file));
            
  if (!attributes->size())
    raiseError("names file contains no variables but class variable");

  skip->push_back(false); // for class
  variables=mlnew TVarList(attributes.getReference());
  variables->push_back(classVar);
}



bool writeValues(FILE *file, PVariable var, bool justDiscrete=false)
{
  TEnumVariable *enumv=NULL;
  var.dynamic_cast_to(enumv);
 
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
  fprintf(file, "| Names file for %s\n", dom->classVar->name.c_str());
  if (!writeValues(file, dom->classVar))
     fprintf(file, "| Error: class variable %s is not discrete\n", dom->classVar->name.c_str());
  
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
bool readC45Atom(TFileExampleIteratorData &fei, TIdList &atoms)
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
