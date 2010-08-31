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

#include <istream>
#include <string>

#include "stladdon.hpp"

#include "vars.hpp"
#include "domain.hpp"
#include "examplegen.hpp"
#include "learn.hpp"

#include "assoc.hpp"

#include "rule_conditions.hpp"

/*~***************************************************************************************
A condition for checking the presence of an attribute or attribute=value in an example
*****************************************************************************************/

TRuleCondAtom::TRuleCondAtom(int ana, int anv)
  : attrIndex(ana), valueIndex(anv)
  {};


bool TRuleCondAtom::operator()(PExample example)
{ return    (example->operator[](attrIndex).isSpecial())
         || (valueIndex>=0) && (example->operator[](attrIndex).intV!=valueIndex) ? 0 : 1; }



TRuleCondOneOf::TRuleCondOneOf(const vector<int> &attrs)
  : attrIndices(attrs)
  {};


bool TRuleCondOneOf::operator()(PExample example)
{ ITERATE(vector<int>, ai, attrIndices)
    if (!example->operator[](*ai).isSpecial())  return true;
  return false; }


/*~***************************************************************************************
A condition which counts the satisfied TRuleCondAtom conditions in a rule
*****************************************************************************************/

TRuleCondCounted::TRuleCondCounted(char as, int ano, char aside)
  : sign(as), occurences(ano), side(aside)
  {};


TRuleCondCounted::TRuleCondCounted(PDomain domain, istream &istr, const vector<pair<string, vector<int> > > &sets)
{ vector<string> atoms;
  if (istr.eof() || !readConditionAtoms(istr, atoms)) return;

  side=atoms.front()[0];
  if ((side!='l') && (side!='r') && (side!='b'))
    raiseErrorWho("ConditionSet", "invalid side specification");
  if (atoms.front().length()==1) {
    sign='=';
    occurences=1;
  }
  else {
    sign=atoms.front()[1];
    if (atoms.front()[2]=='=') {
      if (sign=='<') sign='a';
      else if (sign=='>') sign='i';
      else raiseErrorWho("ConditionSet", "invalid condition specification");

      if (atoms.front().length()==3) occurences=1;
     else sscanf(string(atoms.front().begin()+3, atoms.front().end()).c_str(), "%i", &occurences);
    }

    else {
      if (atoms.front().length()==2) occurences=1;
      else sscanf(string(atoms.front().begin()+2, atoms.front().end()).c_str(), "%i", &occurences);
    }
  }

  vector<string>::iterator ii(atoms.begin()+1);
  for( ; ii!=atoms.end(); ii++) {
    string atom=*ii;
    string::iterator ai(atom.begin());
    for(; (ai!=atom.end()) && (*ai!='='); ai++);
    if (ai==atom.end()) {
      int attrIndex=domain->getVarNum(atom, false);
      if (attrIndex>=0) atomConditions.push_back(mlnew TRuleCondAtom(attrIndex));
      else {
        vector<pair<string, vector<int> > >::const_iterator si(sets.begin());
        for( ; (si!=sets.end()) && ((*si).first!=atom); si++);
        if (si==sets.end()) raiseErrorWho("ConditionSet", "a set or an attribute named '%s' not found.", atom.c_str());
        else atomConditions.push_back(mlnew TRuleCondOneOf((*si).second));
      }
    } else {
      string aname(atom.begin(), ai);
      int attrIndex=domain->getVarNum(aname, false);
      if (attrIndex<0) raiseErrorWho("ConditionSet", "attribute '%s' not found", aname.c_str());

      TValue value;
      domain->variables->at(attrIndex)->str2val(string(ai+1, atom.end()), value);
      atomConditions.push_back(mlnew TRuleCondAtom(attrIndex, value.isSpecial() ? -1 : value.intV));
    }
  }
}


int TRuleCondCounted::count(PExample example)
{ int occs=0;
  for(vector<TCondition<PExample> *>::iterator ci(atomConditions.begin()); ci!=atomConditions.end(); occs+=(**(ci++))(example));
  return occs;
}


bool TRuleCondCounted::operator()(PAssociationRule asr)
{ int occs = -1;
  switch (side) {
    case 'l': occs = count(asr->left); break;
    case 'r': occs = count(asr->right); break;
    case 'b': occs = count(asr->left) + count(asr->right);
  }

  switch (sign) {
    case '*': return occs>=0 ? 1 : 0;
    case '>': return occs>occurences ? 1 : 0;
    case 'i': return occs>=occurences ? 1 : 0;
    case '=': return occs==occurences ? 1 : 0;
    case '<': return occs<occurences ? 1 : 0;
    case 'a': return occs<=occurences ? 1 : 0;
  }

  return false;
}


bool TRuleCondCounted::readConditionAtoms(istream &str, vector<string> &atoms)
{
  #define MAX_LINE_LENGTH 1024
  char line[MAX_LINE_LENGTH], *curr=line;
  str.getline(line, MAX_LINE_LENGTH);
  if (str.gcount()==MAX_LINE_LENGTH-1)
    raiseError("line too long");

  atoms=vector<string>();

  for(;*curr && (*curr<=' ');curr++); // skip whitespace

  string atom;
  while (*curr && (curr-line)<MAX_LINE_LENGTH) {
    switch(*curr) {
      case '|' : if (atom.length()) atoms.push_back(atom);  // end of line
                 return atoms.size()>0;
      case 13:
      case 10:   if (*++curr<=' ') {
                   if (atom.length()) atoms.push_back(atom);
                   atom=string();
                   return atoms.size()>0;
                 }
                 else atom+='.'; break;
      case ' ':  if (atom.length()) { atoms.push_back(atom); atom=""; }
                 while(*++curr==' ');
                 break;
      default:   if (*curr>' ') atom+=*curr;
                 curr++;
    };
  }
  if (atom.length()) atoms.push_back(atom);

  return atoms.size()>0;
  #undef MAX_LINE_LENGTH
}


/*~***************************************************************************************
A conjunction of count condition - defines one `type' of the rule
*****************************************************************************************/

TRuleCondConjunctions::TRuleCondConjunctions() {}

TRuleCondConjunctions::TRuleCondConjunctions(PDomain domain, istream &istr,
                                             const vector<pair<string, vector<int> > > &sets)
{ while(!istr.eof()) {
    TRuleCondCounted *newRule=mlnew TRuleCondCounted(domain, istr, sets);
    if (newRule->atomConditions.size()) push_back(newRule);
    else break;
  }
}


/*~***************************************************************************************
A disjunction of `types' of rules
*****************************************************************************************/


TRuleCondDisjunctions::TRuleCondDisjunctions() {}


TRuleCondDisjunctions::TRuleCondDisjunctions(PDomain domain, istream &istr)
{
  readSets(domain, istr);
  readConjunctions(domain, istr);
}


void TRuleCondDisjunctions::readSets(PDomain domain, istream &istr) {
  while (!istr.eof()) {
    vector<string> atoms;
    if (!readSetAtoms(istr, atoms)) break;

    vector<string>::iterator ai(atoms.begin());
    string name=*ai;

    { ITERATE(TSets, si, sets)
        if ((*si).first==name) raiseError("RuleCondDisjunction: set '%s' already exists", name.c_str());
      if (domain->getVarNum(name, false)>=0)
        raiseError("TRuleCondDisjunction: attribute '%s' already exists", name.c_str());
    }

    vector<int> elements;
    while(++ai!=atoms.end()) {
      int varIndex=domain->getVarNum(*ai, false);
      if (varIndex>=0)
        elements.push_back(varIndex);
      else {
        TSets::iterator si(sets.begin());
        for( ; (si!=sets.end()) && ((*si).first!=*ai); si++);
        if (si!=sets.end())
          ITERATE(vector<int>, v2i, (*si).second) elements.push_back(*v2i);
        else raiseError("TRuleCondDisjunction: attribute or set '%s' not found", (*ai).c_str());
      }
    }

    sets.push_back(make_pair(name, elements));
  }
}


#define MAX_LINE_LENGTH 10240
bool TRuleCondDisjunctions::readSetAtoms(istream &str, vector<string> &atoms)
{
  char line[MAX_LINE_LENGTH], *curr=line;
  str.getline(line, MAX_LINE_LENGTH);
  if (str.gcount()==MAX_LINE_LENGTH-1) raiseError("line too long while reading conditions file");

  atoms = vector<string>();

  for(;*curr && (*curr<=' ');curr++); // skip whitespace

  string atom;
  while (*curr && (curr-line)<MAX_LINE_LENGTH) {
    switch(*curr) {
      case ' ' :
      case ':' : atoms.push_back(atom);
                 atom=string();
                 while( *++curr && (*curr<=' '));
                 break;
      case 13:
      case 10:   if (atom.length()) atoms.push_back(atom);
                 atom=string();
                 return atoms.size()>0;
      default:   if (*curr>' ') atom+=*curr;
                 curr++;
    };
  }
  if (atom.length()) atoms.push_back(atom);

  return atoms.size()>0;
}
#undef MAX_LINE_LENGTH


void TRuleCondDisjunctions::readConjunctions(PDomain domain, istream &istr)
{ while(!istr.eof()) {
    TRuleCondConjunctions *newConj=mlnew TRuleCondConjunctions(domain, istr, sets);
    if (newConj->size()) push_back(newConj);
    //else break;
  }
}


TRuleCondConjunctions *conditionForClassifier(const int &attributes)
{ TRuleCondCounted *classRight=mlnew TRuleCondCounted('=', 1, 'r');
  classRight->atomConditions.push_back(mlnew TRuleCondAtom(attributes, -1));

  TRuleCondCounted *nonoclassRight=mlnew TRuleCondCounted('=', 0, 'r');
  vector<int> noclass;
  for(int i=0; i<attributes; i++) noclass.push_back(i);
  nonoclassRight->atomConditions.push_back(mlnew TRuleCondOneOf(noclass));

  TRuleCondConjunctions *conjs=mlnew TRuleCondConjunctions;
  conjs->push_back(classRight);
  conjs->push_back(nonoclassRight);

  return conjs;
}

