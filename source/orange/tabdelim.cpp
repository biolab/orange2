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
#include <vector>
#include <set>

#include <math.h>
#include "stladdon.hpp"
#include "errors.hpp"

#include "values.hpp"
#include "vars.hpp"
#include "stringvars.hpp"
#include "domain.hpp"
#include "examples.hpp"

#include "tabdelim.ppp"


bool readTabAtom(TFileExampleIteratorData &fei, TIdList &atoms);

// Constructor; sets the name of file and the domain.
TTabDelimExampleGenerator::TTabDelimExampleGenerator(const string &afname, PDomain dom)
: TFileExampleGenerator(afname, dom)
{ 
  TTabDelimDomain *mydomain;
  domain.dynamic_cast_to(mydomain);

  if (!mydomain)
    raiseError("'domain' should be derived from TabDelimDomain");

  startDataPos = mydomain->startDataPos;
  startDataLine = mydomain->startDataLine;
}


bool emptyAtoms(TIdList &atoms)
{ for(TIdList::iterator ii(atoms.begin()); (ii!=atoms.end()); ii++)
    if ((*ii).length())
      return false;
  atoms.erase(atoms.begin(), atoms.end());
  return true;
}


// Reads an example from the file
bool TTabDelimExampleGenerator::readExample(TFileExampleIteratorData &fei, TExample &exam)
{
  TTabDelimDomain *mydomain;
  domain.dynamic_cast_to(mydomain);

  if (!mydomain)
    raiseError("'domain' should be derived from TabDelimDomain");

  TIdList atoms;
  while(!feof(fei.file) && (!readTabAtom(fei, atoms) || emptyAtoms(atoms)));
  if (!atoms.size())
    return false;

  mydomain->atomList2Example(atoms, exam, fei);
  return true;
}


TTabDelimDomain::TTabDelimDomain(const TTabDelimDomain &old)
: TDomain(old),
  kind(mlnew TIntList(old.kind.getReference())),
  DCs(old.DCs),
  classPos(old.classPos)
{}


TTabDelimDomain::TTabDelimDomain(const string &stem, PVarList knownVars, bool autoDetect)
{ TFileExampleIteratorData fei(stem);
  if (autoDetect)
    detectTypes(fei, knownVars);
  else
    readHeader(fei, knownVars);
}


void TTabDelimDomain::readHeader(TFileExampleIteratorData &fei, PVarList knownVars)
{
  TIdList varNames;
  while(!feof(fei.file) && !readTabAtom(fei, varNames));
  if (!varNames.size())
    raiseError("empty file");

  TIdList varTypes;
  while(!feof(fei.file) && !readTabAtom(fei, varTypes));
  if (!varTypes.size())
    raiseError("cannot read types of attributes");

  TIdList varFlags;
  while(!feof(fei.file) && !readTabAtom(fei, varFlags));
  if (!varFlags.size())
    raiseError("cannot read flags for attributes");

  startDataPos = ftell(fei.file);
  startDataLine = fei.line;

  constructDomain(varNames, varTypes, varFlags, knownVars);
}


/* These are the rules for determining the attribute types.

   There are three ways to determine a type.

   1. By header prefixes to attribute names.
      The prefix is formed by [cmi][DCS]#
      c, m and i mean class attribute, meta attribute and ignore,
      respectively.
      D, C and S mean discrete, continuous and string attributes.

   2. By knownVars.
      If the type is not determined from header row (either because
      there was no prefix or it only contained c, m or i)
      knownVars is checked for the attribute with the same name.
      If found, the attribute from knownVars will be used.

   3. From the data.
      These attributes can be either continuous or discrete.
      The file is parsed and values for each attribute are checked.
      Values denoting undefined values ('?', '.', '~', '*', 'NA' and
      empty strings) are ignored.
      If all values can be parsed as numbers, the attribute is continuous.
      An exception to this rule are attributes with values 0, 1, 2, ..., 9.
      These are treated as discrete (the assumption is that those number
      are just codes for otherwise discrete values).
*/

void TTabDelimDomain::detectTypes(TFileExampleIteratorData &fei, PVarList knownVars)
{ vector<string> varNames;
  vector<string> varTypes;
  vector<string> varFlags;
  vector<char> couldBeNumber; // 0 cannot be, 1 can be, 2 can even be coded discrete
  int nCouldBeNumber = 0;


  /**** Parsing the header row */

  while(!feof(fei.file) && !readTabAtom(fei, varNames));
  if (!varNames.size())
    raiseError("unexpected end of file '%s'", fei.filename.c_str());

  startDataPos = ftell(fei.file);
  startDataLine = fei.line;

  bool hasClass = false;

  {
    ITERATE(vector<string>, ni, varNames) {
      varTypes.push_back(string());
      varFlags.push_back(string());
      couldBeNumber.push_back(0);

      const char *cptr = (*ni).c_str();
      if (*cptr && (cptr[1]=='#')) {
        if (*cptr == 'm')
          varFlags.back() = "meta";
        else if (*cptr == 'i')
          varFlags.back() = "i";
        else if (*cptr == 'c') {
          varFlags.back() = "class";
          hasClass = true;
        }
        else if (*cptr == 'D')
          varTypes.back() = "d";
        else if (*cptr == 'C')
          varTypes.back() = "c";
        else if (*cptr == 'S')
          varTypes.back() = "string";
        else
          raiseWarning("unrecognized flags in attribute name '%s'", cptr);

        *ni = string(cptr+2);
      }

      else if (*cptr && cptr[1] && (cptr[2]=='#')) {
        bool beenWarned = false;
        if (*cptr == 'm')
          varFlags.back() = "meta";
        else if (*cptr == 'i')
          varFlags.back() = "i";
        else if (*cptr == 'c') {
          varFlags.back() = "class";
          hasClass = true;
        }
        else {
          raiseWarning("unrecognized flags in attribute name '%s'", cptr);
          beenWarned = true;
        }

        cptr++;
        if (*cptr == 'D')
          varTypes.back() = "d";
        else if (*cptr == 'C')
          varTypes.back() = "c";
        else if (*cptr == 'S')
          varTypes.back() = "string";
        else
          if (!beenWarned)
            raiseWarning("unrecognized flags in attribute name '%s'", cptr);

        *ni = string(cptr+2); // we have already increased cptr once
      }

      // If the type has not been determined, we look at knownVars
      if (!varTypes.back().length() && knownVars)
        PITERATE(TVarList, kni, knownVars)
          if ((*kni)->name == *ni) {
            varTypes.back() = "*";
            break;
          }
          
      /* If we still don't have the type, we request the check by setting
         encountered and couldBeNumber */
      if (!varTypes.back().length()) {
        couldBeNumber.back() = 2;
        nCouldBeNumber++;
      }
    }
  }



  if (nCouldBeNumber) {

    /**** Parsing the data (if needed) */

    vector<string> atoms;
    char numTest[64];
    while (!feof(fei.file) && nCouldBeNumber) {
      if (!readTabAtom(fei, atoms))
        continue;
    
      vector<char>::iterator cni(couldBeNumber.begin());
      for(vector<string>::const_iterator ai(atoms.begin()), ae(atoms.end()); ai!=ae; ai++, cni++) {
        if (*cni) {
          // If it represents a special value, we skip it
          const char *ceni = (*ai).c_str();
          if (   !*ceni
              || !ceni[1] && ((*ceni=='?') || (*ceni=='.') || (*ceni=='~') || (*ceni=='*'))
              || (*ai == "NA"))
            continue;

          // If the attribute can be a number, we check it as a number
          if ((*ai).length()>63) {
            *cni = 0;
            nCouldBeNumber--;
            continue;
          }

          if ((*ai).length()==1) {
            if (((*ai)[0]<'0') || ((*ai)[0]>'9')) {
              *cni = 0;
              nCouldBeNumber--;
            }
            continue;
          }

          *cni = 1; // longer than 1 character

          strcpy(numTest, ceni);
          for(char *sc = numTest; *sc; sc++)
            if (*sc == ',')
              *sc = '.';

          char *eptr;
          strtod(numTest, &eptr);
          if (*eptr) {
            *cni = 0;
            nCouldBeNumber--;
          }
        }
      }
    }


    /**** Setting the missing types */

    vector<string>::iterator ti(varTypes.begin());
    vector<char>::const_iterator cni(couldBeNumber.begin()), cne(couldBeNumber.end());
    for(; cni!=cne; cni++, ti++)
      if (!(*ti).length())
        (*ti) = (*cni==1) ? "c" : "d";
  }
  
  if (!hasClass)
    for(vector<string>::reverse_iterator fri(varFlags.rbegin()), fre(varFlags.rend()); fri!=fre; fri++)
      /* We don't want ignored or meta-attributes as classes; we'll thus find
         the last that is not such */
      if (!(*fri).length()) {
        *fri = "class";
        break;
      }

  constructDomain(varNames, varTypes, varFlags, knownVars);
}


TTabDelimDomain::TTabDelimDomain(TIdList &varNames, TIdList &varTypes, TIdList &varFlags, PVarList knownVars)
  : TDomain()
{ constructDomain(varNames, varTypes, varFlags, knownVars); }



void TTabDelimDomain::constructDomain(TIdList &varNames, TIdList &varTypes, TIdList &varFlags, PVarList knownVars)
{
  PVariable newVar;

  if (varNames.size() != varTypes.size())
    raiseError("mismatching number of attributes and their types.");
  if (varNames.size() < varFlags.size())
    raiseError("too many flags (third line too long)");

  kind=mlnew TIntList(varNames.size(), -1);
  DCs=mlnew TStringList(varNames.size(), "");

  int pos=0;
  classPos=-1;

  // parses the 3rd line; for each attribute, it checks whether the flags are correct,
  // it sets the classPos (position of the class attribute), kind[i] becomes 0 for attribute i
  // which is to be skipped, and id (getMetaID) for meta attributes. It sets DCs[i] for attributes
  // with different DC character.
  TIdList::iterator vni = varNames.begin();
  vector<TProgArguments> arguments;
  ITERATE(TIdList, fi, varFlags) {
    arguments.push_back(TProgArguments("dc: ordered", *fi, false));
    TProgArguments &args = arguments.back();
    if (args.direct.size()) {
      if (args.direct.size()>1)
        raiseError("invalid flags for attribute '%s'", (*vni).c_str());
      string direct = args.direct.front();
      if ((direct=="s") || (direct=="skip") || (direct=="i") || (direct=="ignore"))
        kind->at(pos) = 0;
      else if ((direct=="c") || (direct=="class"))
        if (classPos==-1)
          classPos = pos;
        else 
          raiseError("multiple attributes are specified as class attribute ('%s' and '%s')", varNames[pos].c_str(), (*vni).c_str());
      else if (direct=="meta") {
        long id=getMetaID();
        metas.push_back(TMetaDescriptor(id, PVariable()));
        kind->at(pos)=id;
      }
    }

    if (args.exists("dc"))
      DCs->at(pos) = args["dc"];
    pos++; vni++;
  }
  while (arguments.size()<varNames.size())
    arguments.push_back(TProgArguments());

  // Constructs variables
  vector<int>::iterator si=kind->begin();
  vector<TProgArguments>::const_iterator argi(arguments.begin());
  pos=0;
  for(TIdList::iterator ni=varNames.begin(), ti=varTypes.begin(); ni!=varNames.end(); ni++, ti++, pos++, argi++, si++) {
    if (*si) {
      if (!(*ti).length())
        raiseError("type for attribute '%s' is missing", (*ni).c_str());

      if (*ti=="*")
        newVar = makeVariable(*ni, knownVars, -1);
      else if ((*ti=="c") || (*ti=="continuous") || (*ti=="float") || (*ti=="f"))
        newVar = makeVariable(*ni, knownVars, TValue::FLOATVAR);
        //newVar=mlnew TFloatVariable(*ni);
      else if ((*ti=="d") || (*ti=="discrete") || (*ti=="e") || (*ti=="enum")) {
        newVar = makeVariable(*ni, knownVars, TValue::INTVAR);
        newVar->ordered = (*argi).exists("ordered");
      }
      else if (*ti=="string")
        newVar = makeVariable(*ni, knownVars, stringVarType);
      else {
        string vals;
        newVar = makeVariable(*ni, knownVars, TValue::INTVAR);
        TEnumVariable *evar = newVar.AS(TEnumVariable);
        newVar->ordered = (*argi).exists("ordered");
        ITERATE(string, ci, *ti)
          if (*ci==' ') {
            if (vals.length())
              evar->addValue(vals);
            vals="";
          } 
          else
            vals+=*ci;

        if (vals.length())
          evar->addValue(vals);

        newVar->ordered = (*argi).exists("ordered");
      }

      if (*si==-1) {
        if (pos==classPos)
          classVar=newVar;
        else
          attributes->push_back(newVar);
      }
      else {
        TMetaDescriptor *meta = metas[*si];
        if (!meta) 
          raiseError("error in domain for tab-delimited file (meta descriptor not found)");
        meta->variable = newVar;
      }
    }
  }

  variables = mlnew TVarList(attributes.getReference());
  if (classVar)
    variables->push_back(classVar);
}


void TTabDelimDomain::atomList2Example(TIdList &atoms, TExample &exam, const TFileExampleIteratorData &fei)
{
  // Add an appropriate number of empty atoms, if needed
  while (atoms.size()<kind->size())
    atoms.push_back(string(""));
  _ASSERT(exam.domain==this);

  TExample::iterator ei(exam.begin());
  TVarList::iterator vi(attributes->begin());
  TIdList ::iterator ai(atoms.begin());
  vector<int>::iterator si(kind->begin()), se(kind->end());
  vector<string>::iterator dci(DCs->begin());
  int pos=0;
  for (; (si!=se); pos++, si++, ai++, dci++)
    if (*si) { // if attribute is not to be skipped
      string valstr;

      // Check for don't care
      if (!(*ai).length() || (valstr=="NA"))
        valstr = "?"; // empty fields are treated as don't care
      else { // else check if one of don't care symbols
        valstr = *ai;
        if (valstr.length()==1) {
          if ((*dci).size()) {
            string::iterator dcii = (*dci).begin();
            for(; (dcii!=(*dci).end()) && (*dcii!=valstr[0]); dcii++);
            if (dcii!=(*dci).end())
              valstr[0]='?';
          }
          else
            if (valstr[0]=='.')
              valstr[0]='?';
        }
        else
          if (valstr=="*")
            valstr[0]='~';
      }

      if (*si==-1)
        if (pos==classPos) { // if this is class value
          TValue cval;
          if (classVar->varType == TValue::FLOATVAR) {
            if (!classVar->str2val_try(valstr, cval))
              raiseError("file '%s', line %i: '%s' is not a legal value for the continuous class", fei.filename.c_str(), fei.line, valstr.c_str());
          }
          else
            classVar->str2val_add(valstr, cval);

          exam.setClass(cval);
        }
        else { // if this is a normal value
          // replace the first ',' with '.'
          // (if there is more than one, it's an error anyway
          if ((*vi)->varType == TValue::FLOATVAR) {
            int cp = valstr.find(',');
            if (cp!=string::npos)
              valstr[cp] = '.';
            if (!(*vi)->str2val_try(valstr, *ei))
              raiseError("file '%s', line %i: '%s' is not a legal value for the continuous attribute '%s'", fei.filename.c_str(), fei.line, valstr.c_str(), (*vi)->name.c_str());
          }
          else
            (*vi)->str2val_add(valstr, *ei);

          vi++;
          ei++;
        }
      else { // if this is a meta value
        TMetaDescriptor *md = metas[*si];
        _ASSERT(md!=NULL);
        TValue mval;
        md->variable->str2val_add(valstr, mval);
        exam.meta.setValue(*si, mval);
      }
    }

  if (pos==classPos) // if class is the last value in the line, it is set here
    classVar->str2val_add(ai==atoms.end() ? "?" : *(ai++), exam[variables->size()-1]);

  while ((ai!=atoms.end()) && !(*ai).length()) ai++; // line must be empty from now on

  if (ai!=atoms.end()) {
	TIdList::iterator ii=atoms.begin();
	string s=*ii;
	while(++ii!=atoms.end()) s+=" "+*ii;
    raiseError("example of invalid length (%s)", s.c_str());
  }
}


/*  Reads a list of atoms from a line of tab delimited file. Atom consists of any characters
    except \n, \r and \t. Multiple spaces are replaced by a single space. Atoms are separated
    by \t. Lines end with \n or \r. Lines which begin with | are ignored. */
bool readTabAtom(TFileExampleIteratorData &fei, TIdList &atoms)
{
  atoms.clear();

  if (!fei.file)
    raiseErrorWho("TabDelimExampleGenerator", "file not opened");

  if (feof(fei.file))
    return false;

  char line[32768], *curr=line;

  fei.line++;
  if (!fgets(line, 32768, fei.file)) {
    if (feof(fei.file))
      return false;
    raiseErrorWho("TabDelimExampleGenerator", "error while reading line %i of file '%s'", fei.line, fei.filename.c_str());
  }

  if (strlen(line)>=32768-1)
    raiseErrorWho("TabDelimExampleGenerator", "line %i of file '%s' too long", fei.line, fei.filename.c_str());

  if (*curr=='|')
    return false;

  string atom;
  while (*curr)
    switch(*curr) {
      case '\r':
      case '\n':
        if (atom.length() || atoms.size())
          atoms.push_back(atom);  // end of line
        return atoms.size()>0;

      case '\t':
        atoms.push_back(atom);
        atom = string();
        break;

      case ' ':
        atom += *curr;
        break;

      default:
        if ((*curr>=' ') || (*curr<0))
          atom += *curr++;
        else
          curr++;
    };

  if (atom.length() || atoms.size())
    atoms.push_back(atom);

  return atoms.size()>0;
}




// ********* Output ********* //


void tabDelim_writeExample(FILE *file, const TExample &ex)
{ TVarList::const_iterator vi(ex.domain->variables->begin()), ve(ex.domain->variables->end());
  TExample::const_iterator ri(ex.begin());
  string st;
  (*(vi++))->val2str(*(ri++), st);
  fprintf(file, "%s", st.c_str());
  for(; vi!=ve; vi++, ri++) {
    (*vi)->val2str(*ri, st);
    fprintf(file, "\t%s", st.c_str());
  }

  const_ITERATE(TMetaVector, mi, ex.domain->metas) {
    (*mi).variable->val2str(ex.meta[(*mi).id], st);
    fprintf(file, "\t%s", st.c_str());
  }
  fprintf(file, "\n");
}


void tabDelim_writeExamples(FILE *file, PExampleGenerator rg)
{ PEITERATE(gi, rg)
    tabDelim_writeExample(file, *gi);
}


void printVarType(FILE *file, PVariable var)
{
  TEnumVariable *enumv;
  var.dynamic_cast_to(enumv);
  if (enumv) {
    TValue val;
    string sval;
    if (!enumv->firstValue(val))
      fprintf(file, "d");
    else {
      enumv->val2str(val, sval); 
      fprintf(file, sval.c_str());
      while(enumv->nextValue(val)) {
        enumv->val2str(val, sval);
        fprintf(file, " %s", sval.c_str());
      }
    }
  }
  else if (var.is_derived_from(TFloatVariable))
    fprintf(file, "continuous");
  else if (var.is_derived_from(TStringVariable))
    fprintf(file, "string");
  else
    raiseErrorWho("tabDelim_writeDomain", "tabDelim format supports only discrete, continuous and string variables");
}


void tabDelim_writeDomain(FILE *file, PDomain dom)
{ 
  { int notFirst=0;
    const_PITERATE(TVarList, vi, dom->variables) {
      if (notFirst++)
        fprintf(file, "\t%s", (*vi)->name.c_str());
      else
        fprintf(file, "%s", (*vi)->name.c_str());
    }

    const_ITERATE(TMetaVector, mi, dom->metas) {
      if (notFirst++)
        fprintf(file, "\t%s", (*mi).variable->name.c_str());
      else
        fprintf(file, "%s", (*mi).variable->name.c_str());
    }
    fprintf(file, "\n");
  }

  { int notFirst=0;
    const_PITERATE(TVarList, vi, dom->variables) {
      if (notFirst++)
        fprintf(file, "\t");
      printVarType(file, *vi);
    }
    const_ITERATE(TMetaVector, mi, dom->metas) {
      if (notFirst++)
        fprintf(file, "\t");
      printVarType(file, (*mi).variable);
    }
    fprintf(file, "\n");
  }

  { if (dom->attributes->size())
      for(int i = dom->attributes->size()-1; i--; )
        fprintf(file, "\t");

    if (dom->classVar)
      fprintf(file, "\tclass");

    int notFirst=dom->variables->size();
    
    { const_ITERATE(TMetaVector, mi, dom->metas) {
        if (notFirst++)
          fprintf(file, "\t");
        fprintf(file, "meta");
      }
    }
   fprintf(file, "\n");
  }
}

