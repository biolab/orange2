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
#include <list>

#include <math.h>
#include "stladdon.hpp"
#include "errors.hpp"

#include "values.hpp"
#include "vars.hpp"
#include "stringvars.hpp"
#include "domain.hpp"
#include "examples.hpp"

#include "tabdelim.ppp"

bool readTabAtom(TFileExampleIteratorData &fei, TIdList &atoms, bool escapeSpaces=true);

TTabDelimExampleGenerator::TTabDelimExampleGenerator(const string &afname, PDomain dom)
: TFileExampleGenerator(afname, dom)
{ 
  TTabDelimDomain *mydomain = domain.AS(TTabDelimDomain);
  if (!mydomain)
    raiseError("'domain' should be derived from TabDelimDomain");

  TFileExampleIteratorData fei(afname);
  
  TIdList atoms;
  for (int i = mydomain->headerLines; !feof(fei.file) && i--; )
    while(!feof(fei.file) && !readTabAtom(fei, atoms)) {
      TIdList::iterator ii(atoms.begin()), ie(atoms.end());
      while ((ii!=ie) && !(*ii).length())
        ii++;
      if (ii==ie)
        atoms.clear();
      else
        break;
    }

  startDataPos = ftell(fei.file);
  startDataLine = fei.line;
}


bool TTabDelimExampleGenerator::readExample(TFileExampleIteratorData &fei, TExample &exam)
{
  TTabDelimDomain *mydomain = domain.AS(TTabDelimDomain);
  if (!mydomain)
    raiseError("'domain' should be derived from TabDelimDomain");

  TIdList atoms;
  while(!feof(fei.file) && !readTabAtom(fei, atoms)) {
    TIdList::iterator ii(atoms.begin()), ie(atoms.end());
    while ((ii!=ie) && !(*ii).length())
      ii++;
    if (ii==ie)
      atoms.clear();
    else
      break;
  }
  
  if (!atoms.size())
    return false;

  mydomain->atomList2Example(atoms, exam, fei);
  return true;
}


list<TTabDelimDomain *> TTabDelimDomain::knownDomains;
TKnownVariables TTabDelimDomain::knownVariables;


void TTabDelimDomain::removeKnownVariable(TVariable *var)
{ knownVariables.remove(var);
  var->destroyNotifier = NULL;
}



PVariable TTabDelimDomain::createVariable(const string &name, const int &varType, bool dontStore)
{ PVariable var = ::createVariable(name, varType);
  if (!dontStore)
    knownVariables.add(var.AS(TVariable), removeKnownVariable);

  return var;
}


TTabDelimDomain::TTabDelimDomain(const TTabDelimDomain &old)
: TDomain(old),
  attributeTypes(mlnew TIntList(old.attributeTypes.getReference())),
  DCs(old.DCs),
  classPos(old.classPos)
{}


TTabDelimDomain::TTabDelimDomain()
: TDomain(),
  attributeTypes(mlnew TIntList()),
  DCs(mlnew TStringList()),
  classPos(-1),
  headerLines(0)
{}


TTabDelimDomain::~TTabDelimDomain()
{ knownDomains.remove(this);
}


bool TTabDelimDomain::isSameDomain(TTabDelimDomain const *orig) const
{ if (   !orig
      || (classPos != orig->classPos)
      || !sameDomains(this, orig))
    return false;

  for (TIntList::const_iterator ki1(attributeTypes->begin()), ke1(attributeTypes->end()), ki2(orig->attributeTypes->begin()); ki1!=ke1; ki1++, ki2++)
    if (*ki1 != *ki2)
      return false;

  for (TStringList::const_iterator si1(DCs->begin()), se1(DCs->end()), si2(orig->DCs->begin()); si1!=se1; si1++, si2++)
    if (*si1 != *si2)
      return false;

  return true;
}


PDomain TTabDelimDomain::readDomain(const bool autoDetect, const string &stem, PVarList sourceVars, PDomain sourceDomain, bool dontCheckStored, bool dontStore)
{ PDomain newDomain = (autoDetect ? domainWithDetection : domainWithoutDetection) (stem, sourceVars, sourceDomain, dontCheckStored, dontStore);

  TTabDelimDomain *unewDomain = newDomain.AS(TTabDelimDomain);
  if (!unewDomain)
    return newDomain;
                                 
  if (sourceDomain) {
    TTabDelimDomain *usourceDomain = sourceDomain.AS(TTabDelimDomain);
    if (unewDomain->isSameDomain(usourceDomain))
      return sourceDomain;
  }

  if (dontCheckStored)
    return newDomain;

  ITERATE(list<TTabDelimDomain *>, sdi, knownDomains)
    if (unewDomain->isSameDomain(*sdi))
      // The domain is rewrapped here (we have a pure pointer, but it has already been wrapped)
      return *sdi;

  if (!dontStore && !exists(knownDomains, unewDomain))
    knownDomains.push_back(unewDomain);

  return newDomain;
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

class TSearchWarranty 
{ public:
  int posInFile, posInDomain, suspectedType;
  TSearchWarranty(const int &pif, const int &pid)
  : posInFile(pif), posInDomain(pid), suspectedType(3)
  {}
};

PDomain TTabDelimDomain::domainWithDetection(const string &stem, PVarList sourceVars, PDomain sourceDomain, bool dontCheckStored, bool dontStore)
{ 
  TFileExampleIteratorData fei(stem);
  
  TIdList varNames;
  while(!feof(fei.file) && !readTabAtom(fei, varNames));
  if (varNames.empty())
    ::raiseError("unexpected end of file '%s'", fei.filename.c_str());

  TTabDelimDomain *domain = mlnew TTabDelimDomain();
  PDomain wdomain = domain;
  int &classPos = domain->classPos;
  TVarList &variables = const_cast<TVarList &>(domain->variables.getReference());
  TIntList &attributeTypes = const_cast<TIntList &>(domain->attributeTypes.getReference());
  TMetaVector &metas = domain->metas;

  domain->headerLines = 1;

  list<TSearchWarranty> searchWarranties;

  /**** Parsing the header row */
  
  ITERATE(vector<string>, ni, varNames) {
    /* Parses the header line
       - sets *ni to a real name (without prefix)
       - sets varType to TValue::varType or -1 if the type is not specified
       - sets classPos to the current position, if the attribute is class attribute
         (and reports an error if there is more than one such attribute)
       - to attributeTypes, appends -1 for ordinary atributes, -2 for metas and 0 for ignored */
    int varType = -1; // varType, or -1 for unnown
    attributeTypes.push_back(-1);

    const char *cptr = (*ni).c_str();
    if (*cptr && (cptr[1]=='#')) {
      if (*cptr == 'm')
        attributeTypes.back() = -2;
      else if (*cptr == 'i')
        attributeTypes.back() = 0;
      else if (*cptr == 'c') {
        if (classPos>-1)
          ::raiseError("more than one attribute marked as class");
        else
          classPos = ni-varNames.begin();
      }

      else if (*cptr == 'D')
        varType = TValue::INTVAR;
      else if (*cptr == 'C')
        varType = TValue::FLOATVAR;
      else if (*cptr == 'S')
        varType = stringVarType;

      else
        ::raiseError("unrecognized flags in attribute name '%s'", cptr);

      *ni = string(cptr+2);
    }

    else if (*cptr && cptr[1] && (cptr[2]=='#')) {
      bool beenWarned = false;
      if (*cptr == 'm')
        attributeTypes.back() = -2;
      else if (*cptr == 'i')
        attributeTypes.back() = 0;
      else if (*cptr == 'c') {
        if (classPos>-1)
          ::raiseError("more than one attribute marked as class");
        else
          classPos = ni-varNames.begin();
      }
      else
        ::raiseError("unrecognized flags in attribute name '%s'", cptr);

      cptr++;
      if (*cptr == 'D')
        varType = TValue::INTVAR;
      else if (*cptr == 'C')
        varType = TValue::FLOATVAR;
      else if (*cptr == 'S')
        varType = stringVarType;
      else
        ::raiseError("unrecognized flags in attribute name '%s'", cptr);

      // remove the prefix (we have already increased cptr once)
      *ni = string(cptr+2);
    }

    /* If the attribute is not to be ignored, we attempt to either find its descriptor
       among the known attributes or create a new attribute if the type is given.
       For ordinary attributes, the descriptor (or PVariable()) is pushed to the list of 'variables'.
       For meta attributes, a meta descriptor is pushed to 'metas'. If the attribute was used as
       meta-attribute in some of known domains, the id is reused; otherwise a new id is created.
       If the descriptor was nor found nor created, a warranty is issued.
    */
      
    if (attributeTypes.back()) {
      int id;
      PVariable var = makeVariable<TTabDelimDomain>(*ni, varType, sourceVars, sourceDomain, dontCheckStored ? NULL : &knownVariables, dontCheckStored ? NULL : &knownDomains, id, varType>=0, NULL);
      if (attributeTypes.back() == -2) {
        if (!id)
          id = getMetaID();
        metas.push_back(TMetaDescriptor(id, var));
        attributeTypes.back() = id;

        if (!var)
          searchWarranties.push_back(TSearchWarranty(ni-varNames.begin(), -id));
      }
      else {
        variables.push_back(var);
        if (!var)
          searchWarranties.push_back(TSearchWarranty(ni-varNames.begin(), variables.size()-1));
      }
    }
  }

  if (!searchWarranties.empty()) {
    vector<string> atoms;
    char numTest[64];
    while (!feof(fei.file) && !searchWarranties.empty()) {
      if (!readTabAtom(fei, atoms))
        continue;
    
      for(list<TSearchWarranty>::iterator wi(searchWarranties.begin()), we(searchWarranties.end()); wi!=we; wi++) {
        const string &atom = atoms[(*wi).posInFile];

        // only discrete attributes can have values longer than 63 characters
        if (atom.length()>63) {
          PVariable newVar = createVariable(varNames[(*wi).posInFile], TValue::INTVAR, dontStore);
          if ((*wi).posInDomain >= 0)
            variables[(*wi).posInDomain] = newVar;
          else
            metas[-(*wi).posInDomain]->variable = newVar;
          wi = searchWarranties.erase(wi);
          wi--;
          continue;
        }

        const char *ceni = atom.c_str();
        if (   !*ceni
            || !ceni[1] && ((*ceni=='?') || (*ceni=='.') || (*ceni=='~') || (*ceni=='*'))
            || (atom == "NA"))
          continue;

        // we have encountered some value
        if ((*wi).suspectedType == 3) 
          (*wi).suspectedType = 2;

        // If the attribute is a digit, it can be anything
        if ((!*ceni) && (*ceni>='0') && (*ceni<='9'))
          continue;

        // If it is longer than one character, it cannot be a coded discrete
        if (ceni[1])
          (*wi).suspectedType = 1;

        // Convert commas into dots
        strcpy(numTest, ceni);
        for(char *sc = numTest; *sc; sc++)
          if (*sc == ',')
            *sc = '.';

        // If the attribute cannot be converted into a number, it is enum
        char *eptr;
        strtod(numTest, &eptr);
        if (*eptr) {
          PVariable newVar = createVariable(varNames[(*wi).posInFile], TValue::INTVAR, dontStore);
          if ((*wi).posInDomain >= 0)
            variables[(*wi).posInDomain] = newVar;
          else
            metas[-(*wi).posInDomain]->variable = newVar;
          wi = searchWarranties.erase(wi);
          continue;
        }
      }
    }


    ITERATE(list<TSearchWarranty>, wi, searchWarranties) {
      const string &name = varNames[(*wi).posInFile];
      if ((*wi).suspectedType == 3)
        ::raiseError("cannot determine type for attribute '%s'", name.c_str());

      PVariable var = createVariable(name, (*wi).suspectedType == 2 ? TValue::INTVAR : TValue::FLOATVAR, dontStore);
      if ((*wi).posInDomain >= 0)
        variables[(*wi).posInDomain] = var;
      else
        metas[-(*wi).posInDomain]->variable = var;
    }
  }

  if (classPos>=0) {
    TVarList::iterator ci = variables.begin()+classPos;
    domain->classVar = *ci;
    variables.erase(ci);
    domain->attributes = mlnew TVarList(variables);
    variables.push_back(domain->classVar);
  }
  else {
    classPos = varNames.size()-1;
    vector<int>::reverse_iterator fri(attributeTypes.rbegin()), fre(attributeTypes.rend());
    for(; (fri!=fre) && (*fri != -1); fri++, classPos--);
    if (fri==fre)
      classPos = -1;
    else {
      domain->attributes = mlnew TVarList(variables);
      domain->classVar = variables.back();
      domain->attributes->erase(domain->attributes->end()-1);
    }
  }

  domain->DCs = mlnew TStringList(domain->variables->size(), "");

  return wdomain;
}


PDomain TTabDelimDomain::domainWithoutDetection(const string &stem, PVarList sourceVars, PDomain sourceDomain, bool dontCheckStored, bool dontStore)
{
  TFileExampleIteratorData fei(stem);
  
  TIdList varNames, varTypes, varFlags;
  
  while(!feof(fei.file) && !readTabAtom(fei, varNames));
  if (varNames.empty())
    ::raiseError("empty file");

  while(!feof(fei.file) && !readTabAtom(fei, varTypes, false));
  if (varTypes.empty())
    ::raiseError("cannot read types of attributes");

  while(!feof(fei.file) && !readTabAtom(fei, varFlags));

  if (varNames.size() != varTypes.size())
    ::raiseError("mismatching number of attributes and their types.");
  if (varNames.size() < varFlags.size())
    ::raiseError("too many flags (third line too long)");

  TTabDelimDomain *domain = mlnew TTabDelimDomain();
  domain->attributeTypes = mlnew TIntList(varNames.size(), -1);
  domain->DCs = mlnew TStringList(varNames.size(), "");

  PDomain wdomain = domain;
  int &classPos = domain->classPos;
  TVarList &variables = const_cast<TVarList &>(domain->variables.getReference());
  TIntList &attributeTypes = const_cast<TIntList &>(domain->attributeTypes.getReference());
  TMetaVector &metas = domain->metas;

  domain->headerLines = 3;


  int pos=0;

  // parses the 3rd line; for each attribute, it checks whether the flags are correct,
  // it sets the classPos (position of the class attribute), attributeTypes[i] becomes 0 for attribute i
  // which is to be skipped, and id (getMetaID) for meta attributes. It sets DCs[i] for attributes
  // with different DC character.
  TIdList::iterator vni = varNames.begin();
  vector<TProgArguments> arguments;
  ITERATE(TIdList, fi, varFlags) {
    arguments.push_back(TProgArguments("dc: ordered", *fi, false));
    TProgArguments &args = arguments.back();
    if (args.direct.size()) {
      if (args.direct.size()>1)
        ::raiseError("invalid flags for attribute '%s'", (*vni).c_str());
      string direct = args.direct.front();
      if ((direct=="s") || (direct=="skip") || (direct=="i") || (direct=="ignore"))
        attributeTypes[pos] = 0;
      else if ((direct=="c") || (direct=="class"))
        if (classPos==-1)
          classPos = pos;
        else 
          ::raiseError("multiple attributes are specified as class attribute ('%s' and '%s')", varNames[pos].c_str(), (*vni).c_str());
      else if (direct=="meta")
        attributeTypes[pos]=-2;
    }

    if (args.exists("dc"))
      domain->DCs->at(pos) = args["dc"];

    pos++;
    vni++;
  }
  while (arguments.size()<varNames.size())
    arguments.push_back(TProgArguments());

  TKnownVariables *sknownv = dontCheckStored ? NULL : &knownVariables;
  list<TTabDelimDomain *> *sknownd = dontCheckStored ? NULL : &knownDomains;
  TVariable::TDestroyNotifier *notifier = dontStore ? NULL : removeKnownVariable;

  // Constructs variables
  vector<int>::iterator si = attributeTypes.begin();
  vector<TProgArguments>::const_iterator argi(arguments.begin());
  pos=0;
  for(TIdList::iterator ni=varNames.begin(), ti=varTypes.begin(); ni!=varNames.end(); ni++, ti++, pos++, argi++, si++) {
    if (*si) {
      PVariable newVar;
      int id;

      if (!(*ti).length())
        ::raiseError("type for attribute '%s' is missing", (*ni).c_str());

      if ((*ti=="c") || (*ti=="continuous") || (*ti=="float") || (*ti=="f"))
        newVar = makeVariable<TTabDelimDomain>(*ni, TValue::FLOATVAR, sourceVars, sourceDomain, sknownv, sknownd, id, false, notifier);

      else if ((*ti=="d") || (*ti=="discrete") || (*ti=="e") || (*ti=="enum")) {
        newVar = makeVariable<TTabDelimDomain>(*ni, TValue::INTVAR, sourceVars, sourceDomain, sknownv, sknownd, id, false, notifier);
        newVar->ordered = (*argi).exists("ordered");
      }

      else if (*ti=="string")
        newVar = makeVariable<TTabDelimDomain>(*ni, stringVarType, sourceVars, sourceDomain, sknownv, sknownd, id, false, notifier);

      else {
        string vals;
        newVar = makeVariable<TTabDelimDomain>(*ni, TValue::INTVAR, sourceVars, sourceDomain, sknownv, sknownd, id, false, notifier);
        TEnumVariable *evar = newVar.AS(TEnumVariable);
        newVar->ordered = (*argi).exists("ordered");
        ITERATE(string, ci, *ti)
          if (*ci==' ') {
            if (vals.length())
              evar->addValue(vals);
            vals="";
          } 
          else {
            if ((*ci=='\\') && (ci[1]==' ')) {
              vals += ' ';
              ci++;
            }
            else
              vals += *ci;
          }

        if (vals.length())
          evar->addValue(vals);

        newVar->ordered = (*argi).exists("ordered");
      }

      if (*si==-1) {
        if (pos==classPos)
          domain->classVar = newVar;
        else
          domain->attributes->push_back(newVar);
      }
      else { // *si==-2
        if (!id)
          id = getMetaID();
        *si = id;
        metas.push_back(TMetaDescriptor(id, newVar));
      }
    }
  }

  domain->variables = mlnew TVarList(domain->attributes.getReference());
  if (domain->classVar)
    domain->variables->push_back(domain->classVar);

  return wdomain;
}


void TTabDelimDomain::atomList2Example(TIdList &atoms, TExample &exam, const TFileExampleIteratorData &fei)
{
  // Add an appropriate number of empty atoms, if needed
  while (atoms.size()<attributeTypes->size())
    atoms.push_back(string(""));
  _ASSERT(exam.domain==this);

  TExample::iterator ei(exam.begin());
  TVarList::iterator vi(attributes->begin());
  TIdList ::iterator ai(atoms.begin());
  vector<int>::iterator si(attributeTypes->begin()), se(attributeTypes->end());
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
bool readTabAtom(TFileExampleIteratorData &fei, TIdList &atoms, bool escapeSpaces)
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
  while (*curr) {
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

      case '\\':
        if (escapeSpaces && curr[1]==' ') {
          atom += ' ';
          curr++;
          break;
        }

      default:
        if ((*curr>=' ') || (*curr<0))
          atom += *curr;
    };
    curr++;
  }

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

string escSpaces(const string &s)
{ string res;
  const_ITERATE(string, si, s)
    if (*si==' ')
      res += "\\ ";
    else
      res += *si;
  return res;
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
      fprintf(file, escSpaces(sval).c_str());
      while(enumv->nextValue(val)) {
        enumv->val2str(val, sval);
        fprintf(file, " %s", escSpaces(sval).c_str());
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

