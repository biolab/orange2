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
#include "strings.hpp"

#include "values.hpp"
#include "vars.hpp"
#include "stringvars.hpp"
#include "domain.hpp"
#include "examples.hpp"

#include "tabdelim.ppp"

bool readTabAtom(TFileExampleIteratorData &fei, TIdList &atoms, bool escapeSpaces=true, bool csv = false);
bool atomsEmpty(const TIdList &atoms);

list<TDomain *> TTabDelimExampleGenerator::knownDomains;


TTabDelimExampleGenerator::TTabDelimExampleGenerator(const TTabDelimExampleGenerator &old)
: TFileExampleGenerator(old),
  attributeTypes(mlnew TIntList(old.attributeTypes.getReference())),
  DCs(CLONE(TStringList, old.DCs)),
  classPos(old.classPos),
  headerLines(old.headerLines),
  csv(old.csv)
{}


TTabDelimExampleGenerator::TTabDelimExampleGenerator(const string &afname, bool autoDetect, bool acsv, PVarList sourceVars, TMetaVector *sourceMetas, PDomain sourceDomain, bool dontCheckStored, bool dontStore)
: TFileExampleGenerator(afname, PDomain()),
  attributeTypes(mlnew TIntList()),
  DCs(mlnew TStringList()),
  classPos(-1),
  headerLines(0),
  csv(acsv)
{ 
  // domain needs to be initialized after attributeTypes, DCs, classPos, headerLines
  domain = readDomain(afname, autoDetect, sourceVars, sourceMetas, sourceDomain, dontCheckStored, dontStore);

  TFileExampleIteratorData fei(afname);
  
  TIdList atoms;
  for (int i = headerLines; !feof(fei.file) && i--; )
    while(!feof(fei.file) && !readTabAtom(fei, atoms, true, csv)) {
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


void TTabDelimExampleGenerator::destroyNotifier(TDomain *domain)
{ knownDomains.remove(domain); }


bool TTabDelimExampleGenerator::readExample(TFileExampleIteratorData &fei, TExample &exam)
{
  TIdList atoms;
  while(!feof(fei.file) && (!readTabAtom(fei, atoms, true, csv) || atomsEmpty(atoms))) {
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

  // Add an appropriate number of empty atoms, if needed
  while (atoms.size()<attributeTypes->size())
    atoms.push_back(string(""));
  _ASSERT(exam.domain==domain);

  TExample::iterator ei(exam.begin());
  TVarList::iterator vi(domain->attributes->begin());
  TIdList ::iterator ai(atoms.begin());
  vector<int>::iterator si(attributeTypes->begin()), se(attributeTypes->end());
  bool dcs = DCs && DCs->size();
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
          if (dcs && (*dci).size()) {
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
          if (domain->classVar->varType == TValue::FLOATVAR) {
            if (!domain->classVar->str2val_try(valstr, cval))
              raiseError("file '%s', line %i: '%s' is not a legal value for the continuous class", fei.filename.c_str(), fei.line, valstr.c_str());
          }
          else
            domain->classVar->str2val_add(valstr, cval);

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
        TMetaDescriptor *md = domain->metas[*si];
        _ASSERT(md!=NULL);
        TValue mval;
        md->variable->str2val_add(valstr, mval);
        exam.setMeta(*si, mval);
      }
    }

  if (pos==classPos) // if class is the last value in the line, it is set here
    domain->classVar->str2val_add(ai==atoms.end() ? "?" : *(ai++), exam[domain->variables->size()-1]);

  while ((ai!=atoms.end()) && !(*ai).length()) ai++; // line must be empty from now on

  if (ai!=atoms.end()) {
	TIdList::iterator ii=atoms.begin();
	string s=*ii;
	while(++ii!=atoms.end()) s+=" "+*ii;
    raiseError("example of invalid length (%s)", s.c_str());
  }

  return true;
}



PDomain TTabDelimExampleGenerator::readDomain(const string &stem, const bool autoDetect, PVarList sourceVars, TMetaVector *sourceMetas, PDomain sourceDomain, bool dontCheckStored, bool dontStore)
{ bool domainIsNew;
  PDomain newDomain;
  
  if (autoDetect)
    newDomain = domainWithDetection(stem, domainIsNew, sourceVars, sourceMetas, sourceDomain, dontCheckStored);
  else
    newDomain = domainWithoutDetection(stem, domainIsNew, sourceVars, sourceMetas, sourceDomain, dontCheckStored);

  if (domainIsNew && !dontStore) {
    newDomain->destroyNotifier = destroyNotifier;
    knownDomains.push_back(newDomain.AS(TDomain));
  }

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
  // suspectedType can be 3 (never seen it yet), 2 (can even be coded discrete), 1 (can be float);
  //   if it's found that it cannot be float, it can only be discrete, so the warranty is removed
  TSearchWarranty(const int &pif, const int &pid)
  : posInFile(pif), posInDomain(pid), suspectedType(3)
  {}
};

PDomain TTabDelimExampleGenerator::domainWithDetection(const string &stem, bool &domainIsNew, PVarList sourceVars, TMetaVector *sourceMetas, PDomain sourceDomain, bool dontCheckStored)
{ 
  headerLines = 1;

  TFileExampleIteratorData fei(stem);
  
  TIdList varNames;
  while(!feof(fei.file) && !readTabAtom(fei, varNames, true, csv));
  if (varNames.empty())
    ::raiseError("unexpected end of file '%s'", fei.filename.c_str());

  TAttributeDescriptions attributeDescriptions, metas;
  classPos = -1;
  int classType = -1;


  list<TSearchWarranty> searchWarranties;

  /**** Parsing the header row */
  
  ITERATE(vector<string>, ni, varNames) {
    /* Parses the header line
       - sets *ni to a real name (without prefix)
       - sets varType to TValue::varType or -1 if the type is not specified
       - sets classPos to the current position, if the attribute is class attribute
         (and reports an error if there is more than one such attribute)
       - to attributeTypes, appends -1 for ordinary atributes, 1 for metas and 0 for ignored */
    int varType = -1; // varType, or -1 for unnown
    attributeTypes->push_back(-1);
    int &attributeType = attributeTypes->back();

    const char *cptr = (*ni).c_str();
    if (*cptr && (cptr[1]=='#')) {
      if (*cptr == 'm')
        attributeType = 1;
      else if (*cptr == 'i')
        attributeType = 0;
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
        attributeType = 1;
      else if (*cptr == 'i')
        attributeType = 0;
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
      
    if ((classPos == ni-varNames.begin())) {
      classType = varType;
    }
    else {
      if (attributeType == 1) {
        metas.push_back(TAttributeDescription(*ni, varType));
        if (varType==-1)
          searchWarranties.push_back(TSearchWarranty(ni-varNames.begin(), -metas.size()));
      }
      else if (attributeType) {
        attributeDescriptions.push_back(TAttributeDescription(*ni, varType));
        if (varType=-1)
          searchWarranties.push_back(TSearchWarranty(ni-varNames.begin(), attributeType==-2 ? -1 : attributeDescriptions.size()-1));
      }
    }
  }

  if (classPos > -1) {
    attributeDescriptions.push_back(TAttributeDescription(varNames[classPos], classType));
    if (classType<0)
      searchWarranties.push_back(TSearchWarranty(classPos, attributeDescriptions.size()-1));
  }
  else
    classPos = attributeDescriptions.size()-1;

  if (!searchWarranties.empty()) {
    vector<string> atoms;
    char numTest[64];
    while (!feof(fei.file) && !searchWarranties.empty()) {
      if (!readTabAtom(fei, atoms, true, csv))
        continue;
    
      for(list<TSearchWarranty>::iterator wi(searchWarranties.begin()), we(searchWarranties.end()); wi!=we; wi++) {
        const string &atom = atoms[(*wi).posInFile];

        // only discrete attributes can have values longer than 63 characters
        if (atom.length()>63) {
          if ((*wi).posInDomain<0)
            metas[-(*wi).posInDomain - 1].varType = TValue::INTVAR;
          else
            attributeDescriptions[(*wi).posInDomain].varType = TValue::INTVAR;
          wi = searchWarranties.erase(wi);
          wi--;
          continue;
        }

        const char *ceni = atom.c_str();
        if (   !*ceni
            || !ceni[1] && ((*ceni=='?') || (*ceni=='.') || (*ceni=='~') || (*ceni=='*') || (*ceni=='-'))
            || (atom == "NA"))
          continue;

        // we have encountered some value
        if ((*wi).suspectedType == 3) 
          (*wi).suspectedType = 2;

        // If the attribute is a digit, it can be anything
        if ((!ceni[1]) && (*ceni>='0') && (*ceni<='9'))
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
        while (*eptr==32)
          eptr++;
        if (*eptr) {
          if ((*wi).posInDomain<0)
            metas[-(*wi).posInDomain - 1].varType = TValue::INTVAR;
          else
            attributeDescriptions[(*wi).posInDomain].varType = TValue::INTVAR;
          wi = searchWarranties.erase(wi);
          wi--;
          continue;
        }
      }
    }


    ITERATE(list<TSearchWarranty>, wi, searchWarranties) {
      const string &name = varNames[(*wi).posInFile];
      if ((*wi).suspectedType == 3)
        ::raiseError("cannot determine type for attribute '%s'", name.c_str());

      int type = (*wi).suspectedType == 2 ? TValue::INTVAR : TValue::FLOATVAR;
      if ((*wi).posInDomain<0)
        metas[-(*wi).posInDomain - 1].varType = type;
      else
        attributeDescriptions[(*wi).posInDomain].varType = type;
    }
  }

  if (sourceDomain) {
    if (!checkDomain(sourceDomain.AS(TDomain), &attributeDescriptions, true, NULL))
      raiseError("given domain does not match the file");
    else
      return sourceDomain;
  }

  int *metaIDs = mlnew int[metas.size()];
  PDomain newDomain = prepareDomain(&attributeDescriptions, true, &metas, domainIsNew, dontCheckStored ? NULL : &knownDomains, sourceVars, sourceMetas, metaIDs);

  int *mid = metaIDs;
  PITERATE(TIntList, ii, attributeTypes)
    if (*ii == 1)
      *ii = *(mid++);

  mldelete metaIDs;

  return newDomain;
}


PDomain TTabDelimExampleGenerator::domainWithoutDetection(const string &stem, bool &domainIsNew, PVarList sourceVars, TMetaVector *sourceMetas, PDomain sourceDomain, bool dontCheckStored)
{
  TFileExampleIteratorData fei(stem);
  
  TIdList varNames, varTypes, varFlags;
  
  while(!feof(fei.file) && !readTabAtom(fei, varNames, true, csv));
  if (varNames.empty())
    ::raiseError("empty file");

  while(!feof(fei.file) && !readTabAtom(fei, varTypes, false, csv));
  if (varTypes.empty())
    ::raiseError("cannot read types of attributes");

  while(!feof(fei.file) && !readTabAtom(fei, varFlags, true, csv));

  if (varNames.size() != varTypes.size())
    ::raiseError("mismatching number of attributes and their types.");
  if (varNames.size() < varFlags.size())
    ::raiseError("too many flags (third line too long)");
  while (varFlags.size() < varNames.size())
    varFlags.push_back("");

  TAttributeDescriptions attributeDescriptions, metas;
  TAttributeDescription classDescription("", 0);
  classPos = -1;
  headerLines = 3;

  attributeTypes = mlnew TIntList(varNames.size(), -1);
  DCs = mlnew TStringList(varNames.size(), "");

  TIdList::iterator vni(varNames.begin()), vne(varNames.end());
  TIdList::iterator ti(varTypes.begin());
  TIdList::iterator fi(varFlags.begin()), fe(varFlags.end());
  TIntList::iterator ati(attributeTypes->begin());
  for(; vni!=vne; fi++, vni++, ti++, ati++) {
    TAttributeDescription *attributeDescription = NULL;
    bool ordered = false;

    if (fi!=fe) {
      TProgArguments args("dc: ordered", *fi, false);

      if (args.direct.size()) {
        if (args.direct.size()>1)
          ::raiseError("invalid flags for attribute '%s'", (*vni).c_str());
        string direct = args.direct.front();
        if ((direct=="s") || (direct=="skip") || (direct=="i") || (direct=="ignore"))
          *ati = 0;
        else if ((direct=="c") || (direct=="class"))
          if (classPos==-1) {
            classPos = vni - varNames.begin();
            classDescription.name = *vni;
            attributeDescription = &classDescription;
          }
          else 
            ::raiseError("multiple attributes are specified as class attribute ('%s' and '%s')", (*vni).c_str(), (*vni).c_str());
        else if ((direct=="m") || (direct=="meta"))
          *ati = 1;
      }

      if (args.exists("dc"))
        DCs->at(vni-varNames.begin()) = args["dc"];

      ordered = args.exists("ordered");
    }

    if (!*ati)
      continue;

    if (!attributeDescription) {// this can only be defined if the attribute is a class attribute
      if (*ati==1) {
        metas.push_back(TAttributeDescription(*vni, -1, ordered));
        attributeDescription = &metas.back();
      }
      else {
        attributeDescriptions.push_back(TAttributeDescription(*vni, -1, ordered));
        attributeDescription = &attributeDescriptions.back();
      }
    }
    else
      attributeDescription->ordered = ordered;

    if (!(*ti).length())
      ::raiseError("type for attribute '%s' is missing", (*vni).c_str());

    if ((*ti=="c") || (*ti=="continuous") || (*ti=="float") || (*ti=="f"))
      attributeDescription->varType = TValue::FLOATVAR;
    else if ((*ti=="d") || (*ti=="discrete") || (*ti=="e") || (*ti=="enum"))
      attributeDescription->varType = TValue::INTVAR;
    else if (*ti=="string")
      attributeDescription->varType = stringVarType;
    else {
      attributeDescription->varType = TValue::INTVAR;
      attributeDescription->values = mlnew TStringList;

      string vals;
      ITERATE(string, ci, *ti)
        if (*ci==' ') {
          if (vals.length())
            attributeDescription->values->push_back(vals);
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
        attributeDescription->values->push_back(vals);
    }
  }

  if (classPos > -1)
    attributeDescriptions.push_back(classDescription);

  if (sourceDomain) {
    if (!checkDomain(sourceDomain.AS(TDomain), &attributeDescriptions, true, NULL))
      raiseError("given domain does not match the file");
    else
      return sourceDomain;
  }

  int *metaIDs = mlnew int[metas.size()];
  PDomain newDomain = prepareDomain(&attributeDescriptions, classPos>=0, &metas, domainIsNew, dontCheckStored ? NULL : &knownDomains, sourceVars, sourceMetas, metaIDs);

  int *mid = metaIDs;
  PITERATE(TIntList, ii, attributeTypes)
    if (*ii == 1)
      *ii = *(mid++);

  mldelete metaIDs;

  return newDomain;
}


bool atomsEmpty(const TIdList &atoms)
{ const_ITERATE(TIdList, ai, atoms)
    if ((*ai).length())
      return false;
  return true;
}


/*  Reads a list of atoms from a line of tab or comma delimited file. Atom consists of any characters
    except \n, \r and \t (and ',' if csv=true). Multiple spaces are replaced by a single space. Atoms
    are separated by \t or ',' if csv=true. Lines end with \n or \r. Lines which begin with | are ignored. */
bool readTabAtom(TFileExampleIteratorData &fei, TIdList &atoms, bool escapeSpaces, bool csv)
{
  atoms.clear();

  if (!fei.file)
    raiseErrorWho("TabDelimExampleGenerator", "file not opened");

  if (feof(fei.file))
    return false;

  fei.line++;

  char c;
  int col = 0;
  string atom;
  for(;;) {
    c = fgetc(fei.file);

    if (c==EOF)
      break;
    if (!col && (c=='|')) {
      for (c=fgetc(fei.file); (c!='\r') && (c!='\n') && (c!=EOF); c=fgetc(fei.file));
      return false;
    }

    col++;

    switch(c) {
      case '\r':
      case '\n':
        if (atom.length() || atoms.size())
          atoms.push_back(atom);  // end of line
        return atoms.size()>0;

      case '\t':
        atoms.push_back(atom);
        atom = string();
        break;

      case ',':
        if (csv) {
          atoms.push_back(trim(atom));
          atom = string();
          break;
        }
        // else fallthrough

      case ' ':
        atom += c;
        break;

      case '\\':
        if (escapeSpaces)
          c = fgetc(fei.file);

      default:
        if ((c>=' ') || (c<0))
          atom += c;
    };
  }
  
  if (ferror(fei.file))
    raiseErrorWho("TabDelimExampleGenerator", "error while reading line %i of file '%s'", fei.line, fei.filename.c_str());

  if (atom.length() || atoms.size())
    atoms.push_back(csv ? trim(atom) : atom);

  return atoms.size()>0;
}




// ********* Output ********* //


void tabDelim_writeExample(FILE *file, const TExample &ex, char delim)
{ TVarList::const_iterator vi(ex.domain->variables->begin()), ve(ex.domain->variables->end());
  TExample::const_iterator ri(ex.begin());
  string st;
  (*(vi++))->val2str(*(ri++), st);
  fprintf(file, "%s", st.c_str());
  for(; vi!=ve; vi++, ri++) {
    (*vi)->val2str(*ri, st);
    fprintf(file, "%c%s", delim, st.c_str());
  }

  const_ITERATE(TMetaVector, mi, ex.domain->metas) {
    (*mi).variable->val2str(ex[(*mi).id], st);
    fprintf(file, "%c%s", delim, st.c_str());
  }
  fprintf(file, "\n");
}


void tabDelim_writeExamples(FILE *file, PExampleGenerator rg, char delim)
{ PEITERATE(gi, rg)
    tabDelim_writeExample(file, *gi, delim);
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


void tabDelim_writeDomainWithoutDetection(FILE *file, PDomain dom, char delim)
{ 
  char delims[2] = {delim, 0};

  { int notFirst = 0;
    const_PITERATE(TVarList, vi, dom->variables) {
      if (notFirst++)
        fprintf(file, "%c%s", delim, (*vi)->name.c_str());
      else
        fprintf(file, "%s", (*vi)->name.c_str());
    }

    const_ITERATE(TMetaVector, mi, dom->metas) {
      if (notFirst++)
        fprintf(file, "%c%s", delim, (*mi).variable->name.c_str());
      else
        fprintf(file, "%s", (*mi).variable->name.c_str());
    }
    fprintf(file, "\n");
  }

  { int notFirst=0;
    const_PITERATE(TVarList, vi, dom->variables) {
      if (notFirst++)
        fprintf(file, delims);
      printVarType(file, *vi);
    }
    const_ITERATE(TMetaVector, mi, dom->metas) {
      if (notFirst++)
        fprintf(file, delims);
      printVarType(file, (*mi).variable);
    }
    fprintf(file, "\n");
  }

  { if (dom->attributes->size())
      for(int i = dom->attributes->size()-1; i--; )
        fprintf(file, delims);

    if (dom->classVar)
      fprintf(file, "%cclass", delim);

    int notFirst=dom->variables->size();
    
    { const_ITERATE(TMetaVector, mi, dom->metas) {
        if (notFirst++)
          fprintf(file, delims);
        fprintf(file, "meta");
      }
    }
   fprintf(file, "\n");
  }
}


/* If discrete value can be mistakenly read as continuous, we need to add the prefix.
   This needs to be checked. */
bool tabDelim_checkNeedsD(PVariable var)
{
  bool floated = false;
  TEnumVariable *enumv = var.AS(TEnumVariable);
  if (enumv) {
    TValue val;
    string sval;
    char svalc[65];

    if (!enumv->firstValue(val))
      return true;
    
    do {
      enumv->val2str(val, sval);
      if (sval.size()>63)
        return false;

      if ((sval.size()==1) && (sval[0]>='0') && (sval[0]<='9'))
        continue;

      // Convert commas into dots
      char *sc = svalc;
      ITERATE(string, si, sval) {
        *(sc++) = *si==',' ? '.' : *si;
        *sc = 0;

        char *eptr;
        strtod(svalc, &eptr);
        if (*eptr)
          return false;
        else
          floated = true;
      }
    } while (enumv->nextValue(val));
  }
  
  // All values were either one digit or successfully interpreted as continuous
  // We need to return true if there were some that were not one-digit...
  return floated;
}


void tabDelim_writeDomainWithDetection(FILE *file, PDomain dom, char delim)
{
  char delims[2] = {delim, 0};

  int notFirst = 0;
  const_PITERATE(TVarList, vi, dom->attributes)
    fprintf(file, "%s%s%s", (notFirst++ ? delims : ""), (tabDelim_checkNeedsD(*vi) ? "D#" : ""), (*vi)->name.c_str());
  
  if (dom->classVar)
    fprintf(file, "%s%s%s", (notFirst++ ? delims : ""), (tabDelim_checkNeedsD(dom->classVar) ? "cD#" : "c#"), dom->classVar->name.c_str());

  const_ITERATE(TMetaVector, mi, dom->metas)
    fprintf(file, "%s%s%s", (notFirst++ ? delims : ""), (tabDelim_checkNeedsD((*mi).variable) ? "mD#" : "m#"), (*mi).variable->name.c_str());

  fprintf(file, "\n");
}


void tabDelim_writeDomain(FILE *file, PDomain dom, bool autodetect, char delim)
{ if (autodetect)
    tabDelim_writeDomainWithDetection(file, dom, delim);
  else 
    tabDelim_writeDomainWithoutDetection(file, dom, delim);
}
