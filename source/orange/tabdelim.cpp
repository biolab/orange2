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
#include "strings.hpp"
#include "getarg.hpp"

#include "values.hpp"
#include "vars.hpp"
#include "stringvars.hpp"
#include "pythonvars.hpp"
#include "domain.hpp"
#include "examples.hpp"

#include "tabdelim.ppp"

int readTabAtom(TFileExampleIteratorData &fei, vector<string> &atoms, bool escapeSpaces=true, bool csv = false);
bool atomsEmpty(const vector<string> &atoms);


TDomainDepot TTabDelimExampleGenerator::domainDepot_tab;
TDomainDepot TTabDelimExampleGenerator::domainDepot_txt;


const TTabDelimExampleGenerator::TIdentifierDeclaration TTabDelimExampleGenerator::typeIdentifiers[] =
 {{"discrete", 0, TValue::INTVAR},      {"d", 0, TValue::INTVAR},
  {"continuous", 0, TValue::FLOATVAR},  {"c", 0, TValue::FLOATVAR},
  {"string", 0, STRINGVAR},             {"s", 0, STRINGVAR},
  {"python", 0, PYTHONVAR},             {"python:", 7, PYTHONVAR},
  {NULL, 0}};


TTabDelimExampleGenerator::TTabDelimExampleGenerator(const TTabDelimExampleGenerator &old)
: TFileExampleGenerator(old),
  attributeTypes(mlnew TIntList(old.attributeTypes.getReference())),
  DCs(old.DCs),
  classPos(old.classPos),
  headerLines(old.headerLines),
  csv(old.csv)
{}


TTabDelimExampleGenerator::TTabDelimExampleGenerator(const string &afname, bool autoDetect, bool acsv, PVarList sourceVars, TMetaVector *sourceMetas, PDomain sourceDomain, bool dontCheckStored, bool dontStore, const char *aDK, const char *aDC)
: TFileExampleGenerator(afname, PDomain()),
  attributeTypes(mlnew TIntList()),
  DCs(),
  DK(aDK ? strcpy((char *)malloc(strlen(aDK)+1), aDK) : NULL),
  DC(aDC ? strcpy((char *)malloc(strlen(aDC)+1), aDC) : NULL),
  classPos(-1),
  headerLines(0),
  csv(acsv)
{ 
  // domain needs to be initialized after attributeTypes, DCs, classPos, headerLines
  domain = readDomain(afname, autoDetect, sourceVars, sourceMetas, sourceDomain, dontCheckStored, dontStore);

  TFileExampleIteratorData fei(afname);
  
  vector<string> atoms;
  for (int i = headerLines; !feof(fei.file) && i--; )
    // read one line (not counting comment lines, but counting empty lines)
    while(!feof(fei.file) && (readTabAtom(fei, atoms, true, csv) == -1));

  startDataPos = ftell(fei.file);
  startDataLine = fei.line;
}


TTabDelimExampleGenerator::~TTabDelimExampleGenerator()
{
  if (DK)
    free(DK);

  if (DC)
    free(DC);
}

bool TTabDelimExampleGenerator::readExample(TFileExampleIteratorData &fei, TExample &exam)
{
  vector<string> atoms;
  // read lines until eof or a non-empty line
  while(!feof(fei.file) && ((readTabAtom(fei, atoms, true, csv)>0) || atomsEmpty(atoms))) {
    vector<string>::iterator ii(atoms.begin()), ie(atoms.end());
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
  vector<string>::iterator ai(atoms.begin());
  vector<int>::iterator si(attributeTypes->begin()), se(attributeTypes->end());
  vector<vector<string> >::iterator dci(DCs.begin()), dce(DCs.end());
  int pos=0;
  for (; (si!=se); pos++, si++, ai++) {
    if (*si) { // if attribute is not to be skipped
      string valstr;

      // Check for don't care
      valstr = *ai;
      if (dci != dce)
        ITERATE(vector<string>, dcii, *dci)
          if (*dcii == valstr) {
            valstr = '?';
            break;
          }

      if (valstr != "?") {
        if (!valstr.length() || (valstr == "NA") || (valstr == ".") || (DC && (valstr == DC)))
          valstr = "?";
        else if ((valstr == "*") || (DK && (valstr == DK)))
          valstr = "~";
      }

      try {
        if (*si==-1)
          if (pos==classPos) { // if this is class value
            TValue cval;
            domain->classVar->filestr2val(valstr, cval, exam);
            exam.setClass(cval);
          }
          else { // if this is a normal value
            (*vi++)->filestr2val(valstr, *ei++, exam);
          }
        else { // if this is a meta value
          TMetaDescriptor *md = domain->metas[*si];
          _ASSERT(md!=NULL);
          TValue mval;
          md->variable->filestr2val(valstr, mval, exam);

          exam.setMeta(*si, mval);
        }
      }
      catch (mlexception &err) {
        raiseError("file '%s', line '%i': %s", fei.filename.c_str(), fei.line, err.what());
      }
    }

    if (dci != dce)
      dci++;
  }

  if (pos==classPos) // if class is the last value in the line, it is set here
    domain->classVar->filestr2val(ai==atoms.end() ? "?" : *(ai++), exam[domain->variables->size()-1], exam);

  while ((ai!=atoms.end()) && !(*ai).length()) ai++; // line must be empty from now on

  if (ai!=atoms.end()) {
	vector<string>::iterator ii=atoms.begin();
	string s=*ii;
	while(++ii!=atoms.end()) s+=" "+*ii;
    raiseError("example of invalid length (%s)", s.c_str());
  }

  return true;
}


char *TTabDelimExampleGenerator::mayBeTabFile(const string &stem)
{
  vector<string> varNames, atoms;
  vector<string>::const_iterator vi, ai, ei;

  TFileExampleIteratorData fei(stem);

  // if there is no names line, it is not .tab
  while(!feof(fei.file) && (readTabAtom(fei, varNames, true, csv)==-1));
  if (varNames.empty()) {
    char *res = mlnew char[128];
    res = strcpy(res, "empty file");
    return res;
  }

  // if any name contains the correct hash formatting it is not tab-delim it's more likely .txt
  for(vi = varNames.begin(), ei = varNames.end(); vi!=ei; vi++) {
    const char *c = (*vi).c_str();
    if ((*c=='m') || (*c=='c') || (*c=='i'))
      c++;
    if (   ((*c=='D') || (*c=='C') || (*c=='S'))
        && (c[1]=='#')) {
      char *res= mlnew char[128 + (*vi).size()];
      sprintf(res, "attribute name '%s' looks suspicious", (*vi).c_str());
      return res;
    }
  }

  // if there is no var types line, it is not .tab
  while(!feof(fei.file) && (readTabAtom(fei, atoms, true, csv)==-1));
  if (atoms.empty()) {
    char *res = mlnew char[128];
    res = strcpy(res, "no line with attribute types");
    return res;
  }

  if (atoms.size() != varNames.size())
    raiseError("the number of attribute types does not match the number of attributes");

  // Each atom must be either 'd', 'c' or 's', or contain a space
  for(vi = varNames.begin(), ai = atoms.begin(), ei = atoms.end(); ai != ei; ai++, vi++) {
    const char *c = (*ai).c_str();
    if (!*c) {
      char *res= mlnew char[128 + (*vi).size()];
      sprintf(res, "empty type entry for attribute '%s'", (*vi).c_str());
      return res;
    }

    const TIdentifierDeclaration *tid = typeIdentifiers;
    for(; tid->identifier && (tid->matchRoot ? strncmp(tid->identifier, c, tid->matchRoot) : strcmp(tid->identifier, c)); tid++);
    if (tid->identifier)
      continue;

    for(; *c && (*c!=' '); c++);
      if (!*c) {
        char *res= mlnew char[128 + (*vi).size() + (*ai).size()];
        sprintf(res, "attribute '%s' is defined as having only one value ('%s')", (*vi).c_str(), (*ai).c_str());
        return res;
      }
  }

  // if there is no flags line, it is not .tab
  while(!feof(fei.file) && (readTabAtom(fei, atoms, true, csv)==-1));
  if (feof(fei.file)) {
    char *res = mlnew char[128];
    res = strcpy(res, "file has only two lines");
    return res;
  }

  if (atoms.size() > varNames.size())
    raiseError("the number of attribute options is greater than the number of attributes");

  // Check flags
  for(vi = varNames.begin(), ai = atoms.begin(), ei = atoms.end(); ai != ei; ai++, vi++) {
    TProgArguments args("dc: ordered", *ai, false);

    if (args.unrecognized.size()) {
      char *res= mlnew char[128 + (*vi).size()];
      sprintf(res, "unrecognized options at attribute '%s'", (*vi).c_str());
      return res;
    }

    if (args.direct.size()) {
      if (args.direct.size()>1) {
        char *res= mlnew char[128 + (*vi).size()];
        sprintf(res, "too many direct options at attribute '%s'", (*vi).c_str());
        return res;
      }

      static const char *legalDirects[] = {"s", "skip","i", "ignore", "c", "class", "m", "meta", NULL};
      string &direct = args.direct.front();
      const char **lc = legalDirects;
      while(*lc && strcmp(*lc, direct.c_str()))
        lc++;
      if (!*lc) {
        char *res= mlnew char[128 + (*vi).size() + (*ai).size()];
        sprintf(res, "unrecognized option ('%s') at attribute '%s'", (*ai).c_str(), (*vi).c_str());
        return res;
      }
    }
  }

  return NULL;
}

PDomain TTabDelimExampleGenerator::readDomain(const string &stem, const bool autoDetect, PVarList sourceVars, TMetaVector *sourceMetas, PDomain sourceDomain, bool dontCheckStored, bool dontStore)
{ 
  // non-NULL when this cannot be tab file (reason given as result)
  // NULL if this seems a valid tab file
  char *isNotTab = mayBeTabFile(stem);

  if (autoDetect) {
    if (!isNotTab)
      raiseWarning("'%s' is being loaded as .txt, but could be .tab file", stem.c_str());
    else
      mldelete isNotTab;

    return domainWithDetection(stem, sourceVars, sourceMetas, sourceDomain, dontCheckStored);
  }

  else {
    if (isNotTab) {
      raiseWarning("'%s' is being loaded as .tab, but looks more like .txt file\n(%s)", stem.c_str(), isNotTab);
      mldelete isNotTab;
    }

    return domainWithoutDetection(stem, sourceVars, sourceMetas, sourceDomain, dontCheckStored);
  }
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

PDomain TTabDelimExampleGenerator::domainWithDetection(const string &stem, PVarList sourceVars, TMetaVector *sourceMetas, PDomain sourceDomain, bool dontCheckStored)
{ 
  headerLines = 1;

  TFileExampleIteratorData fei(stem);
  
  vector<string> varNames;
  // read the next non-comment line
  while(!feof(fei.file) && (readTabAtom(fei, varNames, true, csv)==-1));
  if (varNames.empty())
    ::raiseError("unexpected end of file '%s'", fei.filename.c_str());

  TDomainDepot::TAttributeDescriptions attributeDescriptions, metas;
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
        varType = STRINGVAR;

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
        varType = STRINGVAR;
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
        metas.push_back(TDomainDepot::TAttributeDescription(*ni, varType));
        if (varType==-1)
          searchWarranties.push_back(TSearchWarranty(ni-varNames.begin(), -metas.size()));
      }
      else if (attributeType) {
        attributeDescriptions.push_back(TDomainDepot::TAttributeDescription(*ni, varType));
        if (varType=-1)
          searchWarranties.push_back(TSearchWarranty(ni-varNames.begin(), attributeType==-2 ? -1 : attributeDescriptions.size()-1));
      }
    }
  }

  if (classPos > -1) {
    attributeDescriptions.push_back(TDomainDepot::TAttributeDescription(varNames[classPos], classType));
    if (classType<0)
      searchWarranties.push_back(TSearchWarranty(classPos, attributeDescriptions.size()-1));
  }
  else
    classPos = attributeDescriptions.size()-1;

  if (!searchWarranties.empty()) {
    vector<string> atoms;
    char numTest[64];
    while (!feof(fei.file) && !searchWarranties.empty()) {
      // seek to the next line non-empty non-comment line
      if (readTabAtom(fei, atoms, true, csv) <= 0)
        continue;
    
      for(list<TSearchWarranty>::iterator wi(searchWarranties.begin()), we(searchWarranties.end()); wi!=we; wi++) {
        if ((*wi).posInFile >= atoms.size())
          raiseError("line %i too short", fei.line);

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
            || (atom == "NA") || (DC && (atom == DC)) || (DK && (atom == DK)))
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
        raiseWarning("cannot determine type for attribute '%s'; the attribute will be ignored", name.c_str());

      int type = (*wi).suspectedType == 2 ? TValue::INTVAR : TValue::FLOATVAR;
      if ((*wi).posInDomain<0)
        metas[-(*wi).posInDomain - 1].varType = type;
      else
        attributeDescriptions[(*wi).posInDomain].varType = type;
    }

    for(int i = 0; i < attributeDescriptions.size(); )
      if (attributeDescriptions[i].varType == -1)
        attributeDescriptions.erase(attributeDescriptions.begin() + i);
      else
        i++;
  }

  if (sourceDomain) {
    if (!domainDepot_txt.checkDomain(sourceDomain.AS(TDomain), &attributeDescriptions, true, NULL))
      raiseError("given domain does not match the file");
    else
      return sourceDomain;
  }

  int *metaIDs = mlnew int[metas.size()];
  PDomain newDomain = domainDepot_txt.prepareDomain(&attributeDescriptions, true, &metas, sourceVars, sourceMetas, false, dontCheckStored, NULL, metaIDs);

  int *mid = metaIDs;
  PITERATE(TIntList, ii, attributeTypes)
    if (*ii == 1)
      *ii = *(mid++);

  mldelete metaIDs;

  return newDomain;
}


PDomain TTabDelimExampleGenerator::domainWithoutDetection(const string &stem, PVarList sourceVars, TMetaVector *sourceMetas, PDomain sourceDomain, bool dontCheckStored)
{
  TFileExampleIteratorData fei(stem);
  
  vector<string> varNames, varTypes, varFlags;
  
  while(!feof(fei.file) && (readTabAtom(fei, varNames, true, csv) == -1));
  if (varNames.empty())
    ::raiseError("empty file");

  while(!feof(fei.file) && (readTabAtom(fei, varTypes, false, csv) == -1));
  if (varTypes.empty())
    ::raiseError("cannot read types of attributes");

  while(!feof(fei.file) && (readTabAtom(fei, varFlags, true, csv) == -1));

  if (varNames.size() != varTypes.size())
    ::raiseError("mismatching number of attributes and their types.");
  if (varNames.size() < varFlags.size())
    ::raiseError("too many flags (third line too long)");
  while (varFlags.size() < varNames.size())
    varFlags.push_back("");

  TDomainDepot::TAttributeDescriptions attributeDescriptions, metas;
  TDomainDepot::TAttributeDescription classDescription("", 0);
  classPos = -1;
  headerLines = 3;

  attributeTypes = mlnew TIntList(varNames.size(), -1);

  vector<string>::iterator vni(varNames.begin()), vne(varNames.end());
  vector<string>::iterator ti(varTypes.begin());
  vector<string>::iterator fi(varFlags.begin()), fe(varFlags.end());
  TIntList::iterator ati(attributeTypes->begin());
  for(; vni!=vne; fi++, vni++, ti++, ati++) {
    TDomainDepot::TAttributeDescription *attributeDescription = NULL;
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

      if (args.exists("dc")) {
        const int ind = vni-varNames.begin();
        ITERATE(TMultiStringParameters, mi, args.options)
          if ((*mi).first == "dc") {
            while (DCs.size() <= ind)
              DCs.push_back(vector<string>());
            DCs.at(ind).push_back((*mi).second);
          }
      }

      ordered = args.exists("ordered");
    }

    if (!*ati)
      continue;

    if (!attributeDescription) {// this can only be defined if the attribute is a class attribute
      if (*ati==1) {
        metas.push_back(TDomainDepot::TAttributeDescription(*vni, -1, *ti, ordered));
        attributeDescription = &metas.back();
      }
      else {
        attributeDescriptions.push_back(TDomainDepot::TAttributeDescription(*vni, -1, *ti, ordered));
        attributeDescription = &attributeDescriptions.back();
      }
    }
    else
      attributeDescription->ordered = ordered;

    if (!(*ti).length())
      ::raiseError("type for attribute '%s' is missing", (*vni).c_str());

    const TIdentifierDeclaration *tid = typeIdentifiers;
    for(; tid->identifier; tid++)
      if (!(tid->matchRoot ? strncmp(tid->identifier, (*ti).c_str(), tid->matchRoot)
                           : strcmp(tid->identifier, (*ti).c_str()))) {
        attributeDescription->varType = tid->varType;
        break;
      }
    if (!tid->identifier) {
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
    if (!domainDepot_tab.checkDomain(sourceDomain.AS(TDomain), &attributeDescriptions, classPos >= 0, NULL))
      raiseError("given domain does not match the file");
    else
      return sourceDomain;
  }

  int *metaIDs = mlnew int[metas.size()];
  PDomain newDomain = domainDepot_tab.prepareDomain(&attributeDescriptions, classPos>=0, &metas, sourceVars, sourceMetas, false, dontCheckStored, NULL, metaIDs);

  int *mid = metaIDs;
  PITERATE(TIntList, ii, attributeTypes)
    if (*ii == 1)
      *ii = *(mid++);

  mldelete metaIDs;

  return newDomain;
}


bool atomsEmpty(const vector<string> &atoms)
{ const_ITERATE(vector<string>, ai, atoms)
    if ((*ai).length())
      return false;
  return true;
}


/*  Reads a list of atoms from a line of tab or comma delimited file. Atom consists of any characters
    except \n, \r and \t (and ',' if csv=true). Multiple spaces are replaced by a single space. Atoms
    are separated by \t or ',' if csv=true. Lines end with \n or \r. Lines which begin with | are ignored.
   
    Returns number of atoms, -1 for comment line and -2 for EOF
    */
int readTabAtom(TFileExampleIteratorData &fei, vector<string> &atoms, bool escapeSpaces, bool csv)
{
  atoms.clear();

  if (!fei.file)
    raiseErrorWho("TabDelimExampleGenerator", "file not opened");

  if (feof(fei.file))
    return -2;

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
      return -1;
    }

    col++;

    switch(c) {
      case '\r':
      case '\n':
        if (atom.length() || atoms.size())
          atoms.push_back(trim(atom));  // end of line
        if (c == '\r') {
          c = fgetc(fei.file);
          if (c != '\n')
            fseek(fei.file, SEEK_CUR, -1);
        }
        return atoms.size();

      case '\t':
        atoms.push_back(trim(atom));
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
        if (escapeSpaces) {
          c = fgetc(fei.file);
          if (c != ' ')
            atom += '\\';
        }

      default:
        // trim left
        if ((c>=' ') || (c<0))
          atom += c;
    };
  }
  
  if (ferror(fei.file))
    raiseErrorWho("TabDelimExampleGenerator", "error while reading line %i of file '%s'", fei.line, fei.filename.c_str());

  if (atom.length() || atoms.size())
    atoms.push_back(csv ? trim(atom) : atom);

  ITERATE(vector<string>, ai, atoms) {
  }
  return atoms.size();
}




// ********* Output ********* //


#define PUTDELIM { if (ho) putc(delim, file); else ho = true; }

void tabDelim_writeExample(FILE *file, const TExample &ex, char delim)
{ 
}


void tabDelim_writeExamples(FILE *file, PExampleGenerator rg, char delim, const char *DK, const char *DC)
{ 
  PEITERATE(ex, rg) {
    TVarList::const_iterator vi((*ex).domain->variables->begin()), ve((*ex).domain->variables->end());
    TExample::const_iterator ri((*ex).begin());
    string st;
    bool ho = false;

    for(; vi!=ve; vi++, ri++) {
      PUTDELIM;
      if (DK && ((*ri).valueType == valueDK))
        fprintf(file, DK);
      else if (DC && ((*ri).valueType == valueDC))
        fprintf(file, DC);
      else {
        (*vi)->val2filestr(*ri, st, *ex);
        fprintf(file, st.c_str());
      }
    }

    const_ITERATE(TMetaVector, mi, (*ex).domain->metas) {
      PUTDELIM;
      if (DK && ((*ri).valueType == valueDK))
        fprintf(file, DK);
      else if (DC && ((*ri).valueType == valueDC))
        fprintf(file, DC);
      else {
        (*mi).variable->val2filestr((*ex)[(*mi).id], st, *ex);
        fprintf(file, "%s", st.c_str());
      }
    }
    fprintf(file, "\n");
  }
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

extern TOrangeType PyOrPythonVariable_Type;

void printVarType(FILE *file, PVariable var, bool listDiscreteValues)
{
  TEnumVariable *enumv = var.AS(TEnumVariable);
  if (enumv) {
    TValue val;
    string sval;
    if (!enumv->firstValue(val) || !listDiscreteValues)
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
  else if (var.is_derived_from(TPythonVariable)) {
    if (var.counter->ob_type == (PyTypeObject *)&PyOrPythonVariable_Type)
      fprintf(file, "python");
    else {
      PyObject *pyclassname = PyObject_GetAttrString((PyObject *)(var.counter)->ob_type, "__name__");
      fprintf(file, "python:%s", PyString_AsString(pyclassname));
      Py_DECREF(pyclassname);
    }
  }  
  else
    raiseErrorWho("tabDelim_writeDomain", "tabDelim format supports only discrete, continuous and string variables");
}


void tabDelim_writeDomainWithoutDetection(FILE *file, PDomain dom, char delim, bool listDiscreteValues)
{ 
  TVarList::const_iterator vi, vb(dom->variables->begin()), ve(dom->variables->end());
  TMetaVector::const_iterator mi, mb(dom->metas.begin()), me(dom->metas.end());

  bool ho = false;
  // First line: attribute names
  for(vi = vb; vi!=ve; vi++) {
    PUTDELIM;
    fprintf(file, "%s", (*vi)->name.c_str());
  }
  for(mi = mb; mi!=me; mi++) {
    PUTDELIM;
    fprintf(file, "%s", (*mi).variable->name.c_str());
  }
  fprintf(file, "\n");

  
  // Second line: types
  ho = false;
  for(vi = vb; vi!=ve; vi++) {
    PUTDELIM;
    printVarType(file, *vi, listDiscreteValues);
  }
  for(mi = mb; mi!=me; mi++) {
    PUTDELIM;
    printVarType(file, (*mi).variable, listDiscreteValues);
  }
  fprintf(file, "\n");


  // Third line: "meta" and "-ordered"
  ho = false;
  for(vb = vi = dom->attributes->begin(), ve = dom->attributes->end(); vi!=ve; vi++) {
    PUTDELIM;
    if (((*vi)->varType == TValue::INTVAR) && (*vi)->ordered)
      fprintf(file, "-ordered");
  }
  if (dom->classVar) {
    PUTDELIM;
    fprintf(file, "class");
  }
  for(mi = mb; mi!=me; mi++) {
    PUTDELIM;
    fprintf(file, "meta");
    if (((*mi).variable->varType == TValue::INTVAR) && (*mi).variable->ordered)
      fprintf(file, " -ordered");
 }
 fprintf(file, "\n");
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
  bool ho = false;
  const_PITERATE(TVarList, vi, dom->attributes) {
    PUTDELIM;
    fprintf(file, "%s%s", (tabDelim_checkNeedsD(*vi) ? "D#" : ""), (*vi)->name.c_str());
  }
  
  if (dom->classVar) {
    PUTDELIM;
    fprintf(file, "%s%s", (tabDelim_checkNeedsD(dom->classVar) ? "cD#" : "c#"), dom->classVar->name.c_str());
  }

  const_ITERATE(TMetaVector, mi, dom->metas) {
    PUTDELIM;
    fprintf(file, "%s%s", (tabDelim_checkNeedsD((*mi).variable) ? "mD#" : "m#"), (*mi).variable->name.c_str());
  }

  fprintf(file, "\n");
}


void tabDelim_writeDomain(FILE *file, PDomain dom, bool autodetect, char delim, bool listDiscreteValues)
{ if (autodetect)
    tabDelim_writeDomainWithDetection(file, dom, delim);
  else 
    tabDelim_writeDomainWithoutDetection(file, dom, delim, listDiscreteValues);
}
