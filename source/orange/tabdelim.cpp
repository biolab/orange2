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
#include <vector>
#include <list>
#include <map>

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

int readTabAtom(TFileExampleIteratorData &fei, vector<string> &atoms, bool escapeSpaces=true, bool csv = false, bool allowEmpty=false);
bool atomsEmpty(const vector<string> &atoms);


const TTabDelimExampleGenerator::TIdentifierDeclaration TTabDelimExampleGenerator::typeIdentifiers[] =
 {{"discrete", 0, TValue::INTVAR},      {"d", 0, TValue::INTVAR},
  {"continuous", 0, TValue::FLOATVAR},  {"c", 0, TValue::FLOATVAR},
  {"string", 0, STRINGVAR},             {"s", 0, STRINGVAR},
  {"python", 0, PYTHONVAR},             {"python:", 7, PYTHONVAR},
  {NULL, 0}};


TTabDelimExampleGenerator::TTabDelimExampleGenerator(const TTabDelimExampleGenerator &old)
: TFileExampleGenerator(old),
  attributeTypes(mlnew TIntList(old.attributeTypes.getReference())),
  classPos(old.classPos),
  headerLines(old.headerLines),
  csv(old.csv)
{}


TTabDelimExampleGenerator::TTabDelimExampleGenerator(const string &afname, bool autoDetect, bool acsv, const int createNewOn, vector<int> &status, vector<pair<int, int> > &metaStatus, const char *aDK, const char *aDC, bool noCodedDiscrete, bool noClass)
: TFileExampleGenerator(afname, PDomain()),
  attributeTypes(mlnew TIntList()),
  DK(aDK ? strcpy((char *)malloc(strlen(aDK)+1), aDK) : NULL),
  DC(aDC ? strcpy((char *)malloc(strlen(aDC)+1), aDC) : NULL),
  classPos(-1),
  headerLines(0),
  csv(acsv)
{ 
  // domain needs to be initialized after attributeTypes, classPos, headerLines
  domain = readDomain(afname, autoDetect, createNewOn, status, metaStatus, noCodedDiscrete, noClass);

  TFileExampleIteratorData fei(afname);
  
  vector<string> atoms;
  for (int i = headerLines; !feof(fei.file) && i--; )
    // read one line (not counting comment lines, but the flag line may be empty)
    while(!feof(fei.file) && (readTabAtom(fei, atoms, true, csv, (headerLines==3) && !i) == -1));

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

  exam.removeMetas();

  TExample::iterator ei(exam.begin());
  TVarList::iterator vi(domain->attributes->begin());
  vector<string>::iterator ai(atoms.begin());
  TIntList::iterator si(attributeTypes->begin()), se(attributeTypes->end());
  int pos=0;
  for (; (si!=se); pos++, si++, ai++) {
    if (*si) { // if attribute is not to be skipped and is not a basket
      string valstr;

      // Check for don't care
      valstr = *ai;
      if (!valstr.length() || (valstr == "NA") || (valstr == ".") || (DC && (valstr == DC)))
        valstr = "?";
      else if ((valstr == "*") || (DK && (valstr == DK)))
        valstr = "~";

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

    // the attribute is marked to be skipped, but may also be a basket
    else { 
      if (pos == basketPos) {
        TSplits splits;
        split(*ai, splits);
        ITERATE(TSplits, si, splits)
          basketFeeder->addItem(exam, string(si->first, si->second), fei.line);
      }
    }
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

    if (!strcmp("basket", c))
      continue;

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
  while(!feof(fei.file) && (readTabAtom(fei, atoms, true, csv, true)==-1));
  if (feof(fei.file)) {
    char *res = mlnew char[128];
    res = strcpy(res, "file has only two lines");
    return res;
  }

  if (atoms.size() > varNames.size())
    raiseError("the number of attribute options is greater than the number of attributes");

  // Check flags
  for(vi = varNames.begin(), ai = atoms.begin(), ei = atoms.end(); ai != ei; ai++, vi++) {
    TProgArguments args("dc: ordered", *ai, false, true);

/*  Not any more: now they go into the Variable's dictionary

    if (args.unrecognized.size()) {
      char *res= mlnew char[128 + (*vi).size()];
      sprintf(res, "unrecognized options at attribute '%s'", (*vi).c_str());
      return res;
    }
*/
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

PDomain TTabDelimExampleGenerator::readDomain(const string &stem, const bool autoDetect, const int createNewOn, vector<int> &status, vector<pair<int, int> > &metaStatus, bool noCodedDiscrete, bool noClass)
{ 
  // non-NULL when this cannot be tab file (reason given as result)
  // NULL if this seems a valid tab file
  char *isNotTab = mayBeTabFile(stem);

  TDomainDepot::TAttributeDescriptions descriptions;
  
  if (autoDetect) {
    if (!isNotTab)
      raiseWarning("'%s' is being loaded as .txt, but could be .tab file", stem.c_str());
    readTxtHeader(stem, descriptions);
  }
  else {
    if (isNotTab)
      raiseWarning("'%s' is being loaded as .tab, but looks more like .txt file\n(%s)", stem.c_str(), isNotTab);
    readTabHeader(stem, descriptions);
  }

  if (isNotTab)
    mldelete isNotTab;

  scanAttributeValues(stem, descriptions);
  
  TIntList::iterator ati(attributeTypes->begin());
  TDomainDepot::TPAttributeDescriptions attributeDescriptions, metaDescriptions;
  int ind = 0, lastRegular = -1;

  for(TDomainDepot::TAttributeDescriptions::iterator adi(descriptions.begin()), ade(descriptions.end()); adi != ade; adi++, ati++, ind++) {
    if (!*ati)
      continue;
      
    if (adi->varType == -1) {
      switch (detectAttributeType(*adi, noCodedDiscrete)) {
        case 0:
        case 2:
          adi->varType = TValue::INTVAR;
          break;
          
        case 1:
          adi->varType = TValue::FLOATVAR;
          break;

        case 4:
          adi->varType = STRINGVAR;
          *ati = 1;
          break;

        default:
          raiseWarning("cannot determine type for attribute '%s'; the attribute will be ignored", adi->name.c_str());
          *ati = 0;
          continue;
        }
    }
    
    if (*ati == 1)
      metaDescriptions.push_back(&*adi);
      
    else if ((classPos != ind) && (basketPos != ind)) {
      attributeDescriptions.push_back(&*adi);
      lastRegular = ind;
    }
  }
  
  if (classPos > -1)
    attributeDescriptions.push_back(&descriptions[classPos]);
  else if (autoDetect && !noClass)
    classPos = lastRegular;
    
  if (basketPos >= 0)
//    basketFeeder = mlnew TBasketFeeder(sourceDomain, createNewOn == TVariable::OK, false);
    basketFeeder = mlnew TBasketFeeder(PDomain(), createNewOn == TVariable::OK, false);
    
/*  if (sourceDomain) {
    if (!domainDepot_tab.checkDomain(sourceDomain.AS(TDomain), &attributeDescriptions, classPos >= 0, NULL))
      raiseError("given domain does not match the file");

    if (basketFeeder)
      basketFeeder->domain = sourceDomain;
    return sourceDomain;
  }
*/
  PDomain newDomain = domainDepot.prepareDomain(&attributeDescriptions, classPos>-1, &metaDescriptions, createNewOn, status, metaStatus);

  vector<pair<int, int> >::const_iterator mid(metaStatus.begin());
  PITERATE(TIntList, ii, attributeTypes)
    if (*ii == 1)
      *ii = mid++ ->first;

  if (basketFeeder)
    basketFeeder->domain = newDomain;

  return newDomain;
}



int TTabDelimExampleGenerator::detectAttributeType(TDomainDepot::TAttributeDescription &desc, const bool noCodedDiscrete)
{
  char numTest[64];

  int status = 3;  //  3 - not encountered any values, 2 - can be coded discrete, 1 - can be float, 0 - must be nominal
                   //  4 (set later) - string value
  typedef map<string, int> msi;
  ITERATE(msi, vli, desc.values) {

    if (vli->first.length() > 63) {
      status = 0;
      break;
    }
    
    const char *ceni = vli->first.c_str();
    if (   !*ceni
        || !ceni[1] && ((*ceni=='?') || (*ceni=='.') || (*ceni=='~') || (*ceni=='*'))
        || !strcmp(ceni, "NA") || (DC && !strcmp(ceni, DC)) || (DK && !strcmp(ceni, DK)))
      continue;
    
    if (status == 3)
      status = 2;

    if ((status == 2) && (ceni[1] || (*ceni<'0') || (*ceni>'9')))
      status = noCodedDiscrete ? 2 : 1;
      
    if (status == 1) {
      strcpy(numTest, ceni);
      for(char *sc = numTest; *sc; sc++)
        if (*sc == ',')
          *sc = '.';

      char *eptr;
      strtod(numTest, &eptr);
      while (*eptr==32)
        eptr++;
      if (*eptr) {
        status = 0;
        break;
      }
    }
  }
  
  /* Check whether this is a string attribute:
     - has more than 20 values
     - less than half of the values appear more than once */
  if ((status==0) && (desc.values.size() > 20)) {
      int more2 = 0;
      for(map<string, int>::const_iterator dvi(desc.values.begin()), dve(desc.values.end()); dvi != dve; dvi++) {
        if (dvi->second > 1)
          more2++;
      }
      if (more2*2 < desc.values.size()) {
        status = 4;
      }
  }
  return status;
}




/* These are the rules for determining the attribute types.

   There are three ways to determine a type.

   1. By header prefixes to attribute names.
      The prefix is formed by [cmi][DCS]#
      c, m and i mean class attribute, meta attribute and ignore,
      respectively.
      D, C and S mean discrete, continuous and string attributes.


!!! NOT TRUE:

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


void TTabDelimExampleGenerator::scanAttributeValues(const string &stem, TDomainDepot::TAttributeDescriptions &desc)
{
  TFileExampleIteratorData fei(stem);

  vector<string> atoms;
  vector<string>::const_iterator ai, ae;
  TDomainDepot::TAttributeDescriptions::iterator di, db(desc.begin()), de(desc.end());

  for (int i = headerLines; !feof(fei.file) && i--; )
    while(!feof(fei.file) && (readTabAtom(fei, atoms, true, csv, (headerLines==3) && !i) == -1));

  while (!feof(fei.file)) {
    if (readTabAtom(fei, atoms, true, csv) <= 0)
      continue;
    
    for(di = db, ai = atoms.begin(), ae = atoms.end(); (di != de) && (ai != ae); di++, ai++) {

      //skip the attribute if it is a FLOATVAR or STRINGVAR
      if ((di->varType != TValue::FLOATVAR) && (di->varType != STRINGVAR)) {

          const char *ceni = ai->c_str();

          if (   !*ceni
              || !ceni[1] && ((*ceni=='?') || (*ceni=='.') || (*ceni=='~') || (*ceni=='*'))
              || (*ai == "NA") || (DC && (*ai == DC)) || (DK && (*ai == DK)))
             continue;

          //increase counter or insert - THIS PART IS SLOW!
          //maybe it would be faster if di->values was a unordered_map?
          map<string, int>::iterator vf = di->values.lower_bound(*ai);
          if ((vf != di->values.end()) && (vf->first == *ai)) {
            vf->second++;
          }
          else {
            di->values.insert(vf, make_pair(*ai, 1));
          }

      }
    }
  }

}


void TTabDelimExampleGenerator::readTxtHeader(const string &stem, TDomainDepot::TAttributeDescriptions &descs)
{ 
  TFileExampleIteratorData fei(stem);

  vector<string> varNames;
  while(!feof(fei.file) && (readTabAtom(fei, varNames, true, csv)==-1));
  if (varNames.empty())
    ::raiseError("unexpected end of file '%s' while searching for attribute names", fei.filename.c_str());

  headerLines = 1;
  classPos = -1;
  basketPos = -1;
  attributeTypes = mlnew TIntList(varNames.size(), -1);
  TIntList::iterator attributeType(attributeTypes->begin());
  vector<string>::const_iterator ni(varNames.begin()), ne(varNames.end());
  int ind = 0;
  
  for(; ni != ne; ni++, ind++, attributeType++) {
    /* Parses the header line
       - sets *ni to a real name (without prefix)
       - sets varType to TValue::varType or -1 if the type is not specified and -2 if it's a basket
       - sets classPos/basketPos to the current position, if the attribute is class/basket attribute
         (and reports an error if there is more than one such attribute)
       - to attributeTypes, appends -1 for ordinary atributes, 1 for metas and 0 for ignored or baskets*/
    int varType = -1; // varType, or -1 for unnown, -2 for basket

    const char *cptr = (*ni).c_str();
    if (*cptr && (cptr[1]=='#') || (cptr[2] == '#')) {
      if (*cptr == 'm') {
        *attributeType = 1;
        cptr++;
      }
      else if (*cptr == 'i') {
        *attributeType = 0;
        cptr++;
      }
      else if (*cptr == 'c') {
        if (classPos>-1)
          ::raiseError("more than one attribute marked as class");
        else
          classPos = ind;
        cptr++;
      }
      
      // we may have encountered a m, i or c, so cptr points to the second character,
      // or it can still point to the first 
      if (*cptr == 'D') {
        varType = TValue::INTVAR;
        cptr++;
      }
      else if (*cptr == 'C') {
        varType = TValue::FLOATVAR;
        cptr++;
      }
      else if (*cptr == 'S') {
        varType = STRINGVAR;
        cptr++;
      }
      else if (*cptr == 'B') {
        varType = -2;
        if ((*attributeType != -1) || (classPos == ind))
          ::raiseError("flag 'B' is incompatible with 'i', 'm' and 'c'");
        *attributeType = 0;
        if (basketPos > -1)
          ::raiseError("more than one basket attribute");
        else
          basketPos = ind;
        cptr++;
      }
     
      if (*cptr != '#')     
        ::raiseError("unrecognized flags in attribute name '%s'", cptr);
      cptr++;
    }

    descs.push_back(TDomainDepot::TAttributeDescription(cptr, varType));
  }
}



void TTabDelimExampleGenerator::readTabHeader(const string &stem, TDomainDepot::TAttributeDescriptions &descs)
{
  classPos = -1;
  basketPos = -1;
  headerLines = 3;

  TFileExampleIteratorData fei(stem);
  
  vector<string> varNames, varTypes, varFlags;
  
  while(!feof(fei.file) && (readTabAtom(fei, varNames, true, csv) == -1));
  if (varNames.empty())
    ::raiseError("empty file");

  while(!feof(fei.file) && (readTabAtom(fei, varTypes, false, csv) == -1));
  if (varTypes.empty())
    ::raiseError("cannot read types of attributes");

  while(!feof(fei.file) && (readTabAtom(fei, varFlags, true, csv, true) == -1));

  if (varNames.size() != varTypes.size())
    ::raiseError("mismatching number of attributes and their types.");
  if (varNames.size() < varFlags.size())
    ::raiseError("too many flags (third line too long)");
  while (varFlags.size() < varNames.size())
    varFlags.push_back("");

  attributeTypes = mlnew TIntList(varNames.size(), -1);

  vector<string>::iterator vni(varNames.begin()), vne(varNames.end());
  vector<string>::iterator ti(varTypes.begin());
  vector<string>::iterator fi(varFlags.begin()), fe(varFlags.end());
  TIntList::iterator attributeType(attributeTypes->begin());
  int ind = 0;
  
  for(; vni!=vne; fi++, vni++, ti++, attributeType++, ind++) {
  
    descs.push_back(TDomainDepot::TAttributeDescription(*vni, 0));
    TDomainDepot::TAttributeDescription &desc = descs.back();

    bool ordered = false;

    if (fi!=fe) {
      TProgArguments args("dc: ordered", *fi, false, true);

      if (args.direct.size()) {
      
        if (args.direct.size()>1)
          ::raiseError("invalid flags for attribute '%s'", (*vni).c_str());
          
        string direct = args.direct.front();
        if ((direct=="s") || (direct=="skip") || (direct=="i") || (direct=="ignore"))
          *attributeType = 0;

        else if ((direct=="c") || (direct=="class")) {
          if (classPos != -1)
            ::raiseError("multiple attributes are specified as class attribute ('%s' and '%s')", (*vni).c_str(), (*vni).c_str());
          classPos = ind;
        }
        
        else if ((direct=="m") || (direct=="meta"))
          *attributeType = 1;
      }

      ITERATE(TMultiStringParameters, mi, args.options)
        if ((*mi).first == "dc")
          raiseWarning("argument -dc is not supported any more");

      ordered = args.exists("ordered");

      desc.userFlags = args.unrecognized;
    }

    if (!strcmp((*ti).c_str(), "basket")) {
      if (basketPos > -1)
        ::raiseError("multiple basket attributes are defined");
      if (ordered || (classPos == ind) || (*attributeType != -1))
        ::raiseError("'basket' flag is incompatible with other flags");
      basketPos = ind;
      *attributeType = 0;
    }

    if (!*attributeType)
      continue;

    if (!(*ti).length())
      ::raiseError("type for attribute '%s' is missing", (*vni).c_str());

    const TIdentifierDeclaration *tid = typeIdentifiers;
    for(; tid->identifier; tid++)
      if (!(tid->matchRoot ? strncmp(tid->identifier, (*ti).c_str(), tid->matchRoot)
                           : strcmp(tid->identifier, (*ti).c_str()))) {
        desc.varType = tid->varType;
        desc.typeDeclaration = *ti;
        break;
      }
      
    if (!tid->identifier) {
      desc.varType = TValue::INTVAR;

      string vals;
      ITERATE(string, ci, *ti) {
        if (*ci==' ') {
          if (vals.length())
            desc.addValue(vals);
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
      }

      if (vals.length())
        desc.addValue(vals);
    }
  }
}


bool atomsEmpty(const vector<string> &atoms)
{ const_ITERATE(vector<string>, ai, atoms)
    if ((*ai).length())
      return false;
  return true;
}


int trimAtomsList(vector<string> &atoms)
{
  if (!atoms.size())
    return 0;

  vector<string>::iterator ei(atoms.end()-1), bi(atoms.begin());
  for(; !(*ei).length() && ei!=bi; ei--);
  if (!(*ei).length())
    atoms.clear();
  else
    atoms.erase(++ei, atoms.end());
  return atoms.size();
}

/*  Reads a list of atoms from a line of tab or comma delimited file. Atom consists of any characters
    except \n, \r and \t (and ',' if csv=true). Multiple spaces are replaced by a single space. Atoms
    are separated by \t or ',' if csv=true. Lines end with \n or \r. Lines which begin with | are ignored.
   
    Returns number of atoms, -1 for comment line and -2 for EOF
    */
int readTabAtom(TFileExampleIteratorData &fei, vector<string> &atoms, bool escapeSpaces, bool csv, bool allowEmpty)
{
  atoms.clear();

  if (!fei.file)
    raiseErrorWho("TabDelimExampleGenerator", "file not opened");

  if (feof(fei.file))
    return -2;

  fei.line++;

  char c, c2;
  int col = 0;
  string atom;
  for(;;) {
    c = fgetc(fei.file);

    if (c==(char)EOF)
      break;
    if (!col && (c=='|')) { //ignore comment
      for (c=fgetc(fei.file); (c!='\r') && (c!='\n') && (c!=(char)EOF); c=fgetc(fei.file));
      return -1;
    }

    col++;

    switch(c) {
      case '\r':
      case '\n':
        c2 = fgetc(fei.file);
        if ((c2!='\r') && (c2!='\n') || (c2 == c))
          ungetc(c2, fei.file);
        if (atom.length() || atoms.size())
          atoms.push_back(trim(atom));  // end of line
        if (allowEmpty || atoms.size())
          return trimAtomsList(atoms);
        break;

      case '\t':
        atoms.push_back(trim(atom));
        atom.clear();
        break;

      case ',':
        if (csv) {
          atoms.push_back(trim(atom));
          atom.clear();
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

  return trimAtomsList(atoms);
}




// ********* Output ********* //


#define PUTDELIM { if (ho) putc(delim, file); else ho = true; }

void tabDelim_writeExample(FILE *file, const TExample &ex, char delim)
{ 
}


inline const char *checkCtrl(const char *c) {
  for(const char *cc = c; *cc; cc++)
    if ((const unsigned char)(*cc) < 32)
      raiseErrorWho("write", "string '%s' cannot be written to a file since it contains invalid characters", c);
  return c;
}

void tabDelim_writeExamples(FILE *file, PExampleGenerator rg, char delim, const char *DK, const char *DC)
{ 
  const TDomain domain = rg->domain.getReference();
  TVarList::const_iterator vb(domain.variables->begin()), vi, ve(domain.variables->end());

  PEITERATE(ex, rg) {
    vi = vb;
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
        fprintf(file, checkCtrl(st.c_str()));
      }
    }

    TMetaVector::const_iterator mb((*ex).domain->metas.begin()), mi, me((*ex).domain->metas.end());

    for(mi = mb; mi != me; mi++) {
      if (!(*mi).optional) {
        PUTDELIM;
        if (DK && ((*ri).valueType == valueDK))
          fprintf(file, DK);
        else if (DC && ((*ri).valueType == valueDC))
          fprintf(file, DC);
        else {
          (*mi).variable->val2filestr((*ex)[(*mi).id], st, *ex);
          fprintf(file, "%s", checkCtrl(st.c_str()));
        }
      }
    }
    
    bool first = true;
    for(mi = mb; mi != me; mi++) {
      if ((*mi).optional) {
        const TVariable &var = (*mi).variable.getReference();
        if ((var.varType == TValue::FLOATVAR) && (*ex).hasMeta((*mi).id)) {
          const TValue &mval = (*ex).getMeta((*mi).id);
          if (!mval.isSpecial()) {
            if (first) {
              PUTDELIM;
              first = false;
            }
            else
              fprintf(file, " ");

            if (mval.floatV == 1.0)
              fprintf(file, checkCtrl(var.name.c_str()));
            else {
              var.val2filestr(mval, st, *ex);
              fprintf(file, "%s=%s", checkCtrl(var.name.c_str()), checkCtrl(st.c_str()));
            }
          }
        }
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
      fprintf(file, checkCtrl(escSpaces(sval).c_str()));
      while(enumv->nextValue(val)) {
        enumv->val2str(val, sval);
        fprintf(file, " %s", checkCtrl(escSpaces(sval).c_str()));
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
      fprintf(file, "python:%s", checkCtrl(PyString_AsString(pyclassname)));
      Py_DECREF(pyclassname);
    }
  }  
  else
    raiseErrorWho("tabDelim_writeDomain", "tabDelim format supports only discrete, continuous and string variables");
}


void tabDelim_printAttributes(FILE *file, PVariable var, bool needsSpace) {
  TPyOrange *bvar = (TPyOrange *)(var.counter);
  PyObject *attrdict = bvar->orange_dict ? PyDict_GetItemString(bvar->orange_dict, "attributes") : NULL;
  if (attrdict) {
    PyObject *key, *value;
    Py_ssize_t pos = 0;
    while (PyDict_Next(attrdict, &pos, &key, &value)) {
      if (PyString_Check(key))
        Py_INCREF(key);
      else
        key = PyObject_Repr(key);
      if (PyString_Check(value))
        Py_INCREF(value);
      else
        value = PyObject_Repr(value);
      fprintf(file, (pos>1) || needsSpace ? " %s=%s" : "%s=%s", PyString_AsString(key), PyString_AsString(value));
      Py_DECREF(value);
      Py_DECREF(key);
    }
  }
}

void tabDelim_writeDomainWithoutDetection(FILE *file, PDomain dom, char delim, bool listDiscreteValues)
{ 
  TVarList::const_iterator vi, vb(dom->variables->begin()), ve(dom->variables->end());
  TMetaVector::const_iterator mi, mb(dom->metas.begin()), me(dom->metas.end());

  bool ho = false;
  bool hasOptionalFloats = false;

  // First line: attribute names
  for(vi = vb; vi!=ve; vi++) {
    PUTDELIM;
    fprintf(file, "%s", checkCtrl((*vi)->name.c_str()));
  }
  for(mi = mb; mi!=me; mi++) {
    if (mi->optional) {
      if ((*mi).variable->varType == TValue::FLOATVAR)
        hasOptionalFloats = true;
    }
    else {
      PUTDELIM;
      fprintf(file, "%s", checkCtrl((*mi).variable->name.c_str()));
    }
  }

  if (hasOptionalFloats) {
    PUTDELIM;
    fprintf(file, "__basket_foo");
  }

  fprintf(file, "\n");

  
  // Second line: types
  ho = false;
  for(vi = vb; vi!=ve; vi++) {
    PUTDELIM;
    printVarType(file, *vi, listDiscreteValues);
  }
  for(mi = mb; mi!=me; mi++) {
    if (mi->optional)
      continue;
    PUTDELIM;
    printVarType(file, (*mi).variable, listDiscreteValues);
  }

  if (hasOptionalFloats) {
    PUTDELIM;
    fprintf(file, "basket");
  }

  fprintf(file, "\n");


  // Third line: "meta" and "-ordered"
  ho = false;
  for(vb = vi = dom->attributes->begin(), ve = dom->attributes->end(); vi!=ve; vi++) {
    PUTDELIM;
    bool isOrdered = ((*vi)->varType == TValue::INTVAR) && (*vi)->ordered;
    if (isOrdered)
      fprintf(file, "-ordered");
    tabDelim_printAttributes(file, *vi, isOrdered);
  }
  if (dom->classVar) {
    PUTDELIM;
    fprintf(file, "class");
    tabDelim_printAttributes(file, dom->classVar, true);
  }
  for(mi = mb; mi!=me; mi++) {
    if (mi->optional)
      continue;
    PUTDELIM;
    fprintf(file, "meta");
    if (((*mi).variable->varType == TValue::INTVAR) && (*mi).variable->ordered)
      fprintf(file, " -ordered");
    tabDelim_printAttributes(file, (*mi).variable, true);
 }

 if (hasOptionalFloats)
   PUTDELIM;

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
    fprintf(file, "%s%s", (tabDelim_checkNeedsD(*vi) ? "D#" : ""), checkCtrl((*vi)->name.c_str()));
  }
  
  if (dom->classVar) {
    PUTDELIM;
    fprintf(file, "%s%s", (tabDelim_checkNeedsD(dom->classVar) ? "cD#" : "c#"), checkCtrl(dom->classVar->name.c_str()));
  }


  bool hasOptionalFloats = false;

  const_ITERATE(TMetaVector, mi, dom->metas) {
    if (mi->optional) {
      if ((*mi).variable->varType == TValue::FLOATVAR)
        hasOptionalFloats = true;
    }
    else {
      PUTDELIM;
      fprintf(file, "%s%s", (tabDelim_checkNeedsD((*mi).variable) ? "mD#" : "m#"), checkCtrl((*mi).variable->name.c_str()));
    }
  }

  if (hasOptionalFloats) {
    PUTDELIM;
    fprintf(file, "B#__basket_foo");
  }

  fprintf(file, "\n");
}


void tabDelim_writeDomain(FILE *file, PDomain dom, bool autodetect, char delim, bool listDiscreteValues)
{ if (autodetect)
    tabDelim_writeDomainWithDetection(file, dom, delim);
  else 
    tabDelim_writeDomainWithoutDetection(file, dom, delim, listDiscreteValues);
}
