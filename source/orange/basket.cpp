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

// to include Python.h before STL defines a template set (doesn't work with VC 6.0)
#include "garbage.hpp" 

#include "stladdon.hpp"
#include "strings.hpp"
#include <map>

#include "domain.hpp"
#include "examplegen.hpp"

#include "basket.ppp"

map<string, TMetaDescriptor> TBasketFeeder::itemCache;


TBasketFeeder::TBasketFeeder(PDomain sd, bool dcs, bool ds)
: dontStore(ds),
  dontCheckStored(dcs),
  sourceDomain(sd)
{}


void TBasketFeeder::clearCache()
{ 
  itemCache.clear();
}


void TBasketFeeder::addItem(TExample &example, const string &atom2, const int &lineno)
{
  string atom;
  float quantity;

  string::const_iterator bi = atom2.begin();
  string::size_type ei = atom2.find('=');
  if (ei == string::npos) {
    atom = trim(atom2);
    quantity = 1.0;
  }
  else {
    atom = trim(string(bi, bi+ei));
    string trimmed = trim(string(bi+ei+1, atom2.end()));
    char *err;
    quantity = strtod(trimmed.c_str(), &err);
    if (*err)
      raiseError("invalid number after '%s=' in line %i", atom.c_str(), lineno);
  }

  int id = ILLEGAL_INT;

  // Have we seen this item in this file already?
  map<string, int>::const_iterator item(localStore.find(atom));
  if (item != localStore.end())
    id = (*item).second;

  else {
    // Check the sourceDomain, if there is one
    if (sourceDomain) {
      const TMetaDescriptor *md = sourceDomain->metas[atom];

      if (md) {
        id = md->id;

        TMetaDescriptor nmd(id, md->variable, 1); // optional meta!

        // store to global cache, if allowed and if sth with that name is not already there
        if (!dontStore) {
          map<string, TMetaDescriptor>::const_iterator gitem(itemCache.find(atom));
          if (gitem == itemCache.end())
            itemCache[atom] = nmd;
        }

        domain->metas.push_back(nmd);
      }
    }

    if ((id == ILLEGAL_INT) && !dontCheckStored) { // !sourceDomain or not found there
      map<string, TMetaDescriptor>::const_iterator gitem(itemCache.find(atom));
      if (gitem != itemCache.end()) {
        id = (*gitem).second.id;
        domain->metas.push_back((*gitem).second);
      }
    }

    // Not found anywhere - need to create a new attribute
    if (id == ILLEGAL_INT) {
      id = getMetaID();
      // Variable is created solely to hold the name
      TFloatVariable *var = (TFloatVariable *)TVariable::getExisting(atom, TValue::FLOATVAR);
      if (!var)
        var = mlnew TFloatVariable(atom);
      domain->metas.push_back(TMetaDescriptor(id, var, true));

      // store to global cache, if allowed and if not already there
      // Why dontCheckStored? If we have already searched there, we don't have to confirm again it doesn't exist
      if (!dontStore && (!dontCheckStored || (itemCache.find(atom) == itemCache.end())))
          itemCache[atom] = domain->metas.back();
    }

    localStore[atom] = id;
  }

  if (example.hasMeta(id))
    example[id].floatV += quantity;
  else
    example.setMeta(id, TValue(quantity));
}





TBasketExampleGenerator::TBasketExampleGenerator(const string &datafile, PDomain sd, const int createNewOn, vector<int> &status, vector<pair<int, int> > &metaStatus)
: TFileExampleGenerator(datafile, mlnew TDomain()),
  basketFeeder(new TBasketFeeder(sd, false, false))
{
  basketFeeder->domain = domain;
}


bool TBasketExampleGenerator::readExample(TFileExampleIteratorData &fei, TExample &example)
{
  if (!fei.file)
    raiseError("file not opened");
  if (feof(fei.file))
    return false;

  static const char atomends[] = {(char)EOF, '|', '\n', '\r', ',', 0};

  example.meta.clear();

  fei.line++;

  string atom;
  char c = fgetc(fei.file);

  for(;;) {
    const char *ae = atomends;
    while (*ae && (*ae!=c))
      ae++;

    if (*ae) {
      if (atom.length()) {
        basketFeeder->addItem(example, atom, fei.line);
        atom = string();
      }
 
      if (c == ',') {
        do
          c = fgetc(fei.file);
        while (c==' ');
        continue;
      }

      if (c == '|')
        do
          c = fgetc(fei.file);
        while ((c!='\r') && (c!='\n'));

      // we don't exit the loop if there's more to read and we haven't read anything yet
      if (example.meta.empty() && (c != (char)EOF))
        c = fgetc(fei.file);
      else
        break;
    }

    else { // not an end-of-atom character
      if ((c>=' ') || (c<0))
        atom += c;
      c = fgetc(fei.file);
    }
  }

  return !example.meta.empty();
}


void basket_writeExamples(FILE *fle, PExampleGenerator gen, set<int> &missing)
{
  const TDomain &domain = gen->domain.getReference();

  PEITERATE(ei, gen) {
    bool comma = false;
    ITERATE(TMetaValues, mi, (*ei).meta) {
      if ((*mi).second.varType != TValue::FLOATVAR)
        raiseError(".basket files cannot store non-continuous attributes");

      if ((*mi).second.isSpecial() || ((*mi).second.floatV == 0.0))
        continue;
        
      PVariable metaVar = domain.getMetaVar((*mi).first, false);
      if (metaVar) {
        if ((*mi).second.floatV == 1.0)
          if (comma)
            fprintf(fle, ", %s", metaVar->name.c_str());
          else {
            fprintf(fle, metaVar->name.c_str());
            comma = true;
          }
        else
          if (comma)
            fprintf(fle, ", %s=%5.3f", metaVar->name.c_str(), (*mi).second.floatV);
          else {
            fprintf(fle, "%5s=%5.3f", metaVar->name.c_str(), (*mi).second.floatV);
            comma = true;
          }
      }
      else
        missing.insert((*mi).first);
    }

    if (comma)
      fprintf(fle, "\n");
  }
}
