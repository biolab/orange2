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

#include "stladdon.hpp"
#include <map>

#include "domain.hpp"
#include "examplegen.hpp"

#include "basket.ppp"

map<string, TMetaDescriptor> TBasketExampleGenerator::itemCache;

TBasketExampleGenerator::TBasketExampleGenerator(const string &datafile, PDomain sd, bool dcs, bool ds)
: TFileExampleGenerator(datafile, mlnew TDomain()),
  dontStore(ds),
  dontCheckStored(dcs),
  sourceDomain(sd)
{}


void TBasketExampleGenerator::clearCache()
{ 
  itemCache.clear();
}


void TBasketExampleGenerator::addItem(TExample &example, const string &atom, const float &quantity)
{
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

        // store to global cache, if allowed and if sth with that name is not already there
        if (!dontStore) {
          map<string, TMetaDescriptor>::const_iterator gitem(itemCache.find(atom));
          if (gitem == itemCache.end())
            itemCache[atom] = *md;
        }

        domain->metas.push_back(*md);
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
      domain->metas.push_back(TMetaDescriptor(id, mlnew TFloatVariable(atom)));

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

bool TBasketExampleGenerator::readExample(TFileExampleIteratorData &fei, TExample &example)
{
  if (!fei.file)
    raiseError("file not opened");
  if (feof(fei.file))
    return false;

  static const char atomends[] = {EOF, '|', '\n', '\r', ',', 0};

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
        addItem(example, atom);
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
      if (example.meta.empty() && (c != EOF))
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

