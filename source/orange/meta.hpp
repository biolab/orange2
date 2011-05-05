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


#ifndef __META_HPP
#define __META_HPP

#include <string>
#include <vector>
#include "values.hpp"
using namespace std;


WRAPPER(Variable)

extern long metaID;

ORANGE_API long getMetaID(PVariable var = PVariable());

#ifdef _MSC_VER
  #pragma warning (push)
  #pragma warning (disable: 4275 4251)
#endif

extern ORANGE_API TValue missingMetaValue;

// A vector of meta values with id's
class ORANGE_API TMetaValues : public vector<pair<long, TValue> > {
public:
  TValue &operator[](long id);
  const TValue &operator[](long id) const;
  bool exists(long id) const;
  void setValue(const long &id, const TValue &val);
  void removeValue(const long &id);
  void removeValueIfExists(const long &id);

  const TValue &getValueIfExists(long id) const;
  TValue &getValueIfExists(long id);
};

#ifdef _MSC_VER
  #pragma warning (pop)
#endif


class ORANGE_API TMetaDescriptor {
public:
  long   id;
  PVariable variable;
  int optional;

  TMetaDescriptor();
  TMetaDescriptor(const long &ai, const PVariable &avar, const int opt = 0);
  TMetaDescriptor(const TMetaDescriptor &);

  /* We don't need this, but need to provide it to be able to export the class to a DLL*/
  bool operator <(const TMetaDescriptor &other) const  { return id < other.id; }
  bool operator ==(const TMetaDescriptor &other) const  { return id == other.id; }
};

#ifdef _MSC_VER
  ORANGE_EXTERN template class ORANGE_API std::vector<TMetaDescriptor>;
#endif

class ORANGE_API TMetaVector : public vector<TMetaDescriptor> {
public:
  TMetaDescriptor *operator[](PVariable);
  TMetaDescriptor const *operator[](PVariable) const;
  TMetaDescriptor *operator[](const string &sna);
  TMetaDescriptor const *operator[](const string &sna) const;
  TMetaDescriptor *operator[](const long &ai);
  TMetaDescriptor const *operator[](const long &ai) const;
};

extern const char *_getweightwho, *_unknownweightexception, *_noncontinuousweightexception;

inline float _getweight(const TValue &val)
{ if (val.valueType)
    raiseErrorWho(_getweightwho, _unknownweightexception);
  if (val.varType!=TValue::FLOATVAR)
    raiseErrorWho(_getweightwho, _noncontinuousweightexception);
  return val.floatV;
}

// A macro to return a weight of an example or 1 if the weightID == 0
#define WEIGHT(ex) (weightID<0 ? _getweight((ex)[weightID]) : float(1.0))
#define WEIGHT2(ex,w) (w<0 ? _getweight((ex)[w]) : float(1.0))

#endif

