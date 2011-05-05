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


#ifndef __EXAMPLES_HPP
#define __EXAMPLES_HPP

#include "meta.hpp"
#include "orvector.hpp"
WRAPPER(Domain)
WRAPPER(Example)

#define TExampleList TOrangeVector<PExample> 
VWRAPPER(ExampleList)

extern long exampleId;
ORANGE_API long getExampleId();

// A vector of attribute and class values
class ORANGE_API TExample : public TOrange {
public:
  __REGISTER_CLASS

  PDomain domain; //PR Example's domain
  TValue *values, *values_end;
  TMetaValues meta;
  string *name;
  int id; //P

  TExample();
  TExample(PDomain dom, bool initMetas = true);
  TExample(PDomain dom, const TExample &orig, bool copyMetas = true);
  TExample(const TExample &orig, bool copyMetas = true);
  TExample(PDomain dom, PExampleList);
  virtual ~TExample();

private:
  void insertVal(TValue &srcval, PVariable var, const long &metaID, vector<bool> &defined);

public:
  int traverse(visitproc visit, void *arg) const;
  int dropReferences();

  virtual TExample &operator =(const TExample &orig);

  typedef TValue *iterator;
  typedef const TValue *const_iterator;

  inline TValue *begin()
  { return values; }

  inline TValue *begin() const
  { return values; }

  inline TValue *end()
  { return values_end; }

  inline TValue *end() const
  { return values_end; }


  TValue &operator[] (const int &i)
  { return i>=0 ? values[i] : getMeta(i); }

  const TValue &operator[] (const int &i) const
  { return i>=0 ? values[i] : getMetaIfExists(i); }

  // this is duplicated from above, but can be used on non-const examples
  const TValue &getValue(const int &i)
  { return i>=0 ? values[i] : getMetaIfExists(i); }

  TValue &operator[] (PVariable &var);
  const TValue &operator[] (PVariable &var) const;
  TValue &operator[] (const string &name);
  const TValue &operator[] (const string &name) const;

  // this one is similar to the above operator, but calls computeValue if needed
  // (if [] returned it, the computed value would be a reference to a temporary)
  TValue getValue(PVariable &var) const;

  TValue &getClass()
  { return values_end[-1]; }

  TValue &getClass() const
  { return values_end[-1]; }

  void setClass(const TValue &val)
  { values_end[-1] = val; }


  const TValue &missingMeta(const int &i) const;

  // these two require that meta is defined
  const TValue &getMeta(const int &i) const
  { 
    return meta[i];
  }

  TValue &getMeta(const int &i)
  { 
    return meta[i];
  }

  // this one will set it to undefined if it is optional and undefined
/*  TValue &getMetaIfExists(const int &i)
  { 
    TValue &val = meta.getValueIfExists(i);
    return &val != &missingMetaValue ? val : missingMeta(i);
  }
*/

  const TValue &getMetaIfExists(const int &i) const
  { 
    const TValue &val = meta.getValueIfExists(i);
    return &val != &missingMetaValue ? val : missingMeta(i);
  }




  bool hasMeta(const int &i) const
  { return meta.exists(i); }

  void setMeta(const int &i, const TValue &val)
  { meta.setValue(i, val); }

  void removeMeta(const int &i)
  { meta.removeValue(i); }

  void removeMetaIfExists(const int &i)
  { meta.removeValueIfExists(i); }

  void removeMetas()
  { meta.clear(); }

  bool operator < (const TExample &) const;
  bool operator == (const TExample &) const;
  int  compare(const TExample &other, const bool ignoreClass = false) const;
  bool compatible(const TExample &other, const bool ignoreClass = false) const;

  void addToCRC(unsigned long &crc, const bool includeMetas = false) const;
  int sumValues(const bool includeMetas = false) const;
  
  bool hasMissing() const
  {
    for(TValue const *vi = values, *ve = values_end; vi!=ve; vi++)
      if (vi->isSpecial())
        return true;
    return false;
  }
  
  bool missingClass() const
  { return values_end[-1].isSpecial(); }
};


WRAPPER(Example)

#endif

