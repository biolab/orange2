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


#ifndef __EXAMPLES_HPP
#define __EXAMPLES_HPP

#include "meta.hpp"
#include "orvector.hpp"
WRAPPER(Domain)

// A vector of attribute and class values
class TExample : public TOrange {
public:
  __REGISTER_CLASS

  PDomain domain; //PR Example's domain
  TValue *values, *values_end;
  TMetaValues meta;

  TExample();
  TExample(PDomain dom);
  TExample(PDomain dom, const TExample &orig);
  TExample(const TExample &orig);
  virtual ~TExample();

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


  TValue &operator[] (int i)
  { return values[i]; }

  const TValue &operator[] (int i) const
  { return values[i]; }

  TValue &operator[] (PVariable &var);
  const TValue &operator[] (PVariable &var) const;
  TValue &operator[] (const string &name);
  const TValue &operator[] (const string &name) const;

  TValue getClass() const
  { return values_end[-1]; }

  void setClass(const TValue &val)
  { values_end[-1] = val; }


  bool operator < (const TExample &) const;
  bool operator == (const TExample &) const;
  int  compare(const TExample &other) const;
  bool compatible(const TExample &other) const;
};


WRAPPER(Example)

#endif

