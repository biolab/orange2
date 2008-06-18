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


#ifndef __STRINGVARS_HPP
#define __STRINGVARS_HPP

#include "root.hpp"
#include "values.hpp"
#include "vars.hpp"
#include <string>

#define STRINGVAR 6

class ORANGE_API TStringValue : public TSomeValue {
public:
  __REGISTER_CLASS

  string value;

  TStringValue(const string &aval);
  TStringValue(const TStringValue &other);

  virtual void val2str(string &) const;
  virtual void str2val(const string &);

  virtual int  compare(const TSomeValue &v) const;
  virtual bool compatible(const TSomeValue &v) const;
  virtual bool operator < (const TSomeValue &v) const;
  virtual bool operator ==(const TSomeValue &v) const;
  virtual bool operator !=(const TSomeValue &v) const;
  virtual bool operator > (const TSomeValue &v) const;
};



class ORANGE_API TStringVariable : public TVariable {
public:
  __REGISTER_CLASS

  TStringVariable();
  TStringVariable(const string &aname);

  virtual bool firstValue(TValue &val) const;
  virtual bool nextValue(TValue &val) const;
  virtual TValue randomValue(const int &rand);

  virtual int  noOfValues() const;

  virtual void str2val(const string &valname, TValue &valu);
  virtual void val2str(const TValue &valu, string &vname) const;

  virtual void val2filestr(const TValue &val, string &str, const TExample &) const;
  virtual void filestr2val(const string &valname, TValue &valu, TExample &);
};

#endif
