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
