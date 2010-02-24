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


#ifndef __PYTHONVARS_HPP
#define __PYTHONVARS_HPP

#include "root.hpp"
#include "values.hpp"
#include "vars.hpp"

#include "c2py.hpp"

#define PYTHONVAR 7

class ORANGE_API TPythonValue : public TSomeValue {
public:
  __REGISTER_CLASS

  PyObject *value;

  TPythonValue();
  TPythonValue(const TPythonValue &other);
  TPythonValue(PyObject *value);

  TPythonValue &TPythonValue::operator =(const TPythonValue &other);

  ~TPythonValue();

  virtual int  compare(const TSomeValue &v) const;
  virtual bool compatible(const TSomeValue &v) const;
};



class ORANGE_API TPythonVariable : public TVariable {
public:
  __REGISTER_CLASS

  bool usePickle; //P tells whether to use pickle for saving to/loading from files
  bool useSomeValue; //P tells whether the Variable will operate on Value or SomeValue (default)

protected:
  TValue DC_somevalue;
  TValue DK_somevalue;

public:
  TPythonVariable();
  TPythonVariable(const string &aname);

  bool isOverloaded(char *method) const;

  virtual const TValue &DC() const;
  virtual const TValue &DK() const;
  virtual TValue specialValue(int) const;

  virtual void val2str(const TValue &val, string &str) const;
  virtual void str2val(const string &valname, TValue &valu);
  virtual void str2val_add(const string &valname, TValue &valu);

  virtual void val2filestr(const TValue &val, string &str, const TExample &) const;
  virtual void filestr2val(const string &valname, TValue &valu, TExample &);

  virtual bool   firstValue(TValue &val) const;
  virtual bool   nextValue(TValue &val) const;
  virtual TValue randomValue(const int &rand=-1);

  // Returns the number of different values, -1 if it cannot be done (for example, if variable is continuous)
  virtual int  noOfValues() const;

// steals a reference!
  TValue toValue(PyObject *pyvalue) const;

  TValue toNoneValue(const signed char &valueType) const;
  void toValue(PyObject *pyvalue, TValue &val) const;

  PyObject *toPyObject(const TValue &valu) const;
};


class ORANGE_API TPythonValueSpecial : public TOrange {
public:
  __REGISTER_CLASS

  int valueType; //P value type

  TPythonValueSpecial(const int &vt)
  : valueType(vt)
  {}
};

#endif
