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


#include "boolcnt.hpp"
#include "root.hpp"

VWRAPPER(VarList);
VWRAPPER(VarListList);

class TSubsetsGenerator : public TOrange {
public:
  __REGISTER_ABSTRACT_CLASS

  PVarList varList; //P a set of attributes from which subsets are generated

  virtual bool reset();
  virtual bool reset(const TVarList &vl);
  virtual bool nextSubset(TVarList &)=0;
};

WRAPPER(SubsetsGenerator);


class TSubsetsGenerator_constSize : public TSubsetsGenerator {
public:
  __REGISTER_CLASS

  int B; //P subset size

  TSubsetsGenerator_constSize(int B);

  virtual bool reset();
  virtual bool reset(const TVarList &vl);
  virtual bool nextSubset(TVarList &);

protected:
  bool moreToCome;
  TCounter counter;
};


class TSubsetsGenerator_minMaxSize : public TSubsetsGenerator {
public:
  __REGISTER_CLASS

  int min; //P minimal subset size
  int max; //P maximal subset size

  TSubsetsGenerator_minMaxSize(int amin, int amax); // can be -1 to ignore

  virtual bool reset();
  virtual bool reset(const TVarList &vl);
  virtual bool nextSubset(TVarList &);

protected:
  bool moreToCome;
  int B;
  TCounter counter;
};


class TSubsetsGenerator_constant : public TSubsetsGenerator {
public:
  __REGISTER_CLASS

  PVarList constant; //P a subset that is returned (once!)

  TSubsetsGenerator_constant();
  TSubsetsGenerator_constant(const TVarList &);

  virtual bool reset();
  virtual bool reset(const TVarList &);
  virtual bool nextSubset(TVarList &);

protected:
  bool moreToCome;
};


class TSubsetsGenerator_withRestrictions : public TSubsetsGenerator {
public:
  __REGISTER_CLASS

  PSubsetsGenerator subGenerator; //P subset generator
  PVarList required; //P set of required attributes
  PVarList forbidden; //P set of forbidden attributes
  PVarListList forbiddenSubSubsets; //P set of forbidden subsets (attributes that must not occur together)

  TSubsetsGenerator_withRestrictions(PSubsetsGenerator sub=PSubsetsGenerator());
  TSubsetsGenerator_withRestrictions(PSubsetsGenerator sub, const TVarList &areq, const TVarList &aforb);

  virtual bool reset();
  virtual bool reset(const TVarList &);
  virtual bool nextSubset(TVarList &);
};


