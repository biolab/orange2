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


#ifndef __CONTINUIZE_HPP
#define __CONTINUIZE_HPP

#include "transval.hpp"

class TDiscrete2Continuous : public TTransformValue {
public:
  __REGISTER_CLASS

  int value; //P the observed values

  TDiscrete2Continuous(const int =-1);
  virtual void transform(TValue &);
};


class TNormalizeContinuous : public TTransformValue {
public:
  __REGISTER_CLASS

  float average; //P the average value
  float span; //P the value span

  TNormalizeContinuous(const float =0.0, const float =0.0);
  virtual void transform(TValue &);
};

PDomain domain4SVM(PExampleGenerator);
PDomain domain4SVM(PDomain);

#endif
