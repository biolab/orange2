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


#ifndef __COST_HPP
#define __COST_HPP

#include "vars.hpp"
#include "distvars.hpp"

class ORANGE_API TCostMatrix : public TOrange {
public:
  __REGISTER_CLASS

  PVariable classVar; //P attribute to which the matrix applies
  int dimension; //PR dimension (should equal classVar.noOfValues())

  float *costs;

  TCostMatrix(const int &dimension, const float &inside = 1.0);
  TCostMatrix(PVariable, const float &inside = 1.0);
  	  
  inline const float &cost(const int &predicted, const int &correct) const
  { 
    if ((predicted >= dimension) || (correct >= dimension))
      raiseError("value out of range");
    return costs[predicted*dimension + correct];
  }
  
  inline float &cost(const int &predicted, const int &correct)
  { 
    if ((predicted >= dimension) || (correct >= dimension))
      raiseError("value out of range");
    return costs[predicted*dimension + correct];
  }
  

protected:
  void init(const float &inside);
};

WRAPPER(CostMatrix);

#endif

