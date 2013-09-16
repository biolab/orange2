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

