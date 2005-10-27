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

    Authors: Janez Demsar, Blaz Zupan, 1996--2005
    Contact: ales.erjavec324@email.si
*/

#ifndef __MDS_HPP
#define __MDS_HPP

#include "orangeom_globals.hpp"
#include "orvector.hpp"
#include "symmatrix.hpp"
#include "progress.hpp"
#include <iostream>

#define MMAX(a,b) (((a)>(b))? a :b)

#if _MSC_VER!=0 && _MSC_VER <1300
OMVWRAPPER(FloatList)
#else 
VWRAPPER(FloatList)
#endif

#define TFloatListList TOrangeVector<PFloatList>
OMVWRAPPER(FloatListList)

class ORANGEOM_API TStressFunc: public TOrange{
public:
    __REGISTER_ABSTRACT_CLASS
    virtual float operator() (float current, float correct, float weight=1.0)=0;
};

class ORANGEOM_API TKruskalStress : public TStressFunc{
public:
   __REGISTER_CLASS
   virtual float operator() (float current, float correct, float weight=1.0){
       float d=current-correct;
       return d*d*weight;
   }
};

class ORANGEOM_API TSammonStress : public TStressFunc{
public:
    __REGISTER_CLASS
    virtual float operator() (float current, float correct, float weight=1.0){
		//std::cout<<current<<", "<<correct<<std::endl;
       float d=current-correct;
       return d*d*weight/MMAX(1e-6,current);
   }
};

class ORANGEOM_API TSgnSammonStress : public TStressFunc{
public:
    __REGISTER_CLASS
    virtual float operator() (float current, float correct, float weight=1.0){
        float d=current-correct;
        return d*weight/MMAX(1e-6,current);
    }
};

class ORANGEOM_API TSgnRelStress : public TStressFunc{
public:
    __REGISTER_CLASS
    virtual float operator() (float current, float correct, float weight=1.0){
        float d=current-correct;
        return d*weight/MMAX(1e-6,correct);
    }
};

OMWRAPPER(StressFunc)
OMWRAPPER(KruskalStress)
OMWRAPPER(SammonStress)
OMWRAPPER(SgnSammonStress)        
OMWRAPPER(SgnRelStress)

/*this is copy paste form callback.cpp */
inline PyObject *callCallback(PyObject *self, PyObject *args)
{ PyObject *result;
  
  if (PyObject_HasAttrString(self, "__callback")) {
      PyObject *callback = PyObject_GetAttrString(self, "__callback");
      result = PyObject_CallObject(callback, args);
      Py_DECREF(callback);
  }
  else 
      result = PyObject_CallObject(self, args);
  /*
  if (!result)
      throw pyexception();
  */
  return result;
}


class ORANGEOM_API TStressFunc_Python : public TStressFunc{
public:
    __REGISTER_CLASS
    virtual float operator() (float current, float correct, float weight=1.0){
        PyObject *args=Py_BuildValue("fff", &current, &correct, &weight);
        PyObject *result=callCallback((PyObject*) myWrapper, args);
        Py_DECREF(args);
        double f=PyFloat_AsDouble(result);
        Py_DECREF(result);
        return (float)f;
    }    
};

OMWRAPPER(StressFunc_Python)

class ORANGEOM_API TMDS : public TOrange{
public:
    __REGISTER_CLASS
    PSymMatrix distances; //P SymMatrix that holds the original real distances
    PSymMatrix projectedDistances; //P SymMatrix that holds the projected distances
    PSymMatrix stress;    //P SymMatrix that holds the pointwise stress values
    PFloatListList points;    //P holds the current projected point configuration
    PProgressCallback progressCallback; //P progressCallback function
    bool freshD;    //PR
    float avgStress;   //PR
    int dim;    //PR
    int n;      //PR

    TMDS(PSymMatrix matrix, int dim);
    TMDS(){};
    void SMACOFstep();  
    void getDistances();
    float getStress(PStressFunc fun);
	void optimize(int numIter, PStressFunc fun, float eps=1e-3);

};

OMWRAPPER(MDS)


#endif  //__MDS
