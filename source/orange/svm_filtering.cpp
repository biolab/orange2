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


#include "values.hpp"
#include "vars.hpp"
#include "examples.hpp"
#include "examplegen.hpp"

#include "classfromvar.hpp"
#include "basstat.hpp"

#include "svm_filtering.ppp"


TDiscrete2Continuous::TDiscrete2Continuous(const int aval)
 : value(aval)
 {}

void TDiscrete2Continuous::transform(TValue &val)
{ if (val.varType!=TValue::INTVAR)
    raiseError("invalid value type (non-int)");

  val = TValue(float(val.isSpecial() ? 0.0
                                     : (val.intV==value ? 1.0 : -1.0)));
}



TNormalizeContinuous::TNormalizeContinuous(const float av, const float sp)
 : average(av), span(sp)
 { if (span==0.0)
     span=1.0;
 }


void TNormalizeContinuous::transform(TValue &val)
{ if (val.varType!=TValue::FLOATVAR)
    raiseError("invalid value type (non-float)");

  val = TValue(float(val.isSpecial() ? 0.0 : (2*(val.floatV-average)/span)));
}



void addEnumVariable(PVariable var, TVarList &vars)
{ TEnumVariable *evar=var.AS(TEnumVariable);
  if (!evar)
    raiseError("cannot convert non-discrete values if only domain is given");

  if (evar->values->size()>=2) {
    int baseValue=evar->baseValue;
    if ((baseValue<0) && (evar->values->size()==2))
      baseValue=1;

    for(int val=0, mval=evar->values->size(); val<mval; val++)
      if (val!=baseValue) {
        PVariable newvar=mlnew TFloatVariable(evar->name+"="+evar->values->at(val));
        TClassifierFromVar *cfv=mlnew TClassifierFromVar(newvar, var);
        cfv->transformer=mlnew TDiscrete2Continuous(val);
        newvar->getValueFrom=cfv;
        vars.push_back(newvar);
      }
  }
}

void addFloatVariable(PVariable var, const float &avg, const float &span, TVarList &vars)
{ if (var->varType!=TValue::FLOATVAR)
    raiseError("cannot convert non-continuous values if only domain is given");

  PVariable newvar=mlnew TFloatVariable(var->name+"_N");
  TClassifierFromVar *cfv=mlnew TClassifierFromVar(newvar, var);
  cfv->transformer=mlnew TNormalizeContinuous(avg, span);
  newvar->getValueFrom=cfv;
  vars.push_back(newvar);
}

PDomain domain4SVM(PDomain dom)
{ TEnumVariable *eclass=dom->classVar.AS(TEnumVariable);
  if (!eclass || (eclass->values->size()>2))
    raiseErrorWho("domain4SVM", "binary class expected");

  TVarList newvars;
  PITERATE(TVarList, vi, dom->variables)
    addEnumVariable(*vi, newvars);

  return mlnew TDomain(newvars);
}


PDomain domain4SVM(PExampleGenerator egen)
{ PDomain dom=egen->domain; 
  TEnumVariable *eclass=dom->classVar.AS(TEnumVariable);
  if (!eclass || (eclass->values->size()>2))
    raiseErrorWho("domain4SVM" , "binary class expected");

  TVarList newvars;
  TDomainBasicAttrStat dombas(egen);
  TDomainBasicAttrStat::iterator di(dombas.begin());

  PITERATE(TVarList, vi, dom->variables) {
    if ((*vi)->varType==TValue::INTVAR)
      addEnumVariable(*vi, newvars);
    else
      addFloatVariable(*vi, (*di)->avg, (*di)->max-(*di)->min, newvars);
    di++;
  }

  return mlnew TDomain(newvars);
}
