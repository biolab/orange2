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


// to include Python.h before STL defines a template set (doesn't work with VC >6.0)
#include "garbage.hpp" 

#include "domain.hpp"
#include "distvars.hpp"
#include "contingency.hpp"
#include "examplegen.hpp"

#include "learn.ppp"


TLearner::TLearner(const int &aneeds)
: needs(aneeds)
{}


PClassifier TLearner::operator()(PVariable)
{ if (needs==NeedsNothing)
    raiseError("invalid value of 'needs'");
  else
    raiseError("no examples"); 
  return PClassifier();
}


PClassifier TLearner::operator()(PDistribution dist)
{ switch (needs) {
    case NeedsNothing:
      return operator()(dist->variable);
    case NeedsClassDistribution:
      raiseError("invalid value of 'needs'");
    default:
      raiseError("cannot learn from class distribution only"); 
  };
  return PClassifier();
}


PClassifier TLearner::operator()(PDomainDistributions ddist)
{ switch (needs) {
    case NeedsNothing:
      return operator()(ddist->back()->variable);
    case NeedsClassDistribution:
      return operator()(ddist->back());
    case NeedsDomainDistribution:
      raiseError("invalid value of 'needs'");
    default:
      raiseError("cannot learn from distributions only");
  }
  return PClassifier();
}


PClassifier TLearner::operator()(PDomainContingency dcont)
{ switch (needs) {
    case NeedsNothing:
      return operator()(dcont->classes->variable);
    case NeedsClassDistribution:
      return operator()(dcont->classes);
    case NeedsDomainDistribution:
      return operator()(dcont->getDistributions());
    case NeedsDomainContingency:
      raiseError("invalid value of 'needs'");
    default:
      raiseError("cannot learn from contingencies only");
  }
  return PClassifier();
}



PClassifier TLearner::operator()(PExampleGenerator gen, const int &weight)
{ 
  if (!gen || !gen->domain)
    raiseError("TLearner: no examples or invalid example generator");
  if (!gen->domain->classVar)
    raiseError("class-less domain");

  switch (needs) {
    case NeedsNothing:
      return operator()(gen->domain->classVar);
    case NeedsClassDistribution:
      return operator()(getClassDistribution(gen, weight));
    case NeedsDomainDistribution:
      return operator()(PDomainDistributions(mlnew TDomainDistributions(gen, weight)));
    case NeedsDomainContingency:
      return operator()(PDomainContingency(mlnew TDomainContingency(gen, weight)));
    default:
      raiseError("invalid value of 'needs'");
  }
  return PClassifier();
}



PClassifier TLearner::smartLearn(PExampleGenerator gen, const int &weight,
                                 PDomainContingency dcont,
                                 PDomainDistributions ddist,
                                 PDistribution dist)
{ 
  switch (needs) {

    case NeedsNothing:
      if (!gen || !gen->domain)
        raiseError("TLearner: no examples or invalid example generator");
      if (!gen->domain->classVar)
        raiseError("class-less domain");
      return operator()(gen->domain->classVar);

    case NeedsClassDistribution:
      if (dist)
        return operator()(dist);
      else if (ddist)
        return operator()(ddist->back());
      else if (dcont)
        return operator()(dcont->classes);
      else {
        dist = getClassDistribution(gen, weight);
        return operator()(dist);
      }

    case NeedsDomainDistribution:
      if (ddist)
        return operator()(ddist);
      else if (dcont)
        return operator()(dcont->getDistributions());
      else {
        ddist = PDomainDistributions(mlnew TDomainDistributions(gen, weight));
        return operator()(ddist);
      }

    case NeedsDomainContingency:
      if (!dcont)
        dcont = PDomainContingency(mlnew TDomainContingency(gen, weight));
      return operator()(dcont);

    case NeedsExampleGenerator:
      return operator()(gen, weight);

    default:
      raiseError("invalid value of 'needs'");
  }

  return PClassifier();
}

  
TLearnerFD::TLearnerFD()
: TLearner()
{}


TLearnerFD::TLearnerFD(PDomain ad)
: domain(ad)
{}
