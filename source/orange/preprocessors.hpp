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


#ifndef __PREPROCESSORS_HPP
#define __PREPROCESSORS_HPP

#include "stladdon.hpp"

#include "vars.hpp"
#include "domain.hpp"
#include "examples.hpp"
#include "examplegen.hpp"

#include "preprocess.hpp"
#include "filter.hpp"
#include "discretize.hpp"
#include "cost.hpp"

WRAPPER(Filter);

DECLARE(drop)
  PStringList vnames; //P tells which examples to remove
};

DECLARE(take)
  PStringList vnames; //P tells which examples to select
};

DECLARE(ignore)
  PStringList vnames; //P (+attributes) tells which attributes to remove
};

DECLARE(select)
  PStringList vnames; //P (+attributes) tells which attributes to select
};


DECLARE(remove_duplicates) };
DECLARE(skip_missing) };
DECLARE(only_missing) };
DECLARE(skip_missing_classes) };
DECLARE(only_missing_classes) };

typedef map<string, float> TNameProb;
typedef WRAPPEDNML(NameProb) PNameProb;

DECLARE(noise)
  float defaultNoise; //P default noise level
  int randseed; //P random generator seed
  PNameProb probabilities;

private:
  void addNoise(TExampleTable *);
};


DECLARE(gaussian_noise)
  float defaultDeviation; //P default deviation
  int randseed; //P random generator seed
  PNameProb deviations; 
};


DECLARE(missing)
  float defaultMissing; //P default proportion of missing values
  int randseed; //P random generator seed
  PNameProb probabilities; 

private:
  void addMissing(TExampleTable *);
};


DECLARE(class_noise)
  float classNoise; //P class noise level
  int randseed; //P random generator seed

private:
  void addNoise(PExampleGenerator);
};


DECLARE(class_gaussian_noise)
  float classDeviation; //P class deviation
  int randseed; //P random generator seed
};


DECLARE(class_missing)
  float classMissing; //P proportion of missing class values
  int randseed; //P random generator seed

private:
  void addMissing(PExampleGenerator);
};


DECLARE(cost_weight)
  bool equalize; //P reweight examples to equalize class proportions
  PFloatList classWeights; //P weights of examples of particular classes
};

DECLARE(censor_weight)
  string outcomeVar; //P outcome variable name
  string eventValue; //P event (fail) value
  string timeVar; //P time variable
  string method; //P weighting method
  float maxTime; //P maximal time
  string weightName; //P name of new weight
};


WRAPPER(Discretization);

DECLARE(discretize)
  int noOfIntervals; //P number of intervals
  PStringList vnames; //P (>attributes) names of attributes to discretize (all if empty)
  bool notClass; //P do not discretize the class attribute
  PDiscretization method; //P discretization method

  PDomain discretizedDomain(PExampleGenerator, long &);
};


DECLARE(move_to_table) };

DECLARE(filter)
  PFilter filter; //P filter
};

#endif
