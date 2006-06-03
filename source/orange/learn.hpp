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


#ifndef __LEARN_HPP
#define __LEARN_HPP


WRAPPER(Variable)
WRAPPER(Distribution)
WRAPPER(DomainDistributions)
WRAPPER(DomainContingency)
WRAPPER(ExampleGenerator)

/*  A base for classes which can learn to classify examples after presented an appropriate learning set.
    Learning is invoked by calling 'learn' method and can be forgoten by calling 'forget'. */
class ORANGE_API TLearner : public TOrange {
public:
  __REGISTER_CLASS

  enum {NeedsNothing, NeedsClassDistribution, NeedsDomainDistribution, NeedsDomainContingency, NeedsExampleGenerator};
  int needs; //PR the kind of data that learner needs

  TLearner(const int & = NeedsExampleGenerator);
  
  virtual PClassifier operator()(PVariable);
  virtual PClassifier operator()(PDistribution);
  virtual PClassifier operator()(PDomainDistributions);
  virtual PClassifier operator()(PDomainContingency);
  virtual PClassifier operator()(PExampleGenerator, const int &weight = 0);

  virtual PClassifier smartLearn(PExampleGenerator, const int &weight,
	                               PDomainContingency = PDomainContingency(),
                                 PDomainDistributions = PDomainDistributions(),
                                 PDistribution = PDistribution());
};


class ORANGE_API TLearnerFD : public TLearner {
public:
  __REGISTER_CLASS

  PDomain domain; //P domain

  TLearnerFD();
  TLearnerFD(PDomain);
};

WRAPPER(Learner)

#endif

