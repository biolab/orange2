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

#ifndef __SOM_HPP
#define __SOM_HPP

#include "orangeom_globals.hpp"
#include "orange_api.hpp"
#include "orvector.hpp"
#include "learn.hpp"
#include "classify.hpp"
#include "transval.hpp"
#include "examplegen.hpp"
#include "values.hpp"
#include "root.hpp"

#ifdef _MSC_VER
#define NO_PIPED_COMMANDS
#define popen(a, b) NULL
#define pclose(a, b) NULL
#endif

#ifdef _MSC_VER
extern "C"{
#include "som/lvq_pak.h"
#include "som/som_rout.h"
#include "som/datafile.h"
}
#else
#include "som/lvq_pak.h"
#include "som/som_rout.h"
#include "som/datafile.h"
#endif

WRAPPER(ExampleTable)
WRAPPER(Classifier)
WRAPPER(ProbabilityEstimatorConstructor)
VWRAPPER(FloatList)
VWRAPPER(IntList)

#include "majority.hpp"

class ORANGEOM_API TSOMNode : public TOrange{
public:
   __REGISTER_CLASS
   int x;  //P x dimension
   int y;  //P y dimension
   PFloatList  vector; //PR codebook vector
   PExampleTable examples; //PR holds the examples associated with this node
    
   PDomainContinuizer domainContinuizer;    //P domain continuizer used to transform the domain
   PDomain transformedDomain;  //P transformed domain

   PExample referenceExample;	//P reference example
   
   PClassifier classifier; //P
    
   float getDistance(const TExample &example);
};

OMWRAPPER(SOMNode)

#define TSOMNodeList TOrangeVector<PSOMNode>
OMVWRAPPER(SOMNodeList)

class ORANGEOM_API TSOMLearner : public TLearner{
public:
   __REGISTER_CLASS
   
   static const int RectangularTopology;
   static const int HexagonalTopology;
   static const int BubbleNeighborhood;
   static const int GaussianNeighborhood;
   static const int LinearFunction;
   static const int InverseFunction;
   
   int xDim;    //P xDim
   int yDim;    //P yDim
   int steps;   //P number of steps
   int topology;//P topology
   int neighborhood; //P neighborhood
   int alphaType;	//P alphaType
   int randomSeed;	//P random seed for codebook initialization

   PIntList iterations; //P list of iterations for each step
   PIntList radius; //P neighbourghood radius list
   PFloatList alpha;    //P initial alpha values list
   
   PDomainContinuizer domainContinuizer;    //P domain continuizer used to transform the domain
   PDomain transformedDomain;  //P transformed domain
   
   TSOMLearner();
   PClassifier operator() (PExampleGenerator examples, const int &a=0);
};

OMWRAPPER(SOMLearner)

class ORANGEOM_API TSOMClassifier : public TClassifier{
public:
    __REGISTER_CLASS
    
    int xDim;       //PR xDim
    int yDim;       //PR yDim
    int topology;   //PR topology
    
    float trainingError;    //PR 
    
    struct entries *som_pak_data;
    struct entries *som_pak_codes;
	struct teach_params params;
    
    PDomainContinuizer domainContinuizer;   //P domain continuizer used to transform the domain
    PDomain transformedDomain;  //P transformed domain

	PExampleTable examples;	//P examples
    
    PSOMNodeList nodes; //P list of SOMNodes
    
    PClassifier classifier;
	
	TSOMClassifier(): TClassifier(true){};
	TSOMClassifier(PVariable v, bool cp=true): TClassifier(v, cp){};
	TSOMClassifier(const TSOMClassifier &old): TClassifier(old){};

	~TSOMClassifier();

    //virtual TValue operator ()(const TExample &);
    virtual PDistribution classDistribution(const TExample &);
    //virtual void predictionAndDistribution(const TExample &, TValue &, PDistribution &);
	
	float getError(PExampleGenerator examples);
    
    PSOMNode getWinner(const TExample &example);
};

//OMWRAPPER(SOMClassifier)

class ORANGEOM_API TSOMMap : public TSOMClassifier{
public:
    __REGISTER_CLASS
       
    virtual TValue operator ()(const TExample &){
        raiseError("classles domain");
		return TValue();
    }
    virtual PDistribution classDistribution(const TExample &){
        raiseError("classles domain");
		return PDistribution();
    }
    virtual void predictionAndDistribution(const TExample &, TValue &, PDistribution &){
        raiseError("classles domain");
    }
};

OMWRAPPER(SOMMap)

#endif //__SOM_HPP