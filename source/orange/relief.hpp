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


#ifndef __RELIEF_HPP
#define __RELIEF_HPP

#include "measures.hpp"

WRAPPER(ExamplesDistance_Relief);
WRAPPER(Domain);


class ORANGE_API TMeasureAttribute_relief : public TMeasureAttribute {
public:
    __REGISTER_CLASS

    float k; //P number of neighbours
    float m; //P number of reference examples

    TMeasureAttribute_relief(int ak=5, int am=100);
    virtual float operator()(int attrNo, PExampleGenerator, PDistribution apriorClass=PDistribution(), int weightID=0);

    void reset();

    vector<float> measures;
    PExampleGenerator prevGenerator; //C
    PDomain prevDomain; //C
    long prevDomainVersion;
    int prevExamples, prevWeight;
};
    
#endif
