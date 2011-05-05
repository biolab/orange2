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


#ifndef __PROGRESS_HPP
#define __PROGRESS_HPP

#include "root.hpp"

class ORANGE_API TProgressCallback : public TOrange {
public:
  __REGISTER_ABSTRACT_CLASS

  virtual bool operator()(const float &, POrange = POrange()) = 0;

  bool operator()(float *&milestone, POrange = POrange());
  static float *milestones(const int totalSteps, const int nMilestones = 100);
};

WRAPPER(ProgressCallback)

#endif
