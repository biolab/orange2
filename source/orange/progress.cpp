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


#include "progress.ppp"

bool TProgressCallback::operator()(float *&milestone, POrange o)
{ return operator()(*((++milestone)++), o); }


float *TProgressCallback::milestones(const int totalSteps, const int nMilestones)
{
  float *milestones = new float[2*(totalSteps+1)];
  float *mi = milestones;
  const float step = float(totalSteps)/nMilestones;
  for(int i = 0; i<=nMilestones; i++) {
    *mi++ = floor(0.5+i*step);
    *mi++ = float(i)/100.0;
  }
  *mi++ = -1;
  *mi++ = 1.0;

  return milestones;
}
