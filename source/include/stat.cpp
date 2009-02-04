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


#include <functional>

#include "stat.hpp"

using namespace std;

int nUniquePoints(const vector<double> &points)
{ 
  int up = 0;
  for(vector<double>::const_iterator vb(points.begin()), vi(vb), ve(points.end()); vi!=ve; vi++)
    if ((vi == vb) || (*(vi-1) != *vi))
      up++;

  return up;
}


void samplingFactor(const vector<double> &points, int nPoints, vector<double> &result)
{ 
  result.clear();

  for (vector<double>::const_iterator pi(points.begin()), pe(points.end());;) {
    const double &ax = *pi;
    result.push_back(ax);
    
    if (++pi==pe)
      break;
    if (*pi == ax)
      continue;

    if (*pi != ax) {
      // We could write this faster, but we don't want to run into problems with rounding
      double div = (*pi - ax) / nPoints;
      for (int i=1; i < nPoints; i++)
        result.push_back(ax + i*div);
    }
  }
}


void samplingFactor(const map<double, double> &points, int nPoints, vector<double> &result)
{ 
  result.clear();

  for (map<double, double>::const_iterator pi(points.begin()), pe(points.end());;) {
    const double &ax = (*pi).first;
    result.push_back(ax);
    
    if (++pi==pe)
      break;

    // We could write this faster, but we don't want to run into problems with rounding floats
    const double &div = ((*pi).first - ax) / nPoints;
    for (int i=1; i < nPoints; i++)
      result.push_back(ax + i*div);
  }
}


void samplingMinimal(const map<double, double> &points, int nPoints, vector<double> &result)
{ 
  result.clear();

  if (nPoints<=points.size()) {
    for (map<double, double>::const_iterator pi(points.begin()), pe(points.end()); pi != pe; pi++)
      result.push_back((*pi).first);
  }

  else
    samplingFixed(points, nPoints, result);
}


void samplingMinimal(const vector<double> &points, int nPoints, vector<double> &result)
{ 
  int nUnique = nUniquePoints(points);

  if (nPoints<=nUnique)
    result = points;
  else
    samplingFixed(points, nPoints, result);
}


void samplingFixed(const vector<double> &points, int nPoints, vector<double> &result)
{ 
  int nUnique = nUniquePoints(points);

  result.clear();
  const double &ineach = double(nPoints - nUnique) / double(nUnique-1);
  double inthis = 0.0;
  
  for (vector<double>::const_iterator pi(points.begin()), pe(points.end());;) {
    double ax = *pi;
    result.push_back(ax);

    if (++pi==pe)
      break;

    if (*pi != ax) {
      inthis += ineach;
      if (inthis >= 1.0) {
        const double &dif = (*pi - ax) / (int(floor(inthis))+1);
        while (inthis > 0.5) {
          result.push_back(ax += dif);
          inthis -= 1.0;
        }
      }
    }
  }
}


void samplingFixed(const map<double, double> &points, int nPoints, vector<double> &result)
{ 
  result.clear();

  const double &ineach = float(nPoints - points.size()) / float(points.size()-1);
  double inthis = 0.0;
  
  for (map<double, double>::const_iterator pi(points.begin()), pe(points.end());;) {
    double ax = (*pi).first;
    result.push_back(ax);

    if (++pi==pe)
      break;

    inthis += ineach;
    if (inthis >= 0.5) {
      const double &dif = ((*pi).first - ax) / (int(floor(inthis))+1);
      while (inthis > 0.5) {
        result.push_back(ax += dif);
        inthis -= 1.0;
      }
    }
  }
}


void samplingUniform(const vector<double> &points, int nPoints, vector<double> &result)
{
  result.clear();

  const double &fi = points.front();
  const double &rg = (points.back()-fi) / (nPoints-1);
  for (int i = 0; i<nPoints; i++)
    result.push_back(fi + i*rg);
}


void samplingUniform(const map<double, double> &points, int nPoints, vector<double> &result)
{
  result.clear();

  const double &fi = (*points.begin()).first;
  map<double, double>::const_iterator pe(points.end());
  pe--;
  const double &rg = ((*pe).first-fi) / (nPoints-1);
  for (int i = 0; i<nPoints; i++)
    result.push_back(fi + i*rg);
}



bool comp1st(const pair<double, double> &x1, const pair<double, double> &x2)
{ return x1.first < x2.first; }


void vector2weighted(const vector<pair<double, double> > &points, vector<TXYW> &weighted)
{
  if (points.empty())
    throw StatException("lwr/loess: empty sample");

  weighted.clear();

  vector<pair<double, double> > myPoints = points;
  sort(myPoints.begin(), myPoints.end(), comp1st);

  vector<pair<double, double> >::const_iterator mpi(myPoints.begin()), mpe(myPoints.end());
  weighted.push_back(TXYW((*mpi).first, (*mpi).second));
  while(++mpi != mpe) {
    TXYW &last = weighted.back();
    if ((*mpi).first == last.x) {
      last.y += (*mpi).second;
      last.w += 1.0;
    }
    else {
      if (last.w > 1e-6)
        last.y /= last.w;
      weighted.push_back(TXYW((*mpi).first, (*mpi).second));
    }
  }

  TXYW &last = weighted.back();
  if (last.w > 1e-6)
    last.y /= last.w;
}

void loess(const vector<double> &refpoints, const vector<TXYW> &points, const float &windowProp, vector<pair<double, double> > &result)
{ 
  result.clear();
  
  typedef vector<TXYW>::const_iterator iterator;

  iterator lowedge = points.begin();
  iterator highedge = points.end();
  iterator from;
  iterator to;

  double nPoints = 0;
  for(from = points.begin(); from != highedge; nPoints += (*(from++)).w);
  double needpoints = windowProp <= 1.0 ? nPoints * windowProp : windowProp;

  bool stopWindow = needpoints >= nPoints;
  if (stopWindow) {
    from = lowedge;
    to = highedge;
  }
  else
    for(from = to = lowedge; (to != highedge) && (needpoints>0); needpoints -= (*(to++)).w);

  for(vector<double>::const_iterator rpi(refpoints.begin()), rpe(refpoints.end()); rpi != rpe; rpi++) {

    const double &refx = *rpi;

    /* Adjust the window */

    if (!stopWindow) {
      // adjust the top end so that the window includes the reference point
      //   (note that the last point included is to-1, so this one must be >= refx)
      for(; (to != highedge) && (refx > (*(to-1)).x); needpoints -= (*(to++)).w);
      const int diffto = (to == highedge ? (to-1)->x : to->x) - refx;
      // adjust the bottom end as high as it goes but so that the window still covers at least needpoints points
      for(; (from != to) && (diffto < refx - (*from).x) && (needpoints + (*from).w < 0); needpoints += (*(from++)).w);

      while ((to!=highedge) && ((*to).x - refx < refx - (*from).x)) {
        // 'to' is not at the high edge and to's point is closer that from's, so include it
        needpoints -= (*(to++)).w;
        // adjust the bottom end as high as it goes but so that the window still covers at least needpoints points
        for(; (from != to) && (needpoints + (*from).w < 0); needpoints += (*(from++)).w);
      }

      stopWindow = (to==highedge);
    }

 
    /* Determine the window half-width */
    double h = abs(refx - (*from).x);
    const double h2 = abs((*(to-1)).x - refx);
    if (h2 > h)
      h = h2;

    h *= 1.1;


    /* Iterate through the window */

    double Sx = 0.0, Sy = 0.0, Sxx = 0.0, Syy = 0.0, Sxy = 0.0, Sw = 0.0, Swx = 0.0, Swxx = 0.0;
    double n = 0.0;

    for (iterator ii = from; ii != to; ii++) {
      const double &x = (*ii).x;
      const double &y = (*ii).y;

      // compute the weight based on the distance
      double w = abs(refx - x) / h;
      w = 1 - w*w*w;
      w = w*w*w;
      // and multiply it by the point's given weight
      w *= (*ii).w;

      n   += w;
      Sx  += w * x;
      Sxx += w * x * x;
      Sy  += w * y;
      Syy += w * y * y;
      Sxy += w * x * y;

      Sw  += w * w;
      Swx += w * w * x;
      Swxx += w * w * x * x;
    }

    if (n==0) {
      result.push_back(pair<double, double>(Sy, 0));
      continue;
    }

    const double mu_x = Sx / n;
    const double mu_y = Sy / n;
    const double sigma_x2 = (Sxx - mu_x * Sx) / n;
    if (sigma_x2 < 1e-20) {
      result.push_back(pair<double, double>(Sy, 0));
      continue;
    }

    const double sigma_y2 = (Syy - mu_y * Sy) / n;
    const double sigma_xy = (Sxy - Sx * Sy / n) / n;
    const double sigma_y_x = sigma_y2 - sigma_xy * sigma_xy / sigma_x2;

    const double dist_x = refx - mu_x;
    const double y = mu_y + sigma_xy / sigma_x2 * dist_x;

    double var_y = sigma_y_x / n / n * (Sw + dist_x * dist_x / sigma_x2 / sigma_x2 * (Swxx + mu_x * mu_x * Sw - 2 * mu_x * Swx));
    if ((var_y < 0) && (var_y > -1e-6))
      var_y = 0;
    else
      var_y = sqrt(var_y);

    result.push_back(pair<double, double>(y, var_y));
  }
}


void loess(const vector<double> &refpoints, const vector<pair<double, double> > &points, const float &windowProp, vector<pair<double, double> > &result)
{
  vector<TXYW> weighted;
  vector2weighted(points, weighted);
  loess(refpoints, weighted, windowProp, result);
}


void loess(const vector<double> &refpoints, const map<double, double> &points, const float &windowProp, vector<pair<double, double> > &result)
{
  vector<TXYW> opoints;
  for(map<double, double>::const_iterator pi(points.begin()), pe(points.end()); pi != pe; pi++)
    opoints.push_back(TXYW((*pi).first, (*pi).second));

  loess(refpoints, opoints, windowProp, result);
}


void lwr(const vector<double> &refpoints, const vector<TXYW> &points, const float &smoothFactor, vector<pair<double, double> > &result)
{ 
  result.clear();
  
  typedef vector<TXYW>::const_iterator iterator;

  float tot_w = 0.0;
  { 
    const_ITERATE(vector<TXYW>, pi, points)
    tot_w += (*pi).w;
  }
  const float p25 = 0.25 * tot_w;
  const float p75 = 0.75 * tot_w;
  float x25, x75;
  tot_w = 0;
  { 
    vector<TXYW>::const_iterator pi(points.begin()), pe(points.end());
    for(; (pi!=pe) && (tot_w<p25); tot_w += (*pi).w, pi++);
    const float &x1 = (*(pi-1)).x;
    x25 = x1 + ((*pi).x-x1) * (p25 - tot_w + (*pi).w) / (*pi).w;

    if (tot_w >= p75)
      throw StatException("not enough data to compute 25th and 75th percentile");

    for(; (pi!=pe) && (tot_w<p75); tot_w += (*pi).w, pi++);
    const float &x2 = (*(pi-1)).x;
    x75 = x2 + ((*pi).x-x2) * (p75 - tot_w + (*pi).w) / (*pi).w;
  }

  const float sigma = smoothFactor * (x75-x25);

  const_ITERATE(vector<double>, ri, refpoints) {
    const double &refx = *ri;

    double Sx = 0.0, Sy = 0.0, Sxx = 0.0, Syy = 0.0, Sxy = 0.0, Sw = 0.0, Swx = 0.0, Swxx = 0.0;
    double n = 0.0;

    for (vector<TXYW>::const_iterator ii(points.begin()), ie(points.end()); ii != ie; ii++) {
      const double &x = (*ii).x;
      const double &y = (*ii).y;

      // compute the weight based on the distance and the point's given weight
      const double dx = x - *ri;
      double w = (*ii).w * exp(- dx*dx / (sigma*sigma));

      n   += w;
      Sx  += w * x;
      Sxx += w * x * x;
      Sy  += w * y;
      Syy += w * y * y;
      Sxy += w * x * y;

      Sw  += w * w;
      Swx += w * w * x;
      Swxx += w * w * x * x;
    }

    if (n==0) {
      result.push_back(pair<double, double>(Sy, 0));
      continue;
    }

    const double mu_x = Sx / n;
    const double mu_y = Sy / n;
    const double sigma_x2 = (Sxx - mu_x * Sx) / n;
    if (sigma_x2 < 1e-20) {
      result.push_back(pair<double, double>(Sy, 0));
      continue;
    }

    const double sigma_y2 = (Syy - mu_y * Sy) / n;
    const double sigma_xy = (Sxy - Sx * Sy / n) / n;
    const double sigma_y_x = sigma_y2 - sigma_xy * sigma_xy / sigma_x2;

    const double dist_x = refx - mu_x;
    const double y = mu_y + sigma_xy / sigma_x2 * dist_x;

    double var_y = sigma_y_x / n / n * (Sw + dist_x * dist_x / sigma_x2 / sigma_x2 * (Swxx + mu_x * mu_x * Sw - 2 * mu_x * Swx));
    if ((var_y < 0) && (var_y > -1e-6))
      var_y = 0;
    else
      var_y = sqrt(var_y);

    result.push_back(pair<double, double>(y, var_y));
  }
}



void lwr(const vector<double> &refpoints, const vector<pair<double, double> > &points, const float &smoothFactor, vector<pair<double, double> > &result)
{
  vector<TXYW> weighted;
  vector2weighted(points, weighted);
  lwr(refpoints, weighted, smoothFactor, result);
}


void lwr(const vector<double> &refpoints, const map<double, double> &points, const float &smoothFactor, vector<pair<double, double> > &result)
{
  vector<TXYW> opoints;
  for(map<double, double>::const_iterator pi(points.begin()), pe(points.end()); pi != pe; pi++)
    opoints.push_back(TXYW((*pi).first, (*pi).second));

  lwr(refpoints, opoints, smoothFactor, result);
}

