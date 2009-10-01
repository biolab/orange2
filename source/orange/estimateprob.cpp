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


#include "vars.hpp"
#include "contingency.hpp"
#include "examplegen.hpp"

#include "estimateprob.ppp"
#include "stat.hpp"
#include "random.hpp"


DEFINE_TOrangeVector_classDescription(PProbabilityEstimator, "TProbabilityEstimatorList", true, ORANGE_API)
DEFINE_TOrangeVector_classDescription(PConditionalProbabilityEstimator, "TConditionalProbabilityEstimatorList", true, ORANGE_API)

TProbabilityEstimator::TProbabilityEstimator(const bool &disc, const bool &cont)
: supportsDiscrete(disc),
  supportsContinuous(cont)
{}


PDistribution TProbabilityEstimator::operator()() const
{ return PDistribution(); }



TConditionalProbabilityEstimator::TConditionalProbabilityEstimator(const bool &disc, const bool &cont)
: supportsDiscrete(disc),
  supportsContinuous(cont)
{}


PContingency TConditionalProbabilityEstimator::operator()() const
{ return PContingency(); }



TProbabilityEstimator_FromDistribution::TProbabilityEstimator_FromDistribution(PDistribution af)
: TProbabilityEstimator(true, true),
  probabilities(af)
{ /* We try to check if we support discrete/continuous attributes;
     if we can't, we'll promise everything and blame the user for it
     (he should have given us the distributions) */
  if (probabilities) {
    if (probabilities.is_derived_from(TDiscDistribution))
      supportsContinuous = false;
    else if (probabilities.is_derived_from(TContDistribution))
      supportsDiscrete = false;
  }
}


float TProbabilityEstimator_FromDistribution::operator()(const TValue &classVal) const
{ checkProperty(probabilities);
  if (classVal.isSpecial())
    raiseError("undefined attribute value");
  
  /* This is a harmless shortcut to make things run faster in the most usual case */
  if (classVal.varType == TValue::INTVAR) {
    const TDiscDistribution *ddist = probabilities.AS(TDiscDistribution);
    if (ddist)
      return (*ddist)[classVal.intV];
    // else, let probabilities->p do something or (more probably) report an error
  }
  
  return probabilities->p(classVal);
}


PDistribution TProbabilityEstimator_FromDistribution::operator()() const
{ return CLONE(TDistribution, probabilities);
}



PProbabilityEstimator TProbabilityEstimatorConstructor_relative::operator()(PDistribution frequencies, PDistribution, PExampleGenerator, const long &, const int &) const
{ TProbabilityEstimator_FromDistribution *pefd = mlnew TProbabilityEstimator_FromDistribution(CLONE(TDistribution, frequencies));
  PProbabilityEstimator estimator = pefd;
  pefd->probabilities->normalize();
  return estimator;
}



TProbabilityEstimatorConstructor_Laplace::TProbabilityEstimatorConstructor_Laplace(const float &al, const bool &an)
: l(al),
  renormalize(an)
{}


PProbabilityEstimator TProbabilityEstimatorConstructor_Laplace::operator()(PDistribution frequencies, PDistribution, PExampleGenerator, const long &, const int &) const
{ TProbabilityEstimator_FromDistribution *pefd = mlnew TProbabilityEstimator_FromDistribution(CLONE(TDistribution, frequencies));
  PProbabilityEstimator estimator = pefd;
  
  TDiscDistribution *ddist = pefd->probabilities.AS(TDiscDistribution);
  if (ddist) {
    const float &abs = ddist->abs;
    const float &cases = ddist->cases;
    const float div = cases + l * ddist->noOfElements();
    int i = 0;
    if (div) {
      if ((cases == abs) || !renormalize || (abs<1e-20))
        PITERATE(TDiscDistribution, di, ddist)
          ddist->setint(i++, (*di + l) / div);
      else
        PITERATE(TDiscDistribution, di, ddist)
          ddist->setint(i++, (*di / abs * cases + l) / div);
    }
    else
      pefd->probabilities->normalize();
  }
  else
    pefd->probabilities->normalize();
  
  return estimator;
}



TProbabilityEstimatorConstructor_m::TProbabilityEstimatorConstructor_m(const float &am, const bool &an)
: m(am),
  renormalize(an)
{}


PProbabilityEstimator TProbabilityEstimatorConstructor_m::operator()(PDistribution frequencies, PDistribution apriori, PExampleGenerator, const long &weightID, const int &) const
{ TProbabilityEstimator_FromDistribution *pefd = mlnew TProbabilityEstimator_FromDistribution(CLONE(TDistribution, frequencies));
  PProbabilityEstimator estimator = pefd;
  
  TDiscDistribution *ddist = pefd->probabilities.AS(TDiscDistribution);  
  if (ddist && (ddist->cases > 1e-20) && apriori) {
    TDiscDistribution *dapriori = apriori.AS(TDiscDistribution);
    if (!dapriori || (dapriori->abs < 1e-20))
      raiseError("invalid apriori distribution");
    
    float mabs = m/dapriori->abs;
    const float &abs = ddist->abs;
    const float &cases = ddist->cases;
    const float div = cases + m;
    if ((abs==cases) || !renormalize) {
      int i = 0;
      for(TDiscDistribution::iterator di(ddist->begin()), de(ddist->end()), ai(dapriori->begin());
          di != de;
          di++, ai++, i++)
         ddist->setint(i, (*di+*ai*mabs)/div);
    }
    else {
      int i = 0;
      for(TDiscDistribution::iterator di(ddist->begin()), de(ddist->end()), ai(dapriori->begin());
          di != de;
          di++, ai++, i++)
         ddist->setint(i, (*di / abs * cases + *ai*mabs)/div);
    }
  }
  else
    pefd->probabilities->normalize();
    
  return estimator;
}



TProbabilityEstimatorConstructor_kernel::TProbabilityEstimatorConstructor_kernel(const float &minImp, const float &smoo, const int &nP)
: minImpact(minImp),
  smoothing(smoo),
  nPoints(nP)
{}


PProbabilityEstimator TProbabilityEstimatorConstructor_kernel::operator()(PDistribution frequencies, PDistribution apriori, PExampleGenerator, const long &weightID, const int &) const
{ TContDistribution *cdist = frequencies.AS(TContDistribution);
  if (!cdist)
    raiseError("continuous distribution expected");
  if (!cdist->size())
    raiseError("empty distribution");
  if ((minImpact<0.0) || (minImpact>1.0))
    raiseError("'minImpact' should be between 0.0 and 1.0 (not %5.3f)", minImpact);

  vector<float> points;
  distributePoints(cdist->distribution, nPoints, points);

  TContDistribution *curve = mlnew TContDistribution(frequencies->variable);
  PDistribution wcurve = curve;

  /* Bandwidth suggested by Chad Shaw. Also found in http://www.stat.lsa.umich.edu/~kshedden/Courses/Stat606/Notes/interpolate.pdf */
  const float h = smoothing * sqrt(cdist->error()) * exp(- 1.0/5.0 * log(cdist->abs)); // 1.144
  const float hsqrt2pi = h * 2.5066282746310002;
  float t;

  if (minImpact>0) {
    t = -2 * log(minImpact*hsqrt2pi); // 2.5066... == sqrt(2*pi)
    if (t<=0) {
      // minImpact too high, but that's user's problem... 
      ITERATE(vector<float>, pi, points)
        curve->setfloat(*pi, 0.0);
        return wcurve;
    }
    else
      t = h * sqrt(t);
  }
      
      
  ITERATE(vector<float>, pi, points) {
    const float &x = *pi;
    TContDistribution::const_iterator from, to;

    if (minImpact>0) {
      from = cdist->lower_bound(x-t);
      to = cdist->lower_bound(x+t);
      if ((from==cdist->end()) || (to==cdist->begin()) || (from==to)) {
        curve->setfloat(x, 0.0);
        continue;
      }
    }
    else {
      from = cdist->begin();
      to = cdist->end();
    }

    float p = 0.0, n = 0.0;
    for(; from != to; from++) {
      n += (*from).second;
      p += (*from).second * exp( - 0.5 * sqr( (x - (*from).first)/h ) );
    }

    curve->setfloat(x, p/hsqrt2pi/(n*h)); // hsqrt2pi is from the inside (errf), n*h is for the sum average
  }


  return mlnew TProbabilityEstimator_FromDistribution(curve);
}



TProbabilityEstimatorConstructor_loess::TProbabilityEstimatorConstructor_loess(const float &windowProp, const int &ak)
: windowProportion(windowProp),
  nPoints(ak),
  distributionMethod(DISTRIBUTE_MAXIMAL)
{}



PProbabilityEstimator TProbabilityEstimatorConstructor_loess::operator()(PDistribution frequencies, PDistribution, PExampleGenerator, const long &weightID, const int &attrNo) const
{ TContDistribution *cdist = frequencies.AS(TContDistribution);
  if (!cdist)
    if (frequencies && frequencies->variable)
      raiseError("attribute '%s' is not continuous", frequencies->variable->name.c_str());
    else
      raiseError("continuous distribution expected");
  if (!cdist->size())
    raiseError("empty distribution");

  map<float, float> loesscurve;
  loess(cdist->distribution, nPoints, windowProportion, loesscurve, distributionMethod);
  return mlnew TProbabilityEstimator_FromDistribution(mlnew TContDistribution(loesscurve));
}



TConditionalProbabilityEstimator_FromDistribution::TConditionalProbabilityEstimator_FromDistribution(PContingency cont)
: TConditionalProbabilityEstimator(true, true),
  probabilities(cont)
{ if (probabilities) {
    supportsContinuous = (probabilities->varType == TValue::FLOATVAR);
    supportsDiscrete = (probabilities->varType == TValue::INTVAR);
  }
}


float TConditionalProbabilityEstimator_FromDistribution::operator()(const TValue &val, const TValue &condition) const
{ if (condition.varType == TValue::INTVAR)
    return probabilities->operator[](condition)->operator[](val);

  else if (condition.varType == TValue::FLOATVAR) {
    if (condition.isSpecial() || val.isSpecial())
      raiseError("undefined attribute value for condition");
    if (probabilities->varType != TValue::FLOATVAR)
      raiseError("invalid attribute type for condition");

    const TDistributionMap *dm = probabilities->continuous;
    const float &x = condition.floatV;
    TDistributionMap::const_iterator rb = dm->upper_bound(x);
    if (rb==dm->end())
      return 0.0;
    if ((*rb).first==x)
      return (*rb).second->operator[](val);
    if (rb==dm->begin())
      return 0.0;

    const float &x2 = (*rb).first, &y2 = (*rb).second->operator[](val);
    rb--;
    const float &x1 = (*rb).first, &y1 = (*rb).second->operator[](val);

    if (x1 == x2)
      return (y1+y2)/2;

    return y1 + (x - x1) * (y2 - y1) / (x2 - x1);
  }

  raiseError("invalid attribute type for condition");
  return 0.0;
}


PDistribution TConditionalProbabilityEstimator_FromDistribution::operator()(const TValue &condition) const
{ if (condition.varType == TValue::INTVAR)
    return probabilities->operator[](condition);

  else if (condition.varType == TValue::FLOATVAR) {
    if (condition.isSpecial())
      raiseError("undefined attribute value for condition");
    if (probabilities->varType != TValue::FLOATVAR)
      raiseError("invalid attribute value type for condition");

    const float &x = condition.floatV;
    const TDistributionMap *dm = probabilities->continuous;
    TDistributionMap::const_iterator rb = dm->upper_bound(x);
    if (rb==dm->end())
      rb = dm->begin();
    
    TDistribution *result = CLONE(TDistribution, (*rb).second);
    PDistribution wresult = result;

    if ((rb==dm->begin()) && ((*rb).first!=x)) {
      (*result) *= 0;
      return wresult;
    }

    const float &x2 = (*rb).first;
    rb--;
    const float &x1 = (*rb).first;
    const PDistribution &y1 = (*rb).second;

    if (x1 == x2) {
      *result += y1;
      *result *= 0.5;
      return wresult;
    }

    // The normal formula for this is in the function above
    *result -= y1;
    *result *= (x-x1)/(x2-x1);
    *result += y1;
    return wresult;
  }

  raiseError("invalid attribute value for condition");
  return PDistribution();
}


PContingency TConditionalProbabilityEstimator_FromDistribution::operator()() const
{ return CLONE(TContingency, probabilities); }

  

void TConditionalProbabilityEstimator_ByRows::checkCondition(const TValue &condition) const
{ checkProperty(estimatorList);
  if (!estimatorList->size())
    raiseError("empty 'estimatorList'");
  if (condition.isSpecial())
    raiseError("undefined attribute value for condition");
  if (condition.varType != TValue::INTVAR)
    raiseError("value for condition is not discrete");
  if (condition.intV >= estimatorList->size())
    raiseError("value for condition out of range");
}


float TConditionalProbabilityEstimator_ByRows::operator()(const TValue &val, const TValue &condition) const
{ checkCondition(condition);
  return estimatorList->operator[](condition.intV)->call(val);
}


PDistribution TConditionalProbabilityEstimator_ByRows::operator()(const TValue &condition) const
{ checkCondition(condition);
  return estimatorList->operator[](condition.intV)->call();
}


TConditionalProbabilityEstimatorConstructor_ByRows::TConditionalProbabilityEstimatorConstructor_ByRows(PProbabilityEstimatorConstructor pec)
: estimatorConstructor(pec)
{}

PConditionalProbabilityEstimator TConditionalProbabilityEstimatorConstructor_ByRows::operator()(PContingency frequencies, PDistribution apriori, PExampleGenerator gen, const long &weightID, const int &attrNo) const
{ if (!frequencies)
    frequencies = mlnew TContingencyAttrClass(gen, weightID, attrNo);
  if (frequencies->varType != TValue::INTVAR)
    if (frequencies->outerVariable)
      raiseError("attribute '%s' is not discrete", frequencies->outerVariable->name.c_str());
    else
      raiseError("discrete attribute for condition expected");

  /* We first try to construct a list of Distributions; if we suceed, we'll return an instance of
     TConditionProbabilityEstimator_FromDistribution. If we fail, we'll construct an instance of
     TConditionProbabilityEstimator_ByRows. */

  // This list stores conditional estimators for the case we fail
  PProbabilityEstimatorList cpel = mlnew TProbabilityEstimatorList();

  PContingency newcont = mlnew TContingencyAttrClass(frequencies->outerVariable, frequencies->innerVariable);
  TDistributionVector::const_iterator fi(frequencies->discrete->begin()), fe(frequencies->discrete->end());
  for (int i = 0; fi!=fe; fi++, i++) {
    PProbabilityEstimator est = estimatorConstructor->call(*fi, apriori, PExampleGenerator(), 0, attrNo);
    cpel->push_back(est);
    PDistribution dist = est->call();
    if (!dist)
      break;
    if (i >= newcont->discrete->size())
      newcont->discrete->push_back(dist);
    else
      newcont->discrete->operator[](i) = dist;
  }
  

  if (fi==fe)
    return mlnew TConditionalProbabilityEstimator_FromDistribution(newcont);

  /* We failed at constructing a matrix of probabilites. We'll just complete the list of estimators. */
  for (; fi!=fe; fi++)
    cpel->push_back(estimatorConstructor->call(*fi, apriori, gen, weightID));

  TConditionalProbabilityEstimator_ByRows *cbr = mlnew TConditionalProbabilityEstimator_ByRows();
  PConditionalProbabilityEstimator wcbr = cbr;
  cbr->estimatorList = cpel;
  return wcbr;
}



TConditionalProbabilityEstimatorConstructor_loess::TConditionalProbabilityEstimatorConstructor_loess(const float &windowProp, const int &ak)
: windowProportion(windowProp),
  nPoints(ak),
  distributionMethod(DISTRIBUTE_FIXED)
{}


PConditionalProbabilityEstimator TConditionalProbabilityEstimatorConstructor_loess::operator()(PContingency frequencies, PDistribution, PExampleGenerator, const long &, const int &) const
{ if (frequencies->varType != TValue::FLOATVAR)
    if (frequencies->outerVariable)
      raiseError("attribute '%s' is not continuous", frequencies->outerVariable->name.c_str());
    else
      raiseError("continuous attribute expected for condition");

    if (!frequencies->continuous->size())
      // This is ugly, but: if you change this, you should also change the code which catches it in
      // Bayesian learner
      raiseError("distribution (of attribute values, probably) is empty or has only a single value");

  PContingency cont = CLONE(TContingency, frequencies);
  const TDistributionMap &points = *frequencies->continuous;

/*  if (frequencies->continuous->size() == 1) {
    TDiscDistribution *f = (TDiscDistribution *)(points.begin()->second.getUnwrappedPtr());
    f->normalize();
    f->variances = mlnew TFloatList(f->size(), 0.0);
    return mlnew TConditionalProbabilityEstimator_FromDistribution(cont);
  }
*/
  cont->continuous->clear();

  vector<float> xpoints;
  distributePoints(points, nPoints, xpoints, distributionMethod);

  if (!xpoints.size())
    raiseError("no points for the curve (check 'nPoints')");
    
  if (frequencies->continuous->size() == 1) {
    TDiscDistribution *f = (TDiscDistribution *)(points.begin()->second.getUnwrappedPtr());
    f->normalize();
    f->variances = mlnew TFloatList(f->size(), 0.0);
    const_ITERATE(vector<float>, pi, xpoints)
      (*cont->continuous)[*pi] = f;
    return mlnew TConditionalProbabilityEstimator_FromDistribution(cont);
  }    

  TDistributionMap::const_iterator lowedge = points.begin();
  TDistributionMap::const_iterator highedge = points.end();

  bool needAll;
  map<float, PDistribution>::const_iterator from, to;

  vector<float>::const_iterator pi(xpoints.begin()), pe(xpoints.end());
  float refx = *pi;

  from = lowedge;
  to = highedge; 
  int totalNumOfPoints = frequencies->outerDistribution->abs;

  int needpoints = int(ceil(totalNumOfPoints * windowProportion));
  if (needpoints<3)
    needpoints = 3;


  TSimpleRandomGenerator rgen(frequencies->outerDistribution->cases);

  if ((needpoints<=0) || (needpoints>=totalNumOfPoints)) {  //points.size()
    needAll = true;
    from = lowedge;
    to = highedge;
  }
  else {
    needAll = false;

    /* Find the window */
    from = points.lower_bound(refx);
    to = points.upper_bound(refx);
    if (from==to)
      if (to != highedge)
        to++;
      else
        from --;

    /* Extend the interval; we set from to highedge when it would go beyond lowedge, to indicate that only to can be modified now */
    while (needpoints > 0) {
      if ((to == highedge) || ((from != highedge) && (refx - (*from).first < (*to).first - refx))) {
        if (from == lowedge)
          from = highedge;
        else {
          from--;
          needpoints -= (*from).second->cases;
        }
      }
      else {
        to++;
        if (to!=highedge)
          needpoints -= (*to).second->cases;
        else
          needpoints = 0;
      }

    }
    
    if (from == highedge)
      from = lowedge;
/*    else
      from++;*/
  }

  int numOfOverflowing = 0;
  // This follows http://www-2.cs.cmu.edu/afs/cs/project/jair/pub/volume4/cohn96a-html/node7.html
  for(;;) {
    TDistributionMap::const_iterator tt = to;
    --tt;
  
    float h = (refx - (*from).first);
    if ((*tt).first - refx  >  h)
      h = ((*tt).first - refx);

    /* Iterate through the window */

    tt = from;
    const float &x = (*tt).first;
    const PDistribution &y = (*tt).second;
    float cases = y->abs;

    float w = fabs(refx - x) / h;
    w = 1 - w*w*w;
    w = w*w*w;

    const float num = y->abs; // number of instances with this x - value
    float n = w * num;
    float Sww = w * w * num;

    float Sx = w * x * num;
    float Swwx  = w * w * x * num;
    float Swwxx = w * w * x * x * num;
    TDistribution *Sy = CLONE(TDistribution, y);
    PDistribution wSy = Sy;
    *Sy *= w;

    float Sxx = w * x * x * num;
    TDistribution *Syy = CLONE(TDistribution, y);
    PDistribution wSyy = Syy;
    *Syy *= w;

    TDistribution *Sxy = CLONE(TDistribution, y);
    PDistribution wSxy = Sxy;
    *Sxy *= w * x;

    if (tt!=to)
      while (++tt != to) {
        const float &x = (*tt).first;
        const PDistribution &y = (*tt).second;
        cases += y->abs;

        w = fabs(refx - x) / h;
        w = 1 - w*w*w;
        w = w*w*w;

        const float num = y->abs;
        n   += w * num;
        Sww += w * w * num;
        Sx  += w * x * num;
        Swwx += w * w * x * num;
        Swwxx += w * w * x * x * num;
        Sxx += w * x * x * num;

        TDistribution *ty = CLONE(TDistribution, y);
        PDistribution wty = ty;
        *ty *= w;
        *Sy  += wty;
        *Syy += wty;
        *ty *= x;
        *Sxy += wty;

        //*ty *= PDistribution(y);
      }

    float sigma_x2 = n<1e-6 ? 0.0 : (Sxx - Sx * Sx / n)/n;
    if (sigma_x2<1e-10) {
      *Sy *= 0;
      Sy->cases = cases;
      (*cont->continuous)[refx] = (wSy);
    }

    TDistribution *sigma_y2 = CLONE(TDistribution, Sy);
    PDistribution wsigma_y2 = sigma_y2;
    *sigma_y2 *= wsigma_y2;
    *sigma_y2 *= -1/n;
    *sigma_y2 += wSyy;
    *sigma_y2 *= 1/n;

    TDistribution *sigma_xy = CLONE(TDistribution, Sy);
    PDistribution wsigma_xy = sigma_xy;
    *sigma_xy *= -Sx/n;
    *sigma_xy += wSxy; 
    *sigma_xy *= 1/n;

    // This will be sigma_xy / sigma_x2, but we'll multiply it by whatever we need
    TDistribution *sigma_tmp = CLONE(TDistribution, sigma_xy);
    PDistribution wsigma_tmp = sigma_tmp;
    //*sigma_tmp *= wsigma_tmp;
    if (sigma_x2 > 1e-10)
      *sigma_tmp *= 1/sigma_x2;

    const float difx = refx - Sx/n;

    // computation of y
    *sigma_tmp *= difx;
    *Sy *= 1/n;
    *Sy += *sigma_tmp;

    // probabilities that are higher than 0.9 normalize with a logistic function, which produces two positive 
    // effects: prevents overfitting and avoids probabilities that are higher than 1.0. But, on the other hand, this 
    // solution is rather unmathematical. Do the same for probabilities that are lower than 0.1.

    vector<float>::iterator syi(((TDiscDistribution *)(Sy))->distribution.begin()); 
    vector<float>::iterator sye(((TDiscDistribution *)(Sy))->distribution.end()); 
    for (; syi!=sye; syi++) {
      if (*syi > 0.9) {
        Sy->abs -= *syi;
        *syi = 1/(1+exp(-10*((*syi)-0.9)*log(9.0)-log(9.0)));
        Sy->abs += *syi;
      }
      if (*syi < 0.1) {
        Sy->abs -= *syi;
        *syi = 1/(1+exp(10*(0.1-(*syi))*log(9.0)+log(9.0)));
        Sy->abs += *syi;
      }
    }

    Sy->cases = cases;
    Sy->normalize();
    (*cont->continuous)[refx] = (wSy);
 
    // now for the variance
    // restore sigma_tmp and compute the conditional sigma
    if ((fabs(difx) > 1e-10) && (sigma_x2 > 1e-10)) {
      *sigma_tmp *= (1/difx);
      *sigma_tmp *= wsigma_xy;
      *sigma_tmp *= -1; 
      *sigma_tmp += wsigma_y2;
      // fct corresponds to part of (10) in the brackets (see URL above)
   //   float fct = Sww + difx*difx/sigma_x2/sigma_x2 * (Swwxx   - 2/n * Sx*Swwx   +  2/n/n * Sx*Sx*Sww);
      float fct = 1 + difx*difx/sigma_x2; //n + difx*difx/sigma_x2+n*n --- add this product to the overall fct sum if you are estimating error for a single user and not for the line.  
      *sigma_tmp *= fct/n; // fct/n/n;
    }
    ((TDiscDistribution *)(Sy)) ->variances = mlnew TFloatList(((TDiscDistribution *)(sigma_tmp))->distribution);
    

    // on to the next point
    pi++;
    if (pi==pe)
      break; 

    refx = *pi;

    // Adjust the window
    while (to!=highedge) {
      float dif = (refx - (*from).first) - ((*to).first - refx);
      if ((dif>0) || (dif==0) && rgen.randbool()) {
        if (numOfOverflowing > 0) {
          from++;
          numOfOverflowing -= (*from).second->cases;
        }
        else {
          to++;
          if (to!=highedge) 
            numOfOverflowing += (*to).second->cases;
        }
      }
	    else
		    break;
    }
  }

  return mlnew TConditionalProbabilityEstimator_FromDistribution(cont);
}
