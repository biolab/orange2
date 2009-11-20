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

    Authors: Martin Mozina, Janez Demsar, Blaz Zupan, 1996--2004
    Contact: martin.mozina@fri.uni-lj.si
*/

#include "filter.hpp"
#include "table.hpp"
#include "stat.hpp"
#include "measures.hpp"
#include "discretize.hpp"
#include "distvars.hpp"
#include "classfromvar.hpp"
#include "progress.hpp"


#include "rulelearner.ppp"

DEFINE_TOrangeVector_classDescription(PRule, "TRuleList", true, ORANGE_API)
DEFINE_TOrangeVector_classDescription(PEVDist, "TEVDistList", true, ORANGE_API)

#ifdef _MSC_VER
#if _MSC_VER < 1300
template<class T>
inline T &min(const T&x, const T&y)
{ return x<y ? x : y; }
#endif
#endif

TRule::TRule()
: weightID(0),
  quality(ILLEGAL_FLOAT),
  complexity(-1),
  coveredExamples(NULL),
  coveredExamplesLength(-1),
  parentRule(NULL),
  chi(0.0),
  requiredConditions(0),
  baseDist(NULL)
{}


TRule::TRule(PFilter af, PClassifier cl, PLearner lr, PDistribution dist, PExampleTable ce, const int &w, const float &qu)
: filter(af),
  classifier(cl),
  learner(lr),
  classDistribution(dist),
  examples(ce),
  weightID(w),
  quality(qu),
  coveredExamples(NULL),
  coveredExamplesLength(-1),
  parentRule(NULL),
  chi(0.0),
  valuesFilter(NULL),
  requiredConditions(0),
  baseDist(dist)
{}


TRule::TRule(const TRule &other, bool copyData)
: filter(other.filter? other.filter->deepCopy() : PFilter()),
  valuesFilter(other.valuesFilter? other.valuesFilter->deepCopy() : PFilter()),
  classifier(other.classifier),
  learner(other.learner),
  complexity(other.complexity),
  classDistribution(copyData ? other.classDistribution: PDistribution()),
  examples(copyData ? other.examples : PExampleTable()),
  weightID(copyData ? other.weightID : 0),
  quality(copyData ? other.quality : ILLEGAL_FLOAT),
  coveredExamples(copyData && other.coveredExamples && (other.coveredExamplesLength >= 0) ? (int *)memcpy(new int[other.coveredExamplesLength], other.coveredExamples, other.coveredExamplesLength) : NULL),
  coveredExamplesLength(copyData ? other.coveredExamplesLength : -1),
  parentRule(other.parentRule),
  chi(other.chi),
  baseDist(other.baseDist),
  requiredConditions(other.requiredConditions)
{}


TRule::~TRule()
{ delete coveredExamples; }

bool TRule::operator ()(const TExample &ex)
{
  checkProperty(filter);
  return filter->call(ex);
}


#define HIGHBIT 0x80000000

PExampleTable TRule::operator ()(PExampleTable gen, const bool ref, const bool negate)
{
  checkProperty(filter);

  TExampleTable *table = ref ? mlnew TExampleTable(gen, 1) : mlnew TExampleTable(PExampleGenerator(gen));
  PExampleGenerator wtable = table;

  PEITERATE(ei, gen)
    if (filter->call(*ei) != negate)
      table->addExample(*ei);

  return wtable;
}


void TRule::filterAndStore(PExampleTable gen, const int &wei, const int &targetClass, const int *prevCovered, const int anExamples)
{
  checkProperty(filter);
  examples=this->call(gen);
  weightID = wei;
  classDistribution = getClassDistribution(examples, wei);
  if (classDistribution->abs==0)
    return;

  if (targetClass>=0)
    classifier = mlnew TDefaultClassifier(gen->domain->classVar, TValue(targetClass), classDistribution);
  else if (learner) {
    classifier = learner->call(examples,wei);
  }
  else
    classifier = mlnew TDefaultClassifier(gen->domain->classVar, classDistribution);
/*  if (anExamples > 0) {
    const int bitsInInt = sizeof(int)*8;
    coveredExamplesLength = anExamples/bitsInInt + 1;
    coveredExamples = (int *)malloc(coveredExamplesLength);
    if (prevCovered) {
      memcpy(coveredExamples, prevCovered, coveredExamplesLength);

      int *cei = coveredExamples-1;
      int mask = 0;
      int inBit = 0;

      PEITERATE(ei, gen) {
        if (!(*cei & mask)) {
          if (inBit)
            *cei = *cei << inBit;
          while(!*++cei);
          mask = -1;
          inBit = bitsInInt;
        }

        while( (*cei & HIGHBIT) == 0) {
          *cei = *cei << 1;
          *cei = mask << 1;
          inBit--;
        }

        if (filter->call(*ei)) {
          *cei = (*cei << 1) | 1;
          table->addExample(*ei);
        }
        else
          *cei = *cei << 1;

        mask = mask << 1;
        inBit--;
      }
    }

    else {
      int *cei = coveredExamples;
      int inBit = bitsInInt;

      PEITERATE(ei, gen) {
        if (filter->call(*ei)) {
          *cei = (*cei << 1) | 1;
          table->addExample(*ei);
        }
        else
          *cei = *cei << 1;

        if (!--inBit) {
          inBit = bitsInInt;
          cei++;
        }
      }
      *cei = *cei << inBit;
    }
  } */
}



bool haveEqualValues(const TRule &r1, const TRule &r2)
{
  const TDefaultClassifier *clsf1 = r1.classifier.AS(TDefaultClassifier);
  const TDefaultClassifier *clsf2 = r2.classifier.AS(TDefaultClassifier);
  if (!clsf1 || !clsf2)
    return false;

  const TDiscDistribution *dist1 = dynamic_cast<const TDiscDistribution *>(clsf1->defaultDistribution.getUnwrappedPtr());
  const TDiscDistribution *dist2 = dynamic_cast<const TDiscDistribution *>(clsf2->defaultDistribution.getUnwrappedPtr());

  float high1 = dist1->highestProb();
  float high2 = dist2->highestProb();

  for(TDiscDistribution::const_iterator d1i(dist1->begin()), d1e(dist1->end()), d2i(dist2->begin()), d2e(dist2->end());
      (d1i!=d1e) && (d2i!=d2e);
      d1i++, d2i++)
    if ((*d1i == high1) && (*d2i == high2))
      return true;

  return false;
}


bool TRule::operator <(const TRule &other) const
{
  if (!haveEqualValues(*this, other))
    return false;

  bool different = false;

  if (coveredExamples && other.coveredExamples) {
    int *c1i = coveredExamples;
    int *c2i = other.coveredExamples;
    for(int i = coveredExamplesLength; i--; c1i++, c2i++) {
      if (*c1i & ~*c2i)
        return false;
      if (*c1i != *c2i)
        different = true;
    }
  }
  else {
    raiseError("operator not implemented yet");
  }

  return different;
}


bool TRule::operator <=(const TRule &other) const
{
  if (!haveEqualValues(*this, other))
    return false;

  if (coveredExamples && other.coveredExamples) {
    int *c1i = coveredExamples;
    int *c2i = other.coveredExamples;
    for(int i = coveredExamplesLength; i--; c1i++, c2i++) {
      if (*c1i & ~*c2i)
        return false;
    }
  }

  else {
    raiseError("operator not implemented yet");
  }

  return true;
}


bool TRule::operator >(const TRule &other) const
{
  if (!haveEqualValues(*this, other))
    return false;

  bool different = false;
  if (coveredExamples && other.coveredExamples) {
    int *c1i = coveredExamples;
    int *c2i = other.coveredExamples;
    for(int i = coveredExamplesLength; i--; c1i++, c2i++) {
      if (~*c1i & *c2i)
        return false;
      if (*c1i != *c2i)
        different = true;
    }
  }

  else {
    raiseError("operator not implemented yet");
  }

  return different;
}


bool TRule::operator >=(const TRule &other) const
{
  if (!haveEqualValues(*this, other))
    return false;

  if (coveredExamples && other.coveredExamples) {
    int *c1i = coveredExamples;
    int *c2i = other.coveredExamples;
    for(int i = coveredExamplesLength; i--; c1i++, c2i++) {
      if (~*c1i & *c2i)
        return false;
    }
  }

  else {
    raiseError("operator not implemented yet");
  }

  return true;
}


bool TRule::operator ==(const TRule &other) const
{
  if (!haveEqualValues(*this, other))
    return false;

  if (coveredExamples && other.coveredExamples) {
    return !memcmp(coveredExamples, other.coveredExamples, coveredExamplesLength);
  }

  else {
    raiseError("operator not implemented yet");
  }

  return false;
}



TRuleValidator_LRS::TRuleValidator_LRS(const float &a, const float &min_coverage, const int &max_rule_complexity, const float &min_quality)
: alpha(a),
  min_coverage(min_coverage),
  max_rule_complexity(max_rule_complexity),
  min_quality(min_quality)
{}

bool TRuleValidator_LRS::operator()(PRule rule, PExampleTable, const int &, const int &targetClass, PDistribution apriori) const
{
  const TDiscDistribution &obs_dist = dynamic_cast<const TDiscDistribution &>(rule->classDistribution.getReference());
  if (!obs_dist.cases || obs_dist.cases < min_coverage)
    return false;
  if (max_rule_complexity > -1.0 && rule->complexity > max_rule_complexity)
    return false;
  if (min_quality>rule->quality)
    return false;

  const TDiscDistribution &exp_dist = dynamic_cast<const TDiscDistribution &>(apriori.getReference());
  if (obs_dist.abs == exp_dist.abs) //it turns out that this happens quite often
    return false;

  if (alpha >= 1.0)
    return true;

  if (targetClass == -1) {
    float lrs = 0.0;
    for(TDiscDistribution::const_iterator odi(obs_dist.begin()), ode(obs_dist.end()), edi(exp_dist.begin()), ede(exp_dist.end());
        (odi!=ode); odi++, edi++) {
      if ((edi!=ede) && (*edi) && (*odi))
        lrs += *odi * log(*odi / ((edi != ede) & (*edi > 0.0) ? *edi : 1e-5));
    }
    lrs = 2 * (lrs - obs_dist.abs * log(obs_dist.abs / exp_dist.abs));
    return (lrs > 0.0) && (chisqprob(lrs, float(obs_dist.size()-1)) <= alpha);
  }

  float p = (targetClass < obs_dist.size()) ? obs_dist[targetClass] : 1e-5;
  const float P = (targetClass < exp_dist.size()) && (exp_dist[targetClass] > 0.0) ? exp_dist[targetClass] : 1e-5;
  float n = obs_dist.abs - p;
  float N = exp_dist.abs - P;
  if (n>=N)
    return false;
  if (N<=0.0)
    N = 1e-6f;
  if (p<=0.0)
    p = 1e-6f;
  if (n<=0.0)
    n = 1e-6f;

  float lrs = 2 * (p*log(p/P) + n*log(n/N) - obs_dist.abs * log(obs_dist.abs/exp_dist.abs));
  return (lrs > 0.0) && (chisqprob(lrs, 1.0f) <= alpha);
}


float TRuleEvaluator_Entropy::operator()(PRule rule, PExampleTable, const int &, const int &targetClass, PDistribution apriori)
{
  const TDiscDistribution &obs_dist = dynamic_cast<const TDiscDistribution &>(rule->classDistribution.getReference());
  if (!obs_dist.cases)
    return -numeric_limits<float>::max();

  if (targetClass == -1)
    return -getEntropy(dynamic_cast<TDiscDistribution &>(rule->classDistribution.getReference()));

  const TDiscDistribution &exp_dist = dynamic_cast<const TDiscDistribution &>(apriori.getReference());

  float p = (targetClass < obs_dist.size()) ? obs_dist[targetClass] : 0.0;
  const float P = (targetClass < exp_dist.size()) && (exp_dist[targetClass] > 0.0) ? exp_dist[targetClass] : 1e-5;

  float n = obs_dist.abs - p;
  float N = exp_dist.abs - P;
  if (N<=0.0)
    N = 1e-6f;
  if (p<=0.0)
    p = 1e-6f;
  if (n<=0.0)
    n = 1e-6f;

  return ((p*log(p) + n*log(n) - obs_dist.abs * log(obs_dist.abs)) / obs_dist.abs);
}

float TRuleEvaluator_Laplace::operator()(PRule rule, PExampleTable, const int &, const int &targetClass, PDistribution apriori)
{
  const TDiscDistribution &obs_dist = dynamic_cast<const TDiscDistribution &>(rule->classDistribution.getReference());
  if (!obs_dist.cases)
    return 0;

  float p;
  if (targetClass == -1) {
    p = float(obs_dist.highestProb());
    return (p+1)/(obs_dist.abs+obs_dist.size());
  }
  p = float(obs_dist[targetClass]);
  return (p+1)/(obs_dist.abs+2);
}

TRuleEvaluator_LRS::TRuleEvaluator_LRS(const bool &sr)
: storeRules(sr)
{
  TRuleList *ruleList = mlnew TRuleList;
  rules = ruleList;
}

float TRuleEvaluator_LRS::operator()(PRule rule, PExampleTable, const int &, const int &targetClass, PDistribution apriori)
{
  const TDiscDistribution &obs_dist = dynamic_cast<const TDiscDistribution &>(rule->classDistribution.getReference());
  if (!obs_dist.cases)
    return 0.0;

  const TDiscDistribution &exp_dist = dynamic_cast<const TDiscDistribution &>(apriori.getReference());

  if (obs_dist.abs >= exp_dist.abs) //it turns out that this happens quite often
    return 0.0;

  if (targetClass == -1) {
    float lrs = 0.0;
    for(TDiscDistribution::const_iterator odi(obs_dist.begin()), ode(obs_dist.end()), edi(exp_dist.begin()), ede(exp_dist.end());
        (odi!=ode); odi++, edi++) {
      if ((edi!=ede) && (*edi) && (*odi))
        lrs += *odi * log(*odi / ((edi != ede) & (*edi > 0.0) ? *edi : 1e-5));
    }
    lrs = 2 * (lrs - obs_dist.abs * log(obs_dist.abs / exp_dist.abs));
    return lrs;
  }

  float p = (targetClass < obs_dist.size()) ? obs_dist[targetClass] : 1e-5;
  float P = (targetClass < exp_dist.size()) && (exp_dist[targetClass] > 0.0) ? exp_dist[targetClass] : 1e-5;

  if (p/obs_dist.abs <= P/exp_dist.abs)
    return 0.0;

  float n = obs_dist.abs - p;
  float N = exp_dist.abs - P;

  if (N<=0.0)
    N = 1e-6f;
  if (p<=0.0)
    p = 1e-6f;
  if (n<=0.0)
    n = 1e-6f;


  p = p - 0.5;
  n = obs_dist.abs - p;
  if (p<=(p+n)*P/(P+N))
	return 0.0;
  float ep = obs_dist.abs*P/(exp_dist.abs);

  float lrs = 2 * (p*log(p/ep) + n*log(n/obs_dist.abs) +
                   (P-p)*log((P-p)/(exp_dist.abs-obs_dist.abs)) + (N-n)*log((N-n)/(exp_dist.abs-obs_dist.abs)) -
                   (P-p)*log(P/exp_dist.abs)-N*log(N/exp_dist.abs));
  if (storeRules) {
    TRuleList &rlist = rules.getReference();
    rlist.push_back(rule);
  }
  return lrs;
}


TEVDist::TEVDist(const float & mu, const float & beta, PFloatList & percentiles)
: mu(mu),
  beta(beta),
  percentiles(percentiles)
{
  maxPercentile = (float)0.95;
  step = (float)0.1;
}

TEVDist::TEVDist()
{
  maxPercentile = (float)0.95;
  step = (float)0.1;
}

double TEVDist::getProb(const float & chi)
{
  if (!percentiles || percentiles->size()==0 || percentiles->at(percentiles->size()-1)<chi)
    return 1.0-exp(-exp((double)(mu-chi)/beta));
  if (chi < percentiles->at(0))
    return 1.0;
  TFloatList::const_iterator pi(percentiles->begin()), pe(percentiles->end());
  for (int i=0; (pi+1)!=pe; pi++,i++) {
    float a = *pi;
    float b = *(pi+1);
    if (chi>=a && chi <=b)
      return (maxPercentile-i*step)-step*(chi-a)/(b-a);
  }
  return 1.0;
}

float TEVDist::median()
{
  if (!percentiles || percentiles->size()==0)
    return mu + beta*0.36651292; // log(log(2))
  if (percentiles->size()%2 == 0)
    return (percentiles->at(percentiles->size()/2-1)+percentiles->at(percentiles->size()/2))/2;
  else
    return (percentiles->at(percentiles->size()/2));
}

TEVDistGetter_Standard::TEVDistGetter_Standard(PEVDistList dists)
: dists(dists)
{}

TEVDistGetter_Standard::TEVDistGetter_Standard()
{}

PEVDist TEVDistGetter_Standard::operator()(const PRule, const int & parentLength, const int & length) const
{
  // first element (correction for inter - rule optimism
  if (!length)
    return dists->at(0);
  // optimism between rule length of "parentLength" and true length "length"
  int indx = length*(length-1)/2 + parentLength + 1;
  if (dists->size() > indx)
    return dists->at(indx);
  return NULL;
}

float getChi(float p, float n, float P, float N)
{
  p = p - 0.5;
  n = n + 0.5;
  if (p<=(p+n)*P/(P+N))
	  return 0.0;
  float ep = (p+n)*P/(P+N);
  return 2*(p*log(p/ep)+n*log(n/(p+n))+(P-p)*log((P-p)/(P+N-p-n))+(N-n)*log((N-n)/(P+N-p-n))-(P-p)*log(P/(P+N))-N*log(N/(P+N)));
}

LNLNChiSq::LNLNChiSq(PEVDist evd, const float & chi)
: evd(evd),
  chi(chi)
{
  // extreme alpha = probability of obtaining chi with such extreme value distribution
  extremeAlpha = evd->getProb(chi);
  if (extremeAlpha < 1.0-evd->maxPercentile)
    extremeAlpha = -1.0;
  // exponent used in FT cumulative function (min because it is negative)
  exponent = min(float(log(log(1/evd->maxPercentile))),(evd->mu-chi)/evd->beta);
}

double LNLNChiSq::operator()(float chix) {
  if (chix<=0.0)
    return 100.0;
  double chip = chisqprob((double)chix,1.0)/2; // in statc
  if (extremeAlpha > 0.0)
    return chip-extremeAlpha;

  if (chip<=0.0)
    return -100.0;

  if (chip < 1e-6)
    return log(chip)-exponent;
  return log(-log(1-chip))-exponent;
}

LRInv::LRInv(float & n, float & P, float & N, float chiCorrected)
: n(n),
  P(P),
  chiCorrected(chiCorrected),
  N(N)
{}

double LRInv::operator()(float p){
  // check how it is done in call
  return getChi(p,n-p,P,N-P) - chiCorrected;
}

LRInvMean::LRInvMean(float correctedP, PRule rule, PRule groundRule, const int & targetClass)
: n(rule->classDistribution->abs),
  p(correctedP),
  P(groundRule->classDistribution->atint(targetClass)),
  N(groundRule->classDistribution->abs)
{}

double LRInvMean::operator()(float pc){
  return - getChi(p,n-p,pc*N/n,N-pc*N/n) + 0.30;
}


LRInvE::LRInvE(PRule rule, PRule groundRule, const int & targetClass, float chiCorrected)
: n(rule->classDistribution->abs),
  p(rule->classDistribution->atint(targetClass)),
  N(groundRule->classDistribution->abs),
  chiCorrected(chiCorrected)
{}


double LRInvE::operator()(float P){
  // check how it is done in call
  P *= N/n;
  return - getChi(p,n-p,P,N-P) + chiCorrected;
}

// Implementation of Brent's root finding method.
float brent(const float & minv, const float & maxv, const int & maxsteps, DiffFunc * func)
{
  float a = minv;
  float b = maxv;
  float fa = func->call(a);
  float fb = func->call(b);

  float threshold = 0.01 * (maxv - minv);

  if (fb>0 && fa>0 && fb>fa || fb<0 && fa<0 && fb<fa)
    return a;
	if (fb>0 && fa>0 && fb<fa || fb<0 && fa<0 && fb>fa)
    return b;

  float c = a; // c is previous value of b
  float fe, fc = fa;
  float m = 0.0, e = 0.0, d = 0.0;
  int counter = 0;
  while (1) {
    counter += 1;
    if (fb == fa)
      return b;
    else if (fb!=fc && fa!=fc)
      d = a*fb*fc/(fa-fb)/(fa-fc)+b*fa*fc/(fb-fa)/(fb-fc)+c*fa*fb/(fc-fa)/(fc-fb);
    else
      d = b-fb*(b-a)/(fb-fa);

    m = (a+b)/2;
    if (d<=m && d>=b || d>=m && d<=b)
      e = d;
    else
      e = m;
    fe = func->call(e);
    if (fe*fb<0) {
      a = b;
      fa = fb;
    }
    c = b;
    fc = fb;
    b = e;
    fb = fe;

    if (counter>maxsteps)
      return 0.0;
    if ((b>0.1 && fb*func->call(b-0.1)<=0) || fb*func->call(b+0.1)<=0)
      return b;
    if (abs(a-b)<threshold && fa*fb<0)
      return (a+b)/2.;
    if (fb*fa>0 || b>maxv || b<minv)
      return 0.0;
  }
}

TRuleEvaluator_mEVC::TRuleEvaluator_mEVC(const int & m, PEVDistGetter evDistGetter, PVariable probVar, PRuleValidator validator, const int & min_improved, const float & min_improved_perc, const int & optimismReduction)
: m(m),
  evDistGetter(evDistGetter),
  probVar(probVar),
  validator(validator),
  min_improved(min_improved),
  min_improved_perc(min_improved_perc),
  bestRule(NULL),
  ruleAlpha(1.0),
  attributeAlpha(1.0),
  optimismReduction(optimismReduction)
{}

TRuleEvaluator_mEVC::TRuleEvaluator_mEVC()
: m(0),
  evDistGetter(NULL),
  probVar(NULL),
  validator(NULL),
  min_improved(1),
  min_improved_perc(0),
  bestRule(NULL),
  ruleAlpha(1.0),
  attributeAlpha(1.0),
  optimismReduction(0)
{}

void TRuleEvaluator_mEVC::reset()
{
  bestRule = NULL;
}


/* The method validates rule's attributes' significance with respect to its extreme value corrected distribution. */
bool TRuleEvaluator_mEVC::ruleAttSignificant(PRule rule, PExampleTable examples, const int & weightID, const int &targetClass, PDistribution apriori, float & aprioriProb)
{
  PEVDist evd;

  // Should classical LRS be used, or EVC corrected?
  bool useClassicLRS = false;
  if (!optimismReduction)
    useClassicLRS = true;
  if (!useClassicLRS)
  {
    evd = evDistGetter->call(rule, 0, 0);
    if (evd->mu < 1.0)
      useClassicLRS = true;
  }

  TFilter_values *filter;
  if (rule->valuesFilter)
    filter = rule->valuesFilter.AS(TFilter_values);
  else
    filter = rule->filter.AS(TFilter_values);

  // Loop through all attributes - remove each and check significance
  bool rasig = true;
  int i,j;
  float chi;
  TFilter_values *newfilter;
  for (i=0; i<filter->conditions->size(); i++)
  {
      if (i<rule->requiredConditions)
        continue;
      // create a rule without one condition
      TRule *newRule = new TRule();
      PRule wnewRule = newRule;
      wnewRule->filter = new TFilter_values();
      wnewRule->filter->domain = examples->domain;
      wnewRule->complexity = rule->complexity - 1;
      newfilter = newRule->filter.AS(TFilter_values);
      for (j=0; j<filter->conditions->size(); j++)
        if (j!=i)
          newfilter->conditions->push_back(filter->conditions->at(j));
      wnewRule->filterAndStore(examples, weightID, targetClass);

      // compute lrs of rule vs new rule (without one condtion)
      if (!rule->classDistribution->abs || wnewRule->classDistribution->abs == rule->classDistribution->abs)
        return false;
      chi = getChi(rule->classDistribution->atint(targetClass),
                   rule->classDistribution->abs - rule->classDistribution->atint(targetClass),
                   wnewRule->classDistribution->atint(targetClass),
                   wnewRule->classDistribution->abs - wnewRule->classDistribution->atint(targetClass));
      // correct lrs with evc
      if (!useClassicLRS) {
        LNLNChiSq *diffFunc = new LNLNChiSq(evd,chi);
		    chi = brent(0.0,chi,100, diffFunc);
        delete diffFunc;
      }
      // check significance
      rasig = rasig & ((chi > 0.0) && (chisqprob(chi, 1.0f) <= attributeAlpha));
      if (!rasig)
        return false;
  }
  return true;
}

float combineEPositives(float N, float P, float oldQ, float n, float q)
{
  if (oldQ >= P/N)
	  return q*n;
  if (P <= 0.1)
	  return 0.0;
  return N*oldQ/P*q*n;
}

float TRuleEvaluator_mEVC::evaluateRulePessimistic(PRule rule, PExampleTable examples, const int & weightID, const int &targetClass, PDistribution apriori, const int & rLength, const float & aprioriProb)
{
  // extreme value distribution; optimism from ground Rule to rule
  PEVDist evd = evDistGetter->call(rule, 0, 0);

  if (!evd || evd->mu < 0.0)
    return -10e+6;
  //printf("mu=%f, beta=%f\n",evd->mu,evd->beta);
  if (evd->mu == 0.0 || !rule->parentRule)
  {
    // return as if rule distribution is not optimistic
    rule->chi = getChi(rule->classDistribution->atint(targetClass), rule->classDistribution->abs - rule->classDistribution->atint(targetClass),
                apriori->atint(targetClass), apriori->abs - apriori->atint(targetClass));
	  rule->estRF = rule->classDistribution->atint(targetClass)/rule->classDistribution->abs;
    return (rule->classDistribution->atint(targetClass)+m*aprioriProb)/(rule->classDistribution->abs+m);
  }

  // rule's improvement chi
  float chi = getChi(rule->classDistribution->atint(targetClass), rule->classDistribution->abs - rule->classDistribution->atint(targetClass),
                   rule->parentRule->classDistribution->atint(targetClass), rule->parentRule->classDistribution->abs - rule->parentRule->classDistribution->atint(targetClass));

  float ePos = 0.0;
  float median = evd->median();
  float rule_acc = rule->classDistribution->atint(targetClass)/rule->classDistribution->abs;
  float parent_acc = rule->parentRule->classDistribution->atint(targetClass)/rule->parentRule->classDistribution->abs;

   // need correcting? if rule very good, correcting will not change anything
  if ((evd->mu-chi)/evd->beta < -100)
      // return as if rule distribution is not optimistic
    ePos = rule->classDistribution->atint(targetClass);
  else if (rule_acc < parent_acc)
    ePos = rule->classDistribution->atint(targetClass);
  else if (chi<=median)
    ePos = rule->classDistribution->abs * parent_acc;
  else {
      // compute ePos
      LRInvE *diffFunc = new LRInvE(rule, rule->parentRule, targetClass, median);
      ePos = brent(rule->parentRule->classDistribution->atint(targetClass)/rule->parentRule->classDistribution->abs*rule->classDistribution->atint(targetClass), rule->classDistribution->atint(targetClass), 100, diffFunc);
      delete diffFunc;

  }

  float ePosOA; // expected positive examples considering base rule also
  ePosOA = combineEPositives(rule->parentRule->classDistribution->abs, rule->parentRule->classDistribution->atint(targetClass), rule->parentRule->estRF, rule->classDistribution->abs, ePos/rule->classDistribution->abs);
  rule->chi = getChi(ePosOA, rule->classDistribution->abs - ePosOA,
                apriori->atint(targetClass), apriori->abs - apriori->atint(targetClass));

  rule->estRF = ePosOA/rule->classDistribution->abs;
  float quality = (ePosOA + m*aprioriProb)/(rule->classDistribution->abs+m);

  if (quality > aprioriProb)
    return quality;
  return aprioriProb-0.01+0.01*rule_acc;
}

float TRuleEvaluator_mEVC::evaluateRuleM(PRule rule, PExampleTable examples, const int & weightID, const int &targetClass, PDistribution apriori, const int & rLength, const float & aprioriProb)
{
  if (!m & !rule->classDistribution->abs)
    return 0.0;
  rule->chi = getChi(rule->classDistribution->atint(targetClass), rule->classDistribution->abs - rule->classDistribution->atint(targetClass),
                     apriori->atint(targetClass), apriori->abs - apriori->atint(targetClass));
  float p = rule->classDistribution->atint(targetClass)+m*apriori->atint(targetClass)/apriori->abs;
  p = p/(rule->classDistribution->abs+m);
  return p;
}

// evaluates a rule, which can have its base conditions, base rule
// can be evaluted in any way.
float TRuleEvaluator_mEVC::evaluateRuleEVC_Step(PRule rule, PExampleTable examples, const int & weightID, const int &targetClass, PDistribution apriori, const int & rLength, const float & aprioriProb)
{
  // extreme value distribution; optimism from parent Rule to rule
  PEVDist evd = evDistGetter->call(rule, 0, 0);

  if (!evd || evd->mu < 0.0)
    return -10e+6;
  //printf("mu=%f, beta=%f\n",evd->mu,evd->beta);
  if (evd->mu == 0.0 || !rule->parentRule)
  {
    // return as if rule distribution is not optimistic
    rule->chi = getChi(rule->classDistribution->atint(targetClass), rule->classDistribution->abs - rule->classDistribution->atint(targetClass),
                apriori->atint(targetClass), apriori->abs - apriori->atint(targetClass));
	  rule->estRF = rule->classDistribution->atint(targetClass)/rule->classDistribution->abs;
    rule->distP = rule->classDistribution->atint(targetClass);
    return (rule->classDistribution->atint(targetClass)+m*aprioriProb)/(rule->classDistribution->abs+m);
  }

  // rule's chi
  float chi = getChi(rule->classDistribution->atint(targetClass), rule->classDistribution->abs - rule->classDistribution->atint(targetClass),
                   rule->parentRule->distP, rule->parentRule->classDistribution->abs - rule->parentRule->distP);

  float ePos = 0.0;
  float median = evd->median();
  float rule_acc = rule->classDistribution->atint(targetClass)/rule->classDistribution->abs;
  float parent_acc = rule->parentRule->distP/rule->parentRule->classDistribution->abs;

   // need correcting? if rule very good, correcting will not change anything
  //printf("chi = %4.3f, median = %4.3f\n",chi, median);

  // compute new distP
  if (chi<=median)
  {
      LRInv *diffFunc = new LRInv(rule->classDistribution->abs, rule->parentRule->distP, rule->parentRule->classDistribution->abs, median);
      rule->distP = brent(rule->classDistribution->atint(targetClass), rule->classDistribution->abs, 100, diffFunc);
      delete diffFunc;
  }
  else
    rule->distP = rule->classDistribution->atint(targetClass);

//  if (rule->parentRule->distP > rule->parentRule->classDistribution->atint(targetClass))
//  {
//  if (rule->classDistribution->atint(targetClass) >= 38 || rule->parentRule->classDistribution->atint(targetClass) >= 38)
 //   {
//  printf("complexoty = %d\n", rule->complexity);
 //   printf("parent: all = %4.2f, pos = %4.2f, distP = %4.2f\n", rule->parentRule->classDistribution->abs, rule->parentRule->classDistribution->atint(targetClass), rule->parentRule->distP);
 //   printf("child: all = %4.2f, pos = %4.2f, distP = %4.2f\n", rule->classDistribution->abs, rule->classDistribution->atint(targetClass), rule->distP);
 // }

  if ((evd->mu-chi)/evd->beta < -100)
      // return as if rule distribution is not optimistic
    ePos = rule->classDistribution->atint(targetClass);
/*  else if (rule_acc < parent_acc)
    ePos = rule->classDistribution->atint(targetClass); */
  else if (chi<=median)
    ePos = rule->classDistribution->abs * parent_acc;
  else {
      // correct chi
      LNLNChiSq *diffFunc = new LNLNChiSq(evd,chi);
      rule->chi = brent(0.0,chi,100, diffFunc); // this is only the correction of one step chi
      delete diffFunc;

      //printf("rule chi = %4.3f\n",rule->chi);
      // compute expected number of positive examples relatively to base rule
      if (rule->chi > 0.0)
      {
        // correct optimism

        LRInv *diffFunc = new LRInv(rule->classDistribution->abs, rule->parentRule->distP, rule->parentRule->classDistribution->abs, rule->chi); //-0.45);
        ePos = brent(rule->parentRule->distP/rule->parentRule->classDistribution->abs*rule->classDistribution->abs, rule->classDistribution->atint(targetClass), 100, diffFunc);
        //printf("epos = %4.3f\n",ePos);
        delete diffFunc;
      }
      else
        ePos = rule->classDistribution->abs * parent_acc;
  }

  float ePosOA; // expected positive examples considering base rule also
  ePosOA = combineEPositives(rule->parentRule->classDistribution->abs, rule->parentRule->distP, rule->parentRule->estRF, rule->classDistribution->abs, ePos/rule->classDistribution->abs);
  //printf("eposoa = %4.3f\n",ePosOA);

  rule->chi = getChi(ePosOA, rule->classDistribution->abs - ePosOA,
                apriori->atint(targetClass), apriori->abs - apriori->atint(targetClass));

  rule->estRF = ePosOA/rule->classDistribution->abs;
  float quality = (ePosOA + m*aprioriProb)/(rule->classDistribution->abs+m);
  //printf("quality = %4.3f, rf = %4.3f\n",quality, rule->estRF);
 // if (rule->classDistribution->atint(targetClass) >= 38 || rule->parentRule->classDistribution->atint(targetClass) >= 38)
 //   printf("child: all %4.4f %4.4f %4.4f\n", rule->distP, quality, aprioriProb);
  if (quality > aprioriProb)
    return quality;
  if (rule_acc < aprioriProb)
    return rule_acc;
  return aprioriProb; //-0.01+0.01*rule_acc;
}


// evaluates a rule, which can have its base conditions, base rule
// can be evaluted in any way.
float TRuleEvaluator_mEVC::evaluateRuleEVC(PRule rule, PExampleTable examples, const int & weightID, const int &targetClass, PDistribution apriori, const int & rLength, const float & aprioriProb)
{
  // extreme value distribution; optimism from parent Rule to rule
  PEVDist evd = evDistGetter->call(rule, 0, rLength - rule->requiredConditions);

  // get base distribution (normally apriori, otherwise distribution of argument)
  PDistribution base = rule->baseDist;
  float baseProb = base->atint(targetClass)/base->abs;

  if (!evd || evd->mu < 0.0)
    return -10e+6;
  //printf("mu=%f, beta=%f\n",evd->mu,evd->beta);
  if (evd->mu == 0.0)
  {
    // return as if rule distribution is not optimistic
    rule->chi = getChi(rule->classDistribution->atint(targetClass), rule->classDistribution->abs - rule->classDistribution->atint(targetClass),
                apriori->atint(targetClass), apriori->abs - apriori->atint(targetClass));
	  rule->estRF = rule->classDistribution->atint(targetClass)/rule->classDistribution->abs;
    rule->distP = rule->classDistribution->atint(targetClass);
    return (rule->classDistribution->atint(targetClass)+m*aprioriProb)/(rule->classDistribution->abs+m);
  }

  // rule's chi
  float chi = getChi(rule->classDistribution->atint(targetClass), rule->classDistribution->abs - rule->classDistribution->atint(targetClass),
                   base->atint(targetClass), base->abs - base->atint(targetClass));
  float ePos = 0.0;
  float median = evd->median();
  float rule_acc = rule->classDistribution->atint(targetClass)/rule->classDistribution->abs;

  if ((evd->mu-chi)/evd->beta < -100)
    ePos = rule->classDistribution->atint(targetClass);
  else if (rule_acc < baseProb)
    ePos = rule->classDistribution->atint(targetClass);
  else if (chi<=median)
    ePos = rule->classDistribution->abs * baseProb;
  else {
      // correct chi
      LNLNChiSq *diffFunc = new LNLNChiSq(evd,chi);
      rule->chi = brent(0.0,chi,100, diffFunc); // this is only the correction of one step chi
      delete diffFunc;

      //printf("rule chi = %4.3f\n",rule->chi);
      // compute expected number of positive examples relatively to base rule
      if (rule->chi > 0.0)
      {
        // correct optimism

        float baseTarget = base->atint(targetClass);
        LRInv *diffFunc = new LRInv(rule->classDistribution->abs, baseTarget, base->abs, rule->chi); //-0.45);
        ePos = brent(base->atint(targetClass)/base->abs*rule->classDistribution->abs, rule->classDistribution->atint(targetClass), 100, diffFunc);
        //printf("epos = %4.3f\n",ePos);
        delete diffFunc;
      }
      else
        ePos = rule->classDistribution->abs * baseProb;
  }

  rule->chi = getChi(ePos, rule->classDistribution->abs - ePos,
                apriori->atint(targetClass), apriori->abs - apriori->atint(targetClass));

  rule->estRF = ePos/rule->classDistribution->abs;
  float quality = (ePos + m*aprioriProb)/(rule->classDistribution->abs+m);
  //printf("quality = %4.3f, rf = %4.3f\n",quality, rule->estRF);
 // if (rule->classDistribution->atint(targetClass) >= 38 || rule->parentRule->classDistribution->atint(targetClass) >= 38)
 //   printf("child: all %4.4f %4.4f %4.4f\n", rule->distP, quality, aprioriProb);
  if (quality > aprioriProb)
    return quality;
  if (rule_acc < aprioriProb)
    return rule_acc;
  return aprioriProb; //-0.01+0.01*rule_acc;
}




float TRuleEvaluator_mEVC::operator()(PRule rule, PExampleTable examples, const int & weightID, const int &targetClass, PDistribution apriori)
{
  rule->chi = 0.0;
  if (!rule->classDistribution->abs || !rule->classDistribution->atint(targetClass))
    return 0;

  // evaluate rule
  int rLength = rule->complexity;
  float aprioriProb = apriori->atint(targetClass)/apriori->abs;
  float quality;

  if (optimismReduction == 0)
    quality = evaluateRuleM(rule,examples,weightID,targetClass,apriori,rLength,aprioriProb);
  else if (optimismReduction == 1)
    quality = evaluateRulePessimistic(rule,examples,weightID,targetClass,apriori,rLength,aprioriProb);
  else if (optimismReduction == 2)
    quality = evaluateRuleEVC(rule,examples,weightID,targetClass,apriori,rLength,aprioriProb);
  else
    quality = evaluateRuleEVC_Step(rule,examples,weightID,targetClass,apriori,rLength,aprioriProb);

/*  if (optimismReduction == 2)
	printf("rule: %f, %f, %f\n",rule->classDistribution->atint(targetClass), rule->classDistribution->abs, quality); */
  if (quality < 0.0)
    return quality;
  if (!probVar || !returnExpectedProb)
    return quality;

  // get rule's probability coverage
  int improved = 0;
  PEITERATE(ei, rule->examples)
    if ((*ei).getClass().intV == targetClass && quality > (*ei)[probVar].floatV)
      improved ++;

  // compute future quality = expected quality when rule is finalised
  float bestQuality;
  float futureQuality = 0.0;
  if (rule->classDistribution->atint(targetClass) == rule->classDistribution->abs)
    futureQuality = -1.0;
  else {
    PDistribution oldRuleDist = rule->classDistribution;
    float rulesTrueChi = rule->chi;
    float rulesDistP = rule->distP;
    rule->classDistribution = mlnew TDiscDistribution(examples->domain->classVar);
    rule->classDistribution->setint(targetClass, oldRuleDist->atint(targetClass));
    rule->classDistribution->abs = rule->classDistribution->atint(targetClass);
    rule->complexity += 1;

    float estRF = rule->estRF;
    if (optimismReduction == 0)
      bestQuality = evaluateRuleM(rule,examples,weightID,targetClass,apriori,rLength+1,aprioriProb);
    else if (optimismReduction == 1)
      bestQuality = evaluateRulePessimistic(rule,examples,weightID,targetClass,apriori,rLength+1,aprioriProb);
    else if (optimismReduction == 2)
      bestQuality = evaluateRuleEVC(rule,examples,weightID,targetClass,apriori,rLength+1,aprioriProb);
    else
      bestQuality = evaluateRuleEVC_Step(rule,examples,weightID,targetClass,apriori,rLength+1,aprioriProb);

    rule->estRF = estRF;
    rule->classDistribution = oldRuleDist;
    rule->chi = rulesTrueChi;
    rule->complexity -= 1;
    rule->distP = rulesDistP;

    if (bestQuality <= quality)
      futureQuality = -1.0;
    else if (bestRule && bestQuality <= bestRule->quality)
      futureQuality = -1.0;
    else {
      futureQuality = 0.0;
      PEITERATE(ei, rule->examples) {
        if ((*ei).getClass().intV != targetClass)
          continue;
      /*  if (quality >= (*ei)[probVar].floatV) {
          futureQuality += 1.0;
          continue;
        } */
        if (bestQuality <= (*ei)[probVar].floatV) {
          continue;
        }
        float x = ((*ei)[probVar].floatV-quality); //*rule->classDistribution->abs;
        if ((*ei)[probVar].floatV > quality)
          x *= (1.0-quality)/(bestQuality-quality);
        x /= sqrt(quality*(1.0-quality)); // rule->classDistribution->abs*
 //       if (max(1e-12,1.0-zprob(x)) > futureQuality)
 //           futureQuality = max(1e-12,1.0-zprob(x));
 //           futureQuality += max(1e-12,1.0-zprob(x));
        futureQuality += log(1.0-max(1e-12,1.0-2*zprob(x)));
      }
      futureQuality = 1.0 - exp(futureQuality);
      //futureQuality /= rule->classDistribution->atint(targetClass); //apriori->abs; //rule->classDistribution->atint(targetClass);//rule->classDistribution->abs;
    }
  }

  // store best rule as best rule and return expected quality of this rule
  rule->quality = quality;
  if (improved >= min_improved &&
      improved/rule->classDistribution->atint(targetClass) > min_improved_perc*0.01 &&
      quality > (aprioriProb + 1e-3) &&
      (!bestRule || (quality>bestRule->quality+1e-3)) &&
      (!validator || validator->call(rule, examples, weightID, targetClass, apriori))) {

      TRule *pbestRule = new TRule(rule.getReference(), true);
      PRule wpbestRule = pbestRule;
      // check if rule is significant enough
      bool ruleGoodEnough = true;

      if (ruleAlpha < 1.0)
        ruleGoodEnough = ruleGoodEnough & ((rule->chi > 0.0) && (chisqprob(rule->chi, 1.0f) <= ruleAlpha));
      if (ruleGoodEnough && attributeAlpha < 1.0)
        ruleGoodEnough = ruleGoodEnough & ruleAttSignificant(rule, examples, weightID, targetClass, apriori, aprioriProb);
      if (ruleGoodEnough)
      {
        bestRule = wpbestRule;
        bestRule->quality = quality;
        futureQuality += 1.0;
      }
  }
  return futureQuality;
}

bool worstRule(const PRule &r1, const PRule &r2)
{ return    (r1->quality > r2->quality)
          || (r1->quality==r2->quality
          && r1->complexity < r2->complexity);
}
/*         || (r1->quality==r2->quality)
            && (   (r1->complexity < r2->complexity)
                || (r1->complexity == r2->complexity)
                   && ((int(r1.getUnwrappedPtr()) ^ int(r2.getUnwrappedPtr())) & 16) != 0
               ); }  */

bool inRules(PRuleList rules, PRule rule)
{
  TRuleList::const_iterator ri(rules->begin()), re(rules->end());
  PExampleGenerator rulegen = rule->examples;
  for (; ri!=re; ri++) {
    PExampleGenerator rigen = (*ri)->examples;
    if (rigen->numberOfExamples() == rulegen->numberOfExamples()) {
      TExampleIterator rei(rulegen->begin()), ree(rulegen->end());
      TExampleIterator riei(rigen->begin()), riee(rigen->end());
      for (; rei != ree && !(*rei).compare(*riei); ++rei, ++riei) {
      }
        if (rei == ree)
          return true;
    }
  }
  return false;
}

TRuleBeamFilter_Width::TRuleBeamFilter_Width(const int &w)
: width(w)
{}


void TRuleBeamFilter_Width::operator()(PRuleList &rules, PExampleTable, const int &)
{
  if (rules->size() > width) {
    sort(rules->begin(), rules->end(), worstRule);

    TRuleList *filteredRules = mlnew TRuleList;
    PRuleList wFilteredRules = filteredRules;

    int nRules = 0;
    TRuleList::const_iterator ri(rules->begin()), re(rules->end());
    while (nRules < width && ri != re) {
      if (!inRules(wFilteredRules,*ri)) {
        wFilteredRules->push_back(*ri);
        nRules++;
      }
      ri++;
    }
    rules =  wFilteredRules;
  }
}


inline void _selectBestRule(PRule &rule, PRule &bestRule, int &wins, TRandomGenerator &rgen)
{
  if ((rule->quality > bestRule->quality) || (rule->complexity < bestRule->complexity)) {
    bestRule = rule;
    wins = 1;
  }
  else if ((rule->complexity == bestRule->complexity) && rgen.randbool(++wins))
    bestRule = rule;
}



PRuleList TRuleBeamInitializer_Default::operator()(PExampleTable data, const int &weightID, const int &targetClass, PRuleList baseRules, PRuleEvaluator evaluator, PDistribution apriori, PRule &bestRule)
{
  checkProperty(evaluator);

  TRuleList *ruleList = mlnew TRuleList();
  PRuleList wruleList = ruleList;

  TRandomGenerator rgen(data->numberOfExamples());
  int wins;

  if (baseRules && baseRules->size())
    PITERATE(TRuleList, ri, baseRules) {
      TRule *newRule = mlnew TRule((*ri).getReference(), true);
      PRule wNewRule = newRule;
      ruleList->push_back(wNewRule);
      if (!newRule->examples)
      {
        newRule->filterAndStore(data,weightID,targetClass);
        newRule->baseDist = newRule->classDistribution;
      }
      newRule->quality = evaluator->call(wNewRule, data, weightID, targetClass, apriori);
      if (!bestRule || (newRule->quality > bestRule->quality)) {
        bestRule = wNewRule;
        wins = 1;
      }
      else
        if (newRule->quality == bestRule->quality)
          _selectBestRule(wNewRule, bestRule, wins, rgen);
    }

  else {
     TRule *ubestRule = mlnew TRule();
     bestRule = ubestRule;
     ruleList->push_back(bestRule);
     ubestRule->filter = new TFilter_values();
     ubestRule->filter->domain = data->domain;
     ubestRule->filterAndStore(data, weightID,targetClass);
     ubestRule->baseDist = ubestRule->classDistribution;
     ubestRule->complexity = 0;
  }

  return wruleList;
}


PRuleList TRuleBeamRefiner_Selector::operator()(PRule wrule, PExampleTable data, const int &weightID, const int &targetClass)
{
  if (!discretization) {
    discretization = mlnew TEntropyDiscretization();
    dynamic_cast<TEntropyDiscretization *>(discretization.getUnwrappedPtr())->forceAttribute = true;
    dynamic_cast<TEntropyDiscretization *>(discretization.getUnwrappedPtr())->maxNumberOfIntervals = 5;
  }

  TRule &rule = wrule.getReference();
  TFilter_values *filter = wrule->filter.AS(TFilter_values);
  if (!filter)
    raiseError("a filter of type 'Filter_values' expected");

  TRuleList *ruleList = mlnew TRuleList;
  PRuleList wRuleList = ruleList;

  TDomainDistributions ddist(wrule->examples, wrule->weightID);

  const TVarList &attributes = rule.examples->domain->attributes.getReference();

  vector<bool> used(attributes.size(), false);
  PITERATE(TValueFilterList, vfi, filter->conditions)
    used[(*vfi)->position] = true;

  vector<bool>::const_iterator ui(used.begin());
  TDomainDistributions::const_iterator di(ddist.begin());
  TVarList::const_iterator vi(attributes.begin()), ve(attributes.end());
  int pos = 0;
  for(; vi != ve; vi++, ui++, pos++, di++) {
    if ((*vi)->varType == TValue::INTVAR) {
      if (!*ui) {
        vector<float>::const_iterator idi((*di).AS(TDiscDistribution)->begin());
        for(int v = 0, e = (*vi)->noOfValues(); v != e; v++)
          if (*idi>0) {
            TRule *newRule = mlnew TRule(rule, false);
            ruleList->push_back(newRule);
            newRule->complexity++;

            filter = newRule->filter.AS(TFilter_values);

            TValueFilter_discrete *newCondition = mlnew TValueFilter_discrete(pos, *vi, 0);
            filter->conditions->push_back(newCondition);

            TValue value = TValue(v);
            newCondition->values->push_back(value);
            newRule->filterAndStore(rule.examples, rule.weightID,targetClass);
            newRule->parentRule = wrule;
          }
      }
    }

    else if (((*vi)->varType == TValue::FLOATVAR)) {
      if (discretization) {
        PVariable discretized;
        try {
          discretized = discretization->call(rule.examples, *vi, weightID);
        } catch(...) {
          continue;
        }
        TClassifierFromVar *cfv = discretized->getValueFrom.AS(TClassifierFromVar);
        TDiscretizer *discretizer = cfv ? cfv->transformer.AS(TDiscretizer) : NULL;
        if (!discretizer)
          raiseError("invalid or unrecognized discretizer");

        vector<float> cutoffs;
        discretizer->getCutoffs(cutoffs);
        if (cutoffs.size()) {
          TRule *newRule;
          newRule = mlnew TRule(rule, false);
          PRule wnewRule = newRule;
          newRule->complexity++;
          newRule->parentRule = wrule;

          newRule->filter.AS(TFilter_values)->conditions->push_back(mlnew TValueFilter_continuous(pos,  TValueFilter_continuous::LessEqual,		cutoffs.front(), 0, 0));
          newRule->filterAndStore(rule.examples, rule.weightID,targetClass);
          if (wrule->classDistribution->cases > wnewRule->classDistribution->cases)
            ruleList->push_back(newRule);

          for(vector<float>::const_iterator ci(cutoffs.begin()), ce(cutoffs.end()-1); ci != ce; ci++) {
            newRule = mlnew TRule(rule, false);
            wnewRule = newRule;
            newRule->complexity++;
            newRule->parentRule = wrule;
            filter = newRule->filter.AS(TFilter_values);
            filter->conditions->push_back(mlnew TValueFilter_continuous(pos,  TValueFilter_continuous::Greater, *ci, 0, 0));
            newRule->filterAndStore(rule.examples, rule.weightID,targetClass);
            if (wrule->classDistribution->cases > wnewRule->classDistribution->cases)
              ruleList->push_back(newRule);

            newRule = mlnew TRule(rule, false);
            wnewRule = newRule;
            newRule->complexity++;
            newRule->parentRule = wrule;
            filter = newRule->filter.AS(TFilter_values);
            filter->conditions->push_back(mlnew TValueFilter_continuous(pos,  TValueFilter_continuous::LessEqual, *(ci+1), 0, 0));
            newRule->filterAndStore(rule.examples, rule.weightID,targetClass);
            if (wrule->classDistribution->cases > wnewRule->classDistribution->cases)
              ruleList->push_back(newRule);
          }

          newRule = mlnew TRule(rule, false);
          ruleList->push_back(newRule);
          newRule->complexity++;

          newRule->filter.AS(TFilter_values)->conditions->push_back(mlnew TValueFilter_continuous(pos,  TValueFilter_continuous::Greater, cutoffs.back(), 0, 0));
          newRule->filterAndStore(rule.examples, rule.weightID,targetClass);
          newRule->parentRule = wrule;
        }
      }
      else
        raiseWarning("discretizer not given, continuous attributes will be skipped");
    }
  }
  if (!discretization)
    discretization = PDiscretization();
  return wRuleList;
}


PRuleList TRuleBeamCandidateSelector_TakeAll::operator()(PRuleList &existingRules, PExampleTable, const int &)
{
  PRuleList candidates = mlnew TRuleList(existingRules.getReference());
//  existingRules->clear();
  existingRules->erase(existingRules->begin(), existingRules->end());
  return candidates;
}


PRule TRuleBeamFinder::operator()(PExampleTable data, const int &weightID, const int &targetClass, PRuleList baseRules)
{
  // set default values if value not set
  bool tempInitializer = !initializer;
  if (tempInitializer)
    initializer = mlnew TRuleBeamInitializer_Default;
  bool tempCandidateSelector = !candidateSelector;
  if (tempCandidateSelector)
    candidateSelector = mlnew TRuleBeamCandidateSelector_TakeAll;
  bool tempRefiner = !refiner;
  if (tempRefiner)
    refiner = mlnew TRuleBeamRefiner_Selector;
/*  bool tempValidator = !validator;
  if (tempValidator)
    validator = mlnew TRuleValidator_LRS((float)0.01);
  bool tempRuleStoppingValidator = !ruleStoppingValidator;
  if (tempRuleStoppingValidator)
    ruleStoppingValidator = mlnew TRuleValidator_LRS((float)0.05); */
  bool tempEvaluator = !evaluator;
  if (tempEvaluator)
    evaluator = mlnew TRuleEvaluator_Entropy;
  bool tempRuleFilter = !ruleFilter;
  if (tempRuleFilter)
    ruleFilter = mlnew TRuleBeamFilter_Width;

  checkProperty(initializer);
  checkProperty(candidateSelector);
  checkProperty(refiner);
  checkProperty(evaluator);
  checkProperty(ruleFilter);

  PDistribution apriori = getClassDistribution(data, weightID);

  TRandomGenerator rgen(data->numberOfExamples());
  int wins = 1;

  PRule bestRule;
  PRuleList ruleList = initializer->call(data, weightID, targetClass, baseRules, evaluator, apriori, bestRule);

  {
  PITERATE(TRuleList, ri, ruleList) {
    if (!(*ri)->examples)
      (*ri)->filterAndStore(data, weightID,targetClass);
    if ((*ri)->quality == ILLEGAL_FLOAT)
      (*ri)->quality = evaluator->call(*ri, data, weightID, targetClass, apriori);
  }
  }

  if (!bestRule->examples)
    bestRule->filterAndStore(data, weightID,targetClass);
  if (bestRule->quality == ILLEGAL_FLOAT)
    bestRule->quality = evaluator->call(bestRule, data, weightID, targetClass, apriori);

  int bestRuleLength = 0;
  while(ruleList->size()) {
    PRuleList candidateRules = candidateSelector->call(ruleList, data, weightID);
    PITERATE(TRuleList, ri, candidateRules) {
      PRuleList newRules = refiner->call(*ri, data, weightID, targetClass);
      PITERATE(TRuleList, ni, newRules) {
        (*ni)->quality = evaluator->call(*ni, data, weightID, targetClass, apriori);
        if ((*ni)->quality > bestRule->quality && (!validator || validator->call(*ni, data, weightID, targetClass, apriori)))
          _selectBestRule(*ni, bestRule, wins, rgen);
        if (!ruleStoppingValidator || ruleStoppingValidator->call(*ni, (*ri)->examples, weightID, targetClass, (*ri)->classDistribution)) {
          ruleList->push_back(*ni);
        }
      }
    }
    ruleFilter->call(ruleList,data,weightID);
  }

  // set empty values if value was not set (used default)
  if (tempInitializer)
    initializer = PRuleBeamInitializer();
  if (tempCandidateSelector)
    candidateSelector = PRuleBeamCandidateSelector();
  if (tempRefiner)
    refiner = PRuleBeamRefiner();
/*  if (tempValidator)
    validator = PRuleValidator();
  if (tempRuleStoppingValidator)
    ruleStoppingValidator = PRuleValidator();  */
  if (tempEvaluator)
    evaluator = PRuleEvaluator();
  if (tempRuleFilter)
    ruleFilter = PRuleBeamFilter();

  return bestRule;
}


TRuleLearner::TRuleLearner(bool se, int tc, PRuleList rl)
: storeExamples(se),
  targetClass(tc),
  baseRules(rl)
{}


PClassifier TRuleLearner::operator()(PExampleGenerator gen, const int &weightID)
{
  return this->call(gen,weightID,targetClass,baseRules);
}

PClassifier TRuleLearner::operator()(PExampleGenerator gen, const int &weightID, const int &targetClass, PRuleList baseRules)
{
  // Initialize default values if values not set
  bool tempDataStopping = !dataStopping && !ruleStopping;
  if (tempDataStopping)
    dataStopping = mlnew TRuleDataStoppingCriteria_NoPositives;

  bool tempRuleFinder = !ruleFinder;
  if (tempRuleFinder)
    ruleFinder = mlnew TRuleBeamFinder;

  bool tempCoverAndRemove = !coverAndRemove;
  if (tempCoverAndRemove)
    coverAndRemove = mlnew TRuleCovererAndRemover_Default;

  checkProperty(ruleFinder);
  checkProperty(coverAndRemove);

  TExampleTable *data = mlnew TExampleTable(gen);
  PExampleTable wdata = data;

  if (!dataStopping && !ruleStopping)
    raiseError("no stopping criteria; set 'dataStopping' and/or 'ruleStopping'");

  TRuleList *ruleList = mlnew TRuleList;
  PRuleList wruleList = ruleList;

  int currWeightID = weightID;

  float beginwe=0.0, currentwe;
  if (progressCallback) {
    if (targetClass==-1)
      beginwe = wdata->weightOfExamples(weightID);
    else {
      PDistribution classDist = getClassDistribution(wdata, weightID);
      TDiscDistribution *ddist = classDist.AS(TDiscDistribution);
      beginwe = ddist->atint(targetClass);
    }
    progressCallback->call(0.0);
  }

  while (!dataStopping || !dataStopping->call(wdata, currWeightID, targetClass)) {
    PRule rule = ruleFinder->call(wdata, currWeightID, targetClass, baseRules);
    if (!rule)
      raiseError("'ruleFinder' didn't return a rule");

    if (ruleStopping && ruleStopping->call(ruleList, rule, wdata, currWeightID))
      break;

    wdata = coverAndRemove->call(rule, wdata, currWeightID, currWeightID, targetClass);
    ruleList->push_back(rule);

    if (progressCallback) {
      if (targetClass==-1)
        currentwe = wdata->weightOfExamples(weightID);
      else {
        PDistribution classDist = getClassDistribution(wdata, currWeightID);
        TDiscDistribution *ddist = classDist.AS(TDiscDistribution);
        currentwe = ddist->atint(targetClass);
      }
      progressCallback->call(1-currentwe/beginwe);
    }
  }
  if (progressCallback)
    progressCallback->call(1.0);


  // Restore values
  if (tempDataStopping)
    dataStopping = PRuleDataStoppingCriteria();
  if (tempRuleFinder)
    ruleFinder = PRuleFinder();
  if (tempCoverAndRemove)
    coverAndRemove = PRuleCovererAndRemover();

  PRuleClassifierConstructor clConstructor =
    classifierConstructor ? classifierConstructor :
    PRuleClassifierConstructor(mlnew TRuleClassifierConstructor_firstRule());
  return clConstructor->call(ruleList, gen, weightID);
};


bool TRuleDataStoppingCriteria_NoPositives::operator()(PExampleTable data, const int &weightID, const int &targetClass) const
{
  PDistribution classDist = getClassDistribution(data, weightID);
  TDiscDistribution *ddist = classDist.AS(TDiscDistribution);

  return (targetClass >= 0 ? ddist->atint(targetClass) : ddist->abs) == 0.0;
}

bool TRuleStoppingCriteria_NegativeDistribution::operator()(PRuleList ruleList, PRule rule, PExampleTable data, const int &weightID) const
{
  if (rule && rule->classifier)
  {
    PDistribution aprioriDist = getClassDistribution(data, weightID);
    TDiscDistribution *apriori = aprioriDist.AS(TDiscDistribution);

    const TDefaultClassifier *clsf = rule->classifier.AS(TDefaultClassifier);
    if (!clsf)
      return false;
    const TDiscDistribution *dist = dynamic_cast<const TDiscDistribution *>(clsf->defaultDistribution.getUnwrappedPtr());
    const int classVal = clsf->defaultVal.intV;
    if (classVal<0 || classVal>=dist->size())
      return false;
    float acc = dist->atint(clsf->defaultVal.intV)/dist->abs;
    float accApriori = apriori->atint(clsf->defaultVal.intV)/apriori->abs;
    if (accApriori>acc)
      return true;
  }
  return false;
}


PExampleTable TRuleCovererAndRemover_Default::operator()(PRule rule, PExampleTable data, const int &weightID, int &newWeight, const int &targetClass) const
{
  TExampleTable *table = mlnew TExampleTable(data, 1);
  PExampleGenerator wtable = table;

  TFilter &filter = rule->filter.getReference();

  if (targetClass < 0)
  {
    PEITERATE(ei, data)
      if (!filter(*ei))
        table->addExample(*ei);
  }
  else
    PEITERATE(ei, data)
      if (!filter(*ei) || (*ei).getClass().intV!=targetClass)
        table->addExample(*ei);


  newWeight = weightID;
  return wtable;
}

// classifiers
PRuleClassifier TRuleClassifierConstructor_firstRule::operator ()(PRuleList rules, PExampleTable table, const int &weightID)
{
  return mlnew TRuleClassifier_firstRule(rules, table, weightID);
}


TRuleClassifier::TRuleClassifier(PRuleList arules, PExampleTable anexamples, const int &aweightID)
: rules(arules),
  examples(anexamples),
  weightID(aweightID),
  TClassifier(anexamples->domain->classVar,true)
{}

TRuleClassifier::TRuleClassifier()
: TClassifier(true)
{}


TRuleClassifier_firstRule::TRuleClassifier_firstRule(PRuleList arules, PExampleTable anexamples, const int &aweightID)
: TRuleClassifier(arules, anexamples, aweightID)
{
  prior = getClassDistribution(examples, weightID);
}

TRuleClassifier_firstRule::TRuleClassifier_firstRule()
: TRuleClassifier()
{}

PDistribution TRuleClassifier_firstRule::classDistribution(const TExample &ex)
{
  checkProperty(rules);
  checkProperty(prior);

  PITERATE(TRuleList, ri, rules) {
    if ((*ri)->call(ex))
      return (*ri)->classDistribution;
  }
  return prior;
}

void copyTable(float **dest, float **source, int nx, int ny) {
  for (int i=0; i<nx; i++)
    memcpy(dest[i],source[i],ny*sizeof(float));
}

//==============================================================================
// return 1 if system not solving
// nDim - system dimension
// pfMatr - matrix with coefficients
// pfVect - vector with free members
// pfSolution - vector with system solution
// pfMatr becames trianglular after function call
// pfVect changes after function call
//
// Developer: Henry Guennadi Levkin
//
//==============================================================================
int LinearEquationsSolving(int nDim, double* pfMatr, double* pfVect, double* pfSolution)
{
  double fMaxElem;
  double fAcc;

  int i, j, k, m;


  for(k=0; k<(nDim-1); k++) // base row of matrix
  {
    // search of line with max element
    fMaxElem = fabs( pfMatr[k*nDim + k] );
    m = k;
    for(i=k+1; i<nDim; i++)
    {
      if(fMaxElem < fabs(pfMatr[i*nDim + k]) )
      {
        fMaxElem = pfMatr[i*nDim + k];
        m = i;
      }
    }

    // permutation of base line (index k) and max element line(index m)
    if(m != k)
    {
      for(i=k; i<nDim; i++)
      {
        fAcc               = pfMatr[k*nDim + i];
        pfMatr[k*nDim + i] = pfMatr[m*nDim + i];
        pfMatr[m*nDim + i] = fAcc;
      }
      fAcc = pfVect[k];
      pfVect[k] = pfVect[m];
      pfVect[m] = fAcc;
    }

    if( pfMatr[k*nDim + k] == 0.) return 1; // needs improvement !!!

    // triangulation of matrix with coefficients
    for(j=(k+1); j<nDim; j++) // current row of matrix
    {
      fAcc = - pfMatr[j*nDim + k] / pfMatr[k*nDim + k];
      for(i=k; i<nDim; i++)
      {
        pfMatr[j*nDim + i] = pfMatr[j*nDim + i] + fAcc*pfMatr[k*nDim + i];
      }
      pfVect[j] = pfVect[j] + fAcc*pfVect[k]; // free member recalculation
    }
  }

  for(k=(nDim-1); k>=0; k--)
  {
    pfSolution[k] = pfVect[k];
    for(i=(k+1); i<nDim; i++)
    {
      pfSolution[k] -= (pfMatr[k*nDim + i]*pfSolution[i]);
    }
    pfSolution[k] = pfSolution[k] / pfMatr[k*nDim + k];
  }

  return 0;
}

// extracts class value index from target in rule
int getClassIndex(PRule r) {
  const TDefaultClassifier &cl = dynamic_cast<const TDefaultClassifier &>(r->classifier.getReference());
  return cl.defaultVal.intV;
}

// constructor 1
TLogitClassifierState::TLogitClassifierState(PRuleList arules, PExampleTable anexamples, const int &aweightID)
: rules(arules),
  examples(anexamples),
  weightID(aweightID)
{
  // initialize f, p
  f = new float *[examples->domain->classVar->noOfValues()-1];
  p = new float *[examples->domain->classVar->noOfValues()];
  int i;
  for (i=0; i<examples->domain->classVar->noOfValues()-1; i++) {
	  f[i] = new float[examples->numberOfExamples()];
	  p[i] = new float[examples->numberOfExamples()];
  }
  p[examples->domain->classVar->noOfValues()-1] = new float[examples->numberOfExamples()];

  betas = new float[rules->size()];
  priorBetas = new float[examples->domain->classVar->noOfValues()];
  isExampleFixed = new bool[examples->numberOfExamples()];
}

// constructor 2
TLogitClassifierState::TLogitClassifierState(PRuleList arules, const PDistributionList &probList, PExampleTable anexamples, const int &aweightID)
: rules(arules),
  examples(anexamples),
  weightID(aweightID)
{
  // initialize f, p
  f = new float *[examples->domain->classVar->noOfValues()-1];
  p = new float *[examples->domain->classVar->noOfValues()];
  int i, j;
  for (i=0; i<examples->domain->classVar->noOfValues()-1; i++) {
    f[i] = new float[examples->numberOfExamples()];
    p[i] = new float[examples->numberOfExamples()];
    for (j=0; j<examples->numberOfExamples(); j++) {
        f[i][j] = 0.0;
        p[i][j] = 1.0/examples->domain->classVar->noOfValues();
    }
  }
  p[examples->domain->classVar->noOfValues()-1] = new float[examples->numberOfExamples()];
  for (j=0; j<examples->numberOfExamples(); j++)
      p[examples->domain->classVar->noOfValues()-1][j] = 1.0/examples->domain->classVar->noOfValues();

   // if initial example probability is given, update F and P
  if (probList) {
    double *matrix = new double [sqr(examples->domain->classVar->noOfValues()-1)];
    double *fVals = new double [examples->domain->classVar->noOfValues()-1];
    double *results = new double [examples->domain->classVar->noOfValues()-1];
    for (i=0; i<probList->size(); i++) {
      int k1, k2;
      TDistribution *dist = mlnew TDiscDistribution(probList->at(i)->variable);
      PDistribution wdist = dist;
      // Prepare and compute expected f - values (a linear equation)
      for (k1=0; k1<examples->domain->classVar->noOfValues(); k1++) {
        if (probList->at(i)->atint(k1) >= 1.0-1e-4)
          wdist->setint(k1,(float)(1.0-1e-4));
        else if (probList->at(i)->atint(k1) <= 1e-4)
          wdist->setint(k1,(float)(1e-4));
        else
          wdist->setint(k1,probList->at(i)->atint(k1));
      }
      wdist->normalize();
      for (k1=0; k1<examples->domain->classVar->noOfValues()-1; k1++) {
        fVals[k1] = -wdist->atint(k1);
        for (k2=0; k2<examples->domain->classVar->noOfValues()-1; k2++) {
          if (k1==k2)
            matrix[k1*(examples->domain->classVar->noOfValues()-1)+k2] = wdist->atint(k1)-1;
          else
            matrix[k1*(examples->domain->classVar->noOfValues()-1)+k2] = wdist->atint(k1);
        }
      }
      LinearEquationsSolving(examples->domain->classVar->noOfValues()-1, matrix, fVals, results);
      // store values
      for (k1=0; k1<examples->domain->classVar->noOfValues()-1; k1++)
        f[k1][i] = results[k1]>0.0 ? log(results[k1]) : -10.0;
      for (k1=0; k1<examples->domain->classVar->noOfValues(); k1++)
        p[k1][i] = wdist->atint(k1);
    }
    delete [] matrix;
    delete [] fVals;
    delete [] results;
  }

  // compute rule indices
  i=0;
  ruleIndices = mlnew PIntList[rules->size()];
  {
    PITERATE(TRuleList, ri, rules) {
      TIntList *ruleIndicesnw = mlnew TIntList();
      ruleIndices[i] = ruleIndicesnw;
      j=0;
      PEITERATE(ei, examples) {
        if ((*ri)->call(*ei))
          ruleIndices[i]->push_back(j);
        j++;
      }
      i++;
    }
  }

  // set initial values of betas
  betas = new float[rules->size()];
  for (i=0; i<rules->size(); i++)
    betas[i] = (float)0.0;

  // Add default rules
  priorBetas = new float[examples->domain->classVar->noOfValues()];
  for (i=0; i<examples->domain->classVar->noOfValues(); i++)
    priorBetas[i] = 0.0;

  // computer best rule covering
  PDistribution apriori = getClassDistribution(examples, weightID);
  isExampleFixed = new bool[examples->numberOfExamples()];
  for (j=0; j<examples->numberOfExamples(); j++)
    isExampleFixed[j] = false;

  // priorProb and avgProb
  TFloatList *npriorProb = mlnew TFloatList();
  avgPriorProb = npriorProb;
  TFloatList *navgProb = mlnew TFloatList();
  avgProb = navgProb;
  TIntList *pprefixRules = mlnew TIntList();
  prefixRules = pprefixRules;
  computeAvgProbs();
  computePriorProbs();
}

TLogitClassifierState::~TLogitClassifierState()
{
  int i;
  for (i=0; i<examples->domain->classVar->noOfValues()-1; i++)
  	delete [] f[i];
  delete [] f;

  for (i=0; i<examples->domain->classVar->noOfValues(); i++)
  	delete [] p[i];
  delete [] p;
  delete [] betas;
  delete [] priorBetas;
  delete [] ruleIndices;
  delete [] isExampleFixed;
}

void TLogitClassifierState::computeAvgProbs()
{
  // compute new rule avgProbs
  avgProb->clear();
  int classInd = 0;

  float newAvgProb;
  for (int ri = 0; ri<rules->size(); ri++) {
    newAvgProb = 0.0;
    classInd = getClassIndex(rules->at(ri));
    PITERATE(TIntList, ind, ruleIndices[ri])
      newAvgProb += p[classInd][*ind];
    avgProb->push_back(newAvgProb/ruleIndices[ri]->size());
  }
}

// compute new prior probs
void TLogitClassifierState::computePriorProbs()
{
  avgPriorProb->clear();
  for (int pi=0; pi<examples->domain->classVar->noOfValues(); pi++) {
    float newPriorProb = 0.0;
    for (int ei=0; ei<examples->numberOfExamples(); ei++) {
      newPriorProb += p[pi][ei];
    }
    avgPriorProb->push_back(newPriorProb/examples->numberOfExamples());
  }
}

void TLogitClassifierState::copyTo(PLogitClassifierState & wstate)
{
  if (!wstate) {
    TLogitClassifierState *state = mlnew TLogitClassifierState(rules, examples, weightID);
    wstate = state;
    wstate->ruleIndices = mlnew PIntList[rules->size()];
    int i;
    for (i=0; i<rules->size(); i++) {
      TIntList * tIndices = mlnew TIntList(ruleIndices[i].getReference());
      wstate->ruleIndices[i] = tIndices;
    }
  }

  wstate->eval = eval;
  copyTable(wstate->f, f, examples->domain->classVar->noOfValues()-1, examples->numberOfExamples());
  copyTable(wstate->p, p, examples->domain->classVar->noOfValues(), examples->numberOfExamples());
  memcpy(wstate->betas,betas,sizeof(float)*rules->size());
  memcpy(wstate->priorBetas,priorBetas,sizeof(float)*(examples->domain->classVar->noOfValues()-1));
  memcpy(wstate->isExampleFixed, isExampleFixed, sizeof(bool)*examples->numberOfExamples());

  TFloatList *pavgProb = mlnew TFloatList(avgProb.getReference());
  TFloatList *pavgPriorProb = mlnew TFloatList(avgPriorProb.getReference());
  TIntList *pprefixRules = mlnew TIntList(prefixRules.getReference());
  wstate->avgProb = pavgProb;
  wstate->avgPriorProb = pavgPriorProb;
  wstate->prefixRules = pprefixRules;
}

void TLogitClassifierState::newBeta(int i, float b)
{
  // set new beta
  float diff = b-betas[i];
  betas[i] = b;


  // add differences to f
  int classIndex = getClassIndex(rules->at(i));
  PITERATE(TIntList, ind, ruleIndices[i])
    for (int fi=0; fi<examples->domain->classVar->noOfValues()-1; fi++)
      if (fi == classIndex)
        f[fi][*ind] += diff;
      else if (classIndex == examples->domain->classVar->noOfValues()-1)
        f[fi][*ind] -= diff;

  // compute p
  computePs(i);
  computeAvgProbs();
  computePriorProbs();
}

void TLogitClassifierState::newPriorBeta(int i, float b)
{
  // set new beta
  float diff = b-priorBetas[i];
  priorBetas[i] = b;

  // add differences to f
  for (int ei=0; ei<examples->numberOfExamples(); ei++)
    for (int fi=0; fi<examples->domain->classVar->noOfValues()-1; fi++)
      if (fi == i)
        f[fi][ei] += diff;
  computePs(-1);
  computeAvgProbs();
  computePriorProbs();
}

void TLogitClassifierState::updateExampleP(int ei)
{
/*  PITERATE(TIntList, ind, frontRules)
	  if (rules->at(*ind)->call(examples->at(ei))) {
		p[getClassIndex(rules->at(*ind))][ei] = rules->at(*ind)->quality;
		for (int ci=0; ci<examples->domain->classVar->noOfValues(); ci++)
			if (ci!=getClassIndex(rules->at(*ind)))
				p[ci][ei] = (1.0-rules->at(*ind)->quality)/(examples->domain->classVar->noOfValues()-1);
		return;
	  } */
  if (isExampleFixed[ei])
	  return;

  float sum = 1.0;
  int pi;
  for (pi=0; pi<examples->domain->classVar->noOfValues()-1; pi++) {
    if (f[pi][ei] > 10.0)
        p[pi][ei] = 22000.0;
    else
        p[pi][ei] = exp(f[pi][ei]);
    sum += p[pi][ei];
  }
  p[examples->domain->classVar->noOfValues()-1][ei] = 1.0;
  for (pi=0; pi<examples->domain->classVar->noOfValues(); pi+=1) {
    p[pi][ei] /= sum;
  }
}

void TLogitClassifierState::computePs(int beta_i)
{
  if (beta_i<0)
    for (int ei=0; ei<examples->numberOfExamples(); ei++)
      updateExampleP(ei);
  else
    PITERATE(TIntList, ind, ruleIndices[beta_i])
      updateExampleP(*ind);
}

void TLogitClassifierState::setFixed(int rule_i)
{
	PITERATE(TIntList, ind, ruleIndices[rule_i])
	  isExampleFixed[*ind] = true;
}

void TLogitClassifierState::updateFixedPs(int rule_i)
{
	PITERATE(TIntList, ind, ruleIndices[rule_i])
  {
    float bestQuality = 0.0;
		PITERATE(TIntList, fr, prefixRules) {
			  if (rules->at(*fr)->call(examples->at(*ind)) && rules->at(*fr)->quality > bestQuality) {
          bestQuality = rules->at(*fr)->quality;
				  p[getClassIndex(rules->at(*fr))][*ind] = rules->at(*fr)->quality;
				  for (int ci=0; ci<examples->domain->classVar->noOfValues(); ci++)
					  if (ci!=getClassIndex(rules->at(*fr)))
						  p[ci][*ind] = (1.0-rules->at(*fr)->quality)/(examples->domain->classVar->noOfValues()-1);
				  break;
			  }
	  }
  }
}

float TLogitClassifierState::getBrierScore()
{
  float brier = 0.0;
  for (int j=0; j<examples->numberOfExamples(); j++)
    brier += pow(1.0-p[examples->at(j).getClass().intV][j],2.0);
  return brier;
}

float TLogitClassifierState::getAUC()
{
  float auc = 0.0, sum_ranks;
  int n1, n2;
  vector<double> probs, ranks;
//  for each class
  for (int cl=0; cl < examples->domain->classVar->noOfValues() - 1; cl++)
  {
//    probs = prediction probabilities of class
    probs.clear();
    for (int j=0; j<examples->numberOfExamples(); j++)
      probs.push_back(p[cl][j]);

//    get ranks of class examples
//    n1 = elements of examples from class
//    n2 = elements of examples not from class
    rankdata(probs, ranks);
    n1=n2=0;
    sum_ranks=0.0;
    for (int j=0; j<examples->numberOfExamples(); j++)
    {
      if (examples->at(j).getClass().intV == cl)
      {
        n1++;
        sum_ranks+=ranks.at(j);
      }
      else n2++;
    }
//    auc += (sum(ranks) - n1*(n1+1)/2) / (n1*n2)
    auc += (sum_ranks - n1*(n1+1)/2) / (n1*n2);
  }
  return auc / (examples->domain->classVar->noOfValues() - 1);
}

void TLogitClassifierState::setPrefixRule(int rule_i) //, int position)
{
/*	TIntList::iterator frs(frontRules->begin()+position), fre(frontRules->end());
	while (fre > frs)
	{
		frontRules->insert(fre, *(fre-1));
		fre--;
	}

	frontRules->insert(frs, rule_i); */
	prefixRules->push_back(rule_i);
	setFixed(rule_i);
	updateFixedPs(rule_i);
	betas[rule_i] = 0.0;
  computeAvgProbs();
  computePriorProbs();
}

TRuleClassifier_logit::TRuleClassifier_logit()
: TRuleClassifier()
{}

TRuleClassifier_logit::TRuleClassifier_logit(PRuleList arules, const float &minSignificance, const float &minBeta, PExampleTable anexamples, const int &aweightID, const PClassifier &classifier, const PDistributionList &probList, bool setPrefixRules, bool optimizeBetasFlag)
: TRuleClassifier(arules, anexamples, aweightID),
  minSignificance(minSignificance),
  priorClassifier(classifier),
  setPrefixRules(setPrefixRules),
  optimizeBetasFlag(optimizeBetasFlag),
  minBeta(minBeta)
{
  initialize(probList);
  float step = 2.0;
  minStep = (float)0.01;

  // initialize prior betas

  // optimize betas
  if (optimizeBetasFlag)
      optimizeBetas();

  // find front rules
  if (setPrefixRules)
  {
	  bool changed = setBestPrefixRule();
	  while (changed) {
      if (optimizeBetasFlag)
        optimizeBetas();
		  changed = setBestPrefixRule();
	  }
  }

  // prepare results in Orange-like format
  TFloatList *aruleBetas = mlnew TFloatList();
  ruleBetas = aruleBetas;
  TRuleList *aprefixRules = mlnew TRuleList();
  prefixRules = aprefixRules;
  int i;
  for (i=0; i<rules->size(); i++)
    ruleBetas->push_back(currentState->betas[i]);
  for (i=0; i<currentState->prefixRules->size(); i++)
    prefixRules->push_back(rules->at(currentState->prefixRules->at(i)));
  delete [] skipRule;
}

bool TRuleClassifier_logit::setBestPrefixRule()
{
//Each rule should cover at least 50% of examples
//Bries score / example should be highest
//New quality could not be higher than original quality and could not be lower or equal to prior probability.

  PLogitClassifierState tempState;
  currentState->copyTo(tempState);
  int bestRuleI = -1;
  float bestImprovement = 0.0;  // improvement of brier score
  float bestNewQuality = 0.0;

  PDistribution apriori = getClassDistribution(examples, weightID);

  for (int i=0; i<rules->size(); i++) {
    // compute corrected quality
    float new_positive = 0.0, new_covered = 0.0;
    float fixed_prob = 0.0;
    float apriori_prob = apriori->atint(rules->at(i))/apriori->abs;
    int j;
    for (j=0; j<examples->numberOfExamples(); j++)
    {
        if (rules->at(i)->call(examples->at(j)) && !tempState->isExampleFixed[j])
        {
            if (getClassIndex(rules->at(i)) == examples->at(j).getClass().intV)
                new_positive ++;
            new_covered ++;
        }
        if (rules->at(i)->call(examples->at(j)) && tempState->isExampleFixed[j])
          fixed_prob += tempState->p[getClassIndex(rules->at(i))][j];
    }
    // adjust old quality with changed relative frequency
    if (!new_covered)
      continue;

    float oldQuality = rules->at(i)->quality;
    float newQuality = 0.0;
    float rf_rule = rules->at(i)->classDistribution->atint(getClassIndex(rules->at(i))) + 2*apriori_prob;
    rf_rule /= rules->at(i) -> classDistribution->abs + 2;
    float new_rf_rule = new_positive + 2*apriori_prob;
    new_rf_rule /= new_covered + 2;

    newQuality = oldQuality * new_rf_rule / rf_rule;
    float cov_perc = 1.0; //max(min((float)1.0, (float)new_covered / 30), (float) new_covered / rules->at(i)->classDistribution->abs);
    //newQuality = (oldQuality * rules->at(i)->classDistribution->abs - fixed_prob) / (new_covered);
    if (rules->at(i)->classDistribution->abs == apriori->abs)
      newQuality = new_positive / new_covered;
    if (newQuality < apriori_prob && rules->at(i)->classDistribution->abs < apriori->abs)
      newQuality = apriori_prob;
/*    if (newQuality > oldQuality * new_rf_rule / rf_rule && rules->at(i)->classDistribution->abs < apriori->abs)
      newQuality = oldQuality * new_rf_rule / rf_rule; */
    if (newQuality > oldQuality && rules->at(i)->classDistribution->abs < apriori->abs)
      newQuality = oldQuality;

 /*   if (abs(newQuality-oldQuality) > 0.1 && rules->at(i)->classDistribution->abs < apriori->abs)
      continue; */

    //printf("new: %f, old: %f\n", newQuality, oldQuality);
   // if (newQuality > oldQuality && cov_perc < 0.5)
   //   newQuality = oldQuality;
      
/*    if (new_rf_rule > rel_freq)
      if (new_covered > 30)
        newQuality = oldQuality * (new_rf_rule / rel_freq);
      else
        newQuality = oldQuality;
    else
        newQuality = oldQuality * ((new_positive / new_covered) / rel_freq); */


/*
    float fixedExamplesPerc = 0.0;
    float oldBrierScore = 0.0, newBrierScore = 0.0;

    // adjust quality of rule according to previously covered
    float oldQuality = rules->at(i)->quality;
    float sumQualityFixed = 0.0;
    int j, nFixed=0;
    for (j=0; j<examples->numberOfExamples(); j++)
        if (rules->at(i)->call(examples->at(j)) && tempState->isExampleFixed[j])
        {
            sumQualityFixed += tempState->p[examples->at(j).getClass().intV][j];
            nFixed++;
        }
    // the average predicted probability should be as original rule quality
    float requiredSum = oldQuality * rules->at(i)->classDistribution->abs;
    if (nFixed >= rules->at(i)->classDistribution->abs)
        continue;
    float newQuality = (requiredSum - sumQualityFixed) / (rules->at(i)->classDistribution->abs - nFixed);
    if (newQuality <= 0.0 || newQuality >= 1.0)
        continue; */
    rules->at(i)->quality = newQuality;
    currentState->setPrefixRule(i);
    rules->at(i)->quality = oldQuality;

    // it must cover 50% and improve brier score
    float fixedExamples = 0.0;
    float oldBrierScore = 0.0, newBrierScore = 0.0;
    float oldAbsScore = 0.0, newAbsScore = 0.0;
    float oldLLScore = 0.0, newLLScore = 0.0;
    for (j=0; j<examples->numberOfExamples(); j++)
    {
        if (rules->at(i)->call(examples->at(j)) && tempState->isExampleFixed[j])
            fixedExamples ++;
        if (rules->at(i)->call(examples->at(j)) && !tempState->isExampleFixed[j])
        {
            newAbsScore += 1.0-currentState->p[examples->at(j).getClass().intV][j];
            oldAbsScore += 1.0-tempState->p[examples->at(j).getClass().intV][j];
            newBrierScore += pow(1.0-currentState->p[examples->at(j).getClass().intV][j],2.0);
            oldBrierScore += pow(1.0-tempState->p[examples->at(j).getClass().intV][j],2.0);
            newLLScore += log(currentState->p[examples->at(j).getClass().intV][j] > 0 ? currentState->p[examples->at(j).getClass().intV][j] : 1e-6);
            oldLLScore += log(tempState->p[examples->at(j).getClass().intV][j] > 0 ? tempState->p[examples->at(j).getClass().intV][j] : 1e-6);
        }
    }
// !(fixedExamples/rules->at(i)->classDistribution->abs <= 0.5 || (rules->at(i)->classDistribution->abs-fixedExamples)>30)  || 
    // 
    if (newAbsScore >= oldAbsScore || newBrierScore >= oldBrierScore) // || newLLScore <= oldLLScore)
    {
        tempState->copyTo(currentState);
        continue;
    }

/*    if (cov_perc < 0.5 && cov_perc > bestImprovement)
    {
        bestImprovement = cov_perc;
        bestRuleI = i;
        bestNewQuality = newQuality;
    } */
    //if (cov_perc >= 0.5 && (oldAbsScore-newAbsScore)/(rules->at(i)->classDistribution->abs-fixedExamples) + 1.0 > bestImprovement )
    //if ((oldAbsScore-newAbsScore)/(rules->at(i)->classDistribution->abs-fixedExamples) > bestImprovement )
    //if (cov_perc < 0.5 && cov_perc + newQuality > bestImprovement)
    //{
    //    bestImprovement = cov_perc + newQuality; //1.0 + (oldAbsScore-newAbsScore)/(rules->at(i)->classDistribution->abs-fixedExamples);
    //    bestRuleI = i;
    //    bestNewQuality = newQuality;
    //}
    if (newQuality * cov_perc > bestImprovement)
    {
        bestImprovement = newQuality * cov_perc; //1.0 + (oldAbsScore-newAbsScore)/(rules->at(i)->classDistribution->abs-fixedExamples);
        bestRuleI = i;
        bestNewQuality = newQuality;
    }
    tempState->copyTo(currentState);
  }
  if (bestRuleI > -1)
  {
      rules->at(bestRuleI)->quality = bestNewQuality;
      currentState->setPrefixRule(bestRuleI);
      skipRule[bestRuleI] = true;
      // compute new class distribution for this rule
      TExampleTable * newexamples = mlnew TExampleTable(examples->domain);
      PExampleGenerator pnewexamples = PExampleGenerator(newexamples);
      for (int ei=0; ei<examples->numberOfExamples(); ei++)
        if (!tempState->isExampleFixed[ei])
          newexamples->addExample(examples->at(ei));
      rules->at(bestRuleI)->filterAndStore(pnewexamples, rules->at(bestRuleI)->weightID, getClassIndex(rules->at(bestRuleI)));
      return true;
  }
  return false;
}

void TRuleClassifier_logit::optimizeBetas()
{
  bool minSigChange = true;
  bool minBetaChange = true;

  while (minSigChange || minBetaChange)
  {
      // learn initial model
      updateRuleBetas(2.0);

      minSigChange = false;
      minBetaChange = false;
      for (int i=0; i<rules->size(); i++)
      {
         if (skipRule[i] || rules->at(i)->classDistribution->abs == prior->abs)
             continue;
         if (currentState->betas[i] < minBeta)
         {
             skipRule[i] = true;
             minBetaChange = true;
             currentState->newBeta(i,0.0);
         }
      }

      // min significance check, the same story as minBeta
      // loop through rules, check if significant, if not set to 0
      for (int i=0; i<rules->size(); i++)
      {
         if (skipRule[i] || rules->at(i)->classDistribution->abs == prior->abs)
             continue;
         float oldBeta = currentState->betas[i];
         currentState->newBeta(i,0.0);
         if (currentState->avgProb->at(i) < (wSatQ->at(i) - 0.01))
             currentState->newBeta(i,oldBeta);
         else
         {
             skipRule[i] = true;
             minSigChange = true;
         }
      }
  }
}

// Init current state
void TRuleClassifier_logit::initialize(const PDistributionList &probList)
{
  // compute prior distribution of learning examples
  prior = getClassDistribution(examples, weightID);
  domain = examples->domain;
  // set initial state
  TLogitClassifierState *ncurrentState = new TLogitClassifierState(rules, probList, examples, weightID);
  currentState = ncurrentState;
  // compute standard deviations of rules
  TFloatList *sd = new TFloatList();
  wsd = sd;
  TFloatList *sig = new TFloatList();
  wsig = sig;
  PITERATE(TRuleList, ri, rules) {
  	float maxDiff = (*ri)->classDistribution->atint(getClassIndex(*ri))/(*ri)->classDistribution->abs;
	  maxDiff -= (*ri)->quality;
	  wsig->push_back(maxDiff);
   float n = (*ri)->examples->numberOfExamples();
    float a = n*(*ri)->quality;
    float b = n*(1.0-(*ri)->quality);
    float expab = log(a)+log(b)-2*log(a+b)-log(a+b+1);
    //wsd->push_back(exp(0.5*expab));
    // temprorarily, wsd will be expected beta for a rule
    float rf = prior->atint(getClassIndex(*ri))/prior->abs;
    if ((*ri)->classDistribution->abs < prior->abs)
        wsd->push_back(log((*ri)->quality/(1-(*ri)->quality))-log(rf/(1-rf)));
    else if (getClassIndex(*ri) < (*ri)->examples->domain->classVar->noOfValues() - 1)
        wsd->push_back(log(rf/(1-rf)));
    else
        wsd->push_back(0.0);
   // printf("%f %f %f %f\n", n, a, b, exp(0.5*expab));
  }

  // compute satisfiable qualities
  TFloatList *satQ = new TFloatList();
  wSatQ = satQ;
  if (minSignificance >= 0.5 || minSignificance <= 0.0)
    PITERATE(TRuleList, ri, rules)
      wSatQ->push_back((*ri)->quality);
  else
  {
    float n,a,b,error,high,low,av,avf;
    PITERATE(TRuleList, ri, rules) {
      n = (*ri)->examples->numberOfExamples();
      a = n*(*ri)->quality;
      b = n*(1.0-(*ri)->quality);
      error = 1.0;
      high = 1.0;
      low = (float) 0.0;
      while (error > 0.001)
      {
        av = (high+low)/2.0;
        avf = (float)betai(double(a),double(b),double(av));
        if (avf < minSignificance)
          low = av;
        else
          high = av;
        error = abs(avf-minSignificance);
      }
      wSatQ->push_back(av);
    }
  }

  // Compute average example coverage and set index of examples covered by rule
  float **coverages = new float* [examples->domain->classVar->noOfValues()];
  int j=0,k=0;
  for (j=0; j<examples->domain->classVar->noOfValues(); j++)  {
    coverages[j] = new float[examples->numberOfExamples()];
    for (k=0; k<examples->numberOfExamples(); k++)
      coverages[j][k] = 0.0;
  }
  int i=0;
  {
    PITERATE(TRuleList, ri, rules) {
      j=0;
      PEITERATE(ei, examples) {
        if ((*ri)->call(*ei)) {
          //int vv = (*ei).getClass().intV;
		      //if ((*ei).getClass().intV == getClassIndex(*ri))
			    coverages[getClassIndex(*ri)][j] += 1.0;
        }
	      j++;
      }
      i++;
    }
  }

  // compute coverages of rules
  TFloatList *avgCov = new TFloatList();
  wavgCov = avgCov;
  for (i=0; i<rules->size(); i++) {
    float newCov = 0.0;
    float counter = 0.0;
    PITERATE(TIntList, ind, currentState->ruleIndices[i])
    {
      //if (getClassIndex(rules->at(i)) == examples->at(*ind).getClass().intV) {
      newCov += coverages[getClassIndex(rules->at(i))][*ind];
      counter++;
      //}
    }
    //printf(" ... %f ... %f \n", newCov, counter);
    if (counter) {
      wavgCov->push_back(newCov/counter);
    }
    else
      wavgCov->push_back(0.0);
  }

  skipRule = new bool [rules->size()];
  for (i=0; i<rules->size(); i++)
      skipRule[i] = false;
}

// betas update
/*void TRuleClassifier_logit::updateRuleBetas(float step)
{
  stabilizeAndEvaluate(step,-1);
  PLogitClassifierState finalState;
  currentState->copyTo(finalState);

  bool changed = true;
  while (changed)
  {
    changed = false;  

    // loop through rules to find the biggest problem
    int worstI = -1;
    float worstSig = 1000;
    for (int i=0; i<rules->size(); i++) {
        if (currentState->avgProb->at(i) >= rules->at(i)->quality)
            continue;
        if (skipRule[i])
            continue;

        float under_estimate = (rules->at(i)->quality - currentState->avgProb->at(i));//*rules->at(i)->classDistribution->abs;
        // if under estimate error is big enough
        if (under_estimate > 0.01) {
            // is the difference between estimated quality and relative frequency lower than current worstEval?
            //if (wsig->at(i) < worstSig)
            if (currentState->betas[i] - wsd->at(i) < worstSig)
            {
                //worstSig = wsig->at(i);
                worstSig = currentState->betas[i] - wsd->at(i);
                worstI = i;
            }
        }
    }
    if (worstI > -1)
    {
        bool impr_curr = false;
        float curr_step = step;
        while (!impr_curr)
        {
            currentState->newBeta(worstI,currentState->betas[worstI]+curr_step);
            if (currentState->avgProb->at(worstI) >= rules->at(worstI)->quality)
            {
                finalState->copyTo(currentState);
                curr_step /= 2.0;
            }
            else
            {
              stabilizeAndEvaluate(curr_step,-1);
              currentState->copyTo(finalState);
              changed = true;
              impr_curr = true;
            }
        }
    }
  }

  finalState->copyTo(currentState);
}*/

void TRuleClassifier_logit::updateRuleBetas_old(float step_)
{
  /* for (int i=0; i<rules->size(); i++) {
    if (skipRule[i])
        continue;
    currentState->newBeta(i,wsd->at(i));
  } */
  stabilizeAndEvaluate(step_,-1);
  PLogitClassifierState finalState;
  currentState->copyTo(finalState);

  float step = 2.0;
  bool changed;
  float worst_underestimate, underestimate;
  int worst_rule_index;
  while (step > 0.001)
  {
      step /= 2;
      changed = true;
      while (changed)
      {
        changed = false;
        worst_underestimate = (float)0.01;
        worst_rule_index = -1;
        // find rule with greatest underestimate in probability
        for (int i=0; i<rules->size(); i++) {
            if (currentState->avgProb->at(i) >= rules->at(i)->quality)
                continue;
            if (skipRule[i])
                continue;

            underestimate = (rules->at(i)->quality - currentState->avgProb->at(i));//*rules->at(i)->classDistribution->abs;
            // if under estimate error is big enough
            if (underestimate > worst_underestimate)
            {
                worst_underestimate = underestimate;
                worst_rule_index = i;
            }
        }
        if (worst_rule_index > -1)
        {
            currentState->newBeta(worst_rule_index,currentState->betas[worst_rule_index]+step);
            if (currentState->avgProb->at(worst_rule_index) > rules->at(worst_rule_index)->quality)
            {
                finalState->copyTo(currentState);
            }
            else
            {
              stabilizeAndEvaluate(step,-1);
              currentState->copyTo(finalState);
              changed = true;
            }
        }
      }
  }
  finalState->copyTo(currentState);
}

void TRuleClassifier_logit::updateRuleBetas(float step_)
{

  stabilizeAndEvaluate(step_,-1);
  PLogitClassifierState finalState, tempState;
  currentState->copyTo(finalState);

  float step = 2.0;
  int changed;
  float worst_underestimate, underestimate;
  float auc = currentState->getAUC();
  float brier = currentState->getBrierScore();
  float temp_auc, temp_brier;
  int worst_rule_index;
  vector<double> underest;
  vector<int> indices;
  while (step > 0.001)
  {
      step /= 2;
      changed = 0;
//      printf("brier = %4.2f, auc = %4.2f,step = %4.2f\n", brier, auc, step);
      while (changed < 100)
      {
        changed = 0;
        worst_underestimate = (float)0.01;
        worst_rule_index = -1;
        underest.clear();
        indices.clear();
        // find rule with greatest underestimate in probability
        for (int i=0; i<rules->size(); i++) {
            if (currentState->avgProb->at(i) >= rules->at(i)->quality)
                continue;
            if (skipRule[i])
                continue;

            underestimate = (rules->at(i)->quality - currentState->avgProb->at(i));//*rules->at(i)->classDistribution->abs;
            // if under estimate error is big enough
            if (underestimate > worst_underestimate)
            {
                worst_underestimate = underestimate;
                worst_rule_index = i;
            }
            if (underestimate > 0.01)
            {
              // insert and keep list ordered
              int ins;
              for (ins=0; ins<underest.size(); ins++)
                if (underest.at(ins) < underestimate)
                  break;
              underest.insert(underest.begin() + ins, underestimate);
              indices.insert(indices.begin() + ins, i);
            }
        }
        if (worst_rule_index > -1)
        {
/*          int i;
          for (i=0; i<indices.size(); i++)
          {
            currentState->copyTo(tempState);
            currentState->newBeta(indices.at(i),currentState->betas[indices.at(i)]+step);
            if (currentState->avgProb->at(indices.at(i)) > rules->at(indices.at(i))->quality)
            {
              tempState->copyTo(currentState);
              continue;
            }
            else {
              stabilizeAndEvaluate(step,-1);
              temp_auc = currentState->getAUC();
              temp_brier = currentState->getBrierScore();
              printf("TEMP: brier = %4.2f, auc = %4.2f\n", temp_brier, temp_auc);
              if (temp_auc >= auc && temp_brier < brier)
              {
                currentState->copyTo(finalState);
                changed = 0;
                auc = temp_auc;
                brier = temp_brier;
                break;
              }
              else
              {
                tempState->copyTo(currentState);
              }
            }
          } 
          if (i == indices.size())
          {*/
            currentState->newBeta(worst_rule_index,currentState->betas[worst_rule_index]+step);
            if (currentState->avgProb->at(worst_rule_index) > rules->at(worst_rule_index)->quality)
            {
                finalState->copyTo(currentState);
                changed = 100;
            }
            else
            {
              stabilizeAndEvaluate(step,-1);
              temp_auc = currentState->getAUC();
              temp_brier = currentState->getBrierScore();
              if (temp_auc >= auc && temp_brier < brier)
              {
                currentState->copyTo(finalState);
                changed = 0;
                auc = temp_auc;
                brier = temp_brier;
              }
              else
                changed ++;
            }
         // }
        }
        else
        {
          changed = 100;
          finalState->copyTo(currentState);
        }
      }
  }
  finalState->copyTo(currentState);
}

/*void TRuleClassifier_logit::stabilizeAndEvaluate(float & step, int last_changed_rule_index)
{
    PLogitClassifierState tempState;
    currentState->copyTo(tempState);
    bool changed = true;
    while (changed)
    {
        changed = false;
        // first find problematic rule with highest difference between quality and relative frequency
        int worstI = -1;
        float worstSig = -1000.0;
        for (int i=0; i<rules->size(); i++)
        {
            if (currentState->avgProb->at(i) > (rules->at(i)->quality + 0.01) && currentState->betas[i] > 0.0 
                && currentState->betas[i] - wsd->at(i) > worstSig && i != last_changed_rule_index)
            {
                worstSig = currentState->betas[i] - wsd->at(i);
                worstI = i;
            }
        }
        if (worstI > -1)
        {
            float curr_step = step;
            bool impr_curr = false;
            while (!impr_curr)
            {
                float new_beta = currentState->betas[worstI]-curr_step > 0 ? currentState->betas[worstI]-curr_step : 0.0;
                currentState->newBeta(worstI,new_beta);
                if (currentState->avgProb->at(worstI) < rules->at(worstI)->quality)
                {
                    tempState->copyTo(currentState);
                    curr_step /= 2;
                }
                else
                {
                    currentState->copyTo(tempState);
                    impr_curr = true;
                    changed = true;
                }
            }
        }

        for (int i=0; i<rules->size(); i++)
        {
            if (i == last_changed_rule_index)
                continue;
            // if optimistic, decrease beta
            if ( (currentState->avgProb->at(i) > rules->at(i)->quality) && (currentState->betas[i]-step)>=0.0)
            {
                currentState->newBeta(i,currentState->betas[i]-step);
                if (currentState->avgProb->at(i) < rules->at(i)->quality)
                {
                    tempState->copyTo(currentState);
                }
                else
                {
                    currentState->copyTo(tempState);
                    changed = true;
                }
            }
        } 
    }
}*/


void TRuleClassifier_logit::stabilizeAndEvaluate(float & step, int last_changed_rule_index)
{
    PLogitClassifierState tempState;
    currentState->copyTo(tempState);
    bool changed = true;
    while (changed)
    {
        changed = false;
        for (int i=0; i<rules->size(); i++)
        {
            if (currentState->avgProb->at(i) > (rules->at(i)->quality + 0.01) && currentState->betas[i] > 0.0 &&
                i != last_changed_rule_index)
            {
                float new_beta = currentState->betas[i]-step > 0 ? currentState->betas[i]-step : 0.0;
                currentState->newBeta(i,new_beta);
                if (currentState->avgProb->at(i) < rules->at(i)->quality + 1e-6)
                {
                    tempState->copyTo(currentState);
                }
                else
                {
                    currentState->copyTo(tempState);
                    changed = true;
                }
            }
        }
    }
}


void TRuleClassifier_logit::addPriorClassifier(const TExample &ex, double * priorFs) {
  // initialize variables
  double *matrix = new double [sqr(examples->domain->classVar->noOfValues()-1)];
  double *fVals = new double [examples->domain->classVar->noOfValues()-1];
  double *results = new double [examples->domain->classVar->noOfValues()-1];
  int k1, k2;
  TDistribution *dist = mlnew TDiscDistribution(domain->classVar);
  PDistribution wdist = dist;

  PDistribution classifierDist = priorClassifier->classDistribution(ex);
  // correct probablity if equals 1.0
  for (k1=0; k1<examples->domain->classVar->noOfValues(); k1++) {
    if (classifierDist->atint(k1) >= 1.0-1e-4)
      wdist->setint(k1,(float)(1.0-1e-4));
    else if (classifierDist->atint(k1) <= 1e-4)
      wdist->setint(k1,(float)(1e-4));
    else
      wdist->setint(k1,classifierDist->atint(k1));
  }
  wdist->normalize();


  // create matrix
  for (k1=0; k1<examples->domain->classVar->noOfValues()-1; k1++) {
    fVals[k1] = -wdist->atint(k1);
    for (k2=0; k2<examples->domain->classVar->noOfValues()-1; k2++) {
      if (k1==k2)
        matrix[k1*(examples->domain->classVar->noOfValues()-1)+k2] = (wdist->atint(k1)-1);
      else
        matrix[k1*(examples->domain->classVar->noOfValues()-1)+k2] = wdist->atint(k1);
    }
  }
  // solve equation
  LinearEquationsSolving(examples->domain->classVar->noOfValues()-1, matrix, fVals, results);
  for (k1=0; k1<examples->domain->classVar->noOfValues()-1; k1++)
    priorFs[k1] = results[k1]>0.0 ? log(results[k1]) : -10.0;
  // clean up
  delete [] matrix;
  delete [] fVals;
  delete [] results;
}

PDistribution TRuleClassifier_logit::classDistribution(const TExample &ex)
{
  checkProperty(rules);
  checkProperty(prior);
  checkProperty(domain);
  TExample cexample(domain, ex);

  TDiscDistribution *dist = mlnew TDiscDistribution(domain->classVar);
  PDistribution res = dist;

  // if front rule triggers, use it first
  bool foundPrefixRule = false;
  float bestQuality = 0.0;
  PITERATE(TRuleList, rs, prefixRules) {
	  if ((*rs)->call(ex) && (*rs)->quality > bestQuality) {
      bestQuality = (*rs)->quality;
		  dist->setint(getClassIndex(*rs),(*rs)->quality);
		  for (int ci=0; ci<examples->domain->classVar->noOfValues(); ci++)
		    if (ci!=getClassIndex(*rs))
			    dist->setint(ci,(1.0-(*rs)->quality)/(examples->domain->classVar->noOfValues()-1));
      foundPrefixRule = true;
      break;
	  }
  }
  if (foundPrefixRule)
    return dist;

  // if correcting a classifier, use that one first then
  double * priorFs = new double [examples->domain->classVar->noOfValues()-1];
  if (priorClassifier)
    addPriorClassifier(ex, priorFs);
  else
    for (int k=0; k<examples->domain->classVar->noOfValues()-1; k++)
      priorFs[k] = 0.0;
  // compute return probabilities
  for (int i=0; i<res->noOfElements()-1; i++) {
    float f = priorFs[i];
    TFloatList::const_iterator b(ruleBetas->begin()), be(ruleBetas->end());
    TRuleList::iterator r(rules->begin()), re(rules->end());
    for (; r!=re; r++, b++)
      if ((*r)->call(cexample)) {
        if (getClassIndex(*r) == i)
  		    f += (*b);
        else if (getClassIndex(*r) == res->noOfElements()-1)
          f -= (*b);
      }
    dist->addint(i,exp(f));
  }
  dist->addint(res->noOfElements()-1,1.0);
  dist->normalize();

  delete [] priorFs;
  return res;
}
