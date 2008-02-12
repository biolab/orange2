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
DEFINE_TOrangeVector_classDescription(PEVCDist, "TEVCDistList", true, ORANGE_API)

TRule::TRule()
: weightID(0),
  quality(ILLEGAL_FLOAT),
  complexity(ILLEGAL_FLOAT),
  coveredExamples(NULL),
  coveredExamplesLength(-1),
  parentRule(NULL),
  chi(0.0),
  requiredConditions(0)
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
  requiredConditions(0)
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

  if (learner) {
    classifier = learner->call(examples,wei);
  }
  else if (targetClass>=0)
    classifier = mlnew TDefaultClassifier(gen->domain->classVar, TValue(targetClass), classDistribution);
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



TRuleValidator_LRS::TRuleValidator_LRS(const float &a, const float &min_coverage, const float &max_rule_complexity, const float &min_quality)
: alpha(a),
  min_coverage(min_coverage),
  max_rule_complexity(max_rule_complexity),
  min_quality(min_quality)
{}

bool TRuleValidator_LRS::operator()(PRule rule, PExampleTable, const int &, const int &targetClass, PDistribution apriori) const
{
  const TDiscDistribution &obs_dist = dynamic_cast<const TDiscDistribution &>(rule->classDistribution.getReference());
  if (!obs_dist.cases)
    return false;
  
  if (obs_dist.cases < min_coverage)
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

  float p = (targetClass < obs_dist.size()) ? obs_dist[targetClass]-0.5 : 1e-5;
  const float P = (targetClass < exp_dist.size()) && (exp_dist[targetClass] > 0.0) ? exp_dist[targetClass] : 1e-5;

  if (p/obs_dist.abs < P/exp_dist.abs)
    return 0.0;

  float n = obs_dist.abs - p;
  float N = exp_dist.abs - P;

  if (N<=0.0)
    N = 1e-6f;
  if (p<=0.0)
    p = 1e-6f;
  if (n<=0.0)
    n = 1e-6f;

  float lrs = 2 * (p*log(p/obs_dist.abs) + n*log(n/obs_dist.abs) +
                   (P-p)*log((P-p)/(exp_dist.abs-obs_dist.abs)) + (N-n)*log((N-n)/(exp_dist.abs-obs_dist.abs)) -
                   P*log(P/exp_dist.abs)-N*log(N/exp_dist.abs));
  if (storeRules) {
    TRuleList &rlist = rules.getReference();
    rlist.push_back(rule);
  }
  return lrs;
}


TEVCDist::TEVCDist(const float & mu, const float & beta, PFloatList & percentiles) 
: mu(mu),
  beta(beta),
  percentiles(percentiles)
{}

TEVCDist::TEVCDist() 
{}

double TEVCDist::getProb(const float & chi)
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
      return (0.95-i*0.1)-0.1*(chi-a)/(b-a);
  }
  return 1.0;
}

float TEVCDist::median()
{
  if (!percentiles || percentiles->size()==0)
    return mu + beta*0.36651292; // log(log(2))
  return (percentiles->at(4)+percentiles->at(5))/2;
}

TEVCDistGetter_Standard::TEVCDistGetter_Standard(PEVCDistList dists) 
: dists(dists)
{}

TEVCDistGetter_Standard::TEVCDistGetter_Standard()
{}

PEVCDist TEVCDistGetter_Standard::operator()(const PRule, const int & parentLength, const int & length) const
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
  float pn = p+n;
  if (p/(p+n) == P/(P+N))
    return 0.0;
  else if (p/(p+n) < P/(P+N)) {
    p = p+0.5;
    if (p>(p+n)*P/(P+N))
      p = (p+n)*P/(P+N);
    n = pn-p;
  }
  else {
    p = p - 0.5;
    if (p<(p+n)*P/(P+N))
      p = (p+n)*P/(P+N);
    n = pn-p;
  }
  return 2*(p*log(p/(p+n))+n*log(n/(p+n))+(P-p)*log((P-p)/(P+N-p-n))+(N-n)*log((N-n)/(P+N-p-n))-P*log(P/(P+N))-N*log(N/(P+N)));
}

// 2 log likelihood with Yates' correction
float TChiFunction_2LOGLR::operator()(PRule rule, PExampleTable data, const int & weightID, const int & targetClass, PDistribution apriori, float & nonOptimistic_Chi) const
{
  nonOptimistic_Chi = 0.0;
  if (!rule->classDistribution->abs || apriori->abs == rule->classDistribution->abs)
    return 0.0;
  return getChi(rule->classDistribution->atint(targetClass),
                rule->classDistribution->abs - rule->classDistribution->atint(targetClass),
                apriori->atint(targetClass),
                apriori->abs - apriori->atint(targetClass));
}



TRuleEvaluator_mEVC::TRuleEvaluator_mEVC(const int & m, PChiFunction chiFunction, PEVCDistGetter evcDistGetter, PVariable probVar, PRuleValidator validator, const int & min_improved, const float & min_improved_perc)
: m(m),
  chiFunction(chiFunction),
  evcDistGetter(evcDistGetter),
  probVar(probVar),
  validator(validator),
  min_improved(min_improved),
  min_improved_perc(min_improved_perc),
  bestRule(NULL),
  ruleAlpha(1.0),
  attributeAlpha(1.0)
{}

TRuleEvaluator_mEVC::TRuleEvaluator_mEVC()
: m(0),
  chiFunction(NULL),
  evcDistGetter(NULL),
  probVar(NULL),
  validator(NULL),
  min_improved(1),
  min_improved_perc(0),
  bestRule(NULL),
  ruleAlpha(1.0),
  attributeAlpha(1.0)
{}

void TRuleEvaluator_mEVC::reset()
{
  bestRule = NULL;
}

LNLNChiSq::LNLNChiSq(PEVCDist evc, const float & chi)
: evc(evc),
  chi(chi)
{
  extremeAlpha = evc->getProb(chi);
  if (extremeAlpha < 0.05)
    extremeAlpha = 0.0;
}

double LNLNChiSq::operator()(float chix) {
    if (chix<=0.0)
        return 100.0;
    double chip = chisqprob((double)chix,1.0); // in statc
    if (extremeAlpha > 0.0)
        return chip-extremeAlpha;
    if (chip<=0.0 && (evc->mu-chi)/evc->beta < -100)
        return 0.0;
    if (chip<=0.0)
        return -100.0;
    if (chip < 1e-6)
        return log(chip)-(evc->mu-chi)/evc->beta;
    return log(-log(1-chip))-(evc->mu-chi)/evc->beta;
}

LRInv::LRInv(PRule rule, PExampleTable examples, const int & weightID, const int & targetClass, PDistribution apriori, PChiFunction chiFunction, float chiCorrected)
: examples(examples),
  weightID(weightID),
  targetClass(targetClass),
  apriori(apriori),
  chiFunction(chiFunction),
  chiCorrected(chiCorrected)
{
  TRule &rrule = rule.getReference();
  TRule *ntempRule = mlnew TRule(rrule, false);
  tempRule = ntempRule;
  N = rule->classDistribution->abs;

  tempRule->classDistribution = mlnew TDiscDistribution(examples->domain->classVar);
}

double LRInv::operator()(float p){
  // check how it is done in call
    tempRule->classDistribution->setint(targetClass, p);
    tempRule->classDistribution->abs = N;

    return chiFunction->call(tempRule, examples, weightID, targetClass, apriori, nonOptimistic_Chi) - chiCorrected;
}

// Implementation of Brent's root finding method.
float brent(const float & minv, const float & maxv, const int & maxsteps, DiffFunc * func) 
{
    float a = minv;
    float b = maxv;
    float fa = func->call(a);
    float fb = func->call(b);
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
        if (abs(a-b)<0.01 && fa*fb<0)
            return (a+b)/2.;
        if (fb*fa>0 || b>maxv || b<minv)
            return 0.0;
        if ((b>0.1 && fb*func->call(b-0.1)<=0) || fb*func->call(b+0.1)<=0)
            return b;
        if (counter>maxsteps)
            return 0.0;
    }
}

/* The method validates rule's attributes' significance with respect to its extreme value corrected distribution. */
bool TRuleEvaluator_mEVC::ruleAttSignificant(PRule rule, PExampleTable examples, const int & weightID, const int &targetClass, PDistribution apriori, float & aprioriProb)
{
  TFilter_values *filter;
  if (rule->valuesFilter)
    filter = rule->valuesFilter.AS(TFilter_values);
  else
    filter = rule->filter.AS(TFilter_values);
  int rLength = rule->complexity;
  PEVCDist evc = evcDistGetter->call(rule, rLength-1, rLength);
  
  // Should classical LRS be used, or EVC corrected?
  bool useClassicLRS = false;
  if (evc->mu < 1.0)
    useClassicLRS = true;

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
        LNLNChiSq *diffFunc = new LNLNChiSq(evc,chi);
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

float TRuleEvaluator_mEVC::evaluateRule(PRule rule, PExampleTable examples, const int & weightID, const int &targetClass, PDistribution apriori, const int & rLength, const float & aprioriProb) const
{
  PEVCDist evc = evcDistGetter->call(rule, 0, rLength);
  if (!evc || evc->mu < 0.0)
    return -10e+6;
  if (evc->mu == 0.0 || rLength == 0) {
    rule->chi = getChi(rule->classDistribution->atint(targetClass),
                rule->classDistribution->abs - rule->classDistribution->atint(targetClass),
                apriori->atint(targetClass),
                apriori->abs - apriori->atint(targetClass));
    return (rule->classDistribution->atint(targetClass)+m*aprioriProb)/(rule->classDistribution->abs+m);
  }
  PEVCDist evc_inter = evcDistGetter->call(rule, 0, 0);
  float rule_acc = rule->classDistribution->atint(targetClass)/rule->classDistribution->abs;

  // if accuracy of rule is worse than prior probability
  if (rule_acc < aprioriProb)
    return rule_acc - 0.01;

  // correct chi square
  float nonOptimistic_Chi = 0.0;
  float chi = chiFunction->call(rule, examples, weightID, targetClass, apriori, nonOptimistic_Chi);

  float median = evc->median();
  float chiCorrected = nonOptimistic_Chi;

  if (chi<=median || (evc->mu-chi)/evc->beta < -100) {
    rule->chi = getChi(rule->classDistribution->atint(targetClass),
                rule->classDistribution->abs - rule->classDistribution->atint(targetClass),
                apriori->atint(targetClass),
                apriori->abs - apriori->atint(targetClass));
    if (chi <= median)
      return aprioriProb-0.01;
    else
      return (rule->classDistribution->atint(targetClass)+m*aprioriProb)/(rule->classDistribution->abs+m);
  }

  // correct chi
  LNLNChiSq *diffFunc = new LNLNChiSq(evc,chi);
  chiCorrected += brent(0.0,chi,100, diffFunc);
  delete diffFunc;

  // remove inter-length optimism
  chiCorrected -= evc_inter->mu;
  rule->chi = chiCorrected;
  // compute expected number of positive examples
  float ePositives = 0.0;
  if (chiCorrected > 0.0)
  {
    LRInv *diffFunc = new LRInv(rule, examples, weightID, targetClass, apriori, chiFunction, chiCorrected);
    ePositives = brent(rule->classDistribution->abs*aprioriProb, rule->classDistribution->atint(targetClass), 100, diffFunc);
    delete diffFunc;
  }

  // compute true chi (with e positives)
  rule->chi = getChi(ePositives, rule->classDistribution->abs - ePositives,
                     apriori->atint(targetClass), apriori->abs - apriori->atint(targetClass));

  float quality = (ePositives + m*aprioriProb)/(rule->classDistribution->abs+m);

  if (quality > aprioriProb)
    return quality;
  return aprioriProb-0.01;
}

float TRuleEvaluator_mEVC::operator()(PRule rule, PExampleTable examples, const int & weightID, const int &targetClass, PDistribution apriori)
{
  rule->chi = 0.0;
  if (!rule->classDistribution->abs || !rule->classDistribution->atint(targetClass))
    return 0;

  // evaluate rule
  int rLength = rule->complexity;
  float aprioriProb = apriori->atint(targetClass)/apriori->abs;

  rule->quality = evaluateRule(rule,examples,weightID,targetClass,apriori,rLength,aprioriProb);
  if (rule->quality < 0.0)
    return rule->quality;
  if (!probVar)
    return rule->quality;

  // get rule's probability coverage
  int improved = 0;
  PEITERATE(ei, rule->examples)
    if ((*ei).getClass().intV == targetClass && rule->quality > (*ei)[probVar].floatV)
      improved ++;

  // compute future quality
  float futureQuality = 0.0;
  if (rule->classDistribution->atint(targetClass) == rule->classDistribution->abs)
    futureQuality = -1.0;
  else {
    PDistribution oldRuleDist = rule->classDistribution;
    float rulesTrueChi = rule->chi;
    rule->classDistribution = mlnew TDiscDistribution(examples->domain->classVar);
    rule->classDistribution->setint(targetClass, oldRuleDist->atint(targetClass));
    rule->classDistribution->abs = rule->classDistribution->atint(targetClass);
    float bestQuality = evaluateRule(rule,examples,weightID,targetClass,apriori,rLength+1,aprioriProb);
    rule->classDistribution = oldRuleDist;
    rule->chi = rulesTrueChi;
    if (bestQuality <= rule->quality)
      futureQuality = -1;
    else if (bestRule && bestQuality <= bestRule->quality)
      futureQuality = -1;
    else {
      futureQuality = 0.0;
      PEITERATE(ei, rule->examples) {
        if ((*ei).getClass().intV != targetClass)
          continue; 
        if (rule->quality >= (*ei)[probVar].floatV)
          futureQuality += pow(bestQuality-(*ei)[probVar].floatV,2);
        else if (bestQuality > (*ei)[probVar].floatV)
          futureQuality += pow(bestQuality-(*ei)[probVar].floatV,3)/(bestQuality-rule->quality);
      }
      futureQuality /= rule->classDistribution->abs;
    }
  }
  // store best rule and return result
  if (improved >= min_improved && 
      improved/rule->classDistribution->atint(targetClass) > min_improved_perc &&
      rule->quality > aprioriProb && 
      (!bestRule || (rule->quality>bestRule->quality)) &&
      (!validator || validator->call(rule, examples, weightID, targetClass, apriori))) {
      
      TRule *pbestRule = new TRule(rule.getReference(), true);
      PRule wpbestRule = pbestRule;

      // check if rule is significant enough
      bool ruleGoodEnough = true;
      if (ruleAlpha < 1.0) 
        ruleGoodEnough = ruleGoodEnough & ((rule->chi > 0.0) && (chisqprob(rule->chi, 1.0f) <= ruleAlpha));
      if (attributeAlpha < 1.0) 
        ruleGoodEnough = ruleGoodEnough & ruleAttSignificant(rule, examples, weightID, targetClass, apriori, aprioriProb);
      if (ruleGoodEnough)
      {
        bestRule = wpbestRule;
        futureQuality = 1.0+rule->quality;
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
        newRule->filterAndStore(data,weightID,targetClass);
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
     ubestRule->complexity = 0;
  }

  return wruleList;
}


PRuleList TRuleBeamRefiner_Selector::operator()(PRule wrule, PExampleTable data, const int &weightID, const int &targetClass)
{
  if (!discretization) {
    discretization = mlnew TEntropyDiscretization();
    dynamic_cast<TEntropyDiscretization *>(discretization.getUnwrappedPtr())->forceAttribute = true;
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
        if ((*ni)->quality >= bestRule->quality && (!validator || validator->call(*ni, data, weightID, targetClass, apriori)))
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
	  betas[i] = 0.0;

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
  TIntList *pfrontRules = mlnew TIntList();
  frontRules = pfrontRules;
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
  TIntList *pfrontRules = mlnew TIntList(frontRules.getReference());
  wstate->avgProb = pavgProb;
  wstate->avgPriorProb = pavgPriorProb;
  wstate->frontRules = pfrontRules;
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
      else
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
      else
        f[fi][ei] -= diff;
  // compute p
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
    p[pi][ei] = exp(f[pi][ei]);
    sum += p[pi][ei];
  }
  p[examples->domain->classVar->noOfValues()-1][ei] = 1.0;
  for (pi=0; pi<examples->domain->classVar->noOfValues(); pi+=1)
    p[pi][ei] /= sum;
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
		PITERATE(TIntList, fr, frontRules) {
			  if (rules->at(*fr)->call(examples->at(*ind)) && rules->at(*fr)->quality > bestQuality) {
          bestQuality = rules->at(*fr)->quality;
				  p[getClassIndex(rules->at(*fr))][*ind] = rules->at(*fr)->quality;
				  for (int ci=0; ci<examples->domain->classVar->noOfValues(); ci++) 
					  if (ci!=getClassIndex(rules->at(*fr)))
						  p[ci][*ind] = (1.0-rules->at(*fr)->quality)/(examples->domain->classVar->noOfValues()-1);
//				  break;
			  }
	  }
  }
}

void TLogitClassifierState::setFrontRule(int rule_i) //, int position)
{
/*	TIntList::iterator frs(frontRules->begin()+position), fre(frontRules->end());
	while (fre > frs)
	{
		frontRules->insert(fre, *(fre-1));
		fre--;
	}

	frontRules->insert(frs, rule_i); */
	frontRules->push_back(rule_i);
	setFixed(rule_i);
	updateFixedPs(rule_i);
	betas[rule_i] = 0.0;
  computeAvgProbs();
  computePriorProbs();
}

TRuleClassifier_logit::TRuleClassifier_logit()
: TRuleClassifier()
{}

TRuleClassifier_logit::TRuleClassifier_logit(PRuleList arules, const float &minSignificance, PExampleTable anexamples, const int &aweightID, const PClassifier &classifier, const PDistributionList &probList, const int & priorBetaType)
: TRuleClassifier(arules, anexamples, aweightID),
  minSignificance(minSignificance),
  priorClassifier(classifier),
  priorBetaType(priorBetaType)
{
  initialize(probList);

  float step = 2.0;
  bool setFrontRules = false;
  minStep = (float)0.01;

  // optimize betas
  optimizeBetas();
  
  // find front rules
  if (setFrontRules) 
  {
    PLogitClassifierState oldState;
	  currentState->copyTo(oldState);
	  setBestFrontRule();
	  if (currentState->eval > oldState->eval)
		  optimizeBetas(); 
	  while (currentState->eval > oldState->eval) {
		  currentState->copyTo(oldState);
		  setBestFrontRule();
		  if (currentState->eval > oldState->eval)
			  optimizeBetas(); 
	  }
	  oldState->copyTo(currentState);
  }

  // prepare results in Orange-like format
  TFloatList *aruleBetas = mlnew TFloatList();
  ruleBetas = aruleBetas;
  TFloatList *apriorProbBetas = mlnew TFloatList();
  priorProbBetas = apriorProbBetas;
  TRuleList *afrontRules = mlnew TRuleList();
  frontRules = afrontRules;
  int i;
  for (i=0; i<rules->size(); i++)
    ruleBetas->push_back(currentState->betas[i]);
  for (i=0; i<examples->domain->classVar->noOfValues()-1; i++)
    priorProbBetas->push_back(currentState->priorBetas[i]);
  for (i=0; i<currentState->frontRules->size(); i++)
    frontRules->push_back(rules->at(currentState->frontRules->at(i)));
}

void TRuleClassifier_logit::setBestFrontRule()
{
  PLogitClassifierState tempState;
	currentState->copyTo(tempState);
	int bestRuleI = -1;
	float bestEvaluation = 0.0; //currentState->eval;
	for (int i=0; i<rules->size(); i++) {
    if (currentState->betas[i] < 0.001)
      continue;
    bool hasFixedExamples = false;
    for (int j=0; j<examples->numberOfExamples(); j++)
    {
      if (rules->at(i)->call(examples->at(j)) && currentState->isExampleFixed[j] && getClassIndex(rules->at(i))!= examples->at(j).getClass().intV) {
        hasFixedExamples = true;
        break;
      }
      if (rules->at(i)->call(examples->at(j)) && !currentState->isExampleFixed[j] && getClassIndex(rules->at(i))!= examples->at(j).getClass().intV && 
          currentState->p[examples->at(j).getClass().intV][j] > (1.0-rules->at(i)->quality)) {
        hasFixedExamples = true;
        break;
      }
    }
    if (hasFixedExamples)
      continue;
    currentState->setFrontRule(i);
		evaluate();
    // first: classic evaluation improves?
		if ((currentState->eval-tempState->eval)/rules->at(i)->examples->numberOfExamples() > bestEvaluation)
    {
      // penalized evaluation? where some of positive examples are turned into negative still improves?
      float positives = 0.0, negatives = 0.0;
      float worstP = 1.0;
      float avgPos = 0.0;
      for (int j=0; j<examples->numberOfExamples(); j++)
      {
        // examples that are changed by the new rule
        if (rules->at(i)->call(examples->at(j)) && currentState->p[getClassIndex(rules->at(i))][j] == rules->at(i)->quality)
        {
          if (getClassIndex(rules->at(i)) == examples->at(j).getClass().intV)
          {
            positives += 1.0;
            avgPos += tempState->p[getClassIndex(rules->at(i))][j];
          }
          else
            negatives += 1.0;
          if (tempState->p[getClassIndex(rules->at(i))][j] < worstP)
            worstP = tempState->p[getClassIndex(rules->at(i))][j];
        }
      }
      if (positives>0)
        avgPos /= positives;
    //  printf("positives: %f, negatives: %f, worstP: %f\n", positives, negatives, worstP);
    	float diff = rules->at(i)->classDistribution->atint(getClassIndex(rules->at(i)))/(rules->at(i))->quality;
   //   printf("diff1: %f\n",diff);
    	diff -= rules->at(i)->classDistribution->abs;
   //   printf("diff2: %f\n",diff);
      float newPositives, newNegatives;
 /*     if (diff>positives)
      {
        newNegatives = negatives + positives;
        newPositives = 0.0;
      }
      else
      {
        newNegatives = negatives + diff;
        newPositives = positives - diff;
      }*/
      newPositives = positives;
      newNegatives = negatives + diff;
  //    printf("newPositives: %f, newNegatives: %f\n", newPositives, newNegatives);
      float oldEvaluation = 0.0;
      for (int j=0; j<examples->numberOfExamples(); j++)
        // examples that are changed by the new rule
        if (rules->at(i)->call(examples->at(j)) && currentState->p[getClassIndex(rules->at(i))][j] == rules->at(i)->quality) {
           oldEvaluation += log(tempState->p[examples->at(j).getClass().intV][j]);
           if (getClassIndex(rules->at(i)) == examples->at(j).getClass().intV) {
             oldEvaluation += diff/positives*(1.0-tempState->p[examples->at(j).getClass().intV][j])/(1-avgPos)*log(1.0-tempState->p[examples->at(j).getClass().intV][j]);
           }

   /*         oldEvaluation += newPositives/positives*(tempState->p[examples->at(j).getClass().intV][j]/avgPos)*log(tempState->p[examples->at(j).getClass().intV][j]);
            if ((1.0-newPositives/positives*(tempState->p[examples->at(j).getClass().intV][j]/avgPos))>0)
              oldEvaluation += (1.0-newPositives/positives*(tempState->p[examples->at(j).getClass().intV][j]/avgPos))*log(1.0-tempState->p[examples->at(j).getClass().intV][j]);
          }
          else /*
       /*   if (getClassIndex(rules->at(i)) == examples->at(j).getClass().intV)
          {
            oldEvaluation += newPositives/positives*(tempState->p[examples->at(j).getClass().intV/tempState->avgProb->at(i))*log(tempState->p[examples->at(j).getClass().intV][j]);
            //oldEvaluation += (1.0-newPositives/positives)*log(1.0-tempState->p[examples->at(j).getClass().intV][j]);
          }
          else
           oldEvaluation += log(tempState->p[examples->at(j).getClass().intV][j]); */
        }
//      printf("oldEvaluation 1: %f\n", oldEvaluation);
     // oldEvaluation += (newNegatives-negatives)*log(1.0-worstP);
 //     printf("oldEvaluation 2: %f\n", oldEvaluation);
      float newEvaluation = newPositives*log(rules->at(i)->quality)+newNegatives*log(1.0-rules->at(i)->quality);          
 //     printf("old: %f, new: %f\n", oldEvaluation, newEvaluation);
      if (newEvaluation > oldEvaluation)
      {
        bestRuleI = i;
			  bestEvaluation = (currentState->eval-tempState->eval)/rules->at(i)->examples->numberOfExamples();
      }
		}
		tempState->copyTo(currentState);
	}
	if (bestRuleI > -1) {
		currentState->setFrontRule(bestRuleI);
		evaluate();
 //   printf("Eval improvement: from %f to %f\n",tempState->eval,currentState->eval);
	} 
/*	currentState->copyTo(tempState);
	int bestRuleI = -1, bestAt = -1;
	float bestEvaluation = currentState->eval;

    for (int i=0; i<rules->size(); i++)
		for (int at=0; at<=currentState->frontRules->size(); at++) {
			currentState->setFrontRule(i);//,at);
			evaluate(currentState);
			printf("setbest, %f %f\n", bestEvaluation, currentState->eval);
			if (currentState->eval > bestEvaluation) {
				bestRuleI = i;
				bestAt = at;
				bestEvaluation = currentState->eval;
			}
			tempState->copyTo(currentState);
		}
	if (bestRuleI > -1) {
		printf("best Rule: %d %d\n", bestRuleI, bestAt);
		currentState->setFrontRule(bestRuleI, bestAt);
		evaluate(currentState);
	} */
}


void TRuleClassifier_logit::optimizeBetas()
{
  float step = 2.0;
  minStep = (float)0.01;
  
  PLogitClassifierState oldState;
  currentState->copyTo(oldState);
  // first optimize prior betas
  while (priorBetaType > 0 && step > minStep) {
    step /= 2.0;
    correctPriorBetas(step);
    if (currentState->eval >= oldState->eval)
      currentState->copyTo(oldState);
    else
    {
      oldState->copyTo(currentState);
      break;
    }
  }

  step = 2.0;
  while (step > minStep)
  {
    step /= 2.0;
    bool improvedOverAll = true;
  	while (improvedOverAll) {
      stabilizeAndEvaluate(step,-1);
      currentState->copyTo(oldState);
      updateRuleBetas(step);
      if (currentState->eval <= oldState->eval) {
        oldState->copyTo(currentState);
		    improvedOverAll = false;
      }
      else
        improvedOverAll = true;
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
    wsd->push_back(exp(0.5*expab));
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
  float *coverages = new float[examples->numberOfExamples()];
  int j=0;
  for (j=0; j<examples->numberOfExamples(); j++) {
    coverages[j] = 0.0;
  }
  int i=0;
  {
    PITERATE(TRuleList, ri, rules) {
      j=0;
      PEITERATE(ei, examples) {
        if ((*ri)->call(*ei)) {
          //int vv = (*ei).getClass().intV;
		      if ((*ei).getClass().intV == getClassIndex(*ri))
			      coverages[j] += 1.0;
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
      if (getClassIndex(rules->at(i)) == examples->at(*ind).getClass().intV) {
        newCov += coverages[*ind];
        counter++;
      }
    if (counter) {
      wavgCov->push_back(newCov/counter);
    }
    else
      wavgCov->push_back(0.0);
  }
  evaluate();
}

// Iterates through rules and tries to change betas to improve goodness-of-fit
void TRuleClassifier_logit::updateRuleBetas(float & step)
{
  PLogitClassifierState finalState, tempState; 
  currentState->copyTo(finalState);
  currentState->copyTo(tempState);

  bool changed = true; 
  int counter = 0; // so it wont cycle indefinitely
  while (changed && counter<10) {
    changed = false; 
    counter ++;
    if (currentState->eval > finalState->eval)
    {
      currentState->copyTo(finalState);
      counter = 1;
    }

    correctPriorBetas(step);
    for (int i=0; i<rules->size(); i++) {
      // if rule is not significant, set its beta to zero
      if (wSatQ->at(i)<rules->at(i)->quality && currentState->betas[i] > 0)
      {
        currentState->newBeta(i,0.0);
        stabilizeAndEvaluate(step,i);
        if (currentState->avgProb->at(i) < wSatQ->at(i))
          tempState->copyTo(currentState);
        else {
          currentState->copyTo(tempState);
          changed = true;
        }
      }

      if (wSatQ->at(i)<rules->at(i)->quality && currentState->betas[i] == 0 && currentState->avgProb->at(i) >= wSatQ->at(i))
        continue;
      if (currentState->avgProb->at(i) >= rules->at(i)->quality)
        continue;
      float oldStateProb = currentState->avgProb->at(i);

      // sign of error before change
      float errorBefore = 0.0;
      for (int j=0; j<rules->size(); j++)
        if (getClassIndex(rules->at(j)) == getClassIndex(rules->at(i))) // && wsig->at(j) <= wsig->at(i)) // && wsd->at(j) < wsd->at(i)) // && rules->at(j)->classDistribution->abs > rules->at(i)->classDistribution->abs) //wsd->at(j) < wsd->at(i))
          errorBefore -= (currentState->avgProb->at(j) - rules->at(j)->quality)/wsd->at(j);

      if (errorBefore > 0.0)
        currentState->newBeta(i,currentState->betas[i]+step);
      else if (errorBefore < 0.0 && currentState->betas[i]>=step)
        currentState->newBeta(i,currentState->betas[i]-step);
      else
        continue;
      stabilizeAndEvaluate(step,i);

      // compute after error 
      float errorAfter = 0.0;
      for (int j=0; j<rules->size(); j++)
        if (getClassIndex(rules->at(j)) == getClassIndex(rules->at(i))) 
          errorAfter -= (currentState->avgProb->at(j) - rules->at(j)->quality)/wsd->at(j);

      if ((errorBefore < 0.0 && errorAfter>=errorBefore || errorBefore > 0.0 && errorAfter<=errorBefore && errorAfter>=0.0) && currentState->avgProb->at(i) <= rules->at(i)->quality) { 
        currentState->copyTo(tempState);
        changed = true;
      }
      else
        tempState->copyTo(currentState);
    }
	}
  finalState->copyTo(currentState);
}


// If average predicted probability > then its quality and beta > 0 then decrease beta
void TRuleClassifier_logit::stabilizeAndEvaluate(float & step, int last_changed_rule_index)
{
  evaluate();
  bool changed = true;
  while (changed) {
    changed = false;
    for (int i=0; i<rules->size(); i++) {
	    if (i == last_changed_rule_index)
		    continue;
      // if optimistic, always decrease beta 
      if ( (currentState->avgProb->at(i) > rules->at(i)->quality) &&
		       (currentState->betas[i]-step)>=0.0 ) {
        if ((currentState->betas[i]-step)<=0.0)
          currentState->newBeta(i,0.0);
        else
          currentState->newBeta(i,currentState->betas[i]-step);
        evaluate();
  	    changed = true;
      }
    }
  }
}

// Correct prior probabilities by setting prior betas.
void TRuleClassifier_logit::correctPriorBetas(float & step)
{
  PLogitClassifierState tempState;
  currentState->copyTo(tempState);

  bool changed = true;
  while (changed) { 
	  changed = false;
	  for (int i=0; i<examples->domain->classVar->noOfValues()-1; i++) { // there are n-1 prior betas
      // positive change
      if (currentState->avgPriorProb->at(i) < prior->atint(i)/prior->abs) {
        currentState->newPriorBeta(i,currentState->priorBetas[i]+step);
        if (currentState->avgPriorProb->at(i) <= prior->atint(i)/prior->abs) {
          changed = true;
          currentState->copyTo(tempState);
        }
        else
          tempState->copyTo(currentState);
      }
      else {// negative change
        currentState->newPriorBeta(i,currentState->priorBetas[i]-step);
        if (currentState->avgPriorProb->at(i) >= prior->atint(i)/prior->abs) {
          changed = true;
          currentState->copyTo(tempState);
        }
        else
          tempState->copyTo(currentState);
      }
      evaluate();
    }
  }
}

// Computes new probabilities of examples if rule would have beta set as newBeta.
void TRuleClassifier_logit::evaluate()
{
  currentState->eval = 0.0;
  for (int ei=0; ei<examples->numberOfExamples(); ei++)
	    currentState->eval += currentState->p[examples->at(ei).getClass().intV][ei]>0.0 ? log(currentState->p[examples->at(ei).getClass().intV][ei]) : -1e+6;
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
  bool foundFrontRule = false;
  float bestQuality = 0.0;
  PITERATE(TRuleList, rs, frontRules) {
	  if ((*rs)->call(ex) && (*rs)->quality > bestQuality) {
      bestQuality = (*rs)->quality;
		  dist->setint(getClassIndex(*rs),(*rs)->quality);
		  for (int ci=0; ci<examples->domain->classVar->noOfValues(); ci++) 
		    if (ci!=getClassIndex(*rs))
			    dist->setint(ci,(1.0-(*rs)->quality)/(examples->domain->classVar->noOfValues()-1));
      foundFrontRule = true;
	  }
  }
  if (foundFrontRule)
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
    float f = priorProbBetas->at(i) + priorFs[i];
    TFloatList::const_iterator b(ruleBetas->begin()), be(ruleBetas->end());
    TRuleList::iterator r(rules->begin()), re(rules->end());
    for (; r!=re; r++, b++)
      if ((*r)->call(cexample)) {
        if (getClassIndex(*r) == i) 
  		    f += (*b); 
        else
          f -= (*b); 
      }
    dist->addint(i,exp(f));
  }
  dist->addint(res->noOfElements()-1,1.0);
  dist->normalize();
  delete [] priorFs;
  return res;
}



