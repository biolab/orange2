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


#include <math.h>
#include "stladdon.hpp"

#include "vars.hpp"
#include "domain.hpp"
#include "examples.hpp"
#include "examplegen.hpp"
#include "distvars.hpp"
#include "contingency.hpp"
#include "basstat.hpp"
#include "orvector.hpp"

#include "distance.ppp"


TExamplesDistanceConstructor::TExamplesDistanceConstructor(const bool &ic)
: ignoreClass(ic)
{}


TExamplesDistanceConstructor_Hamming::TExamplesDistanceConstructor_Hamming()
: ignoreClass(true),
  ignoreUnknowns(false)
{}


PExamplesDistance TExamplesDistanceConstructor_Hamming::operator()(PExampleGenerator, const int &, PDomainDistributions, PDomainBasicAttrStat) const
{ return mlnew TExamplesDistance_Hamming(ignoreClass, ignoreUnknowns); }


TExamplesDistance_Hamming::TExamplesDistance_Hamming(const bool &ic, const bool &iu)
: ignoreClass(ic),
  ignoreUnknowns(iu)
{}


float TExamplesDistance_Hamming::operator()(const TExample &e1, const TExample &e2) const 
{ if (   (e1.domain != e2.domain)
      && (ignoreClass ? e1.domain->attributes != e2.domain->attributes
                     : e1.domain->variables != e2.domain->variables))
    raiseError("cannot compare examples from different domains");

  float dist = 0.0;
  int Na = e1.domain->attributes->size() + (!ignoreClass && e1.domain->classVar ? 1 : 0);
  for(TExample::const_iterator i1 = e1.begin(), i2 = e2.begin(); Na--; i1++, i2++)
    if (   (!ignoreUnknowns || !(*i1).isSpecial() && !(*i2).isSpecial())
        && (!(*i1).compatible(*i2)))
      dist += 1.0;
  return dist;
}


TExamplesDistanceConstructor_Normalized::TExamplesDistanceConstructor_Normalized()
: TExamplesDistanceConstructor(),
  normalize(true),
  ignoreUnknowns(false)
{}


TExamplesDistanceConstructor_Normalized::TExamplesDistanceConstructor_Normalized(const bool &ic, const bool &no, const bool &iu)
: TExamplesDistanceConstructor(ic),
  normalize(no),
  ignoreUnknowns(iu)
{}


TExamplesDistance_Normalized::TExamplesDistance_Normalized()
: normalizers(PFloatList()),
  domainVersion(-1),
  normalize(true),
  ignoreUnknowns(false)
{}


TExamplesDistance_Normalized::TExamplesDistance_Normalized(const bool &ignoreClass, const bool &no, const bool &iu, PExampleGenerator egen, const int &weightID, PDomainDistributions ddist, PDomainBasicAttrStat bstat)
: normalizers(mlnew TAttributedFloatList()),
  bases(mlnew TAttributedFloatList()),
  averages(mlnew TAttributedFloatList()),
  variances(mlnew TAttributedFloatList()),
  domainVersion(egen ? egen->domain->version : -1),
  normalize(no),
  ignoreUnknowns(iu)
{ TFloatList &unormalizers = normalizers.getReference();

  PVarList varlist;

  if (!bstat && !ddist && egen)
    bstat = mlnew TDomainBasicAttrStat(egen, weightID);

  if (bstat && egen) {
    varlist  = ignoreClass ? egen->domain->attributes : egen->domain->variables;

    TDomainBasicAttrStat::const_iterator si(bstat->begin()), ei(bstat->end());
    TVarList::const_iterator vi (egen->domain->variables->begin()), evi(egen->domain->variables->end());

    if (ignoreClass && egen->domain->classVar) {
      evi--;
      ei--;
    }

    for(; (vi!=evi) && (si!=ei); si++, vi++) {
      if ((*vi)->varType==TValue::FLOATVAR) {
        if (*si && ((*si)->n>0)) {
          normalizers->push_back((*si)->max!=(*si)->min ? 1.0/((*si)->max-(*si)->min) : 0.0);
          bases->push_back((*si)->min);
          averages->push_back((*si)->avg);
          variances->push_back((*si)->dev * (*si)->dev);
        }
        else {
          normalizers->push_back(0.0);
          bases->push_back(0.0);
          averages->push_back(0.0);
          variances->push_back(0.0);
        }
      }
      else if ((*vi)->varType==TValue::INTVAR) {
        if ((*vi)->ordered)
          if ((*vi)->noOfValues()>0)
            normalizers->push_back(1.0/(*vi)->noOfValues());
          else
            normalizers->push_back(0.0);
        else
          normalizers->push_back(-1.0);
        bases->push_back(0.0);
        averages->push_back(0.0);
        variances->push_back(0.0);
      }
      else {
        normalizers->push_back(0.0);
        bases->push_back(0.0);
        averages->push_back(0.0);
        variances->push_back(0.0);
      }
    }

    if ((vi!=evi) || (si!=ei))
      raiseError("lengths of domain and basic attribute statistics do not match");
  }

  else if (ddist) {
    varlist = mlnew TVarList;

    PITERATE(TDomainDistributions, ci, ddist) {
      if (*ci) {
        const PVariable &vi = (*ci)->variable;
        varlist->push_back(vi);

        if (vi->varType==TValue::FLOATVAR) {
          TContDistribution *dcont = (*ci).AS(TContDistribution);
          if (dcont && (dcont->begin() != dcont->end())) {
            const float min = (*dcont->distribution.begin()).first;
            const float dif = (*dcont->distribution.rbegin()).first - min;
            normalizers->push_back(dif > 0.0 ? 1.0/dif : 0.0);
            bases->push_back(min);
            averages->push_back(dcont->average());
            variances->push_back(dcont->var());
          }
          else {
            normalizers->push_back(0.0);
            averages->push_back(0.0);
            variances->push_back(0.0);
          }
        }
        else if (vi->varType==TValue::INTVAR) {
          if (vi->ordered) {
            const int nval = vi->noOfValues();
            normalizers->push_back(nval ? 1.0/float(nval) : 0.0);
          }
          else
            normalizers->push_back(-1.0);
            bases->push_back(0.0);
            averages->push_back(0.0);
            variances->push_back(0.0);
        }
        else {
          normalizers->push_back(0.0);
          bases->push_back(0.0);
          averages->push_back(0.0);
          variances->push_back(0.0);
        }
      }
      else {
        normalizers->push_back(0.0);
        bases->push_back(0.0);
        averages->push_back(0.0);
        variances->push_back(0.0);
      }
    }
  }

  else if (bstat) {
    varlist = mlnew TVarList;

    TDomainBasicAttrStat::const_iterator si(bstat->begin()), ei(bstat->end());

    if (ignoreClass) // can't check it, but suppose there is a class attribute
      ei--;

    for(; si!=ei; si++) {
      if (!*si)
        raiseError("cannot compute normalizers from BasicAttrStat in presence of non-continuous attributes");

      varlist->push_back((*si)->variable);
      if (((*si)->n>0) && ((*si)->max!=(*si)->min)) {
        normalizers->push_back(1.0/((*si)->max-(*si)->min));
        bases->push_back((*si)->min);
        averages->push_back((*si)->avg);
        variances->push_back((*si)->dev * (*si)->dev);
      }
      else {
        normalizers->push_back(0.0);
        bases->push_back(0.0);
        averages->push_back(0.0);
        variances->push_back(0.0);
      }
    }
  }

  else
   raiseError("no data");

  normalizers->attributes = bases->attributes = averages->attributes = variances->attributes = varlist;
}


/* Returns a vector of normalized differences between the two examples.
   Quick checks do not guarantee that domains are really same to the training domain.
   To be really on the safe side, we should know the domain and convert both examples. Too slow...
*/
void TExamplesDistance_Normalized::getDifs(const TExample &e1, const TExample &e2, vector<float> &difs) const
{ checkProperty(normalizers);

  if (   (e1.domain != e2.domain)
      && (e1.domain->variables != e2.domain->variables))
    raiseError("examples are from different domains");
  
/*  if (domainVersion>=0
        ? (domainVersion != e1.domain->version)
        : ((normalizers->size() > e1.domain->variables->size()) || (normalizers->size()< e1.domain->attributes->size())))
    raiseError("examples are from a wrong domain");*/

  difs = vector<float>(normalizers->size(), 0.0);
  vector<float>::iterator di(difs.begin());

  TExample::const_iterator i1(e1.begin()), i2(e2.begin());
  for(TFloatList::const_iterator si(normalizers->begin()), se(normalizers->end()); si!=se; si++, i1++, i2++, di++)
    if ((*i1).isSpecial() || (*i2).isSpecial())
      *di = ((*si!=0) && !ignoreUnknowns) ? 0.5 : 0.0;
    else 
      if (normalize) {
        if (*si>0) {
          if ((*i1).varType == TValue::FLOATVAR)
            *di = *si * fabs((*i1).floatV - (*i2).floatV);
          else if ((*i1).varType == TValue::INTVAR)
            *di = *si * fabs(float((*i1).intV - (*i2).intV));
        }
        else if (*si<0)
          *di = (*i1).compatible(*i2) ? 0.0 : 1.0;
      }
      else {
        if ((*i1).varType == TValue::FLOATVAR) {
          *di = fabs((*i1).floatV - (*i2).floatV);
        }
        else 
          if (*si>0) {
            if ((*i1).varType == TValue::INTVAR)
              *di = fabs(float((*i1).intV - (*i2).intV));
            else
              *di = (*i1).compatible(*i2) ? 0.0 : 1.0;
          }
    }
}



void TExamplesDistance_Normalized::getNormalized(const TExample &e1, vector<float> &normalized) const
{
  checkProperty(normalizers);
  checkProperty(bases);

  if (  domainVersion>=0
        ? (domainVersion != e1.domain->version)
        : ((normalizers->size() > e1.domain->variables->size()) || (normalizers->size()< e1.domain->attributes->size())))
    raiseError("example is from a wrong domain");

  normalized.clear();
  TExample::const_iterator ei(e1.begin());
  for(TFloatList::const_iterator normi(normalizers->begin()), norme(normalizers->end()), basi(bases->begin()); normi!=norme; ei++, normi++, basi++) {
// changed by PJ
/*
    if ((*ei).isSpecial())
      normalized.push_back(numeric_limits<float>::quiet_NaN());
    else
      if ((*normi>0) && ((*ei).varType == TValue::FLOATVAR))
        normalized.push_back(normalize ? ((*ei).floatV - *basi) / *normi : (*ei).floatV);
      else
        normalized.push_back(-1.0);
*/
	if ((*ei).isSpecial() || ei->varType != TValue::FLOATVAR)
		normalized.push_back(numeric_limits<float>::signaling_NaN());
    else
      if (*normi>0 && normalize)
        normalized.push_back(((*ei).floatV - *basi) * (*normi));
      else
        normalized.push_back((*ei).floatV);
  }
}


/*TExamplesDistanceConstructor_Maximal::TExamplesDistanceConstructor_Maximal()
{}
*/

TExamplesDistanceConstructor_Euclidean::TExamplesDistanceConstructor_Euclidean()
{}


TExamplesDistanceConstructor_Manhattan::TExamplesDistanceConstructor_Manhattan()
{}


PExamplesDistance TExamplesDistanceConstructor_Maximal::operator()(PExampleGenerator egen, const int &weightID, PDomainDistributions ddist, PDomainBasicAttrStat bstat) const
{ return mlnew TExamplesDistance_Maximal(ignoreClass, normalize, ignoreUnknowns, egen, weightID, ddist, bstat); }


PExamplesDistance TExamplesDistanceConstructor_Manhattan::operator()(PExampleGenerator egen, const int &weightID, PDomainDistributions ddist, PDomainBasicAttrStat bstat) const
{ return mlnew TExamplesDistance_Manhattan(ignoreClass, normalize, ignoreUnknowns, egen, weightID, ddist, bstat); }


PExamplesDistance TExamplesDistanceConstructor_Euclidean::operator()(PExampleGenerator egen, const int &weightID, PDomainDistributions ddist, PDomainBasicAttrStat bstat) const
{ return mlnew TExamplesDistance_Euclidean(ignoreClass, normalize, ignoreUnknowns, egen, weightID, ddist, bstat); }



TExamplesDistance_Maximal::TExamplesDistance_Maximal()
{}


TExamplesDistance_Manhattan::TExamplesDistance_Manhattan()
{}


TExamplesDistance_Euclidean::TExamplesDistance_Euclidean()
{}



TExamplesDistance_Maximal::TExamplesDistance_Maximal(const bool &ignoreClass, const bool &normalize, const bool &ignoreUnknowns, PExampleGenerator egen, const int &weightID, PDomainDistributions ddist, PDomainBasicAttrStat dstat)
: TExamplesDistance_Normalized(ignoreClass, normalize, ignoreUnknowns, egen, weightID, ddist, dstat)
{}


TExamplesDistance_Manhattan::TExamplesDistance_Manhattan(const bool &ignoreClass, const bool &normalize, const bool &ignoreUnknowns, PExampleGenerator egen, const int &weightID, PDomainDistributions ddist, PDomainBasicAttrStat dstat)
: TExamplesDistance_Normalized(ignoreClass, normalize, ignoreUnknowns, egen, weightID, ddist, dstat)
{}


TExamplesDistance_Euclidean::TExamplesDistance_Euclidean(const bool &ignoreClass, const bool &normalize, const bool &ignoreUnknowns, PExampleGenerator egen, const int &weightID, PDomainDistributions ddist, PDomainBasicAttrStat dstat)
: TExamplesDistance_Normalized(ignoreClass, normalize, ignoreUnknowns, egen, weightID, ddist, dstat),
  distributions(mlnew TDomainDistributions(egen, weightID, false, true)),
  bothSpecialDist(mlnew TAttributedFloatList())
{
  bothSpecialDist->attributes = averages->attributes;

  PITERATE(TDomainDistributions, di, distributions) {
    if (*di) {
      float sum2 = 0;
      TDiscDistribution *distr = (*di).AS(TDiscDistribution);
      ITERATE(vector<float>, pi, distr->distribution)
        sum2 += (*pi) * (*pi);
      sum2 /= distr->abs * distr->abs;
      bothSpecialDist->push_back(1-sum2);
    }
    else
      bothSpecialDist->push_back(0.0);
  }
}



float TExamplesDistance_Maximal::operator ()(const TExample &e1, const TExample &e2) const 
{ 
  vector<float> difs;
  getDifs(e1, e2, difs);
  return difs.size() ? *max_element(difs.begin(), difs.end()) : 0.0;
}


float TExamplesDistance_Manhattan::operator ()(const TExample &e1, const TExample &e2) const 
{ 
  vector<float> difs;
  getDifs(e1, e2, difs);
  float dist = 0.0;
  const_ITERATE(vector<float>, di, difs)
    dist += *di;
  return dist;
}


float TExamplesDistance_Euclidean::operator ()(const TExample &e1, const TExample &e2) const 
{ 
  vector<float> difs;
  getDifs(e1, e2, difs);
  float dist = 0.0;
  TExample::const_iterator e1i(e1.begin()), e2i(e2.begin());
  TFloatList::const_iterator avgi(averages->begin()), vari(variances->begin());
  vector<float>::const_iterator di(difs.begin()), de(difs.end());
  TDomainDistributions::const_iterator disti(distributions->begin());
  TFloatList::const_iterator bsi(bothSpecialDist->begin());
  TFloatList::const_iterator si(normalizers->begin());

  for(; di!=de; di++, e1i++, e2i++, avgi++, vari++, disti++, si++)
    if ((*e1i).varType == TValue::FLOATVAR) {
      if ((*e1i).isSpecial())
        if ((*e2i).isSpecial())
          dist += 2 * *vari;
        else {
          const float e2a = (*e2i).floatV - *avgi;
          if (normalize)
            dist += e2a*e2a + *vari * *si * *si;
          else
            dist += e2a*e2a + *vari;
        }
      else // e1i is not special
        if ((*e2i).isSpecial()) {
          const float e2a = (*e1i).floatV - *avgi;
          if (normalize)
            dist += e2a*e2a + *vari * *si * *si;
          else
            dist += e2a*e2a + *vari;
        }
      else // none is special
        dist += (*di) * (*di);
    }

    else if ((*e1i).varType == TValue::INTVAR) {
      if ((*e1i).isSpecial())
        if ((*e2i).isSpecial())
          dist += *bsi;
        else
          dist += 1 - (*disti)->p((*e2i).intV);
      else // e1i is not special
        if ((*e2i).isSpecial())
          dist += 1 - (*disti)->p((*e1i).intV);
        else
          if ((*e1i).intV != (*e2i).intV)
            dist += 1;
    }
    else
      dist += (*di)*(*di);

  return sqrt(dist);
}




TExamplesDistanceConstructor_Relief::TExamplesDistanceConstructor_Relief()
{}


PExamplesDistance TExamplesDistanceConstructor_Relief::operator()(PExampleGenerator gen, const int &weightID, PDomainDistributions ddist, PDomainBasicAttrStat bstat) const
{ 
  const TDomain &domain = gen->domain.getReference();

  PVariable otherAttribute = domain.hasOtherAttributes();
  if (otherAttribute)
    raiseError("domain has attributes whose type is not supported by ReliefF (e.g. '%s')", otherAttribute->name.c_str());

  // for continuous attributes BasicAttrStat suffices; for discrete it does not
  const bool hasDiscrete = domain.hasDiscreteAttributes() || domain.classVar && (domain.classVar->varType == TValue::INTVAR);
  if (!bstat || (hasDiscrete && !ddist))
    if (!gen)
      raiseError("examples or domain distributions expected");
    else
      if (hasDiscrete)
        ddist = mlnew TDomainDistributions(gen, weightID);
      else
        bstat = mlnew TDomainBasicAttrStat(gen, weightID);

  TExamplesDistance_Relief *edr = mlnew TExamplesDistance_Relief();
  PExamplesDistance res = edr;

  if (!ignoreClass)
    raiseError("'ignoreClass' not supported");

  edr->averages       = mlnew TAttributedFloatList(gen->domain->attributes);
  edr->normalizations = mlnew TAttributedFloatList(gen->domain->attributes);
  edr->bothSpecial    = mlnew TAttributedFloatList(gen->domain->attributes);

  edr->distributions = CLONE(TDomainDistributions, ddist);
  if (ddist)
    edr->distributions->normalize();
  
  for(int attrIndex = 0, nAttrs = gen->domain->variables->size(); attrIndex != nAttrs; attrIndex++)
    if (domain.variables->at(attrIndex)->varType == TValue::FLOATVAR) {
      if (bstat) {
        const TBasicAttrStat &bas = bstat->at(attrIndex).getReference();
        edr->averages->push_back(bas.avg);
        edr->normalizations->push_back(bas.max - bas.min);
      }
      else {
        const TContDistribution *contd = ddist->at(attrIndex).AS(TContDistribution);
        if (contd->size()) {
          edr->averages->push_back(contd->average());
          edr->normalizations->push_back((*contd->distribution.rbegin()).first - (*contd->distribution.begin()).first);
        }
        else {
          edr->averages->push_back(0.0);
          edr->normalizations->push_back(1.0);
        }
      }
      edr->bothSpecial->push_back(0.5);
    }
  else {
    edr->averages->push_back(0.0);
    edr->normalizations->push_back(0.0);
    float dist = 1.0;
    const_PITERATE(TDiscDistribution, di, ddist->at(attrIndex).AS(TDiscDistribution))
      dist -= *di * *di;
    edr->bothSpecial->push_back(dist);
  }

  return res;
}


float TExamplesDistance_Relief::operator()(const TExample &e1, const TExample &e2) const
{ 
  checkProperty(averages);
  checkProperty(normalizations);
  checkProperty(bothSpecial);

  const bool hasDistributions = bool(distributions);

  TExample::const_iterator e1i(e1.begin()), e1e(e1.end());
  TExample::const_iterator e2i(e2.begin());
  TFloatList::const_iterator avgi(averages->begin()),
                             nori(normalizations->begin()),
                             btsi(bothSpecial->begin());

  TDomainDistributions::const_iterator di;
  if (hasDistributions)
    di = distributions->begin();

  float dist = 0.0;
  for(; e1i!=e1e; e1i++, e2i++, avgi++, nori++, btsi++) {
    float dd = 0.0;
    const TValue &v1 = *e1i, &v2 = *e2i;
    if (v1.varType==TValue::INTVAR) {             // discrete
      if (v1.isSpecial())
        if (v2.isSpecial()) 
          dd = *btsi;                               // both special
        else {
          if (!hasDistributions)
            raiseError("'distributions' not set; cannot deal with unknown values");
          dd = 1-(*di)->atint(v2.intV);        // v1 special
        }
      else
        if (v2.isSpecial()) {
          if (!hasDistributions)
            raiseError("'distributions' not set; cannot deal with unknown values");
          dd = 1-(*di)->atint(v1.intV);        // v2 special
        }
        else
          if (v1.intV != v2.intV)
            dd = 1.0;                               // both known, different
    }
    else if (*nori>0) {                           // continuous, and can normalize
      if (v1.isSpecial())
        if (v2.isSpecial()) 
          dd = float(0.5);                          // both special
        else
          dd = fabs(*avgi - v2.floatV) / *nori;      // v1 special
      else
        if (v2.isSpecial())
          dd = fabs(*avgi - v1.floatV) / *nori;      // v2 special
        else
          dd = fabs(v1.floatV - v2.floatV) / *nori;  // both known
    }

    dist += dd>1.0 ? 1.0 : dd;

    if (hasDistributions)
      di++;
  }

  return dist;
}


float TExamplesDistance_Relief::operator()(const int &attrNo, const TValue &v1, const TValue &v2) const
{
  float dd = -1.0;
  if (v1.varType==TValue::INTVAR) {                              // discrete
    if (v1.isSpecial())
      if (v2.isSpecial()) 
        dd = bothSpecial->at(attrNo);                                // both special
        else 
          dd = 1 - distributions->at(attrNo)->atint(v2.intV);   // v1 special
      else
        if (v2.isSpecial())  
          dd = 1 - distributions->at(attrNo)->atint(v1.intV);   // v2 special
        else
          dd = (v1.intV != v2.intV) ? 1.0 : 0.0;                     // both known
    }
    else if (normalizations->at(attrNo)>0) {                     // continuous, and can normalize
      if (v1.isSpecial())
        if (v2.isSpecial()) 
          dd = 0.5;                                                                  // both special
        else
          dd = fabs(averages->at(attrNo) - v2.floatV) / normalizations->at(attrNo);   // v1 special
      else
        if (v2.isSpecial())
          dd = fabs(averages->at(attrNo) - v1.floatV) / normalizations->at(attrNo);   // v2 special
        else
          dd = fabs(v1.floatV - v2.floatV) / normalizations->at(attrNo);              // both known
    }

    return dd>1.0 ? 1.0 : dd;
}


