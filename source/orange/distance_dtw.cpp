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

#include <cmath>
#include <limits>

#include "examplegen.hpp"
#include "distance_dtw.ppp"

DEFINE_TOrangeVector_classDescription(TAlignment, "TAlignmentList", false, ORANGE_API)


TAlignment::TAlignment()
{}


TAlignment::TAlignment(int ai, int aj)
: i(ai),
  j(aj)
{}


TAlignment::TAlignment(const TAlignment &old)
: i(old.i),
  j(old.j)
{}


bool TAlignment::operator==(const TAlignment &o) const
{ return (i==o.i) && (j==o.j);
}


bool TAlignment::operator<(const TAlignment &o) const
{ return (i<o.i) || ((o.i==o.j) && (j==o.j));
}


class TdtwElement
{
public:
	float	dist2;				// squared distance between two points: dist[i,j] = (pi - qj)^2
	float	minSumDist2;		// sum of squared distances on the minimal warping path: sum(distij)
	int K;						// length of the minimal warping path
	const TdtwElement *pParent;	// parent element for reconstruction of warping path

	TdtwElement() : dist2(-1), minSumDist2(-1), K(-1), pParent(NULL) {};
	TdtwElement(float newDist) : dist2(newDist), minSumDist2(-1), K(-1), pParent(NULL) {};

/*    TdtwElement &operator = (const TdtwElement &old)
    { dist2 = old.dist2;
      minSumDist2 = old.minSumDist2;
      K = old.K;
      pParent = old.pParent;
      return *this;
    }*/

	// update minSumDist2, K
	void updateMin(const PdtwVector &d)
	{
		PdtwVector::const_iterator iter, itere;
		vector<float> mm;
		vector<int> kk;
		for ( iter = d.begin(), itere = d.end(); iter != itere; iter++ )
		{
			mm.push_back( (*iter)->minSumDist2 + dist2 );
			kk.push_back( (*iter)->K + 1 );
		}
		int mMinIdx = -1, i;
		float mMin = numeric_limits<float>::max();
		for ( i = 0; i < mm.size(); i++ )
		{
			if ( mm[i] < mMin )
			{
				mMin = mm[i];
				mMinIdx = i;
			}
		}
		minSumDist2 = mm[mMinIdx];
		K = kk[mMinIdx];
		pParent = d[mMinIdx];
	}
};


TExamplesDistance_DTW::TExamplesDistance_DTW()
: dtwDistance(DTW_EUCLIDEAN)
{}


TExamplesDistance_DTW::TExamplesDistance_DTW(const int &distance, const bool &normalize, const bool &ignoreClass, PExampleGenerator egen, PDomainDistributions ddist, PDomainBasicAttrStat dstat)
: TExamplesDistance_Normalized(ignoreClass, normalize, ignoreClass, egen, ddist, dstat),
  dtwDistance(distance)
{}


/*
float TExamplesDistance_DTW::operator ()(const TExample &e1, const TExample &e2) const
{ 
  vector<float> seq1, seq2;
  getNormalized(e1, seq1);
  getNormalized(e2, seq2);
  TdtwMatrix mtrx;
  initMatrix(seq1,seq2, mtrx);
  float dtwDistance = calcDistance(mtrx);
  return dtwDistance;
}
*/

float TExamplesDistance_DTW::operator ()(const TExample &e1, const TExample &e2) const
{ 
  vector<float> seq1, seq2, der1, der2;
  getNormalized(e1, seq1);
  getNormalized(e2, seq2);
  TdtwMatrix mtrx;
  switch (dtwDistance) {
	case DTW_EUCLIDEAN: {
	  initMatrix(seq1, seq2, mtrx);
	  break;
						}
	case DTW_DERIVATIVE: {
		getDerivatives(seq1, der1);
		getDerivatives(seq2, der2);
		initMatrix(der1, der2, mtrx);
		break;
						 }
  }
  float dist = calcDistance(mtrx);
  return dist;
}


float TExamplesDistance_DTW::operator ()(const TExample &e1, const TExample &e2, PWarpPath &path) const
{ 
  vector<float> seq1, seq2, der1, der2;
  getNormalized(e1, seq1);
  getNormalized(e2, seq2);
  TdtwMatrix mtrx;
  switch (dtwDistance) {
	case DTW_EUCLIDEAN: {
	  initMatrix(seq1, seq2, mtrx);
	  break;
						}
	case DTW_DERIVATIVE: {
		getDerivatives(seq1, der1);
		getDerivatives(seq2, der2);
		initMatrix(der1, der2, mtrx);
		break;
						 }
  }
  float dist = calcDistance(mtrx);
  path = setWarpPath(mtrx);
  return dist;
}


/*	REFERENCE: Keogh et al., Derivative Dynamic Time Warping.
	ADAPTATION: derivative of the 1st and last point calculated from 2 adj. points (instead of 3)
	EXAMPLE: seq = [s0,s1,s2], der = [d0,d1,d2]
		d0 = s1-s0
		d1 = ((s1-s0)+((s2-s0)/2))/2
		d2 = s2-s1
*/
void TExamplesDistance_DTW::getDerivatives(vector<float> &seq, vector<float> &der) const
{
  vector<float>::const_iterator sbegin(seq.begin()), send(seq.end()), sip(sbegin), si(sbegin), sin(sbegin+1);
  der.clear();
  if ( send - sbegin > 2 ) {
	  // d0 = s1-s0
	  der.push_back( (*sin)-(*si) );
	  si++;
	  sin++;
	  // d1, ...
	  for(; sin != send; sip++, si++, sin++) {
		der.push_back( ((*si)-(*sip)+((*sin)-(*sip))/2)/2 );
			
	  }
	  sip++;
	  si++;
	  // d2
	  der.push_back( (*si)-(*sip) );
  }
  else {
	  // 2 time points
	  if ( sin < send ) {
		  int diff = (*sin)-(*si);
		  der.push_back(diff);
		  der.push_back(diff);
	  }
	  // single time point -> NaN
	  else {
		  der.push_back(numeric_limits<float>::signaling_NaN());
	  }
  }
}


void TExamplesDistance_DTW::initMatrix(const vector<float> &p, const vector<float> &q, TdtwMatrix &mtrx) const
{
	vector <float>::const_iterator iter_p, iter_q, iter_pe, iter_qe;
	// find max difference between p and q
	float minp = numeric_limits<float>::max(), maxp = numeric_limits<float>::min();
	float minq = numeric_limits<float>::max(), maxq = numeric_limits<float>::min();
	for ( iter_p = p.begin(), iter_pe = p.end(); iter_p != iter_pe; iter_p++ ) {
		if ((*iter_p) < minp ) minp = (*iter_p);
		if ((*iter_p) > maxp ) maxp = (*iter_p);
	}
	for ( iter_q = q.begin(), iter_qe = q.end(); iter_q != iter_qe; iter_q++ ) {
		if ((*iter_q) < minq ) minq = (*iter_q);
		if ((*iter_q) > maxq ) maxq = (*iter_q);
	}
	
      const float m1 = fabs(maxp-minq), m2 = fabs(maxq-minp);
      const float maxDiff = m1 > m2 ? m1 : m2;
	//printf("minp:%5.5f maxp:%5.5f minq:%5.5f maxq:%5.5f maxDiff:%5.5f\n", minp,maxp,minq,maxq,maxDiff);
	// build matrix, calculate dist2
	float diff;
	for ( iter_p = p.begin(), iter_pe = p.end(); iter_p != iter_pe; iter_p++ )
	{
        TdtwVector v;
		for ( iter_q = q.begin(), iter_qe = q.end(); iter_q != iter_qe; iter_q++ )
		{
			if ((*iter_p) == numeric_limits<float>::signaling_NaN() || (*iter_q) == numeric_limits<float>::signaling_NaN()) {
				// set to max in order to avoid the missing points
				diff = maxDiff;
			}
			else {
				diff = (*iter_p) - (*iter_q);
			}
			v.push_back( TdtwElement( diff * diff ) );
			//printf("p:%5.5f, q:%5.5f, diff:%5.5f\n", *iter_p,*iter_q,diff);
		}
		mtrx.push_back(v);
		//printf("\n");
	}
}


float TExamplesDistance_DTW::calcDistance(TdtwMatrix &mtrx) const
{
	TdtwMatrix::iterator iter_i, iter_i1, iter_endi;
	TdtwVector::iterator iter_j, iter_j1, iter_jd, iter_jd1, iter_endj;
	// initiate matrix
	mtrx[0][0].K = 1;
	mtrx[0][0].minSumDist2 = mtrx[0][0].dist2;
	// iterate rows from 1, column = 0
	for ( iter_i = mtrx.begin(), iter_i1 = iter_i + 1, iter_endi = mtrx.end(); iter_i1 < iter_endi; iter_i++, iter_i1++ )
	{
		iter_i1->at(0).minSumDist2 = iter_i->at(0).minSumDist2 + iter_i1->at(0).dist2;
		iter_i1->at(0).K = iter_i->at(0).K + 1;
		iter_i1->at(0).pParent = &(iter_i->at(0));

	}
	// iterate columns from 1, row = 0
	for ( iter_j = mtrx[0].begin(), iter_j1 = iter_j + 1, iter_endj = mtrx[0].end(); iter_j1 < iter_endj; iter_j++, iter_j1++ )
	{
		iter_j1->minSumDist2 = iter_j->minSumDist2 + iter_j1->dist2;
		iter_j1->K = iter_j->K + 1;
		iter_j1->pParent = &(*iter_j);
	}

	// fill matrix
	const int lenp = mtrx.size();
	const int lenq = mtrx.front().size();
	const int lenmin = lenp < lenq ? lenp : lenq;
	int d, d1;
	for ( d = 0, d1 = 1; d1 < lenmin; d++, d1++ )
	{
		// iterate rows from d1->, column = d1
		for ( iter_i = mtrx.begin() + d, iter_i1 = iter_i + 1, iter_endi = mtrx.end() - 1; iter_i < iter_endi; iter_i++, iter_i1++ )
		{
         	PdtwVector minElPVect;
			minElPVect.push_back( &((*iter_i)[d]) );
			minElPVect.push_back( &((*iter_i)[d1]) );
			minElPVect.push_back( &((*iter_i1)[d]) );
			(*iter_i1)[d1].updateMin( minElPVect );
		}
		// iterate columns from d1->, rows = d1
		for ( iter_j = mtrx[d].begin() + d, iter_j1 = iter_j + 1, iter_jd = mtrx[d1].begin() + d, iter_jd1 = iter_jd + 1, \
			  iter_endj = mtrx[d].end() - 1; iter_j < iter_endj; iter_j++, iter_j1++, iter_jd++, iter_jd1++ )
		{
            PdtwVector minElPVect;
			minElPVect.push_back( &(*iter_j) );
			minElPVect.push_back( &(*iter_j1) );
			minElPVect.push_back( &(*iter_jd) );
			(*iter_jd1).updateMin( minElPVect );
		}
	}
	TdtwElement endEl = mtrx[lenp-1][lenq-1];
	return sqrt(endEl.minSumDist2) / endEl.K;
}


void TExamplesDistance_DTW::printMatrix(const TdtwMatrix &mtrx) const
{

  {
  const_ITERATE(TdtwMatrix, mi, mtrx) {
    const_ITERATE(TdtwVector, mii, *mi)
      printf("%5.5f ", (*mii).dist2);
    printf("\n");
  }
  printf("\n\n");
  }

  {
  const_ITERATE(TdtwMatrix, mi, mtrx) {
    const_ITERATE(TdtwVector, mii, *mi)
      printf("%5.5f ", (*mii).minSumDist2);
    printf("\n");
  }
  printf("\n\n");
  }

  {
  const_ITERATE(TdtwMatrix, mi, mtrx) {
    const_ITERATE(TdtwVector, mii, *mi)
      printf("%i ", (*mii).K);
    printf("\n");
  }
  printf("\n\n");
  }
}


PWarpPath TExamplesDistance_DTW::setWarpPath(const TdtwMatrix &mtrx) const
{
	PWarpPath warpPath = new TWarpPath;
	int ii = mtrx.size() - 1, 
		jj = mtrx[0].size() - 1;
	warpPath->push_back( TAlignment(ii, jj) );
	while ( mtrx[ii][jj].pParent != NULL )
	{
		if ( ii > 0 && jj > 0)
		{
			if (mtrx[ii][jj].pParent == &(mtrx[ii-1][jj-1]))
			{
				warpPath->push_back( TAlignment(--ii, --jj) );
			}
			else if (mtrx[ii][jj].pParent == &(mtrx[ii][jj-1]))
			{
				warpPath->push_back( TAlignment(ii, --jj) );
			}
			else if (mtrx[ii][jj].pParent == &(mtrx[ii-1][jj]))
			{
				warpPath->push_back( TAlignment(--ii, jj) );
			}
		}
		else if (ii > 0)
		{
				warpPath->push_back( TAlignment(--ii, jj) );
		}
		else if (jj > 0)
		{
				warpPath->push_back( TAlignment(ii, --jj) );
		}
	}
	return warpPath;
}
/*{
	PWarpPath warpPath = new TWarpPath;
	int ii = mtrx.size() - 1, 
		jj = mtrx.front().size() - 1;

	// traceback for minimal warping path
	warpPath->push_back( TAlignment(ii, jj) );
	while ( ii + jj > 0 )
	{
        float bestSumDist2 = numeric_limits<float>::max();
        int best_i, best_j;

#define TEST(iii,jjj) \
if (	(iii>=0) \
	&&	(jjj>=0) \
	&&	(	(mtrx[iii][jjj].minSumDist2 < bestSumDist2) \
		 ||	(	(mtrx[iii][jjj].minSumDist2 == bestSumDist2) \
			 && ( abs(best_i-best_j) > abs(iii-jjj) ) \
			) \
		) \
	) \
{ bestSumDist2 = mtrx[iii][jjj].minSumDist2; best_i = iii; best_j = jjj; }

        TEST(ii-1, jj-1)
        TEST(ii-1, jj)
        TEST(ii, jj-1)

#undef TEST

        ii = best_i;
        jj = best_j;
        warpPath->push_back( TAlignment(ii, jj) );
	}
    return warpPath;
}*/


TExamplesDistanceConstructor_DTW::TExamplesDistanceConstructor_DTW()
: dtwDistance(TExamplesDistance_DTW::DTW_EUCLIDEAN)
{}


PExamplesDistance TExamplesDistanceConstructor_DTW::operator()(PExampleGenerator egen, const int &, PDomainDistributions ddist, PDomainBasicAttrStat bstat) const
{ return mlnew TExamplesDistance_DTW(dtwDistance, normalize, ignoreClass, egen, ddist, bstat); }

