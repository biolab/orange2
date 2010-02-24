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

/*  Most of the functions on this file are adapted from stat.py, which is in
    turn based on number of various sources. */

#ifndef STAT_HPP
#define STAT_HPP

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <vector>
#include <set>
#include <map>
#include <algorithm>
#include <limits>

using namespace std;

#include "stladdon.hpp"
#include "stat.hpp"
#include "statexceptions.hpp"
//#include "random.hpp"

#define SITERATE(iter, pna)  for(iterator iter((pna).begin()), iter##_end((pna).end()); iter!=iter##_end; iter++)
#define const_SITERATE(iter, pna)  for(const_iterator iter((pna).begin()), iter##_end((pna).end()); iter!=iter##_end; iter++)
#define DEFINE_TYPENAME typedef typename vector<T>::iterator iterator; typedef typename vector<T>::const_iterator const_iterator;


/* *********** AUXILIARY FUNCTIONS ************/

#ifdef _MSC_VER
  #pragma warning (disable : 4251 4996)
#endif

#ifndef _MSC_VER_70
inline double abs(double x)
{ return fabs(x); }
#endif

template<class T>
inline T sqr(const T &x) { return x*x; }


template<class T>
inline int convert_to_int(const T &x) { return int(x); }

template<class T>
inline double convert_to_double(const T &x) { return double(x); }


template<class T>
int compare(const T &x, const T &y)
{ if (x<y) return -1;
  if (x>y) return 1;
  return 0;
}

template<class T>
class CompareByIndex
{ const vector<T> &items;
public:
  
  CompareByIndex(const vector<T> &po)
   : items(po) {}

  int operator()(const int &i1, const int &i2) const
  { return items[i1]<items[i2]; }
};


template<class T, class U>
class CompareByIndex_pred
{ const vector<T> &items;
  U less_than;

public:
  CompareByIndex_pred(const vector<T> &po, U cf)
   : items(po), less_than(cf) {}

  int operator()(const int &i1, const int &i2) const
  { return less_than(items[i1], items[i2]); }
};




/* *********** SUPPORT FUNCTIONS ************/



template<class T>
T sum(const vector<T> &x, const T &init=0.0)
{ DEFINE_TYPENAME
  T sum=init;
  const_SITERATE(ii, x)
    sum += *ii;
  return sum;
}

template<class T>
void cumsum(const vector<T> &x, vector<T> &res, const T &init=0.0)
{ DEFINE_TYPENAME
  T sum=init;
  const_SITERATE(ii, x) {
    sum += *ii;
    res.push_back(sum);
  }
}

template<class T>
T ss(const vector<T> &x, const T &init=0.0)
{ DEFINE_TYPENAME
  T sum=init;
  const_SITERATE(ii, x)
    sum+=sqr(*ii);
  return sum;
}


template<class T>
T summult(const vector<T> &x, const vector<T> &y, const T &init=0.0)
{ DEFINE_TYPENAME
  
  if (x.size() != y.size())
    throw StatException("summult: lists of different sizes");

  T sum=init;
  for(const_iterator xi(x.begin()), xe(x.end()), yi(y.begin());
      xi!=xe; sum+= *(xi++) * *(yi++));
  return sum;
}


template<class T>
T sumdiffsquared(const vector<T> &flist1, const vector<T> &flist2, const T &init=0.0)
{ DEFINE_TYPENAME

  if (flist1.size() != flist2.size())
    throw StatException("sumdiffsquared: lists of different sizes");

  T sum=init;
  for(const_iterator i1(flist1.begin()), e1(flist1.end()), i2(flist2.begin()); i1!=e1; sum+=sqr(*(i1++)-*(i2++)));
  return sum;
}


template<class T>
T sumsquared(vector<T> &x, const T &init=0.0)
{ return sqr(sum(x, init)); }

template<class T>
bool shellsort(const vector<T> &flist, vector<int> &indices, vector<T> &slist)
{ DEFINE_TYPENAME
  
  int len=flist.size();
  indices=vector<int>(len);
  vector<int>::iterator ii=indices.begin();
  for(int idx=0; idx<len; idx++)
    *(ii++)=idx;

  stable_sort(indices.begin(), indices.end(), CompareByIndex<T>(flist));

  slist=vector<T>(len);
  iterator si(slist.begin());
  ITERATE(vector<int>, i2, indices)
    *(si++)=flist[*i2];
    
  return true;
}


template<class T, class U>
bool shellsort(const vector<T> &flist, vector<int> &indices, vector<T> &slist, const U &cf)
{ DEFINE_TYPENAME
  
  int len=flist.size();
  indices=vector<int>(len);
  vector<int>::iterator ii=indices.begin();
  for(int idx=0; idx<len; idx++)
    *(ii++)=idx;

  stable_sort(indices.begin(), indices.end(), CompareByIndex_pred<T, U>(flist, cf));

  slist=vector<T>();
  slist.reserve(len);
  ITERATE(vector<int>, i2, indices)
    slist.push_back(flist[*i2]);
    
  return true;
}


template<class T>
bool rankdata(const vector<T> &flist, vector<double> &ranks)
{ vector<T> items;
  vector<int> indices;
  shellsort(flist, indices, items);

  int sumranks=0, dupcount=0, n=indices.size();
  ranks=vector<double>(n);
  for(int beg=0, en; beg<n; beg=en) {
    const T &begi=items[beg];
    for(en=beg+1; (en<n) && (begi==items[en]); en++);
    double averank=double((en-1)+beg)/2 + 1;
    while(beg<en)
      ranks[indices[beg++]]=averank;
  }

  return true;
}


template<class T, class U>
bool rankdata(const vector<T> &flist, vector<double> &ranks, U cf)
{ vector<T> items;
  vector<int> indices;
  shellsort(flist, indices, items, cf);

  int sumranks=0, dupcount=0, n=indices.size();
  ranks=vector<double>(n);
  for(int beg=0, en; beg<n; beg=en) {
    for(en=beg+1; (en<n) && (!cf(items[beg], items[en])); en++);
    double averank=double((en-1)+beg)/2 + 1;
    while(beg<en)
      ranks[indices[beg++]]=averank;
  }

  return true;
}

/* *********** CENTRAL TENDENCY ************/


template<class T>
T geometricmean(const vector<T> &flist)
{ DEFINE_TYPENAME
  if (!flist.size())
    throw StatException("geometricmean: empty list");

  T mult=1.0;
  const_SITERATE(fi, flist)
    mult *= *fi;

  if (mult<=0.0)
    throw StatException("geometricmean: non-positive product");

  return exp(log(mult)/flist.size());
}


template<class T>
T harmonicmean(const vector<T> &flist)
{ DEFINE_TYPENAME
  if (!flist.size())
    throw StatException("harmonicmean: empty list");

  T sum=0.0;
  const_SITERATE(fi, flist)
    if (*fi==0.0)
      throw StatException("harmonicmean: division by zero");
    else sum+=T(1.0) / *fi;
  return T(flist.size())/sum;
}


template<class T>
T mean(const vector<T> &flist)
{ DEFINE_TYPENAME
  if (!flist.size())
    throw StatException("mean: empty list");

  T sum=0.0;
  const_SITERATE(fi, flist)
    sum += *fi;
  return sum/flist.size();
}


template<class T>
T median(const vector<T> &med)
{ if (!med.size())
    throw StatException("median: empty list");

  vector<T> med2(med);
  nth_element(med2.begin(), med2.begin()+med2.size()/2, med2.end());
  return middleelement(med2);
}


template<class T, class C>
T median(const vector<T> &med, const C &compare)
{ if (!med.size())
    throw StatException("median: empty list");

  vector<T> med2(med);
  nth_element(med2.begin(), med2.begin()+med2.size()/2, med2.end(), compare);
  return middleelement(med2);
}


template<class T>
T middleelement(const vector<T> &med)
{ DEFINE_TYPENAME
  const_iterator medmid(med.begin()+med.size()/2);
  if (med.size()%2)
    return *min_element(medmid, med.end());
  else
    return (*max_element(med.begin(), medmid) + *min_element(medmid, med.end()))/2.0;
}


template<class T>
int mode(const vector<T> &flist, vector<T> &mode)
{ DEFINE_TYPENAME
  typedef typename map<T, int>::iterator mapiterator;
  if (!flist.size())
    throw StatException("mode: empty list");

  map<T, int> bins;
  const_SITERATE(fi, flist) {
    mapiterator bi=bins.lower_bound(*fi);
    if ((bi==bins.end()) || ((*bi).first!=*fi))
      bins[*fi]=1;
    else
      (*bi).second++;
  }

  int count=0;
  for(mapiterator bi(bins.begin()), be(bins.end()); bi!=be; bi++)
    if ((*bi).second>count) {
      count=(*bi).second;
      mode.clear();
      mode.push_back((*bi).first);
    }
    else if ((*bi).second==count) {
      mode.push_back((*bi).first);
    }
  return count;
}



template<class T, class C>
int mode(const vector<T> &flist, vector<T> &mode, const C &compare)
{ DEFINE_TYPENAME
  typedef typename map<T, int, C>::iterator mapiterator;

  if (!flist.size())
    throw StatException("mode: empty list");

  map<T, int, C> bins(compare);
  const_SITERATE(fi, flist) {
    mapiterator bi=bins.lower_bound(*fi);
    if ((bi==bins.end()) || ((*bi).first!=*fi))
      bins[*fi]=1;
    else
      (*bi).second++;
  }

  int count=0;
  for(mapiterator bi(bins.begin()), be(bins.end()); bi!=be; bi++)
    if ((*bi).second>count) {
      count=(*bi).second;
      mode.clear();
      mode.push_back((*bi).first);
    }
    else if ((*bi).second==count) {
      mode.push_back((*bi).first);
    }
  return count;
}



/* *********** MOMENTS ************/

template<class T>
T moment(const vector<T> &flist, const int &mom)
{ DEFINE_TYPENAME
  if (!flist.size())
    throw StatException("moment: empty list");

  T me=mean(flist);

  if (mom==1) return me;
  if (mom==2) return samplevar(flist);

  T sum=0.0;
  const_SITERATE(fi, flist) {
    T dx= *fi-me;
    if (dx>0.0)
      sum+=exp(log(dx)*mom);
    else if (dx<0.0)
      if (mom%2)
        sum-=exp(log(-dx)*mom);
      else
        sum+=exp(log(-dx)*mom);
  }
  return sum/flist.size();
}


template<class T>
T variation(const vector<T> &flist)
{ return samplestdev(flist)/mean(flist) * 100.0;
}


template<class T>
T skewness(const vector<T> &flist)
{ T mom2=samplevar(flist);
  if (mom2==0.0)
    throw StatException("skewness: variation is 0.0");
  return moment(flist, 3) / exp(log(mom2)*1.5);
}


template<class T>
T kurtosis(const vector<T> &flist)
{ T mom2=samplevar(flist);
  if (mom2==0.0)
    throw StatException("skewness: variation is 0.0");
  return moment(flist, 4) / sqr(mom2);
}



/* *********** FREQUENCY STATS************/

template<class T>
T scoreatpercentile(const vector<T> &flist, const double &perc)
{ DEFINE_TYPENAME
  
  if (!flist.size())
    throw StatException("mode: empty list");

  vector<T> l2(flist);
  iterator mid(l2.begin()+int(l2.size()*perc/100.0+0.5));
  nth_element(l2.begin(), mid, l2.end());
  return *min_element(mid, l2.end());
}


template<class T, class C>
T scoreatpercentile(const vector<T> &flist, const double &perc, const C &compare)
{ DEFINE_TYPENAME
  
  if (!flist.size())
    throw StatException("mode: empty list");

  vector<T> l2(flist);
  iterator mid(l2.begin()+int(l2.size()*perc/100.0+0.5));
  nth_element(l2.begin(), mid, l2.end(), compare);
  return *min_element(mid, l2.end());
}


template<class T>
double percentileofscore(const vector<T> &flist, const T &x)
{ DEFINE_TYPENAME
  vector<T> l2(flist);
  iterator part=partition(l2.begin(), l2.end(), bind2nd(less<T>(), x));
  return double(part-l2.begin())/l2.size() * 100.0;
}


template<class T, class C>
double percentileofscore(const vector<T> &flist, const T &x, const C &compare)
{ DEFINE_TYPENAME
  vector<T> l2(flist);
  iterator part=partition(l2.begin(), l2.end(), bind2nd(compare, x));
  return double(part-l2.begin())/l2.size() * 100.0;
}


template<class T>
void histogram (const vector<T> &flist,
                vector<int> &counts, T &min, T &binsize, int &extrapoints,
                int numbins=10)
{ DEFINE_TYPENAME
  T max;

  min=*min_element(flist.begin(), flist.end());
  max=*max_element(flist.begin(), flist.end());
  T ebw=(max-min)/T(numbins) + 1.0;
  binsize=(max-min+ebw)/T(numbins);
  min-=binsize/2;

  counts=vector<int>(numbins, 0);
  extrapoints=0;
  const_SITERATE(ii, flist) {
    int binno=convert_to_int((*ii-min)/binsize);
    if (binno<numbins)
      counts[binno]++;
    else
      extrapoints++;
  }
}


template<class T>
void histogram (const vector<T> &flist,
                vector<int> &counts, T &rmin, T &binsize, int &extrapoints,
                const T &min, const T &max, int numbins=10)
{ DEFINE_TYPENAME
  rmin=min;

  binsize=(max-min)/T(numbins);
  counts=vector<int>(numbins, 0);
  extrapoints=0;
  const_SITERATE(ii, flist) {
    int binno=convert_to_int((*ii-min)/(binsize));
    if (binno<numbins)
      counts[binno]++;
    else
      extrapoints++;
  }
}


template<class T>
void cumfreq  (const vector<T> &flist,
               vector<int> &counts, T &min, T &binsize, int &extrapoints,
               int numbins=10)
{ histogram(flist, counts, min, binsize, extrapoints, numbins);
  for(int i=1; i<numbins; counts[i]+=counts[i-1], i++);
}


template<class T>
void cumfreq  (const vector<T> &flist,
               vector<int> &counts, T &rmin, T &binsize, int &extrapoints,
               const T &min, const T &max, int numbins=10)
{ histogram(flist, counts, rmin, binsize, extrapoints, min, max, numbins);
  for(int i=1; i<numbins; counts[i]+=counts[i-1], i++);
}



template<class T>
void relfreq  (const vector<T> &flist,
               vector<double> &counts, T &min, T &binsize, int &extrapoints,
               int numbins=10)
{ DEFINE_TYPENAME
  vector<int> hcounts;
  histogram(flist, hcounts, min, binsize, extrapoints, numbins);
  counts.clear();
  double ls=flist.size();
  ITERATE(vector<int>, ii, hcounts)
    counts.push_back(*ii/ls);
}

template<class T>
void relfreq  (const vector<T> &flist,
               vector<double> &counts, T &rmin, T &binsize, int &extrapoints,
               const T &min, const T &max, int numbins=10)
{ DEFINE_TYPENAME
  vector<int> hcounts;
  histogram(flist, hcounts, rmin, binsize, extrapoints, min, max, numbins);
  counts.clear();
  double ls=flist.size();
  ITERATE(vector<int>, ii, hcounts)
    counts.push_back(*ii/ls);
}


/* *********** VARIABILITY ************/

template<class T>
T samplevar(const vector<T> &flist)
{ DEFINE_TYPENAME
  if (!flist.size())
    throw StatException("samplevar: empty list");

  T me=mean(flist);
  T sum=0.0;
  const_SITERATE(fi, flist)
    sum+=sqr(*fi-me);
  return sum/flist.size();
}


template<class T>
T samplestdev(const vector<T> &flist)
{ return sqrt(samplevar(flist)); }


template<class T>
T var(const vector<T> &flist)
{ DEFINE_TYPENAME
  if (flist.size()<2)
    throw StatException("samplevar: empty or one-element list");

  T me=mean(flist);
  T sum=0.0;
  const_SITERATE(fi, flist)
    sum+=sqr(*fi-me);
  return sum/(flist.size()-1);
}


template<class T>
T stdev(const vector<T> &flist)
{ return sqrt(var(flist)); }


template<class T>
T sterr(const vector<T> &flist)
{ return stdev(flist)/sqrt(T(flist.size())); }


template<class T>
T z(const vector<T> &flist, const T &x)
{ return (x-mean(flist))/samplestdev(flist); }


template<class T>
bool zs(const vector<T> &flist, vector<T> &zss)
{ DEFINE_TYPENAME
  T me=mean(flist), ss=samplestdev(flist);
  zss=vector<T>(flist.size());
  iterator zi(zss.begin());
  const_SITERATE(fi, flist)
    *(zi++)=(*fi-me)/ss;
  return true;
}



/* *********** TRIMMING FNCS ************/

template<class T>
void trimboth (const vector<T> &flist, double proportion, vector<T> &clist)
{ int tocut=int(flist.size()*proportion);
  if (tocut*2>flist.size())
    throw StatException("trim proportion too large");

  clist=vector<T>(flist.begin()+tocut, flist.end()-tocut);
}


template<class T>
void trim1 (const vector<T> &flist, double proportion, vector<T> &clist, bool right=true)
{ int tocut=int(flist.size()*proportion);
  if (tocut>flist.size())
    throw StatException("trim proportion too large");

  if (right)
    clist=vector<T>(flist.begin(), flist.end()-tocut);
  else
    clist=vector<T>(flist.begin()+tocut, flist.end());
}



/* *********** PROBABILITY CALCS ************/

template<class T>
T gammln(const T &xx)
{
    static T cof[6] = {76.18009173, -86.50532033, 24.01409822, -1.231739516, 0.120858003e-2, -0.536382e-5};

    T x=xx, y=xx, tmp=x+5.5;
    tmp-=(x+0.5)*log(tmp);
    T ser=1.000000000190015;
    for(int j=0; j<6; j++)
     ser+=cof[j]/++y;
    return -tmp + log(T(2.5066282746310005)*ser/x);
}


template<class T>
T gammcf(const T &a, const T &x, T &gln)
{ const T FPMIN(1.0e-30);
  const int ITMAX= 100;
  const T EPS=3.0e-7;

  gln = gammln(a);
  T b=x+1.0-a;
  T c=T(1.0)/FPMIN;
  T d=T(1.0)/b;
  T h=d;
  for(int i=1; i<=ITMAX; i++) {
    T an=(a-double(i))*i;
    b += 2.0;
    d=an*d+b;
    if (abs(d) < FPMIN) d=FPMIN;
    c=b+an/c;
    if (abs(c) < FPMIN) c=FPMIN;
    d=T(1.0)/d;
    T del=d*c;
    h *= del;
    if (abs(del-1.0) < EPS)
      return exp(-x+a*log(x)-gln)*h;
  }

  throw StatException("gcf: a too large, ITMAX too small");
}


template<class T>
T gammser(const T &a, const T &x, T &gln)
{ const int ITMAX=100;
  const T EPS=3.0e-7;
  gln=gammln(a);
  if (x<=0.0)
    throw StatException("gser: negative x");

  T ap=a;
  T sum=T(1.0)/a;
  T del=sum;
  for(int n=1; n<=ITMAX; n++) {
    ++ap;
    del *= x/ap;
    sum += del;
    if (abs(del) < fabs(sum)*EPS)
      return sum*exp(-x+a*log(x)-gln);
  }
      
  throw StatException("gcf: a too large, ITMAX too small");
}
   

template<class T>
T gammp(const T &a, const T &x)
{ if ((x<0.0) || (a<=0.0))
    throw StatException("gammp: invalid arguments");

  T gln;
  return (x<(a+1.0)) ? gammser(a, x, gln) : -gammcf(a, x, gln)+1.0;
}


template<class T>
T gammq(const T &a, const T &x)
{ if ((x<0.0) || (a<=0.0))
    throw StatException("gammp: invalid arguments");

  T gln;
  return (x<(a+1.0)) ? -gammser(a, x, gln)+1.0 : gammcf(a, x, gln);
}

      
template<class T>
T erf(const T &x)
{ return (x<0.0) ? -gammp(T(0.5), x*x) : gammp(T(0.5), x*x); }


template<class T>
T erfc(const T &x)
{ return (x<0.0) ? T(1.0) + gammp(T(0.5), x*x) : gammq(T(0.5), x*x); }


template<class T>
T chisqprob(const T &x, const T &df)
{ return x > 1e-10 ? gammq(df*0.5, x*0.5) : T(1.0);
}



template<class T>
T betacf(const T &a, const T &b, const T &x)
{   const int ITMAX = 200;
    const double EPS = 3.0e-7;

    T bm=1.0, az=1.0, am=1.0;

    T qab=a+b;
    T qap=a+1.0;
    T qam=a-1.0;
    T bz= -qab*x/qap + 1.0;
    for(int i=0; i<=ITMAX; i++) {
	    T em = i+1;
	    T tem = em*2;
	    T d = em*(b-em)*x/((qam+tem)*(a+tem));
	    T ap = az + d*am;
	    T bp = bz+d*bm;
	    d = -(a+em)*(qab+em)*x/((qap+tem)*(a+tem));
	    T app = ap+d*az;
	    T bpp = bp+d*bz;
	    T aold = az;
	    am = ap/bpp;
	    bm = bp/bpp;
	    az = app/bpp;
	    bz = 1.0;
	    if (abs(az-aold)<(fabs(az)*EPS))
	      return az;
    }

    throw StatException("betacf: a or b too big, or ITMAX too small.");
    return -1.0;
}

template<class T>
T betai(const T &a, const T &b, const T &x)
{ if ((x<0.0) || (x>1.0))
    throw StatException("betai: bad x");

  T bt= ((x==0.0) || (x==1.0)) ? 0.0 : exp(gammln(a+b)-gammln(a)-gammln(b)+a*log(x)+b*log(- x + 1.0));
  return (x < (a+1.0)/(a+b+2.0)) ? bt*betacf(a,b,x)/a : -bt*betacf(b,a, -x+1.0)/b + 1.0;
}


template<class T>
T zprob(const T &z)
{ const T Z_MAX = 6.0;
  
  T x;

  if (z == 0.0)
    x = 0.0;

  else {
    T y = abs(z) * 0.5;
    if (y >= Z_MAX*0.5)
      x = 1.0;
    else if (y < 1.0) {
      T w = sqr(y);
	    x = ((((((((w * 0.000124818987
                  -0.001075204047) * w +0.005198775019) * w
                -0.019198292004) * w +0.059054035642) * w
              -0.151968751364) * w +0.319152932694) * w
            -0.531923007300) * w +0.797884560593) * y * 2.0;
    }
	  else {
	    y -= 2.0;
	    x = (((((((((((((y * -0.000045255659
                       +0.000152529290) * y -0.000019538132) * y
                     -0.000676904986) * y +0.001390604284) * y
                   -0.000794620820) * y -0.002034254874) * y
                 +0.006549791214) * y -0.010557625006) * y
               +0.011630447319) * y -0.009279453341) * y
             +0.005353579108) * y -0.002141268741) * y
           +0.000535310849) * y +0.999936657524;
    }
  }

  return (z > 0.0) ? (x+1.0)*0.5 : (-x+1.0)*0.5;
}


inline double fprob(const int &dfnum, const int &dfden, const double &F)
{ return betai(dfden*0.5, dfnum*0.5, dfden/(dfden+dfnum*F)); }


template<class T>
T erfcc(const T& x)
{ T z = abs(x);
  T t = T(1.0) / (z*0.5+1.0);
  T ans = t * exp(-z*z-1.26551223 + t*(t*(t*(t*(t*(t*(t*(t*(t*0.17087277-0.82215223)+1.48851587)-1.13520398)+0.27886807)-0.18628806)+0.9678418)+0.37409196)+1.00002368));
  return (x >= 0.0) ?  ans : -ans +2.0;
}


/* *********** RANDOM NUMBERS ***************/

template<class T>
T gasdev(const T &mean, const T &dev)
{ float r, v1, v2;
  do {
    v1=float(rand())/RAND_MAX*2-1;
    v2=float(rand())/RAND_MAX*2-1;
    r=v1*v1+v2*v2;
  } while ((r>1.0) || (r<0.0));

  return mean + dev * v1 * sqrt(T(-2.0*log(r)/r));
}

template<class T, class RF>
T gasdev(const T &mean, const T &dev, RF &randfunc)
{ float r, v1, v2;
  do {
    v1=randfunc(-1.0, 1.0);
    v2=randfunc(-1.0, 1.0);
    r=v1*v1+v2*v2;
  } while ((r>1.0) || (r<0.0));

  return mean + dev * v1 * sqrt(-2.0*log(r)/r);
}


/* *********** CORRELATION FNCS ************/

template<class T>
T pearsonr(const vector<T> flist1, const vector<T> &flist2, T &probrs)
{ DEFINE_TYPENAME

  const T TINY=T(1.0e-30);

  if (flist1.size() != flist2.size())
    throw StatException("pearsonr: lists of different sizes");

  double n=flist1.size();
  T sumx=0.0, sumx2=0.0, sumy=0.0, sumy2=0.0, summult=0.0;
  for(const_iterator i1(flist1.begin()), i2(flist2.begin()), e1(flist1.end()); i1!=e1; i1++, i2++) {
    sumx += *i1; sumx2 += sqr(*i1);
    sumy += *i2; sumy2 += sqr(*i2);
    summult += *i1 * *i2;
  }
  T r_num = summult * n - sumx * sumy;
  T r_den = sqrt( (sumx2*n - sqr(sumx) ) * (sumy2*n - sqr(sumy)) );
  T r = r_num / r_den;
  T df=n-2;
  T t = r * sqrt( df / ( (-r+1.0+TINY)*(r+1.0+TINY) ) );
  probrs=betai(df*0.5, T(0.5), df/(df+t*t));
  return r;
}


template<class T>
double spearmanr(const vector<T> &flist1, const vector<T> &flist2, double &probrs)
{ if (flist1.size() != flist2.size())
    throw StatException("spearmanr: lists of different sizes");

  double n=flist1.size();
  vector<double> ranks1, ranks2;
  rankdata(flist1, ranks1);
  rankdata(flist2, ranks2);
  double dsq=sumdiffsquared(ranks1, ranks2);
  double rs=1-6*dsq / (n*(sqr(n)-1));
  double df=n-2;
  double t = rs * sqrt(df / ((rs+1.0)*(1.0-rs)));
  probrs=betai(0.5*df,0.5,df/(df+t*t));
  return rs;
}


template<class T>
double pointbiserialr(const vector<T> &flist1, const vector<T> &flist2, double &probrs)
{ throw StatException("pointbiserialr: not implemented"); }


template<class T>
double kendalltau(const vector<T> &x, const vector<T> &y, double &probrs)
{ if (x.size() != y.size())
    throw StatException("kendaltau: lists of different sizes");
  if (!x.size())
    throw StatException("kendaltau: empty lists");

  int n=x.size();
  int n1=0, n2=0, iss=0;
  for (int j=0; j<n-1; j++)
    for (int k=j+1; k<n; k++) {
      int a1=compare(x[j], x[k]);
      int a2=compare(y[j], x[k]);
      int aa=a1*a2;
      if (aa) {
        n1++; n2++;
        if (aa>0) iss++;
        else iss--;
      }
      else {
        if (a1) n1++;
        if (a2) n2++;
      }
    }
  double tau = double(iss) / sqrt (double(n1)*double(n2));
  double nf=n;
  double svar=(4*nf+10)/(9*nf*(nf-1));
  double z= tau / sqrt(svar);
  probrs = erfcc(fabs(z)/1.4142136);
  return tau;
}
        

template<class T>
void linregress(const vector<T> flist1, const vector<T> &flist2, 
                T &slope, T &intercepr, T &r, T &probrs, T &sterrest)
{ DEFINE_TYPENAME

  const T TINY=T(1.0e-30);

  if (flist1.size() != flist2.size())
    throw StatException("pearsonr: lists of different sizes");

  double n=flist1.size();
  T sumx=0.0, sumx2=0.0, sumy=0.0, sumy2=0.0, summult=0.0;
  for(const_iterator i1(flist1.begin()), i2(flist2.begin()), e1(flist1.end()); i1!=e1; i1++, i2++) {
    sumx += *i1; sumx2 += sqr(*i1);
    sumy += *i2; sumy2 += sqr(*i2);
    summult += (*i1 * *i2);
  }
  T meanx = sumx/n;
  T meany = sumy/n;

  T r_num = summult * n - sumx * sumy;
  T r_den = sqrt( (sumx2*n - sqr(sumx) ) * (sumy2*n - sqr(sumy)) );
  r = r_num / r_den;
  T df=n-2;
  T t = r * sqrt( df / ( (-r+1.0+TINY)*(r+1.0+TINY) ) );
  probrs=betai(df*0.5, T(0.5), df/(df+t*t));

//  T z = log((r+1.0+TINY)/(-r+1.0+TINY));

  slope = r_num / (sumx2*n - sqr(sumx));
  intercepr = meany - slope*meanx;
  sterrest = sqrt(-sqr(r)+1.0) * samplestdev(flist2);
}



/* *********** INFERENTIAL STATS ************/

template<class T>
T ttest_1samp(const vector<T> &flist, const T &popmean, T &prob)
{
  T n=flist.size(), df=n-1.0;
  T t= (mean(flist) - popmean) / sqrt(var(flist)/n);
  prob=betai(df*0.5, T(0.5), df/(df+t*t));
  return t;
}


template<class T>
T ttest_ind(const vector<T> &x, const vector<T> &y, T &prob)
{ T n1=x.size(), n2=y.size(), df=n1+n2-2.0;
  T svar= ( sqr(stdev(x))*(n1-1.0) + sqr(stdev(y))*(n2-1.0) )  /  df;
  T t= ( mean(x) - mean(y) ) / sqrt(svar * ( (n1+n2)/(n1*n2) ) );
  prob=betai(df*0.5, T(0.5), df/(df+t*t));
  return t;
} 


template<class T>
T ttest_rel(const vector<T> &x, const vector<T> &y, T &prob)
{ if (x.size() != y.size())
    throw StatException("ttest_rel: lists of different sizes");

  T meanx=mean(x);
  T meany=mean(y);

  T n=x.size();
  T cov=0.0;
  for(int i=0; i<x.size(); i++)
    cov += (x[i]-meanx) * (y[i]-meany);
  T df=n-1.0;
  cov /= df;
  T sd = sqrt((var(x)+var(y) - cov*2.0)/n);
  if (sd==0.0)
    throw StatException("ttest_rel: sd==0, can't divide");
  T t = (mean(x)-mean(y))/sd;
  prob=betai(df*0.5, T(0.5), df/(df+t*t));
  return t;
}


template<class T>
T chisquare(const vector<T> &x, const vector<T> *exp, T &prob)
{ int n=x.size();

  T chisq=0.0;
  if (exp) {
    if (exp->size()!=n)
      throw StatException("chi_square: lists of different sizes");
    for(int i=0; i<n; i++)
      chisq+=sqr(x[i]-(*exp)[i]) / (*exp)[i];
  }
  else {
    T invted=sum(x)/T(n);
    for(int i=0; i<n; i++)
      chisq+=sqr(x[i]-invted) / invted;
  }
    
  prob=chisqprob(chisq, T(n-1));

  return chisq;
}


template<class T>
T min_el(const T &x, const T &y)
{ return x<y ? x : y; }

template<class T>
T max_el(const T &x, const T &y)
{ return x>y ? x : y; }

template<class T>
T chisquare2d(const vector<vector<T> > &cont,
              int &df, T &prob, T &cramerV, T &contingency_coeff)
{ if (!cont.size())
    throw StatException("chisquare2d: invalid contingency table");

  const T TINY=1.0e-30;
  int ni=cont.size();
  int nj=cont[0].size();
  if (!nj)
    throw StatException("chisquare2d: invalid contingency table");

  vector<T> sumi(ni), sumj(nj);
  int ci, cj;
  for(ci=0; ci<ni; ci++)
    sumi.push_back(T(0.0));
  for(cj=0; cj<nj; cj++)
    sumj.push_back(T(0.0));

  for(ci=0; ci<ni; ci++) {
    if (cont[ci].size()!=nj)
      throw StatException("chisquare2d: invalid contingency table");
    for(cj=0; cj<nj; cj++) {
      sumi[ci]+=cont[ci][cj];
      sumj[cj]+=cont[ci][cj];
    }
  }

  T sum=0.0;
  int nnj=0;
  for(cj=0; cj<nj; cj++)
    if (sumj[cj]>0.0) {
      sum+=sumj[cj];
      nnj++;
    }

  int nni=0;
  T chisq=0.0;
  for(ci=0; ci<ni; ci++) {
    if (sumi[ci]>0.0)
      nni++;
    for(cj=0; cj<nj; cj++) {
      T expctd=sumj[cj]*sumi[ci] / sum;
      chisq += sqr(cont[ci][cj]-expctd) / (expctd+TINY);
    }
  }

  df=(nni-1)*(nnj-1);

  prob=chisqprob(chisq, T(df));
  cramerV = sqrt (chisq/(sum*(min_el(nni, nnj)-1.0)));
  contingency_coeff=sqrt(chisq/(chisq+sum));
  return chisq;
}


template<class T>
T anova_rel(const vector<vector<T> > &cont, int &df_bt, int &df_err, T &prob)
{ 
  DEFINE_TYPENAME
  
  int k = cont.size();
  int n = cont[0].size();
  if ((n<2) || (k<2))
    throw StatException("anova_rel: invalid contingency table");

  int N = k*n;
  T G = T(0.0), SS_wt = T(0.0), SS_total = T(0.0), SS_bt = T(0.0), SS_bs = T(0.0), SS_err;
  vector<T> Ps(n, T(0.0));
  iterator Psi, Pse(Ps.end());
  for(typename vector<vector<T> >::const_iterator conti(cont.begin()), conte(cont.end()); conti!=conte; conti++) {
    if ((*conti).size() != n)
      throw StatException("anova_rel: number of subject is not the same in all groups");

    T t = T(0.0), tt = T(0.0);
    Psi = Ps.begin();
    for(typename vector<T>::const_iterator contii((*conti).begin()), contie((*conti).end()); contii!=contie; contii++, Psi++) {
      t += (*contii);
      *Psi += (*contii);
      tt += (*contii) * (*contii);
    }
    G += t;
    SS_total += tt;
    SS_wt += tt - t*t/n;
    SS_bt += t*t;
  }

  for(Psi = Ps.begin(); Psi != Pse; Psi++)
    SS_bs += *Psi * *Psi;

  const T GG_N = G*G/N;
  SS_total -= GG_N;
  SS_bt = SS_bt/n - GG_N;
  SS_bs = SS_bs/k - GG_N;
  SS_err = SS_wt - SS_bs;

  df_bt = (k-1);
  df_err = (N-k) - (n-1);

  if (SS_err < 1e-20) {
    prob = 0.0;
    return T(-1.0);
  }

  T MS_bt = SS_bt / df_bt;
  T MS_err = SS_err / df_err;
  T F = MS_bt / MS_err;
  prob = fprob(df_bt, df_err, F);
  return F;
}


template<class T>
T friedmanf(const vector<vector<T> > &cont, double &chi2, int &dfnum, int &dfden, double &prob)
{
  DEFINE_TYPENAME
  
  int k = cont.size();
  int N = cont[0].size();
  if ((N<2) || (k<2))
    throw StatException("friedmanf: invalid contingency table");

  vector<vector<T> > transposed;
  int line;
  for(line = 0; line < N; line++)
    transposed.push_back(vector<T>());

  typename vector<vector<T> >::iterator trani;
  const typename vector<vector<T> >::iterator tranb(transposed.begin()), trane(transposed.end());
  for(typename vector<vector<T> >::const_iterator conti(cont.begin()), conte(cont.end()); conti != conte; conti++) {
    if ((*conti).size() != N)
      throw StatException("friedmanf: number of subject is not the same in all groups");

    trani = tranb;
    for(const_iterator contii((*conti).begin()); trani!=trane; trani++, contii++)
      (*trani).push_back(*contii);
  }

  vector<double> R(k, 0.0), tranks;
  typename vector<double>::iterator Ri, Rb(R.begin()), Re(R.end()), tri;
  for(trani = tranb; trani != trane; trani++) {
    rankdata(*trani, tranks);
    for(Ri = Rb, tri = tranks.begin(); Ri != Re; Ri++, tri++)
      *Ri += *tri;
  }

  double RR = 0.0;
  for(Ri = Rb; Ri != Re; Ri++)
    RR += *Ri * *Ri;

  chi2 = 12 * N / float(k*(k+1)) *  (RR/N/N - k*(k+1)*(k+1) / 4.0);
  double F = (N-1) * chi2 / (N*(k-1) - chi2);
  prob = fprob(k-1, (k-1)*(N-1), F);
  return F;
}


template<class T>
double mannwhitneyu(const vector<T> &x, const vector<T> &y, double &prob)
{ vector<T> both(x);
  both.insert(both.end(), y.begin(), y.end());
  vector<double> ranks;
  rankdata(both, ranks);

  int n1=x.size(), n2=y.size();
  double u1=n1*n2 + (n1*(n1+1))/2.0;
  double u2=n1*n2 + (n2*(n2+1))/2.0;
  vector<double>::iterator ri;
  for(ri=ranks.begin(); n1--; u1 -= *(ri++));
  while(ri!=ranks.end())
    u2 -= *(ri++);
  double bigu=max_el(u1, u2);
  double smallu=min_el(u1, u2);
  double sd=sqrt(n1*n2*(n1+n2+1)/12.0);
  if (sd==0)
    throw StatException("mannwhitneyu: empty group");
  double z=abs((bigu-n1*n2/2.0) / sd);
  prob = 1.0 - zprob(z);
  return smallu;
}


template<class T, class G, class C>
double mannwhitneyu(const vector<T> &x, double &prob, const G &group, const C &compare)
{ DEFINE_TYPENAME
  vector<double> ranks;
  rankdata(x, ranks, compare);
  double u1=0.0, u2=0.0;
  int n1=0, n2=0;
  vector<double>::iterator ri(ranks.begin());
  const_SITERATE(xi, x)
    if (group(*xi)) {
      n1++;
      u1-= *(ri++);
    }
    else {
      n2++;
      u2-= *(ri++);
    }
  u1+=n1*n2 + (n1*(n1+1))/2.0;
  u2+=n1*n2 + (n2*(n2+1))/2.0;
  double bigu=max_el(u1, u2);
  double smallu=min_el(u1, u2);
  double sd=sqrt(n1*n2*(n1+n2+1)/12.0);
  if (sd==0)
    throw StatException("mannwhitneyu: empty group");
  double z=abs((bigu-n1*n2/2.0) / sd);
  prob = 1.0 - zprob(z);
  return smallu;
}


template<class T>
double ranksums(const vector<T> &x, const vector<T> &y, double &prob)
{ vector<T> both(x);
  both.insert(both.end(), y.begin(), y.end());
  vector<double> ranks;
  rankdata(both, ranks);

  int n1c=x.size();
  double sum=0.0;
  for(vector<double>::iterator ri(ranks.begin()); n1c--; sum += *(ri++));

  double n1=x.size(), n2=y.size();
  double expected = n1*(n1+n2+1) / 2.0;
  double z = (sum-expected) / sqrt(n1*n2*(n1+n2+1)/12.0);
  prob=zprob(z);
  return z;
}


template<class T, class G, class C>
double ranksums(const vector<T> &x, double &prob, const G &group, const C &compare)
{ DEFINE_TYPENAME
  vector <double> ranks;
  rankdata(x, ranks, compare);
  vector<double>::iterator ri(ranks.begin());
  double sum=0.0;
  int obs=0;
  const_SITERATE(xi, x)
    if (group(*xi)) {
      sum += *(ri++);
      obs++;
    }
    else
      ri++;

  double n1=obs, n2=x.size()-n1;
  double expected = n1*(n1+n2+1) / 2.0;
  double z = (sum-expected) / sqrt(n1*n2*(n1+n2+1)/12.0);
  prob=zprob(z);
  return z;
}
 

template<class T>
double wilcoxont(const vector<T> &x, const vector<T> &y, double &prob)
{ DEFINE_TYPENAME
  
  if (x.size() != y.size())
    throw StatException("ttest_rel: lists of different sizes");

  vector<T> d, absd;
  for(const_iterator xi(x.begin()), xe(x.end()), yi(y.begin());
      xi!=xe;
      xi++, yi++)
    if (*xi!=*yi) {
      d.push_back(*xi-*yi);
      absd.push_back(abs(d.back()));
    }

  if (!d.size()) {
    prob = 1.0;
    return 0.0;
  }

  vector<double> absdranks;
  rankdata(absd, absdranks);
  double r_plus=0.0, r_minus=0.0;
  for(int i=0; i<d.size(); *( (d[i]<0.0) ? &r_minus : &r_plus ) += absdranks[i], i++);
  double N=d.size();
  double se = sqrt(N*(N+1)*(2*N+1)/24.0);
  double wt = min_el(r_plus, r_minus);
  double z  = fabs(wt-N*(N+1)*0.25)/se;
  prob=1.0-zprob(z);
  return wt;
}



// This follows http://www-2.cs.cmu.edu/afs/cs/project/jair/pub/volume4/cohn96a-html/node7.html

template<class T, class U>
T loess_y(const T &refx, map<T, U> points, const float &windowProp)
{ typedef typename map<T, U>::const_iterator mapiterator;
  mapiterator from, to;

  /* Find the window */

  mapiterator lowedge = points.begin();
  mapiterator highedge = points.end();

  int needpoints = int(ceil(points.size() * windowProp));
  
  if ((needpoints<=1) || (needpoints>=points.size())) {
    from = lowedge;
    to = highedge;
    return ((*--highedge).first - (*lowedge).first)/2.0;
  }
    
  from = points.lower_bound(refx);
  to = points.upper_bound(refx);
  if (from==to)
    if (to != highedge)
      to++;
    else
      from --;


  /* Extend the interval; we set from to highedge when it would go beyond lowedge, to indicate that only to can be modified now */
  while (needpoints--) {
    if ((to == highedge) || ((from != highedge) && (refx - (*from).first < (*to).first - refx))) {
      if (from == lowedge)
        from = highedge;
      else
        from--;
    }
    else
      to++;
  }

  if (from == highedge)
    from = lowedge;
  else
    from++;

  mapiterator tt = to;
  --tt;
  
  T h = (refx - (*from).first);
  if ((*tt).first - refx  >  h)
    h = ((*tt).first - refx);

  h *= T(1.1);

  /* Iterate through the window */

  T Sx = 0.0, Sy = 0.0, Sxx = 0.0, Sxy = 0.0;

  T n = 0.0;

  for (; from != to; from++) {
    const T &x = (*from).first;
    const T &y = (*from).second;

    T w = abs(refx - x) / h;
    w = 1 - w*w*w;
    w = w*w*w;

    n   += w;
    Sx  += w * x;
    Sxx += w * x * x;
    Sy  += w * y;
    Sxy += w * x * y;
  }

  if (n==0)
    return Sy;

  T div = Sxx - Sx * Sx / n;
  if (!div)
    return Sy;

  return Sy/n + (Sxy - Sx*Sy/n) / div * (refx - Sx/n);
}


enum { DISTRIBUTE_MINIMAL, DISTRIBUTE_FACTOR, DISTRIBUTE_FIXED, DISTRIBUTE_UNIFORM, DISTRIBUTE_MAXIMAL };

template<class T, class U>
void distributePoints(const map<T, U> points, int nPoints, vector<T> &result, int method = DISTRIBUTE_MINIMAL)
{ typedef typename map<T, U>::const_iterator mapiterator;

  if (nPoints<0) {
    nPoints = -nPoints;
    method = DISTRIBUTE_FACTOR;
  }

  result.clear();

  if ((nPoints == 1) || (DISTRIBUTE_MINIMAL && (nPoints <= points.size())) || (DISTRIBUTE_MAXIMAL && (nPoints >= points.size()))) {
    for (mapiterator pi(points.begin()), pe(points.end()); pi != pe; pi++)
      result.push_back((*pi).first);
    return;
  }
  
  switch (method) {
    case DISTRIBUTE_FACTOR: {
      for (mapiterator pi(points.begin()), pe(points.end());;) {
        T ax = (*pi).first;
        result.push_back(ax);

        if (++pi==pe)
          break;

        // We could write this faster, but we don't want to run into problems with rounding floats
        T div = ((*pi).first - ax) / nPoints;
        for (int i=1; i < nPoints; i++)
          result.push_back(ax + i*div);
      }
      return;
    }


    case DISTRIBUTE_MINIMAL: {  // All original points plus some in between to fill up the quota
      T ineach = float(nPoints - points.size()) / float(points.size()-1);
      T inthis = T(0.0);
    
      for (mapiterator pi(points.begin()), pe(points.end());;) {
        T ax = (*pi).first;
        result.push_back(ax);

        if (++pi==pe)
          break;

        inthis += ineach;
        if (inthis>=T(0.5)) {
          T dif = ((*pi).first - ax) / (int(floor(inthis))+1);
          while (inthis>T(0.5)) {
            result.push_back(ax += dif);
            inthis -= T(1.0);
          }
        }
      }
      return;
    }

    case DISTRIBUTE_MAXIMAL: {  // Just as many points as allowed
      T ineach = float(points.size()) / float(nPoints); 
      T inthis = T(0.0);
    
      for(mapiterator pi(points.begin()), pe(points.end()); pi != pe; pi++) {
        inthis += 1;
        if (inthis >= 0) {
          result.push_back((*pi).first);
          inthis -= ineach;
        }
      }
      return;
    }

    case DISTRIBUTE_FIXED: {
      set<float> ppos;
      {
        for(mapiterator pi(points.begin()), pe(points.end()); pi != pe; pi++)
          ppos.insert((*pi).first);
      }

      float step = ppos.size()/float(nPoints-2);
      float up = 1.5;

      T x1 = T(0.0);
      T dx = T(0.0);
      result.push_back(*ppos.begin());
      for(set<float>::const_iterator pi(ppos.begin()), pe(ppos.end());;) {
        do {
          x1 = *pi;
          if (++pi==pe)
            break;
          T dx = *pi - x1;
          up -= 1.0;
        } while (up>1.0);
        if (pi==pe)
          break;

        for (; up<1.0; up += step) {
          const float toPush = x1+dx*up;
          if (result.back() != toPush)
            result.push_back(toPush);
        }
      }
      if (result.back() != x1)
        result.push_back(x1);
      return;
    }


    case DISTRIBUTE_UNIFORM: {
      T fi = (*points.begin()).first;
      mapiterator pe(points.end());
      pe--;
      T rg = ((*pe).first-fi) / (nPoints-1);
      for (int i = 0; i<nPoints; i++)
        result.push_back(fi + i*rg);
      return;
    }
  }
}

int nUniquePoints(const vector<double> &points);

void samplingFactor (const vector<double> &points,      int nPoints, vector<double> &result);
void samplingFactor (const map<double, double> &points, int nPoints, vector<double> &result);
void samplingMinimal(const vector<double> &points,      int nPoints, vector<double> &result);
void samplingMinimal(const map<double, double> &points, int nPoints, vector<double> &result);
void samplingFixed  (const vector<double> &points,      int nPoints, vector<double> &result);
void samplingFixed  (const map<double, double> &points, int nPoints, vector<double> &result);
void samplingUniform(const vector<double> &points,      int nPoints, vector<double> &result);
void samplingUniform(const map<double, double> &points, int nPoints, vector<double> &result);

template<class T, class U>
void loess(const U &points, int nPoints, const float &windowProp, map<T, T> &loess_curve, int distributionMethod = DISTRIBUTE_MINIMAL)
{ DEFINE_TYPENAME
  vector<T> xpoints;
  distributePoints(points, nPoints, xpoints, distributionMethod);
  for (const_iterator xi(xpoints.begin()), xe(xpoints.end()); xi!=xe; xi++) 
    loess_curve[*xi] = loess_y(*xi, points, windowProp);
}

class TXYW {
public:
  double x, y, w;

  TXYW(const double &ax, const double &ay, const double &aw = 1.0)
  : x(ax), y(ay), w(aw)
  {}

  TXYW(const TXYW &o)
  : x(o.x), y(o.y), w(o.w)
  {}
};


// refpoints and points should be sorted
void loess(const vector<double> &refpoints, const vector<TXYW> &points, const float &windowProp, vector<pair<double, double> > &result);
void lwr(const vector<double> &refpoints, const vector<TXYW> &points, const float &smoothFactor, vector<pair<double, double> > &result);

void loess(const vector<double> &refpoints, const vector<pair<double, double> > &points, const float &windowProp, vector<pair<double, double> > &result);
void loess(const vector<double> &refpoints, const map<double, double> &points, const float &windowProp, vector<pair<double, double> > &result);

void lwr(const vector<double> &refpoints, const vector<pair<double, double> > &points, const float &smoothFactor, vector<pair<double, double> > &result);
void lwr(const vector<double> &refpoints, const map<double, double> &points, const float &smoothFactor, vector<pair<double, double> > &result);

#endif
