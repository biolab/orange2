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


#ifdef _MSC_VER
  #pragma warning (disable : 4786 4114 4018 4267 4244 4702 4710)
#endif

#include <math.h>
#include "lcomb.hpp"
#include <vector>

using namespace std;

vector<double> p_fact;
vector<vector<double> > p_comb;
vector<vector<double> > p_stirling2;
vector<double> p_bell;

vector<double> p_logfact;
vector<vector<double> > p_logcomb;


bool init_p() {
  p_fact.push_back(1);
  p_logfact.push_back(0);

  p_stirling2.push_back(vector<double>());
  p_stirling2.back().push_back(0);
  return 0;
}

bool to_init = init_p();


double fact(const int &n)
{
  int sze = p_fact.size();
  if (sze <= n) {
    p_fact.reserve(n+1);
    float bk = p_fact.back();
    for(; sze <= n; sze++)
      p_fact.push_back(bk *= float(sze));
  }
  return p_fact[n];
}


double comb(const int &n, const int &k)
{ if ((int(p_comb.size())>n) && (int(p_comb[n].size())>k)) {
    double &res = p_comb[n][k];
    if (res<0)
      res = fact(n)/fact(k)/fact(n-k);
    return res;
  }

  p_comb.reserve(n+1);
  { for(int i = n-p_comb.size()+1; i--; )
      p_comb.push_back(vector<double>());
  }

  vector<double> &line = p_comb[n];
  line.reserve(k+1);
  { for(int i = k-line.size()+1; i--; )
      line.push_back(-1);
  }

  line[k] = fact(n)/fact(k)/fact(n-k);
  return line[k];
}


double stirling2(const int &n, const int &k)
{ if ((k<1) || (k>n))
    return 0.0;

  if ((k==1) || (k==n))
    return 1.0;

  if ((n<int(p_stirling2.size())) && (k<int(p_stirling2[n].size()))) {
    double &res = p_stirling2[n][k];
    if (res<0)
      res = k*stirling2(n-1,k) + stirling2(n-1,k-1);
    return res;
  }

  if (n >= int(p_stirling2.size())) {
    p_stirling2.reserve(n+1);
    { for(int i = n-p_stirling2.size()+1; i--; )
        p_stirling2.push_back(vector<double>());
    }
  }

  vector<double> &line = p_stirling2[n];
  if (k >= int(line.size())) {
    line.reserve(k+1);
    { for(int i = k-line.size()+1; i--; )
        line.push_back(-1);
    }
  }

  line[k] = k*stirling2(n-1,k) + stirling2(n-1,k-1);
  return line[k];
}


double bell(const int &n)
{ double res = 0.0;
  for(int i = 1; i <= n; res += (stirling2(n, i++)));
  return res;
}


double log(double);

const float log_of_2 = log(2.0);


double logfact(const int &n)
{
  if (int(p_logfact.size()) <= n) {
    p_logfact.reserve(n+1);
    float bk = p_logfact.back();
    for(int sze = p_logfact.size(); sze<=n; sze++)
      p_logfact.push_back(bk += log(float(sze))/log_of_2);
  }
  return p_logfact[n];
}


double logcomb(const int &n, const int &k)
{ if ((int(p_logcomb.size())>n) && (int(p_logcomb[n].size())>k)) {
    double &res = p_logcomb[n][k];
    if (res==-99.0)
      res = logfact(n)-logfact(k)-logfact(n-k);
    return res;
  }

  p_comb.reserve(n+1);
  { for(int i = n-p_logcomb.size()+1; i--; )
      p_logcomb.push_back(vector<double>());
  }

  vector<double> &line = p_logcomb[n];
  line.reserve(k+1);
  { for(int i = k-line.size()+1; i--; )
      line.push_back(-99.0);
  }

  line[k] = logfact(n)-logfact(k)-logfact(n-k);
  return line[k];
}

